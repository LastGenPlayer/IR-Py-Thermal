"""One camera, running.

A runtime pairs a capture thread with a hub and owns the operator-settable
state. Control operations are async because they have to cross onto the capture
thread and wait for it; frame consumption is async because that is where the
consumers live.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any, Self

from irthermal.camera.base import CameraError, CameraSource, SourceFactory
from irthermal.camera.emulator import EmulatorSource
from irthermal.camera.reader import FrameReader, ReaderStats
from irthermal.camera.v4l2 import V4L2Source
from irthermal.config import CameraConfig
from irthermal.core import commands
from irthermal.core.frame import Frame, Measurement
from irthermal.core.radiometry import (
    HIGH_RANGE_MODE,
    NORMAL_RANGE_MODE,
    RadiometryAdjustments,
)
from irthermal.core.resolutions import ResolutionParams
from irthermal.hub import FrameHub, Subscription
from irthermal.logging import get_logger

log = get_logger(__name__)

FIRST_FRAME_TIMEOUT_S = 20.0

FLOAT_PARAMETERS: dict[str, int] = {
    "correction": commands.PARAM_CORRECTION,
    "reflected_temp_c": commands.PARAM_REFLECTION,
    "air_temp_c": commands.PARAM_AMBIENT,
    "humidity": commands.PARAM_HUMIDITY,
    "emissivity": commands.PARAM_EMISSIVITY,
}
INTEGER_PARAMETERS: dict[str, int] = {"distance_m": commands.PARAM_DISTANCE}

# Empirical trim for the extended temperature range, carried over from v1.
_HIGH_RANGE_GAIN_RAW = (0.1, 0.0)
_HIGH_RANGE_GAIN_PROCESSED = (1.17, -40.9)


@dataclass(frozen=True, slots=True)
class CameraStatus:
    """Everything an operator or a health check wants to know."""

    id: str
    connected: bool
    frames: int
    fps: float
    read_errors: int
    reconnects: int
    last_error: str | None
    source: str | None
    width: int | None
    height: int | None
    subscribers: int


def build_source_factory(config: CameraConfig) -> SourceFactory:
    """Turn a camera's config into something that can open it, repeatedly."""
    if config.replay_file is not None:
        path = config.replay_file

        def open_emulator() -> CameraSource:
            return EmulatorSource.from_file(path, camera_raw=config.raw_mode)

        return open_emulator

    device = config.device

    def open_device() -> CameraSource:
        return V4L2Source.open(device, camera_raw=config.raw_mode)

    return open_device


class CameraRuntime:
    """A single camera's capture thread, frame hub and settings."""

    def __init__(self, config: CameraConfig, factory: SourceFactory | None = None) -> None:
        self.config = config
        self._factory = factory or build_source_factory(config)
        self._hub: FrameHub | None = None
        self._reader: FrameReader | None = None

    # --- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Create the hub and start capturing."""
        if self._reader is not None:
            raise RuntimeError(f"camera {self.config.id} already started")

        self._hub = FrameHub()
        self._reader = FrameReader(
            self.config.id,
            self._factory,
            self._hub,
            adjustments=RadiometryAdjustments(user_offset=self.config.temperature_offset),
        )
        self._reader.start()
        log.info("camera.starting", camera=self.config.id)

    async def stop(self) -> None:
        """Stop capturing and end every subscription."""
        if self._reader is not None:
            await asyncio.to_thread(self._reader.stop)
            self._reader = None
        if self._hub is not None:
            self._hub.close()
            self._hub = None
        log.info("camera.stopped", camera=self.config.id)

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    # --- frames -------------------------------------------------------------

    @property
    def hub(self) -> FrameHub:
        if self._hub is None:
            raise RuntimeError(f"camera {self.config.id} is not started")
        return self._hub

    @property
    def latest(self) -> Frame | None:
        """The most recent frame, or None before the first capture."""
        return self._hub.latest if self._hub is not None else None

    def subscribe(self, maxsize: int = 1) -> Subscription:
        """Receive frames as they are captured."""
        return self.hub.subscribe(maxsize)

    async def wait_for_frame(self, timeout: float = FIRST_FRAME_TIMEOUT_S) -> Frame:
        """Return the current frame, waiting for one if none has arrived yet."""
        existing = self.latest
        if existing is not None:
            return existing

        async with self.hub.subscribe() as subscription:
            frame = await asyncio.wait_for(subscription.get(), timeout)

        if frame is None:
            raise CameraError(f"camera {self.config.id} closed while waiting for a frame")
        return frame

    # --- status -------------------------------------------------------------

    @property
    def params(self) -> ResolutionParams | None:
        """Sensor geometry, once a frame has told us what it is."""
        source = self._reader.source if self._reader is not None else None
        return source.params if source is not None else None

    def status(self) -> CameraStatus:
        stats = self._reader.stats if self._reader is not None else ReaderStats()
        params = self.params
        return CameraStatus(
            id=self.config.id,
            connected=stats.connected,
            frames=stats.frames,
            fps=round(stats.fps, 2),
            read_errors=stats.read_errors,
            reconnects=stats.reconnects,
            last_error=stats.last_error,
            source=stats.description,
            width=params.width if params else None,
            height=params.height if params else None,
            subscribers=self._hub.subscriber_count if self._hub else 0,
        )

    # --- control ------------------------------------------------------------

    @property
    def adjustments(self) -> RadiometryAdjustments:
        if self._reader is None:
            return RadiometryAdjustments(user_offset=self.config.temperature_offset)
        return self._reader.adjustments

    async def _run_on_capture_thread(self, command: Any) -> Any:
        if self._reader is None:
            raise RuntimeError(f"camera {self.config.id} is not started")
        return await asyncio.wrap_future(self._reader.submit(command))

    async def calibrate(self) -> None:
        """Trigger a shutter calibration -- the audible click."""

        def run(reader: FrameReader) -> None:
            if reader.source is None:
                raise CameraError(f"camera {reader.camera_id} is not connected")
            if reader.source.camera_raw:
                reader._calibrate_raw()
            else:
                reader.source.send_command(commands.ACTION_SHUTTER_CALIBRATE)

        await self._run_on_capture_thread(run)
        log.info("camera.calibrate", camera=self.config.id)

    async def set_temperature_range(self, *, high: bool) -> None:
        """Switch between the normal and extended temperature ranges."""

        def run(reader: FrameReader) -> None:
            if reader.source is None:
                raise CameraError(f"camera {reader.camera_id} is not connected")

            action = commands.ACTION_RANGE_HIGH if high else commands.ACTION_RANGE_NORMAL
            reader.source.send_command(action)

            if not high:
                gain_m, gain_b = 1.0, 0.0
            elif reader.source.camera_raw:
                gain_m, gain_b = _HIGH_RANGE_GAIN_RAW
            else:
                gain_m, gain_b = _HIGH_RANGE_GAIN_PROCESSED

            reader.adjustments = replace(
                reader.adjustments,
                range_mode=HIGH_RANGE_MODE if high else NORMAL_RANGE_MODE,
                gain_m=gain_m,
                gain_b=gain_b,
            )

        await self._run_on_capture_thread(run)
        log.info("camera.range", camera=self.config.id, high=high)

    async def set_parameter(self, name: str, value: float) -> None:
        """Write one measurement parameter to the camera."""
        if name in FLOAT_PARAMETERS:
            payloads = commands.float_command(FLOAT_PARAMETERS[name], float(value))
        elif name in INTEGER_PARAMETERS:
            payloads = commands.ushort_command(INTEGER_PARAMETERS[name], int(value))
        else:
            known = sorted(FLOAT_PARAMETERS | INTEGER_PARAMETERS)
            msg = f"unknown parameter {name!r}; expected one of {known}"
            raise ValueError(msg)

        def run(reader: FrameReader) -> None:
            if reader.source is None:
                raise CameraError(f"camera {reader.camera_id} is not connected")
            for payload in payloads:
                reader.source.send_command(payload)

        await self._run_on_capture_thread(run)
        log.info("camera.parameter", camera=self.config.id, name=name, value=value)

    def set_temperature_offset(self, celsius: float) -> None:
        """Adjust the constant temperature trim. Takes effect on the next frame."""
        if self._reader is None:
            raise RuntimeError(f"camera {self.config.id} is not started")
        self._reader.adjustments = replace(self._reader.adjustments, user_offset=celsius)

    # --- measurement --------------------------------------------------------

    async def measure_points(self) -> dict[str, Measurement]:
        """Sample every configured point on the current frame."""
        frame = await self.wait_for_frame()
        return frame.measure((p.name, p.x, p.y) for p in self.config.points)
