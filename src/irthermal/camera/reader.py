"""The capture thread.

One thread per camera does the blocking work -- `VideoCapture.read()` is not
async and never will be -- and hands finished frames to the hub. Everything that
needs to touch the device goes through this thread as a queued command, so the
device is only ever used from one place and no lock is needed around OpenCV.

The thread owns reconnection too. A camera that gets unplugged should leave the
service running and reporting itself unhealthy, not take the process down with
it, which is what v1 did.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import numpy as np

from irthermal.camera.base import (
    CameraError,
    CameraSource,
    CameraUnavailableError,
    FrameReadError,
    SourceFactory,
)
from irthermal.camera.processing import RawCorrector, clamp_to_table
from irthermal.core.commands import ACTION_SHUTTER_CALIBRATE
from irthermal.core.frame import Frame
from irthermal.core.radiometry import RadiometryAdjustments, RadiometryInputs, temperature_table
from irthermal.core.telemetry import CameraTelemetry, FrameWords, parse_telemetry
from irthermal.hub import FrameHub
from irthermal.logging import get_logger

log = get_logger(__name__)

RECONNECT_DELAY_S = 1.0
MAX_RECONNECT_DELAY_S = 30.0
READ_ERROR_DELAY_S = 0.1
CONSECUTIVE_ERROR_LIMIT = 10
"""Read failures in a row before the source is treated as lost and reopened."""

_FPS_SMOOTHING = 0.9


@dataclass(frozen=True, slots=True)
class ReaderStats:
    """A snapshot of one camera's capture health."""

    connected: bool = False
    frames: int = 0
    read_errors: int = 0
    reconnects: int = 0
    fps: float = 0.0
    last_error: str | None = None
    description: str | None = None


Command = Callable[["FrameReader"], Any]
"""Runs on the capture thread, with the device and corrector available."""


class FrameReader:
    """Captures from one source and publishes to one hub."""

    def __init__(
        self,
        camera_id: str,
        factory: SourceFactory,
        hub: FrameHub,
        *,
        adjustments: RadiometryAdjustments | None = None,
        calibration_file: Path | None = None,
    ) -> None:
        self.camera_id = camera_id
        self._factory = factory
        self._hub = hub
        self._calibration_file = calibration_file

        # Replaced wholesale, never mutated, so the capture thread always sees a
        # coherent set without locking.
        self.adjustments = adjustments or RadiometryAdjustments()

        self.source: CameraSource | None = None
        self.corrector = RawCorrector()

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._commands: Queue[tuple[Command, Future[Any]]] = Queue()

        self._stats = ReaderStats()
        self._stats_lock = threading.Lock()
        self._sequence = 0
        self._last_capture: float | None = None

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Launch the capture thread."""
        if self._thread is not None:
            raise RuntimeError(f"reader for {self.camera_id} already started")

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"capture-{self.camera_id}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the thread to finish and wait for it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._drain_pending_commands()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def stats(self) -> ReaderStats:
        with self._stats_lock:
            return self._stats

    def _update_stats(self, **changes: Any) -> None:
        with self._stats_lock:
            self._stats = replace(self._stats, **changes)

    # --- commands -----------------------------------------------------------

    def submit(self, command: Command) -> Future[Any]:
        """Queue work for the capture thread. Resolves once it has run."""
        future: Future[Any] = Future()
        if self._stop.is_set():
            future.set_exception(CameraError(f"reader for {self.camera_id} is stopping"))
            return future

        self._commands.put((command, future))
        return future

    def _drain_commands(self) -> None:
        while True:
            try:
                command, future = self._commands.get_nowait()
            except Empty:
                return

            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(command(self))
            except Exception as exc:
                future.set_exception(exc)

    def _drain_pending_commands(self) -> None:
        """Fail anything still queued once the thread is gone."""
        while True:
            try:
                _, future = self._commands.get_nowait()
            except Empty:
                return
            if not future.done():
                future.set_exception(CameraError(f"reader for {self.camera_id} stopped"))

    # --- capture loop -------------------------------------------------------

    def _run(self) -> None:
        reconnect_delay = RECONNECT_DELAY_S
        consecutive_errors = 0

        while not self._stop.is_set():
            if self.source is None:
                if not self._open_source():
                    if self._stop.wait(reconnect_delay):
                        break
                    reconnect_delay = min(reconnect_delay * 2, MAX_RECONNECT_DELAY_S)
                    continue
                reconnect_delay = RECONNECT_DELAY_S
                consecutive_errors = 0

            self._drain_commands()
            if self._stop.is_set():
                break

            try:
                self._capture_once()
            except FrameReadError as exc:
                consecutive_errors += 1
                self._note_error(exc)
                if consecutive_errors >= CONSECUTIVE_ERROR_LIMIT:
                    log.warning(
                        "camera.read_errors_exceeded",
                        camera=self.camera_id,
                        errors=consecutive_errors,
                    )
                    self._close_source(reconnect=True)
                    consecutive_errors = 0
                elif self._stop.wait(READ_ERROR_DELAY_S):
                    break
            except CameraUnavailableError as exc:
                self._note_error(exc)
                log.warning("camera.lost", camera=self.camera_id, error=str(exc))
                self._close_source(reconnect=True)
            else:
                consecutive_errors = 0

        self._close_source(reconnect=False)

    def _open_source(self) -> bool:
        try:
            self.source = self._factory()
        except CameraError as exc:
            self._note_error(exc)
            self._update_stats(connected=False)
            log.warning("camera.open_failed", camera=self.camera_id, error=str(exc))
            return False

        self.corrector.reset()
        self._update_stats(connected=True, last_error=None, description=self.source.description)
        log.info("camera.connected", camera=self.camera_id, source=self.source.description)

        if self.source.camera_raw:
            self._calibrate_on_connect()
        return True

    def _close_source(self, *, reconnect: bool) -> None:
        if self.source is not None:
            try:
                self.source.close()
            except Exception as exc:
                log.debug("camera.close_failed", camera=self.camera_id, error=str(exc))
            self.source = None

        self.corrector.reset()
        self._update_stats(connected=False, fps=0.0)
        self._last_capture = None

        if reconnect:
            with self._stats_lock:
                self._stats = replace(self._stats, reconnects=self._stats.reconnects + 1)

    def _capture_once(self) -> None:
        """Read, decode and publish exactly one frame."""
        source = self.source
        if source is None:  # pragma: no cover - loop guarantees a source
            raise CameraUnavailableError("no source")

        words = source.read_words()
        params = source.params

        telemetry = parse_telemetry(words, params, camera_raw=source.camera_raw)
        visible = words[: params.pixel_count].reshape(params.height, params.width).copy()

        if source.camera_raw:
            image = self.corrector.apply(visible)
            corrected = self.corrector.calibrated
        else:
            image = clamp_to_table(visible)
            corrected = False

        image.setflags(write=False)

        self._sequence += 1
        now = time.monotonic()
        frame = Frame(
            camera_id=self.camera_id,
            sequence=self._sequence,
            captured_at=now,
            raw=image,
            telemetry=telemetry,
            table=temperature_table(
                RadiometryInputs.from_telemetry(telemetry, params, self.adjustments)
            ),
            corrected=corrected,
        )

        self._record_capture(now)
        self._hub.publish_from_thread(frame)

    def _record_capture(self, now: float) -> None:
        fps = self._stats.fps
        if self._last_capture is not None:
            interval = now - self._last_capture
            if interval > 0:
                instant = 1.0 / interval
                fps = (
                    instant if fps == 0.0 else _FPS_SMOOTHING * fps + (1 - _FPS_SMOOTHING) * instant
                )
        self._last_capture = now

        with self._stats_lock:
            self._stats = replace(self._stats, frames=self._stats.frames + 1, fps=fps)

    def _note_error(self, exc: Exception) -> None:
        with self._stats_lock:
            self._stats = replace(
                self._stats, read_errors=self._stats.read_errors + 1, last_error=str(exc)
            )

    # --- operations that run on the capture thread --------------------------

    def _calibrate_on_connect(self) -> None:
        """Flat-field a raw camera, by the best means this transport allows.

        A fresh reference beats a stored one, so a camera whose shutter can be
        driven is calibrated outright. Without a control channel there is no way
        to close the shutter, and a reference taken against an open one would
        capture the scene rather than a uniform field -- then subtract that scene
        from every later frame. Such a camera falls back to whatever was stored
        last time, and failing that streams uncorrected.
        """
        source = self.source
        if source is None:  # pragma: no cover - caller guarantees one
            return

        if source.supports_control:
            try:
                self._calibrate_raw()
            except CameraError as exc:
                # A camera that cannot calibrate is still worth streaming.
                log.warning("camera.calibration_failed", camera=self.camera_id, error=str(exc))
            return

        if self._load_calibration():
            return

        hint = (
            "cover the lens and POST /calibrate"
            if self._calibration_file is None
            else f"cover the lens and POST /calibrate; it will be kept in {self._calibration_file}"
        )
        log.warning(
            "camera.calibration_skipped",
            camera=self.camera_id,
            source=source.description,
            reason=f"transport carries no control channel; {hint}",
        )

    def _load_calibration(self) -> bool:
        """Adopt a stored reference, if one matching this sensor exists."""
        path = self._calibration_file
        source = self.source
        if path is None or source is None:
            return False

        params = source.params
        record = self.corrector.load(path, shape=(params.height, params.width))
        if record is None:
            return False

        log.info(
            "camera.calibration_loaded",
            camera=self.camera_id,
            path=str(path),
            age_s=round(record.age_s, 1),
            dead_pixels=record.dead_pixels,
            fpa_temp_c=round(record.fpa_temp_c, 2),
        )
        return True

    def _store_calibration(self, words: FrameWords) -> None:
        """Persist the reference just taken, so the next run need not ask again."""
        path = self._calibration_file
        source = self.source
        if path is None or source is None:
            return

        telemetry = parse_telemetry(words, source.params, camera_raw=source.camera_raw)
        try:
            self.corrector.save(path, fpa_temp_c=telemetry.fpa_temp_c)
        except OSError as exc:
            # Losing the file costs a manual calibration next time, nothing more.
            log.warning(
                "camera.calibration_save_failed",
                camera=self.camera_id,
                path=str(path),
                error=str(exc),
            )
            return

        log.info("camera.calibration_saved", camera=self.camera_id, path=str(path))

    def _calibrate_raw(self, *, settle_frames: int = 8, use_shutter: bool = True) -> None:
        """Close the shutter, capture a reference frame, build the dead-pixel mask.

        `use_shutter` is False when the transport cannot command one; the caller
        is then responsible for blocking the lens before asking.
        """
        source = self.source
        if source is None:
            raise CameraUnavailableError("cannot calibrate without a source")

        self.corrector.reset()
        if use_shutter:
            source.send_command(ACTION_SHUTTER_CALIBRATE)

        # Let the shutter close and the pipeline flush before sampling.
        for _ in range(settle_frames):
            source.read_words()

        words = source.read_words()
        params = source.params
        reference = words[: params.pixel_count].reshape(params.height, params.width).copy()
        self.corrector.calibrate(reference)
        self._store_calibration(words)

        log.info(
            "camera.calibrated",
            camera=self.camera_id,
            dead_pixels=self.corrector.dead_pixel_count,
            mean=float(np.mean(reference)),
        )

    def latest_telemetry(self) -> CameraTelemetry | None:
        """Telemetry from the most recent published frame, if any."""
        frame = self._hub.latest
        return frame.telemetry if frame is not None else None
