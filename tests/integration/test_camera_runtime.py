"""End-to-end capture, driven entirely by emulated sources.

Nothing here touches hardware: CI runs the full pipeline -- source, capture
thread, hub, runtime, manager -- against replayed captures.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from conftest import FrameSpec, build_capture
from irthermal.camera import CameraManager, CameraRuntime, EmulatorSource, UnknownCameraError
from irthermal.camera.base import CameraSource, CameraUnavailableError, FrameReadError
from irthermal.config import CameraConfig, PointConfig, Settings
from irthermal.core import commands
from irthermal.core.resolutions import (
    ResolutionParams,
    UnsupportedResolutionError,
    params_for_width,
)

REPLAY_FPS = 200.0
"""Fast enough to keep tests short, slow enough to exercise the pacing path."""


def emulator_factory(
    params: ResolutionParams, spec: FrameSpec | None = None, *, camera_raw: bool = False
):
    """A factory serving a synthetic capture."""

    def build() -> CameraSource:
        return EmulatorSource.from_array(
            build_capture(params, spec), camera_raw=camera_raw, fps=REPLAY_FPS
        )

    return build


def camera_config(camera_id: str, **kwargs: object) -> CameraConfig:
    return CameraConfig(id=camera_id, device="/dev/video0", **kwargs)  # type: ignore[arg-type]


# --- the emulator source ----------------------------------------------------


def test_emulator_reads_resolution_from_the_capture(params: ResolutionParams) -> None:
    source = EmulatorSource.from_array(build_capture(params), fps=0)

    assert source.params.width == params.width
    assert source.params.height == params.height
    assert source.read_words().size == params.total_words


def test_emulator_round_trips_a_saved_file(tmp_path: Path, small_params: ResolutionParams) -> None:
    """v1 wrote these with its `r` key; they must still load."""
    path = tmp_path / "capture.npy"
    np.save(path, build_capture(small_params))

    source = EmulatorSource.from_file(path, fps=0)
    assert source.params.height == small_params.height


def test_emulator_rejects_a_capture_missing_its_telemetry_rows(
    small_params: ResolutionParams,
) -> None:
    visible_only = build_capture(small_params)[: small_params.height]
    with pytest.raises(UnsupportedResolutionError, match=r"240x176 but width 240 implies"):
        EmulatorSource.from_array(visible_only)


def test_emulator_rejects_a_capture_with_too_few_rows() -> None:
    with pytest.raises(CameraUnavailableError, match="too few"):
        EmulatorSource.from_array(np.zeros((2, 240), dtype=np.uint16))


def test_emulator_rejects_a_capture_of_the_wrong_shape() -> None:
    with pytest.raises(CameraUnavailableError, match="2-dimensional"):
        EmulatorSource.from_array(np.zeros(240, dtype=np.uint16))


def test_emulator_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CameraUnavailableError, match="not found"):
        EmulatorSource.from_file(tmp_path / "absent.npy")


# --- a single runtime -------------------------------------------------------


async def test_runtime_produces_frames(small_params: ResolutionParams) -> None:
    runtime = CameraRuntime(camera_config("cam"), emulator_factory(small_params))
    async with runtime:
        frame = await runtime.wait_for_frame(timeout=5.0)

    assert frame.camera_id == "cam"
    assert (frame.width, frame.height) == (small_params.width, small_params.height)
    assert frame.sequence >= 1


async def test_frames_keep_arriving_with_increasing_sequence(
    small_params: ResolutionParams,
) -> None:
    runtime = CameraRuntime(camera_config("cam"), emulator_factory(small_params))
    async with runtime, runtime.subscribe() as subscription:
        received = [
            frame.sequence
            for frame in [await asyncio.wait_for(subscription.get(), 5.0) for _ in range(3)]
            if frame
        ]

    assert len(received) == 3
    assert received == sorted(received)
    assert len(set(received)) == 3


async def test_frame_pixels_and_telemetry_come_from_the_same_capture(
    small_params: ResolutionParams,
) -> None:
    """v1 could pair a corrected image with telemetry decoded from another read."""
    spec = FrameSpec(serial="PAIRED")
    runtime = CameraRuntime(camera_config("cam"), emulator_factory(small_params, spec))

    async with runtime:
        frame = await runtime.wait_for_frame(timeout=5.0)

    assert frame.telemetry.serial == "PAIRED"
    expected = float(frame.table[frame.raw[5, 5]])
    assert frame.temperature_at(5, 5) == pytest.approx(expected)


async def test_status_reports_connection_and_throughput(small_params: ResolutionParams) -> None:
    runtime = CameraRuntime(camera_config("cam"), emulator_factory(small_params))
    async with runtime:
        await runtime.wait_for_frame(timeout=5.0)
        await asyncio.sleep(0.05)
        status = runtime.status()

    assert status.id == "cam"
    assert status.connected is True
    assert status.frames >= 1
    assert status.width == small_params.width
    assert status.last_error is None


async def test_temperature_offset_shifts_reported_temperatures(
    small_params: ResolutionParams,
) -> None:
    runtime = CameraRuntime(camera_config("cam"), emulator_factory(small_params))
    async with runtime:
        before = (await runtime.wait_for_frame(timeout=5.0)).temperature_at(5, 5)
        runtime.set_temperature_offset(10.0)

        async with runtime.subscribe() as subscription:
            # Skip frames still carrying the previous table.
            for _ in range(5):
                frame = await asyncio.wait_for(subscription.get(), 5.0)
                assert frame is not None
                after = frame.temperature_at(5, 5)
                if after == pytest.approx(before + 10.0):
                    break
            else:
                pytest.fail("offset never took effect")


async def test_configured_points_are_measured(small_params: ResolutionParams) -> None:
    config = camera_config(
        "cam",
        points=[
            PointConfig(name="a", x=10, y=10),
            PointConfig(name="b", x=20, y=30),
            PointConfig(name="off-frame", x=9_000, y=10),
        ],
    )
    runtime = CameraRuntime(config, emulator_factory(small_params))

    async with runtime:
        measured = await runtime.measure_points()

    assert set(measured) == {"a", "b"}
    assert (measured["b"].x, measured["b"].y) == (20, 30)


# --- control ----------------------------------------------------------------


async def test_calibrate_reaches_the_device(small_params: ResolutionParams) -> None:
    sources: list[EmulatorSource] = []

    def factory() -> CameraSource:
        source = EmulatorSource.from_array(build_capture(small_params), fps=REPLAY_FPS)
        sources.append(source)
        return source

    runtime = CameraRuntime(camera_config("cam"), factory)
    async with runtime:
        await runtime.wait_for_frame(timeout=5.0)
        await runtime.calibrate()

    assert commands.ACTION_SHUTTER_CALIBRATE in sources[0].commands


async def test_setting_the_high_range_applies_its_gain(small_params: ResolutionParams) -> None:
    sources: list[EmulatorSource] = []

    def factory() -> CameraSource:
        source = EmulatorSource.from_array(build_capture(small_params), fps=REPLAY_FPS)
        sources.append(source)
        return source

    runtime = CameraRuntime(camera_config("cam"), factory)
    async with runtime:
        await runtime.wait_for_frame(timeout=5.0)
        await runtime.set_temperature_range(high=True)

        assert commands.ACTION_RANGE_HIGH in sources[0].commands
        assert runtime.adjustments.gain_m == pytest.approx(1.17)

        await runtime.set_temperature_range(high=False)
        assert runtime.adjustments.gain_m == pytest.approx(1.0)


async def test_setting_a_parameter_sends_every_byte(small_params: ResolutionParams) -> None:
    sources: list[EmulatorSource] = []

    def factory() -> CameraSource:
        source = EmulatorSource.from_array(build_capture(small_params), fps=REPLAY_FPS)
        sources.append(source)
        return source

    runtime = CameraRuntime(camera_config("cam"), factory)
    async with runtime:
        await runtime.wait_for_frame(timeout=5.0)
        await runtime.set_parameter("emissivity", 0.95)

    expected = commands.float_command(commands.PARAM_EMISSIVITY, 0.95)
    assert all(payload in sources[0].commands for payload in expected)


async def test_unknown_parameter_is_rejected(small_params: ResolutionParams) -> None:
    runtime = CameraRuntime(camera_config("cam"), emulator_factory(small_params))
    async with runtime:
        with pytest.raises(ValueError, match="unknown parameter"):
            await runtime.set_parameter("nonsense", 1.0)


# --- failure handling -------------------------------------------------------


async def test_a_camera_that_never_opens_stays_unconnected(
    small_params: ResolutionParams,
) -> None:
    """An absent camera must leave the service running and reporting it."""

    def failing_factory() -> CameraSource:
        raise CameraUnavailableError("no such device")

    runtime = CameraRuntime(camera_config("cam"), failing_factory)
    async with runtime:
        await asyncio.sleep(0.2)
        status = runtime.status()

    assert status.connected is False
    assert status.frames == 0
    assert status.last_error is not None
    assert "no such device" in status.last_error


async def test_capture_recovers_after_transient_read_failures(
    small_params: ResolutionParams,
) -> None:
    capture = build_capture(small_params)

    class FlakySource(EmulatorSource):
        def __init__(self) -> None:
            super().__init__(
                np.ascontiguousarray(capture, dtype=np.uint16).ravel(),
                small_params,
                description="flaky",
                fps=REPLAY_FPS,
            )
            self.attempts = 0

        def read_words(self):  # type: ignore[no-untyped-def]
            self.attempts += 1
            if self.attempts <= 3:
                raise FrameReadError("transient glitch")
            return super().read_words()

    source = FlakySource()
    runtime = CameraRuntime(camera_config("cam"), lambda: source)

    async with runtime:
        frame = await runtime.wait_for_frame(timeout=5.0)
        status = runtime.status()

    assert frame.sequence >= 1
    assert status.read_errors >= 3


async def test_stopping_ends_subscriptions(small_params: ResolutionParams) -> None:
    runtime = CameraRuntime(camera_config("cam"), emulator_factory(small_params))
    await runtime.start()
    subscription = runtime.subscribe()
    await runtime.wait_for_frame(timeout=5.0)

    await runtime.stop()

    async def drain() -> None:
        async for _ in subscription:
            pass

    await asyncio.wait_for(drain(), timeout=2.0)


async def test_starting_twice_is_refused(small_params: ResolutionParams) -> None:
    runtime = CameraRuntime(camera_config("cam"), emulator_factory(small_params))
    async with runtime:
        with pytest.raises(RuntimeError, match="already started"):
            await runtime.start()


# --- several cameras at once ------------------------------------------------


async def test_manager_runs_every_camera_concurrently() -> None:
    """The point of multi-camera: independent streams in one process."""
    settings = Settings(
        cameras=[
            camera_config("wide"),
            camera_config("narrow"),
            camera_config("big"),
        ]
    )
    factories = {
        "wide": emulator_factory(params_for_width(240)),
        "narrow": emulator_factory(params_for_width(256)),
        "big": emulator_factory(params_for_width(384)),
    }

    async with CameraManager(settings, factories) as manager:
        frames = await asyncio.gather(*(runtime.wait_for_frame(timeout=5.0) for runtime in manager))

        assert manager.ids == ["wide", "narrow", "big"]
        assert manager.healthy is True
        assert [f.width for f in frames] == [240, 256, 384]
        assert {f.camera_id for f in frames} == {"wide", "narrow", "big"}


async def test_cameras_have_independent_frame_streams() -> None:
    settings = Settings(cameras=[camera_config("a"), camera_config("b")])
    factories = {
        "a": emulator_factory(params_for_width(240), FrameSpec(serial="AAA")),
        "b": emulator_factory(params_for_width(256), FrameSpec(serial="BBB")),
    }

    async with CameraManager(settings, factories) as manager:
        first = await manager.get("a").wait_for_frame(timeout=5.0)
        second = await manager.get("b").wait_for_frame(timeout=5.0)

    assert first.telemetry.serial == "AAA"
    assert second.telemetry.serial == "BBB"


async def test_one_failing_camera_does_not_stop_the_others() -> None:
    settings = Settings(cameras=[camera_config("good"), camera_config("bad")])

    def broken() -> CameraSource:
        raise CameraUnavailableError("unplugged")

    factories = {"good": emulator_factory(params_for_width(240)), "bad": broken}

    async with CameraManager(settings, factories) as manager:
        frame = await manager.get("good").wait_for_frame(timeout=5.0)
        await asyncio.sleep(0.2)

        assert frame.camera_id == "good"
        assert manager.healthy is False
        statuses = {s.id: s.connected for s in manager.statuses()}
        assert statuses == {"good": True, "bad": False}


async def test_unknown_camera_id_is_reported_clearly() -> None:
    settings = Settings(cameras=[camera_config("only")])
    manager = CameraManager(settings, {"only": emulator_factory(params_for_width(240))})

    with pytest.raises(UnknownCameraError, match="unknown camera"):
        manager.get("missing")
