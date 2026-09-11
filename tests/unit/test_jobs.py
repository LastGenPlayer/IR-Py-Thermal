"""The lock-in job store: lifecycle, exclusivity and what it keeps.

Driven against emulated cameras, so these exercise the real driver -- the
subscription, the modulation task and the release of the load -- without a
camera or a serial port.
"""

from __future__ import annotations

import asyncio

import pytest

from conftest import FrameSpec, build_capture
from irthermal.camera import CameraManager, EmulatorSource
from irthermal.camera.base import CameraSource
from irthermal.camera.modulator import ExternalModulator
from irthermal.config import CameraConfig, Settings
from irthermal.core.resolutions import params_for_width
from irthermal.jobs import JobConflictError, JobStore, UnknownJobError

PARAMS = params_for_width(240)
REPLAY_FPS = 200.0


def emulator(spec: FrameSpec | None = None):
    capture = build_capture(PARAMS, spec)

    def build() -> CameraSource:
        return EmulatorSource.from_array(capture, fps=REPLAY_FPS)

    return build


@pytest.fixture
def settings() -> Settings:
    return Settings(
        cameras=[
            CameraConfig(id="rack", device="/dev/video0"),
            CameraConfig(id="bench", device="/dev/video1"),
        ]
    )


@pytest.fixture
async def store(settings: Settings):
    manager = CameraManager(settings, {"rack": emulator(), "bench": emulator()})
    async with manager:
        await manager.get("rack").wait_for_frame(timeout=5.0)
        await manager.get("bench").wait_for_frame(timeout=5.0)
        jobs = JobStore(manager)
        try:
            yield jobs
        finally:
            await jobs.close()


async def drain(store: JobStore, job_id: str, timeout: float = 10.0):
    """Wait for one job to reach a terminal state."""
    job = store.get(job_id)
    assert job.task is not None
    await asyncio.wait_for(asyncio.shield(job.task), timeout)
    return job


# --- lifecycle --------------------------------------------------------------


async def test_a_run_completes_and_keeps_its_result(store: JobStore) -> None:
    job = store.submit("rack", frequency_hz=5.0, integration_s=0.3)

    assert job.state == "running"
    await drain(store, job.id)

    assert job.state == "succeeded"
    assert job.error is None
    assert job.result is not None
    assert job.result.frames > 10
    assert job.result.in_phase.shape == (PARAMS.height, PARAMS.width)
    assert job.progress == 1.0


async def test_the_load_is_driven_and_then_released(store: JobStore) -> None:
    """Whatever happens, the load must not be left energised."""
    switch = ExternalModulator()
    job = store.submit("rack", frequency_hz=10.0, integration_s=0.3, modulator=switch)

    await drain(store, job.id)

    assert switch.states[0][1] is True, "the drive must start energised, which v1 never did"
    assert switch.states[-1][1] is False, "the load was left on"
    # 10 Hz over 0.3 s is six half periods, plus the release.
    assert len(switch.states) >= 5


async def test_a_celsius_run_measures_in_degrees(store: JobStore) -> None:
    job = store.submit("rack", frequency_hz=5.0, integration_s=0.3, signal="celsius")

    await drain(store, job.id)

    assert job.signal == "celsius"
    assert job.result is not None


async def test_progress_advances_before_it_finishes(store: JobStore) -> None:
    job = store.submit("rack", frequency_hz=5.0, integration_s=1.0)

    await asyncio.sleep(0.4)
    midway = job.progress

    await drain(store, job.id)

    assert 0.0 < midway < 1.0
    assert job.frames > 0


# --- exclusivity ------------------------------------------------------------


async def test_one_run_per_camera(store: JobStore) -> None:
    """Two runs would drive the same switch to opposite states."""
    store.submit("rack", frequency_hz=5.0, integration_s=1.0)

    with pytest.raises(JobConflictError, match="already running"):
        store.submit("rack", frequency_hz=5.0, integration_s=1.0)


async def test_other_cameras_are_unaffected(store: JobStore) -> None:
    first = store.submit("rack", frequency_hz=5.0, integration_s=0.3)
    second = store.submit("bench", frequency_hz=5.0, integration_s=0.3)

    await drain(store, first.id)
    await drain(store, second.id)

    assert {first.state, second.state} == {"succeeded"}


async def test_a_camera_is_free_again_once_its_run_ends(store: JobStore) -> None:
    first = store.submit("rack", frequency_hz=5.0, integration_s=0.2)
    await drain(store, first.id)

    second = store.submit("rack", frequency_hz=5.0, integration_s=0.2)

    assert second.id != first.id


# --- cancellation -----------------------------------------------------------


async def test_cancelling_stops_the_run_and_releases_the_load(store: JobStore) -> None:
    switch = ExternalModulator()
    job = store.submit("rack", frequency_hz=5.0, integration_s=30.0, modulator=switch)
    await asyncio.sleep(0.2)

    await store.cancel(job.id)

    assert job.state == "cancelled"
    assert job.result is None
    assert switch.states[-1][1] is False


async def test_cancelling_a_finished_run_changes_nothing(store: JobStore) -> None:
    job = store.submit("rack", frequency_hz=5.0, integration_s=0.2)
    await drain(store, job.id)

    await store.cancel(job.id)

    assert job.state == "succeeded"


async def test_closing_the_store_stops_everything(store: JobStore) -> None:
    switch = ExternalModulator()
    job = store.submit("rack", frequency_hz=5.0, integration_s=30.0, modulator=switch)
    await asyncio.sleep(0.15)

    await store.close()

    assert job.state == "cancelled"
    assert switch.states[-1][1] is False


# --- bookkeeping ------------------------------------------------------------


async def test_unknown_job(store: JobStore) -> None:
    with pytest.raises(UnknownJobError, match="nope"):
        store.get("nope")


async def test_finished_jobs_are_evicted_beyond_the_limit(settings: Settings) -> None:
    """Each result is two float64 images; keeping every one is a slow leak."""
    manager = CameraManager(settings, {"rack": emulator(), "bench": emulator()})
    async with manager:
        await manager.get("rack").wait_for_frame(timeout=5.0)
        store = JobStore(manager, history=2)

        ids = []
        for _ in range(4):
            job = store.submit("rack", frequency_hz=10.0, integration_s=0.1)
            await drain(store, job.id)
            ids.append(job.id)

        await store.close()

    assert len(store) <= 3
    assert ids[0] not in {job.id for job in store}
    assert ids[-1] in {job.id for job in store}
