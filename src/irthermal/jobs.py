"""Lock-in runs as background jobs.

A lock-in measurement takes as long as its integration time -- seconds for a
quick look, minutes for a faint signal -- so it cannot be an HTTP request that
blocks until it finishes. It is submitted, polled and collected instead.

Only one run per camera at a time. The load is a physical thing: two runs would
drive the same switch to opposite states and demodulate each other's heat.

Frames come from the hub like any other consumer, which is the point of having a
hub at all -- a lock-in run and a browser watching the stream share one capture
thread. The subscription is deeper than a viewer's, because a viewer wants the
newest frame and an integration wants every frame.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy.typing as npt

from irthermal.camera.manager import CameraManager
from irthermal.camera.modulator import ExternalModulator, Modulator, SerialModulator
from irthermal.camera.runtime import CameraRuntime
from irthermal.core.frame import Frame
from irthermal.core.lockin import LockInAccumulator, LockInError, LockInResult
from irthermal.logging import get_logger

log = get_logger(__name__)

JobState = Literal["running", "succeeded", "failed", "cancelled"]
SignalDomain = Literal["raw", "celsius"]

FINISHED_STATES: frozenset[str] = frozenset({"succeeded", "failed", "cancelled"})

QUEUE_DEPTH = 16
"""Frames the run may fall behind by before any are dropped."""

DEFAULT_HISTORY = 16
"""Finished jobs kept. Each holds two float64 images, so this is not free."""

FIRST_FRAME_TIMEOUT_S = 20.0


class UnknownJobError(KeyError):
    """No job by that id."""


class JobConflictError(RuntimeError):
    """That camera is already running one."""


@dataclass(slots=True)
class LockInJob:
    """One run: what was asked for, how far it got, and what came out."""

    id: str
    camera_id: str
    frequency_hz: float
    integration_s: float
    signal: SignalDomain
    modulator: str
    created_at: float
    """Unix time, for display. Durations are measured on the monotonic clock."""

    state: JobState = "running"
    frames: int = 0
    elapsed_s: float = 0.0
    error: str | None = None
    result: LockInResult | None = None
    task: asyncio.Task[None] | None = field(default=None, repr=False)
    switch: Modulator | None = field(default=None, repr=False)
    """The load switch itself. `modulator` above is only how it is described."""

    @property
    def progress(self) -> float:
        """Fraction of the integration completed, 0 to 1."""
        if self.state in FINISHED_STATES:
            return 1.0 if self.state == "succeeded" else self.elapsed_s / self.integration_s
        return min(self.elapsed_s / self.integration_s, 1.0)

    @property
    def finished(self) -> bool:
        return self.state in FINISHED_STATES


def signal_of(frame: Frame, domain: SignalDomain) -> npt.NDArray[Any]:
    """The pixels a run demodulates.

    `raw` is v1's choice and gives an amplitude in sensor counts. `celsius`
    gives one in degrees, which is the number anyone actually wants to quote,
    at the cost of the calibration curve's non-linearity across the swing.
    Pixels the calibration cannot express stay NaN rather than being filled in;
    they come out NaN in the result, which is the truth about them.
    """
    return frame.raw if domain == "raw" else frame.temperatures()


async def _modulate(modulator: Modulator, frequency_hz: float, reference_at: float) -> None:
    """Switch the load, starting energised at the reference instant.

    Toggling on its own timer rather than once per frame -- which is what v1
    did -- keeps the drive square regardless of frame rate. v1's schedule got
    coarse as the modulation frequency approached the camera's fps, because a
    half period shorter than a frame interval could not be expressed.

    v1 also never commanded the initial state: it assumed the load was already
    on and only wrote at the first toggle, so the first half period ran against
    whatever the hardware happened to be doing. This writes it.
    """
    half_period = 0.5 / frequency_hz
    on = True
    due = reference_at

    while True:
        delay = due - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

        await asyncio.to_thread(modulator.set, on=on)
        on = not on
        # Advancing by a fixed step rather than from "now", so a late write
        # does not push every later edge out with it.
        due += half_period


async def _integrate(job: LockInJob, camera: CameraRuntime, reference_at: float) -> LockInResult:
    """Fold frames into the accumulator until the integration time is up."""
    accumulator: LockInAccumulator | None = None

    async with camera.subscribe(maxsize=QUEUE_DEPTH) as subscription:
        async for frame in subscription:
            elapsed = frame.captured_at - reference_at
            if elapsed < 0.0:
                # Captured before the drive was commanded; it saw the old state.
                continue
            if elapsed > job.integration_s:
                break

            image = signal_of(frame, job.signal)
            if accumulator is None:
                accumulator = LockInAccumulator(job.frequency_hz, frame.raw.shape)

            accumulator.add(image, elapsed)
            job.frames = accumulator.frames
            job.elapsed_s = elapsed

    if accumulator is None:
        raise LockInError("the camera produced no frames during the integration")

    return accumulator.result()


async def _run(job: LockInJob, camera: CameraRuntime, modulator: Modulator) -> None:
    """Drive the load and integrate, then leave the load off whatever happens."""
    driver: asyncio.Task[None] | None = None
    try:
        # Wait for the camera before starting the clock, so a slow first frame
        # is not counted as integration time that produced nothing.
        await camera.wait_for_frame(timeout=FIRST_FRAME_TIMEOUT_S)

        reference_at = time.monotonic()
        driver = asyncio.create_task(_modulate(modulator, job.frequency_hz, reference_at))

        job.result = await _integrate(job, camera, reference_at)
        job.state = "succeeded"
        log.info(
            "lockin.finished",
            job=job.id,
            camera=job.camera_id,
            frames=job.result.frames,
            periods=round(job.result.periods, 2),
        )
    except asyncio.CancelledError:
        job.state = "cancelled"
        raise
    except Exception as exc:
        job.state = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
        log.warning("lockin.failed", job=job.id, camera=job.camera_id, error=job.error)
    finally:
        if driver is not None:
            driver.cancel()
        try:
            await asyncio.to_thread(modulator.set, on=False)
        except Exception as exc:  # pragma: no cover - the port just went away
            log.warning("lockin.release_failed", job=job.id, error=str(exc))
        modulator.close()


class JobStore:
    """Every lock-in run this process knows about."""

    def __init__(self, manager: CameraManager, *, history: int = DEFAULT_HISTORY) -> None:
        self._manager = manager
        self._history = history
        self._jobs: dict[str, LockInJob] = {}

    def __len__(self) -> int:
        return len(self._jobs)

    def __iter__(self) -> Iterator[LockInJob]:
        return iter(self._jobs.values())

    def get(self, job_id: str) -> LockInJob:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise UnknownJobError(f"unknown job {job_id!r}") from None

    def active(self, camera_id: str) -> LockInJob | None:
        """The run currently using a camera, if any."""
        return next(
            (job for job in self._jobs.values() if job.camera_id == camera_id and not job.finished),
            None,
        )

    def submit(
        self,
        camera_id: str,
        *,
        frequency_hz: float,
        integration_s: float,
        signal: SignalDomain = "raw",
        modulator: Modulator | None = None,
    ) -> LockInJob:
        """Start a run. `modulator` overrides the one the config describes."""
        running = self.active(camera_id)
        if running is not None:
            raise JobConflictError(f"camera {camera_id!r} is already running job {running.id}")

        camera = self._manager.get(camera_id)
        switch = modulator if modulator is not None else self._build_modulator(camera)

        job = LockInJob(
            id=uuid.uuid4().hex[:12],
            camera_id=camera_id,
            frequency_hz=frequency_hz,
            integration_s=integration_s,
            signal=signal,
            modulator=switch.description,
            created_at=time.time(),
            switch=switch,
        )
        job.task = asyncio.create_task(_run(job, camera, switch))

        self._prune()
        self._jobs[job.id] = job
        log.info(
            "lockin.started",
            job=job.id,
            camera=camera_id,
            frequency_hz=frequency_hz,
            integration_s=integration_s,
            modulator=job.modulator,
        )
        return job

    @staticmethod
    def _build_modulator(camera: CameraRuntime) -> Modulator:
        config = camera.config
        if config.lockin_port is None:
            return ExternalModulator()
        return SerialModulator.open(
            config.lockin_port, config.lockin_baudrate, invert=config.lockin_invert
        )

    async def cancel(self, job_id: str) -> LockInJob:
        """Stop a run. Already-finished jobs are left as they are."""
        job = self.get(job_id)
        if job.task is not None and not job.finished:
            await self._stop([job])
        return job

    async def close(self) -> None:
        """Cancel everything still running, and wait for the loads to release."""
        await self._stop([job for job in self._jobs.values() if not job.finished])

    @staticmethod
    async def _stop(jobs: list[LockInJob]) -> None:
        """Cancel these runs and wait for each to finish releasing its load.

        Gathered with `return_exceptions` because awaiting a cancelled task
        re-raises its CancelledError, and here that is the expected outcome
        rather than a failure to report.
        """
        tasks = [job.task for job in jobs if job.task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        for job in jobs:
            if job.finished:
                continue
            # Cancelled before the task ever reached its first await, so the
            # coroutine body never ran and neither did its cleanup. Nothing
            # else will ever close this job or the port it opened.
            job.state = "cancelled"
            if job.switch is not None:
                job.switch.close()

    def _prune(self) -> None:
        """Drop the oldest finished jobs once there are more than we keep."""
        finished = [job for job in self._jobs.values() if job.finished]
        for job in finished[: max(0, len(finished) - self._history)]:
            del self._jobs[job.id]
