"""HTTP routes.

Streaming is MJPEG: `multipart/x-mixed-replace`, which every browser renders in
a plain `<img>` tag with no client-side code, no plugin and no negotiation. At
256x192 the bandwidth cost is irrelevant and the debuggability is worth a great
deal -- a stream you can inspect with curl is a stream you can diagnose.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import time
from collections.abc import AsyncIterator, Awaitable
from dataclasses import asdict
from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import StreamingResponse

from irthermal.api.deps import Camera, CurrentFrame, Job, Jobs, Manager, SocketCamera
from irthermal.api.schemas import (
    AdjustmentsOut,
    CameraDetailOut,
    CameraStatusOut,
    HealthOut,
    LockInJobOut,
    LockInRequest,
    MeasurementOut,
    ParametersRequest,
    RangeRequest,
    RegionRequest,
    RegionStatsOut,
    TelemetryFrameOut,
    TelemetryOut,
    TemperaturesOut,
    celsius,
)
from irthermal.camera.base import CameraError, ControlUnsupportedError
from irthermal.camera.modulator import ModulatorError
from irthermal.core.lockin import LockInResult
from irthermal.core.render import (
    COLORMAPS,
    DEFAULT_COLORMAP,
    RenderOptions,
    UnknownColormapError,
    colorize,
    encode_png,
    normalize,
    resolve_colormap,
)
from irthermal.jobs import JobConflictError, LockInJob
from irthermal.logging import get_logger

log = get_logger(__name__)

router = APIRouter()

MJPEG_BOUNDARY = "frame"
MAX_UPSCALE = 8
DEFAULT_TELEMETRY_FPS = 4.0


async def _guard(operation: Awaitable[None]) -> None:
    """Report a camera fault as a service-state problem, not a server error.

    A control the transport cannot carry is a different thing from a camera that
    is merely down: retrying will never help, so it answers 501 rather than 503.
    """
    try:
        await operation
    except ControlUnsupportedError as exc:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)) from None
    except CameraError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from None


# --- render options ---------------------------------------------------------


def render_options(
    camera: Camera,
    colormap: Annotated[str | None, Query(description="Palette name.")] = None,
    min_celsius: Annotated[float | None, Query(description="Low end of the scale.")] = None,
    max_celsius: Annotated[float | None, Query(description="High end of the scale.")] = None,
    equalize: Annotated[bool, Query(description="Equalise the histogram.")] = False,
    upscale: Annotated[int, Query(ge=1, le=MAX_UPSCALE)] = 1,
) -> RenderOptions:
    """Build render settings from the query string, defaulting to the config."""
    options = RenderOptions(
        colormap=colormap or camera.config.colormap,
        min_celsius=min_celsius,
        max_celsius=max_celsius,
        equalize=equalize,
        upscale=upscale,
    )

    try:
        resolve_colormap(options.colormap)
    except UnknownColormapError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None

    return options


RenderQuery = Annotated[RenderOptions, Depends(render_options)]


# --- service ----------------------------------------------------------------


@router.get("/health", response_model=HealthOut, tags=["service"])
async def health(manager: Manager) -> HealthOut:
    """Liveness: the process is up and serving."""
    statuses = manager.statuses()
    connected = sum(1 for s in statuses if s.connected)
    return HealthOut(
        status="ok" if connected == len(statuses) else "degraded",
        cameras=len(statuses),
        connected=connected,
    )


@router.get("/ready", tags=["service"])
async def ready(manager: Manager) -> Response:
    """Readiness: every camera is connected.

    Deliberately separate from liveness. A camera being unplugged should not
    make an orchestrator restart a process that is otherwise working fine.
    """
    if not manager.healthy:
        offline = [s.id for s in manager.statuses() if not s.connected]
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"cameras not connected: {offline}"
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/colormaps", tags=["service"])
async def colormaps() -> list[str]:
    """Palettes accepted by the rendering endpoints."""
    return sorted(COLORMAPS)


# --- cameras ----------------------------------------------------------------


@router.get("/cameras", response_model=list[CameraStatusOut], tags=["cameras"])
async def list_cameras(manager: Manager) -> list[CameraStatusOut]:
    """Every configured camera and its capture health."""
    return [CameraStatusOut.of(status) for status in manager.statuses()]


@router.get("/cameras/{camera_id}", response_model=CameraDetailOut, tags=["cameras"])
async def get_camera_detail(camera: Camera) -> CameraDetailOut:
    """One camera, including whatever its latest frame reported."""
    detail = CameraDetailOut(**asdict(camera.status()))

    frame = camera.latest
    if frame is not None:
        detail.sequence = frame.sequence
        detail.corrected = frame.corrected
        detail.telemetry = TelemetryOut.of(frame.telemetry)
        detail.adjustments = AdjustmentsOut.of(camera.adjustments)

    return detail


# --- imagery ----------------------------------------------------------------


def _mjpeg_part(payload: bytes) -> bytes:
    return b"".join(
        (
            f"--{MJPEG_BOUNDARY}\r\n".encode(),
            b"Content-Type: image/jpeg\r\n",
            f"Content-Length: {len(payload)}\r\n\r\n".encode(),
            payload,
            b"\r\n",
        )
    )


@router.get("/cameras/{camera_id}/stream.mjpg", tags=["imagery"])
async def stream(
    camera: Camera,
    options: RenderQuery,
    fps: Annotated[float, Query(gt=0, le=60, description="Cap the delivered rate.")] = 25.0,
    quality: Annotated[int, Query(ge=1, le=100)] = 85,
) -> StreamingResponse:
    """Live MJPEG. Drop it straight into an `<img src=...>`."""
    interval = 1.0 / fps

    async def parts() -> AsyncIterator[bytes]:
        sent_at = 0.0
        async with camera.subscribe() as subscription:
            async for frame in subscription:
                now = time.monotonic()
                if now - sent_at < interval:
                    continue
                sent_at = now
                yield _mjpeg_part(camera.encode_frame(frame, options, quality=quality))

    return StreamingResponse(
        parts(),
        media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
        # Buffering a live stream defeats the point; nginx needs telling too.
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.get("/cameras/{camera_id}/snapshot.{fmt}", tags=["imagery"])
async def snapshot(
    camera: Camera,
    frame: CurrentFrame,
    options: RenderQuery,
    fmt: Literal["jpg", "jpeg", "png"],
    quality: Annotated[int, Query(ge=1, le=100)] = 85,
) -> Response:
    """A single rendered frame."""
    encoding = "png" if fmt == "png" else "jpeg"
    payload = camera.encode_frame(frame, options, fmt=encoding, quality=quality)
    return Response(
        content=payload,
        media_type=f"image/{encoding}",
        headers={"Cache-Control": "no-store", "X-Frame-Sequence": str(frame.sequence)},
    )


@router.get("/cameras/{camera_id}/frame.npy", tags=["imagery"])
async def raw_frame(frame: CurrentFrame) -> Response:
    """The uncompressed 16-bit pixels, for analysis rather than viewing.

    Readable with `numpy.load`, and the same shape v1 wrote with its `r` key.
    """
    buffer = io.BytesIO()
    np.save(buffer, frame.raw, allow_pickle=False)
    filename = f"{frame.camera_id}-{frame.sequence}.npy"
    return Response(
        content=buffer.getvalue(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Frame-Sequence": str(frame.sequence),
        },
    )


# --- measurement ------------------------------------------------------------


@router.get(
    "/cameras/{camera_id}/temperatures", response_model=TemperaturesOut, tags=["measurement"]
)
async def temperatures(camera: Camera, frame: CurrentFrame) -> TemperaturesOut:
    """Configured points, plus the frame's own extremes."""
    points = frame.measure((p.name, p.x, p.y) for p in camera.config.points)
    return TemperaturesOut.of(frame, points)


@router.get("/cameras/{camera_id}/temperature", response_model=MeasurementOut, tags=["measurement"])
async def temperature_at(
    frame: CurrentFrame,
    x: Annotated[int, Query(ge=0)],
    y: Annotated[int, Query(ge=0)],
) -> MeasurementOut:
    """Temperature at one pixel."""
    try:
        value = frame.temperature_at(x, y)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None

    return MeasurementOut(x=x, y=y, celsius=celsius(value))


@router.post("/cameras/{camera_id}/region", response_model=RegionStatsOut, tags=["measurement"])
async def region(frame: CurrentFrame, body: RegionRequest) -> RegionStatsOut:
    """Min, max and mean over a rectangle, clipped to the frame."""
    try:
        stats = frame.region_stats(body.x1, body.y1, body.x2, body.y2)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None

    return RegionStatsOut.of(stats)


# --- telemetry socket -------------------------------------------------------


async def _await_disconnect(websocket: WebSocket) -> None:
    """Return once the client goes away.

    A socket nobody reads from never learns that the peer has gone: the
    disconnect arrives on the receive channel, and if nothing is listening there
    the push loop keeps writing into a dead connection.
    """
    with contextlib.suppress(WebSocketDisconnect):
        while True:
            await websocket.receive()


@router.websocket("/cameras/{camera_id}/telemetry")
async def telemetry_socket(
    websocket: WebSocket,
    camera: SocketCamera,
    fps: Annotated[float, Query(gt=0, le=30)] = DEFAULT_TELEMETRY_FPS,
) -> None:
    """Push a measurement summary for each frame.

    The companion to the MJPEG stream: the video carries the picture, this
    carries the numbers over it. Sending them together in one encoded image
    would mean re-encoding per viewer and leave the readings unparseable.
    """
    await websocket.accept()
    disconnected = asyncio.create_task(_await_disconnect(websocket))
    interval = 1.0 / fps
    sent_at = 0.0

    try:
        async with camera.subscribe() as subscription:
            async for frame in subscription:
                if disconnected.done():
                    break

                now = time.monotonic()
                if now - sent_at < interval:
                    continue
                sent_at = now

                points = frame.measure((p.name, p.x, p.y) for p in camera.config.points)
                message = TelemetryFrameOut.of(
                    frame,
                    points,
                    fps=camera.status().fps,
                    adjustments=camera.adjustments,
                )
                await websocket.send_text(message.model_dump_json())
    except WebSocketDisconnect:
        pass
    finally:
        disconnected.cancel()


# --- control ----------------------------------------------------------------


@router.post(
    "/cameras/{camera_id}/calibrate", status_code=status.HTTP_202_ACCEPTED, tags=["control"]
)
async def calibrate(camera: Camera) -> dict[str, str]:
    """Trigger a shutter calibration -- the audible click."""
    await _guard(camera.calibrate())
    return {"status": "calibrated", "camera": camera.config.id}


@router.post("/cameras/{camera_id}/range", status_code=status.HTTP_202_ACCEPTED, tags=["control"])
async def set_range(camera: Camera, body: RangeRequest) -> dict[str, str]:
    """Switch between the normal and extended temperature ranges."""
    await _guard(camera.set_temperature_range(high=body.mode == "high"))
    return {"status": "ok", "mode": body.mode}


@router.patch("/cameras/{camera_id}/parameters", tags=["control"])
async def set_parameters(camera: Camera, body: ParametersRequest) -> AdjustmentsOut:
    """Write measurement parameters. Omitted fields are left as they are."""
    for name, value in body.device_parameters().items():
        await _guard(camera.set_parameter(name, value))

    if body.temperature_offset is not None:
        camera.set_temperature_offset(body.temperature_offset)

    return AdjustmentsOut.of(camera.adjustments)


# --- lock-in ----------------------------------------------------------------

LOCKIN_COMPONENTS = ("magnitude", "phase", "in_phase", "quadrature")
Component = Literal["magnitude", "phase", "in_phase", "quadrature"]


def component_of(result: LockInResult, name: Component) -> npt.NDArray[np.float64]:
    """One image out of a finished run."""
    if name == "magnitude":
        return result.magnitude
    if name == "phase":
        return result.phase_degrees
    return result.in_phase if name == "in_phase" else result.quadrature


def result_render_options(
    colormap: Annotated[str, Query(description="Palette name.")] = DEFAULT_COLORMAP,
    equalize: Annotated[bool, Query(description="Equalise the histogram.")] = False,
    upscale: Annotated[int, Query(ge=1, le=MAX_UPSCALE)] = 1,
) -> RenderOptions:
    """Render settings for a lock-in image.

    Separate from `render_options` because a lock-in component is not a
    temperature field: its units are sensor counts, degrees of phase or Celsius
    of swing depending on what was asked for, so a fixed Celsius window would be
    meaningless. These always auto-scale.
    """
    try:
        resolve_colormap(colormap)
    except UnknownColormapError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None

    return RenderOptions(colormap=colormap, equalize=equalize, upscale=upscale)


def finished_result(job: LockInJob) -> LockInResult:
    """The job's result, or an explanation of why there isn't one yet."""
    if job.result is not None:
        return job.result

    detail = job.error if job.state == "failed" else f"job {job.id} is {job.state}"
    code = (
        status.HTTP_409_CONFLICT
        if job.state == "running"
        else status.HTTP_422_UNPROCESSABLE_CONTENT
    )
    raise HTTPException(code, detail=detail or f"job {job.id} produced no result")


@router.post(
    "/cameras/{camera_id}/lockin",
    response_model=LockInJobOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["lock-in"],
)
async def start_lockin(camera: Camera, jobs: Jobs, body: LockInRequest) -> LockInJobOut:
    """Start a lock-in run. Returns immediately; poll the job for progress."""
    try:
        job = jobs.submit(
            camera.config.id,
            frequency_hz=body.frequency_hz,
            integration_s=body.integration_s,
            signal=body.signal,
        )
    except JobConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from None
    except ModulatorError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from None

    return LockInJobOut.of(job)


@router.get("/jobs", response_model=list[LockInJobOut], tags=["lock-in"])
async def list_jobs(jobs: Jobs) -> list[LockInJobOut]:
    """Every run this process still remembers, oldest first."""
    return [LockInJobOut.of(job) for job in jobs]


@router.get("/jobs/{job_id}", response_model=LockInJobOut, tags=["lock-in"])
async def get_job_status(job: Job) -> LockInJobOut:
    """One run's progress."""
    return LockInJobOut.of(job)


@router.delete("/jobs/{job_id}", response_model=LockInJobOut, tags=["lock-in"])
async def cancel_job(job: Job, jobs: Jobs) -> LockInJobOut:
    """Stop a run and release its load. Finished runs are left alone."""
    return LockInJobOut.of(await jobs.cancel(job.id))


@router.get("/jobs/{job_id}/result.npz", tags=["lock-in"])
async def lockin_arrays(job: Job) -> Response:
    """Both demodulated components, for analysis rather than viewing.

    Readable with `numpy.load`, which gives a mapping of `in_phase` and
    `quadrature`. Magnitude and phase are derived from those two, so storing
    them as well would only be a chance for the file to disagree with itself.
    """
    result = finished_result(job)

    buffer = io.BytesIO()
    np.savez_compressed(buffer, in_phase=result.in_phase, quadrature=result.quadrature)
    return Response(
        content=buffer.getvalue(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{job.camera_id}-{job.id}.npz"',
            "X-Lockin-Frames": str(result.frames),
            "X-Lockin-Frequency-Hz": str(result.frequency_hz),
        },
    )


@router.get("/jobs/{job_id}/result.png", tags=["lock-in"])
async def lockin_image(
    job: Job,
    options: Annotated[RenderOptions, Depends(result_render_options)],
    component: Annotated[Component, Query(description="Which image to render.")] = "magnitude",
) -> Response:
    """One component of a finished run, rendered."""
    result = finished_result(job)
    image = colorize(normalize(component_of(result, component), options), options)
    return Response(
        content=encode_png(image),
        media_type="image/png",
        headers={"Cache-Control": "no-store", "X-Lockin-Component": component},
    )
