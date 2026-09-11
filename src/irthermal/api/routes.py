"""HTTP routes.

Streaming is MJPEG: `multipart/x-mixed-replace`, which every browser renders in
a plain `<img>` tag with no client-side code, no plugin and no negotiation. At
256x192 the bandwidth cost is irrelevant and the debuggability is worth a great
deal -- a stream you can inspect with curl is a stream you can diagnose.
"""

from __future__ import annotations

import io
import time
from collections.abc import AsyncIterator, Awaitable
from dataclasses import asdict
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse

from irthermal.api.deps import Camera, CurrentFrame, Manager
from irthermal.api.schemas import (
    AdjustmentsOut,
    CameraDetailOut,
    CameraStatusOut,
    HealthOut,
    MeasurementOut,
    ParametersRequest,
    RangeRequest,
    RegionRequest,
    RegionStatsOut,
    TelemetryOut,
    TemperaturesOut,
    celsius,
)
from irthermal.camera.base import CameraError
from irthermal.core.render import COLORMAPS, RenderOptions, UnknownColormapError, resolve_colormap
from irthermal.logging import get_logger

log = get_logger(__name__)

router = APIRouter()

MJPEG_BOUNDARY = "frame"
MAX_UPSCALE = 8


async def _guard(operation: Awaitable[None]) -> None:
    """Report a camera fault as a service-state problem, not a server error."""
    try:
        await operation
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
