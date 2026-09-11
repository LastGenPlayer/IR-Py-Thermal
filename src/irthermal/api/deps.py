"""Shared dependencies: reaching the manager, and resolving a camera id."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Path, Request, status

from irthermal.camera.base import CameraError
from irthermal.camera.manager import CameraManager, UnknownCameraError
from irthermal.camera.runtime import CameraRuntime
from irthermal.config import Settings
from irthermal.core.frame import Frame

FIRST_FRAME_TIMEOUT_S = 10.0


def get_manager(request: Request) -> CameraManager:
    """The application's camera manager."""
    manager: CameraManager = request.app.state.manager
    return manager


def get_settings(request: Request) -> Settings:
    """The application's settings."""
    settings: Settings = request.app.state.settings
    return settings


def get_camera(
    camera_id: Annotated[str, Path(description="Configured camera id.")],
    manager: Annotated[CameraManager, Depends(get_manager)],
) -> CameraRuntime:
    """Resolve a camera id, or 404 naming the ones that do exist."""
    try:
        return manager.get(camera_id)
    except UnknownCameraError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc).strip("'")) from None


Camera = Annotated[CameraRuntime, Depends(get_camera)]
Manager = Annotated[CameraManager, Depends(get_manager)]
AppSettings = Annotated[Settings, Depends(get_settings)]


async def get_frame(camera: Camera) -> Frame:
    """The camera's current frame, or 503 if it has not produced one.

    A camera that is unplugged or still warming up is a service-state problem,
    not a malformed request.
    """
    frame = camera.latest
    if frame is not None:
        return frame

    try:
        return await camera.wait_for_frame(timeout=FIRST_FRAME_TIMEOUT_S)
    except (TimeoutError, CameraError) as exc:
        connected = camera.status().connected
        detail = (
            f"camera {camera.config.id!r} has produced no frames"
            f"{'' if connected else ' and is not connected'}"
        )
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail) from exc


CurrentFrame = Annotated[Frame, Depends(get_frame)]
