"""Camera sources, capture threads and their lifecycle."""

from irthermal.camera.base import (
    CameraError,
    CameraSource,
    CameraUnavailableError,
    ControlUnsupportedError,
    FrameReadError,
    SourceFactory,
)
from irthermal.camera.emulator import EmulatorSource
from irthermal.camera.manager import CameraManager, UnknownCameraError
from irthermal.camera.runtime import CameraRuntime, CameraStatus, build_source_factory
from irthermal.camera.transport import Transport, open_transport
from irthermal.camera.uvc import UvcSource

__all__ = [
    "CameraError",
    "CameraManager",
    "CameraRuntime",
    "CameraSource",
    "CameraStatus",
    "CameraUnavailableError",
    "ControlUnsupportedError",
    "EmulatorSource",
    "FrameReadError",
    "SourceFactory",
    "Transport",
    "UnknownCameraError",
    "UvcSource",
    "build_source_factory",
    "open_transport",
]
