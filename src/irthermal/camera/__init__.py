"""Camera sources, capture threads and their lifecycle."""

from irthermal.camera.base import (
    CameraError,
    CameraSource,
    CameraUnavailableError,
    FrameReadError,
    SourceFactory,
)
from irthermal.camera.emulator import EmulatorSource
from irthermal.camera.manager import CameraManager, UnknownCameraError
from irthermal.camera.runtime import CameraRuntime, CameraStatus, build_source_factory
from irthermal.camera.v4l2 import V4L2Source

__all__ = [
    "CameraError",
    "CameraManager",
    "CameraRuntime",
    "CameraSource",
    "CameraStatus",
    "CameraUnavailableError",
    "EmulatorSource",
    "FrameReadError",
    "SourceFactory",
    "UnknownCameraError",
    "V4L2Source",
    "build_source_factory",
]
