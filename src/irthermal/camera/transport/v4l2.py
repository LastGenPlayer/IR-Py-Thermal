"""Frames over V4L2, through OpenCV.

The fast path on Linux: one `VideoCapture` handle carries both the pixels and
the control channel, since these cameras smuggle commands through the UVC zoom
control rather than exposing an endpoint of their own.

It is deliberately not the default anywhere else. OpenCV's Windows backends
(DirectShow, Media Foundation) decode the sensor's 16-bit words to BGR before
handing them over, and accept `CAP_PROP_CONVERT_RGB = 0` while ignoring it, so
the loss is silent. `read_words` detects that and says what to switch to instead
of failing deep in a NumPy view.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from irthermal.camera.base import (
    CameraUnavailableError,
    ControlUnsupportedError,
    FrameReadError,
)
from irthermal.camera.transport.base import DeviceGeometry
from irthermal.core.resolutions import SPECIAL_ROWS, is_supported
from irthermal.core.telemetry import FrameWords
from irthermal.logging import get_logger

log = get_logger(__name__)

DISCOVERY_RANGE = 10
"""How many video device indices to probe when auto-detecting."""


def _open_capture(target: int | str) -> cv2.VideoCapture:
    """Open a device by index or path, using V4L2 where it exists."""
    if sys.platform.startswith("linux"):
        return cv2.VideoCapture(target, cv2.CAP_V4L2)
    return cv2.VideoCapture(target)


class V4L2Transport:
    """A camera held open through OpenCV, control channel included."""

    def __init__(self, capture: cv2.VideoCapture, *, description: str) -> None:
        self._capture = capture
        self._description = description
        self._geometry = DeviceGeometry(
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            rows=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @classmethod
    def open(cls, device: str) -> V4L2Transport:
        """Open a specific device path."""
        capture = _open_capture(device)
        if not capture.isOpened():
            msg = f"cannot open video device {device}"
            raise CameraUnavailableError(msg)
        return cls(capture, description=device)

    @classmethod
    def discover(cls) -> V4L2Transport:
        """Probe video devices and take the first with a supported resolution."""
        seen: list[str] = []

        for index in range(DISCOVERY_RANGE):
            capture = _open_capture(index)
            if not capture.isOpened():
                capture.release()
                continue

            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            rows = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            seen.append(f"{index}:{width}x{rows}")

            if is_supported(width, rows - SPECIAL_ROWS):
                log.info("transport.v4l2.discovered", index=index, width=width, rows=rows)
                return cls(capture, description=f"/dev/video{index}")

            capture.release()

        msg = (
            "no supported thermal camera found. Devices seen (index:WxH): "
            f"{', '.join(seen) or 'none'}"
        )
        raise CameraUnavailableError(msg)

    @property
    def description(self) -> str:
        return self._description

    @property
    def geometry(self) -> DeviceGeometry:
        return self._geometry

    @property
    def supports_control(self) -> bool:
        return True

    def configure_stream(self, payload: int) -> None:
        """Ask for raw 16-bit words rather than decoded video."""
        self._capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self._capture.set(cv2.CAP_PROP_ZOOM, payload)

    def read_words(self) -> FrameWords:
        ok, buffer = self._capture.read()
        if not ok or buffer is None:
            raise FrameReadError("camera returned no frame")

        # A backend that honoured CAP_PROP_CONVERT_RGB leaves an even number of
        # bytes on the last axis. One that decoded to BGR leaves three.
        last_axis_bytes = buffer.shape[-1] * buffer.itemsize
        if last_axis_bytes % 2:
            msg = (
                f"{self._description} delivered decoded {buffer.shape} {buffer.dtype} video "
                "rather than raw 16-bit words: this OpenCV backend ignores "
                'CAP_PROP_CONVERT_RGB. Use transport = "ffmpeg", which reads the '
                "native pixel format untouched."
            )
            raise CameraUnavailableError(msg)

        words: FrameWords = buffer.view(np.uint16).ravel()
        return words

    def send_command(self, payload: int) -> bool:
        if self._capture is None:  # pragma: no cover - close() guards this
            raise ControlUnsupportedError("capture is closed")
        return bool(self._capture.set(cv2.CAP_PROP_ZOOM, payload))

    def close(self) -> None:
        self._capture.release()
