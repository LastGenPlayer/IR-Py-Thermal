"""The real camera, over V4L2/UVC.

Control happens through the zoom property -- see `core.commands` -- because
these devices expose no other channel. The frame itself arrives as raw 16-bit
words with `CAP_PROP_CONVERT_RGB` disabled, so OpenCV hands over the sensor
data untouched rather than trying to interpret it as video.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from irthermal.camera.base import CameraUnavailableError, FrameReadError
from irthermal.core.commands import ACTION_STREAM_CONFIGURE
from irthermal.core.resolutions import (
    SPECIAL_ROWS,
    ResolutionParams,
    is_supported,
    params_for_resolution,
)
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


class V4L2Source:
    """A live thermal camera."""

    def __init__(self, capture: cv2.VideoCapture, *, description: str, camera_raw: bool) -> None:
        self._capture = capture
        self._description = description
        self._camera_raw = camera_raw

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        rows = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._params = params_for_resolution(width, rows - SPECIAL_ROWS)

        self.frame_rate = float(capture.get(cv2.CAP_PROP_FPS))
        self._configure()

    def _configure(self) -> None:
        """Ask for raw 16-bit words rather than decoded video."""
        self._capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self._capture.set(cv2.CAP_PROP_ZOOM, ACTION_STREAM_CONFIGURE)

    @classmethod
    def open(cls, device: str | None = None, *, camera_raw: bool = False) -> V4L2Source:
        """Open a specific device path."""
        if device is None:
            return cls.discover(camera_raw=camera_raw)

        capture = _open_capture(device)
        if not capture.isOpened():
            msg = f"cannot open video device {device}"
            raise CameraUnavailableError(msg)

        try:
            return cls(capture, description=device, camera_raw=camera_raw)
        except Exception:
            capture.release()
            raise

    @classmethod
    def discover(cls, *, camera_raw: bool = False) -> V4L2Source:
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
                log.info("camera.discovered", index=index, width=width, height=rows - SPECIAL_ROWS)
                return cls(capture, description=f"/dev/video{index}", camera_raw=camera_raw)

            capture.release()

        msg = (
            "no supported thermal camera found. Devices seen (index:WxH): "
            f"{', '.join(seen) or 'none'}"
        )
        raise CameraUnavailableError(msg)

    @property
    def params(self) -> ResolutionParams:
        return self._params

    @property
    def camera_raw(self) -> bool:
        return self._camera_raw

    @property
    def description(self) -> str:
        return self._description

    def read_words(self) -> FrameWords:
        """Read one frame, rejecting anything short of a complete capture."""
        ok, buffer = self._capture.read()
        if not ok or buffer is None:
            raise FrameReadError("camera returned no frame")

        words: FrameWords = buffer.view(np.uint16).ravel()
        expected = self._params.total_words
        if words.size < expected:
            msg = f"short frame: {words.size} of {expected} words"
            raise FrameReadError(msg)

        return words

    def send_command(self, payload: int) -> bool:
        return bool(self._capture.set(cv2.CAP_PROP_ZOOM, payload))

    def close(self) -> None:
        self._capture.release()
