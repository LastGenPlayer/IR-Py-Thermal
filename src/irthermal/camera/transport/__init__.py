"""Transports, and the rule for picking one.

`auto` is what almost everyone should use. It prefers V4L2 on Linux, because a
V4L2 handle is the only one that carries the camera's control channel, and falls
back to an ffmpeg pipe when that fails. Everywhere else it goes straight to
ffmpeg: OpenCV's Windows and macOS backends decode the sensor's 16-bit words to
BGR before the driver can see them, which loses the measurement entirely.
"""

from __future__ import annotations

from typing import Literal

from irthermal.camera.base import CameraUnavailableError
from irthermal.camera.transport.base import PLATFORM, DeviceGeometry, Transport
from irthermal.camera.transport.ffmpeg import FFmpegTransport, FFmpegUnavailableError
from irthermal.camera.transport.v4l2 import V4L2Transport
from irthermal.logging import get_logger

log = get_logger(__name__)

TransportName = Literal["auto", "v4l2", "ffmpeg"]
"""How to reach the camera. `auto` resolves per platform."""

TRANSPORT_NAMES: frozenset[str] = frozenset(("auto", "v4l2", "ffmpeg"))


def _open_v4l2(device: str | None) -> Transport:
    return V4L2Transport.discover() if device is None else V4L2Transport.open(device)


def _open_ffmpeg(device: str | None) -> Transport:
    return FFmpegTransport.discover() if device is None else FFmpegTransport.open(device)


def open_transport(device: str | None = None, *, transport: str = "auto") -> Transport:
    """Open a camera, choosing a transport when the caller did not.

    `device` is the transport's own identifier -- a V4L2 path, a DirectShow
    device name, an avfoundation index -- or None to discover one.
    """
    if transport not in TRANSPORT_NAMES:
        known = ", ".join(sorted(TRANSPORT_NAMES))
        msg = f"unknown transport {transport!r}; expected one of {known}"
        raise ValueError(msg)

    if transport == "v4l2":
        return _open_v4l2(device)
    if transport == "ffmpeg":
        return _open_ffmpeg(device)

    if not PLATFORM.startswith("linux"):
        return _open_ffmpeg(device)

    try:
        return _open_v4l2(device)
    except CameraUnavailableError as v4l2_error:
        log.info("transport.v4l2.failed", error=str(v4l2_error))
        try:
            return _open_ffmpeg(device)
        except FFmpegUnavailableError:
            # No ffmpeg to fall back to; the V4L2 failure is the real story.
            raise v4l2_error from None


__all__ = [
    "TRANSPORT_NAMES",
    "DeviceGeometry",
    "FFmpegTransport",
    "FFmpegUnavailableError",
    "Transport",
    "TransportName",
    "V4L2Transport",
    "open_transport",
]
