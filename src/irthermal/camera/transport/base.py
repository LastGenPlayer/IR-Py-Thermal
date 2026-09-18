"""What a source needs from the thing actually moving bytes off the camera.

The driver above this layer knows the XTherm wire format; it does not know
whether those bytes arrived through V4L2, through an ffmpeg pipe or from a file.
Splitting the two is what lets the same camera work on Linux, Windows and macOS
without the frame decoder learning anything about operating systems.

Transports differ in one way that callers cannot ignore: a V4L2 handle carries a
control channel, an ffmpeg pipe does not. `supports_control` says which, so the
API can answer honestly rather than silently dropping a calibration request.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from irthermal.core.telemetry import FrameWords

PLATFORM: str = sys.platform
"""`sys.platform`, widened to `str`.

Comparing against `sys.platform` directly lets a type checker narrow away every
branch that cannot run on the machine it happens to be checking from, so the
other platforms' code is never checked at all. Going through a `str` keeps all
of it live.
"""


@dataclass(frozen=True, slots=True)
class DeviceGeometry:
    """What a device reports before anything has been decoded.

    `rows` is what the device delivers, telemetry rows included -- not the
    visible image height. Resolving one into the other needs the sensor
    constants, which live a layer up.
    """

    width: int
    rows: int


@runtime_checkable
class Transport(Protocol):
    """A stream of complete frames, as flat u16 words."""

    @property
    def description(self) -> str:
        """Human-readable origin, for logs."""
        ...

    @property
    def geometry(self) -> DeviceGeometry:
        """Frame shape as delivered, telemetry rows included."""
        ...

    @property
    def supports_control(self) -> bool:
        """Whether `send_command` can actually reach the device.

        False for pipe-based transports, which carry pixels and nothing else.
        """
        ...

    def configure_stream(self, payload: int) -> None:
        """Put the device into raw 16-bit streaming mode, if that is possible here.

        A V4L2 handle does this by writing the payload through the control
        channel. A pipe cannot, and does not need to: ffmpeg reads the device's
        native format, which for these cameras is already the raw one.
        """
        ...

    def read_words(self) -> FrameWords:
        """Block until one complete frame is available.

        Raises `FrameReadError` for a recoverable miss and
        `CameraUnavailableError` when the device is gone.
        """
        ...

    def send_command(self, payload: int) -> bool:
        """Write one control payload. Returns whether the device accepted it.

        Transports with no control channel raise `ControlUnsupportedError`
        rather than reporting a write that never happened.
        """
        ...

    def close(self) -> None:
        """Release the device. Safe to call more than once."""
        ...
