"""What the reader needs from a frame source, and the errors it can raise.

Keeping this to a protocol rather than a base class is what lets the emulator
be a peer of the real device instead of a subclass fighting its constructor --
which is how v1's emulator ended up broken: `Camera.__init__` reset the very
buffer `CameraEmulator` had just loaded.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from irthermal.core.resolutions import ResolutionParams
from irthermal.core.telemetry import FrameWords


class CameraError(Exception):
    """Base class for camera faults."""


class CameraUnavailableError(CameraError):
    """The device could not be opened, or vanished while streaming."""


class FrameReadError(CameraError):
    """A frame could not be read. Usually transient; the reader will retry."""


@runtime_checkable
class CameraSource(Protocol):
    """A source of complete frames."""

    @property
    def params(self) -> ResolutionParams:
        """Sensor geometry and calibration constants."""
        ...

    @property
    def camera_raw(self) -> bool:
        """Whether this camera streams unprocessed sensor data."""
        ...

    @property
    def description(self) -> str:
        """Human-readable origin, for logs."""
        ...

    def read_words(self) -> FrameWords:
        """Block until one complete frame is available.

        Returns the whole frame -- visible pixels and telemetry rows -- as flat
        u16 words. Raises `FrameReadError` for a recoverable miss and
        `CameraUnavailableError` when the device is gone.
        """
        ...

    def send_command(self, payload: int) -> bool:
        """Write one control payload. Returns whether the device accepted it."""
        ...

    def close(self) -> None:
        """Release the device. Safe to call more than once."""
        ...


SourceFactory = Callable[[], CameraSource]
"""Builds a source on demand, so the reader can reopen one after a disconnect."""
