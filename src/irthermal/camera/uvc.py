"""The real camera: XTherm-family sensors over whatever transport carries them.

This layer knows the wire format and nothing about operating systems. It takes
the frame shape a transport reports, resolves the sensor constants that go with
it, and checks every frame is long enough to hold its telemetry rows.

Control happens through the zoom property -- see `core.commands` -- because these
devices expose no other channel. Not every transport has one: an ffmpeg pipe
carries pixels only. `supports_control` says which, so callers can offer
calibration and parameter writes when they will actually work and refuse them
clearly when they will not.
"""

from __future__ import annotations

from irthermal.camera.base import CameraUnavailableError, FrameReadError
from irthermal.camera.transport import Transport, open_transport
from irthermal.core.commands import ACTION_STREAM_CONFIGURE
from irthermal.core.resolutions import SPECIAL_ROWS, ResolutionParams, params_for_resolution
from irthermal.core.telemetry import FrameWords


class UvcSource:
    """A live thermal camera, reached through a transport."""

    def __init__(self, transport: Transport, *, camera_raw: bool) -> None:
        self._transport = transport
        self._camera_raw = camera_raw

        geometry = transport.geometry
        self._params = params_for_resolution(geometry.width, geometry.rows - SPECIAL_ROWS)

        transport.configure_stream(ACTION_STREAM_CONFIGURE)

    @classmethod
    def open(
        cls,
        device: str | None = None,
        *,
        camera_raw: bool = False,
        transport: str = "auto",
    ) -> UvcSource:
        """Open a camera, discovering one when no device is named."""
        handle = open_transport(device, transport=transport)
        try:
            return cls(handle, camera_raw=camera_raw)
        except Exception:
            handle.close()
            raise

    @property
    def params(self) -> ResolutionParams:
        return self._params

    @property
    def camera_raw(self) -> bool:
        return self._camera_raw

    @property
    def description(self) -> str:
        return self._transport.description

    @property
    def supports_control(self) -> bool:
        return self._transport.supports_control

    def read_words(self) -> FrameWords:
        """Read one frame, rejecting anything short of a complete capture."""
        words = self._transport.read_words()
        expected = self._params.total_words
        if words.size < expected:
            msg = f"short frame: {words.size} of {expected} words"
            raise FrameReadError(msg)
        return words

    def send_command(self, payload: int) -> bool:
        """Write one control payload. Raises when the transport has no channel."""
        return self._transport.send_command(payload)

    def close(self) -> None:
        self._transport.close()


__all__ = ["CameraUnavailableError", "UvcSource"]
