"""Transport selection, device enumeration and the ffmpeg framing loop.

None of this needs a camera: the listings are text ffmpeg writes to stderr, and
the read loop only needs something pipe-shaped.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pytest

from irthermal.camera.base import CameraUnavailableError, ControlUnsupportedError
from irthermal.camera.transport import TRANSPORT_NAMES, open_transport
from irthermal.camera.transport.base import DeviceGeometry
from irthermal.camera.transport.ffmpeg import (
    FFMPEG_ENV_VAR,
    FFmpegSettings,
    FFmpegTransport,
    FFmpegUnavailableError,
    _is_supported,
    _parse_avfoundation_devices,
    _parse_dshow_devices,
    find_ffmpeg,
)

DSHOW_LISTING = """[dshow @ 000001ff1aa75940] "Integrated Webcam" (video)
[dshow @ 000001ff1aa75940]   Alternative name "@device_pnp_\\\\?\\usb#vid_1bcf&pid_28b0"
[dshow @ 000001ff1aa75940] "T2SPro" (video)
[dshow @ 000001ff1aa75940]   Alternative name "@device_pnp_\\\\?\\usb#vid_3474&pid_6002"
[dshow @ 000001ff1aa75940] "Microphone (Realtek Audio)" (audio)
"""

AVFOUNDATION_LISTING = """[AVFoundation indev @ 0x7f8] AVFoundation video devices:
[AVFoundation indev @ 0x7f8] [0] FaceTime HD Camera
[AVFoundation indev @ 0x7f8] [1] T2SPro
[AVFoundation indev @ 0x7f8] AVFoundation audio devices:
[AVFoundation indev @ 0x7f8] [0] Built-in Microphone
"""


# --- enumeration ------------------------------------------------------------


def test_dshow_listing_keeps_video_devices_only() -> None:
    assert _parse_dshow_devices(DSHOW_LISTING) == ["Integrated Webcam", "T2SPro"]


def test_dshow_listing_ignores_alternative_names() -> None:
    """The PnP line carries quotes too, and must not be mistaken for a device."""
    assert not any("device_pnp" in name for name in _parse_dshow_devices(DSHOW_LISTING))


def test_avfoundation_listing_stops_at_the_audio_section() -> None:
    assert _parse_avfoundation_devices(AVFOUNDATION_LISTING) == ["0", "1"]


def test_empty_listing_yields_nothing() -> None:
    assert _parse_dshow_devices("") == []
    assert _parse_avfoundation_devices("") == []


# --- geometry ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("width", "rows", "expected"),
    [
        (256, 196, True),
        (240, 184, True),
        (384, 292, True),
        (640, 516, True),
        (256, 192, False),  # telemetry rows missing
        (640, 480, False),  # an ordinary webcam
    ],
)
def test_geometry_support(width: int, rows: int, expected: bool) -> None:
    assert _is_supported(DeviceGeometry(width=width, rows=rows)) is expected


# --- binary lookup ----------------------------------------------------------


def test_explicit_binary_must_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FFMPEG_ENV_VAR, "definitely-not-a-real-binary")
    with pytest.raises(FFmpegUnavailableError, match="not an executable"):
        find_ffmpeg()


def test_explicit_binary_is_used_when_it_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    binary = tmp_path / "ffmpeg"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setenv(FFMPEG_ENV_VAR, str(binary))
    assert find_ffmpeg() == str(binary)


# --- transport selection ----------------------------------------------------


def test_unknown_transport_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown transport"):
        open_transport(None, transport="usb")


def test_transport_names_are_the_documented_three() -> None:
    assert {"auto", "v4l2", "ffmpeg"} == TRANSPORT_NAMES


# --- the framing loop -------------------------------------------------------


class _ChunkedStream(io.RawIOBase):
    """A pipe that hands back at most `chunk` bytes per read, as a real one does."""

    def __init__(self, payload: bytes, chunk: int) -> None:
        self._payload = payload
        self._chunk = chunk
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._payload) - self._offset
        take = min(size, self._chunk, len(self._payload) - self._offset)
        piece = self._payload[self._offset : self._offset + take]
        self._offset += take
        return piece


class _FakeProcess:
    def __init__(self, stdout: Any, returncode: int | None = None) -> None:
        self.stdout = stdout
        self.stderr = io.BytesIO(b"")
        self._returncode = returncode
        self.terminated = False

    def poll(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True
        self._returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        return self._returncode or 0

    def kill(self) -> None:  # pragma: no cover - terminate always succeeds here
        self._returncode = -9


class _StubTransport(FFmpegTransport):
    """An FFmpegTransport whose subprocess is a scripted pipe."""

    def __init__(self, geometry: DeviceGeometry, process: Any) -> None:
        self._process_stub = process
        super().__init__(FFmpegSettings(ffmpeg="ffmpeg", device="stub", geometry=geometry))

    def _spawn(self) -> Any:
        return self._process_stub


def _geometry() -> DeviceGeometry:
    return DeviceGeometry(width=256, rows=196)


def test_short_reads_are_assembled_into_one_frame() -> None:
    geometry = _geometry()
    expected_words = geometry.width * geometry.rows
    payload = np.arange(expected_words, dtype="<u2").tobytes()

    transport = _StubTransport(geometry, _FakeProcess(_ChunkedStream(payload, chunk=4096)))

    words = transport.read_words()
    assert words.size == expected_words
    assert words.dtype == np.uint16
    np.testing.assert_array_equal(words, np.arange(expected_words, dtype="<u2"))


def test_a_closed_pipe_reports_the_device_as_gone() -> None:
    geometry = _geometry()
    transport = _StubTransport(geometry, _FakeProcess(io.BytesIO(b""), returncode=1))

    with pytest.raises(CameraUnavailableError, match="exited with code 1"):
        transport.read_words()


def test_a_truncated_frame_is_not_returned_as_a_short_one() -> None:
    """Half a frame is a dead pipe, not a frame -- the caller must never see it."""
    geometry = _geometry()
    half = geometry.width * geometry.rows  # bytes, so half the words
    transport = _StubTransport(geometry, _FakeProcess(io.BytesIO(b"\x00" * half)))

    with pytest.raises(CameraUnavailableError):
        transport.read_words()


def test_control_is_refused_rather_than_silently_dropped() -> None:
    transport = _StubTransport(_geometry(), _FakeProcess(io.BytesIO(b"")))

    with pytest.raises(ControlUnsupportedError, match="carries pixels only"):
        transport.send_command(0x8000)

    assert transport.supports_control is False


def test_configure_stream_is_a_no_op() -> None:
    """ffmpeg already opened the native format; there is nothing to configure."""
    transport = _StubTransport(_geometry(), _FakeProcess(io.BytesIO(b"")))
    transport.configure_stream(0x8004)


def test_close_terminates_the_process() -> None:
    process = _FakeProcess(io.BytesIO(b""))
    transport = _StubTransport(_geometry(), process)

    transport.close()
    assert process.terminated is True
