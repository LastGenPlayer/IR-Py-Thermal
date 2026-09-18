"""Frames through an ffmpeg pipe, on any operating system.

ffmpeg opens the capture pin at its native pixel format and copies it out
untouched. For these cameras the native format is `yuyv422` -- two bytes per
pixel -- which is not really 4:2:2 video at all, but the sensor's 16-bit words
wearing a format tag the USB video class understands. Asking for `rawvideo` out
gets those words back exactly as the sensor produced them.

That is what makes Windows and macOS work at all. OpenCV's backends there decode
the frame to BGR before anyone can stop them; ffmpeg never touches it.

The cost is the control channel. A pipe carries pixels and nothing else, so
`send_command` raises and `supports_control` is False. Cameras that need
`ACTION_STREAM_CONFIGURE` to enter raw mode cannot be driven this way -- the
T2S+ and its siblings stream raw words natively, which is why this works for
them without ever sending a command.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from irthermal.camera.base import CameraUnavailableError, ControlUnsupportedError
from irthermal.camera.transport.base import PLATFORM, DeviceGeometry
from irthermal.core.resolutions import SPECIAL_ROWS, SUPPORTED_RESOLUTIONS
from irthermal.core.telemetry import FrameWords
from irthermal.logging import get_logger

log = get_logger(__name__)

FFMPEG_ENV_VAR = "IRTHERMAL_FFMPEG"
"""Explicit path to an ffmpeg binary, ahead of every other lookup."""

PIXEL_FORMAT = "yuyv422"
"""Two bytes per pixel, passed through without conversion."""

_ENUMERATION_TIMEOUT_S = 15.0
_PROBE_TIMEOUT_S = 20.0

_INSTALL_HINT = (
    "install one with: pip install 'irthermal[ffmpeg]' (bundled binary), "
    "winget install Gyan.FFmpeg, apt install ffmpeg, or brew install ffmpeg; "
    f"or point {FFMPEG_ENV_VAR} at a binary"
)


class FFmpegUnavailableError(CameraUnavailableError):
    """No usable ffmpeg binary could be found."""


def find_ffmpeg() -> str:
    """Locate an ffmpeg binary: explicit path, then bundled, then PATH."""
    explicit = os.environ.get(FFMPEG_ENV_VAR)
    if explicit:
        if not shutil.which(explicit) and not Path(explicit).is_file():
            msg = f"{FFMPEG_ENV_VAR} points at {explicit!r}, which is not an executable"
            raise FFmpegUnavailableError(msg)
        return explicit

    try:
        import imageio_ffmpeg
    except ImportError:
        pass
    else:
        return str(imageio_ffmpeg.get_ffmpeg_exe())

    found = shutil.which("ffmpeg")
    if found:
        return found

    raise FFmpegUnavailableError(f"no ffmpeg binary found; {_INSTALL_HINT}")


def input_format() -> str:
    """The ffmpeg input device this platform captures through."""
    if PLATFORM == "win32":
        return "dshow"
    if PLATFORM == "darwin":
        return "avfoundation"
    return "v4l2"


def input_spec(device: str) -> str:
    """Turn a device identifier into what ffmpeg's `-i` expects."""
    if PLATFORM == "win32":
        return f"video={device}"
    return device


def _run(ffmpeg: str, args: list[str], timeout: float) -> str:
    """Run ffmpeg and return its stderr, where it writes all its listings."""
    try:
        completed = subprocess.run(
            [ffmpeg, "-hide_banner", *args],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CameraUnavailableError(f"ffmpeg timed out after {timeout}s") from exc
    return completed.stderr.decode("utf-8", errors="replace")


# --- device enumeration -----------------------------------------------------


def list_devices(ffmpeg: str | None = None) -> list[str]:
    """Every video capture device this platform exposes to ffmpeg."""
    binary = ffmpeg or find_ffmpeg()
    fmt = input_format()

    if fmt == "dshow":
        stderr = _run(
            binary,
            ["-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            _ENUMERATION_TIMEOUT_S,
        )
        return _parse_dshow_devices(stderr)

    if fmt == "avfoundation":
        stderr = _run(
            binary,
            ["-list_devices", "true", "-f", "avfoundation", "-i", ""],
            _ENUMERATION_TIMEOUT_S,
        )
        return _parse_avfoundation_devices(stderr)

    return sorted(str(path) for path in Path("/dev").glob("video*"))


def _parse_dshow_devices(stderr: str) -> list[str]:
    """Pull video device names out of a dshow listing.

    Lines look like `[dshow @ ...] "T2SPro" (video)`; the alternative-name line
    that follows carries the PnP path, which we do not need.
    """
    names: list[str] = []
    for line in stderr.splitlines():
        if not line.rstrip().endswith("(video)"):
            continue
        start = line.find('"')
        end = line.rfind('"')
        if start != -1 and end > start:
            names.append(line[start + 1 : end])
    return names


def _parse_avfoundation_devices(stderr: str) -> list[str]:
    """Pull video device indices out of an avfoundation listing.

    The listing has a video section then an audio section, both using `[N] Name`,
    so the audio header has to stop the scan.
    """
    indices: list[str] = []
    in_video = False
    for line in stderr.splitlines():
        lowered = line.lower()
        if "video devices" in lowered:
            in_video = True
            continue
        if "audio devices" in lowered:
            in_video = False
            continue
        if not in_video:
            continue
        marker = line.rfind("] [")
        if marker == -1:
            continue
        closing = line.find("]", marker + 3)
        if closing != -1:
            indices.append(line[marker + 3 : closing])
    return indices


def dshow_geometries(ffmpeg: str, device: str) -> list[DeviceGeometry]:
    """Every frame size a dshow device advertises.

    Lines look like `pixel_format=yuyv422  min s=256x196 fps=25 max s=256x196`.
    """
    stderr = _run(
        ffmpeg,
        ["-f", "dshow", "-list_options", "true", "-i", f"video={device}"],
        _ENUMERATION_TIMEOUT_S,
    )

    found: list[DeviceGeometry] = []
    for line in stderr.splitlines():
        if "min s=" not in line:
            continue
        chunk = line.split("min s=", 1)[1].split()[0]
        width, _, rows = chunk.partition("x")
        try:
            geometry = DeviceGeometry(width=int(width), rows=int(rows))
        except ValueError:
            continue
        if geometry not in found:
            found.append(geometry)
    return found


def _supported_geometries() -> list[DeviceGeometry]:
    """Frame shapes a supported sensor would deliver, telemetry rows included."""
    return [
        DeviceGeometry(width=width, rows=height + SPECIAL_ROWS)
        for width, height in sorted(SUPPORTED_RESOLUTIONS)
    ]


def _is_supported(geometry: DeviceGeometry) -> bool:
    return (geometry.width, geometry.rows - SPECIAL_ROWS) in SUPPORTED_RESOLUTIONS


def probe_geometry(ffmpeg: str, device: str) -> DeviceGeometry:
    """Work out what shape a device delivers.

    dshow advertises its formats, so one listing settles it. Elsewhere there is
    no reliable listing, so each supported shape is tried until one yields a
    frame of exactly the right size.
    """
    if input_format() == "dshow":
        advertised = dshow_geometries(ffmpeg, device)
        for geometry in advertised:
            if _is_supported(geometry):
                return geometry
        seen = ", ".join(f"{g.width}x{g.rows}" for g in advertised) or "none"
        msg = f"device {device!r} advertises no supported resolution. Sizes seen: {seen}"
        raise CameraUnavailableError(msg)

    for geometry in _supported_geometries():
        if _capture_once(ffmpeg, device, geometry):
            return geometry

    msg = f"device {device!r} produced no frame at any supported resolution"
    raise CameraUnavailableError(msg)


def _capture_once(ffmpeg: str, device: str, geometry: DeviceGeometry) -> bool:
    """Whether one frame of exactly this shape can be read from the device."""
    expected = geometry.width * geometry.rows * 2
    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                input_format(),
                "-pixel_format",
                PIXEL_FORMAT,
                "-video_size",
                f"{geometry.width}x{geometry.rows}",
                "-i",
                input_spec(device),
                "-vframes",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                PIXEL_FORMAT,
                "-",
            ],
            capture_output=True,
            timeout=_PROBE_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return len(completed.stdout) == expected


# --- the transport itself ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class FFmpegSettings:
    """Everything needed to spawn a capture, resolved before the process starts."""

    ffmpeg: str
    device: str
    geometry: DeviceGeometry
    fps: float | None = None


class FFmpegTransport:
    """A camera streaming through an ffmpeg subprocess."""

    def __init__(self, settings: FFmpegSettings) -> None:
        self._settings = settings
        self._geometry = settings.geometry
        self._frame_bytes = settings.geometry.width * settings.geometry.rows * 2
        self._description = f"ffmpeg({input_format()}:{settings.device})"
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._process = self._spawn()

    @classmethod
    def open(cls, device: str, *, ffmpeg: str | None = None) -> FFmpegTransport:
        """Open a named device, working out its geometry first."""
        binary = ffmpeg or find_ffmpeg()
        geometry = probe_geometry(binary, device)
        return cls(FFmpegSettings(ffmpeg=binary, device=device, geometry=geometry))

    @classmethod
    def discover(cls, *, ffmpeg: str | None = None) -> FFmpegTransport:
        """Take the first enumerated device that advertises a supported resolution."""
        binary = ffmpeg or find_ffmpeg()
        seen: list[str] = []

        for device in list_devices(binary):
            try:
                geometry = probe_geometry(binary, device)
            except CameraUnavailableError as exc:
                seen.append(f"{device}: {exc}")
                continue

            log.info(
                "transport.ffmpeg.discovered",
                device=device,
                width=geometry.width,
                rows=geometry.rows,
            )
            return cls(FFmpegSettings(ffmpeg=binary, device=device, geometry=geometry))

        detail = "; ".join(seen) or "no video devices found"
        raise CameraUnavailableError(f"no supported thermal camera found. {detail}")

    def _spawn(self) -> subprocess.Popen[bytes]:
        settings = self._settings
        args = [
            settings.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            input_format(),
            "-pixel_format",
            PIXEL_FORMAT,
            "-video_size",
            f"{settings.geometry.width}x{settings.geometry.rows}",
        ]
        if settings.fps:
            args += ["-framerate", str(settings.fps)]
        args += [
            "-i",
            input_spec(settings.device),
            "-f",
            "rawvideo",
            "-pix_fmt",
            PIXEL_FORMAT,
            "-",
        ]

        # Detach it from the console's signal group. Otherwise a Ctrl+C meant for
        # the server also reaches ffmpeg, the pipe closes, and the reader treats
        # a deliberate shutdown as a lost camera and starts reconnecting.
        # `sys.platform` rather than PLATFORM here on purpose: CREATE_NEW_PROCESS_GROUP
        # only exists on Windows, and a type checker needs the literal comparison to
        # know that this branch cannot run anywhere else.
        detach: dict[str, Any]
        if sys.platform == "win32":
            detach = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        else:
            detach = {"start_new_session": True}

        try:
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                **detach,
            )
        except OSError as exc:
            raise CameraUnavailableError(f"could not start ffmpeg: {exc}") from exc

        threading.Thread(
            target=self._drain_stderr,
            args=(process,),
            name=f"ffmpeg-stderr-{settings.device}",
            daemon=True,
        ).start()
        return process

    def _drain_stderr(self, process: subprocess.Popen[bytes]) -> None:
        """Keep the stderr pipe empty, retaining the last lines for diagnostics.

        A full pipe would block ffmpeg mid-frame, so this has to run whether or
        not anyone ever reads the messages.
        """
        stream = process.stderr
        if stream is None:  # pragma: no cover - always piped
            return
        for raw in stream:
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                self._stderr_tail.append(line)

    @property
    def description(self) -> str:
        return self._description

    @property
    def geometry(self) -> DeviceGeometry:
        return self._geometry

    @property
    def supports_control(self) -> bool:
        return False

    def configure_stream(self, payload: int) -> None:  # noqa: ARG002 - protocol signature
        """Nothing to do: ffmpeg already asked for the device's native format."""
        log.debug("transport.ffmpeg.configure_skipped", device=self._settings.device)

    def read_words(self) -> FrameWords:
        stream = self._process.stdout
        if stream is None:  # pragma: no cover - always piped
            raise CameraUnavailableError("ffmpeg produced no output pipe")

        chunks: list[bytes] = []
        remaining = self._frame_bytes

        # An unbuffered pipe returns short reads freely; a frame is only whole
        # once every byte of it has arrived.
        while remaining > 0:
            chunk = stream.read(remaining)
            if not chunk:
                raise CameraUnavailableError(self._death_reason())
            chunks.append(chunk)
            remaining -= len(chunk)

        payload = chunks[0] if len(chunks) == 1 else b"".join(chunks)
        words: FrameWords = np.frombuffer(payload, dtype="<u2")
        return words

    def _death_reason(self) -> str:
        """Why the pipe closed, using whatever ffmpeg said on its way out."""
        code = self._process.poll()
        tail = "; ".join(self._stderr_tail)
        detail = f": {tail}" if tail else ""
        if code is None:
            return f"{self._description} stopped delivering frames{detail}"
        return f"{self._description} exited with code {code}{detail}"

    def send_command(self, payload: int) -> bool:  # noqa: ARG002 - protocol signature
        msg = (
            f"{self._description} carries pixels only. Camera control needs a V4L2 "
            'handle: run on Linux with transport = "v4l2".'
        )
        raise ControlUnsupportedError(msg)

    def close(self) -> None:
        process = self._process
        if process.poll() is not None:
            return

        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)
