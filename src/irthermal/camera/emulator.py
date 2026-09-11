"""Replay a captured frame instead of opening hardware.

This is a first-class source, not a special case: it is how the test suite and
CI exercise the whole pipeline without a camera attached, and how a recorded
capture gets re-examined later.

Captures are the `.npy` files v1 wrote with its `r` key -- the complete frame,
telemetry rows included, shaped (height + 4, width).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from irthermal.camera.base import CameraUnavailableError
from irthermal.core.resolutions import SPECIAL_ROWS, ResolutionParams, params_for_resolution
from irthermal.core.telemetry import FrameWords

DEFAULT_REPLAY_FPS = 25.0


class EmulatorSource:
    """Serves one captured frame over and over, paced like a real camera."""

    def __init__(
        self,
        words: FrameWords,
        params: ResolutionParams,
        *,
        description: str,
        camera_raw: bool = False,
        fps: float = DEFAULT_REPLAY_FPS,
    ) -> None:
        self._words = words
        self._params = params
        self._description = description
        self._camera_raw = camera_raw
        self._interval = 1.0 / fps if fps > 0 else 0.0
        self._next_due = time.monotonic()
        self._closed = False

        self.commands: list[int] = []
        """Every control payload sent, so tests can assert on them."""

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        camera_raw: bool = False,
        fps: float = DEFAULT_REPLAY_FPS,
    ) -> EmulatorSource:
        """Load a `.npy` capture."""
        if not path.exists():
            msg = f"capture not found: {path}"
            raise CameraUnavailableError(msg)

        data = np.load(path, allow_pickle=False)
        return cls.from_array(data, description=str(path), camera_raw=camera_raw, fps=fps)

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        *,
        description: str = "<array>",
        camera_raw: bool = False,
        fps: float = DEFAULT_REPLAY_FPS,
    ) -> EmulatorSource:
        """Wrap an in-memory capture shaped (height + telemetry rows, width)."""
        if data.ndim != 2:
            msg = f"capture must be 2-dimensional, got shape {data.shape}"
            raise CameraUnavailableError(msg)

        rows, width = int(data.shape[0]), int(data.shape[1])
        height = rows - SPECIAL_ROWS
        if height <= 0:
            msg = f"capture has {rows} rows, too few to contain {SPECIAL_ROWS} telemetry rows"
            raise CameraUnavailableError(msg)

        params = params_for_resolution(width, height)
        words = np.ascontiguousarray(data, dtype=np.uint16).ravel()
        return cls(words, params, description=description, camera_raw=camera_raw, fps=fps)

    @property
    def params(self) -> ResolutionParams:
        return self._params

    @property
    def camera_raw(self) -> bool:
        return self._camera_raw

    @property
    def description(self) -> str:
        return f"emulator({self._description})"

    def read_words(self) -> FrameWords:
        """Return the capture, pacing to the configured frame rate."""
        if self._closed:
            raise CameraUnavailableError("emulator source is closed")

        if self._interval:
            delay = self._next_due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next_due = max(self._next_due + self._interval, time.monotonic())

        return self._words

    def send_command(self, payload: int) -> bool:
        """Record the command. There is no device to send it to."""
        self.commands.append(payload)
        return True

    def close(self) -> None:
        self._closed = True
