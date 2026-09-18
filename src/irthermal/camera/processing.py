"""Flat-field correction for cameras that stream unprocessed sensor data.

Some units (the T2S+ V2, notably) do no onboard processing, so every pixel
carries its own fixed offset and the image looks like noise. The fix is to
capture a reference frame with the shutter closed, subtract it, and inpaint over
the pixels that read as dead.

Correction output is clamped into the temperature table's index range. v1 left
it clipped at 65535 while the table held only 16384 entries, so a single hot
outlier turned into an IndexError somewhere downstream.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from irthermal.core.frame import FrameImage
from irthermal.core.resolutions import MAX_RAW_INDEX
from irthermal.logging import get_logger

log = get_logger(__name__)

DEAD_PIXEL_THRESHOLD = 0.05
"""Fraction of the reference frame's range below which a pixel counts as dead."""

_INPAINT_RADIUS = 3

CALIBRATION_VERSION = 1
"""Bumped whenever the stored layout changes, so old files are ignored not misread."""


@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    """What a stored calibration says about itself."""

    saved_at: float
    """Unix time of the calibration, for judging how stale it is."""

    fpa_temp_c: float
    """Sensor temperature when it was taken. NaN when it was not recorded.

    The fixed pattern itself holds steady as the camera warms; what drifts is
    the overall level, and that lands as a uniform offset which
    `temperature_offset` already exists to trim. This is here so an operator can
    see how far conditions have moved, not because the pattern expires.
    """

    dead_pixels: int

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.saved_at)


class RawCorrector:
    """Holds the reference frame and dead-pixel mask for one camera."""

    def __init__(self) -> None:
        self._reference: npt.NDArray[np.float32] | None = None
        self._offset_mean = 0.0
        self._dead_pixels: npt.NDArray[np.uint8] | None = None

    @property
    def calibrated(self) -> bool:
        return self._reference is not None

    @property
    def dead_pixel_count(self) -> int:
        if self._dead_pixels is None:
            return 0
        return int(np.count_nonzero(self._dead_pixels))

    def reset(self) -> None:
        """Discard the current calibration."""
        self._reference = None
        self._offset_mean = 0.0
        self._dead_pixels = None

    def calibrate(self, reference: FrameImage) -> None:
        """Adopt `reference` -- captured with the shutter closed -- as the baseline."""
        as_float = reference.astype(np.float32)
        self._reference = as_float
        self._offset_mean = float(np.mean(as_float))

        low, high = float(np.min(as_float)), float(np.max(as_float))
        threshold = low + (high - low) * DEAD_PIXEL_THRESHOLD

        if np.count_nonzero(as_float < threshold):
            self._dead_pixels = cv2.inRange(as_float, 0.0, threshold).astype(np.uint8)
        else:
            self._dead_pixels = None

    # --- persistence --------------------------------------------------------

    def save(self, path: Path, *, fpa_temp_c: float = float("nan")) -> None:
        """Write the current calibration so a later run can skip the shutter.

        Written to a neighbouring temporary file and moved into place, so an
        interrupted write cannot leave a half-file that loads as a plausible but
        wrong reference.
        """
        if self._reference is None:
            msg = "nothing to save; calibrate first"
            raise RuntimeError(msg)

        dead = self._dead_pixels
        path.parent.mkdir(parents=True, exist_ok=True)
        scratch = path.with_name(path.name + ".partial")

        with scratch.open("wb") as handle:
            np.savez(
                handle,
                version=np.asarray(CALIBRATION_VERSION),
                reference=self._reference,
                offset_mean=np.asarray(self._offset_mean),
                has_dead=np.asarray(dead is not None),
                dead_pixels=dead if dead is not None else np.zeros((1, 1), dtype=np.uint8),
                saved_at=np.asarray(time.time()),
                fpa_temp_c=np.asarray(fpa_temp_c),
            )
        scratch.replace(path)

    def load(self, path: Path, *, shape: tuple[int, int]) -> CalibrationRecord | None:
        """Adopt a stored calibration. Returns None when there is no usable one.

        Every failure here is recoverable -- no file yet, a file from an older
        layout, one saved against a different sensor -- so none of them raise.
        The camera simply streams uncorrected, which is what it would have done
        anyway.
        """
        if not path.is_file():
            return None

        try:
            with np.load(path) as stored:
                if int(stored["version"]) != CALIBRATION_VERSION:
                    log.warning(
                        "calibration.version_mismatch",
                        path=str(path),
                        found=int(stored["version"]),
                        expected=CALIBRATION_VERSION,
                    )
                    return None

                reference = stored["reference"].astype(np.float32)
                if reference.shape != shape:
                    log.warning(
                        "calibration.shape_mismatch",
                        path=str(path),
                        found=f"{reference.shape[1]}x{reference.shape[0]}",
                        expected=f"{shape[1]}x{shape[0]}",
                    )
                    return None

                dead = stored["dead_pixels"].astype(np.uint8) if bool(stored["has_dead"]) else None
                record = CalibrationRecord(
                    saved_at=float(stored["saved_at"]),
                    fpa_temp_c=float(stored["fpa_temp_c"]),
                    dead_pixels=int(np.count_nonzero(dead)) if dead is not None else 0,
                )
                offset_mean = float(stored["offset_mean"])
        except (OSError, KeyError, ValueError) as exc:
            log.warning("calibration.unreadable", path=str(path), error=str(exc))
            return None

        self._reference = reference
        self._offset_mean = offset_mean
        self._dead_pixels = dead
        return record

    def apply(self, visible: FrameImage) -> FrameImage:
        """Correct one frame. Returns it unchanged if not yet calibrated."""
        if self._reference is None:
            return clamp_to_table(visible)

        difference = visible.astype(np.float32) - self._reference + self._offset_mean
        corrected: npt.NDArray[np.float32] = np.clip(difference, 0.0, float(MAX_RAW_INDEX)).astype(
            np.float32
        )

        if self._dead_pixels is not None:
            # cv2's stubs widen the result; inpaint preserves shape and dtype.
            corrected = cast(
                "npt.NDArray[np.float32]",
                cv2.inpaint(corrected, self._dead_pixels, _INPAINT_RADIUS, cv2.INPAINT_TELEA),
            )

        result: FrameImage = corrected.astype(np.uint16)
        return result


def clamp_to_table(visible: FrameImage) -> FrameImage:
    """Force pixels into the temperature table's index range.

    Sensor values are 14-bit, so this is normally a no-op -- but a garbled read
    should degrade into a wrong pixel, not an IndexError two layers away.
    """
    if int(visible.max(initial=0)) <= MAX_RAW_INDEX:
        return visible
    return np.minimum(visible, MAX_RAW_INDEX)
