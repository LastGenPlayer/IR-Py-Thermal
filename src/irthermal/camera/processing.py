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

from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from irthermal.core.frame import FrameImage
from irthermal.core.resolutions import MAX_RAW_INDEX

DEAD_PIXEL_THRESHOLD = 0.05
"""Fraction of the reference frame's range below which a pixel counts as dead."""

_INPAINT_RADIUS = 3


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
