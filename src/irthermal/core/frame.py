"""One captured frame, and everything derived from it.

A `Frame` is immutable and self-contained: the pixels, the telemetry decoded
from the same capture, and the temperature table those coefficients produce all
travel together. v1 kept them apart -- `read()` returned pixels while `info()`
re-derived telemetry from whatever happened to be in a mutable buffer -- which
meant a caller could pair a corrected image with an uncorrected measurement
without noticing. Bundling them makes that mismatch unrepresentable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from irthermal.core.radiometry import TemperatureTable
from irthermal.core.telemetry import CameraTelemetry

FrameImage = npt.NDArray[np.uint16]
"""Visible pixels, 2D, already clamped into the temperature table's index range."""


@dataclass(frozen=True, slots=True)
class Measurement:
    """A temperature reading at a pixel."""

    x: int
    y: int
    celsius: float


@dataclass(frozen=True, slots=True)
class RegionStats:
    """Temperature summary over a rectangle."""

    minimum: Measurement
    maximum: Measurement
    mean_celsius: float


@dataclass(frozen=True, slots=True)
class Frame:
    """A single capture: pixels, telemetry and temperature table as one unit."""

    camera_id: str
    sequence: int
    captured_at: float
    """`time.monotonic()` at capture, for age and frame-rate measurement."""

    raw: FrameImage
    telemetry: CameraTelemetry
    table: TemperatureTable

    corrected: bool = False
    """Whether software flat-field correction was applied to `raw`.

    When it was, the sensor's own min/max/centre readings in `telemetry` refer
    to the uncorrected pixels and must not be mixed with these.
    """

    @property
    def height(self) -> int:
        return int(self.raw.shape[0])

    @property
    def width(self) -> int:
        return int(self.raw.shape[1])

    def contains(self, x: int, y: int) -> bool:
        """Whether (x, y) is inside the image."""
        return 0 <= x < self.width and 0 <= y < self.height

    def temperatures(self) -> TemperatureTable:
        """Every pixel in Celsius.

        Entries can be NaN where the radiance falls below what the calibration
        model can express. Nothing is substituted for them: an invented
        temperature is worse than an absent one.
        """
        return self.table[self.raw]

    def temperature_at(self, x: int, y: int) -> float:
        """Temperature at one pixel, in Celsius."""
        if not self.contains(x, y):
            msg = f"({x}, {y}) is outside the {self.width}x{self.height} frame"
            raise ValueError(msg)
        return float(self.table[self.raw[y, x]])

    def measure(self, points: Iterable[tuple[str, int, int]]) -> dict[str, Measurement]:
        """Sample named points, skipping any that fall outside the frame."""
        return {
            name: Measurement(x, y, float(self.table[self.raw[y, x]]))
            for name, x, y in points
            if self.contains(x, y)
        }

    def region_stats(self, x1: int, y1: int, x2: int, y2: int) -> RegionStats:
        """Min, max and mean over a rectangle, clipped to the frame.

        `x2`/`y2` are exclusive. Coordinates may arrive in any order.
        """
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))

        left = max(0, min(left, self.width))
        right = max(0, min(right, self.width))
        top = max(0, min(top, self.height))
        bottom = max(0, min(bottom, self.height))

        if left == right or top == bottom:
            msg = f"empty region ({x1}, {y1})-({x2}, {y2}) for a {self.width}x{self.height} frame"
            raise ValueError(msg)

        window = self.table[self.raw[top:bottom, left:right]]
        return RegionStats(
            minimum=_extreme(window, left, top, maximum=False),
            maximum=_extreme(window, left, top, maximum=True),
            mean_celsius=float(np.nanmean(window)),
        )

    def stats(self) -> RegionStats:
        """Min, max and mean over the whole frame."""
        return self.region_stats(0, 0, self.width, self.height)

    def center(self) -> Measurement:
        """Temperature at the centre pixel."""
        x, y = self.width // 2, self.height // 2
        return Measurement(x, y, self.temperature_at(x, y))


def _extreme(window: TemperatureTable, left: int, top: int, *, maximum: bool) -> Measurement:
    """Locate the hottest or coldest pixel in a window, ignoring NaN."""
    finder = np.nanargmax if maximum else np.nanargmin
    try:
        flat = int(finder(window))
    except ValueError:
        # Every pixel is NaN; report the corner rather than inventing a value.
        return Measurement(left, top, float("nan"))

    row, column = np.unravel_index(flat, window.shape)
    return Measurement(left + int(column), top + int(row), float(window[row, column]))
