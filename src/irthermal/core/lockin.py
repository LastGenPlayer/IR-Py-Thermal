"""Lock-in demodulation.

A lock-in measurement modulates whatever is being heated at a known frequency
and then correlates the video against that frequency, pixel by pixel. Anything
not moving at that frequency -- ambient drift, the sensor's own warm-up, a
draught -- averages towards nothing, which is how a hotspot far below the
camera's noise floor becomes visible given enough integration time.

The maths is one bin of a discrete Fourier transform, accumulated a frame at a
time so the whole video never has to be held in memory:

    in_phase   = mean over frames of   2 * pixel * sin(2*pi*f*t)
    quadrature = mean over frames of  -2 * pixel * cos(2*pi*f*t)

The factor of two makes the magnitude come out as the signal's own amplitude
rather than half of it. Both weights and the final division match v1 exactly, so
a capture demodulated by either gives the same numbers.

Two properties of that choice are worth knowing. Frames are weighted by their
capture timestamp rather than their index, so a dropped or late frame shifts
nothing -- it is irregular sampling, which this handles and a fixed-step DFT
would not. And nothing subtracts the mean first, so a constant background leaks
into the result unless the integration covers a whole number of periods; run for
an integer count of periods and it cancels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

LockInImage = npt.NDArray[np.float64]
"""One demodulated component, in whatever units the input frames carried."""

TWO_PI = 2.0 * math.pi


class LockInError(RuntimeError):
    """A lock-in run cannot continue."""


@dataclass(frozen=True, slots=True)
class LockInResult:
    """The outcome of one integration."""

    frequency_hz: float
    frames: int
    duration_s: float
    in_phase: LockInImage
    quadrature: LockInImage

    @property
    def magnitude(self) -> LockInImage:
        """Amplitude of the response at the modulation frequency.

        In the units of the input: raw counts for a raw run, Celsius for a
        temperature one. This is the image the measurement exists to produce.
        """
        return np.hypot(self.in_phase, self.quadrature)

    @property
    def phase_degrees(self) -> LockInImage:
        """How far each pixel lags the drive, in degrees.

        Negated against `quadrature` deliberately. With the `-cos` weighting
        above, a signal `A*sin(2*pi*f*t + p)` accumulates to `in_phase = A*cos p`
        and `quadrature = -A*sin p`, so recovering `p` itself takes the sign
        flip. Lag is the useful quantity: it grows with how far the heat had to
        travel to reach the surface, which is what makes buried defects show up
        here when they are invisible in `magnitude`.
        """
        return np.degrees(np.arctan2(-self.quadrature, self.in_phase))

    @property
    def periods(self) -> float:
        """Modulation cycles covered. Non-integer values leak the background."""
        return self.frequency_hz * self.duration_s

    @property
    def samples_per_period(self) -> float:
        """Frames per modulation cycle. Below 2 the result is aliased."""
        if self.duration_s <= 0.0 or self.frequency_hz <= 0.0:
            return 0.0
        return self.frames / self.periods


class LockInAccumulator:
    """Correlates frames against one frequency as they arrive."""

    def __init__(self, frequency_hz: float, shape: tuple[int, ...]) -> None:
        if frequency_hz <= 0.0:
            raise LockInError(f"modulation frequency must be positive, got {frequency_hz}")

        self.frequency_hz = frequency_hz
        self.shape = shape
        self._in_phase: LockInImage = np.zeros(shape, dtype=np.float64)
        self._quadrature: LockInImage = np.zeros(shape, dtype=np.float64)
        self._frames = 0
        self._duration = 0.0

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def duration_s(self) -> float:
        """Elapsed time of the most recent frame added."""
        return self._duration

    def add(self, image: npt.NDArray[Any], elapsed_s: float) -> None:
        """Fold in one frame, captured `elapsed_s` after the reference instant."""
        if image.shape != self.shape:
            msg = f"frame shape {image.shape} does not match the run's {self.shape}"
            raise LockInError(msg)

        phase = TWO_PI * self.frequency_hz * elapsed_s
        self._in_phase += image * (2.0 * math.sin(phase))
        self._quadrature += image * (-2.0 * math.cos(phase))
        self._frames += 1
        self._duration = elapsed_s

    def result(self) -> LockInResult:
        """The demodulated components so far. Safe to call mid-run."""
        if self._frames == 0:
            raise LockInError("no frames were integrated")

        in_phase = self._in_phase / self._frames
        quadrature = self._quadrature / self._frames
        in_phase.setflags(write=False)
        quadrature.setflags(write=False)

        return LockInResult(
            frequency_hz=self.frequency_hz,
            frames=self._frames,
            duration_s=self._duration,
            in_phase=in_phase,
            quadrature=quadrature,
        )
