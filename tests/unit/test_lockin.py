"""Lock-in demodulation.

Two questions. Does it recover a signal that was deliberately planted, and does
it still agree with v1 to the last bit?
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irthermal.core.lockin import LockInAccumulator, LockInError
from reference_legacy import legacy_lockin

SHAPE = (8, 12)


def modulated(
    amplitude: float,
    frequency_hz: float,
    phase: float = 0.0,
    *,
    background: float = 1000.0,
    frames: int = 200,
    fps: float = 100.0,
) -> tuple[list[np.ndarray], list[float]]:
    """A synthetic capture: a constant scene with one sinusoid on top of it."""
    times = [index / fps for index in range(frames)]
    images = [
        np.full(SHAPE, background + amplitude * math.sin(2 * math.pi * frequency_hz * t + phase))
        for t in times
    ]
    return images, times


def integrate(images: list[np.ndarray], times: list[float], frequency_hz: float):
    accumulator = LockInAccumulator(frequency_hz, SHAPE)
    for image, t in zip(images, times, strict=True):
        accumulator.add(image, t)
    return accumulator.result()


# --- recovery ---------------------------------------------------------------


def test_recovers_the_planted_amplitude() -> None:
    images, times = modulated(amplitude=25.0, frequency_hz=2.0)

    result = integrate(images, times, 2.0)

    assert result.magnitude == pytest.approx(25.0, rel=0.02)


def test_the_background_cancels_over_whole_periods() -> None:
    """A constant a hundred thousand times the signal must not reach the result."""
    images, times = modulated(amplitude=1.0, frequency_hz=2.0, background=100_000.0)

    result = integrate(images, times, 2.0)

    assert result.magnitude == pytest.approx(1.0, rel=0.05)


def test_an_incomplete_window_leaks_the_background() -> None:
    """The caveat that goes with the test above, pinned down rather than implied.

    Nothing subtracts the mean, because v1 did not and parity is the point. A
    window that stops part-way through a period therefore carries some of the
    pedestal into the answer, and the error scales with the pedestal rather
    than with the signal. Integrate over whole periods.
    """
    whole, times = modulated(amplitude=1.0, frequency_hz=2.0, frames=200, fps=100.0)
    partial = 175  # 3.5 periods

    clean = integrate(whole, times, 2.0)
    leaky = integrate(whole[:partial], times[:partial], 2.0)

    assert float(np.max(clean.magnitude)) == pytest.approx(1.0, rel=0.05)
    assert float(np.max(leaky.magnitude)) > 100.0


@pytest.mark.parametrize("phase", [0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 3])
def test_recovers_the_planted_phase(phase: float) -> None:
    """`phase_degrees` reports the signal's own lag, not the accumulator's sign."""
    images, times = modulated(amplitude=10.0, frequency_hz=2.0, phase=phase)

    result = integrate(images, times, 2.0)

    # Compared as an angle: +180 and -180 are the same direction.
    error = (result.phase_degrees - math.degrees(phase) + 180.0) % 360.0 - 180.0
    assert error == pytest.approx(0.0, abs=2.0)


def test_a_signal_at_another_frequency_is_rejected() -> None:
    """The whole point: energy away from the reference averages out."""
    images, times = modulated(amplitude=50.0, frequency_hz=7.0)

    result = integrate(images, times, 2.0)

    assert float(np.max(result.magnitude)) < 5.0


def test_only_the_modulated_pixels_light_up() -> None:
    images, times = modulated(amplitude=0.0, frequency_hz=3.0)
    for image, t in zip(images, times, strict=True):
        image[4, 5] += 40.0 * math.sin(2 * math.pi * 3.0 * t)

    result = integrate(images, times, 3.0)

    assert result.magnitude[4, 5] == pytest.approx(40.0, rel=0.02)
    assert float(np.max(np.delete(result.magnitude, 4 * 12 + 5))) < 1e-6


# --- parity -----------------------------------------------------------------


def test_matches_v1_exactly() -> None:
    """Same weights, same accumulators, same division; so, same bits."""
    rng = np.random.default_rng(20240614)
    times = [index / 30.0 for index in range(120)]
    images = [rng.integers(0, 16384, SHAPE, dtype=np.uint16) for _ in times]

    legacy_in_phase, legacy_quadrature = legacy_lockin(list(images), times, 1.5)
    result = integrate(list(images), times, 1.5)

    assert np.array_equal(result.in_phase, legacy_in_phase)
    assert np.array_equal(result.quadrature, legacy_quadrature)


def test_matches_v1_on_irregular_timestamps() -> None:
    """Frames are weighted by when they arrived, and v1 did the same."""
    rng = np.random.default_rng(7)
    times = sorted(float(t) for t in rng.uniform(0.0, 4.0, 60))
    images = [rng.integers(0, 16384, SHAPE, dtype=np.uint16) for _ in times]

    legacy = legacy_lockin(list(images), times, 2.5)
    result = integrate(list(images), times, 2.5)

    assert np.array_equal(result.in_phase, legacy[0])
    assert np.array_equal(result.quadrature, legacy[1])


# --- bookkeeping ------------------------------------------------------------


def test_reports_periods_and_sampling() -> None:
    images, times = modulated(amplitude=1.0, frequency_hz=2.0, frames=200, fps=100.0)

    result = integrate(images, times, 2.0)

    assert result.frames == 200
    assert result.duration_s == pytest.approx(1.99)
    assert result.periods == pytest.approx(3.98)
    assert result.samples_per_period == pytest.approx(50.25, rel=0.01)


def test_undersampling_is_visible_in_the_result() -> None:
    """Nothing stops an aliased run; it has to be detectable afterwards."""
    images, times = modulated(amplitude=1.0, frequency_hz=20.0, frames=40, fps=25.0)

    result = integrate(images, times, 20.0)

    assert result.samples_per_period < 2.0


def test_result_arrays_are_read_only() -> None:
    """Callers hold these for as long as the job is remembered."""
    images, times = modulated(amplitude=1.0, frequency_hz=2.0, frames=10)

    result = integrate(images, times, 2.0)

    with pytest.raises(ValueError, match="read-only"):
        result.in_phase[0, 0] = 1.0


def test_a_result_can_be_taken_mid_run() -> None:
    images, times = modulated(amplitude=25.0, frequency_hz=2.0)
    accumulator = LockInAccumulator(2.0, SHAPE)

    for image, t in zip(images[:100], times[:100], strict=True):
        accumulator.add(image, t)
    partial = accumulator.result()
    for image, t in zip(images[100:], times[100:], strict=True):
        accumulator.add(image, t)

    assert partial.frames == 100
    assert accumulator.result().frames == 200


# --- refusals ---------------------------------------------------------------


def test_a_changed_frame_shape_is_refused() -> None:
    """A camera that reconnects at another resolution must not be folded in."""
    accumulator = LockInAccumulator(2.0, SHAPE)
    accumulator.add(np.zeros(SHAPE), 0.0)

    with pytest.raises(LockInError, match="does not match"):
        accumulator.add(np.zeros((4, 4)), 0.1)


def test_zero_frames_has_no_result() -> None:
    with pytest.raises(LockInError, match="no frames"):
        LockInAccumulator(2.0, SHAPE).result()


@pytest.mark.parametrize("frequency", [0.0, -1.0])
def test_frequency_must_be_positive(frequency: float) -> None:
    with pytest.raises(LockInError, match="positive"):
        LockInAccumulator(frequency, SHAPE)
