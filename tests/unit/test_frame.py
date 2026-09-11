"""Frame measurement, and the raw-mode correction that feeds it."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import FrameSpec, make_frame
from irthermal.camera.processing import RawCorrector, clamp_to_table
from irthermal.core.resolutions import MAX_RAW_INDEX, ResolutionParams

# --- measurement ------------------------------------------------------------


def test_temperature_at_matches_the_table(params: ResolutionParams) -> None:
    frame = make_frame(params)
    expected = float(frame.table[frame.raw[3, 5]])

    assert frame.temperature_at(5, 3) == pytest.approx(expected)


@pytest.mark.parametrize(("x", "y"), [(-1, 0), (0, -1), (10_000, 0), (0, 10_000)])
def test_temperature_outside_the_frame_is_rejected(
    small_params: ResolutionParams, x: int, y: int
) -> None:
    with pytest.raises(ValueError, match="outside"):
        make_frame(small_params).temperature_at(x, y)


def test_measure_skips_points_outside_the_frame(small_params: ResolutionParams) -> None:
    """A misconfigured coordinate should drop out, not abort the whole sample."""
    frame = make_frame(small_params)
    measured = frame.measure([("inside", 10, 10), ("negative", -1, 5), ("far", frame.width, 5)])

    assert set(measured) == {"inside"}
    assert measured["inside"].x == 10


def test_measure_returns_named_measurements(small_params: ResolutionParams) -> None:
    frame = make_frame(small_params)
    measured = frame.measure([("a", 1, 2), ("b", 3, 4)])

    assert (measured["a"].x, measured["a"].y) == (1, 2)
    assert measured["a"].celsius == pytest.approx(frame.temperature_at(1, 2))
    assert (measured["b"].x, measured["b"].y) == (3, 4)
    assert measured["b"].celsius == pytest.approx(frame.temperature_at(3, 4))


def test_frame_stats_find_the_real_extremes(small_params: ResolutionParams) -> None:
    frame = make_frame(small_params)
    temperatures = frame.temperatures()
    stats = frame.stats()

    assert stats.minimum.celsius == pytest.approx(np.nanmin(temperatures))
    assert stats.maximum.celsius == pytest.approx(np.nanmax(temperatures))
    assert stats.mean_celsius == pytest.approx(np.nanmean(temperatures))


def test_stats_coordinates_point_at_the_extreme_pixels(small_params: ResolutionParams) -> None:
    frame = make_frame(small_params)
    stats = frame.stats()

    assert frame.temperature_at(stats.maximum.x, stats.maximum.y) == pytest.approx(
        stats.maximum.celsius
    )
    assert frame.temperature_at(stats.minimum.x, stats.minimum.y) == pytest.approx(
        stats.minimum.celsius
    )


def test_region_stats_are_confined_to_the_region(small_params: ResolutionParams) -> None:
    frame = make_frame(small_params)
    region = frame.region_stats(10, 10, 30, 30)

    assert 10 <= region.maximum.x < 30
    assert 10 <= region.maximum.y < 30


def test_region_accepts_reversed_coordinates(small_params: ResolutionParams) -> None:
    frame = make_frame(small_params)
    assert frame.region_stats(30, 30, 10, 10) == frame.region_stats(10, 10, 30, 30)


def test_region_is_clipped_to_the_frame(small_params: ResolutionParams) -> None:
    frame = make_frame(small_params)
    assert frame.region_stats(-50, -50, 10_000, 10_000) == frame.stats()


def test_empty_region_is_rejected(small_params: ResolutionParams) -> None:
    with pytest.raises(ValueError, match="empty region"):
        make_frame(small_params).region_stats(10, 10, 10, 20)


def test_center_is_the_middle_pixel(small_params: ResolutionParams) -> None:
    frame = make_frame(small_params)
    center = frame.center()

    assert (center.x, center.y) == (frame.width // 2, frame.height // 2)


def test_frame_pixels_are_read_only(small_params: ResolutionParams) -> None:
    frame = make_frame(small_params)
    with pytest.raises(ValueError, match="read-only"):
        frame.raw[0, 0] = 1


# --- clamping (v1 clipped to 65535 against a 16384-entry table) --------------


def test_clamp_leaves_normal_pixels_untouched(small_params: ResolutionParams) -> None:
    image = np.full((4, 4), 8000, dtype=np.uint16)
    assert clamp_to_table(image) is image


def test_clamp_caps_out_of_range_pixels() -> None:
    image = np.array([[0, 60_000], [16_383, 16_384]], dtype=np.uint16)
    clamped = clamp_to_table(image)

    assert int(clamped.max()) == MAX_RAW_INDEX


def test_clamped_pixels_can_index_the_table(small_params: ResolutionParams) -> None:
    """The point of clamping: no IndexError, whatever the sensor sends.

    v1 clipped corrected pixels to 65535 and then indexed a 16384-entry table
    with them.
    """
    spec = FrameSpec(image=np.full(small_params.pixel_count, 60_000, dtype=np.uint16))
    frame = make_frame(small_params, spec)

    assert int(frame.raw.max()) == MAX_RAW_INDEX
    temperatures = frame.temperatures()
    assert temperatures.shape == (frame.height, frame.width)
    assert frame.temperature_at(0, 0) == pytest.approx(float(frame.table[MAX_RAW_INDEX]))


# --- raw-mode correction ----------------------------------------------------


def test_corrector_passes_frames_through_until_calibrated() -> None:
    corrector = RawCorrector()
    image = np.full((8, 8), 8000, dtype=np.uint16)

    assert corrector.calibrated is False
    assert np.array_equal(corrector.apply(image), image)


def test_corrector_removes_the_fixed_pattern() -> None:
    """A per-pixel offset should vanish once the reference is subtracted."""
    rng = np.random.default_rng(0)
    pattern = rng.integers(0, 500, size=(16, 16), dtype=np.uint16)
    scene = np.full((16, 16), 8000, dtype=np.uint16)

    corrector = RawCorrector()
    corrector.calibrate(pattern)
    corrected = corrector.apply((scene + pattern).astype(np.uint16))

    assert corrected.std() < pattern.std()


def test_corrector_output_stays_in_table_range() -> None:
    corrector = RawCorrector()
    corrector.calibrate(np.zeros((8, 8), dtype=np.uint16))
    corrected = corrector.apply(np.full((8, 8), 60_000, dtype=np.uint16))

    assert int(corrected.max()) <= MAX_RAW_INDEX


def test_corrector_flags_dead_pixels() -> None:
    reference = np.full((8, 8), 8000, dtype=np.uint16)
    reference[0, 0] = 0

    corrector = RawCorrector()
    corrector.calibrate(reference)

    assert corrector.dead_pixel_count >= 1


def test_reset_discards_the_calibration() -> None:
    corrector = RawCorrector()
    corrector.calibrate(np.full((8, 8), 8000, dtype=np.uint16))
    assert corrector.calibrated

    corrector.reset()
    assert corrector.calibrated is False
    assert corrector.dead_pixel_count == 0
