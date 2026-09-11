"""Behaviour that is new in v2, including the v1 bugs it closes."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import RAW_CAPABLE_WIDTHS, FrameSpec, build_frame
from irthermal.core.radiometry import (
    HIGH_RANGE_MODE,
    RadiometryAdjustments,
    RadiometryInputs,
    build_temperature_table,
    clear_table_cache,
    temperature_table,
)
from irthermal.core.resolutions import (
    MAX_RAW_VALUE,
    SUPPORTED_RESOLUTIONS,
    ResolutionParams,
    UnsupportedResolutionError,
    params_for_resolution,
    params_for_width,
)
from irthermal.core.telemetry import TruncatedFrameError, parse_telemetry

NON_RAW_WIDTHS = tuple(w for w in (240, 256, 384, 640) if w not in RAW_CAPABLE_WIDTHS)


def _inputs(params: ResolutionParams, **kwargs: object) -> RadiometryInputs:
    telemetry = parse_telemetry(build_frame(params), params)
    return RadiometryInputs.from_telemetry(telemetry, params, **kwargs)  # type: ignore[arg-type]


# --- truncated frames (v1 accepted them, then read out of bounds) ------------


def test_truncated_frame_is_rejected(params: ResolutionParams) -> None:
    """v1 only checked for the visible pixels, then indexed into telemetry anyway."""
    words = build_frame(params)[: params.pixel_count + 4]

    with pytest.raises(TruncatedFrameError, match="telemetry rows"):
        parse_telemetry(words, params)


def test_frame_of_exact_length_is_accepted(params: ResolutionParams) -> None:
    words = build_frame(params)
    assert words.size == params.total_words
    assert parse_telemetry(words, params).serial == "SN0042"


def test_visible_pixels_alone_are_not_enough(params: ResolutionParams) -> None:
    with pytest.raises(TruncatedFrameError):
        parse_telemetry(build_frame(params)[: params.pixel_count], params)


# --- the out-of-range shutter trim on 384/640 (v1 raised IndexError) ---------


@pytest.mark.parametrize("width", NON_RAW_WIDTHS)
def test_raw_mode_without_trim_byte_degrades_instead_of_crashing(width: int) -> None:
    params = params_for_width(width)
    trim_index = params.pixel_count + params.amount_pixels * 2 + 47
    assert trim_index >= params.total_words, "fixture assumption: trim is out of range here"

    telemetry = parse_telemetry(build_frame(params), params, camera_raw=True)

    assert telemetry.shutter_trim_applied is False
    assert telemetry.shutter_temp_c == telemetry.core_temp_c


@pytest.mark.parametrize("width", RAW_CAPABLE_WIDTHS)
def test_raw_mode_applies_trim_where_the_layout_has_one(width: int) -> None:
    params = params_for_width(width)
    spec = FrameSpec(shutter_trim_word=0x0500)  # high byte 5 -> +0.5 C
    telemetry = parse_telemetry(build_frame(params, spec), params, camera_raw=True)

    assert telemetry.shutter_trim_applied is True
    assert telemetry.shutter_temp_c - telemetry.core_temp_c == pytest.approx(0.5)


# --- resolution lookup ------------------------------------------------------


def test_supported_resolutions_are_the_documented_four() -> None:
    documented = {(240, 180), (256, 192), (384, 288), (640, 512)}
    assert set(SUPPORTED_RESOLUTIONS) == documented


def test_unsupported_width_names_the_supported_ones() -> None:
    with pytest.raises(UnsupportedResolutionError, match="240x180"):
        params_for_width(320)


def test_mismatched_height_is_rejected() -> None:
    """A 256-wide sensor that reports 288 rows is not a camera we understand."""
    with pytest.raises(UnsupportedResolutionError, match="implies"):
        params_for_resolution(256, 288)


def test_total_words_accounts_for_telemetry_rows(params: ResolutionParams) -> None:
    assert params.total_words == params.pixel_count + 4 * params.width


# --- table cache ------------------------------------------------------------


def test_cache_returns_the_same_array_for_equal_inputs(params: ResolutionParams) -> None:
    clear_table_cache()
    inputs = _inputs(params)

    assert temperature_table(inputs) is temperature_table(inputs)


def test_cached_table_is_read_only(params: ResolutionParams) -> None:
    clear_table_cache()
    table = temperature_table(_inputs(params))

    with pytest.raises(ValueError, match="read-only"):
        table[0] = 0.0


def test_cache_rebuilds_when_an_adjustment_changes(params: ResolutionParams) -> None:
    clear_table_cache()
    base = temperature_table(_inputs(params))
    shifted = temperature_table(_inputs(params, adjustments=RadiometryAdjustments(user_offset=5.0)))

    assert base is not shifted
    finite = np.isfinite(base) & np.isfinite(shifted)
    assert np.allclose(shifted[finite] - base[finite], 5.0)


def test_table_covers_the_whole_raw_range(params: ResolutionParams) -> None:
    assert build_temperature_table(_inputs(params)).shape == (MAX_RAW_VALUE,)


# --- range modes ------------------------------------------------------------


def test_high_range_mode_skips_the_fpa_offset_correction() -> None:
    """Only the normal range applies the cal_00 correction; v1 keyed this off
    a `range` attribute that nothing ever set to anything but 120."""
    params = params_for_width(240)  # cal_00_fpa_multiplier is non-zero here
    normal = build_temperature_table(_inputs(params))
    high = build_temperature_table(
        _inputs(params, adjustments=RadiometryAdjustments(range_mode=HIGH_RANGE_MODE))
    )

    finite = np.isfinite(normal) & np.isfinite(high)
    assert not np.allclose(normal[finite], high[finite])


def test_gain_correction_is_linear() -> None:
    params = params_for_width(256)
    base = build_temperature_table(_inputs(params))
    scaled = build_temperature_table(
        _inputs(params, adjustments=RadiometryAdjustments(gain_m=1.17, gain_b=-40.9))
    )

    finite = np.isfinite(base) & np.isfinite(scaled)
    assert np.allclose(scaled[finite], 1.17 * base[finite] - 40.9)
