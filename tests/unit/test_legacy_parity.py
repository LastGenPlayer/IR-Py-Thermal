"""The v1 -> v2 regression guard.

Every number the new core produces is diffed against `reference_legacy`, the
transcribed pre-v2 pipeline. The one intentional divergence is arithmetic
precision: v1 computed the table in float32 (on NumPy 1.x -- on NumPy 2.x the
same source computes it in float64, which is the reason for pinning it), v2
computes in float64 throughout. The tolerance below is what that costs, and it
sits far under the sensor's own ~0.1 C resolution.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import RAW_CAPABLE_WIDTHS, FrameSpec, build_frame, random_spec
from irthermal.core import commands
from irthermal.core.radiometry import (
    RadiometryAdjustments,
    RadiometryInputs,
    build_temperature_table,
)
from irthermal.core.resolutions import ResolutionParams, params_for_width
from irthermal.core.telemetry import high_byte_signed, parse_telemetry
from reference_legacy import (
    bin_to_twos_complement,
    legacy_float_command,
    legacy_info_and_table,
    legacy_ushort_command,
)

# Precision budget for the float32 -> float64 change, in degrees Celsius.
PRECISION_TOLERANCE_C = 1.0e-3


def _new_table(
    words: np.ndarray,
    params: ResolutionParams,
    *,
    camera_raw: bool = False,
    user_offset: float = 0.0,
) -> np.ndarray:
    telemetry = parse_telemetry(words, params, camera_raw=camera_raw)
    adjustments = RadiometryAdjustments(user_offset=user_offset)
    inputs = RadiometryInputs.from_telemetry(telemetry, params, adjustments)
    return build_temperature_table(inputs)


def _max_deviation(new: np.ndarray, old: np.ndarray) -> float:
    """Largest absolute difference, ignoring entries both agree are NaN."""
    new_nan, old_nan = np.isnan(new), np.isnan(old)
    assert np.array_equal(new_nan, old_nan), "NaN entries differ between implementations"
    if new_nan.all():
        return 0.0
    return float(np.nanmax(np.abs(new[~new_nan] - old[~old_nan])))


def test_temperature_table_matches_legacy(params: ResolutionParams) -> None:
    words = build_frame(params)
    new = _new_table(words, params)
    _, old = legacy_info_and_table(words, params.width, params.height)

    assert new.shape == old.shape
    assert _max_deviation(new, old) < PRECISION_TOLERANCE_C


@pytest.mark.parametrize("width", RAW_CAPABLE_WIDTHS)
def test_temperature_table_matches_legacy_in_raw_mode(width: int) -> None:
    params = params_for_width(width)
    words = build_frame(params)

    new = _new_table(words, params, camera_raw=True)
    _, old = legacy_info_and_table(words, params.width, params.height, camera_raw=True)

    assert _max_deviation(new, old) < PRECISION_TOLERANCE_C


def test_temperature_table_matches_legacy_with_user_offset(params: ResolutionParams) -> None:
    words = build_frame(params)
    new = _new_table(words, params, user_offset=-12.3)
    _, old = legacy_info_and_table(words, params.width, params.height, user_offset=-12.3)

    assert _max_deviation(new, old) < PRECISION_TOLERANCE_C


@pytest.mark.parametrize("seed", range(25))
def test_temperature_table_matches_legacy_on_random_coefficients(seed: int) -> None:
    """Parity must hold for arbitrary coefficients, not just plausible ones."""
    rng = np.random.default_rng(seed)
    params = params_for_width(int(rng.choice([240, 256, 384, 640])))
    words = build_frame(params, random_spec(rng))

    new = _new_table(words, params)
    _, old = legacy_info_and_table(words, params.width, params.height)

    assert _max_deviation(new, old) < PRECISION_TOLERANCE_C


def test_telemetry_matches_legacy_info(params: ResolutionParams) -> None:
    words = build_frame(params)
    telemetry = parse_telemetry(words, params)
    old, _ = legacy_info_and_table(words, params.width, params.height)

    assert telemetry.shutter_temp_c == pytest.approx(old["temp_shutter"])
    assert telemetry.core_temp_c == pytest.approx(old["temp_core"])
    assert telemetry.fpa_temp_c == pytest.approx(old["temp_fpa"])
    assert telemetry.fpa_average == old["fpa_average"]
    assert telemetry.max_point == (old["temp_max_x"], old["temp_max_y"])
    assert telemetry.max_raw == old["temp_max_raw"]
    assert telemetry.min_point == (old["temp_min_x"], old["temp_min_y"])
    assert telemetry.min_raw == old["temp_min_raw"]
    assert telemetry.average_raw == old["temp_average_raw"]
    assert telemetry.center_raw == old["temp_center_raw"]
    assert telemetry.user_raw == old["user_raw"]
    assert telemetry.correction == pytest.approx(old["correction"])
    assert telemetry.reflected_temp_c == pytest.approx(old["temp_reflected"])
    assert telemetry.air_temp_c == pytest.approx(old["temp_air"])
    assert telemetry.humidity == pytest.approx(old["humidity"])
    assert telemetry.emissivity == pytest.approx(old["emissivity"])
    assert telemetry.distance_m == old["distance"]
    assert telemetry.firmware == old["cameraSoftVersion"]
    assert telemetry.serial == old["sn"]


@pytest.mark.parametrize("width", RAW_CAPABLE_WIDTHS)
def test_raw_shutter_decode_matches_legacy(width: int) -> None:
    params = params_for_width(width)
    # Both sides of the 2049 fold point, and a negative trim byte.
    for shutter_word, trim in ((1500, 0x0500), (3000, 0x8C00), (2049, 0x0000)):
        spec = FrameSpec(shutter_word=shutter_word, shutter_trim_word=trim)
        words = build_frame(params, spec)

        telemetry = parse_telemetry(words, params, camera_raw=True)
        old, _ = legacy_info_and_table(words, width, params.height, camera_raw=True)

        assert telemetry.shutter_temp_c == pytest.approx(old["temp_shutter"])
        assert telemetry.core_temp_c == pytest.approx(old["temp_core"])
        assert telemetry.shutter_trim_applied


def test_high_byte_signed_matches_legacy_bit_string() -> None:
    """Exhaustive: the arithmetic form must equal the old string slicing."""
    for value in range(0x10000):
        expected = bin_to_twos_complement(bin(value)[2:].zfill(16)[:8])
        assert high_byte_signed(value) == expected


def test_float_command_matches_legacy() -> None:
    for value in (0.0, 0.95, -12.3, 1e-4, 400.0):
        assert commands.float_command(commands.PARAM_EMISSIVITY, value) == (
            legacy_float_command(commands.PARAM_EMISSIVITY, value)
        )


def test_ushort_command_matches_legacy() -> None:
    for value in (0, 1, 20, 65535):
        assert commands.ushort_command(commands.PARAM_DISTANCE, value) == (
            legacy_ushort_command(commands.PARAM_DISTANCE, value)
        )


def test_point_command_matches_legacy_opcodes() -> None:
    # From the original Camera.set_point.
    expected = {0: (0xF000, 0xF200), 1: (0xF400, 0xF600), 2: (0xF800, 0xFA00)}
    for index, (x_base, y_base) in expected.items():
        assert commands.point_command(index, 7, 9) == [x_base + 7, y_base + 9]

    with pytest.raises(ValueError, match=r"0\.\.2"):
        commands.point_command(3, 0, 0)
