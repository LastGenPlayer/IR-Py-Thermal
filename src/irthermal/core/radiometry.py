"""Raw ADC value to temperature conversion.

The sensor never reports temperature directly. Each frame ships calibration
coefficients that define a curve mapping the 14-bit ADC range onto Celsius; the
camera's own thermal state (shutter and focal-plane-array temperature) shifts
that curve as the hardware warms up, and the scene parameters (emissivity,
humidity, distance, reflected temperature) correct for the atmosphere between
the lens and the target.

Building the table is the single most expensive per-frame operation in the
pipeline, so `temperature_table` memoises it. The inputs are all quantised by
the sensor already -- they are u16 or f32 words straight off the wire -- which
means exact-match caching hits on almost every frame without approximating
anything. A rebuild only happens when the camera's thermal state actually moves.

All arithmetic runs in float64. The original implementation mixed float32
arrays with float64 scalars and let NumPy decide, which means its output dtype
-- and therefore the temperatures it reported -- depended on the installed NumPy
version, since NEP 50 changed scalar promotion in NumPy 2.0. Pinning the
precision explicitly removes that dependency. The difference against the old
float32 path is on the order of 1e-4 C, some three orders of magnitude below the
sensor's own resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import numpy.typing as npt

from .resolutions import MAX_RAW_VALUE, ResolutionParams
from .telemetry import ZERO_CELSIUS, CameraTelemetry

TemperatureTable = npt.NDArray[np.float64]

NORMAL_RANGE_MODE = 120
"""Range mode in which the table offset gets an FPA-dependent correction."""

HIGH_RANGE_MODE = 400
"""Extended range mode. Skips the FPA-dependent table offset correction."""

_MAX_USEFUL_DISTANCE_M = 20.0


@dataclass(frozen=True, slots=True)
class RadiometryAdjustments:
    """Operator-controlled corrections applied on top of the camera's own data.

    Kept separate from the per-frame coefficients because these change when a
    human changes them, not every frame -- which is what makes caching the
    resulting table worthwhile.
    """

    range_mode: int = NORMAL_RANGE_MODE
    distance_multiplier: float = 1.0
    shutter_temp_offset: float = 0.0
    fpa_temp_offset: float = 0.0

    gain_m: float = 1.0
    """Slope of the post-hoc linear correction. The extended temperature range
    needs its own slope and intercept; normal range leaves these at identity."""
    gain_b: float = 0.0

    user_offset: float = 0.0
    """Constant trim in Celsius, per camera, found empirically."""


@dataclass(frozen=True, slots=True)
class RadiometryInputs:
    """Everything needed to build one temperature table.

    Frozen and hashable so it can key the table cache directly.
    """

    # Per-frame calibration coefficients.
    cal_00: float
    cal_01: float
    cal_02: float
    cal_03: float
    cal_04: float
    cal_05: float

    # Camera thermal state.
    shutter_temp_c: float
    fpa_temp_c: float

    # Scene parameters.
    correction: float
    reflected_temp_c: float
    air_temp_c: float
    humidity: float
    emissivity: float
    distance_m: int

    # Resolution-dependent constants.
    cal_00_offset: float
    cal_00_fpa_multiplier: float

    adjustments: RadiometryAdjustments = field(default_factory=RadiometryAdjustments)

    @classmethod
    def from_telemetry(
        cls,
        telemetry: CameraTelemetry,
        params: ResolutionParams,
        adjustments: RadiometryAdjustments | None = None,
    ) -> RadiometryInputs:
        """Build inputs from a decoded frame plus the current operator settings."""
        return cls(
            cal_00=telemetry.cal_00,
            cal_01=telemetry.cal_01,
            cal_02=telemetry.cal_02,
            cal_03=telemetry.cal_03,
            cal_04=telemetry.cal_04,
            cal_05=telemetry.cal_05,
            shutter_temp_c=telemetry.shutter_temp_c,
            fpa_temp_c=telemetry.fpa_temp_c,
            correction=telemetry.correction,
            reflected_temp_c=telemetry.reflected_temp_c,
            air_temp_c=telemetry.air_temp_c,
            humidity=telemetry.humidity,
            emissivity=telemetry.emissivity,
            distance_m=telemetry.distance_m,
            cal_00_offset=params.cal_00_offset,
            cal_00_fpa_multiplier=params.cal_00_fpa_multiplier,
            adjustments=adjustments or RadiometryAdjustments(),
        )


def water_vapour_content(humidity: float, air_temp_c: float) -> float:
    """Atmospheric water vapour content, in the vendor's arbitrary units."""
    h1, h2, h3, h4 = 1.5587, 0.06939, -2.7816e-4, 6.8455e-7
    return humidity * math.exp(
        h1 + h2 * air_temp_c + h3 * math.pow(air_temp_c, 2) + h4 * math.pow(air_temp_c, 3)
    )


def atmospheric_transmission(humidity: float, air_temp_c: float, distance_m: float) -> float:
    """Fraction of radiance surviving `distance_m` of air."""
    k_atm = 1.9
    a1, a2 = 0.006569, 0.01262
    b1, b2 = -0.002276, -0.00667

    neg_sqrt_distance = -math.sqrt(distance_m)
    sqrt_water = math.sqrt(water_vapour_content(humidity, air_temp_c))

    return k_atm * math.exp(neg_sqrt_distance * (a1 + b1 * sqrt_water)) + (1.0 - k_atm) * math.exp(
        neg_sqrt_distance * (a2 + b2 * sqrt_water)
    )


def build_temperature_table(inputs: RadiometryInputs) -> TemperatureTable:
    """Compute the raw-ADC -> Celsius lookup table. Always rebuilds.

    The result has `MAX_RAW_VALUE` float64 entries; index it with raw pixels.

    Entries outside the sensor's valid radiance band come back as NaN, which is
    what the vendor arithmetic produces for a radiance below the reflected-
    radiance floor. Callers decide what to do about that; nothing is clamped
    here.
    """
    adjust = inputs.adjustments

    effective_distance = (
        min(float(inputs.distance_m), _MAX_USEFUL_DISTANCE_M) * adjust.distance_multiplier
    )
    transmission = atmospheric_transmission(inputs.humidity, inputs.air_temp_c, effective_distance)

    # Radiance contributed by everything that is not the target itself: the
    # reflected background, plus the air column's own emission.
    background_radiance = (1.0 - inputs.emissivity) * transmission * math.pow(
        inputs.reflected_temp_c + ZERO_CELSIUS, 4
    ) + (1.0 - transmission) * math.pow(inputs.air_temp_c + ZERO_CELSIUS, 4)
    target_weight = inputs.emissivity * transmission

    shutter_temp = inputs.shutter_temp_c + adjust.shutter_temp_offset
    fpa_temp = inputs.fpa_temp_c + adjust.fpa_temp_offset

    curve_shift = inputs.cal_02 / (inputs.cal_01 * 2.0)
    curve_bias = inputs.cal_02 * inputs.cal_02 / (inputs.cal_01 * inputs.cal_01 * 4.0)
    shutter_term = inputs.cal_01 * math.pow(shutter_temp, 2) + shutter_temp * inputs.cal_02
    fpa_gain = inputs.cal_03 * math.pow(fpa_temp, 2) + inputs.cal_04 * fpa_temp + inputs.cal_05

    offset_correction = 0
    if adjust.range_mode == NORMAL_RANGE_MODE:
        offset_correction = int(inputs.cal_00_offset - fpa_temp * inputs.cal_00_fpa_multiplier)
    table_offset = inputs.cal_00 - max(offset_correction, 0)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        normalised = np.sqrt(
            np.abs(
                (
                    (np.arange(MAX_RAW_VALUE, dtype=np.float64) - table_offset) * fpa_gain
                    + shutter_term
                )
                / inputs.cal_01
                + curve_bias
            )
        )
        normalised[np.isnan(normalised)] = 0.0

        total_radiance = np.power(normalised - curve_shift + ZERO_CELSIUS, 4)
        target_temp = (
            np.power((total_radiance - background_radiance) / target_weight, 0.25) - ZERO_CELSIUS
        )

        table = (
            target_temp
            + (effective_distance * 0.85 - 1.125) * (target_temp - inputs.air_temp_c) / 100.0
            + inputs.correction
        )
        corrected: TemperatureTable = adjust.gain_m * table + adjust.gain_b + adjust.user_offset

    return corrected


@lru_cache(maxsize=16)
def _cached_table(inputs: RadiometryInputs) -> TemperatureTable:
    table = build_temperature_table(inputs)
    table.setflags(write=False)
    return table


def temperature_table(inputs: RadiometryInputs) -> TemperatureTable:
    """Memoised `build_temperature_table`.

    The returned array is read-only and shared between callers -- index it, do
    not mutate it.
    """
    return _cached_table(inputs)


def cache_info() -> str:
    """Human-readable hit/miss statistics for the table cache."""
    return str(_cached_table.cache_info())


def clear_table_cache() -> None:
    """Drop every memoised table. Intended for tests and camera teardown."""
    _cached_table.cache_clear()
