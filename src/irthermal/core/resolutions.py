"""Sensor geometry and the per-resolution calibration constants that go with it.

The cameras append a block of telemetry rows after the visible image. Everything
in this module describes where that block sits and which magic numbers apply to
a given sensor width.
"""

from __future__ import annotations

from dataclasses import dataclass

SPECIAL_ROWS = 4
"""Rows of non-image telemetry the sensor appends after the visible frame."""

MAX_RAW_VALUE = 16384
"""Number of entries in the raw-ADC -> temperature lookup table."""

MAX_RAW_INDEX = MAX_RAW_VALUE - 1
"""Highest raw ADC value that can legally index the lookup table."""


@dataclass(frozen=True, slots=True)
class ResolutionParams:
    """Fixed constants for one supported sensor resolution."""

    width: int
    height: int
    fpa_offset: int
    fpa_divisor: float
    amount_pixels: int
    cal_00_offset: float
    cal_00_fpa_multiplier: float

    @property
    def pixel_count(self) -> int:
        """Number of u16 words making up the visible image."""
        return self.width * self.height

    @property
    def user_area(self) -> int:
        """Offset of the user-settable parameter block, relative to `pixel_count`."""
        return self.amount_pixels + 127

    @property
    def total_words(self) -> int:
        """Total u16 words in a complete frame, telemetry rows included."""
        return self.width * (self.height + SPECIAL_ROWS)


_BY_WIDTH: dict[int, ResolutionParams] = {
    240: ResolutionParams(
        width=240,
        height=180,
        fpa_offset=7800,
        fpa_divisor=36.0,
        amount_pixels=240,
        cal_00_offset=390.0,
        cal_00_fpa_multiplier=7.05,
    ),
    256: ResolutionParams(
        width=256,
        height=192,
        fpa_offset=8617,
        fpa_divisor=37.682,
        amount_pixels=256,
        cal_00_offset=170.0,
        cal_00_fpa_multiplier=0.0,
    ),
    384: ResolutionParams(
        width=384,
        height=288,
        fpa_offset=7800,
        fpa_divisor=36.0,
        amount_pixels=384 * 3,
        cal_00_offset=390.0,
        cal_00_fpa_multiplier=7.05,
    ),
    640: ResolutionParams(
        width=640,
        height=512,
        fpa_offset=6867,
        fpa_divisor=33.8,
        amount_pixels=640 * 3,
        cal_00_offset=390.0,
        cal_00_fpa_multiplier=7.05,
    ),
}

SUPPORTED_RESOLUTIONS: frozenset[tuple[int, int]] = frozenset(
    (params.width, params.height) for params in _BY_WIDTH.values()
)
"""Visible-image resolutions this driver understands, telemetry rows excluded."""


class UnsupportedResolutionError(ValueError):
    """Raised when a camera reports a resolution we have no constants for."""


def params_for_width(width: int) -> ResolutionParams:
    """Look up sensor constants by image width."""
    try:
        return _BY_WIDTH[width]
    except KeyError:
        supported = ", ".join(f"{w}x{h}" for w, h in sorted(SUPPORTED_RESOLUTIONS))
        msg = f"Unsupported camera width {width}; supported resolutions: {supported}"
        raise UnsupportedResolutionError(msg) from None


def params_for_resolution(width: int, height: int) -> ResolutionParams:
    """Look up sensor constants for a visible-image resolution.

    `height` must exclude the telemetry rows.
    """
    params = params_for_width(width)
    if params.height != height:
        msg = (
            f"Camera reported {width}x{height} but width {width} implies "
            f"{params.width}x{params.height}"
        )
        raise UnsupportedResolutionError(msg)
    return params


def is_supported(width: int, height: int) -> bool:
    """Whether a visible-image resolution is supported."""
    return (width, height) in SUPPORTED_RESOLUTIONS
