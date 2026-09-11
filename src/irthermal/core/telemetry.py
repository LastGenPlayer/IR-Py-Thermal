"""Decode the telemetry rows the sensor appends after each visible frame.

The camera streams `height + SPECIAL_ROWS` rows. The visible image is the first
`width * height` u16 words; everything after that is a packed struct holding
calibration coefficients, factory strings and the sensor's own min/max/centre
measurements.

Calibration coefficients arrive as little-endian float32 and are widened to
Python floats here. That widening is exact -- every float32 has an exact float64
representation -- so no information is lost, and it keeps the downstream
arithmetic at a single, explicit precision instead of leaving it at the mercy of
NumPy scalar-promotion rules, which changed in NumPy 2.0. See `radiometry`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from irthermal.core.resolutions import SPECIAL_ROWS, ResolutionParams

FrameWords = npt.NDArray[np.uint16]
"""One complete frame as flat u16 words: visible pixels then telemetry rows."""

ZERO_CELSIUS = 273.15

_FIRMWARE_WORDS = 8
_SERIAL_WORDS = 3
_SHUTTER_FOLD_POINT = 2049
_SHUTTER_TRIM_OFFSET = 47


class TruncatedFrameError(ValueError):
    """Raised when a frame is too short to contain its telemetry rows.

    The visible image alone is not enough. A frame that carries the pixels but
    loses the trailing rows would otherwise be parsed out of bounds.
    """


@dataclass(frozen=True, slots=True)
class CameraTelemetry:
    """Everything the sensor reports alongside one frame."""

    # Thermal state of the camera itself.
    shutter_temp_c: float
    core_temp_c: float
    fpa_temp_c: float
    fpa_average: int

    # The sensor's own measurements, as raw ADC values.
    min_point: tuple[int, int]
    min_raw: int
    max_point: tuple[int, int]
    max_raw: int
    average_raw: int
    center_raw: int
    user_raw: tuple[int, int, int]

    # Calibration coefficients.
    cal_00: float
    cal_01: float
    cal_02: float
    cal_03: float
    cal_04: float
    cal_05: float

    # User-settable measurement parameters, read back from the camera.
    correction: float
    reflected_temp_c: float
    air_temp_c: float
    humidity: float
    emissivity: float
    distance_m: int

    # Factory identification.
    firmware: str
    serial: str

    shutter_trim_applied: bool = True
    """False when a raw-mode shutter trim byte was expected but unreachable.

    The trim byte sits at `amount_pixels * 2 + 47`, which falls beyond the end
    of the frame on the 384x288 and 640x512 layouts. Raw mode is a 256x192
    feature in practice, so this has never mattered in the field; the flag lets
    the camera layer warn instead of the parser reading out of bounds.
    """


def _u16(words: FrameWords, offset: int) -> int:
    return int(words[offset])


def _f32(words: FrameWords, offset: int) -> float:
    """Read a little-endian float32 and widen it losslessly to a Python float."""
    return float(words[offset : offset + 2].view(np.float32)[0])


def _ascii(words: FrameWords, offset: int, length: int) -> str:
    raw = words[offset : offset + length].view(np.uint8)
    return raw.tobytes().decode("ascii", errors="ignore").rstrip("\x00")


def high_byte_signed(value: int) -> int:
    """Interpret the high byte of a u16 as a signed 8-bit integer."""
    high = (value >> 8) & 0xFF
    return high - 0x100 if high & 0x80 else high


def _decode_shutter_and_core(
    words: FrameWords,
    params: ResolutionParams,
    *,
    camera_raw: bool,
) -> tuple[float, float, bool]:
    """Recover shutter and core temperatures in Celsius.

    Cameras that do their own processing report both directly in decikelvin.
    Cameras streaming unprocessed sensor data encode the shutter temperature as
    a signed offset that has to be unfolded, and carry a separate trim byte.

    Returns (shutter, core, trim_applied).
    """
    base = params.pixel_count + params.amount_pixels
    shutter_word = _u16(words, base + 1)
    core_word = _u16(words, base + 2)

    if not camera_raw:
        return shutter_word / 10.0 - ZERO_CELSIUS, core_word / 10.0 - ZERO_CELSIUS, True

    if shutter_word < _SHUTTER_FOLD_POINT:
        magnitude = float(shutter_word)
        scale = 0.625
    else:
        magnitude = float(0xFFF - shutter_word)
        scale = -0.625

    core_temp = (magnitude * scale + 2731.5) / 10.0 - ZERO_CELSIUS

    trim_index = params.pixel_count + params.amount_pixels * 2 + _SHUTTER_TRIM_OFFSET
    if trim_index >= params.total_words:
        # No trim byte in this layout -- see CameraTelemetry.shutter_trim_applied.
        return core_temp, core_temp, False

    shutter_fix = high_byte_signed(_u16(words, trim_index)) / 10.0
    return core_temp + shutter_fix, core_temp, True


def parse_telemetry(
    words: FrameWords,
    params: ResolutionParams,
    *,
    camera_raw: bool = False,
) -> CameraTelemetry:
    """Decode the telemetry rows of one complete frame.

    `words` is the whole frame as a flat u16 array, visible pixels included.
    """
    required = params.total_words
    if words.size < required:
        msg = (
            f"Frame holds {words.size} u16 words, need {required} "
            f"({params.width}x{params.height} plus {SPECIAL_ROWS} telemetry rows)"
        )
        raise TruncatedFrameError(msg)

    image = params.pixel_count
    block = image + params.amount_pixels
    user = image + params.user_area

    shutter_temp_c, core_temp_c, trim_applied = _decode_shutter_and_core(
        words, params, camera_raw=camera_raw
    )
    fpa_temp_raw = _u16(words, image + 1)

    return CameraTelemetry(
        shutter_temp_c=shutter_temp_c,
        core_temp_c=core_temp_c,
        fpa_temp_c=20.0 - (float(fpa_temp_raw) - params.fpa_offset) / params.fpa_divisor,
        fpa_average=_u16(words, image + 0),
        max_point=(_u16(words, image + 2), _u16(words, image + 3)),
        max_raw=_u16(words, image + 4),
        min_point=(_u16(words, image + 5), _u16(words, image + 6)),
        min_raw=_u16(words, image + 7),
        average_raw=_u16(words, image + 8),
        center_raw=_u16(words, image + 12),
        user_raw=(
            _u16(words, image + 13),
            _u16(words, image + 14),
            _u16(words, image + 15),
        ),
        cal_00=float(_u16(words, block + 0)),
        cal_01=_f32(words, block + 3),
        cal_02=_f32(words, block + 5),
        cal_03=_f32(words, block + 7),
        cal_04=_f32(words, block + 9),
        cal_05=_f32(words, block + 11),
        correction=_f32(words, user + 0),
        reflected_temp_c=_f32(words, user + 2),
        air_temp_c=_f32(words, user + 4),
        humidity=_f32(words, user + 6),
        emissivity=_f32(words, user + 8),
        distance_m=_u16(words, user + 10),
        firmware=_ascii(words, block + 24, _FIRMWARE_WORDS),
        serial=_ascii(words, block + 32, _SERIAL_WORDS),
        shutter_trim_applied=trim_applied,
    )
