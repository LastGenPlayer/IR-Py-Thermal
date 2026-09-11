"""Synthetic frame construction.

Real captures would be better fixtures, but none are checked in and the
hardware is not available to CI. These frames are built to the same wire layout
the parser expects, with coefficients chosen so the temperature curve lands in a
plausible range rather than saturating into NaN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from irthermal.core.resolutions import ResolutionParams, params_for_width

ALL_WIDTHS = (240, 256, 384, 640)

# Widths whose layout actually contains the raw-mode shutter trim byte.
RAW_CAPABLE_WIDTHS = (240, 256)


@dataclass
class FrameSpec:
    """Values written into a synthetic frame's telemetry rows."""

    fpa_average: int = 8600
    # Chosen per-resolution in `build_frame` so that fpa_temp_c lands on 20 C.
    fpa_temp_raw: int | None = None

    max_point: tuple[int, int] = (100, 50)
    max_raw: int = 9000
    min_point: tuple[int, int] = (10, 20)
    min_raw: int = 7000
    average_raw: int = 8000
    center_raw: int = 8200
    user_raw: tuple[int, int, int] = (8100, 8300, 8400)

    cal_00: int = 8000
    shutter_word: int = 3031
    core_word: int = 3100
    # cal_01/02 set the curve shape; cal_03..05 make the FPA gain ~0.376 at 20 C,
    # which puts mid-scale raw values around body temperature.
    cal_01: float = 1.0
    cal_02: float = 0.5
    cal_03: float = 1.0e-4
    cal_04: float = 1.0e-3
    cal_05: float = 0.316

    correction: float = 0.0
    reflected_temp_c: float = 25.0
    air_temp_c: float = 25.0
    humidity: float = 0.45
    emissivity: float = 0.95
    distance_m: int = 1

    firmware: str = "V1.2.3"
    serial: str = "SN0042"
    shutter_trim_word: int = 0x0500  # high byte 5 -> +0.5 C trim

    image: np.ndarray | None = field(default=None, repr=False)


def _write_f32(words: np.ndarray, offset: int, value: float) -> None:
    words[offset : offset + 2] = np.array([value], dtype=np.float32).view(np.uint16)


def _write_ascii(words: np.ndarray, offset: int, length: int, text: str) -> None:
    buffer = np.zeros(length * 2, dtype=np.uint8)
    encoded = text.encode("ascii")[: length * 2]
    buffer[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    words[offset : offset + length] = buffer.view(np.uint16)


def build_frame(params: ResolutionParams, spec: FrameSpec | None = None) -> np.ndarray:
    """Assemble a complete frame: visible pixels plus telemetry rows."""
    spec = spec or FrameSpec()
    words = np.zeros(params.total_words, dtype=np.uint16)

    if spec.image is not None:
        words[: params.pixel_count] = spec.image.ravel().astype(np.uint16)
    else:
        # A ramp through the plausible ADC band, deterministic across runs.
        ramp = np.linspace(7000, 9500, params.pixel_count, dtype=np.float64)
        words[: params.pixel_count] = ramp.astype(np.uint16)

    image = params.pixel_count
    block = image + params.amount_pixels
    user = image + params.user_area

    fpa_temp_raw = spec.fpa_temp_raw if spec.fpa_temp_raw is not None else params.fpa_offset

    words[image + 0] = spec.fpa_average
    words[image + 1] = fpa_temp_raw
    words[image + 2], words[image + 3] = spec.max_point
    words[image + 4] = spec.max_raw
    words[image + 5], words[image + 6] = spec.min_point
    words[image + 7] = spec.min_raw
    words[image + 8] = spec.average_raw
    words[image + 12] = spec.center_raw
    words[image + 13], words[image + 14], words[image + 15] = spec.user_raw

    words[block + 0] = spec.cal_00
    words[block + 1] = spec.shutter_word
    words[block + 2] = spec.core_word
    _write_f32(words, block + 3, spec.cal_01)
    _write_f32(words, block + 5, spec.cal_02)
    _write_f32(words, block + 7, spec.cal_03)
    _write_f32(words, block + 9, spec.cal_04)
    _write_f32(words, block + 11, spec.cal_05)
    _write_ascii(words, block + 24, 8, spec.firmware)
    _write_ascii(words, block + 32, 3, spec.serial)

    _write_f32(words, user + 0, spec.correction)
    _write_f32(words, user + 2, spec.reflected_temp_c)
    _write_f32(words, user + 4, spec.air_temp_c)
    _write_f32(words, user + 6, spec.humidity)
    _write_f32(words, user + 8, spec.emissivity)
    words[user + 10] = spec.distance_m

    trim_index = image + params.amount_pixels * 2 + 47
    if trim_index < params.total_words:
        words[trim_index] = spec.shutter_trim_word

    return words


def random_spec(rng: np.random.Generator) -> FrameSpec:
    """A spec with randomised but physically plausible coefficients."""
    return FrameSpec(
        fpa_temp_raw=int(rng.integers(6000, 9500)),
        cal_00=int(rng.integers(7000, 9000)),
        shutter_word=int(rng.integers(2900, 3200)),
        core_word=int(rng.integers(2900, 3200)),
        cal_01=float(rng.uniform(0.5, 2.0)),
        cal_02=float(rng.uniform(-1.0, 1.0)),
        cal_03=float(rng.uniform(-1e-3, 1e-3)),
        cal_04=float(rng.uniform(-1e-2, 1e-2)),
        cal_05=float(rng.uniform(0.1, 0.8)),
        correction=float(rng.uniform(-2.0, 2.0)),
        reflected_temp_c=float(rng.uniform(10.0, 35.0)),
        air_temp_c=float(rng.uniform(10.0, 35.0)),
        humidity=float(rng.uniform(0.1, 0.9)),
        emissivity=float(rng.uniform(0.7, 1.0)),
        distance_m=int(rng.integers(0, 30)),
        shutter_trim_word=int(rng.integers(0, 0xFFFF)),
    )


@pytest.fixture(params=ALL_WIDTHS, ids=lambda w: f"{w}px")
def params(request: Any) -> ResolutionParams:
    """Every supported resolution, one test run each."""
    return params_for_width(request.param)
