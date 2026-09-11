"""Verbatim transcription of the pre-v2 temperature pipeline.

This mirrors `irpythermal.Camera.info` / `get_temp_table` / `atmt` / `wvc` and
`_decode_shutter_core_temps` as they stood at commit 73c913c, with the OpenCV
dependency stripped out so it runs anywhere. It exists purely so the new
implementation can be diffed against the old one; production code must never
import it.

Nothing here is tidied. The odd spellings, the float32 arange, the mixed scalar
types and the `max(cal_00_corr, 0)` are all reproduced exactly, because the
whole point is to detect when the new code stops agreeing with them.
"""

from __future__ import annotations

import math

import numpy as np

ZEROC = 273.15
MAX_RAW_VALUE = 16384
ROWS_SPECIAL_DATA = 4

# Class attribute defaults from the original Camera.
DISTANCE_MULTIPLIER = 1.0
OFFSET_TEMP_SHUTTER = 0.0
OFFSET_TEMP_FPA = 0.0
RANGE = 120
CORRECTION_COEFFICIENT_M = 1.0
CORRECTION_COEFFICIENT_B = 0.0


def legacy_resolution_params(width: int) -> tuple[int, float, int, float, float]:
    """(fpa_off, fpa_div, amount_pixels, cal_00_offset, cal_00_fpamul)."""
    if width == 640:
        return (6867, 33.8, width * 3, 390.0, 7.05)
    if width == 384:
        return (7800, 36.0, width * 3, 390.0, 7.05)
    if width == 256:
        return (8617, 37.682, width, 170.0, 0.0)
    if width == 240:
        return (7800, 36.0, width, 390.0, 7.05)
    raise ValueError(f"Unsupported camera width: {width}")


def read_u16(arr_u16: np.ndarray, offset: int) -> np.uint16:
    return arr_u16[offset]


def read_f32(arr_u16: np.ndarray, offset: int, step: int = 2) -> np.float32:
    return arr_u16[offset : offset + step].view(np.float32)[0]


def read_u8(arr_u16: np.ndarray, offset: int, step: int) -> np.ndarray:
    return arr_u16[offset : offset + step].view(np.uint8)


def bin_to_twos_complement(binary: str) -> int:
    if binary[0] == "1":
        return int(binary, 2) - 2 ** len(binary)
    return int(binary, 2)


def wvc(humidity: float, t_atm: float) -> float:
    h1 = 1.5587
    h2 = 0.06939
    h3 = -2.7816e-4
    h4 = 6.8455e-7
    return humidity * math.exp(h1 + h2 * t_atm + h3 * math.pow(t_atm, 2) + h4 * math.pow(t_atm, 3))


def atmt(humidity: float, t_atm: float, distance: float) -> float:
    k_atm = 1.9
    nsqd = -math.sqrt(distance)
    sqw = math.sqrt(wvc(humidity, t_atm))

    a1 = 0.006569
    a2 = 0.01262
    b1 = -0.002276
    b2 = -0.00667

    return k_atm * math.exp(nsqd * (a1 + b1 * sqw)) + (1.0 - k_atm) * math.exp(
        nsqd * (a2 + b2 * sqw)
    )


def legacy_info_and_table(
    frame_raw_u16: np.ndarray,
    width: int,
    height: int,
    *,
    camera_raw: bool = False,
    user_offset: float = 0.0,
) -> tuple[dict, np.ndarray]:
    """The original `Camera.info()`, minus the coordinate sampling."""
    fpa_off, fpa_div, amount_pixels, cal_00_offset, cal_00_fpamul = legacy_resolution_params(width)
    four_line_para = width * height
    user_area = amount_pixels + 127

    # --- _decode_shutter_core_temps ---
    shut_temper = read_u16(frame_raw_u16, four_line_para + amount_pixels + 1)
    core_temper = read_u16(frame_raw_u16, four_line_para + amount_pixels + 2)

    shutter_fix = 0.0
    if camera_raw:
        if shut_temper < 2049:
            float_shutter = float(shut_temper)
            corr_factor = 0.625
        else:
            float_shutter = float(0xFFF - shut_temper)
            corr_factor = -0.625

        shutter_fix = (
            bin_to_twos_complement(
                bin(read_u16(frame_raw_u16, four_line_para + amount_pixels * 2 + 47))[2:].zfill(16)[
                    :8
                ]
            )
            / 10.0
        )
        float_shutter = (float_shutter * corr_factor + 2731.5) / 10.0 - 273.15
        float_shutter += shutter_fix
        float_core = float_shutter - shutter_fix
    else:
        float_shutter = shut_temper / 10.0 - ZEROC
        float_core = core_temper / 10.0 - ZEROC

    float_shut_temper, float_core_temper = float_shutter, float_core

    # --- info() ---
    cal_00 = float(read_u16(frame_raw_u16, four_line_para + amount_pixels))
    cal_01 = read_f32(frame_raw_u16, four_line_para + amount_pixels + 3)
    cal_02 = read_f32(frame_raw_u16, four_line_para + amount_pixels + 5)
    cal_03 = read_f32(frame_raw_u16, four_line_para + amount_pixels + 7)
    cal_04 = read_f32(frame_raw_u16, four_line_para + amount_pixels + 9)
    cal_05 = read_f32(frame_raw_u16, four_line_para + amount_pixels + 11)

    camera_soft_version = (
        read_u8(frame_raw_u16, four_line_para + amount_pixels + 24, step=8)
        .tobytes()
        .decode("ascii", errors="ignore")
        .rstrip("\x00")
    )
    sn = (
        read_u8(frame_raw_u16, four_line_para + amount_pixels + 32, step=3)
        .tobytes()
        .decode("ascii", errors="ignore")
        .rstrip("\x00")
    )

    correction = read_f32(frame_raw_u16, four_line_para + user_area)
    refl_tmp = read_f32(frame_raw_u16, four_line_para + user_area + 2)
    air_tmp = read_f32(frame_raw_u16, four_line_para + user_area + 4)
    humidity = read_f32(frame_raw_u16, four_line_para + user_area + 6)
    emiss = read_f32(frame_raw_u16, four_line_para + user_area + 8)
    distance = read_u16(frame_raw_u16, four_line_para + user_area + 10)

    fpa_avg = read_u16(frame_raw_u16, four_line_para)
    fpa_tmp_raw = read_u16(frame_raw_u16, four_line_para + 1)
    max_x = read_u16(frame_raw_u16, four_line_para + 2)
    max_y = read_u16(frame_raw_u16, four_line_para + 3)
    max_raw = read_u16(frame_raw_u16, four_line_para + 4)
    min_x = read_u16(frame_raw_u16, four_line_para + 5)
    min_y = read_u16(frame_raw_u16, four_line_para + 6)
    min_raw = read_u16(frame_raw_u16, four_line_para + 7)
    avg_raw = read_u16(frame_raw_u16, four_line_para + 8)

    center_raw = read_u16(frame_raw_u16, four_line_para + 12)
    user_raw00 = read_u16(frame_raw_u16, four_line_para + 13)
    user_raw01 = read_u16(frame_raw_u16, four_line_para + 14)
    user_raw02 = read_u16(frame_raw_u16, four_line_para + 15)

    fpa_tmp_c = 20.0 - (float(fpa_tmp_raw) - fpa_off) / fpa_div

    distance_adjusted = min(float(distance), 20.0) * DISTANCE_MULTIPLIER
    atm = atmt(humidity, air_tmp, distance_adjusted)
    numerator_sub = (1.0 - emiss) * atm * math.pow(refl_tmp + ZEROC, 4) + (1.0 - atm) * math.pow(
        air_tmp + ZEROC, 4
    )
    denominator = emiss * atm

    ts = float_shut_temper + OFFSET_TEMP_SHUTTER
    tfpa = fpa_tmp_c + OFFSET_TEMP_FPA

    cal_a = cal_02 / (cal_01 * 2.0)
    cal_b = cal_02 * cal_02 / (cal_01 * cal_01 * 4.0)
    cal_c = cal_01 * math.pow(ts, 2) + ts * cal_02
    cal_d = cal_03 * math.pow(tfpa, 2) + cal_04 * tfpa + cal_05

    cal_00_corr = 0
    if RANGE == 120:
        cal_00_corr = int(cal_00_offset - tfpa * cal_00_fpamul)

    table_offset = cal_00 - max(cal_00_corr, 0)

    # --- get_temp_table ---
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        n = np.sqrt(
            np.abs(
                ((np.arange(MAX_RAW_VALUE, dtype=np.float32) - table_offset) * cal_d + cal_c)
                / cal_01
                + cal_b
            )
        )
        n[np.isnan(n)] = 0.0

        wtot = np.power(n - cal_a + ZEROC, 4)
        ttot = np.power((wtot - numerator_sub) / denominator, 0.25) - ZEROC
        temperature_table = (
            ttot + (distance_adjusted * 0.85 - 1.125) * (ttot - air_tmp) / 100.0 + correction
        )
        temperature_table = CORRECTION_COEFFICIENT_M * temperature_table + CORRECTION_COEFFICIENT_B
        temperature_table = temperature_table + user_offset

    info = {
        "temp_shutter": float_shut_temper,
        "temp_core": float_core_temper,
        "cameraSoftVersion": camera_soft_version,
        "sn": sn,
        "correction": correction,
        "temp_reflected": refl_tmp,
        "temp_air": air_tmp,
        "humidity": humidity,
        "emissivity": emiss,
        "distance": int(distance),
        "fpa_average": int(fpa_avg),
        "temp_fpa": fpa_tmp_c,
        "temp_max_x": int(max_x),
        "temp_max_y": int(max_y),
        "temp_max_raw": int(max_raw),
        "temp_min_x": int(min_x),
        "temp_min_y": int(min_y),
        "temp_min_raw": int(min_raw),
        "temp_average_raw": int(avg_raw),
        "temp_center_raw": int(center_raw),
        "user_raw": (int(user_raw00), int(user_raw01), int(user_raw02)),
    }
    return info, temperature_table


def legacy_float_command(position: int, value: float) -> list[int]:
    """The original `send_float_command` payload construction."""
    bytes_ = np.array([value], dtype=np.float32).view(np.uint8)
    return [((position + i) << 8) | (0xFF & int(byte)) for i, byte in enumerate(bytes_)]


def legacy_ushort_command(position: int, value: int) -> list[int]:
    bytes_ = np.array([value], dtype=np.uint16).view(np.uint8)
    return [((position + i) << 8) | (0xFF & int(byte)) for i, byte in enumerate(bytes_)]


def legacy_lockin(
    frames: list[np.ndarray],
    times: list[float],
    freq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The demodulation kernel out of v1's `pyplot.get_lockin_frame`.

    Only the kernel: the original interleaved serial writes, wall-clock waits
    and a camera read into the same loop, none of which can be replayed. The
    arithmetic is verbatim -- the weights, the float64 accumulators and the
    division by the frame count -- which is what parity has to mean here.
    """
    in_phase_sum = np.zeros(frames[0].shape)
    quadrature_sum = np.zeros(frames[0].shape)
    total_frames = 0

    for raw_frame, current_time in zip(frames, times, strict=True):
        total_frames += 1

        phase = 2 * math.pi * freq * current_time
        sin_weight = 2 * math.sin(phase)
        cos_weight = -2 * math.cos(phase)

        in_phase_sum += raw_frame * sin_weight
        quadrature_sum += raw_frame * cos_weight

    if total_frames > 0:
        in_phase_sum /= total_frames
        quadrature_sum /= total_frames

    return in_phase_sum, quadrature_sum
