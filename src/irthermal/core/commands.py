"""Control opcodes for the camera.

These cameras have no separate control endpoint: commands are smuggled through
the UVC zoom control. A write of `CAP_PROP_ZOOM` with a 16-bit payload either
selects a canned action (the `0x80xx` range) or writes one byte into the
camera's parameter block.

Everything here is pure: the functions return the payloads to write, they do not
touch a device. That keeps the wire format testable without hardware.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# --- parameter block addresses (byte offsets, 4 bytes per parameter) ---------

PARAM_CORRECTION = 0 * 4
PARAM_REFLECTION = 1 * 4
PARAM_AMBIENT = 2 * 4
PARAM_HUMIDITY = 3 * 4
PARAM_EMISSIVITY = 4 * 4
PARAM_DISTANCE = 5 * 4

# --- canned actions ---------------------------------------------------------

ACTION_SHUTTER_CALIBRATE = 0x8000
"""Trigger a shutter (flat-field) calibration. This is the audible click."""

ACTION_STREAM_CONFIGURE = 0x8004
"""Put the camera into the raw 16-bit streaming mode this driver expects."""

ACTION_RANGE_NORMAL = 0x8020
"""Select the normal (-20..120 C) temperature range."""

ACTION_RANGE_HIGH = 0x8021
"""Select the extended (up to ~400 C) temperature range."""

ACTION_SAVE_PARAMETERS = 0x80FF
"""Persist the current parameter block to camera flash."""

# --- on-sensor measurement points -------------------------------------------

_POINT_OPCODES: tuple[tuple[int, int], ...] = (
    (0xF000, 0xF200),
    (0xF400, 0xF600),
    (0xF800, 0xFA00),
)
"""(x opcode, y opcode) base values for the three hardware measurement points."""

MAX_POINT_INDEX = len(_POINT_OPCODES) - 1


def _payloads(position: int, raw_bytes: npt.NDArray[np.uint8]) -> list[int]:
    """Pack consecutive bytes into one zoom payload each."""
    return [((position + index) << 8) | (0xFF & int(byte)) for index, byte in enumerate(raw_bytes)]


def float_command(position: int, value: float) -> list[int]:
    """Payloads that write a little-endian float32 at `position`."""
    return _payloads(position, np.array([value], dtype=np.float32).view(np.uint8))


def ushort_command(position: int, value: int) -> list[int]:
    """Payloads that write a little-endian uint16 at `position`."""
    return _payloads(position, np.array([value], dtype=np.uint16).view(np.uint8))


def byte_command(position: int, value: int) -> list[int]:
    """Payload that writes a single byte at `position`."""
    return _payloads(position, np.array([value], dtype=np.uint8))


def point_command(index: int, x: int, y: int) -> list[int]:
    """Payloads that move on-sensor measurement point `index` to (x, y)."""
    try:
        x_opcode, y_opcode = _POINT_OPCODES[index]
    except IndexError:
        msg = f"Measurement point index must be 0..{MAX_POINT_INDEX}, got {index}"
        raise ValueError(msg) from None
    return [x_opcode + x, y_opcode + y]
