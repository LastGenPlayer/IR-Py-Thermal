"""Driving the load whose heat a lock-in run demodulates.

The wire protocol is v1's, unchanged: one ASCII digit and a newline, `1` to
energise the load and `0` to release it. Firmware written for v1 keeps working.

Switching is not always ours to do. A laser on its own function generator, or a
board already being driven by the thing under test, is modulated perfectly well
without this service touching it -- hence `ExternalModulator`, which commands
nothing and lets the demodulation run against a drive it does not control.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

import serial

WRITE_TIMEOUT_S = 1.0

_ON = b"1\n"
_OFF = b"0\n"


class ModulatorError(RuntimeError):
    """The load switch could not be opened or driven."""


@runtime_checkable
class Modulator(Protocol):
    """Something that can switch a load on and off."""

    @property
    def description(self) -> str:
        """How this modulator is reported in job status."""
        ...

    def set(self, *, on: bool) -> None:
        """Energise or release the load. Blocking."""
        ...

    def close(self) -> None:
        """Release the device. Safe to call more than once."""
        ...


class SerialModulator:
    """A load switch on the far end of a serial line."""

    def __init__(self, port: serial.Serial, name: str, *, invert: bool = False) -> None:
        self._port = port
        self._name = name
        self._invert = invert

    @classmethod
    def open(cls, device: str, baudrate: int, *, invert: bool = False) -> SerialModulator:
        """Open a serial port, or explain why it could not be opened.

        `baudrate` has no default here on purpose: the configuration carries
        one, and a second default in the code is a second thing to disagree.
        """
        try:
            port = serial.Serial(device, baudrate, write_timeout=WRITE_TIMEOUT_S)
        except (serial.SerialException, ValueError, OSError) as exc:
            raise ModulatorError(f"could not open modulator on {device}: {exc}") from exc

        return cls(port, device, invert=invert)

    @property
    def description(self) -> str:
        return f"serial:{self._name}"

    def set(self, *, on: bool) -> None:
        energised = not on if self._invert else on
        try:
            self._port.write(_ON if energised else _OFF)
        except (serial.SerialException, OSError) as exc:
            raise ModulatorError(f"could not drive modulator on {self._name}: {exc}") from exc

    def close(self) -> None:
        self._port.close()


class ExternalModulator:
    """Commands nothing; the load is driven by something else.

    Records what it would have done, which is what the emulated camera does with
    control commands and for the same reason: it keeps the tests honest about
    the schedule without needing a device.
    """

    def __init__(self) -> None:
        self.states: list[tuple[float, bool]] = []
        """Every requested state, as (monotonic time, on)."""

    @property
    def description(self) -> str:
        return "external"

    def set(self, *, on: bool) -> None:
        self.states.append((time.monotonic(), on))

    def close(self) -> None:
        pass
