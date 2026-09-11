"""Request and response bodies.

Temperatures are `float | None`. A pixel outside the calibration model's valid
range has no temperature, and NaN is not representable in JSON -- so it is
reported as null rather than as a number that looks real.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from irthermal.camera.runtime import CameraStatus
from irthermal.core.frame import Frame, Measurement, RegionStats
from irthermal.core.radiometry import RadiometryAdjustments
from irthermal.core.telemetry import CameraTelemetry


def celsius(value: float) -> float | None:
    """Present a temperature, or null where the model could not produce one."""
    return None if math.isnan(value) else value


class MeasurementOut(BaseModel):
    """A temperature reading at a pixel."""

    x: int
    y: int
    celsius: float | None

    @classmethod
    def of(cls, measurement: Measurement) -> Self:
        return cls(x=measurement.x, y=measurement.y, celsius=celsius(measurement.celsius))


class RegionStatsOut(BaseModel):
    """Temperature summary over an area."""

    minimum: MeasurementOut
    maximum: MeasurementOut
    mean_celsius: float | None

    @classmethod
    def of(cls, stats: RegionStats) -> Self:
        return cls(
            minimum=MeasurementOut.of(stats.minimum),
            maximum=MeasurementOut.of(stats.maximum),
            mean_celsius=celsius(stats.mean_celsius),
        )


class TelemetryOut(BaseModel):
    """What the camera reports about itself."""

    shutter_temp_c: float
    core_temp_c: float
    fpa_temp_c: float
    emissivity: float
    humidity: float
    air_temp_c: float
    reflected_temp_c: float
    distance_m: int
    correction: float
    firmware: str
    serial: str
    shutter_trim_applied: bool

    @classmethod
    def of(cls, telemetry: CameraTelemetry) -> Self:
        return cls(
            shutter_temp_c=telemetry.shutter_temp_c,
            core_temp_c=telemetry.core_temp_c,
            fpa_temp_c=telemetry.fpa_temp_c,
            emissivity=telemetry.emissivity,
            humidity=telemetry.humidity,
            air_temp_c=telemetry.air_temp_c,
            reflected_temp_c=telemetry.reflected_temp_c,
            distance_m=telemetry.distance_m,
            correction=telemetry.correction,
            firmware=telemetry.firmware,
            serial=telemetry.serial,
            shutter_trim_applied=telemetry.shutter_trim_applied,
        )


class AdjustmentsOut(BaseModel):
    """Operator corrections currently in effect."""

    range_mode: int
    user_offset: float
    gain_m: float
    gain_b: float

    @classmethod
    def of(cls, adjustments: RadiometryAdjustments) -> Self:
        return cls(
            range_mode=adjustments.range_mode,
            user_offset=adjustments.user_offset,
            gain_m=adjustments.gain_m,
            gain_b=adjustments.gain_b,
        )


class CameraStatusOut(BaseModel):
    """Capture health for one camera."""

    id: str
    connected: bool
    frames: int
    fps: float
    read_errors: int
    reconnects: int
    last_error: str | None
    source: str | None
    width: int | None
    height: int | None
    subscribers: int

    @classmethod
    def of(cls, status: CameraStatus) -> Self:
        # CameraStatus uses slots, so it has no __dict__ for vars() to read.
        return cls(**asdict(status))


class CameraDetailOut(CameraStatusOut):
    """Status plus whatever the most recent frame told us."""

    sequence: int | None = None
    corrected: bool | None = None
    telemetry: TelemetryOut | None = None
    adjustments: AdjustmentsOut | None = None


class TemperaturesOut(BaseModel):
    """Every configured point on one frame, plus frame-wide extremes."""

    camera: str
    sequence: int
    points: dict[str, MeasurementOut]
    minimum: MeasurementOut
    maximum: MeasurementOut
    mean_celsius: float | None
    center: MeasurementOut

    @classmethod
    def of(cls, frame: Frame, points: dict[str, Measurement]) -> Self:
        stats = frame.stats()
        return cls(
            camera=frame.camera_id,
            sequence=frame.sequence,
            points={name: MeasurementOut.of(m) for name, m in points.items()},
            minimum=MeasurementOut.of(stats.minimum),
            maximum=MeasurementOut.of(stats.maximum),
            mean_celsius=celsius(stats.mean_celsius),
            center=MeasurementOut.of(frame.center()),
        )


class RegionRequest(BaseModel):
    """A rectangle to summarise. `x2`/`y2` are exclusive."""

    x1: int
    y1: int
    x2: int
    y2: int

    @model_validator(mode="after")
    def _check_area(self) -> Self:
        if self.x1 == self.x2 or self.y1 == self.y2:
            raise ValueError("region has zero area")
        return self


class RangeRequest(BaseModel):
    """Which temperature range the camera should use."""

    mode: Literal["normal", "high"]


class ParametersRequest(BaseModel):
    """Measurement parameters to write to the camera. Omitted fields are left alone."""

    emissivity: float | None = Field(default=None, gt=0.0, le=1.0)
    distance_m: int | None = Field(default=None, ge=0, le=65535)
    humidity: float | None = Field(default=None, ge=0.0, le=1.0)
    air_temp_c: float | None = None
    reflected_temp_c: float | None = None
    correction: float | None = None
    temperature_offset: float | None = None
    """Applied in software rather than sent to the camera."""

    def device_parameters(self) -> dict[str, float]:
        """The subset that gets written to the camera itself."""
        fields = (
            "emissivity",
            "distance_m",
            "humidity",
            "air_temp_c",
            "reflected_temp_c",
            "correction",
        )
        return {name: getattr(self, name) for name in fields if getattr(self, name) is not None}

    @model_validator(mode="after")
    def _check_not_empty(self) -> Self:
        if not self.device_parameters() and self.temperature_offset is None:
            raise ValueError("no parameters given")
        return self


class HealthOut(BaseModel):
    """Liveness and readiness."""

    status: Literal["ok", "degraded"]
    cameras: int
    connected: int
