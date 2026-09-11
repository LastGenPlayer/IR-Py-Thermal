"""Application configuration.

Settings come from three places, in increasing order of priority: a TOML file,
environment variables, then anything passed explicitly in code. That ordering is
what makes a container deployment workable -- bake a config file into the image
and override the handful of things that differ per host with env vars.

Environment variables use a double underscore to descend into nested settings,
so `IRTHERMAL__SERVER__PORT=9000` sets `settings.server.port`.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

DEFAULT_CONFIG_PATH = Path("/etc/irthermal/config.toml")

# Usable as a URL path segment and a Prometheus label value.
IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]*$"

Identifier = Annotated[str, Field(pattern=IDENTIFIER_PATTERN, max_length=64)]


class PointConfig(BaseModel):
    """A fixed pixel whose temperature gets measured and exported every frame."""

    model_config = {"extra": "forbid"}

    name: Identifier
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    labels: dict[str, str] = Field(default_factory=dict)
    """Extra Prometheus labels, e.g. `{group = "hiend", slot = "1"}`."""


class CameraConfig(BaseModel):
    """One camera and everything specific to it."""

    model_config = {"extra": "forbid"}

    id: Identifier
    device: str | None = None
    """V4L2 device path, e.g. `/dev/video0`. Omit to auto-detect."""

    replay_file: Path | None = None
    """A captured `.npy` frame to replay instead of opening hardware."""

    auto_detect: bool = False
    """Scan for a supported camera when no device path is given."""

    raw_mode: bool = False
    """Camera streams unprocessed sensor data; do the correction in software."""

    temperature_offset: float = 0.0
    """Constant offset added to every temperature, in Celsius."""

    colormap: str = "inferno"
    emissivity: float | None = Field(default=None, gt=0.0, le=1.0)
    """Pushed to the camera on connect. Omit to keep whatever it has stored."""

    points: list[PointConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_source(self) -> Self:
        sources = [self.device is not None, self.replay_file is not None, self.auto_detect]
        if sum(sources) != 1:
            msg = (
                f"camera {self.id!r}: set exactly one of device, replay_file or "
                f"auto_detect (got {sum(sources)})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_unique_point_names(self) -> Self:
        counts = Counter(point.name for point in self.points)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            msg = f"camera {self.id!r}: duplicate point names {duplicates}"
            raise ValueError(msg)
        return self


class ServerConfig(BaseModel):
    """HTTP server settings."""

    model_config = {"extra": "forbid"}

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    log_format: Literal["json", "console"] = "json"


class Settings(BaseSettings):
    """Root configuration object."""

    model_config = SettingsConfigDict(
        env_prefix="IRTHERMAL__",
        env_nested_delimiter="__",
        toml_file=DEFAULT_CONFIG_PATH,
        extra="forbid",
    )

    server: ServerConfig = Field(default_factory=ServerConfig)
    cameras: list[CameraConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_cameras(self) -> Self:
        if not self.cameras:
            raise ValueError("no cameras configured; define at least one [[cameras]] entry")

        counts = Counter(camera.id for camera in self.cameras)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            msg = f"duplicate camera ids {duplicates}"
            raise ValueError(msg)
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003 - fixed hook signature
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003 - fixed signature
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Highest priority first: explicit kwargs, then env, then the TOML file.

        Dotenv and secrets files are deliberately not consulted: this service
        has no secrets, and two more silent config sources is two too many.
        """
        return (init_settings, env_settings, TomlConfigSettingsSource(settings_cls))


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings, optionally from a TOML file other than the default."""
    if config_path is None:
        return Settings()

    class ScopedSettings(Settings):
        model_config = SettingsConfigDict(
            env_prefix="IRTHERMAL__",
            env_nested_delimiter="__",
            toml_file=config_path,
            extra="forbid",
        )

    return ScopedSettings()
