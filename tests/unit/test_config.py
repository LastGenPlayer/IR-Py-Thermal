"""Config loading, precedence and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from irthermal.config import load_settings

MINIMAL = """
[[cameras]]
id = "cam-a"
device = "/dev/video0"
"""


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_loads_minimal_config(tmp_path: Path) -> None:
    settings = load_settings(write_config(tmp_path, MINIMAL))

    assert settings.server.port == 8000
    assert len(settings.cameras) == 1
    assert settings.cameras[0].id == "cam-a"
    assert settings.cameras[0].raw_mode is False


def test_shipped_example_config_is_valid() -> None:
    """The documented example must actually load, or it is worse than nothing."""
    example = Path(__file__).parents[2] / "config.example.toml"
    settings = load_settings(example)

    assert {c.id for c in settings.cameras} == {"pi-rack", "bench"}
    rack = next(c for c in settings.cameras if c.id == "pi-rack")
    assert rack.raw_mode is True
    assert rack.temperature_offset == pytest.approx(8.5)
    assert rack.points[0].labels == {"group": "hiend", "slot": "1"}


def test_environment_overrides_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRTHERMAL__SERVER__PORT", "9100")
    settings = load_settings(write_config(tmp_path, MINIMAL))

    assert settings.server.port == 9100


def test_rejects_config_with_no_cameras(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="no cameras configured"):
        load_settings(write_config(tmp_path, "[server]\nport = 8000\n"))


def test_rejects_duplicate_camera_ids(tmp_path: Path) -> None:
    body = MINIMAL + '\n[[cameras]]\nid = "cam-a"\ndevice = "/dev/video1"\n'
    with pytest.raises(ValidationError, match="duplicate camera ids"):
        load_settings(write_config(tmp_path, body))


def test_rejects_duplicate_point_names(tmp_path: Path) -> None:
    body = (
        MINIMAL
        + """
[[cameras.points]]
name = "p1"
x = 1
y = 1

[[cameras.points]]
name = "p1"
x = 2
y = 2
"""
    )
    with pytest.raises(ValidationError, match="duplicate point names"):
        load_settings(write_config(tmp_path, body))


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ('[[cameras]]\nid = "a"\n', "exactly one of device"),
        ('[[cameras]]\nid = "a"\ndevice = "/dev/video0"\nauto_detect = true\n', "exactly one"),
        ('[[cameras]]\nid = "has space"\ndevice = "/dev/video0"\n', "String should match"),
        ('[[cameras]]\nid = "a"\ndevice = "/dev/video0"\nnope = 1\n', "Extra inputs"),
    ],
)
def test_rejects_malformed_camera(tmp_path: Path, body: str, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        load_settings(write_config(tmp_path, body))


def test_rejects_negative_point_coordinates(tmp_path: Path) -> None:
    body = MINIMAL + '\n[[cameras.points]]\nname = "p1"\nx = -1\ny = 0\n'
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        load_settings(write_config(tmp_path, body))
