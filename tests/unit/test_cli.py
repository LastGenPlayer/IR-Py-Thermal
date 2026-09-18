"""The `openapi` command, which an API client imports.

Generating the schema must not need a config, a camera or a bound port -- the
routes are static and the camera id is only ever a path parameter.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from irthermal.cli import app

runner = CliRunner()

EXPECTED_PATHS = {
    "/api/health",
    "/api/cameras",
    "/api/cameras/{camera_id}/snapshot.{fmt}",
    "/api/cameras/{camera_id}/temperatures",
    "/api/cameras/{camera_id}/calibrate",
    "/api/jobs/{job_id}/result.npz",
}


def _generate(tmp_path: Path, *extra: str) -> dict:
    destination = tmp_path / "openapi.json"
    result = runner.invoke(app, ["openapi", "-o", str(destination), *extra])
    assert result.exit_code == 0, result.output
    return json.loads(destination.read_text(encoding="utf-8"))


def test_schema_generates_without_a_config(tmp_path: Path) -> None:
    schema = _generate(tmp_path)

    assert schema["openapi"].startswith("3.")
    assert schema["info"]["title"] == "IRThermal"
    assert set(schema["paths"]) >= EXPECTED_PATHS


def test_schema_records_a_base_url(tmp_path: Path) -> None:
    """Without one an imported collection has nowhere to send its requests."""
    schema = _generate(tmp_path)

    assert schema["servers"] == [{"url": "http://localhost:8000"}]


def test_base_url_can_be_overridden(tmp_path: Path) -> None:
    schema = _generate(tmp_path, "--server-url", "http://pi-rack.local:8000")

    assert schema["servers"] == [{"url": "http://pi-rack.local:8000"}]


def test_bind_address_becomes_a_reachable_host(tmp_path: Path) -> None:
    """0.0.0.0 is a bind address, not somewhere a client can send a request."""
    config = tmp_path / "config.toml"
    config.write_text(
        '[server]\nhost = "0.0.0.0"\nport = 9000\n\n[[cameras]]\nid = "one"\nauto_detect = true\n',
        encoding="utf-8",
    )

    destination = tmp_path / "out.json"
    result = runner.invoke(app, ["openapi", "-c", str(config), "-o", str(destination)])
    assert result.exit_code == 0, result.output

    schema = json.loads(destination.read_text(encoding="utf-8"))
    assert schema["servers"] == [{"url": "http://localhost:9000"}]


def test_hidden_routes_stay_out_of_the_schema(tmp_path: Path) -> None:
    """`/metrics` and `/health` are deliberately undocumented; the WS cannot be."""
    schema = _generate(tmp_path)

    assert "/metrics" not in schema["paths"]
    assert "/api/cameras/{camera_id}/telemetry" not in schema["paths"]


def test_an_invalid_config_is_reported_not_raised(tmp_path: Path) -> None:
    config = tmp_path / "bad.toml"
    config.write_text('[server]\nport = "not-a-port"\n', encoding="utf-8")

    result = runner.invoke(app, ["openapi", "-c", str(config), "-o", str(tmp_path / "x.json")])

    assert result.exit_code == 1
    assert "Config is invalid" in result.output
