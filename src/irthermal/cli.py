"""Command line entry point."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from irthermal.config import DEFAULT_CONFIG_PATH, load_settings

app = typer.Typer(
    name="irthermal",
    help="Thermal camera server.",
    no_args_is_help=True,
    add_completion=False,
)

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", "-c", help=f"Config file (default: {DEFAULT_CONFIG_PATH})."),
]


@app.command("version")
def show_version() -> None:
    """Print the installed version."""
    try:
        typer.echo(version("irthermal"))
    except PackageNotFoundError:
        typer.echo("unknown (not installed as a package)")


@app.command()
def serve(
    config: ConfigOption = None,
    host: Annotated[str | None, typer.Option(help="Override the configured bind address.")] = None,
    port: Annotated[int | None, typer.Option(help="Override the configured port.")] = None,
    reload: Annotated[bool, typer.Option(help="Reload on source changes (development).")] = False,
) -> None:
    """Run the HTTP server."""
    from irthermal import server
    from irthermal.logging import configure_logging

    if reload:
        # uvicorn's reloader needs the app as an import string, and would run a
        # second copy of it besides -- two processes contending for one camera.
        typer.secho(
            "--reload is not supported: the server owns its camera devices, and "
            "a reloader would start a second copy competing for them.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        settings = load_settings(config)
    except ValidationError as exc:
        _report_invalid(exc)
        raise typer.Exit(code=1) from None

    configure_logging(settings.server.log_level, settings.server.log_format)

    # One process only: several workers would each open the same devices.
    server.run(settings, host=host, port=port)


@app.command()
def openapi(
    config: ConfigOption = None,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Where to write the schema.")
    ] = Path("openapi.json"),
    server_url: Annotated[
        str | None,
        typer.Option("--server-url", help="Base URL to record. Defaults to the configured one."),
    ] = None,
) -> None:
    """Write the OpenAPI schema, for an API client such as Bruno to import.

    No camera is opened and no port is bound: the routes are static, and the
    camera id is a path parameter rather than anything the schema bakes in. A
    config is therefore optional, and a placeholder stands in without it.

    The per-frame telemetry WebSocket is absent because OpenAPI cannot describe
    one, and `/metrics` and `/health` because they are deliberately kept out of
    the schema.
    """
    import json

    from irthermal.api import create_app
    from irthermal.config import CameraConfig, Settings

    if config is not None:
        try:
            settings = load_settings(config)
        except ValidationError as exc:
            _report_invalid(exc)
            raise typer.Exit(code=1) from None
    else:
        settings = Settings(
            cameras=[CameraConfig(id="placeholder", replay_file=Path("unused.npy"))]
        )

    schema = create_app(settings).openapi()

    # FastAPI records no `servers` entry, which would leave an imported
    # collection holding relative paths and no host to send them to.
    bind = settings.server.host
    host = "localhost" if bind in {"0.0.0.0", "::", ""} else bind
    base = server_url or f"http://{host}:{settings.server.port}"
    schema["servers"] = [{"url": base}]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")

    paths = schema.get("paths", {})
    operations = sum(len(methods) for methods in paths.values())
    typer.secho(f"Wrote {output}", fg=typer.colors.GREEN)
    typer.echo(f"  openapi {schema.get('openapi')}: {len(paths)} paths, {operations} operations")
    typer.echo(f"  server: {base}")


@app.command("check-config")
def check_config(config: ConfigOption = None) -> None:
    """Validate a config file and summarise what it declares."""
    try:
        settings = load_settings(config)
    except ValidationError as exc:
        _report_invalid(exc)
        raise typer.Exit(code=1) from None

    typer.secho("Config is valid.", fg=typer.colors.GREEN)
    typer.echo(f"  listen: {settings.server.host}:{settings.server.port}")
    for camera in settings.cameras:
        source = camera.device or camera.replay_file or "auto-detect"
        # A replay reads a file; naming a transport there would only mislead.
        transport = "" if camera.replay_file else f" transport={camera.transport}"
        typer.echo(
            f"  camera {camera.id}: source={source}{transport} raw={camera.raw_mode} "
            f"points={len(camera.points)}"
        )


def _report_invalid(exc: ValidationError) -> None:
    """Print validation failures in a form an operator can act on."""
    typer.secho("Config is invalid:", fg=typer.colors.RED, err=True)
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        typer.echo(f"  {location}: {error['msg']}", err=True)


def main() -> None:
    """Console script entry point."""
    app()


if __name__ == "__main__":
    main()
