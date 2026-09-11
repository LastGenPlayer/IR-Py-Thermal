"""Command line entry point.

`serve` arrives with the HTTP layer. For now this covers the parts that exist:
inspecting the build and validating a config file before it reaches a host where
a bad config means a crash loop.
"""

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


@app.command("check-config")
def check_config(config: ConfigOption = None) -> None:
    """Validate a config file and summarise what it declares."""
    try:
        settings = load_settings(config)
    except ValidationError as exc:
        typer.secho("Config is invalid:", fg=typer.colors.RED, err=True)
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"]) or "(root)"
            typer.echo(f"  {location}: {error['msg']}", err=True)
        raise typer.Exit(code=1) from None

    typer.secho("Config is valid.", fg=typer.colors.GREEN)
    typer.echo(f"  listen: {settings.server.host}:{settings.server.port}")
    for camera in settings.cameras:
        source = camera.device or camera.replay_file or "auto-detect"
        typer.echo(
            f"  camera {camera.id}: source={source} raw={camera.raw_mode} "
            f"points={len(camera.points)}"
        )


def main() -> None:
    """Console script entry point."""
    app()


if __name__ == "__main__":
    main()
