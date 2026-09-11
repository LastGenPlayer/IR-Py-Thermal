"""Logging configuration.

These exist because the original setup crashed on the first log line the server
ever emitted -- `add_logger_name` was paired with a factory that produces
loggers having no name. Nothing in the suite called `configure_logging`, so
nothing caught it until the server was actually started.
"""

from __future__ import annotations

import json
import logging

import pytest

from irthermal.logging import configure_logging, get_logger


@pytest.fixture(autouse=True)
def _restore_logging():
    """Put the root logger back however the test left it."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers, root.level = handlers, level


def test_json_output_is_valid_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("info", "json")
    get_logger("irthermal.test").info("camera.connected", camera="rack", frames=3)

    record = json.loads(capsys.readouterr().out.strip())

    assert record["event"] == "camera.connected"
    assert record["camera"] == "rack"
    assert record["frames"] == 3
    assert record["level"] == "info"
    assert record["logger"] == "irthermal.test"
    assert "timestamp" in record


def test_console_output_is_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("info", "console")
    get_logger("irthermal.test").warning("camera.lost", camera="bench")

    out = capsys.readouterr().out

    assert "camera.lost" in out
    assert "bench" in out


def test_level_filtering_suppresses_debug(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("warning", "json")
    log = get_logger("irthermal.test")
    log.info("quiet")
    log.warning("loud")

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]

    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "loud"


def test_stdlib_loggers_use_the_same_format(capsys: pytest.CaptureFixture[str]) -> None:
    """uvicorn logs through stdlib; it must not come out in a second format."""
    configure_logging("info", "json")
    logging.getLogger("uvicorn.error").info("Application startup complete.")

    record = json.loads(capsys.readouterr().out.strip())

    assert record["event"] == "Application startup complete."
    assert record["logger"] == "uvicorn.error"


def test_exceptions_are_rendered(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("info", "json")
    try:
        raise ValueError("camera exploded")
    except ValueError:
        get_logger("irthermal.test").exception("camera.failed")

    record = json.loads(capsys.readouterr().out.strip())

    assert record["event"] == "camera.failed"
    assert "camera exploded" in record["exception"]
