"""Structured logging setup.

JSON by default, because the usual home for this service is a container whose
stdout is being scraped. `console` renders colourised key-value lines instead,
which is what you want when running it by hand.
"""

from __future__ import annotations

import logging
import sys
from typing import Literal

import structlog

LogFormat = Literal["json", "console"]


def configure_logging(level: str = "info", log_format: LogFormat = "json") -> None:
    """Route stdlib logging and structlog through one renderer."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if log_format == "json"
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Make uvicorn's stdlib loggers land in the same stream, same shape.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=shared_processors,
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(numeric_level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """A bound logger for `name`."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
