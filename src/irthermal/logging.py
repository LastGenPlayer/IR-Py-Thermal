"""Structured logging setup.

JSON by default, because the usual home for this service is a container whose
stdout is being scraped. `console` renders colourised key-value lines instead,
which is what you want when running it by hand.

Everything goes through stdlib logging, structlog's own calls included. That is
what lets uvicorn's loggers come out in the same shape as ours rather than in a
second, competing format -- and it is why `add_logger_name` works, since the
records carry a real logger behind them.
"""

from __future__ import annotations

import logging
import sys
from typing import Literal

import structlog

LogFormat = Literal["json", "console"]

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def configure_logging(level: str = "info", log_format: LogFormat = "json") -> None:
    """Route structlog and stdlib logging through one renderer on stdout."""
    numeric_level = _LEVELS.get(level.lower(), logging.INFO)

    # Applied to every record, whether it came from structlog or from stdlib.
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if log_format == "json"
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[
            *shared_processors,
            # Hand off to the stdlib handler, which does the actual rendering.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.format_exc_info,
                renderer,
            ],
        )
    )

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(numeric_level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """A bound logger for `name`."""
    return structlog.stdlib.get_logger(name)
