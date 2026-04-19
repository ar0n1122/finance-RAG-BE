"""
Structured logging configuration via *structlog*.

Produces JSON logs in production and coloured console logs during development.
Automatically binds ``request_id`` and ``user_id`` when available.  PII fields
are redacted before serialisation.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

# Fields whose values are replaced with ``***`` before serialisation.
_PII_FIELDS = frozenset({
    "password", "token", "api_key", "secret", "authorization",
    "x-api-key", "cookie", "credit_card", "ssn", "phone",
})


def _redact_pii(
    _logger: Any,
    _method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Structlog processor that masks PII fields."""
    for key in list(event_dict.keys()):
        if key.lower() in _PII_FIELDS:
            event_dict[key] = "***"
    return event_dict


def _drop_color_message(
    _logger: Any,
    _method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Drop uvicorn's ``color_message`` key to keep logs clean."""
    event_dict.pop("color_message", None)
    return event_dict


def setup_logging(*, log_level: str = "INFO", json_logs: bool = True) -> None:
    """
    Configure *structlog* and the standard-library root logger.

    Parameters
    ----------
    log_level:
        Root log level (DEBUG, INFO, WARNING, …).
    json_logs:
        ``True`` → JSON renderer (production).
        ``False`` → coloured console renderer (development).
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.ExtraAdder(),
        _redact_pii,
        _drop_color_message,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_logs:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Align stdlib logging so uvicorn/httpx messages go through structlog.
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet noisy third-party loggers.
    for name in ("uvicorn.access", "httpx", "httpcore", "google", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a *structlog* bound logger, optionally with a name."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name or "app")
    return logger
