"""
backend/core/logging.py
-----------------------
Structured JSON logging with correlation ID injection.
Every request gets a unique correlation_id attached to all its log lines.
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

from backend.core.config import settings

# Context variable carrying the per-request correlation ID
correlation_id_var: ContextVar[str] = ContextVar(
    "correlation_id", default="no-correlation-id"
)


def _add_correlation_id(
    logger: Any, method: str, event_dict: dict
) -> dict:
    """structlog processor that injects the current correlation_id."""
    event_dict["correlation_id"] = correlation_id_var.get()
    return event_dict


def setup_logging() -> None:
    """Configure structlog for structured JSON (prod) or pretty console (dev)."""

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        _add_correlation_id,
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.log_json:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Silence noisy third-party loggers
    for name in ("uvicorn.access", "sqlalchemy.engine", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """Return a structlog-bound logger for the given module."""
    return structlog.get_logger(name)


def new_correlation_id() -> str:
    """Generate and store a new correlation ID in the current context."""
    cid = str(uuid.uuid4())
    correlation_id_var.set(cid)
    return cid
