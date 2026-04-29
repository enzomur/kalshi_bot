"""Structured JSON logging with correlation IDs and secret redaction.

This module provides a structlog-based logging system that:
- Outputs JSON for production (easy parsing by log aggregators)
- Outputs colored text for development
- Adds correlation IDs for request tracing
- Redacts sensitive values (API keys, signatures)
- Logs to both stdout and file
"""

from __future__ import annotations

import logging
import re
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

# Context variable for correlation ID (thread-safe)
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str:
    """Get or create a correlation ID for the current context."""
    cid = correlation_id_var.get()
    if cid is None:
        cid = str(uuid.uuid4())[:8]
        correlation_id_var.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current context."""
    correlation_id_var.set(cid)


def new_correlation_id() -> str:
    """Create and set a new correlation ID."""
    cid = str(uuid.uuid4())[:8]
    correlation_id_var.set(cid)
    return cid


# Patterns for secret redaction
SECRET_PATTERNS = [
    (re.compile(r"(KALSHI[-_]?ACCESS[-_]?KEY[:\s]*)[^\s,}\"']+", re.I), r"\1[REDACTED]"),
    (re.compile(r"(KALSHI[-_]?ACCESS[-_]?SIGNATURE[:\s]*)[^\s,}\"']+", re.I), r"\1[REDACTED]"),
    (re.compile(r"(api[-_]?key[-_]?id[:\s]*)[^\s,}\"']+", re.I), r"\1[REDACTED]"),
    (re.compile(r"(private[-_]?key[:\s]*)[^\s,}\"']+", re.I), r"\1[REDACTED]"),
    (re.compile(r"(signature[:\s]*)[A-Za-z0-9+/=]{20,}", re.I), r"\1[REDACTED]"),
    (re.compile(r"(password[:\s]*)[^\s,}\"']+", re.I), r"\1[REDACTED]"),
    (re.compile(r"(secret[:\s]*)[^\s,}\"']+", re.I), r"\1[REDACTED]"),
    (re.compile(r"(token[:\s]*)[A-Za-z0-9_-]{20,}", re.I), r"\1[REDACTED]"),
]


def redact_secrets(value: Any) -> Any:
    """Recursively redact secrets from a value."""
    if isinstance(value, str):
        for pattern, replacement in SECRET_PATTERNS:
            value = pattern.sub(replacement, value)
        return value
    elif isinstance(value, dict):
        return {k: redact_secrets(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def add_correlation_id(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add correlation ID to log event."""
    cid = correlation_id_var.get()
    if cid:
        event_dict["correlation_id"] = cid
    return event_dict


def add_timestamp(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add ISO timestamp to log event."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def redact_event(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Redact secrets from log event."""
    return redact_secrets(event_dict)


def setup_logging(
    log_level: str = "INFO",
    json_format: bool = False,
    log_file: str | None = None,
    log_file_max_mb: int = 10,
    log_file_backup_count: int = 5,
) -> None:
    """
    Set up structured logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: If True, output JSON; otherwise colored text
        log_file: Optional path to log file
        log_file_max_mb: Maximum log file size in MB before rotation
        log_file_backup_count: Number of backup files to keep
    """
    # Determine if we're in a TTY (interactive terminal)
    is_tty = sys.stdout.isatty()

    # Build processor chain
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        add_timestamp,
        add_correlation_id,
        redact_event,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format or not is_tty:
        # JSON output for production
        shared_processors.append(structlog.processors.format_exc_info)
        renderer = structlog.processors.JSONRenderer()
    else:
        # Colored console output for development
        shared_processors.append(structlog.dev.set_exc_info)
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # Configure structlog
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set up stdlib logging
    log_level_num = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_num)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level_num)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        from logging.handlers import RotatingFileHandler

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=log_file_max_mb * 1024 * 1024,
            backupCount=log_file_backup_count,
        )
        file_handler.setLevel(log_level_num)

        # Always use JSON for file output
        file_formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger with optional initial context.

    Args:
        name: Logger name (usually __name__)
        **initial_context: Initial context to bind to all log messages

    Returns:
        Bound structlog logger

    Example:
        >>> logger = get_logger(__name__, strategy="weather")
        >>> logger.info("Processing signal", ticker="WEATHER-NYC")
        # Output: {"timestamp": "...", "strategy": "weather", "ticker": "WEATHER-NYC", ...}
    """
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger


class LogContext:
    """Context manager for temporary log context.

    Example:
        >>> with LogContext(request_id="abc123"):
        ...     logger.info("Processing request")
        # Output includes request_id="abc123"
    """

    def __init__(self, **context: Any) -> None:
        self.context = context
        self._token: Any = None

    def __enter__(self) -> LogContext:
        self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._token:
            structlog.contextvars.unbind_contextvars(*self.context.keys())


def log_signal(
    logger: structlog.stdlib.BoundLogger,
    signal_id: str,
    strategy: str,
    market: str,
    direction: str,
    edge: float,
    confidence: float,
) -> None:
    """Log a trading signal with standard fields."""
    logger.info(
        "Signal generated",
        signal_id=signal_id[:8],
        strategy=strategy,
        market=market,
        direction=direction,
        edge=f"{edge:.2%}",
        confidence=f"{confidence:.2%}",
    )


def log_decision(
    logger: structlog.stdlib.BoundLogger,
    decision_id: str,
    signal_id: str,
    approved: bool,
    size: int,
    reason: str | None = None,
) -> None:
    """Log a risk decision with standard fields."""
    if approved:
        logger.info(
            "Risk decision: APPROVED",
            decision_id=decision_id[:8],
            signal_id=signal_id[:8],
            approved_size=size,
        )
    else:
        logger.info(
            "Risk decision: REJECTED",
            decision_id=decision_id[:8],
            signal_id=signal_id[:8],
            reason=reason,
        )


def log_execution(
    logger: structlog.stdlib.BoundLogger,
    order_id: str,
    market: str,
    side: str,
    quantity: int,
    price: int,
    success: bool,
    error: str | None = None,
) -> None:
    """Log trade execution with standard fields."""
    if success:
        logger.info(
            "Trade executed",
            order_id=order_id[:12] if len(order_id) > 12 else order_id,
            market=market,
            side=side,
            quantity=quantity,
            price=price,
        )
    else:
        logger.error(
            "Trade execution failed",
            order_id=order_id[:12] if len(order_id) > 12 else order_id,
            market=market,
            side=side,
            quantity=quantity,
            price=price,
            error=error,
        )
