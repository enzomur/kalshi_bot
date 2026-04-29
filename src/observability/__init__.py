"""Structured logging and observability."""

from src.observability.logging import (
    setup_logging,
    get_logger,
    get_correlation_id,
    set_correlation_id,
    new_correlation_id,
    LogContext,
    log_signal,
    log_decision,
    log_execution,
)

from src.observability.dashboard import (
    CLIDashboard,
    DashboardManager,
    print_startup_banner,
    print_shutdown_summary,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    "get_correlation_id",
    "set_correlation_id",
    "new_correlation_id",
    "LogContext",
    "log_signal",
    "log_decision",
    "log_execution",
    # Dashboard
    "CLIDashboard",
    "DashboardManager",
    "print_startup_banner",
    "print_shutdown_summary",
]
