"""
Structured Logging Module for Autonomous Research Assistant

Provides:
- Structured logging with levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Multiple handlers (console, file)
- Context support for request_id and topic
- JSON-formatted logs for better parsing
"""

import json
import logging
import logging.handlers
import sys
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any

# Context variables for request tracking
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
topic_var: ContextVar[str | None] = ContextVar("topic", default=None)


class BaseContextFormatter(logging.Formatter):
    """Base formatter that injects context (request_id and topic) into log records"""

    def _inject_context(self, record: logging.LogRecord) -> None:
        """Inject context variables into the log record"""
        record.request_id = request_id_var.get() or "N/A"
        record.topic = topic_var.get() or "N/A"


class ContextualFormatter(BaseContextFormatter):
    """JSON formatter that includes request_id and topic in log records"""

    def format(self, record: logging.LogRecord) -> str:
        self._inject_context(record)
        record.timestamp = datetime.utcnow().isoformat()

        log_data: dict[str, Any] = {
            "timestamp": record.timestamp,
            "level": record.levelname,
            "logger": record.name,
            "request_id": record.request_id,
            "topic": record.topic,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data, default=str)


class ConsoleFormatter(BaseContextFormatter):
    """Console formatter that includes request_id and topic"""

    def format(self, record: logging.LogRecord) -> str:
        self._inject_context(record)
        return super().format(record)


class StructuredLogger:
    """Structured logger with context support"""

    _instance: "StructuredLogger | None" = None

    def __new__(cls) -> "StructuredLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._setup_logger()
        return cls._instance

    def _setup_logger(self) -> None:
        """Configure the logger with handlers"""
        self.logger: logging.Logger = logging.getLogger("autonomous_research_assistant")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.logger.handlers.clear()

        # Console handler with INFO level
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            ConsoleFormatter(
                "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] [%(topic)s] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self.logger.addHandler(console_handler)

        # File handler with DEBUG level (structured JSON)
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        file_formatter = ContextualFormatter()
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "research_assistant.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Error file handler (only ERROR and above)
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "research_assistant_errors.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)

    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance"""
        return self.logger


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger instance with context support.

    Args:
        name: Optional logger name (defaults to main logger)

    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"autonomous_research_assistant.{name}")
    return StructuredLogger().get_logger()


def set_logging_context(request_id: str | None = None, topic: str | None = None) -> None:
    """
    Set logging context (request_id and topic) for current execution.

    Args:
        request_id: Unique identifier for the request
        topic: Research topic
    """
    if request_id:
        request_id_var.set(request_id)
    if topic:
        topic_var.set(topic)


def clear_logging_context() -> None:
    """Clear logging context"""
    request_id_var.set(None)
    topic_var.set(None)


def get_logging_context() -> dict[str, str | None]:
    """
    Get current logging context.

    Returns:
        Dictionary with request_id and topic
    """
    return {"request_id": request_id_var.get(), "topic": topic_var.get()}


# Initialize logger on import
_logger = StructuredLogger()
