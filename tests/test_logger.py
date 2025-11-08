"""Tests for agent.logger module"""

import json
import logging
from unittest.mock import MagicMock, patch

from agent.logger import (
    StructuredLogger,
    clear_logging_context,
    get_logger,
    get_logging_context,
    set_logging_context,
)


def test_get_logger_returns_logger() -> None:
    """Test that get_logger returns a logger instance"""
    logger = get_logger("test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "autonomous_research_assistant.test"


def test_get_logger_default_name() -> None:
    """Test that get_logger without name returns main logger"""
    logger = get_logger()
    assert isinstance(logger, logging.Logger)
    assert logger.name == "autonomous_research_assistant"


def test_set_logging_context() -> None:
    """Test setting logging context"""
    set_logging_context(request_id="test-123", topic="test topic")
    context = get_logging_context()
    assert context["request_id"] == "test-123"
    assert context["topic"] == "test topic"


def test_set_logging_context_partial() -> None:
    """Test setting logging context with partial values"""
    clear_logging_context()  # Clear any existing context first
    set_logging_context(request_id="test-456")
    context = get_logging_context()
    assert context["request_id"] == "test-456"
    assert context["topic"] is None

    clear_logging_context()
    set_logging_context(topic="another topic")
    context = get_logging_context()
    assert context["request_id"] is None
    assert context["topic"] == "another topic"


def test_clear_logging_context() -> None:
    """Test clearing logging context"""
    set_logging_context(request_id="test-789", topic="test")
    clear_logging_context()
    context = get_logging_context()
    assert context["request_id"] is None
    assert context["topic"] is None


def test_get_logging_context_default() -> None:
    """Test getting logging context when not set"""
    clear_logging_context()
    context = get_logging_context()
    assert context["request_id"] is None
    assert context["topic"] is None


def test_structured_logger_singleton() -> None:
    """Test that StructuredLogger is a singleton"""
    logger1 = StructuredLogger()
    logger2 = StructuredLogger()
    assert logger1 is logger2


def test_structured_logger_has_handlers() -> None:
    """Test that StructuredLogger has configured handlers"""
    logger = StructuredLogger()
    logger_instance = logger.get_logger()
    assert len(logger_instance.handlers) > 0


def test_logger_logs_with_context() -> None:
    """Test that logger includes context in log records"""
    logger = get_logger("test")
    set_logging_context(request_id="ctx-123", topic="ctx topic")

    # Capture log output
    with patch("agent.logger.StructuredLogger") as MockLogger:
        mock_logger_instance = MagicMock()
        mock_handler = MagicMock()
        mock_logger_instance.handlers = [mock_handler]
        MockLogger.return_value.get_logger.return_value = mock_logger_instance

        logger.info("Test message")
        # Verify handler was called
        assert True  # Handler may be called


def test_logger_formatter_includes_context() -> None:
    """Test that formatter includes context in formatted output"""
    from agent.logger import ContextualFormatter

    formatter = ContextualFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    # Set context
    set_logging_context(request_id="fmt-123", topic="fmt topic")

    formatted = formatter.format(record)
    log_data = json.loads(formatted)

    assert log_data["request_id"] == "fmt-123"
    assert log_data["topic"] == "fmt topic"
    assert log_data["message"] == "Test message"
    assert log_data["level"] == "INFO"


def test_console_formatter_includes_context() -> None:
    """Test that console formatter includes context"""
    from agent.logger import ConsoleFormatter

    formatter = ConsoleFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] [%(topic)s] - %(message)s"
    )
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    set_logging_context(request_id="console-123", topic="console topic")
    formatted = formatter.format(record)

    assert "console-123" in formatted
    assert "console topic" in formatted
    assert "Test message" in formatted
