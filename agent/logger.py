"""
Structured Logging Module for Autonomous Research Assistant

Provides:
- Structured logging with levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Multiple handlers (console, file)
- Context support for request_id and topic
- JSON-formatted logs for better parsing
"""

import logging
import logging.handlers
import json
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
topic_var: ContextVar[Optional[str]] = ContextVar('topic', default=None)


class ContextualFormatter(logging.Formatter):
    """Custom formatter that includes request_id and topic in log records"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Add context to record
        record.request_id = request_id_var.get() or "N/A"
        record.topic = topic_var.get() or "N/A"
        
        # Format timestamp
        record.timestamp = datetime.utcnow().isoformat()
        
        # Use structured format
        log_data = {
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
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        return json.dumps(log_data, default=str)


class ConsoleFormatter(logging.Formatter):
    """Console formatter that includes request_id and topic"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Add context to record
        record.request_id = request_id_var.get() or "N/A"
        record.topic = topic_var.get() or "N/A"
        
        # Call parent format
        return super().format(record)


class StructuredLogger:
    """Structured logger with context support"""
    
    _instance: Optional['StructuredLogger'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.logger = logging.getLogger("autonomous_research_assistant")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Console handler with INFO level
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ConsoleFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] [%(topic)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with DEBUG level (structured JSON)
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "research_assistant.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = ContextualFormatter()
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Error file handler (only ERROR and above)
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "research_assistant_errors.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
        
        self._initialized = True
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance"""
        return self.logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
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


def set_logging_context(request_id: Optional[str] = None, topic: Optional[str] = None):
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


def clear_logging_context():
    """Clear logging context"""
    request_id_var.set(None)
    topic_var.set(None)


def get_logging_context() -> Dict[str, Optional[str]]:
    """
    Get current logging context.
    
    Returns:
        Dictionary with request_id and topic
    """
    return {
        "request_id": request_id_var.get(),
        "topic": topic_var.get()
    }


# Initialize logger on import
_logger = StructuredLogger()

