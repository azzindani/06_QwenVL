"""Logging infrastructure with JSON and text format support."""

import json
import logging
import sys
from datetime import datetime
from typing import Optional


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """Standard text formatter with colors for terminal output."""

    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        message = f"{timestamp} | {color}{record.levelname:8}{self.RESET} | {record.name} | {record.getMessage()}"

        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)

        return message


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    file_path: Optional[str] = None,
) -> None:
    """
    Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ('json' or 'text')
        file_path: Optional file path for log output
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Select formatter
    if format_type == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if path provided
    if file_path:
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(JSONFormatter())  # Always JSON for files
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for adding extra context to log messages."""

    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        if self.extra:
            extra.update(self.extra)
        kwargs["extra"] = {"extra_fields": extra}
        return msg, kwargs


def get_context_logger(name: str, **context) -> LoggerAdapter:
    """
    Get a logger with additional context fields.

    Args:
        name: Logger name
        **context: Additional context fields to include in all logs

    Returns:
        LoggerAdapter with context
    """
    logger = get_logger(name)
    return LoggerAdapter(logger, context)


if __name__ == "__main__":
    print("=" * 60)
    print("LOGGER TEST")
    print("=" * 60)

    # Test text format
    setup_logging(level="DEBUG", format_type="text")
    logger = get_logger("test")

    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Test JSON format
    print("\nJSON format:")
    setup_logging(level="INFO", format_type="json")
    logger = get_logger("test.json")
    logger.info("JSON log message")

    # Test context logger
    ctx_logger = get_context_logger("test.context", request_id="123", user="test")
    ctx_logger.info("Message with context")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
