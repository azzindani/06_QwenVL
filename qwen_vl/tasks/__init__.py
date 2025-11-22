"""Task handlers for document processing."""

from .base import (
    BaseTaskHandler,
    TaskResult,
    TaskType,
    get_handler,
    list_handlers,
    register_handler,
)

# Import handlers to register them
from .layout import LayoutHandler
from .ocr import OCRHandler

__all__ = [
    "BaseTaskHandler",
    "TaskResult",
    "TaskType",
    "get_handler",
    "list_handlers",
    "register_handler",
    "OCRHandler",
    "LayoutHandler",
]
