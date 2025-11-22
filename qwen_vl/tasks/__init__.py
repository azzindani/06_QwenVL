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
from .field_extraction import PRESET_SCHEMAS, FieldExtractionHandler
from .layout import LayoutHandler
from .ner import ENTITY_TYPES, NERHandler
from .ocr import OCRHandler
from .table import TableHandler

__all__ = [
    "BaseTaskHandler",
    "TaskResult",
    "TaskType",
    "get_handler",
    "list_handlers",
    "register_handler",
    "OCRHandler",
    "LayoutHandler",
    "TableHandler",
    "FieldExtractionHandler",
    "NERHandler",
    "PRESET_SCHEMAS",
    "ENTITY_TYPES",
]
