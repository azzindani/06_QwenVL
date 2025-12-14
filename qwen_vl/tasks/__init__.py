"""Task handlers for document processing."""

from .base import (
    BaseTaskHandler,
    TaskResult,
    TaskType,
    get_handler,
    list_handlers,
    register_handler,
)

# Import existing handlers to register them
from .field_extraction import PRESET_SCHEMAS, FieldExtractionHandler
from .layout import LayoutHandler
from .ner import ENTITY_TYPES, NERHandler
from .ocr import OCRHandler
from .table import TableHandler
from .form import FormHandler
from .invoice import InvoiceHandler
from .contract import ContractHandler

# Import new handlers from Phase 2
from .recognition import RecognitionHandler
from .spatial import SpatialHandler
from .video import VideoHandler
from .document_parsing import DocumentParsingHandler
from .computer_agent import ComputerAgentHandler
from .mobile_agent import MobileAgentHandler

__all__ = [
    # Base classes and utilities
    "BaseTaskHandler",
    "TaskResult",
    "TaskType",
    "get_handler",
    "list_handlers",
    "register_handler",
    # Phase 1 handlers
    "OCRHandler",
    "LayoutHandler",
    "RecognitionHandler",
    "SpatialHandler",
    "VideoHandler",
    "DocumentParsingHandler",
    # Phase 2 handlers
    "TableHandler",
    "FieldExtractionHandler",
    "NERHandler",
    # Phase 3 handlers
    "FormHandler",
    "InvoiceHandler",
    "ContractHandler",
    # Agent handlers
    "ComputerAgentHandler",
    "MobileAgentHandler",
    # Constants
    "PRESET_SCHEMAS",
    "ENTITY_TYPES",
]

