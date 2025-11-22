"""API schema generation utilities."""

from typing import Any, Dict, List, Optional, Type
from datetime import datetime
from pydantic import BaseModel, Field, create_model


def schema_to_pydantic(
    name: str,
    schema: Dict[str, Any],
) -> Type[BaseModel]:
    """
    Generate a Pydantic model from a field extraction schema.

    Args:
        name: Name for the generated model
        schema: Field extraction schema with field definitions

    Returns:
        Generated Pydantic model class
    """
    field_definitions = {}

    fields = schema.get("fields", [])
    for field in fields:
        field_name = field.get("name", "unknown")
        field_type = field.get("type", "text")
        required = field.get("required", False)
        description = field.get("description", "")

        # Map schema types to Python types
        python_type = _get_python_type(field_type)

        # Create field with default
        if required:
            field_definitions[field_name] = (
                python_type,
                Field(..., description=description)
            )
        else:
            field_definitions[field_name] = (
                Optional[python_type],
                Field(None, description=description)
            )

    # Create the model dynamically
    model = create_model(name, **field_definitions)
    return model


def _get_python_type(field_type: str) -> type:
    """Map schema field type to Python type."""
    type_mapping = {
        "text": str,
        "string": str,
        "number": float,
        "integer": int,
        "date": str,
        "email": str,
        "phone": str,
        "currency": str,
        "url": str,
        "boolean": bool,
        "list": List[str],
        "array": List[str],
    }
    return type_mapping.get(field_type.lower(), str)


def generate_extraction_models() -> Dict[str, Type[BaseModel]]:
    """
    Generate Pydantic models for all preset schemas.

    Returns:
        Dict of model name to Pydantic model class
    """
    from ..tasks.field_extraction import PRESET_SCHEMAS

    models = {}
    for preset_name, schema in PRESET_SCHEMAS.items():
        model_name = _to_class_name(preset_name)
        models[preset_name] = schema_to_pydantic(model_name, schema)

    return models


def _to_class_name(name: str) -> str:
    """Convert snake_case to PascalCase."""
    return "".join(word.capitalize() for word in name.split("_"))


# Pre-built response models
class ExtractionResult(BaseModel):
    """Base extraction result model."""
    success: bool = Field(..., description="Whether extraction succeeded")
    text: str = Field(..., description="Raw model response")
    data: Optional[Dict[str, Any]] = Field(None, description="Extracted structured data")
    confidence: Optional[float] = Field(None, description="Confidence score 0-1")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")


class OCRResult(BaseModel):
    """OCR extraction result."""
    success: bool
    text: str
    bounding_boxes: Optional[List[Dict[str, Any]]] = None
    word_count: Optional[int] = None


class TableResult(BaseModel):
    """Table extraction result."""
    success: bool
    tables: List[Dict[str, Any]]
    csv_data: Optional[str] = None


class FormResult(BaseModel):
    """Form extraction result."""
    success: bool
    fields: List[Dict[str, Any]]
    checkboxes: List[Dict[str, Any]]
    signatures: List[Dict[str, Any]]


class InvoiceResult(BaseModel):
    """Invoice parsing result."""
    success: bool
    header: Dict[str, Any]
    line_items: List[Dict[str, Any]]
    summary: Dict[str, Any]
    payment: Dict[str, Any]
    validation: Dict[str, Any]


class ContractResult(BaseModel):
    """Contract analysis result."""
    success: bool
    parties: List[Dict[str, Any]]
    dates: Dict[str, Any]
    clauses: List[Dict[str, Any]]
    obligations: List[Dict[str, Any]]
    key_terms: Dict[str, Any]


class NERResult(BaseModel):
    """Named entity recognition result."""
    success: bool
    entities: List[Dict[str, Any]]
    entity_counts: Dict[str, int]


class BatchJobStatus(BaseModel):
    """Batch job status."""
    job_id: str
    status: str  # pending, processing, completed, failed
    total_items: int
    processed_items: int
    failed_items: int
    created_at: datetime
    completed_at: Optional[datetime] = None
    results: Optional[List[Dict[str, Any]]] = None


class WebhookPayload(BaseModel):
    """Webhook event payload."""
    event_type: str  # extraction.completed, extraction.failed, batch.completed
    job_id: Optional[str] = None
    document_id: Optional[str] = None
    timestamp: datetime
    data: Dict[str, Any]


if __name__ == "__main__":
    print("=" * 60)
    print("API SCHEMA GENERATION TEST")
    print("=" * 60)

    # Test schema to Pydantic
    test_schema = {
        "fields": [
            {"name": "vendor_name", "type": "text", "required": True},
            {"name": "total", "type": "currency", "required": True},
            {"name": "date", "type": "date", "required": False},
        ]
    }

    Model = schema_to_pydantic("TestInvoice", test_schema)
    print(f"  Generated model: {Model.__name__}")
    print(f"  Fields: {list(Model.model_fields.keys())}")

    # Test preset models
    models = generate_extraction_models()
    print(f"  Preset models: {list(models.keys())}")

    print("=" * 60)
