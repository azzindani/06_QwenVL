"""Field extraction task handler with schema support."""

from typing import Any, Dict, List, Optional, Union

from PIL import Image

from ..utils.parsers import parse_json_from_markdown
from ..utils.visualization import draw_bounding_boxes
from .base import BaseTaskHandler, TaskResult, TaskType, register_handler


# Preset schemas for common document types
PRESET_SCHEMAS = {
    "invoice": {
        "fields": [
            {"name": "invoice_number", "type": "string", "description": "Invoice number or ID"},
            {"name": "date", "type": "date", "description": "Invoice date"},
            {"name": "due_date", "type": "date", "description": "Payment due date"},
            {"name": "vendor_name", "type": "string", "description": "Vendor/seller name"},
            {"name": "vendor_address", "type": "string", "description": "Vendor address"},
            {"name": "customer_name", "type": "string", "description": "Customer/buyer name"},
            {"name": "subtotal", "type": "currency", "description": "Subtotal amount"},
            {"name": "tax", "type": "currency", "description": "Tax amount"},
            {"name": "total", "type": "currency", "description": "Total amount"},
        ]
    },
    "receipt": {
        "fields": [
            {"name": "store_name", "type": "string", "description": "Store/merchant name"},
            {"name": "date", "type": "date", "description": "Transaction date"},
            {"name": "items", "type": "array", "description": "List of purchased items"},
            {"name": "subtotal", "type": "currency", "description": "Subtotal"},
            {"name": "tax", "type": "currency", "description": "Tax amount"},
            {"name": "total", "type": "currency", "description": "Total amount"},
            {"name": "payment_method", "type": "string", "description": "Payment method used"},
        ]
    },
    "id_card": {
        "fields": [
            {"name": "full_name", "type": "string", "description": "Full name"},
            {"name": "date_of_birth", "type": "date", "description": "Date of birth"},
            {"name": "id_number", "type": "string", "description": "ID/document number"},
            {"name": "expiry_date", "type": "date", "description": "Expiration date"},
            {"name": "address", "type": "string", "description": "Address"},
            {"name": "nationality", "type": "string", "description": "Nationality/country"},
        ]
    },
    "business_card": {
        "fields": [
            {"name": "name", "type": "string", "description": "Person's name"},
            {"name": "title", "type": "string", "description": "Job title"},
            {"name": "company", "type": "string", "description": "Company name"},
            {"name": "email", "type": "email", "description": "Email address"},
            {"name": "phone", "type": "phone", "description": "Phone number"},
            {"name": "address", "type": "string", "description": "Address"},
            {"name": "website", "type": "url", "description": "Website URL"},
        ]
    },
}


@register_handler(TaskType.FIELD_EXTRACTION)
class FieldExtractionHandler(BaseTaskHandler):
    """Handler for schema-based field extraction."""

    @property
    def task_type(self) -> TaskType:
        return TaskType.FIELD_EXTRACTION

    @property
    def system_prompt(self) -> str:
        return (
            "You are a helpful assistant specialized in extracting specific fields "
            "from documents. Extract only the requested fields and provide confidence "
            "scores for each extracted value."
        )

    def process(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        preset: Optional[str] = None,
        **kwargs,
    ) -> TaskResult:
        """
        Extract fields from an image using a schema.

        Args:
            image: Image path or PIL Image
            prompt: Optional custom prompt
            schema: Custom schema with 'fields' list
            preset: Preset schema name ('invoice', 'receipt', 'id_card', 'business_card')

        Returns:
            TaskResult with extracted fields
        """
        img = self._load_image(image)

        # Get schema
        if preset and preset in PRESET_SCHEMAS:
            schema = PRESET_SCHEMAS[preset]
        elif schema is None:
            schema = PRESET_SCHEMAS["invoice"]  # Default

        # Build prompt from schema
        if prompt:
            user_prompt = prompt
        else:
            user_prompt = self._build_schema_prompt(schema)

        messages = self._build_messages(img, user_prompt)
        response = self._generate(messages, **kwargs)

        # Parse extracted fields
        data = parse_json_from_markdown(response)
        fields = data.get("fields", data) if data else {}

        # Extract bounding boxes if present
        boxes = []
        if isinstance(fields, dict):
            for field_name, field_data in fields.items():
                if isinstance(field_data, dict) and "bbox" in field_data:
                    boxes.append({
                        "bbox": field_data["bbox"],
                        "label": field_name,
                    })

        # Create visualization
        vis_image = draw_bounding_boxes(img, boxes) if boxes else None

        return TaskResult(
            text=response,
            data={"fields": fields, "schema": schema},
            bounding_boxes=boxes,
            visualization=vis_image,
            metadata={
                "field_count": len(fields) if isinstance(fields, dict) else 0,
                "preset": preset,
            },
        )

    def _build_schema_prompt(self, schema: Dict[str, Any]) -> str:
        """Build extraction prompt from schema."""
        fields = schema.get("fields", [])

        field_descriptions = []
        for field in fields:
            name = field.get("name", "unknown")
            field_type = field.get("type", "string")
            description = field.get("description", "")
            field_descriptions.append(f"- {name} ({field_type}): {description}")

        fields_text = "\n".join(field_descriptions)

        return (
            f"Extract the following fields from this document:\n\n"
            f"{fields_text}\n\n"
            "Return as JSON with field names as keys. For each field, provide:\n"
            "- value: the extracted value\n"
            "- confidence: confidence score (0.0 to 1.0)\n"
            "- bbox: bounding box coordinates (optional)\n\n"
            "Example:\n"
            "```json\n"
            '{\n'
            '  "fields": {\n'
            '    "field_name": {\n'
            '      "value": "extracted value",\n'
            '      "confidence": 0.95,\n'
            '      "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 50}\n'
            '    }\n'
            '  }\n'
            '}\n'
            "```"
        )

    @staticmethod
    def list_presets() -> List[str]:
        """List available preset schemas."""
        return list(PRESET_SCHEMAS.keys())

    @staticmethod
    def get_preset_schema(name: str) -> Optional[Dict[str, Any]]:
        """Get a preset schema by name."""
        return PRESET_SCHEMAS.get(name)


if __name__ == "__main__":
    print("=" * 60)
    print("FIELD EXTRACTION HANDLER TEST")
    print("=" * 60)
    print("  Field extraction handler registered successfully")
    print(f"  Task type: {TaskType.FIELD_EXTRACTION}")
    print(f"  Available presets: {FieldExtractionHandler.list_presets()}")
    print("=" * 60)
