"""Schema builder UI components for Gradio."""

from typing import Any, Dict, List, Optional
import json


def create_schema_builder_ui():
    """
    Create Gradio components for schema builder.

    Returns:
        Tuple of Gradio components
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("gradio is required. Install with: pip install gradio")

    # Field type options
    field_types = [
        "text", "number", "integer", "date", "email",
        "phone", "currency", "url", "boolean"
    ]

    # Validator options
    validators = [
        "none", "email", "phone", "date", "currency", "url", "percentage", "integer"
    ]

    with gr.Blocks() as builder:
        gr.Markdown("## Schema Builder")
        gr.Markdown("Define fields to extract from documents")

        with gr.Row():
            schema_name = gr.Textbox(
                label="Schema Name",
                placeholder="e.g., invoice, receipt, contract",
            )

        # Field list state
        fields_state = gr.State([])

        # Add field section
        gr.Markdown("### Add Field")

        with gr.Row():
            field_name = gr.Textbox(label="Field Name", placeholder="e.g., vendor_name")
            field_type = gr.Dropdown(label="Type", choices=field_types, value="text")
            field_required = gr.Checkbox(label="Required", value=False)

        with gr.Row():
            field_description = gr.Textbox(
                label="Description",
                placeholder="Description of what to extract",
            )
            field_validator = gr.Dropdown(
                label="Validator",
                choices=validators,
                value="none",
            )

        add_btn = gr.Button("Add Field", variant="primary")

        # Current fields display
        gr.Markdown("### Current Fields")
        fields_display = gr.JSON(label="Fields", value=[])

        # Actions
        with gr.Row():
            clear_btn = gr.Button("Clear All")
            export_btn = gr.Button("Export Schema", variant="secondary")

        # Output
        schema_output = gr.Code(label="Generated Schema", language="json")

        # Event handlers
        def add_field(fields, name, ftype, required, desc, validator):
            if not name:
                return fields, fields

            field = {
                "name": name,
                "type": ftype,
                "required": required,
                "description": desc,
            }

            if validator != "none":
                field["validator"] = validator

            fields = fields + [field]
            return fields, fields

        def clear_fields():
            return [], []

        def export_schema(name, fields):
            schema = {
                "name": name or "custom_schema",
                "fields": fields,
            }
            return json.dumps(schema, indent=2)

        add_btn.click(
            add_field,
            inputs=[fields_state, field_name, field_type, field_required, field_description, field_validator],
            outputs=[fields_state, fields_display],
        )

        clear_btn.click(
            clear_fields,
            outputs=[fields_state, fields_display],
        )

        export_btn.click(
            export_schema,
            inputs=[schema_name, fields_state],
            outputs=[schema_output],
        )

    return builder


def schema_to_ui_fields(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert schema to UI-friendly format.

    Args:
        schema: Extraction schema

    Returns:
        List of field definitions for UI
    """
    fields = schema.get("fields", [])
    return [
        {
            "name": f.get("name", ""),
            "type": f.get("type", "text"),
            "required": f.get("required", False),
            "description": f.get("description", ""),
            "validator": f.get("validator", "none"),
        }
        for f in fields
    ]


def ui_fields_to_schema(name: str, fields: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert UI fields to schema format.

    Args:
        name: Schema name
        fields: List of field definitions from UI

    Returns:
        Extraction schema
    """
    schema_fields = []

    for f in fields:
        field = {
            "name": f["name"],
            "type": f["type"],
            "required": f["required"],
        }

        if f.get("description"):
            field["description"] = f["description"]

        if f.get("validator") and f["validator"] != "none":
            field["validator"] = f["validator"]

        schema_fields.append(field)

    return {
        "name": name,
        "fields": schema_fields,
    }


# Preset schema templates for quick start
SCHEMA_TEMPLATES = {
    "invoice": {
        "name": "invoice",
        "fields": [
            {"name": "invoice_number", "type": "text", "required": True, "description": "Invoice ID"},
            {"name": "date", "type": "date", "required": True, "description": "Invoice date"},
            {"name": "vendor_name", "type": "text", "required": True, "description": "Vendor/company name"},
            {"name": "total", "type": "currency", "required": True, "description": "Total amount"},
            {"name": "tax", "type": "currency", "required": False, "description": "Tax amount"},
        ],
    },
    "receipt": {
        "name": "receipt",
        "fields": [
            {"name": "store_name", "type": "text", "required": True},
            {"name": "date", "type": "date", "required": True},
            {"name": "total", "type": "currency", "required": True},
            {"name": "payment_method", "type": "text", "required": False},
        ],
    },
    "business_card": {
        "name": "business_card",
        "fields": [
            {"name": "name", "type": "text", "required": True},
            {"name": "title", "type": "text", "required": False},
            {"name": "company", "type": "text", "required": False},
            {"name": "email", "type": "email", "required": False, "validator": "email"},
            {"name": "phone", "type": "phone", "required": False, "validator": "phone"},
        ],
    },
}


if __name__ == "__main__":
    print("=" * 60)
    print("SCHEMA BUILDER UI")
    print("=" * 60)
    print("  Templates available:", list(SCHEMA_TEMPLATES.keys()))
    print("=" * 60)
