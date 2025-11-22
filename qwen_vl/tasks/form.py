"""Form understanding task handler."""

from typing import Any, Dict, List, Optional, Union

from PIL import Image

from ..utils.parsers import parse_json_from_markdown
from ..utils.visualization import draw_bounding_boxes
from .base import BaseTaskHandler, TaskResult, TaskType, register_handler


@register_handler(TaskType.FORM)
class FormHandler(BaseTaskHandler):
    """Handler for form understanding tasks."""

    @property
    def task_type(self) -> TaskType:
        return TaskType.FORM

    @property
    def system_prompt(self) -> str:
        return (
            "You are a helpful assistant specialized in form understanding. "
            "Extract key-value pairs, detect checkboxes and radio buttons, "
            "identify signatures, and understand form structure from document images."
        )

    def process(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        extract_signatures: bool = True,
        extract_checkboxes: bool = True,
        **kwargs,
    ) -> TaskResult:
        """
        Extract form elements from an image.

        Args:
            image: Image path or PIL Image
            prompt: Optional custom prompt
            extract_signatures: Whether to detect signatures
            extract_checkboxes: Whether to detect checkboxes/radio buttons

        Returns:
            TaskResult with extracted form data
        """
        img = self._load_image(image)

        user_prompt = prompt or self._build_form_prompt(
            extract_signatures, extract_checkboxes
        )

        messages = self._build_messages(img, user_prompt)
        response = self._generate(messages, **kwargs)

        # Parse form data
        data = parse_json_from_markdown(response)

        if data:
            fields = data.get("fields", [])
            checkboxes = data.get("checkboxes", [])
            signatures = data.get("signatures", [])
        else:
            fields = []
            checkboxes = []
            signatures = []

        # Create visualization
        boxes = []

        for field in fields:
            if "bbox" in field:
                boxes.append({
                    "bbox": field["bbox"],
                    "label": f"{field.get('key', 'field')[:20]}",
                    "color": "#00FF00",
                })

        for checkbox in checkboxes:
            if "bbox" in checkbox:
                state = "☑" if checkbox.get("checked", False) else "☐"
                boxes.append({
                    "bbox": checkbox["bbox"],
                    "label": f"{state} {checkbox.get('label', '')[:15]}",
                    "color": "#0000FF",
                })

        for sig in signatures:
            if "bbox" in sig:
                boxes.append({
                    "bbox": sig["bbox"],
                    "label": "Signature",
                    "color": "#FF00FF",
                })

        vis_image = draw_bounding_boxes(img, boxes) if boxes else None

        return TaskResult(
            text=response,
            data={
                "fields": fields,
                "checkboxes": checkboxes,
                "signatures": signatures,
            },
            bounding_boxes=boxes,
            visualization=vis_image,
            metadata={
                "field_count": len(fields),
                "checkbox_count": len(checkboxes),
                "signature_count": len(signatures),
            },
        )

    def _build_form_prompt(
        self, extract_signatures: bool, extract_checkboxes: bool
    ) -> str:
        """Build form extraction prompt."""
        prompt_parts = [
            "Analyze this form and extract the following information:\n\n",
            "1. **Key-Value Pairs**: All form fields with their labels and values\n",
        ]

        if extract_checkboxes:
            prompt_parts.append(
                "2. **Checkboxes/Radio Buttons**: All selection controls with their state (checked/unchecked)\n"
            )

        if extract_signatures:
            prompt_parts.append(
                "3. **Signatures**: Detect any signature fields or handwritten signatures\n"
            )

        prompt_parts.append(
            "\nReturn as JSON:\n"
            "```json\n"
            "{\n"
            '  "fields": [\n'
            '    {\n'
            '      "key": "field label",\n'
            '      "value": "field value",\n'
            '      "type": "text|date|number|dropdown",\n'
            '      "required": true,\n'
            '      "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 50}\n'
            '    }\n'
            '  ],\n'
            '  "checkboxes": [\n'
            '    {\n'
            '      "label": "checkbox label",\n'
            '      "checked": true,\n'
            '      "group": "group name if radio button",\n'
            '      "bbox": {"x1": 0, "y1": 0, "x2": 20, "y2": 20}\n'
            '    }\n'
            '  ],\n'
            '  "signatures": [\n'
            '    {\n'
            '      "type": "handwritten|digital|stamp",\n'
            '      "name": "signer name if visible",\n'
            '      "date": "date if present",\n'
            '      "bbox": {"x1": 0, "y1": 0, "x2": 200, "y2": 100}\n'
            '    }\n'
            '  ]\n'
            '}\n'
            "```"
        )

        return "".join(prompt_parts)

    def extract_fields_only(
        self,
        image: Union[str, Image.Image],
        **kwargs,
    ) -> TaskResult:
        """
        Extract only key-value fields from form.

        Args:
            image: Image path or PIL Image

        Returns:
            TaskResult with field data only
        """
        return self.process(
            image,
            extract_signatures=False,
            extract_checkboxes=False,
            **kwargs,
        )


if __name__ == "__main__":
    print("=" * 60)
    print("FORM HANDLER TEST")
    print("=" * 60)
    print("  Form handler registered successfully")
    print(f"  Task type: {TaskType.FORM}")
    print("=" * 60)
