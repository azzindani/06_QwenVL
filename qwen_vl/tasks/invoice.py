"""Invoice and receipt parsing task handler."""

from typing import Any, Dict, List, Optional, Union

from PIL import Image

from ..utils.parsers import parse_json_from_markdown, extract_malformed_bbox, normalize_bbox
from ..utils.visualization import draw_bounding_boxes
from .base import BaseTaskHandler, TaskResult, TaskType, register_handler


@register_handler(TaskType.INVOICE)
class InvoiceHandler(BaseTaskHandler):
    """Handler for invoice and receipt parsing tasks."""

    @property
    def task_type(self) -> TaskType:
        return TaskType.INVOICE

    @property
    def system_prompt(self) -> str:
        return (
            "You are a helpful assistant specialized in invoice and receipt parsing. "
            "Extract vendor information, line items, taxes, totals, dates, and payment details "
            "from invoice and receipt documents with high accuracy."
        )

    def process(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        document_type: str = "invoice",
        **kwargs,
    ) -> TaskResult:
        """
        Parse invoice or receipt from an image.

        Args:
            image: Image path or PIL Image
            prompt: Optional custom prompt
            document_type: 'invoice' or 'receipt'

        Returns:
            TaskResult with parsed invoice/receipt data
        """
        img = self._load_image(image)

        user_prompt = prompt or self._build_invoice_prompt(document_type)

        messages = self._build_messages(img, user_prompt)
        response = self._generate(messages, **kwargs)

        # Parse invoice data
        data = parse_json_from_markdown(response)

        if data:
            header = data.get("header", {})
            line_items = data.get("line_items", [])
            summary = data.get("summary", {})
            payment = data.get("payment", {})
        else:
            header = {}
            line_items = []
            summary = {}
            payment = {}

        # Validate totals
        validation = self._validate_invoice(line_items, summary)

        # Create visualization
        boxes = []
        print(f"[INVOICE] Image size: {img.size}, Model input: ({self.last_input_width}x{self.last_input_height})")
        if data and "bounding_boxes" in data:
            raw_boxes = data["bounding_boxes"]
            for box_data in raw_boxes:
                if isinstance(box_data, dict):
                    bbox_raw = box_data.get("bbox") or box_data.get("bbox_2d")
                    if bbox_raw:
                        bbox = normalize_bbox(bbox_raw)
                        if bbox:
                            boxes.append({"bbox": bbox, "label": box_data.get("label", "")})
                    else:
                        # Maybe the box_data itself is a bbox
                        bbox = normalize_bbox(box_data)
                        if bbox:
                            boxes.append({"bbox": bbox})

        print(f"[INVOICE] Drawing {len(boxes)} boxes")
        vis_image = draw_bounding_boxes(
            img, 
            boxes,
            input_width=self.last_input_width,
            input_height=self.last_input_height
        ) if boxes else None

        return TaskResult(
            text=response,
            data={
                "header": header,
                "line_items": line_items,
                "summary": summary,
                "payment": payment,
                "validation": validation,
            },
            bounding_boxes=boxes,
            visualization=vis_image,
            metadata={
                "document_type": document_type,
                "item_count": len(line_items),
                "is_valid": validation.get("is_valid", False),
            },
        )

    def _build_invoice_prompt(self, document_type: str) -> str:
        """Build invoice/receipt parsing prompt."""
        doc_name = "invoice" if document_type == "invoice" else "receipt"

        return (
            f"Parse this {doc_name} and extract all information.\n\n"
            "Return as JSON with this structure:\n"
            "```json\n"
            "{\n"
            '  "header": {\n'
            '    "vendor_name": "Company Name",\n'
            '    "vendor_address": "Full address",\n'
            '    "vendor_phone": "Phone number",\n'
            '    "vendor_email": "Email",\n'
            '    "invoice_number": "INV-001",\n'
            '    "date": "2024-01-15",\n'
            '    "due_date": "2024-02-15",\n'
            '    "customer_name": "Customer",\n'
            '    "customer_address": "Address"\n'
            '  },\n'
            '  "line_items": [\n'
            '    {\n'
            '      "description": "Item description",\n'
            '      "quantity": 1,\n'
            '      "unit_price": 10.00,\n'
            '      "amount": 10.00,\n'
            '      "tax_rate": 0.1\n'
            '    }\n'
            '  ],\n'
            '  "summary": {\n'
            '    "subtotal": 10.00,\n'
            '    "tax": 1.00,\n'
            '    "discount": 0.00,\n'
            '    "total": 11.00,\n'
            '    "currency": "USD"\n'
            '  },\n'
            '  "payment": {\n'
            '    "method": "Bank Transfer",\n'
            '    "terms": "Net 30",\n'
            '    "bank_details": "Account info if present"\n'
            '  }\n'
            '}\n'
            "```\n\n"
            "Extract all visible information. Use null for missing fields."
        )

    def _validate_invoice(
        self, line_items: List[Dict], summary: Dict
    ) -> Dict[str, Any]:
        """
        Validate invoice calculations.

        Returns:
            Validation results with any errors
        """
        errors = []
        warnings = []

        # Calculate expected subtotal
        if line_items:
            calculated_subtotal = sum(
                item.get("amount", 0) or 0 for item in line_items
            )
            reported_subtotal = summary.get("subtotal", 0) or 0

            if abs(calculated_subtotal - reported_subtotal) > 0.01:
                errors.append(
                    f"Subtotal mismatch: calculated {calculated_subtotal:.2f}, "
                    f"reported {reported_subtotal:.2f}"
                )

        def safe_float(val):
            """Convert a value to float, handling strings and None."""
            if val is None:
                return 0.0
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                # Remove currency symbols and whitespace
                cleaned = val.strip().replace('$', '').replace(',', '').replace('€', '').replace('£', '')
                try:
                    return float(cleaned) if cleaned else 0.0
                except ValueError:
                    return 0.0
            return 0.0

        # Check total = subtotal + tax - discount
        if summary is None:
            summary = {}
        subtotal = safe_float(summary.get("subtotal", 0))
        tax = safe_float(summary.get("tax", 0))
        discount = safe_float(summary.get("discount", 0))
        total = safe_float(summary.get("total", 0))

        expected_total = subtotal + tax - discount
        if abs(expected_total - total) > 0.01:
            errors.append(
                f"Total mismatch: expected {expected_total:.2f}, "
                f"reported {total:.2f}"
            )

        # Validate line items
        if line_items and isinstance(line_items, list):
            for i, item in enumerate(line_items):
                if not isinstance(item, dict):
                    continue
                qty = safe_float(item.get("quantity", 0))
                price = safe_float(item.get("unit_price", 0))
                amount = safe_float(item.get("amount", 0))

                expected = qty * price
                if abs(expected - amount) > 0.01:
                    warnings.append(
                        f"Line {i+1}: qty*price ({expected:.2f}) != amount ({amount:.2f})"
                    )

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def parse_receipt(
        self,
        image: Union[str, Image.Image],
        **kwargs,
    ) -> TaskResult:
        """
        Parse a receipt (convenience method).

        Args:
            image: Image path or PIL Image

        Returns:
            TaskResult with parsed receipt data
        """
        return self.process(image, document_type="receipt", **kwargs)


if __name__ == "__main__":
    print("=" * 60)
    print("INVOICE HANDLER TEST")
    print("=" * 60)
    print("  Invoice handler registered successfully")
    print(f"  Task type: {TaskType.INVOICE}")
    print("=" * 60)
