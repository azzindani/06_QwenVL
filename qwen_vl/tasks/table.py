"""Table extraction task handler."""

import csv
import io
import json
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from ..utils.parsers import parse_json_array_from_markdown, parse_json_from_markdown
from ..utils.visualization import draw_bounding_boxes
from .base import BaseTaskHandler, TaskResult, TaskType, register_handler


@register_handler(TaskType.TABLE)
class TableHandler(BaseTaskHandler):
    """Handler for table extraction tasks."""

    @property
    def task_type(self) -> TaskType:
        return TaskType.TABLE

    @property
    def system_prompt(self) -> str:
        return (
            "You are a helpful assistant specialized in extracting tables from documents. "
            "Identify tables, extract their structure including headers and cells, "
            "and convert them to structured formats like JSON or CSV."
        )

    def process(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        output_format: str = "json",
        **kwargs,
    ) -> TaskResult:
        """
        Extract tables from an image.

        Args:
            image: Image path or PIL Image
            prompt: Optional custom prompt
            output_format: 'json' or 'csv'

        Returns:
            TaskResult with extracted tables
        """
        img = self._load_image(image)

        user_prompt = prompt or (
            "Extract all tables from this image. For each table:\n"
            "1. Identify the table boundaries (bounding box)\n"
            "2. Extract headers (column names)\n"
            "3. Extract all rows of data\n\n"
            "Return as JSON:\n"
            "```json\n"
            '{\n'
            '  "tables": [\n'
            '    {\n'
            '      "bbox": {"x1": 0, "y1": 0, "x2": 500, "y2": 300},\n'
            '      "headers": ["Column1", "Column2"],\n'
            '      "rows": [\n'
            '        ["value1", "value2"],\n'
            '        ["value3", "value4"]\n'
            '      ]\n'
            '    }\n'
            '  ]\n'
            '}\n'
            "```"
        )

        messages = self._build_messages(img, user_prompt)
        response = self._generate(messages, **kwargs)

        # Parse tables
        data = parse_json_from_markdown(response)
        tables = data.get("tables", []) if data else []

        # Create visualization
        vis_image = None
        if tables:
            boxes = [{"bbox": t["bbox"], "label": f"Table {i+1}"} for i, t in enumerate(tables) if "bbox" in t]
            if boxes:
                vis_image = draw_bounding_boxes(img, boxes)

        # Convert to CSV if requested
        csv_output = None
        if output_format == "csv" and tables:
            csv_output = self._tables_to_csv(tables)

        return TaskResult(
            text=response,
            data={"tables": tables, "csv": csv_output},
            bounding_boxes=[t.get("bbox") for t in tables if "bbox" in t],
            visualization=vis_image,
            metadata={
                "table_count": len(tables),
                "output_format": output_format,
            },
        )

    def _tables_to_csv(self, tables: List[Dict[str, Any]]) -> str:
        """Convert tables to CSV string."""
        output = io.StringIO()

        for i, table in enumerate(tables):
            if i > 0:
                output.write("\n\n")

            writer = csv.writer(output)

            # Write headers
            headers = table.get("headers", [])
            if headers:
                writer.writerow(headers)

            # Write rows
            rows = table.get("rows", [])
            for row in rows:
                writer.writerow(row)

        return output.getvalue()

    def extract_to_dataframe(
        self,
        image: Union[str, Image.Image],
        table_index: int = 0,
        **kwargs,
    ) -> Any:
        """
        Extract table and return as pandas DataFrame.

        Args:
            image: Image path or PIL Image
            table_index: Index of table to extract (if multiple)

        Returns:
            pandas DataFrame (requires pandas installed)
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for DataFrame output")

        result = self.process(image, **kwargs)
        tables = result.data.get("tables", [])

        if not tables or table_index >= len(tables):
            return pd.DataFrame()

        table = tables[table_index]
        headers = table.get("headers", [])
        rows = table.get("rows", [])

        if headers:
            return pd.DataFrame(rows, columns=headers)
        else:
            return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 60)
    print("TABLE HANDLER TEST")
    print("=" * 60)
    print("  Table handler registered successfully")
    print(f"  Task type: {TaskType.TABLE}")
    print("=" * 60)
