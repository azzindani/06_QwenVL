"""Layout analysis task handler."""

from typing import Optional, Union

from PIL import Image

from ..utils.parsers import parse_coordinates, parse_json_from_markdown
from ..utils.visualization import draw_bounding_boxes
from .base import BaseTaskHandler, TaskResult, TaskType, register_handler


@register_handler(TaskType.LAYOUT)
class LayoutHandler(BaseTaskHandler):
    """Handler for document layout analysis tasks."""

    @property
    def task_type(self) -> TaskType:
        return TaskType.LAYOUT

    @property
    def system_prompt(self) -> str:
        return (
            "You are a helpful assistant specialized in document layout analysis. "
            "Identify and locate different structural elements in documents such as "
            "headers, paragraphs, tables, figures, lists, and other components."
        )

    def process(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        **kwargs,
    ) -> TaskResult:
        """
        Analyze document layout and identify structural elements.

        Args:
            image: Image path or PIL Image
            prompt: Optional custom prompt

        Returns:
            TaskResult with layout analysis
        """
        img = self._load_image(image)

        user_prompt = prompt or (
            "Analyze the layout of this document. Identify all structural elements "
            "including headers, paragraphs, tables, figures, lists, and captions. "
            "For each element, provide:\n"
            "- type: the element type (header, paragraph, table, figure, list, caption, etc.)\n"
            "- bbox: bounding box coordinates (x1, y1, x2, y2)\n"
            "- level: hierarchy level for headers (1, 2, 3...)\n\n"
            "Return as JSON array:\n"
            '```json\n'
            '[{"type": "header", "level": 1, "bbox": {"x1": 0, "y1": 0, "x2": 500, "y2": 50}}]\n'
            '```'
        )

        messages = self._build_messages(img, user_prompt)
        response = self._generate(messages, **kwargs)

        # Parse layout elements
        elements = parse_coordinates(response)

        # Add labels based on type
        for elem in elements:
            if "type" in elem:
                level = elem.get("level", "")
                elem["label"] = f"{elem['type']}{level}"

        # Create visualization
        vis_image = None
        if elements:
            vis_image = draw_bounding_boxes(img, elements)

        return TaskResult(
            text=response,
            bounding_boxes=elements,
            visualization=vis_image,
            data={"elements": elements},
            metadata={"element_count": len(elements)},
        )

    def detect_sections(
        self,
        image: Union[str, Image.Image],
        **kwargs,
    ) -> TaskResult:
        """
        Detect major sections in a document.

        Args:
            image: Image path or PIL Image

        Returns:
            TaskResult with section information
        """
        img = self._load_image(image)

        prompt = (
            "Identify the major sections in this document. "
            "For each section, provide the title and bounding box. "
            "Return as JSON array:\n"
            '```json\n'
            '[{"title": "Introduction", "bbox": {"x1": 0, "y1": 0, "x2": 500, "y2": 200}}]\n'
            '```'
        )

        messages = self._build_messages(img, prompt)
        response = self._generate(messages, **kwargs)

        sections = parse_coordinates(response)

        # Add labels
        for section in sections:
            if "title" in section:
                section["label"] = section["title"]

        vis_image = draw_bounding_boxes(img, sections) if sections else None

        return TaskResult(
            text=response,
            bounding_boxes=sections,
            visualization=vis_image,
            data={"sections": sections},
            metadata={"section_count": len(sections)},
        )

    def detect_reading_order(
        self,
        image: Union[str, Image.Image],
        **kwargs,
    ) -> TaskResult:
        """
        Determine the reading order of elements in a document.

        Args:
            image: Image path or PIL Image

        Returns:
            TaskResult with ordered elements
        """
        img = self._load_image(image)

        prompt = (
            "Analyze this document and determine the correct reading order of all text elements. "
            "Number each element in the order it should be read. "
            "Return as JSON array with order numbers:\n"
            '```json\n'
            '[{"order": 1, "type": "header", "bbox": {"x1": 0, "y1": 0, "x2": 500, "y2": 50}}]\n'
            '```'
        )

        messages = self._build_messages(img, prompt)
        response = self._generate(messages, **kwargs)

        elements = parse_coordinates(response)

        # Add order labels
        for elem in elements:
            if "order" in elem:
                elem["label"] = f"{elem.get('order', '?')}"

        vis_image = draw_bounding_boxes(img, elements) if elements else None

        return TaskResult(
            text=response,
            bounding_boxes=elements,
            visualization=vis_image,
            data={"reading_order": elements},
            metadata={"element_count": len(elements)},
        )


if __name__ == "__main__":
    print("=" * 60)
    print("LAYOUT HANDLER TEST")
    print("=" * 60)
    print("  Layout handler registered successfully")
    print(f"  Task type: {TaskType.LAYOUT}")
    print("  (Actual inference requires loaded model)")
    print("=" * 60)
