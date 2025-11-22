"""OCR task handler for text extraction."""

from typing import Optional, Union

from PIL import Image

from ..utils.parsers import parse_coordinates, parse_json_from_markdown
from ..utils.visualization import draw_bounding_boxes
from .base import BaseTaskHandler, TaskResult, TaskType, register_handler


@register_handler(TaskType.OCR)
class OCRHandler(BaseTaskHandler):
    """Handler for OCR (Optical Character Recognition) tasks."""

    @property
    def task_type(self) -> TaskType:
        return TaskType.OCR

    @property
    def system_prompt(self) -> str:
        return (
            "You are a helpful assistant specialized in optical character recognition "
            "and text extraction. Extract all text from the image accurately, "
            "preserving the layout and structure as much as possible."
        )

    def process(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        with_boxes: bool = False,
        **kwargs,
    ) -> TaskResult:
        """
        Extract text from an image using OCR.

        Args:
            image: Image path or PIL Image
            prompt: Optional custom prompt
            with_boxes: If True, also extract bounding boxes

        Returns:
            TaskResult with extracted text
        """
        img = self._load_image(image)

        if with_boxes:
            return self._process_with_boxes(img, prompt, **kwargs)
        else:
            return self._process_text_only(img, prompt, **kwargs)

    def _process_text_only(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        **kwargs,
    ) -> TaskResult:
        """Extract text without bounding boxes."""
        user_prompt = prompt or "Extract all text from this image. Preserve the layout and structure."

        messages = self._build_messages(image, user_prompt)
        response = self._generate(messages, **kwargs)

        return TaskResult(
            text=response,
            metadata={"mode": "text_only"},
        )

    def _process_with_boxes(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        **kwargs,
    ) -> TaskResult:
        """Extract text with bounding boxes."""
        user_prompt = prompt or (
            "Extract all text from this image with bounding box coordinates. "
            "Return the result as a JSON array where each item has 'text' and 'bbox' "
            "(with x1, y1, x2, y2 coordinates). Format:\n"
            '```json\n'
            '[{"text": "extracted text", "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 50}}]\n'
            '```'
        )

        messages = self._build_messages(image, user_prompt)
        response = self._generate(messages, **kwargs)

        # Parse bounding boxes
        boxes = parse_coordinates(response)

        # Create visualization
        vis_image = None
        if boxes:
            vis_image = draw_bounding_boxes(image, boxes)

        return TaskResult(
            text=response,
            bounding_boxes=boxes,
            visualization=vis_image,
            metadata={"mode": "with_boxes", "box_count": len(boxes)},
        )

    def extract_lines(
        self,
        image: Union[str, Image.Image],
        **kwargs,
    ) -> TaskResult:
        """
        Extract text line by line with coordinates.

        Args:
            image: Image path or PIL Image

        Returns:
            TaskResult with line-level text and boxes
        """
        img = self._load_image(image)

        prompt = (
            "Extract all text from this image line by line. "
            "For each line, provide the text and bounding box coordinates. "
            "Return as JSON array:\n"
            '```json\n'
            '[{"text": "line text", "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 20}}]\n'
            '```'
        )

        messages = self._build_messages(img, prompt)
        response = self._generate(messages, **kwargs)

        boxes = parse_coordinates(response)
        vis_image = draw_bounding_boxes(img, boxes) if boxes else None

        return TaskResult(
            text=response,
            bounding_boxes=boxes,
            visualization=vis_image,
            metadata={"mode": "lines", "line_count": len(boxes)},
        )


if __name__ == "__main__":
    print("=" * 60)
    print("OCR HANDLER TEST")
    print("=" * 60)
    print("  OCR handler registered successfully")
    print(f"  Task type: {TaskType.OCR}")
    print("  (Actual inference requires loaded model)")
    print("=" * 60)
