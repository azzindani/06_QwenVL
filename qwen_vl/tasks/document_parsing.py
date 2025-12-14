"""Document parsing task handler for HTML structure extraction."""

import logging
import re
from typing import Any, Dict, List, Optional, Union

from PIL import Image, ImageDraw, ImageFont

from .base import BaseTaskHandler, TaskResult, TaskType, register_handler

logger = logging.getLogger(__name__)


def parse_html_bboxes(html_content: str) -> List[Dict[str, Any]]:
    """
    Parse bounding boxes from HTML data-bbox attributes.
    
    Args:
        html_content: HTML string with data-bbox attributes
        
    Returns:
        List of bounding box dictionaries
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    soup = BeautifulSoup(html_content, "html.parser")
    elements = soup.find_all(attrs={"data-bbox": True})

    bboxes = []
    for el in elements:
        # Skip ol tags, include li tags within ol
        if el.name == "ol":
            continue

        bbox_str = el.get("data-bbox", "")
        text = el.get_text(strip=True)
        
        try:
            coords = list(map(int, bbox_str.split()))
            if len(coords) == 4:
                bboxes.append({
                    "text": text,
                    "bbox": coords,
                    "tag": el.name,
                })
        except ValueError:
            continue

    return bboxes


def draw_document_boxes(
    image: Image.Image,
    bboxes: List[Dict[str, Any]],
    input_width: int,
    input_height: int,
) -> Image.Image:
    """
    Draw bounding boxes from parsed HTML on an image.
    
    Args:
        image: PIL Image to draw on
        bboxes: List of bbox dictionaries from parse_html_bboxes
        input_width: Model input width for scaling
        input_height: Model input height for scaling
        
    Returns:
        Image with drawn bounding boxes
    """
    img = image.copy()
    width, height = img.size
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", size=12)
    except Exception:
        font = ImageFont.load_default()

    scale_x = input_width / width
    scale_y = input_height / height

    for bbox_data in bboxes:
        coords = bbox_data["bbox"]
        text = bbox_data.get("text", "")[:30]  # Truncate long text

        # Scale coordinates
        x1 = int(coords[0] / scale_x)
        y1 = int(coords[1] / scale_y)
        x2 = int(coords[2] / scale_x)
        y2 = int(coords[3] / scale_y)

        # Ensure proper ordering
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Draw
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        if text:
            draw.text((x1, y2 + 2), text, fill="black", font=font)

    return img


def clean_html_output(html_content: str) -> str:
    """
    Clean and format HTML output from document parsing.
    
    Args:
        html_content: Raw HTML from model
        
    Returns:
        Cleaned HTML string
    """
    try:
        from bs4 import BeautifulSoup, Tag
    except ImportError:
        return html_content

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove color styles
    color_pattern = re.compile(r"\bcolor:[^;]+;?")
    for tag in soup.find_all(style=True):
        original_style = tag.get("style", "")
        new_style = color_pattern.sub("", original_style)
        if not new_style.strip():
            del tag["style"]
        else:
            tag["style"] = new_style.rstrip(";")

    # Remove data attributes
    for attr in ["data-bbox", "data-polygon"]:
        for tag in soup.find_all(attrs={attr: True}):
            del tag[attr]

    return str(soup)


@register_handler(TaskType.DOCUMENT_PARSING)
class DocumentParsingHandler(BaseTaskHandler):
    """Handler for document parsing with HTML structure extraction."""

    @property
    def task_type(self) -> TaskType:
        return TaskType.DOCUMENT_PARSING

    @property
    def system_prompt(self) -> str:
        return (
            "You are an AI specialized in recognizing and extracting text from images. "
            "Your mission is to analyze the image document and generate the result in "
            "QwenVL Document Parser HTML format using specified tags while maintaining "
            "user privacy and data integrity."
        )

    def process(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        output_format: str = "html",
        **kwargs,
    ) -> TaskResult:
        """
        Parse a document image into structured HTML.

        Args:
            image: Image path or PIL Image
            prompt: Custom prompt (optional)
            output_format: 'html' or 'qwenvl_html'

        Returns:
            TaskResult with parsed document structure
        """
        img = self._load_image(image)

        if output_format == "qwenvl_html":
            user_prompt = prompt or "QwenVL HTML"
        else:
            user_prompt = prompt or "Parse this document into structured HTML."

        messages = self._build_messages(img, user_prompt)
        response = self._generate(messages, **kwargs)

        # Parse bboxes from response
        bboxes = parse_html_bboxes(response)

        return TaskResult(
            text=response,
            data={"bboxes": bboxes, "format": output_format},
            metadata={
                "mode": "document_parsing",
                "output_format": output_format,
                "element_count": len(bboxes),
            },
        )

    def parse_to_html(
        self,
        image: Union[str, Image.Image],
        clean_output: bool = True,
        **kwargs,
    ) -> TaskResult:
        """
        Parse document to standard HTML.

        Args:
            image: Image path or PIL Image
            clean_output: Whether to clean the HTML output

        Returns:
            TaskResult with HTML content
        """
        result = self.process(image, output_format="html", **kwargs)

        if clean_output and result.text:
            result.text = clean_html_output(result.text)

        return result

    def parse_with_layout(
        self,
        image: Union[str, Image.Image],
        **kwargs,
    ) -> TaskResult:
        """
        Parse document with QwenVL HTML format including bounding boxes.

        Args:
            image: Image path or PIL Image

        Returns:
            TaskResult with QwenVL HTML and bbox data
        """
        return self.process(image, output_format="qwenvl_html", **kwargs)


if __name__ == "__main__":
    print("=" * 60)
    print("DOCUMENT PARSING HANDLER TEST")
    print("=" * 60)
    print("  Document parsing handler registered successfully")
    print(f"  Task type: {TaskType.DOCUMENT_PARSING}")
    print("  Note: Best results with BeautifulSoup installed")
    print("  (Actual inference requires loaded model)")
    print("=" * 60)
