"""Spatial understanding task handler for object detection and localization."""

import ast
import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageColor, ImageDraw, ImageFont

from ..utils.parsers import parse_json_from_markdown, parse_spatial_output
from .base import BaseTaskHandler, TaskResult, TaskType, register_handler

logger = logging.getLogger(__name__)


def decode_xml_points(text: str) -> Optional[Dict[str, Any]]:
    """
    Decode XML format point coordinates.
    
    Args:
        text: XML string with point coordinates
        
    Returns:
        Dictionary with points, alt, and phrase, or None if parsing fails
    """
    try:
        # Clean up XML markers
        xml_text = text.replace("```xml", "").replace("```", "").strip()
        root = ET.fromstring(xml_text)
        num_points = (len(root.attrib) - 1) // 2
        points = []
        for i in range(num_points):
            x = root.attrib.get(f"x{i+1}") or root.attrib.get("x")
            y = root.attrib.get(f"y{i+1}") or root.attrib.get("y")
            if x and y:
                points.append([x, y])
        alt = root.attrib.get("alt")
        phrase = root.text.strip() if root.text else None
        return {
            "points": points,
            "alt": alt,
            "phrase": phrase,
        }
    except Exception:
        return None


def draw_spatial_boxes(
    image: Image.Image,
    bounding_boxes: str,
    input_width: int,
    input_height: int,
) -> Image.Image:
    """
    Draw bounding boxes on an image with labels.
    
    Args:
        image: PIL Image to draw on
        bounding_boxes: JSON string with bounding box data
        input_width: Model input width for coordinate scaling
        input_height: Model input height for coordinate scaling
        
    Returns:
        Image with drawn bounding boxes
    """
    img = image.copy()
    actual_width, actual_height = img.size
    draw = ImageDraw.Draw(img)

    colors = [
        "red", "green", "blue", "yellow", "orange", "pink", "purple",
        "brown", "cyan", "magenta", "lime", "navy", "teal", "coral",
    ]

    # Parse boxes using robust utility
    boxes = parse_spatial_output(bounding_boxes)
    
    # Check if dimensions are nearly identical (within 2%) - skip scaling
    width_ratio = abs(input_width - actual_width) / max(input_width, actual_width) if input_width else 1
    height_ratio = abs(input_height - actual_height) / max(input_height, actual_height) if input_height else 1
    skip_scaling = width_ratio < 0.02 and height_ratio < 0.02
    
    print(f"[SPATIAL] Image: {actual_width}x{actual_height}, Model: {input_width}x{input_height}, Skip scaling: {skip_scaling}")

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except Exception:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        if "bbox_2d" not in box:
            continue
            
        color = colors[i % len(colors)]
        bbox = box["bbox_2d"]

        # Convert coordinates
        if skip_scaling:
            abs_x1, abs_y1, abs_x2, abs_y2 = [int(b) for b in bbox]
            print(f"[SPATIAL] No scaling: raw={bbox}, used=({abs_x1},{abs_y1},{abs_x2},{abs_y2})")
        else:
            abs_x1 = int(bbox[0] / input_width * actual_width)
            abs_y1 = int(bbox[1] / input_height * actual_height)
            abs_x2 = int(bbox[2] / input_width * actual_width)
            abs_y2 = int(bbox[3] / input_height * actual_height)
            print(f"[SPATIAL] Scaling: raw={bbox}, scaled=({abs_x1},{abs_y1},{abs_x2},{abs_y2})")

        # Ensure proper ordering
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        # Draw rectangle
        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=3)

        # Draw label if present
        if "label" in box:
            draw.text((abs_x1 + 5, abs_y1 + 5), box["label"], fill=color, font=font)

    return img


def draw_points(
    image: Image.Image,
    points_data: Dict[str, Any],
    input_width: int,
    input_height: int,
) -> Image.Image:
    """
    Draw points on an image.
    
    Args:
        image: PIL Image to draw on
        points_data: Dictionary with points and description
        input_width: Model input width for coordinate scaling
        input_height: Model input height for coordinate scaling
        
    Returns:
        Image with drawn points
    """
    img = image.copy()
    actual_width, actual_height = img.size
    draw = ImageDraw.Draw(img)

    colors = ["red", "green", "blue", "yellow", "orange", "pink", "purple"]

    points = points_data.get("points", [])
    description = points_data.get("phrase", "")
    
    # Check if dimensions are nearly identical (within 2%) - skip scaling
    width_ratio = abs(input_width - actual_width) / max(input_width, actual_width) if input_width else 1
    height_ratio = abs(input_height - actual_height) / max(input_height, actual_height) if input_height else 1
    skip_scaling = width_ratio < 0.02 and height_ratio < 0.02

    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except Exception:
        font = ImageFont.load_default()

    for i, point in enumerate(points):
        color = colors[i % len(colors)]
        raw_x = float(point[0])
        raw_y = float(point[1])
        
        if skip_scaling:
            abs_x = int(raw_x)
            abs_y = int(raw_y)
        else:
            abs_x = int(raw_x / input_width * actual_width)
            abs_y = int(raw_y / input_height * actual_height)
        
        radius = 5

        draw.ellipse(
            [(abs_x - radius, abs_y - radius), (abs_x + radius, abs_y + radius)],
            fill=color,
        )
        if description:
            draw.text((abs_x + 8, abs_y + 6), description, fill=color, font=font)

    return img


@register_handler(TaskType.SPATIAL)
class SpatialHandler(BaseTaskHandler):
    """Handler for spatial understanding tasks (object detection, localization)."""

    @property
    def task_type(self) -> TaskType:
        return TaskType.SPATIAL

    @property
    def system_prompt(self) -> str:
        return (
            "You are a helpful assistant specialized in spatial understanding and object detection. "
            "Analyze images to locate and identify objects with precise bounding box coordinates. "
            "Return coordinates in the requested format (JSON or XML)."
        )

    def process(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        **kwargs,
    ) -> TaskResult:
        """
        Detect and locate objects in an image.

        Args:
            image: Image path or PIL Image
            prompt: Custom prompt for detection

        Returns:
            TaskResult with detection results and visualization
        """
        img = self._load_image(image)

        user_prompt = prompt or (
            "Detect all objects in this image and provide their bounding box coordinates "
            "in JSON format: [{\"label\": \"object\", \"bbox_2d\": [x1, y1, x2, y2]}]"
        )

        # Set default pixel limits to avoid OOM for large images
        min_pixels = kwargs.get("min_pixels", 512 * 28 * 28)
        max_pixels = kwargs.get("max_pixels", 1024 * 28 * 28)

        messages = self._build_messages(
            img, 
            user_prompt,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        response = self._generate(messages, **kwargs)

        # Visualize if JSON boxes found
        vis_image = None
        if self.last_input_width and self.last_input_height:
            vis_image = draw_spatial_boxes(img, response, self.last_input_width, self.last_input_height)

        return TaskResult(
            text=response,
            visualization=vis_image,
            metadata={"mode": "detection", "input_width": self.last_input_width, "input_height": self.last_input_height},
        )

    def detect_objects(
        self,
        image: Union[str, Image.Image],
        object_type: Optional[str] = None,
        **kwargs,
    ) -> TaskResult:
        """
        Detect specific objects with bounding boxes.

        Args:
            image: Image path or PIL Image
            object_type: Specific object to detect (optional)

        Returns:
            TaskResult with bounding boxes and visualization
        """
        img = self._load_image(image)

        if object_type:
            prompt = (
                f"Locate all {object_type} in this image. "
                f"Output the bounding box coordinates in JSON format: "
                f'[{{"label": "{object_type}", "bbox_2d": [x1, y1, x2, y2]}}]'
            )
        else:
            prompt = (
                "Detect all objects in this image with their bounding boxes. "
                'Output in JSON format: [{"label": "object", "bbox_2d": [x1, y1, x2, y2]}]'
            )

        # Set default pixel limits to avoid OOM for large images
        min_pixels = kwargs.get("min_pixels", 512 * 28 * 28)
        max_pixels = kwargs.get("max_pixels", 1024 * 28 * 28)

        messages = self._build_messages(
            img, 
            prompt,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        response = self._generate(messages, **kwargs)

        # Visualize if JSON boxes found
        vis_image = None
        if self.last_input_width and self.last_input_height:
            vis_image = draw_spatial_boxes(img, response, self.last_input_width, self.last_input_height)

        return TaskResult(
            text=response,
            visualization=vis_image,
            metadata={
                "mode": "object_detection", 
                "object_type": object_type,
                "input_width": self.last_input_width,
                "input_height": self.last_input_height
            },
        )

    def point_to_object(
        self,
        image: Union[str, Image.Image],
        object_description: str,
        **kwargs,
    ) -> TaskResult:
        """
        Point to a specific object in the image.

        Args:
            image: Image path or PIL Image
            object_description: Description of object to locate

        Returns:
            TaskResult with point coordinates
        """
        img = self._load_image(image)

        prompt = (
            f"Point to {object_description} in this image. "
            f"Output the coordinates in XML format: <points x1=\"X\" y1=\"Y\">{object_description}</points>"
        )

        # Set default pixel limits to avoid OOM for large images
        min_pixels = kwargs.get("min_pixels", 512 * 28 * 28)
        max_pixels = kwargs.get("max_pixels", 1024 * 28 * 28)

        messages = self._build_messages(
            img, 
            prompt,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        response = self._generate(messages, **kwargs)

        # Parse XML response
        points_data = decode_xml_points(response)
        
        # Visualize points
        vis_image = None
        if points_data and self.last_input_width and self.last_input_height:
            vis_image = draw_points(img, points_data, self.last_input_width, self.last_input_height)

        return TaskResult(
            text=response,
            data=points_data,
            visualization=vis_image,
            metadata={
                "mode": "point_to_object", 
                "target": object_description,
                "input_width": self.last_input_width,
                "input_height": self.last_input_height
            },
        )


if __name__ == "__main__":
    print("=" * 60)
    print("SPATIAL HANDLER TEST")
    print("=" * 60)
    print("  Spatial handler registered successfully")
    print(f"  Task type: {TaskType.SPATIAL}")
    print("  (Actual inference requires loaded model)")
    print("=" * 60)
