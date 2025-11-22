"""Visualization utilities for bounding boxes and annotations."""

import colorsys
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFont


# Color palette for bounding boxes
COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
    "#FFA500", "#800080", "#008000", "#000080", "#808000", "#800000",
    "#008080", "#FFC0CB", "#FFD700", "#ADFF2F", "#7FFFD4", "#D2691E",
    "#DC143C", "#00CED1", "#9400D3", "#FF1493", "#1E90FF",
]


def get_color(index: int) -> str:
    """Get a color from the palette by index."""
    return COLORS[index % len(COLORS)]


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def draw_bounding_box(
    image: Image.Image,
    bbox: Dict[str, int],
    label: Optional[str] = None,
    color: str = "#FF0000",
    width: int = 2,
    font_size: int = 12,
) -> Image.Image:
    """
    Draw a bounding box on an image.

    Args:
        image: PIL Image
        bbox: Dict with x1, y1, x2, y2 keys
        label: Optional label text
        color: Box color (hex)
        width: Line width
        font_size: Font size for label

    Returns:
        Image with bounding box drawn
    """
    # Make a copy to avoid modifying original
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Draw rectangle
    coords = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
    draw.rectangle(coords, outline=color, width=width)

    # Draw label if provided
    if label:
        font = _get_font(font_size)
        # Calculate text position
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Position above the box
        text_x = bbox["x1"]
        text_y = max(0, bbox["y1"] - text_height - 4)

        # Draw background for text
        draw.rectangle(
            [text_x, text_y, text_x + text_width + 4, text_y + text_height + 4],
            fill=color,
        )

        # Draw text
        draw.text((text_x + 2, text_y + 2), label, fill="white", font=font)

    return img


def draw_bounding_boxes(
    image: Image.Image,
    boxes: List[Dict[str, Any]],
    width: int = 2,
    font_size: int = 12,
) -> Image.Image:
    """
    Draw multiple bounding boxes on an image.

    Args:
        image: PIL Image
        boxes: List of dicts with 'bbox' and optional 'label' keys
        width: Line width
        font_size: Font size for labels

    Returns:
        Image with all bounding boxes drawn
    """
    img = image.copy()

    for i, box_data in enumerate(boxes):
        # Get bbox coordinates
        if "bbox" in box_data:
            bbox = box_data["bbox"]
        elif all(k in box_data for k in ["x1", "y1", "x2", "y2"]):
            bbox = box_data
        else:
            continue

        # Get label and color
        label = box_data.get("label")
        color = box_data.get("color", get_color(i))

        img = draw_bounding_box(
            img,
            bbox,
            label=label,
            color=color,
            width=width,
            font_size=font_size,
        )

    return img


def draw_point(
    image: Image.Image,
    x: int,
    y: int,
    radius: int = 10,
    color: str = "#00FF00",
    width: int = 2,
) -> Image.Image:
    """
    Draw a point (circle) on an image.

    Args:
        image: PIL Image
        x: X coordinate
        y: Y coordinate
        radius: Circle radius
        color: Circle color (hex)
        width: Line width

    Returns:
        Image with point drawn
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Draw circle
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        outline=color,
        width=width,
    )

    return img


def draw_points(
    image: Image.Image,
    points: List[Tuple[int, int]],
    radius: int = 10,
    color: str = "#00FF00",
    width: int = 2,
) -> Image.Image:
    """
    Draw multiple points on an image.

    Args:
        image: PIL Image
        points: List of (x, y) tuples
        radius: Circle radius
        color: Circle color (hex)
        width: Line width

    Returns:
        Image with points drawn
    """
    img = image.copy()

    for x, y in points:
        img = draw_point(img, x, y, radius=radius, color=color, width=width)

    return img


def draw_text_regions(
    image: Image.Image,
    regions: List[Dict[str, Any]],
    width: int = 1,
    font_size: int = 10,
) -> Image.Image:
    """
    Draw text regions with their content.

    Args:
        image: PIL Image
        regions: List of dicts with 'bbox' and 'text' keys
        width: Line width
        font_size: Font size

    Returns:
        Image with text regions drawn
    """
    img = image.copy()

    for i, region in enumerate(regions):
        bbox = region.get("bbox", region)
        text = region.get("text", "")

        # Use text as label, truncate if too long
        label = text[:20] + "..." if len(text) > 20 else text

        img = draw_bounding_box(
            img,
            bbox,
            label=label,
            color=get_color(i),
            width=width,
            font_size=font_size,
        )

    return img


def _get_font(size: int = 12) -> ImageFont.FreeTypeFont:
    """
    Get a font for drawing text.

    Args:
        size: Font size

    Returns:
        PIL ImageFont
    """
    # Try to load a good font
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]

    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue

    # Fall back to default
    return ImageFont.load_default()


def create_comparison_image(
    original: Image.Image,
    annotated: Image.Image,
    orientation: str = "horizontal",
) -> Image.Image:
    """
    Create a side-by-side or top-bottom comparison image.

    Args:
        original: Original image
        annotated: Annotated image
        orientation: 'horizontal' or 'vertical'

    Returns:
        Combined comparison image
    """
    if orientation == "horizontal":
        width = original.width + annotated.width
        height = max(original.height, annotated.height)
        combined = Image.new("RGB", (width, height))
        combined.paste(original, (0, 0))
        combined.paste(annotated, (original.width, 0))
    else:
        width = max(original.width, annotated.width)
        height = original.height + annotated.height
        combined = Image.new("RGB", (width, height))
        combined.paste(original, (0, 0))
        combined.paste(annotated, (0, original.height))

    return combined


if __name__ == "__main__":
    print("=" * 60)
    print("VISUALIZATION TEST")
    print("=" * 60)

    # Create test image
    img = Image.new("RGB", (400, 300), color="white")

    # Test bounding box
    bbox = {"x1": 50, "y1": 50, "x2": 150, "y2": 150}
    img = draw_bounding_box(img, bbox, label="Test", color="#FF0000")
    print("  ✓ Single bounding box drawn")

    # Test multiple boxes
    boxes = [
        {"bbox": {"x1": 200, "y1": 50, "x2": 350, "y2": 150}, "label": "Box 1"},
        {"bbox": {"x1": 50, "y1": 200, "x2": 150, "y2": 280}, "label": "Box 2"},
    ]
    img = draw_bounding_boxes(img, boxes)
    print("  ✓ Multiple bounding boxes drawn")

    # Test point
    img = draw_point(img, 300, 240, color="#00FF00")
    print("  ✓ Point drawn")

    # Test color palette
    for i in range(5):
        color = get_color(i)
        print(f"  Color {i}: {color}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
