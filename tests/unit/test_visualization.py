"""Unit tests for visualization utilities."""

import pytest
from PIL import Image

from qwen_vl.utils.visualization import (
    create_comparison_image,
    draw_bounding_box,
    draw_bounding_boxes,
    draw_point,
    draw_points,
    draw_text_regions,
    get_color,
    hex_to_rgb,
)


@pytest.fixture
def test_image():
    """Create a test image."""
    return Image.new("RGB", (400, 300), color="white")


@pytest.mark.unit
class TestGetColor:
    """Tests for get_color function."""

    def test_get_first_colors(self):
        """Test getting first few colors."""
        colors = [get_color(i) for i in range(5)]
        assert all(c.startswith("#") for c in colors)
        assert len(set(colors)) == 5  # All different

    def test_color_wrapping(self):
        """Test that colors wrap around."""
        # There are 23 colors, so index 23 should wrap to 0
        assert get_color(0) == get_color(23)
        assert get_color(1) == get_color(24)


@pytest.mark.unit
class TestHexToRgb:
    """Tests for hex_to_rgb function."""

    def test_basic_colors(self):
        """Test basic color conversion."""
        assert hex_to_rgb("#FF0000") == (255, 0, 0)
        assert hex_to_rgb("#00FF00") == (0, 255, 0)
        assert hex_to_rgb("#0000FF") == (0, 0, 255)

    def test_without_hash(self):
        """Test without # prefix."""
        assert hex_to_rgb("FFFFFF") == (255, 255, 255)
        assert hex_to_rgb("000000") == (0, 0, 0)


@pytest.mark.unit
class TestDrawBoundingBox:
    """Tests for draw_bounding_box function."""

    def test_basic_box(self, test_image):
        """Test drawing a basic bounding box."""
        bbox = {"x1": 50, "y1": 50, "x2": 150, "y2": 150}
        result = draw_bounding_box(test_image, bbox)

        # Should return a new image
        assert result is not test_image
        assert result.size == test_image.size

    def test_with_label(self, test_image):
        """Test drawing box with label."""
        bbox = {"x1": 50, "y1": 50, "x2": 150, "y2": 150}
        result = draw_bounding_box(test_image, bbox, label="Test")

        assert result is not test_image

    def test_custom_color(self, test_image):
        """Test drawing with custom color."""
        bbox = {"x1": 50, "y1": 50, "x2": 150, "y2": 150}
        result = draw_bounding_box(test_image, bbox, color="#00FF00")

        assert result is not test_image


@pytest.mark.unit
class TestDrawBoundingBoxes:
    """Tests for draw_bounding_boxes function."""

    def test_multiple_boxes(self, test_image):
        """Test drawing multiple boxes."""
        boxes = [
            {"bbox": {"x1": 50, "y1": 50, "x2": 150, "y2": 100}, "label": "Box 1"},
            {"bbox": {"x1": 200, "y1": 50, "x2": 350, "y2": 100}, "label": "Box 2"},
        ]
        result = draw_bounding_boxes(test_image, boxes)

        assert result is not test_image
        assert result.size == test_image.size

    def test_flat_bbox_format(self, test_image):
        """Test with flat bbox format (x1, y1, x2, y2 at top level)."""
        boxes = [
            {"x1": 50, "y1": 50, "x2": 150, "y2": 100, "label": "Direct"},
        ]
        result = draw_bounding_boxes(test_image, boxes)

        assert result is not test_image

    def test_empty_boxes(self, test_image):
        """Test with empty box list."""
        result = draw_bounding_boxes(test_image, [])

        # Should still return a copy
        assert result is not test_image


@pytest.mark.unit
class TestDrawPoint:
    """Tests for draw_point function."""

    def test_basic_point(self, test_image):
        """Test drawing a basic point."""
        result = draw_point(test_image, 100, 150)

        assert result is not test_image
        assert result.size == test_image.size

    def test_custom_radius(self, test_image):
        """Test with custom radius."""
        result = draw_point(test_image, 100, 150, radius=20)

        assert result is not test_image


@pytest.mark.unit
class TestDrawPoints:
    """Tests for draw_points function."""

    def test_multiple_points(self, test_image):
        """Test drawing multiple points."""
        points = [(100, 100), (200, 150), (300, 200)]
        result = draw_points(test_image, points)

        assert result is not test_image

    def test_empty_points(self, test_image):
        """Test with empty points list."""
        result = draw_points(test_image, [])

        assert result is not test_image


@pytest.mark.unit
class TestDrawTextRegions:
    """Tests for draw_text_regions function."""

    def test_text_regions(self, test_image):
        """Test drawing text regions."""
        regions = [
            {"bbox": {"x1": 10, "y1": 10, "x2": 200, "y2": 50}, "text": "Hello"},
            {"bbox": {"x1": 10, "y1": 60, "x2": 200, "y2": 100}, "text": "World"},
        ]
        result = draw_text_regions(test_image, regions)

        assert result is not test_image

    def test_truncate_long_text(self, test_image):
        """Test that long text is truncated."""
        regions = [
            {
                "bbox": {"x1": 10, "y1": 10, "x2": 200, "y2": 50},
                "text": "This is a very long text that should be truncated",
            },
        ]
        result = draw_text_regions(test_image, regions)

        assert result is not test_image


@pytest.mark.unit
class TestCreateComparisonImage:
    """Tests for create_comparison_image function."""

    def test_horizontal_comparison(self, test_image):
        """Test horizontal comparison."""
        annotated = test_image.copy()
        result = create_comparison_image(test_image, annotated, "horizontal")

        assert result.width == test_image.width * 2
        assert result.height == test_image.height

    def test_vertical_comparison(self, test_image):
        """Test vertical comparison."""
        annotated = test_image.copy()
        result = create_comparison_image(test_image, annotated, "vertical")

        assert result.width == test_image.width
        assert result.height == test_image.height * 2
