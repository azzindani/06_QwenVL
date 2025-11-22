"""Unit tests for parser utilities."""

import pytest

from qwen_vl.utils.parsers import (
    clean_html,
    extract_key_value_pairs,
    parse_bounding_box,
    parse_coordinates,
    parse_json_array_from_markdown,
    parse_json_from_markdown,
    parse_xml_points,
)


@pytest.mark.unit
class TestParseJsonFromMarkdown:
    """Tests for parse_json_from_markdown function."""

    def test_json_in_code_block(self):
        """Test parsing JSON from markdown code block."""
        text = '''Here is the result:
        ```json
        {"name": "test", "value": 123}
        ```
        '''
        result = parse_json_from_markdown(text)
        assert result == {"name": "test", "value": 123}

    def test_json_in_plain_code_block(self):
        """Test parsing JSON from plain code block."""
        text = '''
        ```
        {"key": "value"}
        ```
        '''
        result = parse_json_from_markdown(text)
        assert result == {"key": "value"}

    def test_raw_json(self):
        """Test parsing raw JSON without code block."""
        text = 'The result is {"x": 1, "y": 2}'
        result = parse_json_from_markdown(text)
        assert result == {"x": 1, "y": 2}

    def test_no_json(self):
        """Test with text containing no JSON."""
        text = "This is just plain text"
        result = parse_json_from_markdown(text)
        assert result is None

    def test_invalid_json(self):
        """Test with invalid JSON."""
        text = '{"invalid": json}'
        result = parse_json_from_markdown(text)
        assert result is None

    def test_nested_json(self):
        """Test parsing nested JSON."""
        text = '''```json
        {"outer": {"inner": "value"}}
        ```'''
        result = parse_json_from_markdown(text)
        assert result == {"outer": {"inner": "value"}}


@pytest.mark.unit
class TestParseJsonArrayFromMarkdown:
    """Tests for parse_json_array_from_markdown function."""

    def test_array_in_code_block(self):
        """Test parsing JSON array from code block."""
        text = '''```json
        [{"a": 1}, {"b": 2}]
        ```'''
        result = parse_json_array_from_markdown(text)
        assert result == [{"a": 1}, {"b": 2}]

    def test_raw_array(self):
        """Test parsing raw JSON array."""
        text = 'Results: [1, 2, 3]'
        result = parse_json_array_from_markdown(text)
        assert result == [1, 2, 3]

    def test_no_array(self):
        """Test with text containing no array."""
        text = "Just text"
        result = parse_json_array_from_markdown(text)
        assert result is None


@pytest.mark.unit
class TestParseBoundingBox:
    """Tests for parse_bounding_box function."""

    def test_list_format(self):
        """Test parsing [x1, y1, x2, y2] format."""
        result = parse_bounding_box("[10, 20, 100, 200]")
        assert result == {"x1": 10, "y1": 20, "x2": 100, "y2": 200}

    def test_tuple_format(self):
        """Test parsing (x1, y1, x2, y2) format."""
        result = parse_bounding_box("(50, 60, 150, 160)")
        assert result == {"x1": 50, "y1": 60, "x2": 150, "y2": 160}

    def test_dict_format(self):
        """Test parsing dict format."""
        result = parse_bounding_box('{"x1": 0, "y1": 0, "x2": 100, "y2": 100}')
        assert result == {"x1": 0, "y1": 0, "x2": 100, "y2": 100}

    def test_comma_separated(self):
        """Test parsing comma-separated values."""
        result = parse_bounding_box("10, 20, 30, 40")
        assert result == {"x1": 10, "y1": 20, "x2": 30, "y2": 40}

    def test_invalid_format(self):
        """Test with invalid format."""
        result = parse_bounding_box("not a bbox")
        assert result is None


@pytest.mark.unit
class TestParseCoordinates:
    """Tests for parse_coordinates function."""

    def test_json_array(self):
        """Test parsing coordinates from JSON array."""
        text = '''```json
        [
            {"label": "text1", "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 50}},
            {"label": "text2", "bbox": {"x1": 0, "y1": 60, "x2": 100, "y2": 110}}
        ]
        ```'''
        result = parse_coordinates(text)
        assert len(result) == 2
        assert result[0]["label"] == "text1"
        assert result[1]["label"] == "text2"

    def test_labeled_boxes(self):
        """Test parsing labeled bounding boxes."""
        text = '"Title": [10, 20, 200, 50]\n"Body": [10, 60, 200, 300]'
        result = parse_coordinates(text)
        assert len(result) == 2
        labels = [r["label"] for r in result]
        assert "Title" in labels
        assert "Body" in labels


@pytest.mark.unit
class TestParseXmlPoints:
    """Tests for parse_xml_points function."""

    def test_single_point(self):
        """Test parsing single point."""
        text = '<point x="100" y="200"/>'
        result = parse_xml_points(text)
        assert result == [(100, 200)]

    def test_multiple_points(self):
        """Test parsing multiple points."""
        text = '<point x="10" y="20"/><point x="30" y="40"/>'
        result = parse_xml_points(text)
        assert result == [(10, 20), (30, 40)]

    def test_no_points(self):
        """Test with no points."""
        text = "No points here"
        result = parse_xml_points(text)
        assert result == []


@pytest.mark.unit
class TestCleanHtml:
    """Tests for clean_html function."""

    def test_remove_colors(self):
        """Test removing color styles."""
        html = '<div style="color: red; font-size: 12px;">Text</div>'
        result = clean_html(html)
        assert "color:" not in result
        assert "font-size" in result

    def test_remove_background_color(self):
        """Test removing background-color."""
        html = '<span style="background-color: #fff;">Text</span>'
        result = clean_html(html)
        assert "background-color" not in result

    def test_remove_empty_style(self):
        """Test removing empty style attributes."""
        html = '<div style="">Text</div>'
        result = clean_html(html)
        assert 'style=""' not in result


@pytest.mark.unit
class TestExtractKeyValuePairs:
    """Tests for extract_key_value_pairs function."""

    def test_colon_format(self):
        """Test key: value format."""
        text = "Name: John\nAge: 30"
        result = extract_key_value_pairs(text)
        assert result["Name"] == "John"
        assert result["Age"] == "30"

    def test_equals_format(self):
        """Test key = value format."""
        text = "x = 100\ny = 200"
        result = extract_key_value_pairs(text)
        assert result["x"] == "100"
        assert result["y"] == "200"

    def test_json_format(self):
        """Test JSON format."""
        text = '```json\n{"a": "1", "b": "2"}\n```'
        result = extract_key_value_pairs(text)
        assert result["a"] == "1"
        assert result["b"] == "2"

    def test_quoted_format(self):
        """Test "key": "value" format."""
        text = '"field1": "value1"\n"field2": "value2"'
        result = extract_key_value_pairs(text)
        assert result["field1"] == "value1"
        assert result["field2"] == "value2"
