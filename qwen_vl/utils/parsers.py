"""Parser utilities for extracting structured data from model outputs."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple


def parse_json_from_markdown(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from markdown code blocks or raw text.

    Args:
        text: Text that may contain JSON in markdown blocks

    Returns:
        Parsed JSON dict or None if not found
    """
    # Try to find JSON in markdown code blocks
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
        r"\{[\s\S]*\}",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                # Clean up the match
                json_str = match.strip()
                if not json_str.startswith("{"):
                    # Find the first { and last }
                    start = json_str.find("{")
                    end = json_str.rfind("}") + 1
                    if start != -1 and end > start:
                        json_str = json_str[start:end]

                return json.loads(json_str)
            except json.JSONDecodeError:
                continue

    return None


def parse_json_array_from_markdown(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Extract JSON array from markdown code blocks or raw text.

    Args:
        text: Text that may contain JSON array

    Returns:
        Parsed JSON list or None if not found
    """
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
        r"\[[\s\S]*\]",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                json_str = match.strip()
                if not json_str.startswith("["):
                    start = json_str.find("[")
                    end = json_str.rfind("]") + 1
                    if start != -1 and end > start:
                        json_str = json_str[start:end]

                result = json.loads(json_str)
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                continue

    return None


def parse_bounding_box(text: str) -> Optional[Dict[str, int]]:
    """
    Parse a single bounding box from text.

    Supports formats:
    - [x1, y1, x2, y2]
    - {"x1": 0, "y1": 0, "x2": 100, "y2": 100}
    - (x1, y1, x2, y2)

    Args:
        text: Text containing bounding box

    Returns:
        Dict with x1, y1, x2, y2 keys or None
    """
    # Try JSON format
    try:
        data = json.loads(text)
        if isinstance(data, dict) and all(k in data for k in ["x1", "y1", "x2", "y2"]):
            return {k: int(data[k]) for k in ["x1", "y1", "x2", "y2"]}
        if isinstance(data, list) and len(data) == 4:
            return {"x1": int(data[0]), "y1": int(data[1]), "x2": int(data[2]), "y2": int(data[3])}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Try regex patterns
    patterns = [
        r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]",
        r"\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)",
        r"(\d+),\s*(\d+),\s*(\d+),\s*(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return {
                "x1": int(match.group(1)),
                "y1": int(match.group(2)),
                "x2": int(match.group(3)),
                "y2": int(match.group(4)),
            }

    return None


def parse_coordinates(text: str) -> List[Dict[str, Any]]:
    """
    Parse multiple bounding boxes with labels from text.

    Args:
        text: Text containing bounding boxes

    Returns:
        List of dicts with 'bbox' and optional 'label' keys
    """
    results = []

    # Try to parse as JSON first
    json_data = parse_json_from_markdown(text)
    if json_data:
        if isinstance(json_data, dict):
            # Single item
            if "bbox" in json_data or all(k in json_data for k in ["x1", "y1", "x2", "y2"]):
                results.append(json_data)
        return results

    json_array = parse_json_array_from_markdown(text)
    if json_array:
        return json_array

    # Try to find labeled bounding boxes in text
    # Pattern: "label": [x1, y1, x2, y2] or label: (x1, y1, x2, y2)
    pattern = r'"?([^":\[\]]+)"?\s*:\s*[\[\(](\d+),\s*(\d+),\s*(\d+),\s*(\d+)[\]\)]'
    matches = re.findall(pattern, text)

    for match in matches:
        results.append({
            "label": match[0].strip(),
            "bbox": {
                "x1": int(match[1]),
                "y1": int(match[2]),
                "x2": int(match[3]),
                "y2": int(match[4]),
            }
        })

    return results


def parse_xml_points(text: str) -> List[Tuple[int, int]]:
    """
    Parse point coordinates from XML-like format.

    Args:
        text: Text containing points like <point x="100" y="200"/>

    Returns:
        List of (x, y) tuples
    """
    points = []
    pattern = r'<point\s+x="(\d+)"\s+y="(\d+)"\s*/>'
    matches = re.findall(pattern, text)

    for match in matches:
        points.append((int(match[0]), int(match[1])))

    return points


def clean_html(html: str) -> str:
    """
    Clean and format HTML output from model.

    Args:
        html: Raw HTML string

    Returns:
        Cleaned HTML string
    """
    # Remove color styles
    html = re.sub(r'color:\s*[^;]+;?', '', html)
    html = re.sub(r'background-color:\s*[^;]+;?', '', html)

    # Remove empty style attributes
    html = re.sub(r'\s*style="\s*"', '', html)

    # Clean up whitespace
    html = re.sub(r'\n\s*\n', '\n', html)
    html = html.strip()

    return html


def extract_key_value_pairs(text: str) -> Dict[str, str]:
    """
    Extract key-value pairs from text.

    Supports formats:
    - key: value
    - key = value
    - "key": "value"

    Args:
        text: Text containing key-value pairs

    Returns:
        Dict of extracted pairs
    """
    pairs = {}

    # Try JSON first
    json_data = parse_json_from_markdown(text)
    if json_data and isinstance(json_data, dict):
        return {str(k): str(v) for k, v in json_data.items()}

    # Pattern matching
    patterns = [
        r'"([^"]+)"\s*:\s*"([^"]*)"',  # "key": "value"
        r'([^:\n=]+)\s*[:=]\s*([^\n]+)',  # key: value or key = value
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for key, value in matches:
            key = key.strip().strip('"')
            value = value.strip().strip('"')
            if key and key not in pairs:
                pairs[key] = value

    return pairs


if __name__ == "__main__":
    print("=" * 60)
    print("PARSER TEST")
    print("=" * 60)

    # Test JSON parsing
    test_text = '''Here is the result:
    ```json
    {"name": "test", "value": 123}
    ```
    '''
    result = parse_json_from_markdown(test_text)
    print(f"  JSON parsing: {result}")
    assert result == {"name": "test", "value": 123}

    # Test bounding box parsing
    bbox = parse_bounding_box("[10, 20, 100, 200]")
    print(f"  Bbox parsing: {bbox}")
    assert bbox == {"x1": 10, "y1": 20, "x2": 100, "y2": 200}

    # Test key-value extraction
    kv_text = 'Name: John\nAge: 30\nCity: NYC'
    pairs = extract_key_value_pairs(kv_text)
    print(f"  Key-value pairs: {pairs}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
