"""Unit tests for export functionality."""

import json
import pytest

from qwen_vl.api.export import (
    export_to_json,
    export_to_csv,
    ExportManager,
    get_export_manager,
)


@pytest.mark.unit
class TestExportToJson:
    """Tests for JSON export."""

    def test_export_dict(self):
        """Test exporting dictionary."""
        data = {"key": "value", "number": 42}
        result = export_to_json(data)

        parsed = json.loads(result)
        assert parsed == data

    def test_export_list(self):
        """Test exporting list."""
        data = [{"a": 1}, {"a": 2}]
        result = export_to_json(data)

        parsed = json.loads(result)
        assert parsed == data

    def test_export_pretty(self):
        """Test pretty printing."""
        data = {"key": "value"}

        pretty = export_to_json(data, pretty=True)
        compact = export_to_json(data, pretty=False)

        assert len(pretty) > len(compact)
        assert "\n" in pretty
        assert "\n" not in compact


@pytest.mark.unit
class TestExportToCsv:
    """Tests for CSV export."""

    def test_export_simple_list(self):
        """Test exporting simple list of dicts."""
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]

        result = export_to_csv(data)
        lines = result.strip().split("\n")

        assert len(lines) == 3  # Header + 2 rows
        assert "name" in lines[0]
        assert "Alice" in lines[1]

    def test_export_with_columns(self):
        """Test exporting with specific column order."""
        data = [
            {"a": 1, "b": 2, "c": 3},
        ]

        result = export_to_csv(data, columns=["c", "a"])
        lines = result.strip().replace("\r", "").split("\n")

        assert lines[0] == "c,a"

    def test_export_empty_list(self):
        """Test exporting empty list."""
        result = export_to_csv([])
        assert result == ""

    def test_export_nested_values(self):
        """Test nested values are JSON-encoded."""
        data = [
            {"name": "test", "data": {"nested": "value"}},
        ]

        result = export_to_csv(data)
        # CSV escapes quotes, so check for the key content
        assert "nested" in result
        assert "value" in result


@pytest.mark.unit
class TestExportManager:
    """Tests for ExportManager."""

    def test_export_json(self):
        """Test JSON export through manager."""
        manager = ExportManager()
        data = {"test": "value"}

        result = manager.export(data, "json")
        assert json.loads(result) == data

    def test_export_csv(self):
        """Test CSV export through manager."""
        manager = ExportManager()
        data = [{"a": 1}, {"a": 2}]

        result = manager.export(data, "csv")
        assert "a" in result

    def test_export_csv_from_dict(self):
        """Test CSV export extracts list from dict."""
        manager = ExportManager()
        data = {
            "items": [{"name": "test"}],
        }

        result = manager.export(data, "csv")
        assert "name" in result

    def test_unknown_format(self):
        """Test unknown format raises error."""
        manager = ExportManager()

        with pytest.raises(ValueError) as exc_info:
            manager.export({}, "unknown")

        assert "Unknown format" in str(exc_info.value)

    def test_available_formats(self):
        """Test listing available formats."""
        manager = ExportManager()
        formats = manager.available_formats

        assert "json" in formats
        assert "csv" in formats
        assert "excel" in formats
        assert "pdf" in formats


@pytest.mark.unit
class TestGlobalExportManager:
    """Tests for global export manager."""

    def test_get_export_manager_singleton(self):
        """Test singleton behavior."""
        import qwen_vl.api.export as export_module
        export_module._manager = None

        m1 = get_export_manager()
        m2 = get_export_manager()

        assert m1 is m2

        export_module._manager = None
