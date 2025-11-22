"""Unit tests for multi-page document processing."""

import pytest
from unittest.mock import MagicMock

from qwen_vl.tasks.multipage import (
    MultiPageProcessor,
    PageResult,
    DocumentResult,
    detect_document_boundaries,
)
from qwen_vl.tasks.base import TaskResult


@pytest.mark.unit
class TestPageResult:
    """Tests for PageResult dataclass."""

    def test_create_page_result(self):
        """Test creating page result."""
        task_result = TaskResult(text="Page content")
        page = PageResult(page_number=1, result=task_result)

        assert page.page_number == 1
        assert page.result.text == "Page content"


@pytest.mark.unit
class TestDocumentResult:
    """Tests for DocumentResult dataclass."""

    def test_total_pages(self):
        """Test total pages calculation."""
        pages = [
            PageResult(1, TaskResult(text="Page 1")),
            PageResult(2, TaskResult(text="Page 2")),
            PageResult(3, TaskResult(text="Page 3")),
        ]

        doc = DocumentResult(
            pages=pages,
            merged_text="All content",
        )

        assert doc.total_pages == 3


@pytest.mark.unit
class TestMultiPageProcessor:
    """Tests for MultiPageProcessor."""

    def test_merge_concatenate(self):
        """Test concatenate merge strategy."""
        handler = MagicMock()
        processor = MultiPageProcessor(handler)

        page_results = [
            PageResult(1, TaskResult(text="First page")),
            PageResult(2, TaskResult(text="Second page")),
        ]

        merged_text, merged_data = processor._merge_concatenate(page_results)

        assert "Page 1" in merged_text
        assert "First page" in merged_text
        assert "Page 2" in merged_text
        assert "Second page" in merged_text

    def test_merge_structured(self):
        """Test structured merge strategy."""
        handler = MagicMock()
        processor = MultiPageProcessor(handler)

        page_results = [
            PageResult(1, TaskResult(text="Page 1", data={"items": ["a", "b"]})),
            PageResult(2, TaskResult(text="Page 2", data={"items": ["c", "d"]})),
        ]

        merged_text, merged_data = processor._merge_structured(page_results)

        assert "Page 1" in merged_text
        assert "Page 2" in merged_text
        assert merged_data["items"] == ["a", "b", "c", "d"]


@pytest.mark.unit
class TestDocumentBoundaries:
    """Tests for document boundary detection."""

    def test_detect_boundaries(self):
        """Test boundary detection returns groups."""
        images = ["img1.png", "img2.png", "img3.png"]
        groups = detect_document_boundaries(images)

        assert len(groups) == 3  # Default: each image is separate
        assert groups[0] == [0]
        assert groups[1] == [1]
        assert groups[2] == [2]
