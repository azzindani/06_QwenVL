"""Unit tests for task handlers."""

import pytest

from qwen_vl.tasks import TaskType, list_handlers, register_handler
from qwen_vl.tasks.base import BaseTaskHandler, TaskResult, _handlers


@pytest.mark.unit
class TestTaskType:
    """Tests for TaskType enum."""

    def test_task_types_exist(self):
        """Test that expected task types exist."""
        assert TaskType.OCR.value == "ocr"
        assert TaskType.LAYOUT.value == "layout"
        assert TaskType.TABLE.value == "table"
        assert TaskType.NER.value == "ner"

    def test_task_type_from_string(self):
        """Test creating TaskType from string."""
        assert TaskType("ocr") == TaskType.OCR
        assert TaskType("layout") == TaskType.LAYOUT


@pytest.mark.unit
class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_basic_result(self):
        """Test creating basic result."""
        result = TaskResult(text="Extracted text")

        assert result.text == "Extracted text"
        assert result.data is None
        assert result.bounding_boxes is None
        assert result.confidence is None

    def test_full_result(self):
        """Test creating full result."""
        result = TaskResult(
            text="Text",
            data={"key": "value"},
            bounding_boxes=[{"x1": 0, "y1": 0, "x2": 100, "y2": 100}],
            confidence=0.95,
            metadata={"mode": "test"},
        )

        assert result.text == "Text"
        assert result.data == {"key": "value"}
        assert len(result.bounding_boxes) == 1
        assert result.confidence == 0.95
        assert result.metadata["mode"] == "test"


@pytest.mark.unit
class TestHandlerRegistry:
    """Tests for handler registration."""

    def test_handlers_registered(self):
        """Test that OCR and Layout handlers are registered."""
        handlers = list_handlers()

        assert TaskType.OCR in handlers
        assert TaskType.LAYOUT in handlers

    def test_register_decorator(self):
        """Test the register_handler decorator."""
        # Save original handlers
        original = _handlers.copy()

        try:
            # Create a test handler
            @register_handler(TaskType.TABLE)
            class TestHandler(BaseTaskHandler):
                @property
                def task_type(self):
                    return TaskType.TABLE

                @property
                def system_prompt(self):
                    return "Test"

                def process(self, image, prompt=None, **kwargs):
                    return TaskResult(text="test")

            # Check it's registered
            assert TaskType.TABLE in _handlers
            assert _handlers[TaskType.TABLE] == TestHandler

        finally:
            # Restore original handlers
            _handlers.clear()
            _handlers.update(original)


@pytest.mark.unit
class TestBaseTaskHandler:
    """Tests for BaseTaskHandler."""

    def test_load_image_from_path(self, tmp_path):
        """Test loading image from path."""
        from PIL import Image

        # Create test image
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100), color="white")
        img.save(img_path)

        # Create mock handler
        class MockHandler(BaseTaskHandler):
            @property
            def task_type(self):
                return TaskType.OCR

            @property
            def system_prompt(self):
                return "Test"

            def process(self, image, prompt=None, **kwargs):
                return TaskResult(text="test")

        handler = MockHandler(model=None, processor=None)
        loaded = handler._load_image(str(img_path))

        assert isinstance(loaded, Image.Image)
        assert loaded.mode == "RGB"

    def test_load_image_from_pil(self):
        """Test loading PIL Image directly."""
        from PIL import Image

        img = Image.new("RGBA", (100, 100), color="red")

        class MockHandler(BaseTaskHandler):
            @property
            def task_type(self):
                return TaskType.OCR

            @property
            def system_prompt(self):
                return "Test"

            def process(self, image, prompt=None, **kwargs):
                return TaskResult(text="test")

        handler = MockHandler(model=None, processor=None)
        loaded = handler._load_image(img)

        # Should convert to RGB
        assert loaded.mode == "RGB"

    def test_build_messages(self):
        """Test building chat messages."""
        from PIL import Image

        img = Image.new("RGB", (100, 100))

        class MockHandler(BaseTaskHandler):
            @property
            def task_type(self):
                return TaskType.OCR

            @property
            def system_prompt(self):
                return "System prompt"

            def process(self, image, prompt=None, **kwargs):
                return TaskResult(text="test")

        handler = MockHandler(model=None, processor=None)
        messages = handler._build_messages(img, "User prompt")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System prompt"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"][1]["text"] == "User prompt"
