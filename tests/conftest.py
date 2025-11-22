"""Shared test fixtures and configuration."""

import os
import sys

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_image():
    """Provide test image path."""
    return "tests/fixtures/sample.jpg"


@pytest.fixture
def sample_video():
    """Provide test video path."""
    return "tests/fixtures/sample.mp4"


@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment variables for testing."""
    env_vars = [
        "QWEN_MODEL_SIZE",
        "QWEN_MODEL_VARIANT",
        "QWEN_QUANTIZATION",
        "QWEN_DEVICE_MAP",
        "QWEN_MAX_NEW_TOKENS",
        "QWEN_MIN_PIXELS",
        "QWEN_MAX_PIXELS",
        "QWEN_TEMPERATURE",
        "QWEN_TOP_P",
        "GRADIO_SERVER_NAME",
        "GRADIO_SERVER_PORT",
        "GRADIO_SHARE",
        "MAX_FILE_SIZE_MB",
        "MAX_VIDEO_SIZE_MB",
        "LOG_LEVEL",
        "LOG_FORMAT",
        "LOG_FILE_PATH",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture
def mock_torch_cuda(monkeypatch):
    """Mock torch.cuda for testing without GPU."""
    class MockCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            return 0

    mock_torch = type("MockTorch", (), {"cuda": MockCuda, "version": type("", (), {"cuda": None})})()

    monkeypatch.setattr("torch.cuda", MockCuda)
    return mock_torch


# Markers
def pytest_configure(config):
    config.addinivalue_line("markers", "unit: Unit tests (no GPU required)")
    config.addinivalue_line("markers", "integration: Integration tests (GPU required)")
