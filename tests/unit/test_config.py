"""Unit tests for configuration module."""

import os

import pytest

from qwen_vl.config import (
    Config,
    InferenceConfig,
    LoggingConfig,
    ModelConfig,
    ServerConfig,
    get_config,
    load_config,
    reset_config,
)


@pytest.mark.unit
class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.family == "qwen2.5"
        assert config.size == "7B"
        assert config.variant == "instruct"
        assert config.quantization == "4bit"
        assert config.device_map == "auto"

    def test_model_id_generation(self):
        """Test model ID generation."""
        # Qwen2.5 models
        config = ModelConfig(family="qwen2.5", size="7B", variant="instruct")
        assert config.model_id == "Qwen/Qwen2.5-VL-7B-Instruct"

        config = ModelConfig(family="qwen2.5", size="2B", variant="instruct")
        assert config.model_id == "Qwen/Qwen2.5-VL-2B-Instruct"

        # Qwen3 models
        config = ModelConfig(family="qwen3", size="4B", variant="instruct")
        assert config.model_id == "Qwen/Qwen3-VL-4B-Instruct"

        config = ModelConfig(family="qwen3", size="8B", variant="thinking")
        assert config.model_id == "Qwen/Qwen3-VL-8B-Thinking"

    def test_local_path(self):
        """Test local model path configuration."""
        # Without local path
        config = ModelConfig()
        assert config.is_local is False
        assert config.local_path is None
        assert "Qwen/" in config.model_id

        # With local path
        config = ModelConfig(local_path="/models/qwen3-vl-4b")
        assert config.is_local is True
        assert config.model_id == "/models/qwen3-vl-4b"

    def test_vram_estimation(self):
        """Test VRAM estimation for different configs."""
        # 4-bit quantization
        config = ModelConfig(size="2B", quantization="4bit")
        assert config.estimated_vram_gb == 4.0

        config = ModelConfig(size="4B", quantization="4bit")
        assert config.estimated_vram_gb == 8.0

        config = ModelConfig(size="8B", quantization="4bit")
        assert config.estimated_vram_gb == 16.0

        # No quantization (2x memory)
        config = ModelConfig(size="4B", quantization="none")
        assert config.estimated_vram_gb == 16.0

        # 8-bit quantization (1.5x)
        config = ModelConfig(size="4B", quantization="8bit")
        assert config.estimated_vram_gb == 12.0


@pytest.mark.unit
class TestInferenceConfig:
    """Tests for InferenceConfig."""

    def test_default_values(self):
        """Test default inference configuration."""
        config = InferenceConfig()
        assert config.max_new_tokens == 4096
        assert config.min_pixels == 512 * 28 * 28
        assert config.max_pixels == 2048 * 28 * 28
        assert config.temperature == 0.7
        assert config.top_p == 0.9


@pytest.mark.unit
class TestServerConfig:
    """Tests for ServerConfig."""

    def test_default_values(self):
        """Test default server configuration."""
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 7860
        assert config.share is False
        assert config.max_file_size_mb == 20
        assert config.max_video_size_mb == 100


@pytest.mark.unit
class TestLoggingConfig:
    """Tests for LoggingConfig."""

    def test_default_values(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.format == "json"
        assert config.file_path is None


@pytest.mark.unit
class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_defaults(self, clean_env):
        """Test loading default configuration."""
        reset_config()
        config = load_config()

        assert config.model.size == "7B"
        assert config.model.variant == "instruct"
        assert config.model.quantization == "4bit"
        assert config.inference.max_new_tokens == 4096
        assert config.server.port == 7860
        assert config.logging.level == "INFO"

    def test_env_var_override(self, clean_env, monkeypatch):
        """Test environment variable overrides."""
        monkeypatch.setenv("QWEN_MODEL_SIZE", "8B")
        monkeypatch.setenv("QWEN_MODEL_VARIANT", "thinking")
        monkeypatch.setenv("QWEN_QUANTIZATION", "8bit")
        monkeypatch.setenv("QWEN_MAX_NEW_TOKENS", "2048")
        monkeypatch.setenv("GRADIO_SERVER_PORT", "8080")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("GRADIO_SHARE", "true")

        reset_config()
        config = load_config()

        assert config.model.size == "8B"
        assert config.model.variant == "thinking"
        assert config.model.quantization == "8bit"
        assert config.inference.max_new_tokens == 2048
        assert config.server.port == 8080
        assert config.server.share is True
        assert config.logging.level == "DEBUG"

    def test_partial_override(self, clean_env, monkeypatch):
        """Test partial environment variable override."""
        monkeypatch.setenv("QWEN_MODEL_SIZE", "2B")

        reset_config()
        config = load_config()

        # Changed
        assert config.model.size == "2B"

        # Defaults
        assert config.model.variant == "instruct"
        assert config.model.quantization == "4bit"


@pytest.mark.unit
class TestGetConfig:
    """Tests for get_config singleton."""

    def test_singleton_behavior(self, clean_env):
        """Test that get_config returns the same instance."""
        reset_config()
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_reset_config(self, clean_env, monkeypatch):
        """Test config reset functionality."""
        monkeypatch.setenv("QWEN_MODEL_SIZE", "2B")

        reset_config()
        config1 = get_config()
        assert config1.model.size == "2B"

        monkeypatch.setenv("QWEN_MODEL_SIZE", "8B")
        reset_config()
        config2 = get_config()
        assert config2.model.size == "8B"

        # Different instances after reset
        assert config1 is not config2


@pytest.mark.unit
class TestConfigValidation:
    """Tests for configuration validation."""

    def test_invalid_model_size_type(self, clean_env, monkeypatch):
        """Test that non-numeric port raises error."""
        monkeypatch.setenv("GRADIO_SERVER_PORT", "invalid")

        reset_config()
        with pytest.raises(ValueError):
            load_config()

    def test_numeric_conversion(self, clean_env, monkeypatch):
        """Test numeric string conversion."""
        monkeypatch.setenv("QWEN_MAX_NEW_TOKENS", "1024")
        monkeypatch.setenv("QWEN_TEMPERATURE", "0.5")

        reset_config()
        config = load_config()

        assert config.inference.max_new_tokens == 1024
        assert config.inference.temperature == 0.5
        assert isinstance(config.inference.max_new_tokens, int)
        assert isinstance(config.inference.temperature, float)
