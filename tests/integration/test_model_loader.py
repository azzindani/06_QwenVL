"""Integration tests for model loader (requires GPU)."""

import pytest

# Skip all tests in this module if torch is not available
torch = pytest.importorskip("torch")

from qwen_vl.config import Config, ModelConfig, get_config, reset_config
from qwen_vl.core.model_loader import ModelLoader, get_model, load_model, unload_model


@pytest.mark.integration
class TestModelLoaderSingleton:
    """Tests for ModelLoader singleton pattern."""

    def setup_method(self):
        """Reset model loader before each test."""
        ModelLoader.reset()
        reset_config()

    def test_singleton_behavior(self):
        """Test that ModelLoader is a singleton."""
        loader1 = ModelLoader()
        loader2 = ModelLoader()
        assert loader1 is loader2

    def test_reset_singleton(self):
        """Test reset clears singleton."""
        loader1 = ModelLoader()
        ModelLoader.reset()
        loader2 = ModelLoader()

        # After reset, should be different instance
        # (actually same due to __new__, but state should be clean)
        assert loader2.is_loaded() is False


@pytest.mark.integration
class TestModelLoaderWithoutGPU:
    """Tests for ModelLoader that don't require actual model loading."""

    def setup_method(self):
        """Reset before each test."""
        ModelLoader.reset()
        reset_config()

    def test_is_loaded_initially_false(self):
        """Test that no model is loaded initially."""
        loader = ModelLoader()
        assert loader.is_loaded() is False

    def test_get_model_info_when_not_loaded(self):
        """Test get_model_info returns None when not loaded."""
        loader = ModelLoader()
        assert loader.get_model_info() is None

    def test_unload_when_not_loaded(self):
        """Test unload doesn't error when nothing loaded."""
        loader = ModelLoader()
        loader.unload()  # Should not raise
        assert loader.is_loaded() is False


@pytest.mark.integration
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)
class TestModelLoaderWithGPU:
    """Tests for ModelLoader that require GPU and actual model loading."""

    def setup_method(self):
        """Reset before each test."""
        ModelLoader.reset()
        reset_config()

    def teardown_method(self):
        """Clean up after each test."""
        ModelLoader.reset()

    def test_load_model(self):
        """Test loading a model."""
        # This test will actually download and load the model
        # Only run if explicitly enabled
        pytest.skip("Skipping actual model load - enable for full integration test")

        loader = ModelLoader()
        config = get_config()

        loaded = loader.load(config)

        assert loaded is not None
        assert loaded.model is not None
        assert loaded.processor is not None
        assert loader.is_loaded() is True

    def test_cached_model_returned(self):
        """Test that cached model is returned on second load."""
        pytest.skip("Skipping actual model load - enable for full integration test")

        loader = ModelLoader()
        config = get_config()

        loaded1 = loader.load(config)
        loaded2 = loader.load(config)

        assert loaded1 is loaded2

    def test_force_reload(self):
        """Test force reload creates new model."""
        pytest.skip("Skipping actual model load - enable for full integration test")

        loader = ModelLoader()
        config = get_config()

        loaded1 = loader.load(config)
        loaded2 = loader.load(config, force_reload=True)

        # Should be different instances
        assert loaded1 is not loaded2

    def test_unload_frees_memory(self):
        """Test that unload frees GPU memory."""
        pytest.skip("Skipping actual model load - enable for full integration test")

        loader = ModelLoader()
        config = get_config()

        loader.load(config)
        assert loader.is_loaded() is True

        # Get memory before unload
        mem_before = torch.cuda.memory_allocated()

        loader.unload()
        assert loader.is_loaded() is False

        # Memory should be reduced
        mem_after = torch.cuda.memory_allocated()
        assert mem_after < mem_before

    def test_model_info_after_load(self):
        """Test model info is correct after loading."""
        pytest.skip("Skipping actual model load - enable for full integration test")

        loader = ModelLoader()
        config = get_config()

        loader.load(config)
        info = loader.get_model_info()

        assert info is not None
        assert info["size"] == config.model.size
        assert info["variant"] == config.model.variant
        assert info["quantization"] == config.model.quantization
        assert info["device"] in ["cuda", "cpu"]


@pytest.mark.integration
class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def setup_method(self):
        """Reset before each test."""
        ModelLoader.reset()
        reset_config()

    def test_get_model_returns_none_initially(self):
        """Test get_model returns None when not loaded."""
        assert get_model() is None

    def test_unload_model_convenience(self):
        """Test unload_model convenience function."""
        unload_model()  # Should not raise
        assert get_model() is None
