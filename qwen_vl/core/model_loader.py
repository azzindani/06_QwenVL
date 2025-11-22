"""Model loading with singleton pattern and quantization support."""

import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from ..config import Config, get_config
from .hardware_detection import get_hardware_detector

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """Container for loaded model and processor."""

    model: Any
    processor: Any
    config: Config
    device: str


class ModelLoader:
    """Singleton model loader with caching."""

    _instance: Optional["ModelLoader"] = None
    _loaded_model: Optional[LoadedModel] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(
        self,
        config: Optional[Config] = None,
        force_reload: bool = False,
    ) -> LoadedModel:
        """
        Load the model and processor.

        Args:
            config: Configuration to use (defaults to global config)
            force_reload: Force reload even if already loaded

        Returns:
            LoadedModel with model and processor
        """
        if self._loaded_model is not None and not force_reload:
            logger.info("Using cached model")
            return self._loaded_model

        if config is None:
            config = get_config()

        logger.info(f"Loading model: {config.model.model_id}")
        logger.info(f"Quantization: {config.model.quantization}")

        try:
            model, processor = self._load_model_and_processor(config)

            # Determine device
            detector = get_hardware_detector()
            hardware = detector.detect()

            if hardware.cuda_available:
                device = "cuda"
            else:
                device = "cpu"
                logger.warning("No GPU available, using CPU (will be slow)")

            self._loaded_model = LoadedModel(
                model=model,
                processor=processor,
                config=config,
                device=device,
            )

            logger.info(f"Model loaded successfully on {device}")
            return self._loaded_model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_model_and_processor(self, config: Config) -> Tuple[Any, Any]:
        """
        Internal method to load model and processor from HuggingFace.

        Args:
            config: Configuration

        Returns:
            Tuple of (model, processor)
        """
        # Import here to avoid slow startup when not loading model
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig

        # Try to import Qwen model class
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as QwenModel
        except ImportError:
            from transformers import AutoModelForCausalLM as QwenModel
            logger.warning("Using AutoModelForCausalLM as fallback")

        model_id = config.model.model_id

        # Configure quantization
        quantization_config = None
        if config.model.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif config.model.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load processor
        logger.info("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        # Load model
        logger.info("Loading model weights...")
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": config.model.device_map,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch.float16

        model = QwenModel.from_pretrained(model_id, **model_kwargs)

        return model, processor

    def unload(self) -> None:
        """Unload the model and free memory."""
        if self._loaded_model is not None:
            logger.info("Unloading model...")

            # Clear references
            del self._loaded_model.model
            del self._loaded_model.processor
            self._loaded_model = None

            # Force garbage collection
            import gc

            gc.collect()

            # Clear CUDA cache if available
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("Model unloaded")

    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._loaded_model is not None

    def get_model_info(self) -> Optional[dict]:
        """Get information about the currently loaded model."""
        if self._loaded_model is None:
            return None

        return {
            "model_id": self._loaded_model.config.model.model_id,
            "size": self._loaded_model.config.model.size,
            "variant": self._loaded_model.config.model.variant,
            "quantization": self._loaded_model.config.model.quantization,
            "device": self._loaded_model.device,
        }

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        if cls._instance is not None:
            cls._instance.unload()
        cls._instance = None
        cls._loaded_model = None


def load_model(config: Optional[Config] = None) -> LoadedModel:
    """
    Convenience function to load model.

    Args:
        config: Optional configuration

    Returns:
        LoadedModel instance
    """
    loader = ModelLoader()
    return loader.load(config)


def get_model() -> Optional[LoadedModel]:
    """Get the currently loaded model."""
    loader = ModelLoader()
    return loader._loaded_model


def unload_model() -> None:
    """Unload the current model."""
    loader = ModelLoader()
    loader.unload()


if __name__ == "__main__":
    print("=" * 60)
    print("MODEL LOADER TEST")
    print("=" * 60)

    # This will only work with actual model files
    # For testing, we just verify the loader initializes correctly

    loader = ModelLoader()
    print(f"  Loader initialized: {loader is not None}")
    print(f"  Is singleton: {ModelLoader() is loader}")
    print(f"  Model loaded: {loader.is_loaded()}")

    # Get config and show what would be loaded
    config = get_config()
    print(f"  Would load: {config.model.model_id}")
    print(f"  Quantization: {config.model.quantization}")
    print(f"  Estimated VRAM: {config.model.estimated_vram_gb} GB")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
