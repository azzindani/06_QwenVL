"""Qwen3-VL Production Deployment Package."""

__version__ = "0.1.0"

def __getattr__(name):
    """Lazy import for heavy modules."""
    if name == "InferenceEngine":
        from .core.inference_engine import InferenceEngine
        return InferenceEngine
    if name == "load_model":
        from .core.model_loader import load_model
        return load_model
    if name == "Config":
        from .config import Config
        return Config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["InferenceEngine", "load_model", "Config", "__version__"]
