"""Core inference components."""

def __getattr__(name):
    """Lazy import for heavy modules."""
    if name == "ModelLoader":
        from .model_loader import ModelLoader
        return ModelLoader
    if name == "load_model":
        from .model_loader import load_model
        return load_model
    if name == "HardwareDetector":
        from .hardware_detection import HardwareDetector
        return HardwareDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ModelLoader", "load_model", "HardwareDetector"]
