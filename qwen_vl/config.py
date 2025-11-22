"""Centralized configuration management for Qwen VL models."""

import os
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ModelConfig:
    """Model configuration settings."""

    family: Literal["qwen2.5", "qwen3"] = "qwen2.5"
    size: str = "7B"  # 3B, 7B, 72B for Qwen2.5; 2B, 4B, 8B for Qwen3
    variant: Literal["instruct", "thinking"] = "instruct"
    quantization: Literal["none", "4bit", "8bit"] = "4bit"
    device_map: str = "auto"
    local_path: Optional[str] = None  # Local model directory path

    @property
    def model_id(self) -> str:
        """Get the Hugging Face model ID or local path."""
        if self.local_path:
            return self.local_path

        # Map to correct HuggingFace repository names
        if self.family == "qwen2.5":
            # Qwen/Qwen2.5-VL-2B-Instruct, Qwen/Qwen2.5-VL-7B-Instruct, etc.
            return f"Qwen/Qwen2.5-VL-{self.size}-Instruct"
        else:
            # Qwen/Qwen3-VL-2B-Instruct, Qwen/Qwen3-VL-4B-Thinking, etc.
            variant_name = "Instruct" if self.variant == "instruct" else "Thinking"
            return f"Qwen/Qwen3-VL-{self.size}-{variant_name}"

    @property
    def is_local(self) -> bool:
        """Check if using local model path."""
        return self.local_path is not None

    @property
    def estimated_vram_gb(self) -> float:
        """Estimate VRAM usage based on model size and quantization."""
        base_vram = {
            "2B": 4.0, "3B": 6.0, "4B": 8.0, "7B": 14.0, "8B": 16.0, "72B": 144.0
        }
        quant_multiplier = {"none": 2.0, "4bit": 1.0, "8bit": 1.5}
        return base_vram.get(self.size, 8.0) * quant_multiplier[self.quantization]


@dataclass
class InferenceConfig:
    """Inference configuration settings."""

    max_new_tokens: int = 4096
    min_pixels: int = 512 * 28 * 28  # 401408
    max_pixels: int = 2048 * 28 * 28  # 1605632
    total_pixels: int = 20480 * 28 * 28  # For video
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


@dataclass
class ServerConfig:
    """Server configuration settings."""

    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False
    max_file_size_mb: int = 20
    max_video_size_mb: int = 100


@dataclass
class LoggingConfig:
    """Logging configuration settings."""

    level: str = "INFO"
    format: Literal["json", "text"] = "json"
    file_path: Optional[str] = None


@dataclass
class Config:
    """Main configuration container."""

    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config() -> Config:
    """Load configuration from environment variables."""
    # Get local path (None if not set or empty)
    local_path = os.getenv("QWEN_MODEL_PATH")
    if local_path == "":
        local_path = None

    return Config(
        model=ModelConfig(
            family=os.getenv("QWEN_MODEL_FAMILY", "qwen2.5"),
            size=os.getenv("QWEN_MODEL_SIZE", "7B"),
            variant=os.getenv("QWEN_MODEL_VARIANT", "instruct"),
            quantization=os.getenv("QWEN_QUANTIZATION", "4bit"),
            device_map=os.getenv("QWEN_DEVICE_MAP", "auto"),
            local_path=local_path,
        ),
        inference=InferenceConfig(
            max_new_tokens=int(os.getenv("QWEN_MAX_NEW_TOKENS", "4096")),
            min_pixels=int(os.getenv("QWEN_MIN_PIXELS", str(512 * 28 * 28))),
            max_pixels=int(os.getenv("QWEN_MAX_PIXELS", str(2048 * 28 * 28))),
            temperature=float(os.getenv("QWEN_TEMPERATURE", "0.7")),
            top_p=float(os.getenv("QWEN_TOP_P", "0.9")),
        ),
        server=ServerConfig(
            host=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
            port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
            share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "20")),
            max_video_size_mb=int(os.getenv("MAX_VIDEO_SIZE_MB", "100")),
        ),
        logging=LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format=os.getenv("LOG_FORMAT", "json"),
            file_path=os.getenv("LOG_FILE_PATH"),
        ),
    )


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None


if __name__ == "__main__":
    print("=" * 60)
    print("CONFIG TEST")
    print("=" * 60)

    config = load_config()
    print(f"  Model size: {config.model.size}")
    print(f"  Model variant: {config.model.variant}")
    print(f"  Model ID: {config.model.model_id}")
    print(f"  Is local: {config.model.is_local}")
    print(f"  Local path: {config.model.local_path}")
    print(f"  Quantization: {config.model.quantization}")
    print(f"  Estimated VRAM: {config.model.estimated_vram_gb} GB")
    print(f"  Max tokens: {config.inference.max_new_tokens}")
    print(f"  Server port: {config.server.port}")
    print(f"  Log level: {config.logging.level}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
