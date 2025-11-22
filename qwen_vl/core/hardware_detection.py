"""Hardware detection and GPU resource management."""

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a single GPU."""

    index: int
    name: str
    total_memory_gb: float
    free_memory_gb: float
    used_memory_gb: float

    @property
    def utilization_percent(self) -> float:
        """Get memory utilization percentage."""
        if self.total_memory_gb == 0:
            return 0.0
        return (self.used_memory_gb / self.total_memory_gb) * 100


@dataclass
class HardwareInfo:
    """Complete hardware information."""

    cuda_available: bool
    cuda_version: Optional[str]
    gpu_count: int
    gpus: List[GPUInfo]
    total_vram_gb: float
    free_vram_gb: float

    @property
    def has_sufficient_vram(self) -> bool:
        """Check if there's at least 4GB free VRAM for smallest model."""
        return self.free_vram_gb >= 4.0

    def get_recommended_model(self) -> str:
        """Get recommended model size based on available VRAM."""
        if self.free_vram_gb >= 16:
            return "8B"
        elif self.free_vram_gb >= 8:
            return "4B"
        elif self.free_vram_gb >= 4:
            return "2B"
        else:
            return "none"


class HardwareDetector:
    """Detect and manage hardware resources."""

    def __init__(self):
        self._hardware_info: Optional[HardwareInfo] = None

    def detect(self) -> HardwareInfo:
        """
        Detect available hardware resources.

        Returns:
            HardwareInfo with GPU details
        """
        if self._hardware_info is not None:
            return self._hardware_info

        try:
            import torch

            cuda_available = torch.cuda.is_available()

            if not cuda_available:
                self._hardware_info = HardwareInfo(
                    cuda_available=False,
                    cuda_version=None,
                    gpu_count=0,
                    gpus=[],
                    total_vram_gb=0.0,
                    free_vram_gb=0.0,
                )
                return self._hardware_info

            cuda_version = torch.version.cuda
            gpu_count = torch.cuda.device_count()
            gpus = []
            total_vram = 0.0
            free_vram = 0.0

            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory / (1024**3)  # Convert to GB

                # Get current memory usage
                torch.cuda.set_device(i)
                free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
                used_mem = total_mem - free_mem

                gpu_info = GPUInfo(
                    index=i,
                    name=props.name,
                    total_memory_gb=round(total_mem, 2),
                    free_memory_gb=round(free_mem, 2),
                    used_memory_gb=round(used_mem, 2),
                )
                gpus.append(gpu_info)
                total_vram += total_mem
                free_vram += free_mem

            self._hardware_info = HardwareInfo(
                cuda_available=True,
                cuda_version=cuda_version,
                gpu_count=gpu_count,
                gpus=gpus,
                total_vram_gb=round(total_vram, 2),
                free_vram_gb=round(free_vram, 2),
            )

        except ImportError:
            logger.warning("PyTorch not installed, cannot detect GPU")
            self._hardware_info = HardwareInfo(
                cuda_available=False,
                cuda_version=None,
                gpu_count=0,
                gpus=[],
                total_vram_gb=0.0,
                free_vram_gb=0.0,
            )

        return self._hardware_info

    def reset(self) -> None:
        """Reset cached hardware info (useful for testing)."""
        self._hardware_info = None

    def get_device_map(self, model_size: str) -> str:
        """
        Get optimal device map based on model size and available hardware.

        Args:
            model_size: Model size (2B, 4B, 8B)

        Returns:
            Device map string for model loading
        """
        info = self.detect()

        if not info.cuda_available:
            return "cpu"

        vram_required = {"2B": 4.0, "4B": 8.0, "8B": 16.0}
        required = vram_required.get(model_size, 8.0)

        if info.free_vram_gb >= required:
            return "auto"
        elif info.gpu_count > 1:
            # Multi-GPU setup
            return "auto"
        else:
            logger.warning(
                f"Insufficient VRAM for {model_size} model. "
                f"Required: {required}GB, Available: {info.free_vram_gb}GB"
            )
            return "auto"  # Let accelerate handle memory management

    def print_summary(self) -> None:
        """Print a summary of detected hardware."""
        info = self.detect()

        print("=" * 60)
        print("HARDWARE SUMMARY")
        print("=" * 60)
        print(f"  CUDA Available: {info.cuda_available}")

        if info.cuda_available:
            print(f"  CUDA Version: {info.cuda_version}")
            print(f"  GPU Count: {info.gpu_count}")
            print(f"  Total VRAM: {info.total_vram_gb} GB")
            print(f"  Free VRAM: {info.free_vram_gb} GB")
            print()

            for gpu in info.gpus:
                print(f"  GPU {gpu.index}: {gpu.name}")
                print(f"    Total: {gpu.total_memory_gb} GB")
                print(f"    Free: {gpu.free_memory_gb} GB")
                print(f"    Used: {gpu.used_memory_gb} GB ({gpu.utilization_percent:.1f}%)")
                print()

            print(f"  Recommended Model: {info.get_recommended_model()}")
        else:
            print("  No CUDA-capable GPU detected")

        print("=" * 60)


# Global instance
_detector: Optional[HardwareDetector] = None


def get_hardware_detector() -> HardwareDetector:
    """Get the global hardware detector instance."""
    global _detector
    if _detector is None:
        _detector = HardwareDetector()
    return _detector


def detect_hardware() -> HardwareInfo:
    """Convenience function to detect hardware."""
    return get_hardware_detector().detect()


if __name__ == "__main__":
    detector = HardwareDetector()
    detector.print_summary()

    info = detector.detect()
    if info.cuda_available:
        print(f"\nDevice map for 4B model: {detector.get_device_map('4B')}")
