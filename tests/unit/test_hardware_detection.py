"""Unit tests for hardware detection module."""

from unittest.mock import MagicMock, patch

import pytest

from qwen_vl.core.hardware_detection import (
    GPUInfo,
    HardwareDetector,
    HardwareInfo,
    detect_hardware,
    get_hardware_detector,
)


@pytest.mark.unit
class TestGPUInfo:
    """Tests for GPUInfo dataclass."""

    def test_utilization_calculation(self):
        """Test memory utilization percentage calculation."""
        gpu = GPUInfo(
            index=0,
            name="Test GPU",
            total_memory_gb=24.0,
            free_memory_gb=12.0,
            used_memory_gb=12.0,
        )
        assert gpu.utilization_percent == 50.0

    def test_zero_total_memory(self):
        """Test utilization with zero total memory."""
        gpu = GPUInfo(
            index=0,
            name="Test GPU",
            total_memory_gb=0.0,
            free_memory_gb=0.0,
            used_memory_gb=0.0,
        )
        assert gpu.utilization_percent == 0.0

    def test_full_utilization(self):
        """Test 100% utilization."""
        gpu = GPUInfo(
            index=0,
            name="Test GPU",
            total_memory_gb=16.0,
            free_memory_gb=0.0,
            used_memory_gb=16.0,
        )
        assert gpu.utilization_percent == 100.0


@pytest.mark.unit
class TestHardwareInfo:
    """Tests for HardwareInfo dataclass."""

    def test_sufficient_vram_check(self):
        """Test sufficient VRAM check."""
        # Sufficient VRAM
        info = HardwareInfo(
            cuda_available=True,
            cuda_version="12.1",
            gpu_count=1,
            gpus=[],
            total_vram_gb=24.0,
            free_vram_gb=20.0,
        )
        assert info.has_sufficient_vram is True

        # Insufficient VRAM
        info = HardwareInfo(
            cuda_available=True,
            cuda_version="12.1",
            gpu_count=1,
            gpus=[],
            total_vram_gb=4.0,
            free_vram_gb=2.0,
        )
        assert info.has_sufficient_vram is False

    def test_recommended_model(self):
        """Test model recommendation based on VRAM."""
        # 8B recommended
        info = HardwareInfo(
            cuda_available=True,
            cuda_version="12.1",
            gpu_count=1,
            gpus=[],
            total_vram_gb=24.0,
            free_vram_gb=20.0,
        )
        assert info.get_recommended_model() == "8B"

        # 4B recommended
        info.free_vram_gb = 10.0
        assert info.get_recommended_model() == "4B"

        # 2B recommended
        info.free_vram_gb = 5.0
        assert info.get_recommended_model() == "2B"

        # None recommended
        info.free_vram_gb = 2.0
        assert info.get_recommended_model() == "none"


@pytest.mark.unit
class TestHardwareDetector:
    """Tests for HardwareDetector class."""

    def test_no_cuda_available(self):
        """Test detection when CUDA is not available."""
        detector = HardwareDetector()
        detector.reset()

        # Manually set hardware info to simulate no CUDA
        detector._hardware_info = HardwareInfo(
            cuda_available=False,
            cuda_version=None,
            gpu_count=0,
            gpus=[],
            total_vram_gb=0.0,
            free_vram_gb=0.0,
        )

        info = detector.detect()
        assert info.cuda_available is False
        assert info.gpu_count == 0
        assert info.total_vram_gb == 0.0

    def test_caching(self):
        """Test that hardware info is cached."""
        detector = HardwareDetector()
        detector.reset()

        # Set mock info
        mock_info = HardwareInfo(
            cuda_available=True,
            cuda_version="12.1",
            gpu_count=1,
            gpus=[],
            total_vram_gb=16.0,
            free_vram_gb=12.0,
        )
        detector._hardware_info = mock_info

        # Should return cached
        info1 = detector.detect()
        info2 = detector.detect()
        assert info1 is info2

    def test_reset(self):
        """Test reset clears cache."""
        detector = HardwareDetector()

        mock_info = HardwareInfo(
            cuda_available=True,
            cuda_version="12.1",
            gpu_count=1,
            gpus=[],
            total_vram_gb=16.0,
            free_vram_gb=12.0,
        )
        detector._hardware_info = mock_info

        detector.reset()
        assert detector._hardware_info is None

    def test_device_map_no_cuda(self):
        """Test device map when CUDA is not available."""
        detector = HardwareDetector()
        detector._hardware_info = HardwareInfo(
            cuda_available=False,
            cuda_version=None,
            gpu_count=0,
            gpus=[],
            total_vram_gb=0.0,
            free_vram_gb=0.0,
        )

        assert detector.get_device_map("4B") == "cpu"

    def test_device_map_sufficient_vram(self):
        """Test device map with sufficient VRAM."""
        detector = HardwareDetector()
        detector._hardware_info = HardwareInfo(
            cuda_available=True,
            cuda_version="12.1",
            gpu_count=1,
            gpus=[],
            total_vram_gb=24.0,
            free_vram_gb=20.0,
        )

        # Sufficient for all models
        assert detector.get_device_map("2B") == "auto"
        assert detector.get_device_map("4B") == "auto"
        assert detector.get_device_map("8B") == "auto"

    def test_device_map_limited_vram(self):
        """Test device map with limited VRAM."""
        detector = HardwareDetector()
        detector._hardware_info = HardwareInfo(
            cuda_available=True,
            cuda_version="12.1",
            gpu_count=1,
            gpus=[],
            total_vram_gb=6.0,
            free_vram_gb=5.0,
        )

        # Only 2B fits comfortably
        assert detector.get_device_map("2B") == "auto"
        # 4B and 8B need more, but we return auto to let accelerate handle it
        assert detector.get_device_map("4B") == "auto"
        assert detector.get_device_map("8B") == "auto"


@pytest.mark.unit
class TestGlobalFunctions:
    """Tests for global convenience functions."""

    def test_get_hardware_detector_singleton(self):
        """Test that get_hardware_detector returns singleton."""
        detector1 = get_hardware_detector()
        detector2 = get_hardware_detector()
        assert detector1 is detector2

    def test_detect_hardware_convenience(self):
        """Test detect_hardware convenience function."""
        detector = get_hardware_detector()
        detector._hardware_info = HardwareInfo(
            cuda_available=True,
            cuda_version="12.1",
            gpu_count=1,
            gpus=[],
            total_vram_gb=16.0,
            free_vram_gb=12.0,
        )

        info = detect_hardware()
        assert info.cuda_available is True
        assert info.free_vram_gb == 12.0
