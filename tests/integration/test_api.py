"""
API Endpoint Tests

Tests API endpoints without requiring model initialization.
Uses httpx for async HTTP testing.

Usage:
    pytest tests/integration/test_api.py -v
"""

import pytest
from pathlib import Path

# Test configuration
API_BASE_URL = "http://localhost:8000"
ASSET_DIR = Path(__file__).parent.parent / "asset"


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client):
        """Health check should return 200 and healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestTasksEndpoint:
    """Tests for the /tasks endpoint."""

    def test_tasks_returns_list(self, client):
        """Tasks endpoint should return a list of available tasks."""
        response = client.get("/tasks")
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        assert isinstance(data["tasks"], list)


class TestOCREndpoint:
    """Tests for the /extract/ocr endpoint."""

    def test_ocr_requires_file(self, client):
        """OCR endpoint should require a file."""
        response = client.post("/extract/ocr")
        assert response.status_code == 422  # Validation error

    def test_ocr_with_valid_image(self, client, sample_image):
        """OCR endpoint should process valid image."""
        with open(sample_image, "rb") as f:
            response = client.post(
                "/extract/ocr",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={"include_boxes": "false"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "text" in data


class TestTableEndpoint:
    """Tests for the /extract/table endpoint."""

    def test_table_with_valid_image(self, client, sample_image):
        """Table endpoint should process valid image."""
        with open(sample_image, "rb") as f:
            response = client.post(
                "/extract/table",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={"output_format": "json"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "tables" in data


class TestFormEndpoint:
    """Tests for the /extract/form endpoint."""

    def test_form_with_valid_image(self, client, sample_image):
        """Form endpoint should process valid image."""
        with open(sample_image, "rb") as f:
            response = client.post(
                "/extract/form",
                files={"file": ("test.jpg", f, "image/jpeg")},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "fields" in data


class TestInvoiceEndpoint:
    """Tests for the /extract/invoice endpoint."""

    def test_invoice_with_valid_image(self, client, sample_image):
        """Invoice endpoint should process valid image."""
        with open(sample_image, "rb") as f:
            response = client.post(
                "/extract/invoice",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={"document_type": "invoice"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestNEREndpoint:
    """Tests for the /extract/ner endpoint."""

    def test_ner_with_valid_image(self, client, sample_image):
        """NER endpoint should process valid image."""
        with open(sample_image, "rb") as f:
            response = client.post(
                "/extract/ner",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={"entity_types": "all"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "entities" in data


class TestLayoutEndpoint:
    """Tests for the /layout endpoint."""

    def test_layout_with_valid_image(self, client, sample_image):
        """Layout endpoint should process valid image."""
        with open(sample_image, "rb") as f:
            response = client.post(
                "/layout",
                files={"file": ("test.jpg", f, "image/jpeg")},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "elements" in data


class TestSpatialEndpoint:
    """Tests for the /spatial endpoint."""

    def test_spatial_with_valid_image(self, client, sample_image):
        """Spatial endpoint should process valid image."""
        with open(sample_image, "rb") as f:
            response = client.post(
                "/spatial",
                files={"file": ("test.jpg", f, "image/jpeg")},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


# Pytest fixtures
@pytest.fixture
def client():
    """Create test client."""
    from fastapi.testclient import TestClient
    from qwen_vl.api.server import create_app
    
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Get a sample test image."""
    # Find first available image
    for pattern in ["ocr_example*.jpg", "docparsing_example*.jpg"]:
        images = list(ASSET_DIR.glob(pattern))
        if images:
            return images[0]
    
    # Create a simple test image if none found
    from PIL import Image
    import tempfile
    
    img = Image.new("RGB", (100, 100), color="white")
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name)
    return tmp.name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
