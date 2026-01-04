"""
API client for UI to communicate with API backend.

Provides:
- HTTP client wrapper for API calls
- Image encoding/decoding
- Error handling and retries
"""

import base64
import io
from typing import Any, Dict, Optional

import requests
from PIL import Image

from ..utils.logger import get_logger

logger = get_logger(__name__)


class APIClient:
    """
    HTTP client for communicating with the Qwen VL API.
    
    Example usage:
        client = APIClient("http://localhost:8000")
        result = client.extract_ocr(image)
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 120,
    ):
        """
        Initialize API client.
        
        Args:
            base_url: API base URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
    
    def is_healthy(self) -> bool:
        """Check if API is running and healthy."""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5,
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def _image_to_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL Image to bytes."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    
    def _bytes_to_image(self, data: bytes) -> Image.Image:
        """Convert bytes to PIL Image."""
        return Image.open(io.BytesIO(data))
    
    def _base64_to_image(self, b64: str) -> Optional[Image.Image]:
        """Convert base64 string to PIL Image."""
        if not b64:
            return None
        try:
            data = base64.b64decode(b64)
            return self._bytes_to_image(data)
        except Exception as e:
            logger.warning(f"Failed to decode image: {e}")
            return None
    
    def _post_image(
        self,
        endpoint: str,
        image: Image.Image,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        POST image to endpoint.
        
        Args:
            endpoint: API endpoint path
            image: PIL Image to upload
            data: Additional form data
            
        Returns:
            Response JSON
        """
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"POST {url}")
        
        img_bytes = self._image_to_bytes(image)
        
        try:
            response = self.session.post(
                url,
                files={"file": ("image.png", img_bytes, "image/png")},
                data=data or {},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def extract_ocr(
        self,
        image: Image.Image,
        include_boxes: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract text using OCR.
        
        Args:
            image: Image to process
            include_boxes: Whether to include bounding boxes
            
        Returns:
            OCR result with text and optional boxes
        """
        result = self._post_image(
            "/extract/ocr",
            image,
            {"include_boxes": str(include_boxes).lower()},
        )
        
        # Decode visualization if present
        if "visualization" in result and result["visualization"]:
            result["visualization_image"] = self._base64_to_image(result["visualization"])
        
        return result
    
    def extract_table(
        self,
        image: Image.Image,
        output_format: str = "json",
    ) -> Dict[str, Any]:
        """Extract tables from document."""
        result = self._post_image(
            "/extract/table",
            image,
            {"output_format": output_format},
        )
        
        if "visualization" in result and result["visualization"]:
            result["visualization_image"] = self._base64_to_image(result["visualization"])
        
        return result
    
    def extract_form(
        self,
        image: Image.Image,
        extract_signatures: bool = True,
        extract_checkboxes: bool = True,
    ) -> Dict[str, Any]:
        """Extract form fields."""
        result = self._post_image(
            "/extract/form",
            image,
            {
                "extract_signatures": str(extract_signatures).lower(),
                "extract_checkboxes": str(extract_checkboxes).lower(),
            },
        )
        
        if "visualization" in result and result["visualization"]:
            result["visualization_image"] = self._base64_to_image(result["visualization"])
        
        return result
    
    def extract_invoice(
        self,
        image: Image.Image,
        document_type: str = "invoice",
    ) -> Dict[str, Any]:
        """Parse invoice or receipt."""
        result = self._post_image(
            "/extract/invoice",
            image,
            {"document_type": document_type},
        )
        
        if "visualization" in result and result["visualization"]:
            result["visualization_image"] = self._base64_to_image(result["visualization"])
        
        return result
    
    def extract_ner(
        self,
        image: Image.Image,
        entity_types: str = "all",
    ) -> Dict[str, Any]:
        """Extract named entities."""
        result = self._post_image(
            "/extract/ner",
            image,
            {"entity_types": entity_types},
        )
        
        if "visualization" in result and result["visualization"]:
            result["visualization_image"] = self._base64_to_image(result["visualization"])
        
        return result
    
    def analyze_layout(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze document layout."""
        result = self._post_image("/layout", image)
        
        if "visualization" in result and result["visualization"]:
            result["visualization_image"] = self._base64_to_image(result["visualization"])
        
        return result
    
    def detect_objects(
        self,
        image: Image.Image,
        object_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect objects with spatial understanding."""
        data = {}
        if object_type:
            data["object_type"] = object_type
        
        result = self._post_image("/spatial", image, data)
        
        if "visualization" in result and result["visualization"]:
            result["visualization_image"] = self._base64_to_image(result["visualization"])
        
        return result
    
    def recognize(
        self,
        image: Image.Image,
        recognition_type: str = "general",
    ) -> Dict[str, Any]:
        """Recognize content in image."""
        result = self._post_image(
            "/recognition",
            image,
            {"recognition_type": recognition_type},
        )
        
        if "visualization" in result and result["visualization"]:
            result["visualization_image"] = self._base64_to_image(result["visualization"])
        
        return result
    
    def parse_document(
        self,
        image: Image.Image,
        output_format: str = "html",
    ) -> Dict[str, Any]:
        """Parse document structure."""
        result = self._post_image(
            "/document",
            image,
            {"output_format": output_format},
        )
        
        if "visualization" in result and result["visualization"]:
            result["visualization_image"] = self._base64_to_image(result["visualization"])
        
        return result


def create_client(
    base_url: str = "http://localhost:8000",
    timeout: int = 120,
) -> APIClient:
    """Create an API client instance."""
    return APIClient(base_url=base_url, timeout=timeout)


if __name__ == "__main__":
    print("="*60)
    print("API CLIENT TEST")
    print("="*60)
    
    client = create_client()
    
    if client.is_healthy():
        print("✅ API is healthy")
    else:
        print("❌ API is not running")
        print("   Start the API: python main.py --api")
    
    print("="*60)
