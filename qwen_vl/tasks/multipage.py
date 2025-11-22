"""Multi-page document processing support."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from PIL import Image

from .base import BaseTaskHandler, TaskResult, TaskType


@dataclass
class PageResult:
    """Result from processing a single page."""
    page_number: int
    result: TaskResult
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentResult:
    """Result from processing a multi-page document."""
    pages: List[PageResult]
    merged_text: str
    merged_data: Optional[Dict[str, Any]] = None
    document_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_pages(self) -> int:
        return len(self.pages)


class MultiPageProcessor:
    """Process multi-page documents."""

    def __init__(self, handler: BaseTaskHandler):
        """
        Initialize multi-page processor.

        Args:
            handler: Task handler to use for each page
        """
        self.handler = handler

    def process_pages(
        self,
        images: List[Union[str, Image.Image]],
        merge_strategy: str = "concatenate",
        **kwargs,
    ) -> DocumentResult:
        """
        Process multiple pages/images.

        Args:
            images: List of image paths or PIL Images
            merge_strategy: How to merge results (concatenate, structured)
            **kwargs: Additional handler arguments

        Returns:
            DocumentResult with all pages and merged content
        """
        page_results = []

        for i, image in enumerate(images):
            result = self.handler.process(image, **kwargs)
            page_results.append(PageResult(
                page_number=i + 1,
                result=result,
            ))

        # Merge results
        if merge_strategy == "concatenate":
            merged_text, merged_data = self._merge_concatenate(page_results)
        elif merge_strategy == "structured":
            merged_text, merged_data = self._merge_structured(page_results)
        else:
            merged_text = "\n\n".join(pr.result.text for pr in page_results)
            merged_data = None

        return DocumentResult(
            pages=page_results,
            merged_text=merged_text,
            merged_data=merged_data,
            document_metadata={
                "total_pages": len(page_results),
                "merge_strategy": merge_strategy,
            },
        )

    def process_pdf(
        self,
        pdf_path: str,
        dpi: int = 200,
        **kwargs,
    ) -> DocumentResult:
        """
        Process a PDF document.

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion
            **kwargs: Additional handler arguments

        Returns:
            DocumentResult with all pages
        """
        images = self._pdf_to_images(pdf_path, dpi)
        return self.process_pages(images, **kwargs)

    def process_folder(
        self,
        folder_path: str,
        patterns: Optional[List[str]] = None,
        sort_by: str = "name",
        **kwargs,
    ) -> DocumentResult:
        """
        Process all images in a folder as a document.

        Args:
            folder_path: Path to folder
            patterns: File patterns to match
            sort_by: Sort order (name, modified)
            **kwargs: Additional handler arguments

        Returns:
            DocumentResult with all pages
        """
        folder = Path(folder_path)
        patterns = patterns or ["*.png", "*.jpg", "*.jpeg", "*.tiff"]

        files = []
        for pattern in patterns:
            files.extend(folder.glob(pattern))

        # Sort files
        if sort_by == "name":
            files = sorted(files, key=lambda f: f.name)
        elif sort_by == "modified":
            files = sorted(files, key=lambda f: f.stat().st_mtime)

        images = [str(f) for f in files]
        return self.process_pages(images, **kwargs)

    def _merge_concatenate(
        self,
        page_results: List[PageResult],
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """Merge by concatenating text with page markers."""
        parts = []
        all_data = []

        for pr in page_results:
            parts.append(f"--- Page {pr.page_number} ---")
            parts.append(pr.result.text)

            if pr.result.data:
                all_data.append({
                    "page": pr.page_number,
                    "data": pr.result.data,
                })

        merged_text = "\n\n".join(parts)
        merged_data = {"pages": all_data} if all_data else None

        return merged_text, merged_data

    def _merge_structured(
        self,
        page_results: List[PageResult],
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """Merge by combining structured data."""
        all_text = []
        combined_data: Dict[str, Any] = {}

        for pr in page_results:
            all_text.append(pr.result.text)

            if pr.result.data:
                for key, value in pr.result.data.items():
                    if key not in combined_data:
                        combined_data[key] = []

                    if isinstance(value, list):
                        combined_data[key].extend(value)
                    else:
                        combined_data[key].append(value)

        merged_text = "\n\n".join(all_text)
        return merged_text, combined_data if combined_data else None

    def _pdf_to_images(self, pdf_path: str, dpi: int) -> List[Image.Image]:
        """Convert PDF pages to images."""
        try:
            import pdf2image
            return pdf2image.convert_from_path(pdf_path, dpi=dpi)
        except ImportError:
            raise ImportError(
                "pdf2image is required for PDF processing. "
                "Install with: pip install pdf2image"
            )


def detect_document_boundaries(
    images: List[Union[str, Image.Image]],
    threshold: float = 0.5,
) -> List[List[int]]:
    """
    Detect document boundaries in a sequence of images.

    Args:
        images: List of images
        threshold: Similarity threshold for boundary detection

    Returns:
        List of document groups (each group is list of page indices)
    """
    # Simple heuristic: each image is a separate document
    # In production, use visual similarity or content analysis
    return [[i] for i in range(len(images))]


if __name__ == "__main__":
    print("=" * 60)
    print("MULTI-PAGE PROCESSOR TEST")
    print("=" * 60)
    print("  Multi-page processor module loaded")
    print("  Supports: PDF, folder of images")
    print("  Merge strategies: concatenate, structured")
    print("=" * 60)
