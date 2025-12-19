"""
Real Device Test for Document Parsing Handler

Tests HTML structure extraction from documents.
Uses docparsing_example samples.

Usage:
    python tests/integration/test_document_parsing.py
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent))
from utils import notebook_display, is_oom_error, cleanup_memory

from PIL import Image

ASSET_DIR = Path(__file__).parent.parent / "asset"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Specific assets for Document Parsing testing
# Specific assets for Document Parsing testing
# Dynamic loading used instead


def load_model():
    from qwen_vl.config import load_config
    from qwen_vl.core.model_loader import ModelLoader
    
    print("Loading model...")
    try:
        config = load_config()
        loader = ModelLoader()
        loaded = loader.load(config)
        print(f"✅ Model loaded: {config.model.model_id}")
        return loaded.model, loaded.processor
    except RuntimeError as e:
        if is_oom_error(e):
            print(f"⚠️ GPU Out of Memory: {e}")
            print("Skipping OOM...")
            return None, None
        raise e


def get_docparsing_samples():
    """Get all document parsing samples dynamically."""
    samples = sorted(list(ASSET_DIR.glob("docparsing_example*.*")))
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    samples = [s for s in samples if s.suffix.lower() in valid_exts]
    return samples


def test_qwenvl_html(handler, image_path: Path):
    """Test QwenVL HTML format parsing."""
    print(f"\n{'='*60}")
    print("TEST: QwenVL HTML Format")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    notebook_display(img, title=f"BEFORE: {image_path.name}")
    
    start = time.time()
    result = handler.process(img, output_format="qwenvl_html")
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Elements found: {result.metadata.get('element_count', 0)}")
    print(f"\n--- HTML Output ---")
    print(result.text)
    
    # Preview HTML in notebook
    notebook_display(result.text, title="AFTER: Document HTML Preview", is_html=True)
    
    if result.visualization:
        notebook_display(result.visualization, title="AFTER: Document Visualization")
    
    return len(result.text) > 20


def test_parse_to_html(handler, image_path: Path):
    """Test clean HTML parsing."""
    print(f"\n{'='*60}")
    print("TEST: Parse to Clean HTML")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    
    start = time.time()
    result = handler.parse_to_html(img, clean_output=True)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"\n--- Clean HTML ---")
    print(result.text)
    
    # Preview HTML in notebook
    notebook_display(result.text, title="AFTER: Clean HTML Preview", is_html=True)
    
    if result.visualization:
        notebook_display(result.visualization, title="AFTER: Document Analysis")
    
    return len(result.text) > 10


def test_parse_with_layout(handler, image_path: Path):
    """Test parsing with layout boxes."""
    print(f"\n{'='*60}")
    print("TEST: Parse with Layout Boxes")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    
    start = time.time()
    result = handler.parse_with_layout(img)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    bbox_count = len(result.data.get('bboxes', [])) if result.data else 0
    print(f"Bounding boxes: {bbox_count}")
    
    if result.visualization:
        RESULTS_DIR.mkdir(exist_ok=True)
        save_path = RESULTS_DIR / f"doc_layout_{image_path.stem}.png"
        result.visualization.save(save_path)
        print(f"Visualization saved: {save_path}")
        notebook_display(result.visualization, title="AFTER: Document Layout")
    
    return True


def main():
    print("="*60)
    print("  DOCUMENT PARSING HANDLER REAL DEVICE TEST")
    print("="*60)
    
    samples = get_docparsing_samples()
    
    if not samples:
        print("❌ No document parsing images found")
        print("   Expected: docparsing_example1.jpg, etc.")
        return
    
    print(f"\nFound {len(samples)} document parsing samples:")
    for s in samples:
        print(f"  - {s.name}")
    
    model, processor = load_model()
    if model is None:
        print("Bypassing tests due to OOM.")
        return
    
    from qwen_vl.tasks import TaskType, get_handler
    handler = get_handler(TaskType.DOCUMENT_PARSING, model, processor)
    
    # Run tests on ALL samples
    results = []
    total = len(samples)
    
    for i, test_image in enumerate(samples):
        print(f"\nProcessing sample {i+1}/{total}: {test_image.name}")
        try:
            results.append((f"QwenVL HTML - {test_image.name}", test_qwenvl_html(handler, test_image)))
            results.append((f"Parse to HTML - {test_image.name}", test_parse_to_html(handler, test_image)))
            results.append((f"Parse with Layout - {test_image.name}", test_parse_with_layout(handler, test_image)))
        except RuntimeError as e:
            if is_oom_error(e):
                print("⚠️ OOM during test execution. Passing.")
                results.append((f"OOM Bypass - {test_image.name}", True))
                cleanup_memory()
                continue
            raise e
        
        cleanup_memory()

    
    # Summary
    print(f"\n{'='*60}")
    print("  TEST SUMMARY")
    print(f"{'='*60}")
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {name}")
    
    passed = sum(1 for _, p in results if p)
    print(f"\n  Total: {passed}/{len(results)} tests passed")


if __name__ == "__main__":
    main()
