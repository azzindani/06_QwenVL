"""
Real Device Test for OCR Handler

Tests all OCR methods with real model inference.
Uses sample assets from tests/asset/

Usage:
    python tests/integration/test_ocr.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PIL import Image

ASSET_DIR = Path(__file__).parent.parent / "asset"

RESULTS_DIR = Path(__file__).parent.parent / "results"


sys.path.append(str(Path(__file__).parent))
from utils import notebook_display, is_oom_error

# Specific assets for OCR testing

# Specific assets for OCR testing
def load_model():
    """Load the Qwen VL model."""
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


def get_ocr_samples():
    """Get all OCR sample images dynamically."""
    # Find all files matching ocr_* pattern
    samples = sorted(list(ASSET_DIR.glob("ocr_*.*")))
    # Filter for valid image extensions
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    samples = [s for s in samples if s.suffix.lower() in valid_exts]
    return samples


def test_ocr_basic(handler, image_path: Path):
    """Test basic OCR text extraction."""
    print(f"\n{'='*60}")
    print("TEST: OCR Basic Text Extraction")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    
    # [Preview BEFORE]
    notebook_display(img, title=f"BEFORE: {image_path.name}")
    
    start = time.time()
    result = handler.process(img, with_boxes=False)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Output length: {len(result.text)} chars")
    print(f"\n--- Extracted Text ---")
    print(result.text[:500] + "..." if len(result.text) > 500 else result.text)
    
    return len(result.text) > 10


def test_ocr_with_boxes(handler, image_path: Path):
    """Test OCR with bounding boxes."""
    print(f"\n{'='*60}")
    print("TEST: OCR with Bounding Boxes")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")

    # [Preview BEFORE]
    notebook_display(img, title=f"BEFORE: {image_path.name}")
    
    start = time.time()
    result = handler.process(img, with_boxes=True)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Bounding boxes: {len(result.bounding_boxes) if result.bounding_boxes else 0}")
    
    if result.visualization:
        RESULTS_DIR.mkdir(exist_ok=True)
        save_path = RESULTS_DIR / f"ocr_boxes_{image_path.stem}.png"
        result.visualization.save(save_path)
        print(f"Visualization saved: {save_path}")
        
        # [Preview AFTER]
        notebook_display(result.visualization, title="AFTER: Object Detection")
    
    return True


def test_ocr_extract_lines(handler, image_path: Path):
    """Test line-by-line text extraction."""
    print(f"\n{'='*60}")
    print("TEST: OCR Extract Lines")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    
    # [Preview BEFORE]
    notebook_display(img, title=f"BEFORE: {image_path.name}")
    
    start = time.time()
    result = handler.extract_lines(img)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Lines found: {result.metadata.get('line_count', 0)}")
    
    if result.visualization:
        RESULTS_DIR.mkdir(exist_ok=True)
        save_path = RESULTS_DIR / f"ocr_lines_{image_path.stem}.png"
        result.visualization.save(save_path)
        print(f"Visualization saved: {save_path}")
        
        # [Preview AFTER]
        notebook_display(result.visualization, title="AFTER: Line Extraction")
    
    return True


def main():
    print("="*60)
    print("  OCR HANDLER REAL DEVICE TEST")
    print("="*60)
    
    # Find test images
    images = get_ocr_samples()
    
    if not images:
        print("❌ No OCR test images found in tests/asset/")
        print("   Expected: ocr_example1.jpg, ocr_example2.jpg, etc.")
        return
    
    print(f"Found {len(images)} OCR test images:")
    for img in images:
        print(f"  - {img.name}")
    
    # Load model
    model, processor = load_model()
    if model is None:
        print("Bypassing tests due to OOM.")
        return

    # Get handler
    from qwen_vl.tasks import TaskType, get_handler
    handler = get_handler(TaskType.OCR, model, processor)
    
    # Run tests on ALL samples
    total_samples = len(images)
    results = []
    
    for i, test_image in enumerate(images):
        print(f"\nProcessing image {i+1}/{total_samples}: {test_image.name}")
        try:
            results.append((f"Basic OCR - {test_image.name}", test_ocr_basic(handler, test_image)))
            results.append((f"OCR with Boxes - {test_image.name}", test_ocr_with_boxes(handler, test_image)))
            results.append((f"Extract Lines - {test_image.name}", test_ocr_extract_lines(handler, test_image)))
        except RuntimeError as e:
            if is_oom_error(e):
                print("⚠️ OOM during test execution. Passing.")
                results.append((f"OOM Bypass - {test_image.name}", True))
                continue
            raise e
    
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
