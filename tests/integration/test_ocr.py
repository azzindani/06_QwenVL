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
from utils import notebook_display

# Specific assets for OCR testing

# Specific assets for OCR testing
OCR_SAMPLES = [
    "ocr_example1.jpg",
    "ocr_example2.jpg",
    "ocr_example3.jpg",
    "ocr_example4.jpg",
    "ocr_example5.jpg",
    "ocr_example6.jpg",
]


def load_model():
    """Load the Qwen VL model."""
    from qwen_vl.config import load_config
    from qwen_vl.core.model_loader import ModelLoader
    
    print("Loading model...")
    config = load_config()
    loader = ModelLoader()
    loaded = loader.load(config)
    print(f"✅ Model loaded: {config.model.model_id}")
    return loaded.model, loaded.processor


def get_ocr_samples():
    """Get OCR sample images."""
    samples = []
    for name in OCR_SAMPLES:
        path = ASSET_DIR / name
        if path.exists():
            samples.append(path)
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
    
    # Get handler
    from qwen_vl.tasks import TaskType, get_handler
    handler = get_handler(TaskType.OCR, model, processor)
    
    # Run tests on first sample
    test_image = images[0]
    results = []
    
    results.append(("Basic OCR", test_ocr_basic(handler, test_image)))
    results.append(("OCR with Boxes", test_ocr_with_boxes(handler, test_image)))
    results.append(("Extract Lines", test_ocr_extract_lines(handler, test_image)))
    
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
