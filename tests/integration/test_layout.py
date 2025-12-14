"""
Real Device Test for Layout Handler

Tests document layout analysis and structure detection.
Uses docparsing_example samples.

Usage:
    python tests/integration/test_layout.py
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils import notebook_display

from PIL import Image

ASSET_DIR = Path(__file__).parent.parent / "asset"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Specific assets for Layout testing
LAYOUT_SAMPLES = [
    "docparsing_example1.jpg",
    "docparsing_example2.jpg",
    "docparsing_example3.jpg",
    "docparsing_example4.jpg",
    "docparsing_example5.png",
    "docparsing_example6.png",
    "docparsing_example7.jpg",
    "docparsing_example8.png",
]


def load_model():
    from qwen_vl.config import load_config
    from qwen_vl.core.model_loader import ModelLoader
    
    print("Loading model...")
    config = load_config()
    loader = ModelLoader()
    loaded = loader.load(config)
    print(f"✅ Model loaded: {config.model.model_id}")
    return loaded.model, loaded.processor


def get_layout_samples():
    """Get available layout samples."""
    samples = []
    for name in LAYOUT_SAMPLES:
        path = ASSET_DIR / name
        if path.exists():
            samples.append(path)
    return samples


def test_basic_layout(handler, image_path: Path):
    """Test basic layout analysis."""
    print(f"\n{'='*60}")
    print("TEST: Basic Layout Analysis")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    notebook_display(img, title=f"BEFORE: {image_path.name}")
    notebook_display(img, title=f"BEFORE: {image_path.name}")
    
    start = time.time()
    result = handler.process(img)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Elements: {result.metadata.get('element_count', 0)}")
    print(f"\n--- Layout Analysis ---")
    print(result.text[:500] if len(result.text) > 500 else result.text)
    
    if result.visualization:
        RESULTS_DIR.mkdir(exist_ok=True)
        save_path = RESULTS_DIR / f"layout_{image_path.stem}.png"
        result.visualization.save(save_path)
        print(f"Visualization saved: {save_path}")

        try:
            from IPython.display import display
            display(result.visualization)
        except ImportError:
            pass
    
    return len(result.text) > 10


def test_detect_sections(handler, image_path: Path):
    """Test section detection."""
    print(f"\n{'='*60}")
    print("TEST: Detect Sections")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    notebook_display(img, title=f"BEFORE: {image_path.name}")
    
    start = time.time()
    result = handler.detect_sections(img)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Sections: {result.metadata.get('section_count', 0)}")
    print(f"\n--- Sections ---")
    print(result.text[:500] if len(result.text) > 500 else result.text)

    if result.visualization:
        RESULTS_DIR.mkdir(exist_ok=True)
        save_path = RESULTS_DIR / f"layout_sections_{image_path.stem}.png"
        result.visualization.save(save_path)
        print(f"Visualization saved: {save_path}")
        notebook_display(result.visualization, title="AFTER: Layout Sections")
    
    return True


def test_reading_order(handler, image_path: Path):
    """Test reading order detection."""
    print(f"\n{'='*60}")
    print("TEST: Detect Reading Order")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    notebook_display(img, title=f"BEFORE: {image_path.name}")
    
    start = time.time()
    result = handler.detect_reading_order(img)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"\n--- Reading Order ---")
    print(result.text[:500] if len(result.text) > 500 else result.text)

    if result.visualization:
        RESULTS_DIR.mkdir(exist_ok=True)
        save_path = RESULTS_DIR / f"layout_reading_order_{image_path.stem}.png"
        result.visualization.save(save_path)
        print(f"Visualization saved: {save_path}")
        notebook_display(result.visualization, title="AFTER: Reading Order")
    
    return True


def main():
    print("="*60)
    print("  LAYOUT HANDLER REAL DEVICE TEST")
    print("="*60)
    
    samples = get_layout_samples()
    
    if not samples:
        print("❌ No layout test images found")
        print("   Expected: docparsing_example1.jpg, etc.")
        return
    
    print(f"\nFound {len(samples)} layout samples:")
    for s in samples:
        print(f"  - {s.name}")
    
    model, processor = load_model()
    
    from qwen_vl.tasks import TaskType, get_handler
    handler = get_handler(TaskType.LAYOUT, model, processor)
    
    results = []
    
    # Test on first sample
    test_image = samples[0]
    
    results.append(("Basic Layout", test_basic_layout(handler, test_image)))
    results.append(("Detect Sections", test_detect_sections(handler, test_image)))
    results.append(("Reading Order", test_reading_order(handler, test_image)))
    
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
