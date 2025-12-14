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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PIL import Image

ASSET_DIR = Path(__file__).parent.parent / "asset"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Specific assets for Document Parsing testing
DOCPARSING_SAMPLES = [
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


def get_docparsing_samples():
    """Get available document parsing samples."""
    samples = []
    for name in DOCPARSING_SAMPLES:
        path = ASSET_DIR / name
        if path.exists():
            samples.append(path)
    return samples


def test_qwenvl_html(handler, image_path: Path):
    """Test QwenVL HTML format parsing."""
    print(f"\n{'='*60}")
    print("TEST: QwenVL HTML Format")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    
    start = time.time()
    result = handler.process(img, output_format="qwenvl_html")
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Elements found: {result.metadata.get('element_count', 0)}")
    print(f"\n--- HTML Output (first 500 chars) ---")
    print(result.text[:500])
    
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
    print(f"\n--- Clean HTML (first 500 chars) ---")
    print(result.text[:500])
    
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

        try:
            from IPython.display import display
            display(result.visualization)
        except ImportError:
            pass
    
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
    
    from qwen_vl.tasks import TaskType, get_handler
    handler = get_handler(TaskType.DOCUMENT_PARSING, model, processor)
    
    results = []
    
    # Test on first sample
    test_image = samples[0]
    
    results.append(("QwenVL HTML", test_qwenvl_html(handler, test_image)))
    results.append(("Parse to HTML", test_parse_to_html(handler, test_image)))
    results.append(("Parse with Layout", test_parse_with_layout(handler, test_image)))
    
    # Test on additional samples
    if len(samples) > 1:
        print(f"\n{'='*60}")
        print("TESTING ADDITIONAL SAMPLES")
        print(f"{'='*60}")
        for sample in samples[1:3]:  # Test 2 more samples
            result = test_qwenvl_html(handler, sample)
            results.append((f"QwenVL HTML - {sample.name}", result))
    
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
