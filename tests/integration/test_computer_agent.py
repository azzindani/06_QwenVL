"""
Real Device Test for Computer Agent Handler

Tests desktop automation and screenshot analysis.
Uses computer_use samples.

Usage:
    python tests/integration/test_computer_agent.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PIL import Image

ASSET_DIR = Path(__file__).parent.parent / "asset"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Specific assets for Computer Agent testing
COMPUTER_SAMPLES = [
    "computer_use1.jpeg",    # Desktop screenshot 1
    "computer_use2.jpeg",    # Desktop screenshot 2
    "screenshot_demo.png",   # Demo screenshot
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


def get_computer_samples():
    """Get available computer use samples."""
    samples = []
    for name in COMPUTER_SAMPLES:
        path = ASSET_DIR / name
        if path.exists():
            samples.append(path)
    return samples


def test_screen_analysis(handler, image_path: Path):
    """Test screenshot analysis."""
    print(f"\n{'='*60}")
    print("TEST: Screen Analysis")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    print(f"Size: {img.size}")
    
    start = time.time()
    result = handler.process(img, prompt="Describe this desktop screen. What application is open? What UI elements are visible?")
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"\n--- Screen Analysis ---")
    print(result.text)
    
    return len(result.text) > 20


def test_find_button(handler, image_path: Path):
    """Test finding a button."""
    print(f"\n{'='*60}")
    print("TEST: Find Button")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    
    start = time.time()
    result = handler.find_element(img, "any clickable button")
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"\n--- Button Found ---")
    print(result.text)
    
    return True


def test_find_text_input(handler, image_path: Path):
    """Test finding text input field."""
    print(f"\n{'='*60}")
    print("TEST: Find Text Input")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    
    start = time.time()
    result = handler.find_element(img, "text input field or search bar")
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"\n--- Text Input Found ---")
    print(result.text)
    
    return True


def test_suggest_action(handler, image_path: Path, task: str):
    """Test action suggestion."""
    print(f"\n{'='*60}")
    print(f"TEST: Suggest Action - '{task}'")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    
    start = time.time()
    result = handler.suggest_action(img, task=task)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"\n--- Suggested Action ---")
    print(result.text)
    if result.data:
        print(f"Action data: {result.data}")
    
    if result.visualization:
        RESULTS_DIR.mkdir(exist_ok=True)
        save_path = RESULTS_DIR / f"computer_action_{image_path.stem}.png"
        result.visualization.save(save_path)
        print(f"Visualization saved: {save_path}")
    
    return True


def main():
    print("="*60)
    print("  COMPUTER AGENT HANDLER REAL DEVICE TEST")
    print("="*60)
    
    samples = get_computer_samples()
    
    if not samples:
        print("❌ No computer use screenshots found")
        print("   Expected: computer_use1.jpeg, computer_use2.jpeg, screenshot_demo.png")
        return
    
    print(f"\nFound {len(samples)} computer use samples:")
    for s in samples:
        print(f"  - {s.name}")
    
    model, processor = load_model()
    
    from qwen_vl.tasks import TaskType, get_handler
    handler = get_handler(TaskType.COMPUTER_AGENT, model, processor)
    
    results = []
    
    # Test on first sample
    test_image = samples[0]
    
    results.append(("Screen Analysis", test_screen_analysis(handler, test_image)))
    results.append(("Find Button", test_find_button(handler, test_image)))
    results.append(("Find Text Input", test_find_text_input(handler, test_image)))
    results.append(("Suggest Action - Close Window", test_suggest_action(handler, test_image, "close this window")))
    results.append(("Suggest Action - Open Settings", test_suggest_action(handler, test_image, "open settings or preferences")))
    
    # Test on second sample if available
    if len(samples) > 1:
        print(f"\n{'='*60}")
        print("TESTING ON SECOND SAMPLE")
        print(f"{'='*60}")
        results.append(("Screen Analysis #2", test_screen_analysis(handler, samples[1])))
    
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
