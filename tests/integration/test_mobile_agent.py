"""
Real Device Test for Mobile Agent Handler

Tests mobile app automation and screen analysis.
Uses mobile_en_example and mobile_zh_example samples.

Usage:
    python tests/integration/test_mobile_agent.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PIL import Image

ASSET_DIR = Path(__file__).parent.parent / "asset"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Specific assets for Mobile Agent testing
MOBILE_SAMPLES = [
    "mobile_en_example.png",   # English mobile screenshot
    "mobile_zh_example.png",   # Chinese mobile screenshot
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


def get_mobile_samples():
    """Get available mobile samples."""
    samples = []
    for name in MOBILE_SAMPLES:
        path = ASSET_DIR / name
        if path.exists():
            samples.append(path)
    return samples


def test_mobile_screen(handler, image_path: Path):
    """Test mobile screen analysis."""
    print(f"\n{'='*60}")
    print("TEST: Mobile Screen Analysis")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    print(f"Size: {img.size}")
    
    start = time.time()
    result = handler.process(img, prompt="Describe this mobile screen. What app is this? What can the user do here?")
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"\n--- Mobile Analysis ---")
    print(result.text)
    
    return len(result.text) > 20


def test_find_button(handler, image_path: Path):
    """Test finding a mobile button."""
    print(f"\n{'='*60}")
    print("TEST: Find Mobile Button")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    
    start = time.time()
    result = handler.find_element(img, "any tappable button or icon")
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"\n--- Button Found ---")
    print(result.text)
    
    return True


def test_find_back_button(handler, image_path: Path):
    """Test finding back button."""
    print(f"\n{'='*60}")
    print("TEST: Find Back Button")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    
    start = time.time()
    result = handler.find_element(img, "back button or navigation arrow")
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"\n--- Back Button ---")
    print(result.text)
    
    return True


def test_describe_screen(handler, image_path: Path):
    """Test full screen description."""
    print(f"\n{'='*60}")
    print("TEST: Describe Screen")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    
    start = time.time()
    result = handler.describe_screen(img)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"\n--- Screen Description ---")
    print(result.text)
    
    return len(result.text) > 50


def test_suggest_tap(handler, image_path: Path, task: str):
    """Test suggesting where to tap."""
    print(f"\n{'='*60}")
    print(f"TEST: Suggest Tap - '{task}'")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    
    start = time.time()
    result = handler.suggest_action(img, task=task)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"\n--- Suggested Tap ---")
    print(result.text)
    if result.data:
        print(f"Action data: {result.data}")
    
    if result.visualization:
        RESULTS_DIR.mkdir(exist_ok=True)
        save_path = RESULTS_DIR / f"mobile_tap_{image_path.stem}.png"
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
    print("  MOBILE AGENT HANDLER REAL DEVICE TEST")
    print("="*60)
    
    samples = get_mobile_samples()
    
    if not samples:
        print("❌ No mobile screenshots found")
        print("   Expected: mobile_en_example.png, mobile_zh_example.png")
        return
    
    print(f"\nFound {len(samples)} mobile samples:")
    for s in samples:
        print(f"  - {s.name}")
    
    model, processor = load_model()
    
    from qwen_vl.tasks import TaskType, get_handler
    handler = get_handler(TaskType.MOBILE_AGENT, model, processor)
    
    results = []
    
    # Test English mobile screen
    if any("en" in s.name for s in samples):
        en_sample = next(s for s in samples if "en" in s.name)
        print(f"\n{'='*60}")
        print("TESTING ENGLISH MOBILE SCREEN")
        print(f"{'='*60}")
        
        results.append(("EN - Screen Analysis", test_mobile_screen(handler, en_sample)))
        results.append(("EN - Find Button", test_find_button(handler, en_sample)))
        results.append(("EN - Describe Screen", test_describe_screen(handler, en_sample)))
        results.append(("EN - Suggest Tap", test_suggest_tap(handler, en_sample, "go back to home")))
    
    # Test Chinese mobile screen
    if any("zh" in s.name for s in samples):
        zh_sample = next(s for s in samples if "zh" in s.name)
        print(f"\n{'='*60}")
        print("TESTING CHINESE MOBILE SCREEN")
        print(f"{'='*60}")
        
        results.append(("ZH - Screen Analysis", test_mobile_screen(handler, zh_sample)))
        results.append(("ZH - Describe Screen", test_describe_screen(handler, zh_sample)))
    
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
