"""
Real Device Test for Recognition Handler

Tests image recognition capabilities with real model inference.
Uses celebrity, animal, food, and scene samples.

Usage:
    python tests/integration/test_recognition.py
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils import notebook_display

from PIL import Image

ASSET_DIR = Path(__file__).parent.parent / "asset"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Specific assets for Recognition testing
RECOGNITION_SAMPLES = {
    "celebrity": [
        "sample-celebrity.jpeg",
        "sample-celebrity-2.jpg",
    ],
    "animal": [
        "sample-bird.jpg",
    ],
    "food": [
        "sample-food.jpeg",
    ],
    "anime": [
        "sample-anime.jpeg",
    ],
    "scene": [
        "football_field.jpg",
        "office.jpg",
        "lounge.jpg",
        "dining_table.png",
    ],
    "vehicles": [
        "autonomous_driving.jpg",
        "lots_of_cars.png",
        "drone_cars2.png",
    ],
    "people": [
        "lots_of_people.jpeg",
    ],
}


def load_model():
    from qwen_vl.config import load_config
    from qwen_vl.core.model_loader import ModelLoader
    
    print("Loading model...")
    config = load_config()
    loader = ModelLoader()
    loaded = loader.load(config)
    print(f"✅ Model loaded: {config.model.model_id}")
    return loaded.model, loaded.processor


def get_sample(category: str) -> Path:
    """Get first available sample from category."""
    for name in RECOGNITION_SAMPLES.get(category, []):
        path = ASSET_DIR / name
        if path.exists():
            return path
    return None


def test_celebrity_recognition(handler):
    """Test celebrity recognition."""
    print(f"\n{'='*60}")
    print("TEST: Celebrity Recognition")
    print(f"{'='*60}")
    
    image_path = get_sample("celebrity")
    if not image_path:
        print("⚠️ No celebrity sample found, skipping")
        return None
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    notebook_display(img, title=f"BEFORE: {image_path.name}")
    
    start = time.time()
    result = handler.process(img, prompt="Who is this person? Identify them if possible.")
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"\n--- Recognition Result ---")
    print(result.text)

    if result.visualization:
        RESULTS_DIR.mkdir(exist_ok=True)
        save_path = RESULTS_DIR / f"rec_celeb_{image_path.stem}.png"
        result.visualization.save(save_path)
        print(f"Visualization saved: {save_path}")
        notebook_display(result.visualization, title="AFTER: Celebrity Recognition")
    
    return len(result.text) > 20


def test_animal_recognition(handler):
    """Test animal recognition."""
    print(f"\n{'='*60}")
    print("TEST: Animal Recognition")
    print(f"{'='*60}")
    
    image_path = get_sample("animal")
    if not image_path:
        print("⚠️ No animal sample found, skipping")
        return None
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    notebook_display(img, title=f"BEFORE: {image_path.name}")
    
    start = time.time()
    result = handler.process(img, prompt="What animal is this? Identify the species.")
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"\n--- Recognition Result ---")
    print(result.text)

    if result.visualization:
        RESULTS_DIR.mkdir(exist_ok=True)
        save_path = RESULTS_DIR / f"rec_animal_{image_path.stem}.png"
        result.visualization.save(save_path)
        print(f"Visualization saved: {save_path}")
        notebook_display(result.visualization, title="AFTER: Animal Recognition")
    
    return len(result.text) > 10


def test_food_recognition(handler):
    """Test food recognition."""
    print(f"\n{'='*60}")
    print("TEST: Food Recognition")
    print(f"{'='*60}")
    
    image_path = get_sample("food")
    if not image_path:
        print("⚠️ No food sample found, skipping")
        return None
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    notebook_display(img, title=f"BEFORE: {image_path.name}")
    
    start = time.time()
    result = handler.process(img, prompt="What food is shown? List all dishes visible.")
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"\n--- Recognition Result ---")
    print(result.text)

    if result.visualization:
        RESULTS_DIR.mkdir(exist_ok=True)
        save_path = RESULTS_DIR / f"rec_food_{image_path.stem}.png"
        result.visualization.save(save_path)
        print(f"Visualization saved: {save_path}")
        notebook_display(result.visualization, title="AFTER: Food Recognition")
    
    return len(result.text) > 10


def test_scene_recognition(handler):
    """Test scene recognition."""
    print(f"\n{'='*60}")
    print("TEST: Scene Recognition")
    print(f"{'='*60}")
    
    image_path = get_sample("scene")
    if not image_path:
        print("⚠️ No scene sample found, skipping")
        return None
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    notebook_display(img, title=f"BEFORE: {image_path.name}")
    
    start = time.time()
    result = handler.process(img, prompt="Describe this scene. What kind of place is this?")
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"\n--- Scene Description ---")
    print(result.text)

    if result.visualization:
        RESULTS_DIR.mkdir(exist_ok=True)
        save_path = RESULTS_DIR / f"rec_scene_{image_path.stem}.png"
        result.visualization.save(save_path)
        print(f"Visualization saved: {save_path}")
        notebook_display(result.visualization, title="AFTER: Scene Recognition")
    
    return len(result.text) > 20


def test_identify_objects(handler):
    """Test object identification."""
    print(f"\n{'='*60}")
    print("TEST: Identify Objects")
    print(f"{'='*60}")
    
    image_path = get_sample("scene") or get_sample("vehicles")
    if not image_path:
        print("⚠️ No suitable sample found, skipping")
        return None
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    notebook_display(img, title=f"BEFORE: {image_path.name}")
    
    start = time.time()
    result = handler.identify_objects(img)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"\n--- Objects Found ---")
    print(result.text)

    if result.visualization:
        RESULTS_DIR.mkdir(exist_ok=True)
        save_path = RESULTS_DIR / f"rec_objects_{image_path.stem}.png"
        result.visualization.save(save_path)
        print(f"Visualization saved: {save_path}")
        notebook_display(result.visualization, title="AFTER: Object Identification")
    
    return len(result.text) > 10


def test_identify_landmarks(handler):
    """Test landmark identification."""
    print(f"\n{'='*60}")
    print("TEST: Identify Landmarks")
    print(f"{'='*60}")
    
    image_path = get_sample("scene")
    if not image_path:
        print("⚠️ No scene sample found, skipping")
        return None
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    
    start = time.time()
    result = handler.identify_landmarks(img)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"\n--- Landmarks ---")
    print(result.text)

    if result.visualization:
        RESULTS_DIR.mkdir(exist_ok=True)
        save_path = RESULTS_DIR / f"rec_landmarks_{image_path.stem}.png"
        result.visualization.save(save_path)
        print(f"Visualization saved: {save_path}")
        notebook_display(result.visualization, title="AFTER: Landmark Recognition")
    
    return True


def main():
    print("="*60)
    print("  RECOGNITION HANDLER REAL DEVICE TEST")
    print("="*60)
    
    # Show available samples
    print("\nAvailable samples:")
    for category, files in RECOGNITION_SAMPLES.items():
        available = [f for f in files if (ASSET_DIR / f).exists()]
        print(f"  {category}: {len(available)}/{len(files)} files")
    
    model, processor = load_model()
    
    from qwen_vl.tasks import TaskType, get_handler
    handler = get_handler(TaskType.RECOGNITION, model, processor)
    
    results = []
    
    # Run all tests
    result = test_celebrity_recognition(handler)
    if result is not None:
        results.append(("Celebrity Recognition", result))
    
    result = test_animal_recognition(handler)
    if result is not None:
        results.append(("Animal Recognition", result))
    
    result = test_food_recognition(handler)
    if result is not None:
        results.append(("Food Recognition", result))
    
    result = test_scene_recognition(handler)
    if result is not None:
        results.append(("Scene Recognition", result))
    
    result = test_identify_objects(handler)
    if result is not None:
        results.append(("Identify Objects", result))
    
    result = test_identify_landmarks(handler)
    if result is not None:
        results.append(("Identify Landmarks", result))
    
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
