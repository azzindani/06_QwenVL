"""
Real Device Test for Spatial Handler

Tests object detection and localization with bounding boxes.
Uses spatio_case samples and scene images.

Usage:
    python tests/integration/test_spatial.py
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

# Specific assets for Spatial testing
SPATIAL_SAMPLES = [
    "spatio_case1.jpg",           # Specific spatial case
    "spatio_case2_aff.png",       # Affordance detection
    "spatio_case2_aff2.png",      # Affordance detection 2
    "spatio_case2_plan.png",      # Planning
    "spatio_case2_plan2.png",     # Planning 2
    "dining_table.png",           # Objects on table
    "football_field.jpg",         # Players on field
    "lots_of_cars.png",           # Multiple cars
    "lots_of_people.jpeg",        # Multiple people
    "autonomous_driving.jpg",     # Driving scene
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


def get_spatial_samples():
    """Get available spatial samples."""
    samples = []
    for name in SPATIAL_SAMPLES:
        path = ASSET_DIR / name
        if path.exists():
            samples.append(path)
    return samples


def test_basic_spatial(handler, image_path: Path):
    """Test basic spatial detection."""
    print(f"\n{'='*60}")
    print("TEST: Basic Spatial Detection")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    notebook_display(img, title=f"BEFORE: {image_path.name}")
    
    start = time.time()
    result = handler.process(img)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"\n--- Detection Result ---")
    print(result.text[:500] if len(result.text) > 500 else result.text)
    
    return len(result.text) > 10


def test_detect_objects(handler, image_path: Path, object_type: str = None):
    """Test object detection with bounding boxes."""
    print(f"\n{'='*60}")
    print(f"TEST: Detect Objects{' (' + object_type + ')' if object_type else ''}")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    notebook_display(img, title=f"BEFORE: {image_path.name}")
    
    start = time.time()
    result = handler.detect_objects(img, object_type=object_type)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"\n--- Objects Detected ---")
    print(result.text[:500] if len(result.text) > 500 else result.text)

    if result.visualization:
        RESULTS_DIR.mkdir(exist_ok=True)
        save_path = RESULTS_DIR / f"spatial_objects_{image_path.stem}.png"
        result.visualization.save(save_path)
        print(f"Visualization saved: {save_path}")
        notebook_display(result.visualization, title="AFTER: Spatial Visual")
    
    return True


def test_detect_cars(handler):
    """Test detecting cars specifically."""
    print(f"\n{'='*60}")
    print("TEST: Detect Cars")
    print(f"{'='*60}")
    
    # Use car-heavy images
    for name in ["lots_of_cars.png", "autonomous_driving.jpg", "drone_cars2.png"]:
        path = ASSET_DIR / name
        if path.exists():
            img = Image.open(path)
            print(f"Image: {name}")
            notebook_display(img, title=f"BEFORE: {name}")
            
            start = time.time()
            result = handler.detect_objects(img, object_type="car")
            elapsed = time.time() - start
            
            print(f"Time: {elapsed:.2f}s")
            print(f"\n--- Cars Detected ---")
            print(result.text[:300])
            
            if result.visualization:
                RESULTS_DIR.mkdir(exist_ok=True)
                save_path = RESULTS_DIR / f"spatial_cars_{path.stem}.png"
                result.visualization.save(save_path)
                print(f"Visualization saved: {save_path}")
        notebook_display(result.visualization, title="AFTER: Spatial Visual")

            return True
    
    print("⚠️ No car images found")
    return None


def test_detect_people(handler):
    """Test detecting people specifically."""
    print(f"\n{'='*60}")
    print("TEST: Detect People")
    print(f"{'='*60}")
    
    # Use people-heavy images
    for name in ["lots_of_people.jpeg", "football_field.jpg"]:
        path = ASSET_DIR / name
        if path.exists():
            img = Image.open(path)
            print(f"Image: {name}")
            notebook_display(img, title=f"BEFORE: {name}")
            
            start = time.time()
            result = handler.detect_objects(img, object_type="person")
            elapsed = time.time() - start
            
            print(f"Time: {elapsed:.2f}s")
            print(f"\n--- People Detected ---")
            print(result.text[:300])

            if result.visualization:
                RESULTS_DIR.mkdir(exist_ok=True)
                save_path = RESULTS_DIR / f"spatial_people_{path.stem}.png"
                result.visualization.save(save_path)
                print(f"Visualization saved: {save_path}")
        notebook_display(result.visualization, title="AFTER: Spatial Visual")

            return True
    
    print("⚠️ No people images found")
    return None


def test_point_to_object(handler, image_path: Path, target: str):
    """Test pointing to specific object."""
    print(f"\n{'='*60}")
    print(f"TEST: Point to Object - '{target}'")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    notebook_display(img, title=f"BEFORE: {image_path.name}")
    
    start = time.time()
    result = handler.point_to_object(img, target)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"\n--- Point Result ---")
    print(result.text)
    if result.data:
        print(f"Data: {result.data}")
    
    if result.visualization:
        RESULTS_DIR.mkdir(exist_ok=True)
        save_path = RESULTS_DIR / f"spatial_point_{image_path.stem}.png"
        result.visualization.save(save_path)
        print(f"Visualization saved: {save_path}")
        notebook_display(result.visualization, title="AFTER: Spatial Visual")
    
    return True


def main():
    print("="*60)
    print("  SPATIAL HANDLER REAL DEVICE TEST")
    print("="*60)
    
    samples = get_spatial_samples()
    
    if not samples:
        print("❌ No spatial test images found")
        return
    
    print(f"\nFound {len(samples)} spatial samples:")
    for s in samples[:5]:
        print(f"  - {s.name}")
    if len(samples) > 5:
        print(f"  ... and {len(samples) - 5} more")
    
    model, processor = load_model()
    
    from qwen_vl.tasks import TaskType, get_handler
    handler = get_handler(TaskType.SPATIAL, model, processor)
    
    results = []
    
    # Basic spatial on first sample
    results.append(("Basic Spatial", test_basic_spatial(handler, samples[0])))
    
    # Detect all objects
    results.append(("Detect Objects", test_detect_objects(handler, samples[0])))
    
    # Detect specific objects
    result = test_detect_cars(handler)
    if result is not None:
        results.append(("Detect Cars", result))
    
    result = test_detect_people(handler)
    if result is not None:
        results.append(("Detect People", result))
    
    # Point to object
    results.append(("Point to Object", test_point_to_object(handler, samples[0], "the main subject")))
    
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
