"""
Quick Test Script - Single Handler Test

A simple script to quickly test a single handler with one image.
Useful for debugging and quick validation.

Usage:
    python tests/integration/quick_test.py ocr tests/asset/ocr_example1.jpg
    python tests/integration/quick_test.py recognition tests/asset/docparsing_example1.jpg "What is this?"
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    if len(sys.argv) < 3:
        print("Usage: python quick_test.py <task_type> <image_path> [prompt]")
        print("\nAvailable tasks:")
        print("  ocr, layout, recognition, spatial, video, document_parsing,")
        print("  table, ner, form, invoice, contract, computer_agent, mobile_agent")
        print("\nExample:")
        print("  python quick_test.py ocr tests/asset/ocr_example1.jpg")
        sys.exit(1)
    
    task_name = sys.argv[1].lower()
    image_path = Path(sys.argv[2])
    custom_prompt = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    print("=" * 60)
    print(f"  Quick Test: {task_name.upper()}")
    print("=" * 60)
    print(f"  Image: {image_path}")
    if custom_prompt:
        print(f"  Prompt: {custom_prompt}")
    
    # Import and load
    print("\n  Loading model...")
    from PIL import Image
    from qwen_vl.config import load_config
    from qwen_vl.core.model_loader import ModelLoader
    from qwen_vl.tasks import TaskType, get_handler
    
    config = load_config()
    print(f"  Model: {config.model.model_id}")
    
    loader = ModelLoader()
    loaded = loader.load(config)
    
    print("  ✅ Model loaded")
    
    # Get task type
    try:
        task_type = TaskType(task_name)
    except ValueError:
        print(f"  ❌ Unknown task type: {task_name}")
        print(f"  Available: {[t.value for t in TaskType]}")
        sys.exit(1)
    
    # Process
    print(f"\n  Processing with {task_type.value}...")
    
    handler = get_handler(task_type, loaded.model, loaded.processor)
    img = Image.open(image_path)
    
    start = time.time()
    
    if custom_prompt:
        result = handler.process(img, prompt=custom_prompt)
    else:
        result = handler.process(img)
    
    elapsed = time.time() - start
    
    # Output
    print("\n" + "=" * 60)
    print(f"  ⏱️  Time: {elapsed:.2f}s")
    print("=" * 60)
    print("\n📝 RESULT:\n")
    print(result.text)
    
    if result.data:
        print("\n📊 DATA:")
        print(result.data)
    
    if result.visualization:
        out_path = Path("tests/results") / f"quick_{task_name}_{image_path.stem}.png"
        out_path.parent.mkdir(exist_ok=True)
        result.visualization.save(out_path)
        print(f"\n🖼️  Visualization saved: {out_path}")
    
    print("\n" + "=" * 60)
    print("  ✅ Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
