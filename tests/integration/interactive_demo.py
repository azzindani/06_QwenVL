"""
Interactive Demo Script for Qwen VL Task Handlers

This script provides an interactive command-line interface to test
individual task handlers with your own prompts and images.

Usage:
    python tests/integration/interactive_demo.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

ASSET_DIR = Path(__file__).parent.parent / "asset"


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_menu():
    print("\n" + "=" * 60)
    print("  Qwen VL Interactive Demo")
    print("=" * 60)
    print("""
  Available Tasks:
    1. OCR - Extract text from images
    2. Layout - Analyze document structure
    3. Recognition - Identify objects and content
    4. Spatial - Detect objects with bounding boxes
    5. Video - Analyze video content
    6. Document Parsing - Parse document to HTML
    7. Table - Extract table data
    8. NER - Named Entity Recognition
    9. Form - Extract form fields
    10. Invoice - Parse invoice data
    11. Contract - Analyze contracts
    12. Computer Agent - Desktop automation
    13. Mobile Agent - Mobile automation
    
    0. Exit
    """)


def list_assets():
    print("\n  Available sample images:")
    for i, f in enumerate(sorted(ASSET_DIR.glob("*.*"))):
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            print(f"    {i+1}. {f.name}")
    return list(sorted(ASSET_DIR.glob("*.*")))


def get_task_type(choice: int):
    from qwen_vl.tasks import TaskType
    
    task_map = {
        1: TaskType.OCR,
        2: TaskType.LAYOUT,
        3: TaskType.RECOGNITION,
        4: TaskType.SPATIAL,
        5: TaskType.VIDEO,
        6: TaskType.DOCUMENT_PARSING,
        7: TaskType.TABLE,
        8: TaskType.NER,
        9: TaskType.FORM,
        10: TaskType.INVOICE,
        11: TaskType.CONTRACT,
        12: TaskType.COMPUTER_AGENT,
        13: TaskType.MOBILE_AGENT,
    }
    return task_map.get(choice)


def run_interactive():
    print("\n" + "=" * 60)
    print("  Loading Qwen VL Model...")
    print("=" * 60)
    
    from qwen_vl.config import load_config
    from qwen_vl.core.model_loader import ModelLoader
    from qwen_vl.tasks import get_handler
    from PIL import Image
    
    config = load_config()
    loader = ModelLoader()
    
    print(f"  Model: {config.model.model_id}")
    print(f"  Loading...")
    
    loaded = loader.load(config)
    model = loaded.model
    processor = loaded.processor
    
    print("  ✅ Model loaded successfully!")
    
    while True:
        print_menu()
        
        try:
            choice = input("  Select task (0-13): ").strip()
            if choice == "0":
                print("\n  Goodbye!")
                break
            
            choice = int(choice)
            task_type = get_task_type(choice)
            
            if not task_type:
                print("  Invalid choice. Try again.")
                continue
            
            print(f"\n  Selected: {task_type.value}")
            
            # List assets
            assets = list_assets()
            image_assets = [a for a in assets if a.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            
            if not image_assets:
                print("  No image assets found!")
                continue
            
            img_choice = input("\n  Select image number (or path): ").strip()
            
            try:
                img_idx = int(img_choice) - 1
                if 0 <= img_idx < len(image_assets):
                    img_path = image_assets[img_idx]
                else:
                    print("  Invalid image selection.")
                    continue
            except ValueError:
                img_path = Path(img_choice)
                if not img_path.exists():
                    print(f"  Image not found: {img_path}")
                    continue
            
            # Get custom prompt
            custom_prompt = input("\n  Custom prompt (or Enter for default): ").strip()
            
            # Process
            print(f"\n  Processing {img_path.name}...")
            print("  " + "-" * 50)
            
            handler = get_handler(task_type, model, processor)
            img = Image.open(img_path)
            
            import time
            start = time.time()
            
            if custom_prompt:
                result = handler.process(img, prompt=custom_prompt)
            else:
                result = handler.process(img)
            
            elapsed = time.time() - start
            
            print(f"\n  ⏱️  Time: {elapsed:.2f}s")
            print("\n  📝 Result:")
            print("  " + "-" * 50)
            print(result.text)
            print("  " + "-" * 50)
            
            if result.data:
                print(f"\n  📊 Data: {result.data}")
            
            if result.visualization:
                save_path = ASSET_DIR.parent / "results" / f"viz_{task_type.value}_{img_path.stem}.png"
                save_path.parent.mkdir(exist_ok=True)
                result.visualization.save(save_path)
                print(f"\n  🖼️  Visualization saved: {save_path}")
            
            input("\n  Press Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n  Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            input("\n  Press Enter to continue...")


if __name__ == "__main__":
    run_interactive()
