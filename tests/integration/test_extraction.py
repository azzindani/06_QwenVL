"""
Real Device Test for Table, NER, Form, Invoice, Contract Handlers

Tests structured extraction handlers.

Usage:
    python tests/integration/test_extraction.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PIL import Image

ASSET_DIR = Path(__file__).parent.parent / "asset"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_model():
    from qwen_vl.config import load_config
    from qwen_vl.core.model_loader import ModelLoader
    
    print("Loading model...")
    config = load_config()
    loader = ModelLoader()
    loaded = loader.load(config)
    print(f"✅ Model loaded: {config.model.model_id}")
    return loaded.model, loaded.processor


def test_handler(handler, handler_name: str, image_path: Path):
    """Generic test for a handler."""
    print(f"\n{'='*60}")
    print(f"TEST: {handler_name}")
    print(f"{'='*60}")
    
    img = Image.open(image_path)
    print(f"Image: {image_path.name}")
    
    start = time.time()
    result = handler.process(img)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"\n--- Result ---")
    print(result.text[:500] if len(result.text) > 500 else result.text)

    if result.visualization:
        RESULTS_DIR.mkdir(exist_ok=True)
        save_path = RESULTS_DIR / f"{handler_name.lower().replace(' ', '_')}_{image_path.stem}.png"
        result.visualization.save(save_path)
        print(f"Visualization saved: {save_path}")

        try:
            from IPython.display import display
            display(result.visualization)
        except ImportError:
            pass
    
    return len(result.text) > 10


def main():
    print("="*60)
    print("  EXTRACTION HANDLERS REAL DEVICE TEST")
    print("="*60)
    
    images = list(ASSET_DIR.glob("docparsing_example*.jpg"))
    if not images:
        print("❌ No test images found")
        return
    
    model, processor = load_model()
    
    from qwen_vl.tasks import TaskType, get_handler
    
    test_image = images[0]
    results = []
    
    # Test Table
    print("\n" + "="*60)
    print("  TABLE HANDLER")
    print("="*60)
    table_handler = get_handler(TaskType.TABLE, model, processor)
    results.append(("Table", test_handler(table_handler, "Table Extraction", test_image)))
    
    # Test NER
    print("\n" + "="*60)
    print("  NER HANDLER")
    print("="*60)
    ner_handler = get_handler(TaskType.NER, model, processor)
    results.append(("NER", test_handler(ner_handler, "Named Entity Recognition", test_image)))
    
    # Test Form
    print("\n" + "="*60)
    print("  FORM HANDLER")
    print("="*60)
    form_handler = get_handler(TaskType.FORM, model, processor)
    results.append(("Form", test_handler(form_handler, "Form Extraction", test_image)))
    
    # Test Invoice
    print("\n" + "="*60)
    print("  INVOICE HANDLER")
    print("="*60)
    invoice_handler = get_handler(TaskType.INVOICE, model, processor)
    results.append(("Invoice", test_handler(invoice_handler, "Invoice Parsing", test_image)))
    
    # Test Contract
    print("\n" + "="*60)
    print("  CONTRACT HANDLER")
    print("="*60)
    contract_handler = get_handler(TaskType.CONTRACT, model, processor)
    results.append(("Contract", test_handler(contract_handler, "Contract Analysis", test_image)))
    
    # Test Field Extraction
    print("\n" + "="*60)
    print("  FIELD EXTRACTION HANDLER")
    print("="*60)
    field_handler = get_handler(TaskType.FIELD_EXTRACTION, model, processor)
    
    img = Image.open(test_image)
    start = time.time()
    result = field_handler.process(img, schema={"fields": [{"name": "title", "type": "text"}]})
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s")
    print(f"\n--- Result ---")
    print(result.text[:500] if len(result.text) > 500 else result.text)
    results.append(("Field Extraction", len(result.text) > 10))
    
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
