"""
Real Device Integration Tests for Qwen VL Task Handlers

This script runs actual inference on real samples to test all task handlers.
It requires a GPU with sufficient VRAM and a loaded model.

Usage:
    # Run all tests
    python tests/integration/test_real_inference.py
    
    # Run specific test
    python tests/integration/test_real_inference.py --task ocr
    
    # Quick mode (fewer samples)
    python tests/integration/test_real_inference.py --quick
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PIL import Image

# Test configuration
ASSET_DIR = Path(__file__).parent.parent / "asset"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def get_test_assets() -> Dict[str, List[Path]]:
    """Get all test assets organized by type."""
    assets = {
        "ocr": list(ASSET_DIR.glob("ocr_example*.jpg")),
        "document_parsing": list(ASSET_DIR.glob("docparsing_example*.*")),
        "invoice": list(ASSET_DIR.glob("invoice*.pdf")),
        "screenshot": list(ASSET_DIR.glob("screenshot*.png")),
        "chart": list(ASSET_DIR.glob("chart*.png")),
        "sketch": list(ASSET_DIR.glob("sketch*.jpeg")),
    }
    return assets


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(name: str, success: bool, time_sec: float, details: str = ""):
    """Print a test result."""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"  {status} | {name} | {time_sec:.2f}s")
    if details:
        print(f"         └─ {details}")


class RealInferenceTest:
    """Real device inference test runner."""

    def __init__(self, quick_mode: bool = False):
        self.quick_mode = quick_mode
        self.model = None
        self.processor = None
        self.results = []
        
        # Create results directory
        RESULTS_DIR.mkdir(exist_ok=True)

    def load_model(self) -> bool:
        """Load the model for testing."""
        print_header("Loading Model")
        
        try:
            from qwen_vl.config import load_config
            from qwen_vl.core.model_loader import ModelLoader
            
            config = load_config()
            print(f"  Model: {config.model.model_id}")
            print(f"  Quantization: {config.model.quantization}")
            
            start = time.time()
            loader = ModelLoader()
            loaded = loader.load(config)
            self.model = loaded.model
            self.processor = loaded.processor
            
            load_time = time.time() - start
            print(f"  ✅ Model loaded in {load_time:.1f}s")
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to load model: {e}")
            return False

    def test_ocr(self, assets: List[Path]) -> List[dict]:
        """Test OCR handler on sample images."""
        print_header("Testing OCR Handler")
        
        from qwen_vl.tasks import TaskType, get_handler
        
        results = []
        handler = get_handler(TaskType.OCR, self.model, self.processor)
        
        test_assets = assets[:2] if self.quick_mode else assets
        
        for asset in test_assets:
            try:
                start = time.time()
                img = Image.open(asset)
                result = handler.process(img, with_boxes=False)
                elapsed = time.time() - start
                
                success = len(result.text) > 10
                details = result.text.replace("\n", " ")
                print_result(asset.name, success, elapsed, details)
                
                results.append({
                    "task": "ocr",
                    "asset": asset.name,
                    "success": success,
                    "time": elapsed,
                    "output_length": len(result.text),
                })
                
            except Exception as e:
                print_result(asset.name, False, 0, str(e))
                results.append({
                    "task": "ocr",
                    "asset": asset.name,
                    "success": False,
                    "error": str(e),
                })
        
        return results

    def test_layout(self, assets: List[Path]) -> List[dict]:
        """Test Layout handler on sample images."""
        print_header("Testing Layout Handler")
        
        from qwen_vl.tasks import TaskType, get_handler
        
        results = []
        handler = get_handler(TaskType.LAYOUT, self.model, self.processor)
        
        test_assets = assets[:2] if self.quick_mode else assets[:4]
        
        for asset in test_assets:
            try:
                start = time.time()
                img = Image.open(asset)
                result = handler.process(img)
                elapsed = time.time() - start
                
                success = len(result.text) > 10
                details = result.text.replace("\n", " ")
                print_result(asset.name, success, elapsed, details)
                
                results.append({
                    "task": "layout",
                    "asset": asset.name,
                    "success": success,
                    "time": elapsed,
                })
                
            except Exception as e:
                print_result(asset.name, False, 0, str(e))
                results.append({
                    "task": "layout",
                    "asset": asset.name,
                    "success": False,
                    "error": str(e),
                })
        
        return results

    def test_recognition(self, assets: List[Path]) -> List[dict]:
        """Test Recognition handler on sample images."""
        print_header("Testing Recognition Handler")
        
        from qwen_vl.tasks import TaskType, get_handler
        
        results = []
        handler = get_handler(TaskType.RECOGNITION, self.model, self.processor)
        
        # Use document examples as they contain recognizable content
        test_assets = assets[:2] if self.quick_mode else assets[:3]
        
        for asset in test_assets:
            try:
                start = time.time()
                img = Image.open(asset)
                result = handler.process(img, prompt="What is shown in this image?")
                elapsed = time.time() - start
                
                success = len(result.text) > 20
                details = result.text.replace("\n", " ")
                print_result(asset.name, success, elapsed, details)
                
                results.append({
                    "task": "recognition",
                    "asset": asset.name,
                    "success": success,
                    "time": elapsed,
                })
                
            except Exception as e:
                print_result(asset.name, False, 0, str(e))
                results.append({
                    "task": "recognition",
                    "asset": asset.name,
                    "success": False,
                    "error": str(e),
                })
        
        return results

    def test_spatial(self, assets: List[Path]) -> List[dict]:
        """Test Spatial handler on sample images."""
        print_header("Testing Spatial Handler")
        
        from qwen_vl.tasks import TaskType, get_handler
        
        results = []
        handler = get_handler(TaskType.SPATIAL, self.model, self.processor)
        
        test_assets = assets[:1] if self.quick_mode else assets[:2]
        
        for asset in test_assets:
            try:
                start = time.time()
                img = Image.open(asset)
                result = handler.detect_objects(img)
                elapsed = time.time() - start
                
                success = len(result.text) > 10
                details = result.text.replace("\n", " ")
                print_result(asset.name, success, elapsed, details)
                
                results.append({
                    "task": "spatial",
                    "asset": asset.name,
                    "success": success,
                    "time": elapsed,
                })
                
            except Exception as e:
                print_result(asset.name, False, 0, str(e))
                results.append({
                    "task": "spatial",
                    "asset": asset.name,
                    "success": False,
                    "error": str(e),
                })
        
        return results

    def test_document_parsing(self, assets: List[Path]) -> List[dict]:
        """Test Document Parsing handler on sample images."""
        print_header("Testing Document Parsing Handler")
        
        from qwen_vl.tasks import TaskType, get_handler
        
        results = []
        handler = get_handler(TaskType.DOCUMENT_PARSING, self.model, self.processor)
        
        test_assets = assets[:2] if self.quick_mode else assets[:4]
        
        for asset in test_assets:
            if asset.suffix.lower() == ".pdf":
                continue  # Skip PDFs for now
                
            try:
                start = time.time()
                img = Image.open(asset)
                result = handler.process(img, output_format="qwenvl_html")
                elapsed = time.time() - start
                
                success = len(result.text) > 20
                details = f"Elements: {result.metadata.get('element_count', 0)}"
                print_result(asset.name, success, elapsed, details)
                
                results.append({
                    "task": "document_parsing",
                    "asset": asset.name,
                    "success": success,
                    "time": elapsed,
                })
                
            except Exception as e:
                print_result(asset.name, False, 0, str(e))
                results.append({
                    "task": "document_parsing",
                    "asset": asset.name,
                    "success": False,
                    "error": str(e),
                })
        
        return results

    def test_computer_agent(self, assets: List[Path]) -> List[dict]:
        """Test Computer Agent handler on screenshot."""
        print_header("Testing Computer Agent Handler")
        
        from qwen_vl.tasks import TaskType, get_handler
        
        results = []
        handler = get_handler(TaskType.COMPUTER_AGENT, self.model, self.processor)
        
        for asset in assets[:1]:
            try:
                start = time.time()
                img = Image.open(asset)
                result = handler.process(img, prompt="What buttons or links can I click on this screen?")
                elapsed = time.time() - start
                
                success = len(result.text) > 20
                details = result.text.replace("\n", " ")
                print_result(asset.name, success, elapsed, details)
                
                results.append({
                    "task": "computer_agent",
                    "asset": asset.name,
                    "success": success,
                    "time": elapsed,
                })
                
            except Exception as e:
                print_result(asset.name, False, 0, str(e))
                results.append({
                    "task": "computer_agent",
                    "asset": asset.name,
                    "success": False,
                    "error": str(e),
                })
        
        return results

    def test_table(self, assets: List[Path]) -> List[dict]:
        """Test Table handler on document images."""
        print_header("Testing Table Handler")
        
        from qwen_vl.tasks import TaskType, get_handler
        
        results = []
        handler = get_handler(TaskType.TABLE, self.model, self.processor)
        
        test_assets = assets[:2] if self.quick_mode else assets[:3]
        
        for asset in test_assets:
            try:
                start = time.time()
                img = Image.open(asset)
                result = handler.process(img)
                elapsed = time.time() - start
                
                success = len(result.text) > 10
                details = result.text.replace("\n", " ")
                print_result(asset.name, success, elapsed, details)
                
                results.append({
                    "task": "table",
                    "asset": asset.name,
                    "success": success,
                    "time": elapsed,
                })
                
            except Exception as e:
                print_result(asset.name, False, 0, str(e))
                results.append({
                    "task": "table",
                    "asset": asset.name,
                    "success": False,
                    "error": str(e),
                })
        
        return results

    def test_ner(self, assets: List[Path]) -> List[dict]:
        """Test NER handler on document images."""
        print_header("Testing NER Handler")
        
        from qwen_vl.tasks import TaskType, get_handler
        
        results = []
        handler = get_handler(TaskType.NER, self.model, self.processor)
        
        test_assets = assets[:2] if self.quick_mode else assets[:3]
        
        for asset in test_assets:
            try:
                start = time.time()
                img = Image.open(asset)
                result = handler.process(img)
                elapsed = time.time() - start
                
                success = len(result.text) > 10
                details = result.text.replace("\n", " ")
                print_result(asset.name, success, elapsed, details)
                
                results.append({
                    "task": "ner",
                    "asset": asset.name,
                    "success": success,
                    "time": elapsed,
                })
                
            except Exception as e:
                print_result(asset.name, False, 0, str(e))
                results.append({
                    "task": "ner",
                    "asset": asset.name,
                    "success": False,
                    "error": str(e),
                })
        
        return results

    def run_all_tests(self, task_filter: Optional[str] = None):
        """Run all integration tests."""
        print_header("Qwen VL Real Device Integration Tests")
        print(f"  Asset directory: {ASSET_DIR}")
        print(f"  Quick mode: {self.quick_mode}")
        
        # Get assets
        assets = get_test_assets()
        print(f"\n  Available assets:")
        for asset_type, files in assets.items():
            print(f"    - {asset_type}: {len(files)} files")
        
        # Load model
        if not self.load_model():
            print("\n❌ Cannot run tests without model. Exiting.")
            return
        
        # Run tests
        all_results = []
        
        test_map = {
            "ocr": (self.test_ocr, assets.get("ocr", [])),
            "layout": (self.test_layout, assets.get("document_parsing", [])),
            "recognition": (self.test_recognition, assets.get("document_parsing", [])),
            "spatial": (self.test_spatial, assets.get("document_parsing", [])),
            "document_parsing": (self.test_document_parsing, assets.get("document_parsing", [])),
            "computer_agent": (self.test_computer_agent, assets.get("screenshot", [])),
            "table": (self.test_table, assets.get("document_parsing", [])),
            "ner": (self.test_ner, assets.get("document_parsing", [])),
        }
        
        for task_name, (test_func, task_assets) in test_map.items():
            if task_filter and task_filter != task_name:
                continue
            if not task_assets:
                print(f"\n  ⚠️  Skipping {task_name}: no assets found")
                continue
            try:
                results = test_func(task_assets)
                all_results.extend(results)
            except Exception as e:
                print(f"\n  ❌ Error in {task_name} tests: {e}")
        
        # Summary
        self.print_summary(all_results)
        
        return all_results

    def print_summary(self, results: List[dict]):
        """Print test summary."""
        print_header("Test Summary")
        
        if not results:
            print("  No tests were run.")
            return
        
        total = len(results)
        passed = sum(1 for r in results if r.get("success"))
        failed = total - passed
        
        total_time = sum(r.get("time", 0) for r in results)
        
        print(f"  Total tests: {total}")
        print(f"  ✅ Passed: {passed}")
        print(f"  ❌ Failed: {failed}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average time: {total_time/total:.1f}s per test")
        
        if failed > 0:
            print("\n  Failed tests:")
            for r in results:
                if not r.get("success"):
                    print(f"    - {r['task']}/{r['asset']}: {r.get('error', 'Unknown error')}")


def main():
    parser = argparse.ArgumentParser(description="Run real device integration tests")
    parser.add_argument(
        "--task",
        type=str,
        help="Run only specific task test (ocr, layout, recognition, spatial, etc.)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in quick mode with fewer samples",
    )
    
    args = parser.parse_args()
    
    runner = RealInferenceTest(quick_mode=args.quick)
    runner.run_all_tests(task_filter=args.task)


if __name__ == "__main__":
    main()
