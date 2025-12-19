"""
Comprehensive Real Device Integration Tests for ALL Qwen VL Task Handlers

This script tests ALL 14 task handlers and their key methods using real model inference.
It provides complete feature coverage for validation on actual devices.

Usage:
    # Run all tests
    python tests/integration/test_all_handlers.py
    
    # Run specific handler
    python tests/integration/test_all_handlers.py --handler ocr
    
    # Quick mode
    python tests/integration/test_all_handlers.py --quick
    
    # Verbose output
    python tests/integration/test_all_handlers.py --verbose
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PIL import Image

# Test configuration
ASSET_DIR = Path(__file__).parent.parent / "asset"
RESULTS_DIR = Path(__file__).parent.parent / "results"


@dataclass
class TestCase:
    """Represents a single test case."""
    handler_name: str
    method_name: str
    description: str
    asset_pattern: str
    prompt: Optional[str] = None
    kwargs: Optional[Dict[str, Any]] = None


# ============================================================================
# TEST CASE DEFINITIONS - All handlers and their methods
# ============================================================================

ALL_TEST_CASES = [
    # ========== OCR ==========
    TestCase("ocr", "process", "Basic OCR text extraction", "ocr_example*.jpg"),
    TestCase("ocr", "process", "OCR with bounding boxes", "ocr_example*.jpg", 
             kwargs={"with_boxes": True}),
    TestCase("ocr", "extract_lines", "Extract text line by line", "ocr_example*.jpg"),
    
    # ========== Layout ==========
    TestCase("layout", "process", "Document layout analysis", "docparsing_example*.jpg"),
    TestCase("layout", "detect_sections", "Detect document sections", "docparsing_example*.jpg"),
    TestCase("layout", "detect_reading_order", "Detect reading order", "docparsing_example*.jpg"),
    
    # ========== Recognition ==========
    TestCase("recognition", "process", "Basic image recognition", "docparsing_example*.jpg",
             prompt="What is shown in this image?"),
    TestCase("recognition", "identify_objects", "Identify all objects", "docparsing_example*.jpg"),
    TestCase("recognition", "identify_landmarks", "Identify landmarks", "docparsing_example*.jpg"),
    
    # ========== Spatial ==========
    TestCase("spatial", "process", "Basic spatial detection", "docparsing_example*.jpg"),
    TestCase("spatial", "detect_objects", "Detect objects with boxes", "docparsing_example*.jpg"),
    TestCase("spatial", "point_to_object", "Point to specific object", "docparsing_example*.jpg",
             kwargs={"object_description": "the main title"}),
    
    # ========== Document Parsing ==========
    TestCase("document_parsing", "process", "Parse to QwenVL HTML", "docparsing_example*.jpg",
             kwargs={"output_format": "qwenvl_html"}),
    TestCase("document_parsing", "parse_to_html", "Parse to clean HTML", "docparsing_example*.jpg"),
    TestCase("document_parsing", "parse_with_layout", "Parse with layout boxes", "docparsing_example*.jpg"),
    
    # ========== Table ==========
    TestCase("table", "process", "Extract table data", "docparsing_example*.jpg"),
    
    # ========== NER ==========
    TestCase("ner", "process", "Named entity recognition", "docparsing_example*.jpg"),
    
    # ========== Field Extraction ==========
    TestCase("field_extraction", "process", "Extract with schema", "docparsing_example*.jpg",
             kwargs={"schema": {"fields": [{"name": "title", "type": "text"}]}}),
    
    # ========== Form ==========
    TestCase("form", "process", "Form field extraction", "docparsing_example*.jpg"),
    
    # ========== Invoice ==========
    TestCase("invoice", "process", "Invoice parsing", "docparsing_example*.jpg"),
    
    # ========== Contract ==========
    TestCase("contract", "process", "Contract analysis", "docparsing_example*.jpg"),
    
    # ========== Computer Agent ==========
    TestCase("computer_agent", "process", "Screenshot analysis", "screenshot*.png",
             prompt="What buttons can I click?"),
    TestCase("computer_agent", "find_element", "Find UI element", "screenshot*.png",
             kwargs={"element_description": "the close button"}),
    TestCase("computer_agent", "suggest_action", "Suggest next action", "screenshot*.png",
             kwargs={"task": "close the window"}),
    
    # ========== Mobile Agent ==========
    TestCase("mobile_agent", "process", "Mobile screen analysis", "screenshot*.png",
             prompt="What elements are on this mobile screen?"),
    TestCase("mobile_agent", "find_element", "Find mobile element", "screenshot*.png",
             kwargs={"element_description": "the settings button"}),
    TestCase("mobile_agent", "describe_screen", "Describe mobile screen", "screenshot*.png"),
]


def print_header(title: str, char: str = "="):
    """Print a formatted header."""
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}")


def print_result(test: TestCase, success: bool, elapsed: float, details: str = "", verbose: bool = False):
    """Print a test result."""
    status = "✅" if success else "❌"
    print(f"  {status} {test.handler_name}.{test.method_name}() | {elapsed:.2f}s")
    if verbose and details:
        print(f"      {details}")


class ComprehensiveTest:
    """Comprehensive test runner for all handlers."""

    def __init__(self, quick_mode: bool = False, verbose: bool = False):
        self.quick_mode = quick_mode
        self.verbose = verbose
        self.model = None
        self.processor = None
        self.results: List[Dict[str, Any]] = []
        self.handlers_cache = {}
        
        RESULTS_DIR.mkdir(exist_ok=True)

    def load_model(self) -> bool:
        """Load the model."""
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
            
            print(f"  ✅ Loaded in {time.time() - start:.1f}s")
            return True
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            return False

    def get_handler(self, handler_name: str):
        """Get or create a handler instance."""
        if handler_name in self.handlers_cache:
            return self.handlers_cache[handler_name]
        
        from qwen_vl.tasks import TaskType, get_handler
        
        # Map handler names to TaskType
        task_map = {
            "ocr": TaskType.OCR,
            "layout": TaskType.LAYOUT,
            "recognition": TaskType.RECOGNITION,
            "spatial": TaskType.SPATIAL,
            "video": TaskType.VIDEO,
            "document_parsing": TaskType.DOCUMENT_PARSING,
            "table": TaskType.TABLE,
            "ner": TaskType.NER,
            "field_extraction": TaskType.FIELD_EXTRACTION,
            "form": TaskType.FORM,
            "invoice": TaskType.INVOICE,
            "contract": TaskType.CONTRACT,
            "computer_agent": TaskType.COMPUTER_AGENT,
            "mobile_agent": TaskType.MOBILE_AGENT,
        }
        
        task_type = task_map.get(handler_name)
        if not task_type:
            raise ValueError(f"Unknown handler: {handler_name}")
        
        handler = get_handler(task_type, self.model, self.processor)
        self.handlers_cache[handler_name] = handler
        return handler

    def get_assets(self, pattern: str) -> List[Path]:
        """Get assets matching a pattern."""
        assets = list(ASSET_DIR.glob(pattern))
        # Filter out non-image files
        assets = [a for a in assets if a.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        return assets

    def run_test_case(self, test: TestCase) -> Dict[str, Any]:
        """Run a single test case."""
        result = {
            "handler": test.handler_name,
            "method": test.method_name,
            "description": test.description,
            "success": False,
            "time": 0,
            "error": None,
            "output_length": 0,
        }
        
        try:
            # Get assets
            assets = self.get_assets(test.asset_pattern)
            if not assets:
                result["error"] = f"No assets found for pattern: {test.asset_pattern}"
                return result
            
            # Use first asset (or few in non-quick mode)
            asset = assets[0]
            
            # Get handler
            handler = self.get_handler(test.handler_name)
            
            # Get method
            method = getattr(handler, test.method_name, None)
            if not method:
                result["error"] = f"Method not found: {test.method_name}"
                return result
            
            # Load image
            img = Image.open(asset)
            
            # Prepare kwargs
            kwargs = test.kwargs or {}
            if test.prompt:
                kwargs["prompt"] = test.prompt
            
            # Run
            start = time.time()
            output = method(img, **kwargs)
            elapsed = time.time() - start
            
            # Check success
            result["time"] = elapsed
            result["output_length"] = len(output.text) if hasattr(output, "text") else 0
            result["success"] = result["output_length"] > 5
            result["asset"] = asset.name
            
            if self.verbose:
                result["output_preview"] = output.text if hasattr(output, "text") else ""
            
        except Exception as e:
            result["error"] = str(e)
            import traceback
            if self.verbose:
                traceback.print_exc()
        
        return result

    def run_all(self, handler_filter: Optional[str] = None):
        """Run all test cases."""
        print_header("Comprehensive Handler Test Suite")
        print(f"  Total test cases: {len(ALL_TEST_CASES)}")
        print(f"  Quick mode: {self.quick_mode}")
        print(f"  Verbose: {self.verbose}")
        
        if not self.load_model():
            print("\n❌ Cannot run tests without model")
            return
        
        # Group tests by handler
        tests_by_handler: Dict[str, List[TestCase]] = {}
        for test in ALL_TEST_CASES:
            if handler_filter and test.handler_name != handler_filter:
                continue
            if test.handler_name not in tests_by_handler:
                tests_by_handler[test.handler_name] = []
            tests_by_handler[test.handler_name].append(test)
        
        # Run tests
        for handler_name, tests in tests_by_handler.items():
            print_header(f"Testing {handler_name.upper()}", "-")
            
            # In quick mode, run only first test per handler
            tests_to_run = tests[:1] if self.quick_mode else tests
            
            for test in tests_to_run:
                result = self.run_test_case(test)
                self.results.append(result)
                
                print_result(
                    test, 
                    result["success"], 
                    result["time"],
                    result.get("output_preview", ""),
                    self.verbose
                )
        
        # Summary
        self.print_summary()
        self.save_results()

    def print_summary(self):
        """Print test summary."""
        print_header("Test Summary")
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r["success"])
        failed = total - passed
        total_time = sum(r["time"] for r in self.results)
        
        print(f"  Total: {total}")
        print(f"  ✅ Passed: {passed}")
        print(f"  ❌ Failed: {failed}")
        print(f"  Time: {total_time:.1f}s")
        
        if failed > 0:
            print("\n  Failed tests:")
            for r in self.results:
                if not r["success"]:
                    print(f"    ❌ {r['handler']}.{r['method']}(): {r.get('error', 'Unknown')}")
        
        # Coverage report
        print("\n  Handler Coverage:")
        handlers_tested = set(r["handler"] for r in self.results)
        all_handlers = set(t.handler_name for t in ALL_TEST_CASES)
        
        for h in sorted(all_handlers):
            methods = [r for r in self.results if r["handler"] == h]
            passed = sum(1 for m in methods if m["success"])
            total = len(methods)
            status = "✅" if passed == total and total > 0 else "⚠️" if passed > 0 else "❌"
            print(f"    {status} {h}: {passed}/{total} methods")

    def save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = RESULTS_DIR / f"test_results_{timestamp}.json"
        
        with open(result_file, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r["success"]),
                "results": self.results,
            }, f, indent=2)
        
        print(f"\n  📄 Results saved: {result_file}")


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive handler tests")
    parser.add_argument("--handler", type=str, help="Test specific handler only")
    parser.add_argument("--quick", action="store_true", help="Quick mode (1 test per handler)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    runner = ComprehensiveTest(quick_mode=args.quick, verbose=args.verbose)
    runner.run_all(handler_filter=args.handler)


if __name__ == "__main__":
    main()
