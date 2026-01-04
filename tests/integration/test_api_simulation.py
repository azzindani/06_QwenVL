"""
API Simulation Test

Integration test that runs one sample through each API endpoint,
similar to other integration tests but via HTTP.

Usage:
    # Start API server first:
    python main.py --api
    
    # Then run this test:
    python tests/integration/test_api_simulation.py
"""

import sys
import time
from pathlib import Path

import requests

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Test configuration
API_BASE_URL = "http://localhost:8000"
ASSET_DIR = Path(__file__).parent.parent / "asset"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# One sample per handler type
TEST_SAMPLES = {
    "OCR": {
        "sample": "ocr_example1.jpg",
        "endpoint": "/extract/ocr",
        "data": {"include_boxes": "true"},
    },
    "Table": {
        "sample": "docparsing_example1.jpg",
        "endpoint": "/extract/table",
        "data": {"output_format": "json"},
    },
    "Form": {
        "sample": "docparsing_example2.jpg",
        "endpoint": "/extract/form",
        "data": {},
    },
    "Invoice": {
        "sample": "docparsing_example3.jpg",
        "endpoint": "/extract/invoice",
        "data": {"document_type": "invoice"},
    },
    "NER": {
        "sample": "docparsing_example1.jpg",
        "endpoint": "/extract/ner",
        "data": {"entity_types": "all"},
    },
    "Layout": {
        "sample": "docparsing_example1.jpg",
        "endpoint": "/layout",
        "data": {},
    },
    "Spatial": {
        "sample": "spatio_case1.png",
        "endpoint": "/spatial",
        "data": {},
    },
    "Recognition": {
        "sample": "sample-celebrity1.jpg",
        "endpoint": "/recognition",
        "data": {"recognition_type": "celebrity"},
    },
    "Document": {
        "sample": "docparsing_example1.jpg",
        "endpoint": "/document",
        "data": {"output_format": "html"},
    },
    "ComputerAgent": {
        "sample": "computer_use1.jpeg",
        "endpoint": "/agent/computer",
        "data": {"instruction": "Click on the search button"},
    },
    "MobileAgent": {
        "sample": "mobile_main_example.png",
        "endpoint": "/agent/mobile",
        "data": {"instruction": "Tap on the settings icon"},
    },
}


def find_sample(pattern: str) -> Path:
    """Find sample file matching pattern."""
    # Try exact match first
    exact = ASSET_DIR / pattern
    if exact.exists():
        return exact
    
    # Try glob
    matches = list(ASSET_DIR.glob(pattern))
    if matches:
        return matches[0]
    
    # Try without extension
    base = pattern.rsplit(".", 1)[0]
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        matches = list(ASSET_DIR.glob(f"{base}*{ext}"))
        if matches:
            return matches[0]
    
    return None


def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def test_endpoint(name: str, config: dict) -> dict:
    """Test a single API endpoint."""
    sample_path = find_sample(config["sample"])
    
    if not sample_path:
        return {
            "name": name,
            "status": "SKIP",
            "reason": f"Sample not found: {config['sample']}",
            "time": 0,
        }
    
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    print(f"Sample: {sample_path.name}")
    print(f"Endpoint: {config['endpoint']}")
    
    start = time.time()
    
    try:
        with open(sample_path, "rb") as f:
            files = {"file": (sample_path.name, f, "image/jpeg")}
            response = requests.post(
                f"{API_BASE_URL}{config['endpoint']}",
                files=files,
                data=config.get("data", {}),
                timeout=120,  # 2 min timeout for large models
            )
        
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            success = data.get("success", False)
            
            print(f"Time: {elapsed:.2f}s")
            print(f"Status: {response.status_code}")
            
            # Print key info
            if "text" in data:
                text_preview = data["text"][:200] + "..." if len(data.get("text", "")) > 200 else data.get("text", "")
                print(f"Text: {text_preview}")
            
            if "entities" in data:
                print(f"Entities: {len(data['entities'])} found")
            
            if "tables" in data:
                print(f"Tables: {len(data['tables'])} found")
            
            if "visualization" in data and data["visualization"]:
                print("Visualization: Available (base64)")
            
            return {
                "name": name,
                "status": "PASS" if success else "FAIL",
                "time": elapsed,
                "response": data,
            }
        else:
            print(f"Error: {response.status_code}")
            print(response.text[:500])
            return {
                "name": name,
                "status": "FAIL",
                "reason": f"HTTP {response.status_code}",
                "time": time.time() - start,
            }
            
    except Exception as e:
        elapsed = time.time() - start
        print(f"Error: {e}")
        return {
            "name": name,
            "status": "ERROR",
            "reason": str(e),
            "time": elapsed,
        }


def main():
    print("="*60)
    print("  API SIMULATION TEST")
    print("="*60)
    
    # Check API health
    print("\nChecking API health...")
    if not check_api_health():
        print("❌ API is not running!")
        print("   Start the API first: python main.py --api")
        return
    
    print("✅ API is healthy")
    
    # Run tests
    results = []
    
    for name, config in TEST_SAMPLES.items():
        result = test_endpoint(name, config)
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("  TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    skipped = 0
    
    for result in results:
        status = result["status"]
        time_str = f"({result['time']:.2f}s)" if result.get("time") else ""
        
        if status == "PASS":
            print(f"  ✅ PASS - {result['name']} {time_str}")
            passed += 1
        elif status == "SKIP":
            print(f"  ⏭️  SKIP - {result['name']}: {result.get('reason', '')}")
            skipped += 1
        else:
            print(f"  ❌ FAIL - {result['name']}: {result.get('reason', '')} {time_str}")
            failed += 1
    
    print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")
    
    return passed > 0 and failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
