# Quick verification script
import sys
sys.path.insert(0, ".")

try:
    from qwen_vl.tasks import *
    from qwen_vl.tasks.base import logger
    
    print("=" * 50)
    print("IMPORT VERIFICATION")
    print("=" * 50)
    print(f"  Logger: {logger.name}")
    print(f"  All handlers imported successfully")
    print("=" * 50)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
