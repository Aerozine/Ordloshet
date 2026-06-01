"""tests/validation.py  -  Validate all pipeline modules import."""
import sys
from pathlib import Path

# Add project root to path for importing pipeline submodules
sys.path.insert(0, str(Path(__file__).parent.parent))


def validate_module(module_name: str, module_path: str, optional: bool = False):
    """Check if module imports successfully."""
    try:
        __import__(module_path)
        print(f"  [OK] {module_name}")
        return True
    except ImportError as e:
        if optional:
            print(f"  [WARN]  {module_name}: skip (missing optional dependency)")
            return True
        else:
            print(f"  [FAIL] {module_name}: {e}")
            return False


def main():
    print("Validating pipeline modules...")
    print("-" * 20)
    
    core_modules = [
        ("pipeline.chunker", "pipeline.chunker"),
        ("pipeline.register", "pipeline.register"),
        ("pipeline.alignment", "pipeline.alignment"),
        ("pipeline.translator", "pipeline.translator"),
        ("pipeline.output", "pipeline.output"),
        ("pipeline.adapters.base", "pipeline.adapters.base"),
        ("pipeline.adapters.seq2seq", "pipeline.adapters.seq2seq"),
        ("pipeline.adapters.llm", "pipeline.adapters.llm"),
        ("pipeline.adapters.seamless", "pipeline.adapters.seamless"),
    ]
    
    optional_modules = [
        ("pipeline.strategies.light", "pipeline.strategies.light"),
        ("pipeline.strategies.heavy", "pipeline.strategies.heavy"),
    ]
    
    all_modules = core_modules + optional_modules
    
    passed = 0
    failed = 0
    
    for name, path in all_modules:
        if validate_module(name, path):
            passed += 1
        else:
            failed += 1
    
    print("-" * 20)
    print(f"Results: {passed} passed, {failed} failed")
    print("Note: Missing optional deps (bs4, torch) are expected before pip install")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main() or 0)

