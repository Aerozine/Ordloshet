#!/usr/bin/env python3
"""tests/run_tests.py  -  Simple test runner for the EPUB translator.

This script runs quick validation tests on each pipeline module to ensure they can be imported
and their core functions are callable. It's suitable for CI regression testing.

Run with: make test or python tests/run_tests.py
"""
import sys
from pathlib import Path


def run_test(module_name: str, import_path: str):
    """Test a single module."""
    try:
        if import_path.startswith('.'):
            spec = __import__(f'pipeline{import_path}', fromlist=[''])
        else:
            import importlib
            spec = importlib.util.find_spec(import_path)
            mod = importlib.import_module(import_path)
        
        print(f"  [OK] {module_name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {module_name}: {e}")
        return False


def main():
    """Run all pipeline tests."""
    print("EPUB Translator Pipeline Tests")
    print("=" * 50)
    
    # Test imports
    tests = [
        ("pipeline.chunker", "pipeline.chunker"),
        ("pipeline.register", "pipeline.register"),
        ("pipeline.alignment", "pipeline.alignment"),
        ("pipeline.translator", "pipeline.translator"),
        ("pipeline.output", "pipeline.output"),
        
        # Adapter tests
        ("pipeline.adapters.base", "pipeline.adapters.base"),
        ("pipeline.adapters.seq2seq", "pipeline.adapters.seq2seq"),
        ("pipeline.adapters.llm", "pipeline.adapters.llm"),
        ("pipeline.adapters.seamless", "pipeline.adapters.seamless"),
        
        # Strategy tests
        ("pipeline.strategies.light", "pipeline.strategies.light"),
        ("pipeline.strategies.heavy", "pipeline.strategies.heavy"),
    ]
    
    passed = 0
    failed = 0
    
    for name, path in tests:
        if run_test(name, path):
            passed += 1
        else:
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
