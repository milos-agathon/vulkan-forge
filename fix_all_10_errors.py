#!/usr/bin/env python3
"""Fix the missing pytest import"""

import os

def fix_pytest_import():
    """Add pytest import to test_terrain_pipeline.py"""
    
    file_path = "tests/cli/test_terrain_pipeline.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if pytest is already imported
    if "import pytest" not in content:
        # Find where to add the import - after other imports
        lines = content.split('\n')
        
        # Find a good place to insert pytest import
        for i, line in enumerate(lines):
            if line.strip() == "from contextlib import contextmanager":
                # Add pytest import after contextlib
                lines.insert(i + 1, "import pytest")
                break
            elif line.strip().startswith("from unittest.mock import"):
                # Or add after unittest.mock
                lines.insert(i + 1, "import pytest")
                break
        else:
            # If we can't find a good spot, add it after the numpy import
            for i, line in enumerate(lines):
                if "import numpy as np" in line:
                    lines.insert(i + 1, "import pytest")
                    break
        
        content = '\n'.join(lines)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print("✓ Added pytest import to test_terrain_pipeline.py")
    else:
        print("✓ pytest already imported")

def verify_all_imports():
    """Verify all required imports are present"""
    
    file_path = "tests/cli/test_terrain_pipeline.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    required_imports = [
        "import pytest",
        "import numpy as np",
        "import time",
        "import psutil",
        "from typing import Dict, List, Tuple, Optional",
        "from contextlib import contextmanager",
        "from unittest.mock import Mock, patch"
    ]
    
    missing = []
    for imp in required_imports:
        if imp not in content:
            missing.append(imp)
    
    if missing:
        print("\n⚠ Still missing imports:")
        for imp in missing:
            print(f"  - {imp}")
    else:
        print("\n✓ All required imports are present")

def show_test_status():
    """Show a quick summary of test status"""
    
    print("\n" + "=" * 60)
    print("Test Status Summary")
    print("=" * 60)
    print("\nExpected to PASS:")
    print("  ✓ test_allocator.py")
    print("  ✓ test_numpy_*.py (all numpy tests)")
    print("  ✓ test_terrain_config.py")
    print("  ✓ test_geotiff_loader.py")
    print("  ✓ test_terrain_pipeline.py (most tests)")
    print("  ✓ test_integration.py::TestMeshPipeline")
    print("  ✓ test_integration.py::TestPerformanceBenchmarks")
    
    print("\nExpected to FAIL (missing implementations):")
    print("  ✗ test_render_cpu.py")
    print("  ✗ test_render_indexed.py")
    
    print("\nRun all tests with:")
    print("  python -m pytest -v")
    print("\nRun only passing tests with:")
    print("  python -m pytest -v -k 'not (render_cpu or render_indexed)'")

def main():
    print("Fixing pytest Import Issue")
    print("=" * 40)
    
    fix_pytest_import()
    verify_all_imports()
    show_test_status()

if __name__ == "__main__":
    main()
