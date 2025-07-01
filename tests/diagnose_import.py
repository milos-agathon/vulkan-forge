#!/usr/bin/env python3
"""Diagnose why the import is still failing with old error message."""

import sys
import os
from pathlib import Path
import importlib.util

def diagnose_import():
    print("=== Import Diagnosis ===\n")
    
    # 1. Show Python version and executable
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    # 2. Show sys.path
    print("\n=== Python sys.path ===")
    for i, path in enumerate(sys.path):
        print(f"{i}: {path}")
    
    # 3. Search for ALL vulkan_forge directories
    print("\n=== Searching for all vulkan_forge directories ===")
    found_dirs = []
    
    # Search in sys.path
    for path in sys.path:
        if os.path.exists(path):
            vf_path = os.path.join(path, "vulkan_forge")
            if os.path.exists(vf_path) and os.path.isdir(vf_path):
                found_dirs.append(vf_path)
                print(f"Found: {vf_path}")
    
    # Also check the expected location
    expected_path = Path(__file__).parent.parent / "python" / "vulkan_forge"
    if expected_path.exists() and str(expected_path) not in [str(Path(p)) for p in found_dirs]:
        found_dirs.append(str(expected_path))
        print(f"Found (expected): {expected_path}")
    
    # 4. Check each found directory
    print("\n=== Checking each vulkan_forge directory ===")
    for vf_dir in found_dirs:
        print(f"\nDirectory: {vf_dir}")
        init_file = os.path.join(vf_dir, "__init__.py")
        if os.path.exists(init_file):
            print(f"  __init__.py exists")
            # Read first few lines
            with open(init_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(500)
            print(f"  First 500 chars of __init__.py:")
            print("  " + "\n  ".join(content.split('\n')[:10]))
            
            # Check for the error message
            with open(init_file, 'r', encoding='utf-8', errors='ignore') as f:
                full_content = f.read()
            if "Failed to import vulkan_forge native extension" in full_content:
                print("  ⚠️  WARNING: Contains old error message!")
            else:
                print("  ✓ Does not contain old error message")
    
    # 5. Try to find which module Python would import
    print("\n=== Testing import resolution ===")
    try:
        spec = importlib.util.find_spec("vulkan_forge")
        if spec:
            print(f"Import would load from: {spec.origin}")
            if spec.cached:
                print(f"Cached bytecode at: {spec.cached}")
        else:
            print("No import spec found for vulkan_forge")
    except Exception as e:
        print(f"Error finding import spec: {e}")
    
    # 6. Check for .pyc files
    print("\n=== Checking for compiled bytecode ===")
    for vf_dir in found_dirs:
        pycache_dir = os.path.join(vf_dir, "__pycache__")
        if os.path.exists(pycache_dir):
            print(f"Found __pycache__ in: {vf_dir}")
            for file in os.listdir(pycache_dir):
                if file.endswith('.pyc'):
                    print(f"  - {file}")
    
    # 7. Try importing with detailed error catching
    print("\n=== Attempting import with detailed error catching ===")
    
    # First, ensure the expected path is at the front of sys.path
    expected_parent = str(Path(__file__).parent.parent / "python")
    if expected_parent not in sys.path:
        sys.path.insert(0, expected_parent)
        print(f"Added to sys.path: {expected_parent}")
    
    # Clear any cached imports
    if 'vulkan_forge' in sys.modules:
        del sys.modules['vulkan_forge']
        print("Cleared cached vulkan_forge module")
    
    # Try the import
    try:
        print("\nAttempting: import vulkan_forge")
        import vulkan_forge
        print("✓ Import successful!")
        print(f"Loaded from: {vulkan_forge.__file__}")
    except Exception as e:
        print(f"✗ Import failed with: {type(e).__name__}: {e}")
        
        # Try to get more details
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_import()