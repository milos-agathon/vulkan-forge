#!/usr/bin/env python3
"""Debug script to identify import issues."""

import sys
from pathlib import Path
import os

def debug_imports():
    print("=== VulkanForge Import Debug ===")
    
    # Show current working directory and script location
    script_path = Path(__file__).resolve()
    cwd = Path.cwd()
    
    print(f"Script location: {script_path}")
    print(f"Current working directory: {cwd}")
    print(f"Parent directory: {script_path.parent}")
    
    # Check if vulkan_forge directory exists
    possible_paths = [
        cwd / "vulkan_forge",
        script_path.parent / "vulkan_forge", 
        script_path.parent.parent / "vulkan_forge",
        cwd / "python" / "vulkan_forge",
        script_path.parent / "python" / "vulkan_forge"
    ]
    
    print("\n=== Checking possible vulkan_forge locations ===")
    vulkan_forge_path = None
    for path in possible_paths:
        print(f"Checking: {path}")
        if path.exists() and path.is_dir():
            print(f"  ✓ Found directory")
            init_file = path / "__init__.py"
            if init_file.exists():
                print(f"  ✓ Found __init__.py")
                vulkan_forge_path = path
                break
            else:
                print(f"  ✗ No __init__.py found")
        else:
            print(f"  ✗ Directory does not exist")
    
    if not vulkan_forge_path:
        print("\n✗ Could not find vulkan_forge module directory!")
        return
    
    print(f"\n✓ Using vulkan_forge at: {vulkan_forge_path}")
    
    # Add parent directory to path
    parent_dir = vulkan_forge_path.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
        print(f"Added to Python path: {parent_dir}")
    
    # Check individual module files
    modules = ["__init__.py", "backend.py", "renderer.py", "matrices.py"]
    print(f"\n=== Checking module files in {vulkan_forge_path} ===")
    for module in modules:
        module_path = vulkan_forge_path / module
        if module_path.exists():
            print(f"  ✓ {module}")
        else:
            print(f"  ✗ {module} (missing!)")
    
    # Try importing step by step
    print(f"\n=== Testing imports ===")
    
    try:
        print("1. Testing basic import...")
        import vulkan_forge
        print("  ✓ vulkan_forge imported successfully")
    except Exception as e:
        print(f"  ✗ vulkan_forge import failed: {e}")
        return
    
    try:
        print("2. Testing individual modules...")
        import vulkan_forge.backend
        print("  ✓ vulkan_forge.backend imported")
    except Exception as e:
        print(f"  ✗ vulkan_forge.backend failed: {e}")
    
    try:
        import vulkan_forge.matrices  
        print("  ✓ vulkan_forge.matrices imported")
    except Exception as e:
        print(f"  ✗ vulkan_forge.matrices failed: {e}")
    
    try:
        import vulkan_forge.renderer
        print("  ✓ vulkan_forge.renderer imported")
    except Exception as e:
        print(f"  ✗ vulkan_forge.renderer failed: {e}")
    
    try:
        print("3. Testing main functions...")
        from vulkan_forge import create_renderer, Matrix4x4
        print("  ✓ create_renderer and Matrix4x4 imported")
    except Exception as e:
        print(f"  ✗ Main functions import failed: {e}")
    
    try:
        from vulkan_forge.renderer import RenderTarget, Mesh, Material, Light
        print("  ✓ Renderer classes imported")
    except Exception as e:
        print(f"  ✗ Renderer classes import failed: {e}")
    
    print("\n=== Debug complete ===")

if __name__ == "__main__":
    debug_imports()