#!/usr/bin/env python3
"""Modified debug_imports.py that forces use of local vulkan_forge."""

import sys
from pathlib import Path
import os

def debug_imports():
    print("=== VulkanForge Import Debug ===")
    
    # CRITICAL: Add the python directory to the FRONT of sys.path
    # This ensures we use the local version, not the site-packages version
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    python_dir = project_root / "python"
    
    # Remove any existing vulkan_forge imports
    if 'vulkan_forge' in sys.modules:
        del sys.modules['vulkan_forge']
        print("Removed cached vulkan_forge module")
    
    # Add python directory to the front of sys.path
    sys.path.insert(0, str(python_dir))
    print(f"Added to front of sys.path: {python_dir}")
    
    # Show current working directory and script location
    cwd = Path.cwd()
    
    print(f"Script location: {script_path}")
    print(f"Current working directory: {cwd}")
    print(f"Parent directory: {script_path.parent}")
    
    # Check if vulkan_forge directory exists
    vulkan_forge_path = python_dir / "vulkan_forge"
    
    print(f"\n=== Checking vulkan_forge location ===")
    print(f"Looking at: {vulkan_forge_path}")
    if vulkan_forge_path.exists() and vulkan_forge_path.is_dir():
        print(f"  ✓ Found directory")
        init_file = vulkan_forge_path / "__init__.py"
        if init_file.exists():
            print(f"  ✓ Found __init__.py")
        else:
            print(f"  ✗ No __init__.py found")
    else:
        print(f"  ✗ Directory does not exist")
        return
    
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
        print(f"  Loaded from: {vulkan_forge.__file__}")
    except Exception as e:
        print(f"  ✗ vulkan_forge import failed: {e}")
        import traceback
        traceback.print_exc()
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