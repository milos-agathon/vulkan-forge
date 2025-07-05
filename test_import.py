#!/usr/bin/env python
"""Test if vulkan_forge can be imported correctly."""

import sys
import os

# Add the python directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

try:
    import vulkan_forge as vf
    print("✓ Successfully imported vulkan_forge")
    print(f"  Version: {vf.__version__}")
    print(f"  Native available: {vf._native_available}")
    
    # Try creating a renderer
    renderer = vf.Renderer(640, 480)
    print("✓ Successfully created renderer")
    
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()