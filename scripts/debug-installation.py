#!/usr/bin/env python3
"""
Debug script for vulkan-forge installation issues
Helps diagnose common problems with the installation
"""

import sys
import os
import importlib
import traceback
from pathlib import Path

def print_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_section(title):
    print(f"\n--- {title} ---")

def safe_import_test():
    """Test importing vulkan-forge and inspect what's available"""
    print_header("VULKAN-FORGE IMPORT DEBUGGING")
    
    # Test basic import
    print_section("Basic Import Test")
    try:
        import vulkan_forge
        print("✅ vulkan_forge imported successfully")
        print(f"   Module location: {vulkan_forge.__file__}")
        
        # Check version
        try:
            version = vulkan_forge.__version__
            print(f"   Version: {version}")
        except AttributeError:
            print("⚠️  No __version__ attribute found")
        
        # List all attributes
        print(f"   Available attributes: {dir(vulkan_forge)}")
        
    except ImportError as e:
        print(f"❌ Failed to import vulkan_forge: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error importing vulkan_forge: {e}")
        traceback.print_exc()
        return None
    
    return vulkan_forge

def test_classes(vf_module):
    """Test the main classes"""
    print_section("Class Availability Test")
    
    required_classes = ["HeightFieldScene", "Renderer"]
    available_classes = {}
    
    for class_name in required_classes:
        try:
            if hasattr(vf_module, class_name):
                cls = getattr(vf_module, class_name)
                print(f"✅ {class_name}: {cls}")
                print(f"   Type: {type(cls)}")
                print(f"   Callable: {callable(cls)}")
                available_classes[class_name] = cls
            else:
                print(f"❌ {class_name}: Not found")
        except Exception as e:
            print(f"❌ {class_name}: Error accessing - {e}")
    
    return available_classes

def test_object_creation(available_classes):
    """Test creating objects from the classes"""
    print_section("Object Creation Test")
    
    # Test HeightFieldScene
    if "HeightFieldScene" in available_classes:
        try:
            scene_class = available_classes["HeightFieldScene"]
            print(f"Attempting to create HeightFieldScene...")
            print(f"Class type: {type(scene_class)}")
            
            scene = scene_class()
            print(f"✅ HeightFieldScene created: {scene}")
            print(f"   Object type: {type(scene)}")
            print(f"   Available methods: {[m for m in dir(scene) if not m.startswith('_')]}")
            
            # Test if build method exists
            if hasattr(scene, 'build'):
                build_method = getattr(scene, 'build')
                print(f"   build method: {build_method}")
                print(f"   build callable: {callable(build_method)}")
            else:
                print("❌ No 'build' method found")
                
        except Exception as e:
            print(f"❌ Failed to create HeightFieldScene: {e}")
            traceback.print_exc()
    
    # Test Renderer
    if "Renderer" in available_classes:
        try:
            renderer_class = available_classes["Renderer"]
            print(f"\nAttempting to create Renderer...")
            print(f"Class type: {type(renderer_class)}")
            
            renderer = renderer_class(64, 64)
            print(f"✅ Renderer created: {renderer}")
            print(f"   Object type: {type(renderer)}")
            print(f"   Available methods: {[m for m in dir(renderer) if not m.startswith('_')]}")
            
        except Exception as e:
            print(f"❌ Failed to create Renderer: {e}")
            traceback.print_exc()

def test_numpy_integration():
    """Test NumPy integration"""
    print_section("NumPy Integration Test")
    
    try:
        import numpy as np
        print(f"✅ NumPy imported: {np.__version__}")
        
        # Test creating data
        heights = np.ones((4, 4), dtype=np.float32)
        print(f"✅ Created test data: {heights.shape}, {heights.dtype}")
        
    except ImportError:
        print("❌ NumPy not available")
    except Exception as e:
        print(f"❌ NumPy error: {e}")

def check_c_extension():
    """Check if C++ extension is properly built"""
    print_section("C++ Extension Check")
    
    try:
        import vulkan_forge
        
        # Look for native module
        module_dir = Path(vulkan_forge.__file__).parent
        print(f"Module directory: {module_dir}")
        
        # List all files in the module
        print("Files in module directory:")
        for file_path in module_dir.iterdir():
            print(f"  {file_path.name}")
        
        # Look for compiled extensions
        extensions = list(module_dir.glob("*.pyd")) + list(module_dir.glob("*.so")) + list(module_dir.glob("*.dll"))
        
        if extensions:
            print(f"Found compiled extensions:")
            for ext in extensions:
                print(f"  ✅ {ext.name}")
        else:
            print("❌ No compiled extensions found (.pyd, .so, .dll)")
            print("This suggests the C++ extension was not built properly")
        
        # Try to import the native module directly
        try:
            import vulkan_forge._vulkan_forge_native
            print("✅ Native module imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import native module: {e}")
        except Exception as e:
            print(f"❌ Native module error: {e}")
            
    except Exception as e:
        print(f"❌ Extension check failed: {e}")

def check_installation_method():
    """Check how vulkan-forge was installed"""
    print_section("Installation Method Check")
    
    try:
        import vulkan_forge
        import pkg_resources
        
        # Get package info
        try:
            dist = pkg_resources.get_distribution("vulkan-forge")
            print(f"✅ Package installed via pip: {dist.version}")
            print(f"   Location: {dist.location}")
            print(f"   Metadata: {dist.project_name}")
        except pkg_resources.DistributionNotFound:
            print("⚠️  Not installed via pip (development install?)")
        
        # Check if this is an editable install
        module_path = Path(vulkan_forge.__file__).parent
        if module_path.name == "vulkan_forge" and (module_path.parent / "setup.py").exists():
            print("⚠️  Appears to be an editable/development install")
        
    except Exception as e:
        print(f"❌ Installation check failed: {e}")

def check_environment():
    """Check environment variables and dependencies"""
    print_section("Environment Check")
    
    # Python info
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Vulkan SDK
    vulkan_sdk = os.environ.get('VULKAN_SDK')
    if vulkan_sdk:
        print(f"✅ VULKAN_SDK: {vulkan_sdk}")
        
        # Check if SDK files exist
        if os.path.exists(vulkan_sdk):
            print("✅ VULKAN_SDK path exists")
            bin_dir = Path(vulkan_sdk) / "Bin"
            if bin_dir.exists():
                print("✅ VULKAN_SDK/Bin exists")
                vulkaninfo = bin_dir / "vulkaninfo.exe"
                if vulkaninfo.exists():
                    print("✅ vulkaninfo.exe found")
                else:
                    print("❌ vulkaninfo.exe not found")
            else:
                print("❌ VULKAN_SDK/Bin not found")
        else:
            print("❌ VULKAN_SDK path does not exist")
    else:
        print("❌ VULKAN_SDK environment variable not set")
    
    # PATH check
    path_dirs = os.environ.get('PATH', '').split(os.pathsep)
    vulkan_in_path = any('vulkan' in p.lower() for p in path_dirs)
    if vulkan_in_path:
        print("✅ Vulkan-related directories found in PATH")
    else:
        print("⚠️  No Vulkan directories found in PATH")

def suggest_fixes():
    """Suggest potential fixes based on findings"""
    print_header("SUGGESTED FIXES")
    
    print("""
Based on the diagnostic results above, here are potential fixes:

1. **If no compiled extensions found (.pyd/.so/.dll):**
   - The C++ extension wasn't built properly
   - Try reinstalling: pip uninstall vulkan-forge && pip install vulkan-forge
   - Or if building from source: python -m pip install -e . --force-reinstall

2. **If 'NoneType' errors persist:**
   - The native module may be built but not working correctly
   - Check that you have the Visual Studio redistributables installed
   - Try installing in a fresh virtual environment

3. **If VULKAN_SDK issues:**
   - Run: scripts/install-vulkan-sdk.bat /force
   - Restart your command prompt after installation
   - Verify: vulkaninfo --summary

4. **If development install issues:**
   - Run: python -m pip install -e . --force-reinstall
   - Make sure CMake and Visual Studio are properly installed
   - Check: cmake --version

5. **General debugging steps:**
   - Create fresh virtual environment: python -m venv fresh_env
   - Activate: fresh_env\\Scripts\\activate
   - Install: pip install vulkan-forge
   - Test: python -c "import vulkan_forge; print('OK')"
""")

def main():
    """Main debugging function"""
    try:
        # Run all diagnostic tests
        vf_module = safe_import_test()
        
        if vf_module:
            available_classes = test_classes(vf_module)
            test_object_creation(available_classes)
        
        test_numpy_integration()
        check_c_extension()
        check_installation_method()
        check_environment()
        suggest_fixes()
        
    except KeyboardInterrupt:
        print("\n⚠️  Debugging interrupted by user")
    except Exception as e:
        print(f"\n❌ Debugging script failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()