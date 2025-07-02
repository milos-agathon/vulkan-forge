#!/usr/bin/env python3
"""Test Vulkan setup and GPU availability."""

import sys
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_vulkan_installation():
    """Check if Vulkan is properly installed on the system."""
    print("=== Checking Vulkan Installation ===\n")
    
    # Check for vulkaninfo command
    try:
        result = subprocess.run(['vulkaninfo', '--summary'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ Vulkan runtime is installed")
            # Parse output for GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GPU' in line or 'deviceName' in line:
                    print(f"  {line.strip()}")
        else:
            print("✗ Vulkan runtime not found or not working")
            print("  Please install Vulkan SDK or update GPU drivers")
    except FileNotFoundError:
        print("✗ vulkaninfo command not found")
        print("  Please install Vulkan SDK from: https://vulkan.lunarg.com/")
    except subprocess.TimeoutExpired:
        print("✗ vulkaninfo timed out")
    except Exception as e:
        print(f"✗ Error checking vulkaninfo: {e}")
    
    print()

def check_python_vulkan():
    """Check if Python vulkan package is installed and working."""
    print("=== Checking Python Vulkan Package ===\n")
    
    try:
        import vulkan as vk
        print("✓ Python vulkan package is installed")
        print(f"  Version info available: {hasattr(vk, 'VK_MAKE_VERSION')}")
        
        # Try to create instance
        try:
            app_info = vk.VkApplicationInfo(
                sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
                pApplicationName="VulkanTest",
                applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
                pEngineName="TestEngine",
                engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
                apiVersion=vk.VK_API_VERSION_1_0
            )
            
            create_info = vk.VkInstanceCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                pApplicationInfo=app_info
            )
            
            instance = vk.vkCreateInstance(create_info, None)
            print("✓ Successfully created Vulkan instance")
            
            # Enumerate devices
            devices = vk.vkEnumeratePhysicalDevices(instance)
            print(f"✓ Found {len(devices)} Vulkan device(s)")
            
            for i, device in enumerate(devices):
                props = vk.vkGetPhysicalDeviceProperties(device)
                print(f"  Device {i}: {props.deviceName}")
                
            vk.vkDestroyInstance(instance, None)
            
        except Exception as e:
            print(f"✗ Failed to create Vulkan instance: {e}")
            print("  This might mean no Vulkan-capable GPU or drivers are available")
            
    except ImportError:
        print("✗ Python vulkan package not installed")
        print("  Install with: pip install vulkan")
    except Exception as e:
        print(f"✗ Error with vulkan package: {e}")
    
    print()

def test_vulkan_forge():
    """Test VulkanForge renderer creation."""
    print("=== Testing VulkanForge ===\n")
    
    try:
        import vulkan_forge
        from vulkan_forge import create_renderer, RenderTarget
        
        print(f"VulkanForge version: {vulkan_forge.__version__}")
        print(f"Package location: {vulkan_forge.__file__}")
        
        # Check backend status
        from vulkan_forge.backend import VULKAN_AVAILABLE
        print(f"Backend VULKAN_AVAILABLE: {VULKAN_AVAILABLE}")
        
        # Try to create renderer
        print("\nCreating renderer with GPU preference...")
        renderer = create_renderer(prefer_gpu=True, enable_validation=False)
        print(f"✓ Created renderer: {type(renderer).__name__}")
        
        if hasattr(renderer, 'gpu_active'):
            print(f"  GPU active: {renderer.gpu_active}")
            
        if hasattr(renderer, 'logical_devices'):
            print(f"  Logical devices: {len(renderer.logical_devices)}")
            
        # Test render
        target = RenderTarget(width=256, height=256)
        renderer.set_render_target(target)
        
        from vulkan_forge import Matrix4x4
        view = Matrix4x4.look_at((0, 0, 5), (0, 0, 0), (0, 1, 0))
        proj = Matrix4x4.perspective(1.0472, 1.0, 0.1, 100.0)
        
        frame = renderer.render([], [], [], view, proj)
        print(f"✓ Rendered frame: {frame.shape}, dtype={frame.dtype}")
        
        # Check if GPU rendering produced output
        if frame.max() > 0:
            unique_values = len(np.unique(frame))
            print(f"  Frame has {unique_values} unique values")
            if unique_values > 10:  # GPU renderer creates gradients
                print("  ✓ GPU rendering appears to be working!")
            else:
                print("  ⚠ Renderer may be using CPU fallback")
        
        renderer.cleanup()
        
    except ImportError as e:
        print(f"✗ Failed to import vulkan_forge: {e}")
    except Exception as e:
        print(f"✗ Error testing VulkanForge: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all checks."""
    print("VulkanForge GPU Setup Diagnostic\n" + "="*40 + "\n")
    
    # System checks
    check_vulkan_installation()
    check_python_vulkan()
    
    # VulkanForge test
    test_vulkan_forge()
    
    print("\n" + "="*40)
    print("\nNext Steps:")
    print("1. If Vulkan runtime is not installed, download from:")
    print("   https://vulkan.lunarg.com/sdk/home")
    print("2. If Python vulkan package is missing:")
    print("   pip install vulkan")
    print("3. Update GPU drivers to latest version")
    print("4. For NVIDIA GPUs, ensure driver 390+ is installed")

if __name__ == "__main__":
    # Need numpy for frame analysis
    try:
        import numpy as np
    except ImportError:
        print("NumPy not installed, skipping frame analysis")
        np = None
    
    main()