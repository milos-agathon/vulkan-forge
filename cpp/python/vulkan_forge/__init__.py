# File: cpp/python/vulkan_forge/__init__.py
"""Vulkan Forge native C++ extension module."""

# Import the compiled C++ extension module
try:
    from .vulkan_forge_native import *
    __version__ = "2.0.0"
    print(f"Vulkan Forge {__version__} - Native module loaded from {__file__}")
except ImportError as e:
    raise ImportError(
        "Failed to import vulkan_forge native extension. "
        "Please ensure the C++ module is built and installed correctly."
    ) from e

# Expose the native module's API
__all__ = [
    # These would be populated based on what's exported from bindings.cpp
    "VulkanRenderer",
    "VulkanDevice", 
    "VulkanBuffer",
    "VulkanPipeline",
    "HeightfieldScene",
    # ... other exported classes/functions
]