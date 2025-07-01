# vulkan_forge/vulkan_forge_native.py
"""Stub for the native extension module.

This file exists to prevent import errors when the C++ extension
is not built. The actual native extension would be a .pyd/.so file
that replaces this stub when properly built and installed.
"""

# Placeholder for native functions that might be exposed
def native_version():
    """Return version of native extension."""
    return "stub"

# Any other native functions would go here
# They should raise NotImplementedError or return mock values