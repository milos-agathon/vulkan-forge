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

# Dummy buffer usage flags mirroring Vulkan constants
BUFFER_USAGE_VERTEX_BUFFER = 0x1
BUFFER_USAGE_INDEX_BUFFER = 0x2
BUFFER_USAGE_UNIFORM_BUFFER = 0x10
BUFFER_USAGE_STORAGE_BUFFER = 0x20
BUFFER_USAGE_TRANSFER_SRC = 0x40
BUFFER_USAGE_TRANSFER_DST = 0x80

# Any other native functions would go here
# They should raise NotImplementedError or return mock values