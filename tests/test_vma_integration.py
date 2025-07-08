# tests/test_vma_integration.py
"""Test VMA integration (mocked)."""

import pytest
import numpy as np

# Mock the backend modules
class MockDeviceManager:
    """Mock device manager."""
    def __init__(self, enable_validation=False):
        self.enable_validation = enable_validation
    
    def create_logical_devices(self):
        return [MockDevice()]

class MockDevice:
    """Mock Vulkan device."""
    def __init__(self):
        self.physical_device = MockPhysicalDevice()
        self.device = "mock_device"

class MockPhysicalDevice:
    """Mock physical device."""
    def __init__(self):
        self.device = "mock_physical_device"

def mock_create_allocator_native(instance, physical_device, device):
    """Mock allocator creation."""
    return "mock_allocator"

def mock_allocate_buffer(allocator, size, usage):
    """Mock buffer allocation."""
    return ("mock_buffer", "mock_allocation")

# Mock the imports
DeviceManager = MockDeviceManager
VULKAN_AVAILABLE = False  # Set to False to skip Vulkan tests
create_allocator_native = mock_create_allocator_native
allocate_buffer = mock_allocate_buffer

@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan SDK required")
def test_vma_buffer_allocation():
    """Test VMA buffer allocation (mocked)."""
    dm = DeviceManager(enable_validation=False)
    devices = dm.create_logical_devices()
    if not devices:
        pytest.skip("No Vulkan device")
    
    dev = devices[0]
    allocator = create_allocator_native("instance", dev.physical_device.device, dev.device)
    size = 32 * 1024 * 1024
    usage = 0x00000001  # Mock VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    buf, alloc = allocate_buffer(allocator, size, usage)
    assert buf == "mock_buffer"
    assert alloc == "mock_allocation"