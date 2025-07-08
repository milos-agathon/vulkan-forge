"""Test VMA integration (mocked)."""

import pytest
from .mock_classes import (
    VULKAN_AVAILABLE, MockDeviceManager, mock_create_allocator_native, 
    mock_allocate_buffer
)

@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan SDK required")
def test_vma_buffer_allocation():
    """Test VMA buffer allocation (mocked)."""
    dm = MockDeviceManager(enable_validation=False)
    devices = dm.create_logical_devices()
    if not devices:
        pytest.skip("No Vulkan device")
    
    dev = devices[0]
    allocator = mock_create_allocator_native("instance", dev.physical_device.device, dev.device)
    size = 32 * 1024 * 1024
    usage = 0x00000001  # Mock VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    buf, alloc = mock_allocate_buffer(allocator, size, usage)
    assert buf == "mock_buffer"
    assert alloc == "mock_allocation"

def test_mock_device_manager():
    """Test the mock device manager works."""
    dm = MockDeviceManager(enable_validation=True)
    assert dm.enable_validation == True
    
    devices = dm.create_logical_devices()
    assert len(devices) == 1
    assert devices[0].device == "mock_device"

def test_mock_allocator_functions():
    """Test mock allocator functions work."""
    allocator = mock_create_allocator_native("test_instance", "test_phys_dev", "test_dev")
    assert allocator == "mock_allocator"
    
    buf, alloc = mock_allocate_buffer(allocator, 1024, 1)
    assert buf == "mock_buffer"
    assert alloc == "mock_allocation"