import pytest
import numpy as np

from vulkan_forge.backend import (
    DeviceManager,
    VULKAN_AVAILABLE,
    create_allocator_native,
    allocate_buffer,
)

if VULKAN_AVAILABLE:
    import vulkan as vk


@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan SDK required")
def test_vma_buffer_allocation():
    dm = DeviceManager(enable_validation=False)
    devices = dm.create_logical_devices()
    if not devices:
        pytest.skip("No Vulkan device")
    dev = devices[0]
    allocator = create_allocator_native(dm.instance, dev.physical_device.device, dev.device)
    size = 32 * 1024 * 1024
    usage = vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    buf, alloc = allocate_buffer(allocator, size, usage)
    assert buf != 0
    assert alloc != 0

