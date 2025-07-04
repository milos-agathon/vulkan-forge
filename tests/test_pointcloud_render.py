import numpy as np
from vulkan_forge import RenderTarget
from vulkan_forge.backend import DeviceManager
from vulkan_forge.renderer import VulkanRenderer
from vulkan_forge.numpy_buffer import numpy_buffer


def test_pointcloud_render_cpu():
    dm = DeviceManager(enable_validation=False)
    renderer = VulkanRenderer(dm, dm.create_logical_devices())
    renderer.width = 32
    renderer.height = 32
    renderer.set_render_target(RenderTarget(width=32, height=32))
    points = np.array([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]], dtype=np.float32)
    allocator = object()
    with numpy_buffer(allocator, points) as buf:
        img = renderer.render_points(buf)
    assert img.sum() > 0
