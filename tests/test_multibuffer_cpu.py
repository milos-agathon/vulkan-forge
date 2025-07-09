import numpy as np
from vulkan_forge import create_renderer, RenderTarget
from vulkan_forge.numpy_buffer import MultiBuffer


def test_render_indexed_multibuffer_cpu():
    renderer = create_renderer(prefer_gpu=False)
    renderer = create_renderer(prefer_gpu=False, width=4, height=4)
    allocator = object()
    buffers = MultiBuffer(allocator)
    buffers.add_vertex_buffer(
        "vertices", np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    )
    buffers.add_index_buffer(np.array([0, 1, 2], dtype=np.uint32))
    img = renderer.render_indexed(buffers, 3)
    assert img.shape == (4, 4, 4)
