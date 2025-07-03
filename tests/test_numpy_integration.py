import numpy as np
from vulkan_forge.numpy_buffer import StructuredBuffer


def test_structured_buffer_roundtrip():
    buf = StructuredBuffer(64)
    assert buf.data.nbytes == 64
    assert buf.data.dtype == np.float32
    buf.data.fill(1.0)
    buf.upload()
    buf.data.fill(0.0)
    buf.download()
    assert np.allclose(buf.data, 1.0)
