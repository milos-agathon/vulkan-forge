import numpy as np
from vulkan_forge import create_renderer, RenderTarget, Mesh, Material
from vulkan_forge.matrices import Matrix4x4


def test_render_cpu_indices_matrix():
    renderer = create_renderer(prefer_gpu=False)
    renderer.set_render_target(RenderTarget(width=4, height=4))

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    normals = np.zeros_like(vertices)
    uvs = np.zeros((3, 2), dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = Mesh(vertices=vertices, normals=normals, uvs=uvs, indices=indices)
    material = Material()

    img = renderer.render([
        mesh], [material], [], Matrix4x4.identity(), Matrix4x4.identity())
    assert img.shape == (4, 4, 4)
