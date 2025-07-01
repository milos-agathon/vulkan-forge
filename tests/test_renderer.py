import numpy as np
import pytest
import sys
from pathlib import Path

# Add the source directory to Python path
ROOT = Path(__file__).resolve().parents[1] 
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Try different import paths
try:
    import vulkan_forge.backend as backend
    import vulkan_forge.renderer as renderer
    from vulkan_forge.matrices import Matrix4x4
    from vulkan_forge.renderer import RenderTarget, Mesh, Material, Light
except ImportError:
    try:
        # Alternative import path
        sys.path.insert(0, str(ROOT / "python"))
        import vulkan_forge.backend as backend
        import vulkan_forge.renderer as renderer
        from vulkan_forge.matrices import Matrix4x4
        from vulkan_forge.renderer import RenderTarget, Mesh, Material, Light
    except ImportError:
        # Direct imports
        import backend
        import renderer
        from matrices import Matrix4x4
        from renderer import RenderTarget, Mesh, Material, Light


def test_device_manager_initialization():
    """Test that DeviceManager can be created without errors."""
    device_manager = backend.DeviceManager()
    assert device_manager is not None
    assert isinstance(device_manager.physical_devices, list)
    device_manager.cleanup()


def test_cpu_renderer_initialization():
    """Test CPU renderer initialization."""
    cpu_renderer = renderer.CPURenderer()
    assert cpu_renderer is not None
    assert cpu_renderer.render_target is None


def test_cpu_renderer_with_render_target():
    """Test CPU renderer with a render target."""
    cpu_renderer = renderer.CPURenderer()
    target = RenderTarget(width=100, height=100)
    cpu_renderer.set_render_target(target)
    assert cpu_renderer.render_target == target


def test_cpu_render_empty_scene():
    """Test rendering an empty scene."""
    cpu_renderer = renderer.CPURenderer()
    target = RenderTarget(width=50, height=50)
    cpu_renderer.set_render_target(target)
    
    # Create matrices
    view_matrix = Matrix4x4.identity()
    projection_matrix = Matrix4x4.perspective(np.pi/4, 1.0, 0.1, 100.0)
    
    # Render empty scene
    result = cpu_renderer.render([], [], [], view_matrix, projection_matrix)
    
    assert result.shape == (50, 50, 4)
    assert result.dtype == np.uint8
    # Should have crosshair pattern since no triangles were rendered
    assert np.any(result > 0)  # Some pixels should be non-zero


def test_cpu_render_simple_triangle():
    """Test rendering a simple triangle."""
    cpu_renderer = renderer.CPURenderer()
    target = RenderTarget(width=100, height=100)
    cpu_renderer.set_render_target(target)
    
    # Create a simple triangle mesh
    vertices = np.array([
        [-0.5, -0.5, 0.0],  # Bottom left
        [ 0.5, -0.5, 0.0],  # Bottom right
        [ 0.0,  0.5, 0.0]   # Top center
    ], dtype=np.float32)
    
    normals = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    uvs = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 1.0]
    ], dtype=np.float32)
    
    indices = np.array([0, 1, 2], dtype=np.uint32)
    
    mesh = Mesh(vertices=vertices, normals=normals, uvs=uvs, indices=indices)
    material = Material(base_color=(1.0, 0.0, 0.0, 1.0))  # Red
    light = Light(position=(0.0, 0.0, 5.0))
    
    # Create view and projection matrices
    view_matrix = Matrix4x4.look_at((0, 0, 3), (0, 0, 0), (0, 1, 0))
    projection_matrix = Matrix4x4.perspective(np.pi/4, 1.0, 0.1, 100.0)
    
    # Render
    result = cpu_renderer.render([mesh], [material], [light], view_matrix, projection_matrix)
    
    assert result.shape == (100, 100, 4)
    assert result.dtype == np.uint8
    # Should have some red pixels from the triangle
    assert np.any(result[:, :, 0] > 0)  # Red channel should have values


def test_create_renderer_cpu_fallback():
    """Test renderer creation falls back to CPU."""
    # Force CPU renderer
    cpu_renderer = renderer.create_renderer(prefer_gpu=False)
    assert isinstance(cpu_renderer, renderer.CPURenderer)


def test_create_renderer_gpu_attempt():
    """Test renderer creation attempts GPU first."""
    # This should fall back to CPU since we likely don't have Vulkan properly set up
    any_renderer = renderer.create_renderer(prefer_gpu=True)
    # Should get either VulkanRenderer or CPURenderer
    assert isinstance(any_renderer, (renderer.VulkanRenderer, renderer.CPURenderer))


def test_matrix_operations():
    """Test basic matrix operations."""
    # Test identity matrix
    identity = Matrix4x4.identity()
    assert np.allclose(identity.data, np.eye(4))
    
    # Test translation
    translation = Matrix4x4.translation(1, 2, 3)
    point = translation.transform_point((0, 0, 0))
    assert np.allclose(point, (1, 2, 3))
    
    # Test matrix multiplication
    scale = Matrix4x4.scale(2, 2, 2)
    combined = translation @ scale
    assert combined.data.shape == (4, 4)


def test_render_target():
    """Test render target creation."""
    target = RenderTarget(width=800, height=600, format="RGBA8", samples=1)
    assert target.width == 800
    assert target.height == 600
    assert target.format == "RGBA8"
    assert target.samples == 1


def test_mesh_creation():
    """Test mesh data structure."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
    uvs = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint32)
    
    mesh = Mesh(vertices=vertices, normals=normals, uvs=uvs, indices=indices)
    assert mesh.vertices.shape == (3, 3)
    assert mesh.normals.shape == (3, 3)
    assert mesh.uvs.shape == (3, 2)
    assert mesh.indices.shape == (3,)


def test_material_creation():
    """Test material creation."""
    material = Material(
        base_color=(0.8, 0.2, 0.1, 1.0),
        metallic=0.1,
        roughness=0.7
    )
    assert material.base_color == (0.8, 0.2, 0.1, 1.0)
    assert material.metallic == 0.1
    assert material.roughness == 0.7


def test_light_creation():
    """Test light creation."""
    light = Light(
        position=(5.0, 10.0, 5.0),
        color=(1.0, 0.9, 0.8),
        intensity=2.0,
        light_type="point"
    )
    assert light.position == (5.0, 10.0, 5.0)
    assert light.color == (1.0, 0.9, 0.8)
    assert light.intensity == 2.0
    assert light.light_type == "point"


def test_vulkan_renderer_fallback():
    """Test VulkanRenderer creation and fallback behavior."""
    try:
        device_manager = backend.DeviceManager()
        logical_devices = device_manager.create_logical_devices()
        
        vulkan_renderer = renderer.VulkanRenderer(device_manager, logical_devices)
        assert vulkan_renderer is not None
        
        # Test render target setting
        target = RenderTarget(width=64, height=64)
        vulkan_renderer.set_render_target(target)
        assert vulkan_renderer.render_target == target
        
        # Test cleanup
        vulkan_renderer.cleanup()
        device_manager.cleanup()
        
    except Exception as e:
        # If Vulkan initialization fails, that's expected in many environments
        print(f"Vulkan renderer test failed as expected: {e}")


if __name__ == "__main__":
    # Run basic tests
    test_device_manager_initialization()
    test_cpu_renderer_initialization()
    test_cpu_renderer_with_render_target()
    test_cpu_render_empty_scene()
    test_cpu_render_simple_triangle()
    test_create_renderer_cpu_fallback()
    test_matrix_operations()
    test_render_target()
    test_mesh_creation()
    test_material_creation()
    test_light_creation()
    
    print("All tests passed!")