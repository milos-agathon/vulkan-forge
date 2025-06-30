"""Test suite for vulkan-forge new features."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import List, Any

import vulkan_forge
from vulkan_forge.backend import DeviceManager, VulkanForgeError, PhysicalDeviceInfo, LogicalDevice
from vulkan_forge.renderer import VulkanRenderer, CPURenderer, create_renderer, RenderTarget, Mesh, Material, Light
from vulkan_forge.matrices import Matrix4x4


# Fixtures
@pytest.fixture
def render_target():
    """Standard render target for tests."""
    return RenderTarget(width=640, height=480, format="RGBA8")


@pytest.fixture
def simple_mesh():
    """Basic triangle mesh for rendering tests."""
    vertices = np.array([[0, 1, 0], [-1, -1, 0], [1, -1, 0]], dtype=np.float32)
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
    uvs = np.array([[0.5, 0], [0, 1], [1, 1]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint32)
    return Mesh(vertices=vertices, normals=normals, uvs=uvs, indices=indices)


@pytest.fixture
def mock_vulkan_instance(monkeypatch):
    """Mock Vulkan instance creation."""
    mock_instance = Mock()
    monkeypatch.setattr("vulkan.vkCreateInstance", lambda *args: mock_instance)
    monkeypatch.setattr("vulkan.vkDestroyInstance", lambda *args: None)
    return mock_instance


@pytest.fixture
def mock_physical_devices(monkeypatch):
    """Mock physical device enumeration with configurable device count."""
    def _create_mock_devices(count: int, device_types: List[int] = None):
        if device_types is None:
            device_types = [1] * count  # Default to discrete GPUs
        
        devices = []
        for i, device_type in enumerate(device_types[:count]):
            device = Mock()
            properties = Mock()
            properties.deviceName = f"Mock GPU {i}".encode()
            properties.deviceType = device_type
            
            features = Mock()
            memory_props = Mock()
            
            queue_family = Mock()
            queue_family.queueFlags = 1  # VK_QUEUE_GRAPHICS_BIT
            
            devices.append(device)
            
            # Setup return values for device queries
            monkeypatch.setattr(
                "vulkan.vkGetPhysicalDeviceProperties",
                lambda d: properties if d == device else None
            )
            monkeypatch.setattr(
                "vulkan.vkGetPhysicalDeviceFeatures",
                lambda d: features if d == device else None
            )
            monkeypatch.setattr(
                "vulkan.vkGetPhysicalDeviceMemoryProperties",
                lambda d: memory_props if d == device else None
            )
            monkeypatch.setattr(
                "vulkan.vkGetPhysicalDeviceQueueFamilyProperties",
                lambda d: [queue_family] if d == device else None
            )
        
        monkeypatch.setattr("vulkan.vkEnumeratePhysicalDevices", lambda *args: devices)
        return devices
    
    return _create_mock_devices


@pytest.fixture
def mock_logical_device(monkeypatch):
    """Mock logical device creation."""
    mock_device = Mock()
    mock_queue = Mock()
    mock_command_pool = Mock()
    
    monkeypatch.setattr("vulkan.vkCreateDevice", lambda *args: mock_device)
    monkeypatch.setattr("vulkan.vkGetDeviceQueue", lambda *args: mock_queue)
    monkeypatch.setattr("vulkan.vkCreateCommandPool", lambda *args: mock_command_pool)
    monkeypatch.setattr("vulkan.vkDestroyDevice", lambda *args: None)
    monkeypatch.setattr("vulkan.vkDestroyCommandPool", lambda *args: None)
    
    return mock_device, mock_queue, mock_command_pool


# Helper to check if real GPU is available
def has_gpu():
    """Check if system has a real GPU available."""
    try:
        import vulkan as vk
        # UNKNOWN: Direct vkEnumeratePhysicalDevices check without instance
        return False
    except:
        return False


# Tests
class TestDeviceEnumeration:
    """Test device enumeration and categorization."""
    
    def test_device_manager_init(self, mock_vulkan_instance, mock_physical_devices):
        """Test DeviceManager initialization."""
        mock_physical_devices(2)  # Create 2 mock devices
        
        manager = DeviceManager(app_name="TestApp", enable_validation=False)
        
        assert manager.instance is not None
        assert len(manager.physical_devices) == 2
        assert all(isinstance(d, PhysicalDeviceInfo) for d in manager.physical_devices)
    
    def test_device_type_detection(self, mock_vulkan_instance, mock_physical_devices):
        """Test correct device type detection."""
        # Create devices with different types
        device_types = [
            2,  # VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
            1,  # VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
            4,  # VK_PHYSICAL_DEVICE_TYPE_CPU
        ]
        mock_physical_devices(3, device_types)
        
        manager = DeviceManager()
        
        assert manager.physical_devices[0].is_discrete_gpu
        assert manager.physical_devices[1].is_integrated_gpu
        assert manager.physical_devices[2].is_cpu
    
    def test_no_devices_available(self, mock_vulkan_instance, monkeypatch):
        """Test handling when no devices are available."""
        monkeypatch.setattr("vulkan.vkEnumeratePhysicalDevices", lambda *args: [])
        
        manager = DeviceManager()
        
        with pytest.raises(VulkanForgeError, match="No physical devices"):
            manager.create_logical_devices()


class TestAutomaticFallback:
    """Test GPU to CPU fallback behavior."""
    
    def test_fallback_when_no_gpu(self, mock_vulkan_instance, monkeypatch):
        """Test CPU fallback when GPU initialization fails."""
        # Make vkCreateInstance raise to trigger fallback
        monkeypatch.setattr("vulkan.vkCreateInstance", Mock(side_effect=Exception("No Vulkan")))
        
        renderer = create_renderer(prefer_gpu=True)
        
        assert isinstance(renderer, CPURenderer)
        renderer.cleanup()
    
    def test_fallback_with_cpu_only_device(self, mock_vulkan_instance, mock_physical_devices, mock_logical_device):
        """Test fallback when only CPU devices are available."""
        # Create only CPU device
        mock_physical_devices(1, device_types=[4])  # VK_PHYSICAL_DEVICE_TYPE_CPU
        
        renderer = create_renderer(prefer_gpu=True)
        
        assert isinstance(renderer, CPURenderer)
        renderer.cleanup()
    
    def test_no_fallback_when_gpu_preferred_false(self):
        """Test direct CPU renderer creation when GPU not preferred."""
        renderer = create_renderer(prefer_gpu=False)
        
        assert isinstance(renderer, CPURenderer)
        renderer.cleanup()


class TestMultiGPULoadBalancing:
    """Test multi-GPU load balancing functionality."""
    
    @pytest.mark.parametrize("num_gpus", [1, 2, 4])
    def test_round_robin_device_selection(self, mock_vulkan_instance, mock_physical_devices, 
                                         mock_logical_device, render_target, simple_mesh, num_gpus):
        """Test round-robin device selection across multiple GPUs."""
        mock_physical_devices(num_gpus)
        
        manager = DeviceManager()
        logical_devices = manager.create_logical_devices()
        renderer = VulkanRenderer(manager, logical_devices)
        renderer.set_render_target(render_target)
        
        # Track device selection
        initial_device_index = renderer.current_device_index
        
        # Render multiple times
        view = Matrix4x4.identity()
        proj = Matrix4x4.identity()
        
        for i in range(num_gpus * 2):
            renderer.render([simple_mesh], [Material()], [Light((0, 0, 0))], view, proj)
            expected_index = (initial_device_index + i + 1) % num_gpus
            assert renderer.current_device_index == expected_index
        
        renderer.cleanup()
        manager.cleanup()
    
    def test_load_balancing_with_failed_device(self, mock_vulkan_instance, mock_physical_devices,
                                              mock_logical_device):
        """Test load balancing continues when a device fails."""
        mock_physical_devices(3)
        
        manager = DeviceManager()
        logical_devices = manager.create_logical_devices()
        
        # Simulate one device failure by removing it
        if len(logical_devices) > 1:
            logical_devices.pop(1)
        
        renderer = VulkanRenderer(manager, logical_devices)
        
        # Should still work with remaining devices
        assert len(renderer.logical_devices) == 2
        
        renderer.cleanup()
        manager.cleanup()


class TestRendererLifecycle:
    """Test renderer initialization and cleanup."""
    
    def test_vulkan_renderer_init_cleanup(self, mock_vulkan_instance, mock_physical_devices,
                                         mock_logical_device):
        """Test VulkanRenderer initialization and cleanup."""
        mock_physical_devices(1)
        
        manager = DeviceManager()
        logical_devices = manager.create_logical_devices()
        
        renderer = VulkanRenderer(manager, logical_devices)
        
        assert renderer.device_manager == manager
        assert len(renderer.logical_devices) == 1
        assert renderer.render_target is None
        
        # Set render target
        target = RenderTarget(800, 600)
        renderer.set_render_target(target)
        assert renderer.render_target == target
        
        # Cleanup should not raise
        renderer.cleanup()
        manager.cleanup()
    
    def test_cpu_renderer_lifecycle(self):
        """Test CPURenderer initialization and cleanup."""
        renderer = CPURenderer()
        
        assert renderer.render_target is None
        
        target = RenderTarget(320, 240)
        renderer.set_render_target(target)
        assert renderer.render_target == target
        
        renderer.cleanup()


class TestCPURendererOutput:
    """Test CPU renderer output correctness."""
    
    def test_cpu_renderer_buffer_dimensions(self, render_target, simple_mesh):
        """Test CPU renderer produces correct buffer dimensions."""
        renderer = CPURenderer()
        renderer.set_render_target(render_target)
        
        view = Matrix4x4.look_at((0, 0, 5), (0, 0, 0))
        proj = Matrix4x4.perspective(np.pi/4, render_target.width/render_target.height, 0.1, 100)
        
        material = Material(base_color=(1, 0, 0, 1))
        light = Light(position=(0, 5, 5))
        
        framebuffer = renderer.render([simple_mesh], [material], [light], view, proj)
        
        assert framebuffer.shape == (render_target.height, render_target.width, 4)
        assert framebuffer.dtype == np.uint8
        assert np.all(framebuffer >= 0) and np.all(framebuffer <= 255)
        
        renderer.cleanup()
    
    def test_cpu_renderer_with_multiple_meshes(self, render_target):
        """Test CPU renderer with multiple meshes."""
        renderer = CPURenderer()
        renderer.set_render_target(render_target)
        
        # Create two meshes
        mesh1 = Mesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            normals=np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32),
            uvs=np.zeros((3, 2), dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32)
        )
        
        mesh2 = Mesh(
            vertices=np.array([[0, 0, -1], [1, 0, -1], [0, 1, -1]], dtype=np.float32),
            normals=np.array([[0, 0, -1], [0, 0, -1], [0, 0, -1]], dtype=np.float32),
            uvs=np.zeros((3, 2), dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32)
        )
        
        materials = [Material(base_color=(1, 0, 0, 1)), Material(base_color=(0, 1, 0, 1))]
        lights = [Light(position=(0, 0, 10)), Light(position=(0, 0, -10))]
        
        view = Matrix4x4.identity()
        proj = Matrix4x4.orthographic(-2, 2, -2, 2, -10, 10)
        
        framebuffer = renderer.render([mesh1, mesh2], materials, lights, view, proj)
        
        assert framebuffer.shape == (render_target.height, render_target.width, 4)
        
        renderer.cleanup()


class TestVulkanErrorHandling:
    """Test Vulkan error handling and reporting."""
    
    def test_vulkan_forge_error_with_vk_result(self):
        """Test VulkanForgeError captures vkResult properly."""
        error = VulkanForgeError("Pipeline creation failed", vk_result=-1000)
        
        assert str(error) == "Pipeline creation failed"
        assert error.vk_result == -1000
    
    def test_device_creation_failure(self, mock_vulkan_instance, mock_physical_devices, monkeypatch):
        """Test error handling when device creation fails."""
        mock_physical_devices(1)
        
        # Make vkCreateDevice fail
        monkeypatch.setattr("vulkan.vkCreateDevice", Mock(side_effect=Exception("Device creation failed")))
        
        manager = DeviceManager()
        
        with pytest.raises(VulkanForgeError, match="Failed to create any logical devices"):
            manager.create_logical_devices()
        
        manager.cleanup()
    
    def test_command_pool_creation_failure(self, mock_vulkan_instance, mock_physical_devices, monkeypatch):
        """Test error handling when command pool creation fails."""
        mock_physical_devices(1)
        
        mock_device = Mock()
        mock_queue = Mock()
        
        monkeypatch.setattr("vulkan.vkCreateDevice", lambda *args: mock_device)
        monkeypatch.setattr("vulkan.vkGetDeviceQueue", lambda *args: mock_queue)
        monkeypatch.setattr("vulkan.vkCreateCommandPool", Mock(side_effect=Exception("Command pool failed")))
        monkeypatch.setattr("vulkan.vkDestroyDevice", lambda *args: None)
        
        manager = DeviceManager()
        
        with pytest.raises(VulkanForgeError):
            manager.create_logical_devices()
        
        manager.cleanup()


class TestMatrixOperations:
    """Test 3D transformation matrix operations."""
    
    def test_matrix_composition(self):
        """Test matrix multiplication composition."""
        translation = Matrix4x4.translation(1, 2, 3)
        rotation = Matrix4x4.rotation_y(np.pi / 2)
        scale = Matrix4x4.scale(2, 2, 2)
        
        # Compose transformations
        transform = translation @ rotation @ scale
        
        assert transform.data.shape == (4, 4)
        assert not np.allclose(transform.data, np.eye(4))
    
    def test_view_projection_matrices(self):
        """Test view and projection matrix creation."""
        view = Matrix4x4.look_at((0, 0, 10), (0, 0, 0), (0, 1, 0))
        proj = Matrix4x4.perspective(np.pi/4, 16/9, 0.1, 1000)
        
        # View matrix should move camera back
        assert view.data[2, 3] != 0
        
        # Projection matrix should have perspective divide
        assert proj.data[3, 2] == -1
    
    def test_matrix_inverse(self):
        """Test matrix inverse calculation."""
        # Create invertible matrix
        mat = Matrix4x4.translation(5, -3, 2)
        inv = mat.inverse()
        
        # M * M^-1 = I
        identity = mat @ inv
        assert np.allclose(identity.data, np.eye(4), atol=1e-6)


class TestRenderPipeline:
    """Integration tests for complete render pipeline."""
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not has_gpu(), reason="No GPU available")
    def test_gpu_render_pipeline(self):
        """Test complete GPU render pipeline with real device."""
        # UNKNOWN: Full GPU pipeline test requires actual Vulkan device
        pass
    
    def test_cpu_render_pipeline_integration(self, render_target, simple_mesh):
        """Test complete CPU render pipeline."""
        renderer = create_renderer(prefer_gpu=False)
        renderer.set_render_target(render_target)
        
        # Setup scene
        material = Material(
            base_color=(0.8, 0.2, 0.2, 1.0),
            metallic=0.3,
            roughness=0.7,
            emissive=(0.1, 0, 0)
        )
        
        lights = [
            Light(position=(5, 5, 5), color=(1, 1, 1), intensity=2.0),
            Light(position=(-5, 5, -5), color=(0.5, 0.5, 1), intensity=1.0)
        ]
        
        view = Matrix4x4.look_at((3, 3, 3), (0, 0, 0))
        proj = Matrix4x4.perspective(np.pi/3, render_target.width/render_target.height, 0.1, 50)
        
        # Render
        framebuffer = renderer.render([simple_mesh], [material], lights, view, proj)
        
        # Validate output
        assert framebuffer.shape == (render_target.height, render_target.width, 4)
        assert framebuffer.dtype == np.uint8
        
        # Check that some pixels are non-black (mesh was rendered)
        non_black_pixels = np.any(framebuffer[:, :, :3] > 0, axis=2)
        assert np.sum(non_black_pixels) > 0
        
        renderer.cleanup()
    
    def test_render_without_lights(self, render_target, simple_mesh):
        """Test rendering with no lights (ambient only)."""
        renderer = CPURenderer()
        renderer.set_render_target(render_target)
        
        material = Material(base_color=(1, 1, 1, 1))
        
        view = Matrix4x4.identity()
        proj = Matrix4x4.orthographic(-1, 1, -1, 1, -1, 1)
        
        framebuffer = renderer.render([simple_mesh], [material], [], view, proj)
        
        # Should still produce valid output with ambient lighting
        assert framebuffer.shape == (render_target.height, render_target.width, 4)
        
        renderer.cleanup()


# Benchmark test
@pytest.mark.parametrize("num_triangles", [10, 100, 1000])
def test_cpu_renderer_performance(render_target, num_triangles):
    """Benchmark CPU renderer with varying triangle counts."""
    renderer = CPURenderer()
    renderer.set_render_target(render_target)
    
    # Generate random triangles
    vertices = np.random.randn(num_triangles * 3, 3).astype(np.float32) * 5
    normals = np.tile([0, 0, 1], (num_triangles * 3, 1)).astype(np.float32)
    uvs = np.random.rand(num_triangles * 3, 2).astype(np.float32)
    indices = np.arange(num_triangles * 3, dtype=np.uint32)
    
    mesh = Mesh(vertices=vertices, normals=normals, uvs=uvs, indices=indices)
    material = Material()
    light = Light(position=(0, 10, 10))
    
    view = Matrix4x4.look_at((0, 0, 20), (0, 0, 0))
    proj = Matrix4x4.perspective(np.pi/4, render_target.width/render_target.height, 0.1, 100)
    
    # Should complete without errors
    framebuffer = renderer.render([mesh], [material], [light], view, proj)
    assert framebuffer is not None
    
    renderer.cleanup()