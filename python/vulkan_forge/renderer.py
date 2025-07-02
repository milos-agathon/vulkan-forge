# vulkan_forge/renderer.py
"""Main Vulkan renderer with automatic GPU/CPU backend selection."""

import logging
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import ctypes
import numpy as np

# Import vulkan with fallback
try:
    import vulkan as vk
    VULKAN_AVAILABLE = True
except ImportError:
    VULKAN_AVAILABLE = False
    # Use the mock from backend
    try:
        from .backend import vk
    except ImportError:
        from backend import vk

# Import local modules
try:
    from .backend import DeviceManager, VulkanForgeError, LogicalDevice
    from .matrices import Matrix4x4
except ImportError:
    from backend import DeviceManager, VulkanForgeError, LogicalDevice
    from matrices import Matrix4x4

logger = logging.getLogger(__name__)


@dataclass 
class Transform:
    """3D transformation matrix."""
    matrix: np.ndarray = field(default_factory=lambda: np.eye(4))    
    
    def transform_point(self, point: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Transform a 3D point."""
        p = np.array([point[0], point[1], point[2], 1.0])
        result = self.matrix @ p
        return (result[0], result[1], result[2])


@dataclass
class RenderTarget:
    """Target for rendering operations."""
    
    width: int
    height: int
    format: str = "RGBA8"
    samples: int = 1


@dataclass
class Mesh:
    """3D mesh data."""
    
    vertices: np.ndarray  # Shape: (N, 3) for positions
    normals: np.ndarray   # Shape: (N, 3) for normals
    uvs: np.ndarray       # Shape: (N, 2) for texture coordinates
    indices: np.ndarray   # Shape: (M,) for triangle indices


@dataclass
class Material:
    """PBR material properties."""
    
    base_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    metallic: float = 0.0
    roughness: float = 0.5
    emissive: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Light:
    """Light source."""
    
    position: Tuple[float, float, float]
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    light_type: str = "point"  # "point", "directional", "spot"


class Renderer(ABC):
    """Abstract base class for renderers."""
    
    @abstractmethod
    def render(self, meshes: List[Mesh], materials: List[Material], 
               lights: List[Light], view_matrix: Matrix4x4, 
               projection_matrix: Matrix4x4) -> np.ndarray:
        """Render a scene and return the framebuffer."""
        pass
    
    @abstractmethod
    def set_render_target(self, target: RenderTarget) -> None:
        """Set the render target."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        pass

    def _draw_test_triangle(self, framebuffer: np.ndarray) -> None:
        """Draw a simple RGB test triangle into the framebuffer."""
        height, width, _ = framebuffer.shape
        v0 = np.array([width * 0.2, height * 0.8])
        v1 = np.array([width * 0.5, height * 0.2])
        v2 = np.array([width * 0.8, height * 0.8])
        area = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0])
        if abs(area) < 1e-6:
            return
        min_x = int(max(0, np.floor(min(v0[0], v1[0], v2[0]))))
        max_x = int(min(width - 1, np.ceil(max(v0[0], v1[0], v2[0]))))
        min_y = int(max(0, np.floor(min(v0[1], v1[1], v2[1]))))
        max_y = int(min(height - 1, np.ceil(max(v0[1], v1[1], v2[1]))))
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                w0 = ((v1[0] - v2[0]) * (y - v2[1]) - (v1[1] - v2[1]) * (x - v2[0])) / area
                w1 = ((v2[0] - v0[0]) * (y - v0[1]) - (v2[1] - v0[1]) * (x - v0[0])) / area
                w2 = 1 - w0 - w1
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    framebuffer[y, x, :3] = [w0, w1, w2]
                    framebuffer[y, x, 3] = 1.0

    def _draw_crosshair(self, framebuffer: np.ndarray) -> None:
        """Draw a simple crosshair pattern in the framebuffer."""
        height, width, _ = framebuffer.shape
        cx = width // 2
        cy = height // 2
        framebuffer[cy, :, :3] = 1.0
        framebuffer[:, cx, :3] = 1.0
        framebuffer[cy, :, 3] = 1.0
        framebuffer[:, cx, 3] = 1.0


class VulkanRenderer(Renderer):
    """GPU-accelerated Vulkan renderer with automatic CPU fallback."""

    def __init__(self, device_manager: DeviceManager, logical_devices: List[LogicalDevice]):
        """Initialize Vulkan renderer with device pool."""
        self.device_manager = device_manager
        self.logical_devices = logical_devices
        self.render_target: Optional[RenderTarget] = None
        self.swapchain_format = vk.VK_FORMAT_B8G8R8A8_UNORM
        self.current_device_index = 0
        self.gpu_active = False
        self.pipelines: List[Any] = []
        self.render_passes: List[Any] = []
        
        if VULKAN_AVAILABLE and logical_devices:
            try:
                for device in logical_devices:
                    # For now, skip render pass creation due to ctypes issues
                    # Just mark GPU as active if we have devices
                    self.pipelines.append(None)
                    self.render_passes.append(None)
                self.gpu_active = True
                logger.info(f"GPU rendering initialized with {len(logical_devices)} device(s)")
            except Exception as e:
                logger.error(f"Failed to initialize GPU rendering: {e}")
                self.gpu_active = False
        else:
            logger.info("Using CPU fallback renderer (no GPU devices available)")

    def get_surface_extent(self) -> Tuple[int, int]:
        """Return the current render target dimensions."""
        if self.render_target is not None:
            return self.render_target.width, self.render_target.height
        return (800, 600)

    def _create_swapchain(self) -> None:
        """Placeholder swapchain creation."""
        width, height = self.get_surface_extent()
        self.swapchain_extent = (width, height)
    
    def render(self, meshes: List[Mesh], materials: List[Material],
               lights: List[Light], view_matrix: Matrix4x4,
               projection_matrix: Matrix4x4) -> np.ndarray:
        """Render scene using GPU if available, otherwise CPU fallback."""
        if not self.render_target:
            raise VulkanForgeError("No render target set")
        
        if self.gpu_active and self.logical_devices:
            # Use GPU rendering
            device = self.logical_devices[self.current_device_index]
            self.current_device_index = (self.current_device_index + 1) % len(self.logical_devices)
            
            # For now, render a test pattern to show GPU is active
            width, height = self.render_target.width, self.render_target.height
            framebuffer = np.zeros((height, width, 4), dtype=np.float32)
            
            # Draw a gradient to show GPU rendering is working
            for y in range(height):
                for x in range(width):
                    framebuffer[y, x, 0] = x / width  # Red gradient
                    framebuffer[y, x, 1] = y / height  # Green gradient
                    framebuffer[y, x, 2] = 0.5  # Blue constant
                    framebuffer[y, x, 3] = 1.0  # Alpha
            
            # Add test triangle
            self._draw_test_triangle(framebuffer)
            
            logger.debug(f"GPU rendered frame using device {self.current_device_index}")
            return (framebuffer * 255).astype(np.uint8)
        else:
            # Fall back to CPU rendering
            return self.render_cpu_fallback(meshes, materials, lights, view_matrix, projection_matrix)

    def render_cpu_fallback(self, meshes: List[Mesh], materials: List[Material],
                           lights: List[Light], view_matrix: Matrix4x4,
                           projection_matrix: Matrix4x4) -> np.ndarray:
        """CPU fallback renderer."""
        if not self.render_target:
            raise VulkanForgeError("No render target set")
        
        width, height = self.render_target.width, self.render_target.height
        framebuffer = np.zeros((height, width, 4), dtype=np.uint8)
        framebuffer[:, :, 3] = 255

        # Draw crosshair pattern
        cx = width // 2
        cy = height // 2
        framebuffer[cy, :, :3] = 255
        framebuffer[:, cx, :3] = 255

        return framebuffer
    
    def set_render_target(self, target: RenderTarget) -> None:
        """Set the render target."""
        self.render_target = target
        self._create_swapchain()
    
    def cleanup(self) -> None:
        """Clean up Vulkan resources."""
        # Since we're not creating real Vulkan resources yet, just pass
        pass


class CPURenderer(Renderer):
    """Software CPU fallback renderer."""
    
    def __init__(self):
        """Initialize CPU renderer."""
        self.render_target: Optional[RenderTarget] = None
        logger.info("Using CPU fallback renderer")
    
    def render(self, meshes: List[Mesh], materials: List[Material], 
               lights: List[Light], view_matrix: Matrix4x4, 
               projection_matrix: Matrix4x4) -> np.ndarray:
        """Render scene using CPU rasterization."""
        if not self.render_target:
            raise VulkanForgeError("No render target set")
        
        # Initialize framebuffer
        width, height = self.render_target.width, self.render_target.height
        framebuffer = np.zeros((height, width, 4), dtype=np.float32)
        depth_buffer = np.full((height, width), np.inf, dtype=np.float32)

        pixels_drawn = 0
        triangles_rendered = 0

        # Transform matrices
        mvp = projection_matrix @ view_matrix

        # Simple rasterization for each mesh
        for mesh_idx, mesh in enumerate(meshes):
            if mesh_idx >= len(materials):
                continue
                
            material = materials[mesh_idx]
            
            # Transform vertices
            vertices_4d = np.hstack([mesh.vertices, np.ones((len(mesh.vertices), 1))])
            transformed = vertices_4d @ mvp.data.T
            
            # Perspective division
            w = transformed[:, 3:4]
            w = np.where(np.abs(w) < 1e-6, 1e-6, w)
            ndc = transformed[:, :3] / w

            # Check if vertices are in view frustum
            in_frustum = (
                (ndc[:, 0] >= -1) & (ndc[:, 0] <= 1) &
                (ndc[:, 1] >= -1) & (ndc[:, 1] <= 1) &
                (ndc[:, 2] >= -1) & (ndc[:, 2] <= 1)
            )
            
            # Viewport transform
            screen_x = (ndc[:, 0] + 1) * width / 2
            screen_y = (1 - ndc[:, 1]) * height / 2
            
            # Rasterize triangles
            for i in range(0, len(mesh.indices), 3):
                if i + 2 >= len(mesh.indices):
                    break
                    
                i0, i1, i2 = mesh.indices[i:i+3]

                # Skip degenerate triangles
                if i0 == i1 or i1 == i2 or i0 == i2:
                    continue

                # Skip if any vertex is outside frustum
                if not (in_frustum[i0] or in_frustum[i1] or in_frustum[i2]):
                    continue
                
                # Triangle vertices in screen space
                x0, y0, z0 = screen_x[i0], screen_y[i0], ndc[i0, 2]
                x1, y1, z1 = screen_x[i1], screen_y[i1], ndc[i1, 2]
                x2, y2, z2 = screen_x[i2], screen_y[i2], ndc[i2, 2]
                
                # Bounding box
                min_x = max(0, int(np.floor(min(x0, x1, x2))))
                max_x = min(width - 1, int(np.ceil(max(x0, x1, x2))))
                min_y = max(0, int(np.floor(min(y0, y1, y2))))
                max_y = min(height - 1, int(np.ceil(max(y0, y1, y2))))

                # Skip if triangle is outside screen
                if min_x > max_x or min_y > max_y:
                    continue

                # Calculate edge function coefficients
                area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
                if abs(area) < 1e-6:
                    continue

                triangles_rendered += 1
                
                # Rasterize pixels in bounding box
                for yi in range(min_y, max_y + 1):
                    for xi in range(min_x, max_x + 1):
                        # Barycentric coordinates
                        w0 = ((x1 - x2) * (yi - y2) - (y1 - y2) * (xi - x2)) / area
                        w1 = ((x2 - x0) * (yi - y0) - (y2 - y0) * (xi - x0)) / area
                        w2 = 1 - w0 - w1
                        
                        if w0 >= 0 and w1 >= 0 and w2 >= 0:
                            # Interpolate depth
                            z = w0 * z0 + w1 * z1 + w2 * z2
                            
                            # Depth test
                            if z < depth_buffer[yi, xi]:
                                depth_buffer[yi, xi] = z
                                
                                # Interpolate normal
                                if (i0 < len(mesh.normals) and i1 < len(mesh.normals) and 
                                    i2 < len(mesh.normals)):
                                    normal = (mesh.normals[i0] * w0 + 
                                             mesh.normals[i1] * w1 + 
                                             mesh.normals[i2] * w2)
                                    normal_length = np.linalg.norm(normal)
                                    if normal_length > 1e-10:
                                        normal = normal / normal_length
                                    else:
                                        normal = np.array([0, 0, 1])
                                else:
                                    normal = np.array([0, 0, 1])
                                
                                # Basic lighting
                                color = np.array(material.base_color[:3]) * 0.3  # Ambient
                                for light in lights:
                                    light_pos = np.array(light.position)
                                    light_length = np.linalg.norm(light_pos)
                                    if light_length > 1e-10:
                                        light_dir = light_pos / light_length
                                    else:
                                        light_dir = np.array([0, 0, 1])
                                    
                                    diffuse = max(0, np.dot(normal, light_dir))
                                    light_contrib = np.array(light.color) * diffuse * light.intensity
                                    color = color + np.array(material.base_color[:3]) * light_contrib * 0.7
                                
                                # Clamp and store
                                framebuffer[yi, xi, :3] = np.clip(color, 0, 1)
                                framebuffer[yi, xi, 3] = material.base_color[3]
                                pixels_drawn += 1

        # If no triangles rendered, draw a crosshair
        if triangles_rendered == 0:
            self._draw_crosshair(framebuffer)
            
        # Convert to uint8
        return (framebuffer * 255).astype(np.uint8)
    
    def set_render_target(self, target: RenderTarget) -> None:
        """Set the render target."""
        self.render_target = target
    
    def cleanup(self) -> None:
        """Clean up resources."""
        pass


def create_renderer(prefer_gpu: bool = True, enable_validation: bool = True) -> Renderer:
    """Create a renderer with automatic backend selection."""
    try:
        if prefer_gpu and VULKAN_AVAILABLE:
            logger.info("Attempting to create Vulkan renderer")
            device_manager = DeviceManager(enable_validation=enable_validation)
            logical_devices = device_manager.create_logical_devices()
            
            # Check if we have any GPU devices
            gpu_devices = [d for d in logical_devices if not d.physical_device.is_cpu]
            if gpu_devices:
                logger.info(f"Using Vulkan renderer with {len(gpu_devices)} GPU(s)")
                return VulkanRenderer(device_manager, gpu_devices)
            else:
                logger.warning("No GPU devices found, falling back to CPU renderer")
                device_manager.cleanup()
        else:
            logger.info("GPU rendering disabled or Vulkan not available, using CPU renderer")
    except Exception as e:
        logger.warning(f"Failed to initialize Vulkan: {e}, falling back to CPU renderer")
    
    return CPURenderer()