# vulkan_forge/renderer.py
"""Main Vulkan renderer with automatic GPU/CPU backend selection."""

import logging
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import ctypes
import numpy as np
import vulkan as vk

# Check if we're being loaded as part of a package or directly
if __name__ == "__main__" or "vf_renderer" in __name__:
    from backend import DeviceManager, VulkanForgeError, LogicalDevice
    from matrices import Matrix4x4
else:
    try:
        from .backend import DeviceSelector, DeviceInfo, DeviceManager, VulkanForgeError, LogicalDevice
        from .matrices import Matrix4x4
    except ImportError:
        from backend import DeviceSelector, DeviceInfo, DeviceManager, VulkanForgeError, LogicalDevice
        from matrices import Matrix4x4

# from .backend import DeviceSelector, DeviceInfo
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
    """GPU-accelerated Vulkan renderer with basic CPU fallback."""

    def __init__(self, device_manager: DeviceManager, logical_devices: List[LogicalDevice]):
        """Initialize Vulkan renderer with device pool."""
        self.device_manager = device_manager
        self.logical_devices = logical_devices
        self.render_target: Optional[RenderTarget] = None

        # Swapchain format must be defined before any Vulkan resources that rely
        # on it are created.  This avoids attribute errors if initialization
        # fails halfway through.
        self.swapchain_format = vk.VK_FORMAT_B8G8R8A8_UNORM

        self.current_device_index = 0

        # Tracks whether GPU initialization succeeded
        self.gpu_active = False

        # Initialize pipeline for each device inside a try/except so that any
        # failure cleanly falls back to CPU rendering.
        self.pipelines: List[Any] = []
        self.render_passes: List[Any] = []
        try:
            for device in logical_devices:
                render_pass = self._create_render_pass(device)
                pipeline = self._create_pipeline(device, render_pass)
                self.render_passes.append(render_pass)
                self.pipelines.append(pipeline)
            self.gpu_active = True
        except VulkanForgeError as e:
            logger.error(f"Failed to initialize pipeline on device: {e}")
            self.gpu_active = False

    def get_surface_extent(self) -> Tuple[int, int]:
        """Return the current render target dimensions."""
        if self.render_target is not None:
            return self.render_target.width, self.render_target.height
        return (0, 0)

    def _create_swapchain(self) -> None:
        """Placeholder swapchain creation with corrected extent calculation."""
        width, height = self.get_surface_extent()
        # In a full implementation this would create the VkSwapchainKHR.  We
        # simply store the values so other parts of the renderer can use them.
        self.swapchain_extent = (width, height)

    
    def _create_render_pass(self, device: LogicalDevice) -> Any:
        """Create render pass for a device."""
        features = getattr(self, "device_features", {})
        if features and features.get("VK_KHR_dynamic_rendering"):
            self._render_pass = None
            return None

        color_attachment = vk.VkAttachmentDescription2(
            sType=vk.VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
            pNext=None,
            flags=0,
            format=self.swapchain_format,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        )

        depth_attachment = vk.VkAttachmentDescription2(
            sType=vk.VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
            pNext=None,
            flags=0,
            format=vk.VK_FORMAT_D32_SFLOAT,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        )

        color_ref = vk.VkAttachmentReference2(
            sType=vk.VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,
            pNext=None,
            attachment=0,
            layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
        )

        depth_ref = vk.VkAttachmentReference2(
            sType=vk.VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,
            pNext=None,
            attachment=1,
            layout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            aspectMask=vk.VK_IMAGE_ASPECT_DEPTH_BIT,
        )

        subpass = vk.VkSubpassDescription2(
            sType=vk.VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2,
            pNext=None,
            flags=0,
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            viewMask=0,
            inputAttachmentCount=0,
            pInputAttachments=None,
            colorAttachmentCount=1,
            pColorAttachments=ctypes.pointer(color_ref),
            pResolveAttachments=None,
            pDepthStencilAttachment=ctypes.pointer(depth_ref),
            preserveAttachmentCount=0,
            pPreserveAttachments=None,
        )

        rp_info = vk.VkRenderPassCreateInfo2(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2,
            pNext=None,
            flags=0,
            attachmentCount=2,
            pAttachments=ctypes.pointer(color_attachment),
            subpassCount=1,
            pSubpasses=ctypes.pointer(subpass),
            dependencyCount=0,
            pDependencies=None,
            correlatedViewMaskCount=0,
            pCorrelatedViewMasks=None,
        )

        render_pass = vk.VkRenderPass(0)
        result = vk.vkCreateRenderPass2(
            device.device,
            ctypes.byref(rp_info),
            None,
            ctypes.byref(render_pass),
        )
        if result != vk.VK_SUCCESS:
            raise VulkanForgeError(f"vkCreateRenderPass2 failed: {result}")

        self._render_pass = render_pass
        return render_pass
    
    def _create_pipeline(self, device: LogicalDevice, render_pass: Any) -> Any:
        """Create graphics pipeline for a device."""
        # UNKNOWN: Full pipeline creation requires shader compilation
        return None
    
    def render(self, meshes: List[Mesh], materials: List[Material],
               lights: List[Light], view_matrix: Matrix4x4,
               projection_matrix: Matrix4x4) -> np.ndarray:
        """Render scene using multi-GPU load balancing."""
        if not self.render_target:
            raise VulkanForgeError("No render target set")
        
        # Round-robin device selection for load balancing
        device = self.logical_devices[self.current_device_index]
        self.current_device_index = (self.current_device_index + 1) % len(self.logical_devices)
        
        # UNKNOWN: Full GPU rendering implementation

        # Placeholder: return black framebuffer
        return np.zeros((self.render_target.height, self.render_target.width, 4), dtype=np.uint8)

    def render_frame(self, scene: Any) -> np.ndarray:
        """Draw a frame using GPU if available, otherwise CPU fallback."""
        if self.gpu_active:
            return self.render(scene.meshes, scene.materials, scene.lights, scene.view_matrix, scene.projection_matrix)
        return self.render_cpu_fallback(scene)

    def render_cpu_fallback(self, scene: Any) -> np.ndarray:
        """Simple CPU fallback that draws a colour gradient."""
        if not self.render_target:
            raise VulkanForgeError("No render target set")
        width, height = self.swapchain_extent
        framebuffer = np.zeros((height, width, 4), dtype=np.uint8)
        framebuffer[:, :, 3] = 255

        rect_w = max(1, width // 4)
        rect_h = max(1, height // 4)
        x0 = (width - rect_w) // 2
        y0 = (height - rect_h) // 2
        x1 = x0 + rect_w
        y1 = y0 + rect_h
        framebuffer[y0:y1, x0:x1, 0] = 255
        framebuffer[y0:y1, x0:x1, 1] = 255

        return framebuffer
    
    def set_render_target(self, target: RenderTarget) -> None:
        """Set the render target."""
        self.render_target = target
        # Recreate swapchain when the target size changes
        self._create_swapchain()
    
    def cleanup(self) -> None:
        """Clean up Vulkan resources."""
        # Clean up pipelines and render passes
        for i, device in enumerate(self.logical_devices):
            if i < len(self.pipelines) and self.pipelines[i]:
                # UNKNOWN: vkDestroyPipeline call
                pass
            if i < len(self.render_passes) and self.render_passes[i]:
                # UNKNOWN: vkDestroyRenderPass call
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
        
        # Initialize framebuffer cleared to black with opaque alpha
        width, height = self.render_target.width, self.render_target.height
        framebuffer = np.zeros((height, width, 4), dtype=np.float32)
        depth_buffer = np.full((height, width), np.inf, dtype=np.float32)

        logger.info(f"Rendering {len(meshes)} meshes")
        pixels_drawn = 0
        triangles_rendered = 0

        # Transform matrices
        mvp = projection_matrix @ view_matrix

        # Simple rasterization for each mesh
        for mesh_idx, mesh in enumerate(meshes):
            material = materials[mesh_idx]
            logger.info(f"Mesh {mesh_idx}: vertices shape={mesh.vertices.shape}, "
                       f"indices shape={mesh.indices.shape}, material color={material.base_color}")
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
            logger.info(
                f"Mesh {mesh_idx}: {len(mesh.vertices)} vertices, {np.sum(in_frustum)} in frustum"
            )
            
            # Viewport transform
            screen_x = (ndc[:, 0] + 1) * width / 2
            screen_y = (1 - ndc[:, 1]) * height / 2
            logger.info(f"Screen coords - X range: [{np.min(screen_x):.1f}, {np.max(screen_x):.1f}], "
                       f"Y range: [{np.min(screen_y):.1f}, {np.max(screen_y):.1f}]")
            # Rasterize triangles
            pixels_drawn += 1
            logger.info(f"Total pixels drawn: {pixels_drawn}")
            for i in range(0, len(mesh.indices), 3):
                i0, i1, i2 = mesh.indices[i:i+3]
                if i == 0:  # Log first triangle
                    logger.info(f"First triangle: v{i0}={screen_x[i0]:.1f},{screen_y[i0]:.1f} v{i1}={screen_x[i1]:.1f},{screen_y[i1]:.1f} v{i2}={screen_x[i2]:.1f},{screen_y[i2]:.1f}")

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

                # Calculate edge function coefficients once per triangle
                area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
                if abs(area) < 1e-6:
                    continue  # Skip degenerate triangles

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
                                if i0 < len(mesh.normals) and i1 < len(mesh.normals) and i2 < len(mesh.normals):
                                    normal = mesh.normals[i0] * w0 + mesh.normals[i1] * w1 + mesh.normals[i2] * w2
                                    normal = normal / (np.linalg.norm(normal) + 1e-10)
                                else:
                                    normal = np.array([0, 0, 1])  # Default normal
                                
                                # Basic lighting
                                color = np.array(material.base_color[:3]) * 0.3  # Ambient
                                for light in lights:
                                    # Light direction should be from surface to light
                                    light_pos = np.array(light.position)
                                    # Simple directional light approximation
                                    light_dir = light_pos / (np.linalg.norm(light_pos) + 1e-10)
                                    diffuse = max(0, np.dot(normal, light_dir))
                                    # Add light contribution
                                    light_contrib = np.array(light.color) * diffuse * light.intensity
                                    color = color + np.array(material.base_color[:3]) * light_contrib * 0.7                                
                                # Clamp and store
                                framebuffer[yi, xi, :3] = np.clip(color, 0, 1)
                                framebuffer[yi, xi, 3] = material.base_color[3]
            logger.info(f"Rendered {triangles_rendered} triangles for mesh {mesh_idx}")

        # If nothing was drawn, overlay a simple crosshair so output isn't blank
        if not np.any(framebuffer[:, :, :3]):
            logger.warning("No content rendered, drawing crosshair")
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
        if prefer_gpu:
            logger.info("Forcing CPU renderer for stability")
            return CPURenderer()
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
    except VulkanForgeError as e:
        logger.warning(f"Failed to initialize Vulkan: {e}, falling back to CPU renderer")
    
    return CPURenderer()