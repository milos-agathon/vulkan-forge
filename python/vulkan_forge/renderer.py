# vulkan_forge/renderer.py
"""Main Vulkan renderer with automatic GPU/CPU backend selection."""

import logging
import ctypes  # ← REQUIRED for c_uint32 array (shader module)
from pathlib import Path  # ← Path used in _create_pipeline
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import ctypes
import numpy as np
import os
import subprocess
from ..renderer import Renderer
from ..terrain_config import TerrainConfig
from ..backend import VulkanContext

# Import vulkan with fallback
try:
    import vulkan as vk

    VULKAN_AVAILABLE = True
except ImportError:
    VULKAN_AVAILABLE = False
    # Fallback mock or stub (for CPU path or CI tests)
    try:
        from .backend import vk
    except ImportError:
        import types

        vk = types.SimpleNamespace()  # minimal stub

# Import local modules (relative when in package, absolute when standalone)
try:
    from .backend import DeviceManager, VulkanForgeError, LogicalDevice
    from .matrices import Matrix4x4
    from .numpy_buffer import NumpyBuffer
except ImportError:
    from backend import DeviceManager, VulkanForgeError, LogicalDevice
    from matrices import Matrix4x4
    from numpy_buffer import NumpyBuffer

logger = logging.getLogger(__name__)

# Default colour used when a material lacks a valid base_color
_DEFAULT_BASE_COLOR = np.array([0.7, 0.7, 0.7, 1.0], dtype=np.float32)


def _extract_base_color(material: "Material") -> Tuple[np.ndarray, float]:
    """Return RGB array and alpha from material, falling back to defaults."""
    try:
        base = np.asarray(getattr(material, "base_color"), dtype=np.float32).ravel()
        if base.size < 3:
            raise ValueError
        rgb = base[:3]
        alpha = float(base[3]) if base.size > 3 else 1.0
    except Exception:
        rgb = _DEFAULT_BASE_COLOR[:3]
        alpha = float(_DEFAULT_BASE_COLOR[3])
    return rgb, alpha


# ─────────────────────────────────────────────────────────────────────────────
# Utility dataclasses
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Transform:
    """4 × 4 matrix wrapper with a convenience transform method."""

    matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))

    def transform_point(
        self, point: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        p = np.array([*point, 1.0], dtype=np.float32)
        r = self.matrix @ p
        return (float(r[0]), float(r[1]), float(r[2]))


@dataclass
class RenderTarget:
    """Off-screen or swap-chain render target."""

    width: int
    height: int
    format: str = "RGBA8"
    samples: int = 1


@dataclass
class Mesh:
    """Simple triangle-mesh container."""

    vertices: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    indices: np.ndarray


@dataclass
class Material:
    base_color: Tuple[float, float, float, float] = (1, 1, 1, 1)
    metallic: float = 0.0
    roughness: float = 0.5
    emissive: Tuple[float, float, float] = (0, 0, 0)


@dataclass
class Light:
    position: Tuple[float, float, float]
    color: Tuple[float, float, float] = (1, 1, 1)
    intensity: float = 1.0
    light_type: str = "point"  # "point" | "directional" | "spot"


# ─────────────────────────────────────────────────────────────────────────────
# Abstract renderer
# ─────────────────────────────────────────────────────────────────────────────
class Renderer(ABC):
    """API-agnostic base class and convenience factory.

    Parameters
    ----------
    width : int, optional
        Target framebuffer width. Defaults to ``1280``.
    height : int, optional
        Target framebuffer height. Defaults to ``720``.
    """

    def __new__(
        cls,
        width: int = 1280,
        height: int = 720,
        prefer_gpu: bool = True,
        enable_validation: bool = True,
    ) -> "Renderer":
        if cls is Renderer:
            impl = create_renderer(
                prefer_gpu=prefer_gpu, enable_validation=enable_validation
            )
            impl.set_target(RenderTarget(width, height))
            impl.width = width
            impl.height = height
            return impl
        return super().__new__(cls)

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self.width = width
        self.height = height
        self._render_target: Optional[RenderTarget] = None

    @abstractmethod
    def render(
        self,
        meshes: List[Mesh],
        materials: List[Material],
        lights: List[Light],
        view_matrix: Matrix4x4,
        projection_matrix: Matrix4x4,
    ) -> np.ndarray: ...

    @abstractmethod
    def set_target(self, target: RenderTarget) -> None: ...

    # --- CPU-only helpers for test suite ---------------------------------
    def set_render_target(self, target: "RenderTarget") -> None:
        """Attach (or replace) the active CPU fallback render target."""
        self._rt = target

    @abstractmethod
    def cleanup(self) -> None: ...

    # ─────────────────────────────────────────────────────────────────────
    # Small CPU helpers for visual debug
    # ─────────────────────────────────────────────────────────────────────
    def _draw_test_triangle(self, fb: np.ndarray) -> None:
        h, w, _ = fb.shape
        v0 = np.array([w * 0.25, h * 0.75])
        v1 = np.array([w * 0.50, h * 0.25])
        v2 = np.array([w * 0.75, h * 0.75])
        area = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0])
        if abs(area) < 1e-6:
            return
        min_x = int(max(0, np.floor(min(v0[0], v1[0], v2[0]))))
        max_x = int(min(w - 1, np.ceil(max(v0[0], v1[0], v2[0]))))
        min_y = int(max(0, np.floor(min(v0[1], v1[1], v2[1]))))
        max_y = int(min(h - 1, np.ceil(max(v0[1], v1[1], v2[1]))))
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                w0 = (
                    (v1[0] - v2[0]) * (y - v2[1]) - (v1[1] - v2[1]) * (x - v2[0])
                ) / area
                w1 = (
                    (v2[0] - v0[0]) * (y - v0[1]) - (v2[1] - v0[1]) * (x - v0[0])
                ) / area
                w2 = 1 - w0 - w1
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    fb[y, x, :3] = [w0, w1, w2]
                    fb[y, x, 3] = 1.0

    def _draw_crosshair(self, fb: np.ndarray) -> None:
        h, w, _ = fb.shape
        cx, cy = w // 2, h // 2
        fb[cy, :, :3] = 1
        fb[:, cx, :3] = 1
        fb[cy, :, 3] = 1
        fb[:, cx, 3] = 1


# ─────────────────────────────────────────────────────────────────────────────
# Vulkan (GPU) renderer
# ─────────────────────────────────────────────────────────────────────────────
# Around line 217 in VulkanRenderer.__init__, add this attribute:
class VulkanRenderer(Renderer):
    """GPU backend with automatic graceful CPU fallback."""

    def __init__(
        self,
        device_manager: Optional[DeviceManager] = None,
        logical_devices: Optional[List[LogicalDevice]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not isinstance(device_manager, DeviceManager):
            # Secondary __init__ call from Renderer factory
            return

        self.device_manager = device_manager
        self.logical_devices = (
            logical_devices if isinstance(logical_devices, list) else []
        )
        self.render_target: Optional[RenderTarget] = None
        self._render_pass: Optional[int] = None
        self._framebuffer: Optional[np.ndarray] = None
        self._enable_runtime_compilation = False

        self.swapchain_format = (
            vk.VK_FORMAT_B8G8R8A8_UNORM if VULKAN_AVAILABLE else None
        )
        self.current_device_index = 0
        self.gpu_active = False

        self.pipelines: List[Any] = []
        self.point_pipelines: List[Any] = []
        self.render_passes: List[Any] = []
        self._compiled_shaders: Dict[str, bytes] = {}
        self.pipeline_layouts: List[Any] = []
        self.point_pipeline_layouts: List[Any] = []
        self.descriptor_set_layouts: List[Any] = []
        self.point_descriptor_set_layouts: List[Any] = []
        self.vertex_buffers: Dict[int, Any] = {}

        if VULKAN_AVAILABLE and self.logical_devices:
            try:
                for dev in self.logical_devices:
                    rp = self._create_render_pass(dev)
                    pl = self._create_pipeline(dev, rp)
                    self.render_passes.append(rp)
                    self.pipelines.append(pl)
                    # Add layouts if created inside pipeline creation
                    # Create point rendering pipeline
                    point_pl = self._create_point_pipeline(dev, rp)
                    self.point_pipelines.append(point_pl)
                self.gpu_active = any(self.pipelines)
                self.gpu_point_active = any(self.point_pipelines)
                logger.info(
                    "GPU rendering enabled"
                    if self.gpu_active
                    else "GPU init failed; CPU fallback"
                )
            except Exception as e:
                logger.exception("Vulkan init failed, switching to CPU: %s", e)
                self.gpu_active = False
        else:
            logger.info("No Vulkan devices detected – CPU fallback renderer engaged.")

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────
    def get_surface_extent(self) -> Tuple[int, int]:
        if self.render_target:
            return self.render_target.width, self.render_target.height
        return (800, 600)

    # (Placeholder swap-chain for headless off-screen rendering)
    def _create_swapchain(self) -> None:
        self.swapchain_extent = self.get_surface_extent()

    def _load_shader_bytecode(self, shader_name: str, stage: str) -> bytes:
        """Load shader bytecode with multiple fallback strategies."""
        # Check cache first
        if shader_name in self._compiled_shaders:
            return self._compiled_shaders[shader_name]

        shader_dir = os.path.join(os.path.dirname(__file__), "shaders")

        # Strategy 1: Try pre-compiled .spv file
        spv_path = os.path.join(shader_dir, f"{shader_name}.spv")
        if not os.path.exists(spv_path):
            # Try without .glsl extension if present
            base_name = shader_name.replace(".glsl", "")
            spv_path = os.path.join(shader_dir, f"{base_name}.spv")

        # Also check in python/vulkan_forge/shaders
        if not os.path.exists(spv_path):
            alt_shader_dir = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "python",
                "vulkan_forge",
                "shaders",
            )
            spv_path = os.path.join(alt_shader_dir, f"{shader_name}.spv")
            if not os.path.exists(spv_path):
                base_name = shader_name.replace(".glsl", "")
                spv_path = os.path.join(alt_shader_dir, f"{base_name}.spv")

        if os.path.exists(spv_path):
            try:
                with open(spv_path, "rb") as f:
                    data = f.read()
                    # Validate SPIR-V magic number
                    if len(data) >= 4 and data[:4] == b"\x03\x02\x23\x07":
                        self._compiled_shaders[shader_name] = data
                        logger.debug(f"Loaded pre-compiled shader: {shader_name}.spv")
                        return data
                    else:
                        logger.warning(f"Invalid SPIR-V file: {spv_path}")
            except Exception as e:
                logger.warning(f"Failed to load {spv_path}: {e}")

        # Strategy 2: Try embedded SPIR-V
        try:
            from .shaders.embedded_spirv import get_shader, has_shader

            base_name = shader_name.replace(".glsl", "")
            if has_shader(base_name):
                data = get_shader(base_name)
                if data:
                    self._compiled_shaders[shader_name] = data
                    logger.debug(f"Loaded embedded shader: {base_name}")
                    return data
        except ImportError:
            logger.debug("No embedded_spirv module available")
        except Exception as e:
            logger.debug(f"Failed to load embedded shader: {e}")

        # Additional check: try loading from current directory shaders
        try:
            direct_spv = os.path.join(
                shader_dir, f"{shader_name.replace('.glsl', '')}.spv"
            )
            if os.path.exists(direct_spv):
                with open(direct_spv, "rb") as f:
                    return f.read()
        except:
            pass

        # Strategy 3: Try runtime compilation (development only)
        if self._enable_runtime_compilation:
            glsl_path = os.path.join(shader_dir, f"{shader_name}.glsl")
            if os.path.exists(glsl_path):
                data = self._compile_shader_runtime(glsl_path, stage)
                if data:
                    return data

        # Strategy 4: Final fallback - minimal valid SPIR-V
        logger.warning(f"All strategies failed for shader: {shader_name}")
        return self._get_minimal_spirv(stage)

        # For point shaders that don't exist, return None instead of minimal SPIR-V
        if "point" in shader_name:
            logger.info(
                f"Point shader {shader_name} not found, will use regular shaders"
            )
            return None

    def _get_minimal_spirv(self, stage: str) -> bytes:
        """Get minimal valid SPIR-V for testing."""
        # Minimal SPIR-V headers for different stages
        # These are valid but empty shaders
        if stage == "vertex":
            return (
                b"\x03\x02\x23\x07\x00\x00\x01\x00\x0b\x00\x08\x00\x01\x00\x00\x00"
                b"\x00\x00\x00\x00\x11\x00\x02\x00\x01\x00\x00\x00\x0b\x00\x06\x00"
                b"\x01\x00\x00\x00\x47\x4c\x53\x4c\x2e\x73\x74\x64\x2e\x34\x35\x30"
                b"\x00\x00\x00\x00\x0e\x00\x03\x00\x00\x00\x00\x00\x01\x00\x00\x00"
                b"\x0f\x00\x07\x00\x00\x00\x00\x00\x01\x00\x00\x00\x6d\x61\x69\x6e"
                b"\x00\x00\x00\x00\x03\x00\x03\x00\x02\x00\x00\x00\xc2\x01\x00\x00"
                b"\x05\x00\x04\x00\x01\x00\x00\x00\x6d\x61\x69\x6e\x00\x00\x00\x00"
                b"\x13\x00\x02\x00\x02\x00\x00\x00\x21\x00\x03\x00\x03\x00\x00\x00"
                b"\x02\x00\x00\x00\x36\x00\x05\x00\x02\x00\x00\x00\x01\x00\x00\x00"
                b"\x00\x00\x00\x00\x03\x00\x00\x00\xf8\x00\x02\x00\x05\x00\x00\x00"
                b"\xfd\x00\x01\x00\x38\x00\x01\x00"
            )
        else:  # fragment
            return (
                b"\x03\x02\x23\x07\x00\x00\x01\x00\x0b\x00\x08\x00\x01\x00\x00\x00"
                b"\x00\x00\x00\x00\x11\x00\x02\x00\x01\x00\x00\x00\x0b\x00\x06\x00"
                b"\x01\x00\x00\x00\x47\x4c\x53\x4c\x2e\x73\x74\x64\x2e\x34\x35\x30"
                b"\x00\x00\x00\x00\x0e\x00\x03\x00\x00\x00\x00\x00\x01\x00\x00\x00"
                b"\x0f\x00\x06\x00\x04\x00\x00\x00\x01\x00\x00\x00\x6d\x61\x69\x6e"
                b"\x00\x00\x00\x00\x03\x00\x03\x00\x02\x00\x00\x00\xc2\x01\x00\x00"
                b"\x05\x00\x04\x00\x01\x00\x00\x00\x6d\x61\x69\x6e\x00\x00\x00\x00"
                b"\x13\x00\x02\x00\x02\x00\x00\x00\x21\x00\x03\x00\x03\x00\x00\x00"
                b"\x02\x00\x00\x00\x36\x00\x05\x00\x02\x00\x00\x00\x01\x00\x00\x00"
                b"\x00\x00\x00\x00\x03\x00\x00\x00\xf8\x00\x02\x00\x05\x00\x00\x00"
                b"\xfd\x00\x01\x00\x38\x00\x01\x00"
            )

    def _compile_shader(self, shader_name: str, stage: str) -> bytes:
        """Main shader compilation entry point."""
        return self._load_shader_bytecode(shader_name, stage)

    def _create_shader_module(self, device: LogicalDevice, spirv_code: bytes) -> Any:
        """Create Vulkan shader module from SPIR-V bytecode."""
        if not spirv_code or len(spirv_code) < 4:
            logger.error("Invalid SPIR-V bytecode")
            return None

        # Validate SPIR-V magic number
        if spirv_code[:4] != b"\x03\x02\x23\x07":
            logger.error("Invalid SPIR-V magic number")
            return None

        try:
            # Convert bytes to ctypes array of uint32
            # Ensure spirv_code is properly aligned and in the right format
            # SPIR-V requires 32-bit words
            if len(spirv_code) % 4 != 0:
                logger.error("SPIR-V bytecode size not aligned to 4 bytes")
                return None

            # Convert bytes to ctypes array of uint32
            num_words = len(spirv_code) // 4
            uint32_array = (ctypes.c_uint32 * num_words)()

            # Use memmove for efficient copying
            ctypes.memmove(ctypes.addressof(uint32_array), spirv_code, len(spirv_code))

            create_info = vk.VkShaderModuleCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                pNext=None,
                flags=0,
                codeSize=len(spirv_code),
                pCode=ctypes.cast(uint32_array, ctypes.POINTER(ctypes.c_uint32)),
            )

            return vk.vkCreateShaderModule(device.device, create_info, None)
        except Exception as e:
            logger.error(f"Failed to create shader module: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────
    # Render-pass / pipeline
    # ─────────────────────────────────────────────────────────────────────
    def _create_render_pass(self, dev: LogicalDevice) -> Any:
        if not VULKAN_AVAILABLE:
            return None
        try:
            color_attachment = vk.VkAttachmentDescription(
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

            # Depth attachment
            depth_attachment = vk.VkAttachmentDescription(
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

            attachments = [color_attachment, depth_attachment]

            # Attachment references
            color_ref = vk.VkAttachmentReference(
                attachment=0, layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
            )

            depth_ref = vk.VkAttachmentReference(
                attachment=1, layout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            )

            # Subpass
            subpass = vk.VkSubpassDescription(
                flags=0,
                pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                inputAttachmentCount=0,
                pInputAttachments=None,
                colorAttachmentCount=1,
                pColorAttachments=[color_ref],
                pResolveAttachments=None,
                pDepthStencilAttachment=depth_ref,
                preserveAttachmentCount=0,
                pPreserveAttachments=None,
            )

            # Subpass dependency
            dependency = vk.VkSubpassDependency(
                srcSubpass=vk.VK_SUBPASS_EXTERNAL,
                dstSubpass=0,
                srcStageMask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                srcAccessMask=0,
                dstStageMask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                dstAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            )

            # Create render pass
            render_pass_info = vk.VkRenderPassCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                pNext=None,
                flags=0,
                attachmentCount=len(attachments),
                pAttachments=attachments,
                subpassCount=1,
                pSubpasses=[subpass],
                dependencyCount=1,
                pDependencies=[dependency],
            )

            # Create the render pass
            render_pass = vk.vkCreateRenderPass(dev.device, render_pass_info, None)
            self._render_pass = render_pass
            return render_pass

        except Exception as e:
            logger.error(f"Failed to create render pass: {e}")
            raise VulkanForgeError(f"Failed to create render pass: {e}")

    def _create_pipeline(self, dev: LogicalDevice, render_pass: Any) -> Any:
        if not VULKAN_AVAILABLE or render_pass is None:
            return None

        # Skip pipeline creation if we don't have shader support
        if not hasattr(self, "_compile_shader"):
            logger.warning(
                "Shader compilation not available - skipping pipeline creation"
            )
            return None

        try:
            # Compile shaders
            shader_dir = Path(__file__).with_name("shaders")
            vert_spv = shader_dir / "vertex.spv"
            if vert_spv.is_file():
                vert_code = vert_spv.read_bytes()
            else:
                vert_code = self._compile_shader("vertex", "vertex")

            frag_spv = shader_dir / "fragment.spv"
            if frag_spv.is_file():
                frag_code = frag_spv.read_bytes()
            else:
                frag_code = self._compile_shader("fragment", "fragment")

            # Check if shader compilation succeeded
            if not vert_code or not frag_code:
                logger.warning("Shader compilation failed - skipping pipeline creation")
                return None

            # Validate SPIR-V magic number before creating modules
            if len(vert_code) < 4 or vert_code[:4] != b"\x03\x02\x23\x07":
                logger.warning("Invalid vertex shader SPIR-V")
                return None

            if len(frag_code) < 4 or frag_code[:4] != b"\x03\x02\x23\x07":
                logger.warning("Invalid fragment shader SPIR-V")
                return None

            vert_module = self._create_shader_module(dev, vert_code)
            frag_module = self._create_shader_module(dev, frag_code)

            # If shader modules couldn't be created, we can't create pipeline
            if not vert_module or not frag_module:
                logger.warning(
                    "Shader modules not available - falling back to CPU rendering"
                )
                return None

            # Shader stages
            vert_stage = vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                pNext=None,
                flags=0,
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                module=vert_module,
                pName="main",
                pSpecializationInfo=None,
            )

            frag_stage = vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                pNext=None,
                flags=0,
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=frag_module,
                pName="main",
                pSpecializationInfo=None,
            )

            shader_stages = [vert_stage, frag_stage]

            # Vertex input
            binding_desc = vk.VkVertexInputBindingDescription(
                binding=0,
                stride=32,  # 3 floats pos + 3 floats normal + 2 floats uv
                inputRate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
            )

            attr_descs = [
                vk.VkVertexInputAttributeDescription(
                    location=0,
                    binding=0,
                    format=vk.VK_FORMAT_R32G32B32_SFLOAT,
                    offset=0,
                ),
                vk.VkVertexInputAttributeDescription(
                    location=1,
                    binding=0,
                    format=vk.VK_FORMAT_R32G32B32_SFLOAT,
                    offset=12,
                ),
                vk.VkVertexInputAttributeDescription(
                    location=2, binding=0, format=vk.VK_FORMAT_R32G32_SFLOAT, offset=24
                ),
            ]

            vertex_input = vk.VkPipelineVertexInputStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                pNext=None,
                flags=0,
                vertexBindingDescriptionCount=1,
                pVertexBindingDescriptions=ctypes.pointer(binding_desc),
                vertexAttributeDescriptionCount=len(attr_descs),
                pVertexAttributeDescriptions=(
                    vk.VkVertexInputAttributeDescription * len(attr_descs)
                )(*attr_descs),
            )

            # Input assembly
            input_assembly = vk.VkPipelineInputAssemblyStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                pNext=None,
                flags=0,
                topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                primitiveRestartEnable=vk.VK_FALSE,
            )

            # Viewport state
            viewport = vk.VkViewport(
                x=0.0,
                y=0.0,
                width=float(self.swapchain_extent[0]),
                height=float(self.swapchain_extent[1]),
                minDepth=0.0,
                maxDepth=1.0,
            )

            scissor = vk.VkRect2D(
                offset=vk.VkOffset2D(x=0, y=0),
                extent=vk.VkExtent2D(
                    width=self.swapchain_extent[0], height=self.swapchain_extent[1]
                ),
            )

            viewport_state = vk.VkPipelineViewportStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                pNext=None,
                flags=0,
                viewportCount=1,
                pViewports=ctypes.pointer(viewport),
                scissorCount=1,
                pScissors=ctypes.pointer(scissor),
            )

            # Rasterizer
            rasterizer = vk.VkPipelineRasterizationStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                pNext=None,
                flags=0,
                depthClampEnable=vk.VK_FALSE,
                rasterizerDiscardEnable=vk.VK_FALSE,
                polygonMode=vk.VK_POLYGON_MODE_FILL,
                lineWidth=1.0,
                cullMode=vk.VK_CULL_MODE_BACK_BIT,
                frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
                depthBiasEnable=vk.VK_FALSE,
                depthBiasConstantFactor=0.0,
                depthBiasClamp=0.0,
                depthBiasSlopeFactor=0.0,
            )

            # Multisampling
            multisampling = vk.VkPipelineMultisampleStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                pNext=None,
                flags=0,
                sampleShadingEnable=vk.VK_FALSE,
                rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
                minSampleShading=1.0,
                pSampleMask=None,
                alphaToCoverageEnable=vk.VK_FALSE,
                alphaToOneEnable=vk.VK_FALSE,
            )

            # Depth stencil
            depth_stencil = vk.VkPipelineDepthStencilStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                pNext=None,
                flags=0,
                depthTestEnable=vk.VK_TRUE,
                depthWriteEnable=vk.VK_TRUE,
                depthCompareOp=vk.VK_COMPARE_OP_LESS,
                depthBoundsTestEnable=vk.VK_FALSE,
                stencilTestEnable=vk.VK_FALSE,
                front=vk.VkStencilOpState(),
                back=vk.VkStencilOpState(),
                minDepthBounds=0.0,
                maxDepthBounds=1.0,
            )

            # Color blending
            color_blend_attachment = vk.VkPipelineColorBlendAttachmentState(
                colorWriteMask=vk.VK_COLOR_COMPONENT_R_BIT
                | vk.VK_COLOR_COMPONENT_G_BIT
                | vk.VK_COLOR_COMPONENT_B_BIT
                | vk.VK_COLOR_COMPONENT_A_BIT,
                blendEnable=vk.VK_FALSE,
                srcColorBlendFactor=vk.VK_BLEND_FACTOR_ONE,
                dstColorBlendFactor=vk.VK_BLEND_FACTOR_ZERO,
                colorBlendOp=vk.VK_BLEND_OP_ADD,
                srcAlphaBlendFactor=vk.VK_BLEND_FACTOR_ONE,
                dstAlphaBlendFactor=vk.VK_BLEND_FACTOR_ZERO,
                alphaBlendOp=vk.VK_BLEND_OP_ADD,
            )

            color_blending = vk.VkPipelineColorBlendStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                pNext=None,
                flags=0,
                logicOpEnable=vk.VK_FALSE,
                logicOp=vk.VK_LOGIC_OP_COPY,
                attachmentCount=1,
                pAttachments=ctypes.pointer(color_blend_attachment),
                blendConstants=[0.0, 0.0, 0.0, 0.0],
            )

            # Create descriptor set layout
            ubo_layout_binding = vk.VkDescriptorSetLayoutBinding(
                binding=0,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT,
                pImmutableSamplers=None,
            )

            layout_bindings = [ubo_layout_binding]

            layout_info = vk.VkDescriptorSetLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                pNext=None,
                flags=0,
                bindingCount=len(layout_bindings),
                pBindings=(vk.VkDescriptorSetLayoutBinding * len(layout_bindings))(
                    *layout_bindings
                ),
            )

            desc_layout = vk.vkCreateDescriptorSetLayout(
                dev.device.value if hasattr(dev.device, "value") else dev.device,
                layout_info,
                None,
            )
            self.descriptor_set_layouts.append(desc_layout)

            # Pipeline layout
            pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                pNext=None,
                flags=0,
                setLayoutCount=1,
                pSetLayouts=ctypes.pointer(desc_layout),
                pushConstantRangeCount=0,
                pPushConstantRanges=None,
            )

            pipeline_layout = vk.vkCreatePipelineLayout(
                dev.device.value if hasattr(dev.device, "value") else dev.device,
                pipeline_layout_info,
                None,
            )
            self.pipeline_layouts.append(pipeline_layout)

            # Create pipeline
            pipeline_info = vk.VkGraphicsPipelineCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                pNext=None,
                flags=0,
                stageCount=len(shader_stages),
                pStages=(vk.VkPipelineShaderStageCreateInfo * len(shader_stages))(
                    *shader_stages
                ),
                pVertexInputState=ctypes.pointer(vertex_input),
                pInputAssemblyState=ctypes.pointer(input_assembly),
                pTessellationState=None,
                pViewportState=ctypes.pointer(viewport_state),
                pRasterizationState=ctypes.pointer(rasterizer),
                pMultisampleState=ctypes.pointer(multisampling),
                pDepthStencilState=ctypes.pointer(depth_stencil),
                pColorBlendState=ctypes.pointer(color_blending),
                pDynamicState=None,
                layout=pipeline_layout,
                renderPass=render_pass,
                subpass=0,
                basePipelineHandle=None,
                basePipelineIndex=-1,
            )

            pipelines = vk.vkCreateGraphicsPipelines(
                dev.device.value if hasattr(dev.device, "value") else dev.device,
                None,
                1,
                ctypes.pointer(pipeline_info),
                None,
            )

            # Clean up shader modules
            vk.vkDestroyShaderModule(
                dev.device.value if hasattr(dev.device, "value") else dev.device,
                vert_module,
                None,
            )
            vk.vkDestroyShaderModule(
                dev.device.value if hasattr(dev.device, "value") else dev.device,
                frag_module,
                None,
            )

            return pipelines[0] if pipelines else None

        except Exception as e:
            logger.error(f"Failed to create pipeline: {e}")
            if hasattr(self, "descriptor_set_layouts") and self.descriptor_set_layouts:
                self.descriptor_set_layouts.pop()  # Remove the failed layout
            return None

    def _create_point_pipeline(self, dev: LogicalDevice, render_pass: Any) -> Any:
        """Create a pipeline specifically for point rendering."""
        if not VULKAN_AVAILABLE or render_pass is None:
            return None

        try:
            # Compile point shaders
            shader_dir = Path(__file__).with_name("shaders")

            # Try pre-compiled first
            vert_spv = shader_dir / "point_vertex.spv"
            if vert_spv.is_file():
                vert_code = vert_spv.read_bytes()
            else:
                logger.debug(
                    "Point vertex shader not found, using regular vertex shader"
                )
                vert_code = self._compile_shader("vertex", "vertex")

            frag_spv = shader_dir / "point_fragment.spv"
            if frag_spv.is_file():
                frag_code = frag_spv.read_bytes()
            else:
                logger.debug(
                    "Point fragment shader not found, using regular fragment shader"
                )
                frag_code = self._compile_shader("fragment", "fragment")

            if not vert_code or not frag_code:
                logger.warning("Shader compilation failed - skipping point pipeline")
                return None

            vert_module = self._create_shader_module(dev, vert_code)
            frag_module = self._create_shader_module(dev, frag_code)

            if not vert_module or not frag_module:
                logger.warning("Shader modules not available")
                return None

            # Shader stages
            vert_stage = vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                pNext=None,
                flags=0,
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                module=vert_module,
                pName="main",
                pSpecializationInfo=None,
            )

            frag_stage = vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                pNext=None,
                flags=0,
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=frag_module,
                pName="main",
                pSpecializationInfo=None,
            )

            shader_stages = [vert_stage, frag_stage]

            # Vertex input for points (position + color)
            binding_descs = [
                vk.VkVertexInputBindingDescription(
                    binding=0,
                    stride=12,  # 3 floats for position
                    inputRate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
                ),
                vk.VkVertexInputBindingDescription(
                    binding=1,
                    stride=16,  # 4 floats for color
                    inputRate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
                ),
            ]

            attr_descs = [
                vk.VkVertexInputAttributeDescription(
                    location=0,
                    binding=0,
                    format=vk.VK_FORMAT_R32G32B32_SFLOAT,
                    offset=0,
                ),
                vk.VkVertexInputAttributeDescription(
                    location=1,
                    binding=1,
                    format=vk.VK_FORMAT_R32G32B32A32_SFLOAT,
                    offset=0,
                ),
            ]

            vertex_input = vk.VkPipelineVertexInputStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                pNext=None,
                flags=0,
                vertexBindingDescriptionCount=len(binding_descs),
                pVertexBindingDescriptions=(
                    vk.VkVertexInputBindingDescription * len(binding_descs)
                )(*binding_descs),
                vertexAttributeDescriptionCount=len(attr_descs),
                pVertexAttributeDescriptions=(
                    vk.VkVertexInputAttributeDescription * len(attr_descs)
                )(*attr_descs),
            )

            # Input assembly - POINT_LIST topology
            input_assembly = vk.VkPipelineInputAssemblyStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                pNext=None,
                flags=0,
                topology=vk.VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
                primitiveRestartEnable=vk.VK_FALSE,
            )

            # Viewport state
            viewport = vk.VkViewport(
                x=0.0,
                y=0.0,
                width=float(self.swapchain_extent[0]),
                height=float(self.swapchain_extent[1]),
                minDepth=0.0,
                maxDepth=1.0,
            )

            scissor = vk.VkRect2D(
                offset=vk.VkOffset2D(x=0, y=0),
                extent=vk.VkExtent2D(
                    width=self.swapchain_extent[0], height=self.swapchain_extent[1]
                ),
            )

            viewport_state = vk.VkPipelineViewportStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                pNext=None,
                flags=0,
                viewportCount=1,
                pViewports=ctypes.pointer(viewport),
                scissorCount=1,
                pScissors=ctypes.pointer(scissor),
            )

            # Rasterizer with point size enabled
            rasterizer = vk.VkPipelineRasterizationStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                pNext=None,
                flags=0,
                depthClampEnable=vk.VK_FALSE,
                rasterizerDiscardEnable=vk.VK_FALSE,
                polygonMode=vk.VK_POLYGON_MODE_FILL,
                lineWidth=1.0,
                cullMode=vk.VK_CULL_MODE_NONE,  # No culling for points
                frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
                depthBiasEnable=vk.VK_FALSE,
                depthBiasConstantFactor=0.0,
                depthBiasClamp=0.0,
                depthBiasSlopeFactor=0.0,
            )

            # Multisampling
            multisampling = vk.VkPipelineMultisampleStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                pNext=None,
                flags=0,
                sampleShadingEnable=vk.VK_FALSE,
                rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
                minSampleShading=1.0,
                pSampleMask=None,
                alphaToCoverageEnable=vk.VK_FALSE,
                alphaToOneEnable=vk.VK_FALSE,
            )

            # Depth stencil
            depth_stencil = vk.VkPipelineDepthStencilStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                pNext=None,
                flags=0,
                depthTestEnable=vk.VK_TRUE,
                depthWriteEnable=vk.VK_TRUE,
                depthCompareOp=vk.VK_COMPARE_OP_LESS,
                depthBoundsTestEnable=vk.VK_FALSE,
                stencilTestEnable=vk.VK_FALSE,
                front=vk.VkStencilOpState(),
                back=vk.VkStencilOpState(),
                minDepthBounds=0.0,
                maxDepthBounds=1.0,
            )

            # Color blending
            color_blend_attachment = vk.VkPipelineColorBlendAttachmentState(
                colorWriteMask=vk.VK_COLOR_COMPONENT_R_BIT
                | vk.VK_COLOR_COMPONENT_G_BIT
                | vk.VK_COLOR_COMPONENT_B_BIT
                | vk.VK_COLOR_COMPONENT_A_BIT,
                blendEnable=vk.VK_TRUE,
                srcColorBlendFactor=vk.VK_BLEND_FACTOR_SRC_ALPHA,
                dstColorBlendFactor=vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                colorBlendOp=vk.VK_BLEND_OP_ADD,
                srcAlphaBlendFactor=vk.VK_BLEND_FACTOR_ONE,
                dstAlphaBlendFactor=vk.VK_BLEND_FACTOR_ZERO,
                alphaBlendOp=vk.VK_BLEND_OP_ADD,
            )

            color_blending = vk.VkPipelineColorBlendStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                pNext=None,
                flags=0,
                logicOpEnable=vk.VK_FALSE,
                logicOp=vk.VK_LOGIC_OP_COPY,
                attachmentCount=1,
                pAttachments=ctypes.pointer(color_blend_attachment),
                blendConstants=[0.0, 0.0, 0.0, 0.0],
            )

            # Create descriptor set layout for uniform buffer
            ubo_layout_binding = vk.VkDescriptorSetLayoutBinding(
                binding=0,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT,
                pImmutableSamplers=None,
            )

            layout_info = vk.VkDescriptorSetLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                pNext=None,
                flags=0,
                bindingCount=1,
                pBindings=ctypes.pointer(ubo_layout_binding),
            )

            desc_layout = vk.vkCreateDescriptorSetLayout(
                dev.device.value if hasattr(dev.device, "value") else dev.device,
                layout_info,
                None,
            )
            self.point_descriptor_set_layouts.append(desc_layout)

            # Pipeline layout
            pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                pNext=None,
                flags=0,
                setLayoutCount=1,
                pSetLayouts=ctypes.pointer(desc_layout),
                pushConstantRangeCount=0,
                pPushConstantRanges=None,
            )

            pipeline_layout = vk.vkCreatePipelineLayout(
                dev.device.value if hasattr(dev.device, "value") else dev.device,
                pipeline_layout_info,
                None,
            )
            self.point_pipeline_layouts.append(pipeline_layout)

            # Create pipeline
            pipeline_info = vk.VkGraphicsPipelineCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                pNext=None,
                flags=0,
                stageCount=len(shader_stages),
                pStages=(vk.VkPipelineShaderStageCreateInfo * len(shader_stages))(
                    *shader_stages
                ),
                pVertexInputState=ctypes.pointer(vertex_input),
                pInputAssemblyState=ctypes.pointer(input_assembly),
                pTessellationState=None,
                pViewportState=ctypes.pointer(viewport_state),
                pRasterizationState=ctypes.pointer(rasterizer),
                pMultisampleState=ctypes.pointer(multisampling),
                pDepthStencilState=ctypes.pointer(depth_stencil),
                pColorBlendState=ctypes.pointer(color_blending),
                pDynamicState=None,
                layout=pipeline_layout,
                renderPass=render_pass,
                subpass=0,
                basePipelineHandle=None,
                basePipelineIndex=-1,
            )

            pipelines = vk.vkCreateGraphicsPipelines(
                dev.device.value if hasattr(dev.device, "value") else dev.device,
                None,
                1,
                ctypes.pointer(pipeline_info),
                None,
            )

            # Clean up shader modules
            vk.vkDestroyShaderModule(
                dev.device.value if hasattr(dev.device, "value") else dev.device,
                vert_module,
                None,
            )
            vk.vkDestroyShaderModule(
                dev.device.value if hasattr(dev.device, "value") else dev.device,
                frag_module,
                None,
            )

            return pipelines[0] if pipelines else None

        except Exception as e:
            logger.error(f"Failed to create point pipeline: {e}")
            if (
                hasattr(self, "point_descriptor_set_layouts")
                and self.point_descriptor_set_layouts
            ):
                self.point_descriptor_set_layouts.pop()
            if hasattr(self, "point_pipeline_layouts") and self.point_pipeline_layouts:
                self.point_pipeline_layouts.pop()
            return None

    def set_target(self, target: RenderTarget) -> None:
        self.render_target = target
        self._create_swapchain()

    def set_vertex_buffer(self, numpy_buf, binding: int = 0) -> None:
        """Bind a vertex buffer for rendering."""
        if not hasattr(self, "vertex_buffers"):
            self.vertex_buffers = {}
        if self.gpu_active and hasattr(numpy_buf, "gpu_buffer"):
            self.vertex_buffers[binding] = int(numpy_buf.gpu_buffer)
        else:
            array = getattr(numpy_buf, "_array", numpy_buf)
            self.vertex_buffers[binding] = np.asarray(array)

    def render_points(
        self,
        vertex_buffer: Any,
        model_matrix: Optional[Matrix4x4] = None,
        view_matrix: Optional[Matrix4x4] = None,
        projection_matrix: Optional[Matrix4x4] = None,
        point_size: int = 2,
        color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    ) -> np.ndarray:
        """Render a point cloud from ``vertex_buffer``.

        Falls back to a simple CPU rasteriser when GPU shaders are unavailable.
        """

        if model_matrix is None:
            model_matrix = Matrix4x4.identity()
        if view_matrix is None:
            view_matrix = Matrix4x4.identity()
        if projection_matrix is None:
            projection_matrix = Matrix4x4.identity()

        if self.gpu_active and self.point_pipelines and any(self.point_pipelines):
            # GPU path not yet implemented
            try:
                return self._render_points_gpu(
                    vertex_buffer,
                    model_matrix,
                    view_matrix,
                    projection_matrix,
                    point_size,
                    color,
                )
            except Exception as e:  # pragma: no cover - GPU optional
                logger.error("GPU point rendering failed: %s", e)

        return self._render_points_cpu(
            vertex_buffer, model_matrix, view_matrix, projection_matrix, color
        )

    def _render_points_cpu(
        self,
        vertex_buffer: Any,
        model_matrix: Matrix4x4,
        view_matrix: Matrix4x4,
        projection_matrix: Matrix4x4,
        color: Tuple[float, float, float],
    ) -> np.ndarray:
        verts = np.asarray(
            getattr(
                vertex_buffer,
                "_array",
                getattr(
                    vertex_buffer,
                    "host_view",
                    getattr(vertex_buffer, "array", vertex_buffer),
                ),
            ),
            dtype=np.float32,
            order="C",
        )
        if verts.ndim != 2 or verts.shape[1] < 3:
            raise ValueError("vertex_buffer must contain Nx3 positions")

        mvp = projection_matrix.data @ view_matrix.data @ model_matrix.data
        coords = np.c_[verts[:, :3], np.ones(len(verts))] @ mvp.T
        ndc = coords[:, :3] / np.where(
            np.abs(coords[:, 3:4]) < 1e-6, 1e-6, coords[:, 3:4]
        )

        x = ((ndc[:, 0] + 1) * 0.5 * self.width).astype(int)
        y = ((1 - ndc[:, 1]) * 0.5 * self.height).astype(int)
        in_bounds = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)

        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        col = (np.array(color) * 255).astype(np.uint8)
        img[y[in_bounds], x[in_bounds]] = col

        return img

    def _render_points_gpu(
        self,
        vertex_buffer: Any,
        model_matrix: Matrix4x4,
        view_matrix: Matrix4x4,
        projection_matrix: Matrix4x4,
        point_size: int,
        color: Tuple[float, float, float],
    ) -> np.ndarray:
        """Placeholder GPU implementation (not yet available)."""
        raise NotImplementedError

    def render_indexed(
        self,
        vertex_buffer: Any,
        index_buffer: Any,
        model_matrix: Optional[Matrix4x4] = None,
        view_matrix: Optional[Matrix4x4] = None,
        projection_matrix: Optional[Matrix4x4] = None,
        wireframe: bool = False,
    ) -> np.ndarray:
        """Render indexed geometry.

        Parameters
        ----------
        vertex_buffer : NumpyBuffer or MultiBuffer or numpy.ndarray
            Source vertex buffer. ``MultiBuffer`` instances must expose a
            ``host_view`` attribute for CPU rendering.
        index_buffer : Any
            Source index buffer or index count.

        model_matrix, view_matrix, projection_matrix : Matrix4x4
            4×4 transformation matrices applied in that order.

        Notes
        -----
        Accepts any object exposing ``.array``/``._array``/``.host_view`` or a
        ``numpy.ndarray``.
        Compatible with NumPy 1.x / 2.x; uses ``copy='if_needed'``.
        """
        if model_matrix is None:
            model_matrix = Matrix4x4.identity()
        if view_matrix is None:
            view_matrix = Matrix4x4.identity()
        if projection_matrix is None:
            projection_matrix = Matrix4x4.identity()
        if not self.render_target:
            raise VulkanForgeError("Render target not set")

        if isinstance(index_buffer, NumpyBuffer):
            index_count = index_buffer.count
            idx_ptr = index_buffer.gpu_buffer or None
        elif hasattr(index_buffer, "host_view"):
            index_count = getattr(index_buffer.host_view, "size", 0)
            idx_ptr = getattr(index_buffer, "gpu_buffer", None)
        elif isinstance(index_buffer, np.ndarray):
            index_count = index_buffer.size
            idx_ptr = None
        elif isinstance(index_buffer, int):
            index_count = index_buffer
            idx_ptr = None
        else:
            raise TypeError("index_buffer must be NumpyBuffer, numpy.ndarray or int")

        if self.gpu_active and hasattr(vk, "vkCmdDrawIndexed"):
            try:
                cmd_buf = getattr(self, "command_buffer", None)
                if cmd_buf is not None:
                    vb_handle = int(getattr(vertex_buffer, "gpu_buffer", 0))
                    buffers = (ctypes.c_uint64 * 1)(ctypes.c_uint64(vb_handle))
                    offsets = (ctypes.c_ulonglong * 1)(0)
                    vk.vkCmdBindVertexBuffers(cmd_buf, 0, 1, buffers, offsets)

                    if idx_ptr is not None:
                        ib_handle = int(idx_ptr)
                        dtype = getattr(
                            getattr(index_buffer, "_array", index_buffer),
                            "dtype",
                            np.uint32,
                        )
                        index_type = (
                            vk.VK_INDEX_TYPE_UINT16
                            if dtype == np.uint16
                            else vk.VK_INDEX_TYPE_UINT32
                        )
                        vk.vkCmdBindIndexBuffer(cmd_buf, ib_handle, 0, index_type)

                    vk.vkCmdDrawIndexed(cmd_buf, index_count, 1, 0, 0, 0)
            except Exception as e:
                logger.error("GPU indexed draw failed: %s", e)

        # -- resolve vertex data for CPU raster --
        if hasattr(vertex_buffer, "get_vertex_buffer"):
            entry = vertex_buffer.get_vertex_buffer("vertices")
            if not entry:
                raise ValueError("MultiBuffer missing 'vertices' entry")
            _, buf = entry
            verts_arr = getattr(buf, "_array", getattr(buf, "array", buf))
        elif hasattr(vertex_buffer, "_array"):
            verts_arr = vertex_buffer._array
        elif hasattr(vertex_buffer, "host_view"):
            verts_arr = vertex_buffer.host_view
        elif hasattr(vertex_buffer, "array"):
            verts_arr = vertex_buffer.array
        elif isinstance(vertex_buffer, np.ndarray):
            verts_arr = vertex_buffer
        else:
            raise TypeError(
                "Unsupported vertex_buffer type; expected NumpyBuffer, "
                "MultiBuffer, NumpyBufferCtx, or ndarray."
            )

        verts_arr = np.asarray(verts_arr, dtype=np.float32, order="C")
        vertices = verts_arr

        if isinstance(index_buffer, NumpyBuffer):
            idx_arr = index_buffer._array
        elif hasattr(index_buffer, "host_view"):
            idx_arr = index_buffer.host_view
        elif isinstance(index_buffer, np.ndarray):
            idx_arr = index_buffer
        elif isinstance(index_buffer, int):
            # When index_buffer is just a count, look for indices in vertex_buffer
            if hasattr(vertex_buffer, "get_index_buffer"):
                idx_buf = vertex_buffer.get_index_buffer()
                if idx_buf:
                    idx_arr = idx_buf._array if hasattr(idx_buf, "_array") else idx_buf
                else:
                    # No index buffer, create sequential indices
                    idx_arr = np.arange(index_buffer, dtype=np.int32)
            else:
                # Create sequential indices
                idx_arr = np.arange(index_buffer, dtype=np.int32)
        else:
            raise TypeError("index_buffer must be NumpyBuffer, numpy.ndarray or int")

        indices = np.asarray(idx_arr, dtype=np.int32, order="C")
        verts = np.hstack([vertices, np.ones((len(vertices), 1), dtype=np.float32)])
        world = verts @ model_matrix.data.T
        mesh = Mesh(
            vertices=world[:, :3],
            normals=np.zeros_like(world[:, :3]),
            uvs=np.zeros((len(vertices), 2), dtype=np.float32),
            indices=indices,
        )
        material = Material()
        return self.render_cpu_fallback(
            [mesh], [material], [], view_matrix, projection_matrix
        )

    # ─────────────────────────────────────────────────────────────────────
    # Main render entry
    # ─────────────────────────────────────────────────────────────────────
    def render(
        self,
        meshes: List[Mesh],
        materials: List[Material],
        lights: List[Light],
        view_matrix: Matrix4x4,
        projection_matrix: Matrix4x4,
    ) -> np.ndarray:
        if not self.render_target:
            raise VulkanForgeError("Render target not set")

        w, h = self.render_target.width, self.render_target.height
        # Since GPU rendering isn't fully implemented, use CPU fallback
        return self.render_cpu_fallback(
            meshes, materials, lights, view_matrix, projection_matrix
        )

    def render_cpu_fallback(
        self,
        meshes: List[Mesh],
        materials: List[Material],
        lights: List[Light],
        view_matrix: Matrix4x4,
        projection_matrix: Matrix4x4,
    ) -> np.ndarray:
        """CPU software rasterizer."""
        if not self.render_target:
            raise VulkanForgeError("Render target not set")

        w, h = self.render_target.width, self.render_target.height
        fb = np.zeros((h, w, 4), dtype=np.float32)
        depth_buffer = np.full((h, w), np.inf, dtype=np.float32)

        # Transform matrices
        mvp = (projection_matrix @ view_matrix).data

        for mesh_idx, mesh in enumerate(meshes):
            if mesh_idx >= len(materials):
                continue

            mat = materials[mesh_idx]
            base_rgb, alpha = _extract_base_color(mat)
            base_rgb[:] = 0.5

            verts = np.hstack(
                [mesh.vertices, np.ones((len(mesh.vertices), 1), dtype=np.float32)]
            )
            clip = verts @ mvp.T
            w_vals = np.where(np.abs(clip[:, 3:4]) < 1e-6, 1e-6, clip[:, 3:4])
            ndc = clip[:, :3] / w_vals

            screen_x = (ndc[:, 0] + 1) * w / 2
            screen_y = (1 - ndc[:, 1]) * h / 2

            indices = np.ascontiguousarray(mesh.indices, dtype=np.int32)
            for tri in indices.reshape(-1, 3):
                i0, i1, i2 = map(int, tri)
                if i0 >= len(screen_x) or i1 >= len(screen_x) or i2 >= len(screen_x):
                    continue
                tri_ndc = ndc[[i0, i1, i2]]
                if not np.any(np.all(np.abs(tri_ndc) <= 1, axis=1)):
                    continue

                x0, y0, z0 = screen_x[i0], screen_y[i0], ndc[i0, 2]
                x1, y1, z1 = screen_x[i1], screen_y[i1], ndc[i1, 2]
                x2, y2, z2 = screen_x[i2], screen_y[i2], ndc[i2, 2]

                min_x = max(0, int(np.floor(min(x0, x1, x2))))
                max_x = min(w - 1, int(np.ceil(max(x0, x1, x2))))
                min_y = max(0, int(np.floor(min(y0, y1, y2))))
                max_y = min(h - 1, int(np.ceil(max(y0, y1, y2))))
                if min_x > max_x or min_y > max_y:
                    continue

                area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
                if abs(area) < 1e-6:
                    continue

                xs = np.arange(min_x, max_x + 1)
                ys = np.arange(min_y, max_y + 1)
                xi, yi = np.meshgrid(xs, ys)

                w0 = ((x1 - xi) * (y2 - yi) - (y1 - yi) * (x2 - xi)) / area
                w1 = ((x2 - xi) * (y0 - yi) - (y2 - yi) * (x0 - xi)) / area
                w2 = 1.0 - w0 - w1
                mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
                if not np.any(mask):
                    continue

                z = w0 * z0 + w1 * z1 + w2 * z2
                depth = depth_buffer[min_y : max_y + 1, min_x : max_x + 1]
                depth_mask = mask & (z < depth)
                if not np.any(depth_mask):
                    continue
                depth[depth_mask] = z[depth_mask]

                if len(mesh.normals):
                    n0 = mesh.normals[i0]
                    n1 = mesh.normals[i1]
                    n2 = mesh.normals[i2]
                    normal = (
                        n0 * w0[..., None] + n1 * w1[..., None] + n2 * w2[..., None]
                    )
                    nlen = np.linalg.norm(normal, axis=2, keepdims=True)
                    normal = np.divide(
                        normal, nlen, out=np.zeros_like(normal), where=nlen > 1e-10
                    )
                else:
                    normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    normal = np.broadcast_to(normal, (*w0.shape, 3))

                if not lights:
                    color = np.broadcast_to(base_rgb, (*w0.shape, 3)).astype(np.float32)
                else:
                    color = base_rgb * 0.3
                    color = np.broadcast_to(color, (*w0.shape, 3)).astype(np.float32)
                    for light in lights:
                        ldir = np.array(light.position, dtype=np.float32)
                        lnorm = np.linalg.norm(ldir)
                        if lnorm > 1e-10:
                            ldir = ldir / lnorm
                            diffuse = np.maximum(0.0, np.sum(normal * ldir, axis=2))
                            color += base_rgb * diffuse[..., None] * 0.7

                fb_slice = fb[min_y : max_y + 1, min_x : max_x + 1]
                color = np.clip(color, 0, 1)
                fb_slice[depth_mask, :3] = color[depth_mask]
                fb_slice[depth_mask, 3] = alpha

        return (fb * 255).astype(np.uint8)

    # ─────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────
    def cleanup(self) -> None:
        if not VULKAN_AVAILABLE:
            return
        for idx, dev in enumerate(self.logical_devices):
            if idx < len(self.pipelines) and self.pipelines[idx]:
                vk.vkDestroyPipeline(dev.device, self.pipelines[idx], None)
            if idx < len(self.render_passes) and self.render_passes[idx]:
                vk.vkDestroyRenderPass(dev.device, self.render_passes[idx], None)
            if idx < len(self.pipeline_layouts) and self.pipeline_layouts[idx]:
                vk.vkDestroyPipelineLayout(dev.device, self.pipeline_layouts[idx], None)
            # Clean up point pipeline resources
            if idx < len(self.point_pipelines) and self.point_pipelines[idx]:
                vk.vkDestroyPipeline(dev.device, self.point_pipelines[idx], None)
            if (
                idx < len(self.point_pipeline_layouts)
                and self.point_pipeline_layouts[idx]
            ):
                vk.vkDestroyPipelineLayout(
                    dev.device, self.point_pipeline_layouts[idx], None
                )
            if (
                idx < len(self.point_descriptor_set_layouts)
                and self.point_descriptor_set_layouts[idx]
            ):
                vk.vkDestroyDescriptorSetLayout(
                    dev.device, self.point_descriptor_set_layouts[idx], None
                )
            if (
                idx < len(self.descriptor_set_layouts)
                and self.descriptor_set_layouts[idx]
            ):
                vk.vkDestroyDescriptorSetLayout(
                    dev.device, self.descriptor_set_layouts[idx], None
                )
        self.device_manager.cleanup()

    def render_points(
        self,
        vertex_buffer: Any,
        model_matrix: Optional[Matrix4x4] = None,
        view_matrix: Optional[Matrix4x4] = None,
        projection_matrix: Optional[Matrix4x4] = None,
        point_size: int = 2,
        color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    ) -> np.ndarray:
        """Render points using CPU rasterization."""
        if model_matrix is None:
            model_matrix = Matrix4x4.identity()
        if view_matrix is None:
            view_matrix = Matrix4x4.identity()
        if projection_matrix is None:
            projection_matrix = Matrix4x4.identity()

        if not self.render_target:
            self.set_target(RenderTarget(800, 600))

        # Use the same CPU rendering logic as VulkanRenderer
        return self._render_points_cpu(
            vertex_buffer, model_matrix, view_matrix, projection_matrix, color
        )


# ─────────────────────────────────────────────────────────────────────────────
# CPU-only renderer (stub)
# ─────────────────────────────────────────────────────────────────────────────
class CPURenderer(Renderer):
    """Simple software renderer placeholder."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.render_target: Optional[RenderTarget] = None

    def set_target(self, target: RenderTarget) -> None:
        self.render_target = target

    def render_indexed(
        self,
        vertex_buffer: Any,
        index_buffer: Any,
        model_matrix: Optional[Matrix4x4] = None,
        view_matrix: Optional[Matrix4x4] = None,
        projection_matrix: Optional[Matrix4x4] = None,
        wireframe: bool = False,
    ) -> np.ndarray:
        if model_matrix is None:
            model_matrix = Matrix4x4.identity()
        if view_matrix is None:
            view_matrix = Matrix4x4.identity()
        if projection_matrix is None:
            projection_matrix = Matrix4x4.identity()

        if self.render_target is None:
            self.set_target(RenderTarget(self.width, self.height))

        # Handle MultiBuffer case
        if hasattr(vertex_buffer, "get_vertex_buffer"):
            entry = vertex_buffer.get_vertex_buffer("vertices")
            if entry:
                _, vertex_buffer = entry
            # Also check for index buffer in MultiBuffer
            idx_buf = (
                vertex_buffer.get_index_buffer()
                if hasattr(vertex_buffer, "get_index_buffer")
                else None
            )
            if idx_buf is not None and not isinstance(
                index_buffer, (NumpyBuffer, np.ndarray)
            ):
                index_buffer = idx_buf

        # -- resolve vertex data for CPU raster --
        if hasattr(vertex_buffer, "_array"):
            verts_arr = vertex_buffer._array
        elif hasattr(vertex_buffer, "host_view"):
            verts_arr = vertex_buffer.host_view
        elif hasattr(vertex_buffer, "array"):
            verts_arr = vertex_buffer.array
        elif isinstance(vertex_buffer, np.ndarray):
            verts_arr = vertex_buffer
        else:
            raise TypeError(
                "Unsupported vertex_buffer type; expected NumpyBuffer, "
                "MultiBuffer, NumpyBufferCtx, or ndarray."
            )

        verts_arr = np.asarray(verts_arr, dtype=np.float32, order="C")
        vertices = verts_arr

        if isinstance(index_buffer, NumpyBuffer):
            idx_arr = index_buffer._array
            index_count = len(idx_arr)
        elif hasattr(index_buffer, "host_view"):
            idx_arr = index_buffer.host_view
            index_count = len(idx_arr)
        elif isinstance(index_buffer, np.ndarray):
            idx_arr = index_buffer
            index_count = len(idx_arr)
        elif isinstance(index_buffer, int):
            index_count = index_buffer
            idx_arr = np.arange(index_count, dtype=np.int32)
        else:
            raise TypeError("index_buffer must be NumpyBuffer, numpy.ndarray or int")

        indices = np.asarray(idx_arr, dtype=np.int32, order="C")

        verts = np.hstack([vertices, np.ones((len(vertices), 1), dtype=np.float32)])
        world = verts @ model_matrix.data.T
        mesh = Mesh(
            vertices=world[:, :3],
            normals=np.zeros_like(world[:, :3]),
            uvs=np.zeros((len(vertices), 2), dtype=np.float32),
            indices=indices,
        )
        material = Material()
        return self.render([mesh], [material], [], view_matrix, projection_matrix)

    def render(
        self,
        meshes: List[Mesh],
        materials: List[Material],
        lights: List[Light],
        view_matrix: Matrix4x4,
        projection_matrix: Matrix4x4,
    ) -> np.ndarray:
        if not self.render_target:
            raise RuntimeError("Render target not set")
        w, h = self.render_target.width, self.render_target.height
        fb = np.zeros((h, w, 4), dtype=np.float32)
        depth_buffer = np.full((h, w), np.inf, dtype=np.float32)

        mvp = (projection_matrix @ view_matrix).data

        for mesh_idx, mesh in enumerate(meshes):
            if mesh_idx >= len(materials):
                continue

            mat = materials[mesh_idx]
            base_rgb, alpha = _extract_base_color(mat)
            base_rgb[:] = 0.5

            verts = np.hstack(
                [mesh.vertices, np.ones((len(mesh.vertices), 1), dtype=np.float32)]
            )
            clip = verts @ mvp.T
            w_vals = np.where(np.abs(clip[:, 3:4]) < 1e-6, 1e-6, clip[:, 3:4])
            ndc = clip[:, :3] / w_vals

            screen_x = (ndc[:, 0] + 1) * w / 2
            screen_y = (1 - ndc[:, 1]) * h / 2

            indices = np.ascontiguousarray(mesh.indices, dtype=np.int32)
            for tri in indices.reshape(-1, 3):
                i0, i1, i2 = map(int, tri)
                if i0 >= len(screen_x) or i1 >= len(screen_x) or i2 >= len(screen_x):
                    continue
                tri_ndc = ndc[[i0, i1, i2]]
                if not np.any(np.all(np.abs(tri_ndc) <= 1, axis=1)):
                    continue

                x0, y0, z0 = screen_x[i0], screen_y[i0], ndc[i0, 2]
                x1, y1, z1 = screen_x[i1], screen_y[i1], ndc[i1, 2]
                x2, y2, z2 = screen_x[i2], screen_y[i2], ndc[i2, 2]

                min_x = max(0, int(np.floor(min(x0, x1, x2))))
                max_x = min(w - 1, int(np.ceil(max(x0, x1, x2))))
                min_y = max(0, int(np.floor(min(y0, y1, y2))))
                max_y = min(h - 1, int(np.ceil(max(y0, y1, y2))))
                if min_x > max_x or min_y > max_y:
                    continue

                area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
                if abs(area) < 1e-6:
                    continue

                xs = np.arange(min_x, max_x + 1)
                ys = np.arange(min_y, max_y + 1)
                xi, yi = np.meshgrid(xs, ys)

                w0 = ((x1 - xi) * (y2 - yi) - (y1 - yi) * (x2 - xi)) / area
                w1 = ((x2 - xi) * (y0 - yi) - (y2 - yi) * (x0 - xi)) / area
                w2 = 1.0 - w0 - w1
                mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
                if not np.any(mask):
                    continue

                z = w0 * z0 + w1 * z1 + w2 * z2
                depth = depth_buffer[min_y : max_y + 1, min_x : max_x + 1]
                depth_mask = mask & (z < depth)
                if not np.any(depth_mask):
                    continue
                depth[depth_mask] = z[depth_mask]

                if len(mesh.normals):
                    n0 = mesh.normals[i0]
                    n1 = mesh.normals[i1]
                    n2 = mesh.normals[i2]
                    normal = (
                        n0 * w0[..., None] + n1 * w1[..., None] + n2 * w2[..., None]
                    )
                    nlen = np.linalg.norm(normal, axis=2, keepdims=True)
                    normal = np.divide(
                        normal, nlen, out=np.zeros_like(normal), where=nlen > 1e-10
                    )
                else:
                    normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    normal = np.broadcast_to(normal, (*w0.shape, 3))

                if not lights:
                    color = np.broadcast_to(base_rgb, (*w0.shape, 3)).astype(np.float32)
                else:
                    color = base_rgb * 0.3
                    color = np.broadcast_to(color, (*w0.shape, 3)).astype(np.float32)
                    for light in lights:
                        ldir = np.array(light.position, dtype=np.float32)
                        lnorm = np.linalg.norm(ldir)
                        if lnorm > 1e-10:
                            ldir = ldir / lnorm
                            diffuse = np.maximum(0.0, np.sum(normal * ldir, axis=2))
                            color += base_rgb * diffuse[..., None] * 0.7

                fb_slice = fb[min_y : max_y + 1, min_x : max_x + 1]
                color = np.clip(color, 0, 1)
                fb_slice[depth_mask, :3] = color[depth_mask]
                fb_slice[depth_mask, 3] = alpha

        return (fb * 255).astype(np.uint8)

    def cleanup(self) -> None:
        pass


# Backwards-compat alias (set after class fully defined)
VulkanRenderer._render_points_cpu_alias = VulkanRenderer._render_points_cpu


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
def create_renderer(
    prefer_gpu: bool = True, enable_validation: bool = True
) -> Renderer:
    if prefer_gpu and VULKAN_AVAILABLE:
        dm = DeviceManager(enable_validation=enable_validation)
        devices = dm.create_logical_devices()
        if devices:
            return VulkanRenderer(dm, devices)
    return CPURenderer()


def save_image(image: np.ndarray, filename: str) -> None:
    """Save an RGBA image array to disk."""
    try:
        from PIL import Image

        Image.fromarray(image).save(filename)
    except Exception as e:
        logger.error(f"Failed to save image to {filename}: {e}")


def set_vertex_buffer(renderer: Renderer, numpy_buf, binding: int = 0) -> None:
    """Convenience wrapper for :meth:`Renderer.set_vertex_buffer`."""
    renderer.set_vertex_buffer(numpy_buf, binding)


def set_render_target(renderer: Renderer, target: RenderTarget) -> None:
    """Convenience wrapper for :meth:`Renderer.set_render_target`."""
    renderer.set_render_target(target)

@dataclass
class TerrainBounds:
    """Terrain boundary information"""
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_elevation: float
    max_elevation: float


class TerrainRenderer:
    """GPU-accelerated terrain renderer using Vulkan"""
    
    def __init__(self, config: Optional[TerrainConfig] = None):
        """
        Initialize terrain renderer.
        
        Args:
            config: Terrain configuration (uses default if None)
        """
        self.config = config or TerrainConfig()
        self.is_initialized = False
        
        # Renderer and context
        self._renderer = None
        self._context = None
        
        # Terrain data
        self.heightmap = None
        self.bounds = None
        self.tiles = []
        
        # GPU resources
        self._vertex_buffers = []
        self._index_buffers = []
        self._uniform_buffers = []
        
        # Camera state
        self.camera_position = np.array([0.0, -5.0, 2.0], dtype=np.float32)
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
    def initialize(self, width: int = 1920, height: int = 1080) -> bool:
        """
        Initialize the renderer with a window.
        
        Args:
            width: Window width
            height: Window height
            
        Returns:
            True if initialization successful
        """
        try:
            # Create Vulkan context
            self._context = VulkanContext()
            self._context.initialize()
            
            # Create renderer
            from ..core import create_renderer_auto
            self._renderer = create_renderer_auto(width, height)
            
            self.is_initialized = True
            logger.info("TerrainRenderer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TerrainRenderer: {e}")
            return False
    
    def load_heightmap(self, heightmap: np.ndarray, 
                      bounds: Optional[TerrainBounds] = None) -> bool:
        """
        Load heightmap data for rendering.
        
        Args:
            heightmap: 2D array of height values
            bounds: Geographic bounds (optional)
            
        Returns:
            True if successful
        """
        if not self.is_initialized:
            logger.error("Renderer not initialized")
            return False
            
        try:
            self.heightmap = heightmap.astype(np.float32)
            height, width = heightmap.shape
            
            # Set bounds if not provided
            if bounds is None:
                self.bounds = TerrainBounds(
                    min_x=0.0, max_x=float(width),
                    min_y=0.0, max_y=float(height),
                    min_elevation=float(np.min(heightmap)),
                    max_elevation=float(np.max(heightmap))
                )
            else:
                self.bounds = bounds
            
            # Generate terrain mesh
            self._generate_terrain_mesh()
            
            logger.info(f"Loaded heightmap: {width}x{height}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load heightmap: {e}")
            return False
    
    def _generate_terrain_mesh(self):
        """Generate 3D mesh from heightmap"""
        if self.heightmap is None:
            return
            
        height, width = self.heightmap.shape
        
        # Generate vertices
        vertices = []
        for y in range(height):
            for x in range(width):
                # Position
                px = (x / (width - 1)) * (self.bounds.max_x - self.bounds.min_x) + self.bounds.min_x
                py = (y / (height - 1)) * (self.bounds.max_y - self.bounds.min_y) + self.bounds.min_y
                pz = self.heightmap[y, x] * self.config.height_scale
                
                # Normal (will be calculated properly later)
                nx, ny, nz = 0.0, 0.0, 1.0
                
                # Texture coordinates
                u = x / (width - 1)
                v = y / (height - 1)
                
                vertices.extend([px, py, pz, nx, ny, nz, u, v])
        
        # Generate indices
        indices = []
        for y in range(height - 1):
            for x in range(width - 1):
                # Vertex indices for quad
                v0 = y * width + x
                v1 = v0 + 1
                v2 = (y + 1) * width + x
                v3 = v2 + 1
                
                # Two triangles
                indices.extend([v0, v2, v1])
                indices.extend([v1, v2, v3])
        
        # Convert to numpy arrays
        self.vertices = np.array(vertices, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.uint32)
        
        # Update GPU buffers
        if self._renderer:
            self._upload_to_gpu()
    
    def _upload_to_gpu(self):
        """Upload mesh data to GPU"""
        # This would use the actual Vulkan backend
        # For now, it's a placeholder
        pass
    
    def render(self, view_matrix: Optional[np.ndarray] = None,
              proj_matrix: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Render the terrain.
        
        Args:
            view_matrix: View matrix (4x4)
            proj_matrix: Projection matrix (4x4)
            
        Returns:
            Rendered image as numpy array, or None if failed
        """
        if not self.is_initialized or self.heightmap is None:
            return None
            
        try:
            # Use default matrices if not provided
            if view_matrix is None:
                view_matrix = self._create_view_matrix()
            if proj_matrix is None:
                proj_matrix = self._create_projection_matrix()
            
            # Render using the backend
            if self._renderer:
                # This would call the actual rendering
                pass
                
            return None  # Placeholder
            
        except Exception as e:
            logger.error(f"Render failed: {e}")
            return None
    
    def update_camera(self, position: np.ndarray, target: Optional[np.ndarray] = None):
        """Update camera position and target"""
        self.camera_position = position.copy()
        if target is not None:
            self.camera_target = target.copy()
    
    def _create_view_matrix(self) -> np.ndarray:
        """Create view matrix from camera state"""
        eye = self.camera_position
        target = self.camera_target
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
        # Calculate view matrix
        f = target - eye
        f = f / np.linalg.norm(f)
        
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)
        
        u = np.cross(s, f)
        
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = s
        view[1, :3] = u
        view[2, :3] = -f
        view[3, 0] = -np.dot(s, eye)
        view[3, 1] = -np.dot(u, eye)
        view[3, 2] = np.dot(f, eye)
        
        return view.T
    
    def _create_projection_matrix(self, fov: float = 45.0, 
                                 near: float = 0.1, 
                                 far: float = 1000.0) -> np.ndarray:
        """Create perspective projection matrix"""
        aspect = 16.0 / 9.0  # Default aspect ratio
        fov_rad = np.radians(fov)
        
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = 1.0 / (aspect * np.tan(fov_rad / 2))
        proj[1, 1] = 1.0 / np.tan(fov_rad / 2)
        proj[2, 2] = far / (far - near)
        proj[2, 3] = 1.0
        proj[3, 2] = -(far * near) / (far - near)
        
        return proj
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get rendering performance statistics"""
        return {
            'triangles_rendered': len(self.indices) // 3 if hasattr(self, 'indices') else 0,
            'tiles_rendered': len(self.tiles),
            'frame_time_ms': 0.0,  # Would be tracked by renderer
            'fps': 0.0,
            'gpu_memory_mb': 0.0,
        }
    
    def cleanup(self):
        """Clean up renderer resources"""
        try:
            self._vertex_buffers.clear()
            self._index_buffers.clear()
            self._uniform_buffers.clear()
            
            if self._renderer:
                self._renderer.cleanup()
            
            if self._context:
                self._context.cleanup()
                
            self.is_initialized = False
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Additional classes for complete terrain system

class TerrainStreamer:
    """Terrain data streaming system"""
    
    def __init__(self, cache_size_mb: int = 512):
        """
        Initialize terrain streamer.
        
        Args:
            cache_size_mb: Size of terrain cache in MB
        """
        self.cache_size_mb = cache_size_mb
        self.loaded_tiles = {}
        self.streaming_queue = []
    
    def load_tile(self, tile_x: int, tile_y: int, lod_level: int = 0) -> Optional[np.ndarray]:
        """Load a terrain tile"""
        tile_key = f"{tile_x}_{tile_y}_{lod_level}"
        
        if tile_key in self.loaded_tiles:
            return self.loaded_tiles[tile_key]
        
        # Generate or load tile data
        # This is a placeholder
        return None
    
    def update_streaming(self, camera_position: Tuple[float, float, float],
                        view_distance: float = 1000.0):
        """Update terrain streaming based on camera position"""
        pass


class TerrainLODManager:
    """Level of detail management for terrain"""
    
    def __init__(self, config: Optional[TerrainConfig] = None):
        """Initialize LOD manager"""
        self.config = config or TerrainConfig()
        self.active_lod_levels = {}
    
    def calculate_lod(self, position: Tuple[float, float, float], 
                     camera_position: Tuple[float, float, float]) -> int:
        """Calculate appropriate LOD level for a position"""
        distance = np.linalg.norm(np.array(position) - np.array(camera_position))
        
        # Simple distance-based LOD
        for i, threshold in enumerate(self.config.lod.distances):
            if distance < threshold:
                return i
        
        return len(self.config.lod.distances)


__all__ = [
    "RenderTarget",
    "Mesh",
    "Material",
    "Light",
    "Transform",
    "Renderer",
    "VulkanRenderer",
    "CPURenderer",
    "create_renderer",
    "set_vertex_buffer",
    "set_render_target",
    "save_image",
]
