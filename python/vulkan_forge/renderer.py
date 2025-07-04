# vulkan_forge/renderer.py
"""Main Vulkan renderer with automatic GPU/CPU backend selection."""

import logging
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import ctypes
import numpy as np
import os
import subprocess

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
            impl.set_render_target(RenderTarget(width, height))
            impl.width = width
            impl.height = height
            return impl
        return super().__new__(cls)

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self.width = width
        self.height = height

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
    def set_render_target(self, target: RenderTarget) -> None: ...

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

        self.swapchain_format = (
            vk.VK_FORMAT_B8G8R8A8_UNORM if VULKAN_AVAILABLE else None
        )
        self.current_device_index = 0
        self.gpu_active = False

        self.pipelines: List[Any] = []
        self.render_passes: List[Any] = []
        self._compiled_shaders: Dict[str, bytes] = {}
        self.pipeline_layouts: List[Any] = []
        self.descriptor_set_layouts: List[Any] = []

        if VULKAN_AVAILABLE and self.logical_devices:
            try:
                for dev in self.logical_devices:
                    rp = self._create_render_pass(dev)
                    pl = self._create_pipeline(dev, rp)
                    self.render_passes.append(rp)
                    self.pipelines.append(pl)
                    # Add layouts if created inside pipeline creation
                self.gpu_active = any(self.pipelines)
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

    # ─────────────────────────────────────────────────────────────────────
    # Shader compilation utilities
    # ─────────────────────────────────────────────────────────────────────
    def _compile_shader(self, shader_name: str, stage: str) -> bytes:
        """Compile GLSL → SPIR-V or return a cached blob."""
        if shader_name in self._compiled_shaders:
            return self._compiled_shaders[shader_name]

        shader_dir = os.path.join(os.path.dirname(__file__), "shaders")
        glsl = os.path.join(shader_dir, f"{shader_name}.glsl")
        spv = os.path.join(shader_dir, f"{shader_name}.spv")

        # Use cached SPIR-V if it’s newer than GLSL source
        if os.path.exists(spv) and os.path.getmtime(spv) >= os.path.getmtime(glsl):
            with open(spv, "rb") as fh:
                blob = fh.read()
                self._compiled_shaders[shader_name] = blob
                return blob

        # Try to invoke glslc
        try:
            subprocess.run(
                ["glslc", f"-fshader-stage={stage}", glsl, "-o", spv],
                check=True,
                capture_output=True,
            )
            with open(spv, "rb") as fh:
                blob = fh.read()
                self._compiled_shaders[shader_name] = blob
                return blob
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.warning(
                "glslc unavailable – using embedded placeholder for %s", shader_name
            )
            blob = self._get_embedded_spirv(shader_name)
            self._compiled_shaders[shader_name] = blob
            return blob

    def _get_embedded_spirv(self, shader_name: str) -> bytes:
        """Tiny dummy SPIR-V blobs for fallback (minimal header only)."""
        # We don't have valid embedded SPIR-V, so return empty bytes
        # This will signal that shader compilation is not available
        return b""

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
            vert_code = self._compile_shader("vertex", "vertex")
            frag_code = self._compile_shader("fragment", "fragment")

            # Check if shader compilation succeeded
            if not vert_code or not frag_code:
                logger.warning("Shader compilation failed - skipping pipeline creation")
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

    def set_render_target(self, target: RenderTarget) -> None:
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
        self, first_vertex: int = 0, vertex_count: Optional[int] = None
    ) -> np.ndarray:
        """Render bound vertex buffers as a point cloud."""
        if not self.render_target:
            raise VulkanForgeError("Render target not set")

        if vertex_count is None:
            buf = self.vertex_buffers.get(0)
            if isinstance(buf, np.ndarray):
                vertex_count = len(buf) - first_vertex
            else:
                vertex_count = 0

        if self.gpu_active and hasattr(vk, "vkCmdDraw"):
            try:
                cmd_buf = getattr(self, "command_buffer", None)
                if cmd_buf is not None:
                    buffers = (ctypes.c_uint64 * 1)(
                        ctypes.c_uint64(self.vertex_buffers.get(0, 0))
                    )
                    offsets = (ctypes.c_ulonglong * 1)(0)
                    vk.vkCmdBindVertexBuffers(cmd_buf, 0, 1, buffers, offsets)
                    vk.vkCmdDraw(cmd_buf, vertex_count, 1, first_vertex, 0)
            except Exception as e:
                logger.error("GPU draw failed: %s", e)

        w, h = self.render_target.width, self.render_target.height
        fb = np.zeros((h, w, 4), dtype=np.uint8)
        positions = self.vertex_buffers.get(0)
        colors = self.vertex_buffers.get(1)
        if isinstance(positions, np.ndarray):
            pts = positions[first_vertex : first_vertex + vertex_count]
            if not isinstance(colors, np.ndarray):
                colors = np.ones((len(pts), 4), dtype=np.float32)
            cols = (
                colors[first_vertex : first_vertex + len(pts)]
                if isinstance(colors, np.ndarray)
                else None
            )
            xs = ((pts[:, 0] + 1) * w / 2).astype(int)
            ys = ((1 - pts[:, 1]) * h / 2).astype(int)
            for i, (x, y) in enumerate(zip(xs, ys)):
                if 0 <= x < w and 0 <= y < h:
                    c = cols[i] if cols is not None else np.array([1, 1, 1, 1])
                    fb[y, x, :3] = np.clip(c[:3] * 255, 0, 255)
                    fb[y, x, 3] = int(c[3] * 255)
        self._framebuffer = fb
        return fb

    def render_indexed(
        self,
        vertex_buffer: Any,
        index_buffer: Any,
        model_matrix: Optional[Matrix4x4] = None,
        view_matrix: Optional[Matrix4x4] = None,
        projection_matrix: Optional[Matrix4x4] = None,
        wireframe: bool = False,
    ) -> np.ndarray:
        """Render indexed geometry from buffers.

        Parameters
        ----------
        vertex_buffer : Any
            Vertex data as ``NumpyBuffer`` or ``np.ndarray``.
        index_buffer : NumpyBuffer | np.ndarray | int
            Index data or raw element count when already bound on GPU.
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

        vertices = np.asarray(
            getattr(vertex_buffer, "_array", vertex_buffer), dtype=np.float32
        )
        if isinstance(index_buffer, int):
            indices = np.arange(index_count, dtype=np.int32)
        else:
            indices = np.asarray(
                getattr(index_buffer, "_array", index_buffer), dtype=np.int32
            )
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
        fb = np.zeros((h, w, 4), dtype=np.float32)
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
            if (
                idx < len(self.descriptor_set_layouts)
                and self.descriptor_set_layouts[idx]
            ):
                vk.vkDestroyDescriptorSetLayout(
                    dev.device, self.descriptor_set_layouts[idx], None
                )
        self.device_manager.cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# CPU-only renderer (stub)
# ─────────────────────────────────────────────────────────────────────────────
class CPURenderer(Renderer):
    """Simple software renderer placeholder."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.render_target: Optional[RenderTarget] = None

    def set_render_target(self, target: RenderTarget) -> None:
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
            self.set_render_target(RenderTarget(self.width, self.height))

        if hasattr(vertex_buffer, "get_vertex_buffer"):
            entry = vertex_buffer.get_vertex_buffer("vertices")
            if entry:
                vertex_buffer = entry[1]
            buf = (
                vertex_buffer.get_index_buffer()
                if hasattr(vertex_buffer, "get_index_buffer")
                else None
            )
            if buf is not None and not isinstance(
                index_buffer, (NumpyBuffer, np.ndarray)
            ):
                index_buffer = buf

        vertices = np.asarray(
            getattr(vertex_buffer, "_array", vertex_buffer), dtype=np.float32
        )
        if isinstance(index_buffer, int):
            index_count = index_buffer
            indices = np.arange(index_count, dtype=np.int32)
        else:
            index_count = len(getattr(index_buffer, "_array", index_buffer))
            indices = np.asarray(
                getattr(index_buffer, "_array", index_buffer), dtype=np.int32
            )

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
    "render_indexed",
    "set_vertex_buffer",
    "save_image",
]
