"""Vulkan renderer with CPU fallback."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, Optional

from numpy.typing import NDArray

import numpy as np

from . import backend
from .matrices import Matrix4x4
from .backend import vk


class VulkanRenderer:
    """Renderer that uses Vulkan if available, otherwise CPU."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.device_info = backend.select_device()
        self.device = self.device_info.handle
        self.is_cpu = self.device is None or self.device_info.device_type == vk.VK_PHYSICAL_DEVICE_TYPE_CPU
        if not self.is_cpu:
            self.render_pass = self._create_render_pass()
            self.pipeline_layout, self.pipeline = self._create_pipeline()

    # Vulkan helpers -----------------------------------------------------
    def _create_render_pass(self) -> Any:
        color_attachment = vk.VkAttachmentDescription2(
            sType=vk.VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
            format=vk.VK_FORMAT_B8G8R8A8_UNORM,
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
            attachment=0,
            layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        )
        depth_ref = vk.VkAttachmentReference2(
            sType=vk.VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,
            attachment=1,
            layout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        )
        subpass = vk.VkSubpassDescription2(
            sType=vk.VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2,
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=[color_ref],
            pDepthStencilAttachment=depth_ref,
        )
        info = vk.VkRenderPassCreateInfo2(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2,
            attachmentCount=2,
            pAttachments=[color_attachment, depth_attachment],
            subpassCount=1,
            pSubpasses=[subpass],
        )
        render_pass = vk.VkRenderPass(0)
        result = vk.vkCreateRenderPass2(self.device, info, None, render_pass)
        if result != vk.VK_SUCCESS:
            raise RuntimeError("vkCreateRenderPass2 failed")
        return render_pass

    def _load_shader(self, name: str) -> bytes:
        path = Path(__file__).resolve().parent / "shaders" / name
        return path.read_bytes()

    def _create_pipeline(self) -> tuple[Any, Any]:
        vert_data = self._load_shader("vertex.glsl")
        frag_data = self._load_shader("fragment.glsl")

        vert_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(vert_data),
            pCode=vert_data,
        )
        frag_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(frag_data),
            pCode=frag_data,
        )
        vert_module = vk.vkCreateShaderModule(self.device, vert_info, None)
        frag_module = vk.vkCreateShaderModule(self.device, frag_info, None)

        layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        )
        layout = vk.vkCreatePipelineLayout(self.device, layout_info, None)

        pipeline_info = vk.VkGraphicsPipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            stageCount=2,
            pStages=None,
            layout=layout,
            renderPass=self.render_pass,
        )
        pipeline = vk.vkCreateGraphicsPipelines(self.device, None, 1, pipeline_info, None)
        return layout, pipeline

    # Public API ---------------------------------------------------------
    def render(
        self,
        meshes: Optional[list[Any]] = None,
        view: Optional[Matrix4x4] = None,
        proj: Optional[Matrix4x4] = None,
    ) -> NDArray[np.uint8]:
        del meshes, view, proj  # Unused for stub
        if self.is_cpu:
            buf: NDArray[np.uint8] = np.full((self.height, self.width, 4), 128, dtype=np.uint8)
            buf[:, :, 3] = 255
            return buf

        # Command buffer recording would go here
        return np.zeros((self.height, self.width, 4), dtype=np.uint8)
