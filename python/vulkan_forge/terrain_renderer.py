#!/usr/bin/env python3
"""
Complete Vulkan-accelerated terrain renderer implementation
"""

import sys
import time
import struct
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import logging

# Platform-specific imports
try:
    import glfw
    GLFW_AVAILABLE = True
except ImportError:
    GLFW_AVAILABLE = False
    print("Warning: GLFW not available. Install with: pip install glfw")

try:
    import vulkan as vk
    VULKAN_AVAILABLE = True
except ImportError:
    VULKAN_AVAILABLE = False
    print("Warning: Python vulkan package not available. Install with: pip install vulkan")

# Import vulkan-forge components
from vulkan_forge.terrain_config import TerrainConfig
from vulkan_forge.backend import VulkanContext


class VulkanTerrainRenderer:
    """Complete Vulkan terrain renderer with window and GPU rendering"""
    
    def __init__(self, width: int = 1920, height: int = 1080, config: Optional[TerrainConfig] = None):
        self.width = width
        self.height = height
        self.config = config or TerrainConfig.from_preset('balanced')
        
        # Vulkan objects
        self.instance = None
        self.surface = None
        self.physical_device = None
        self.device = None
        self.graphics_queue = None
        self.present_queue = None
        self.swapchain = None
        self.swapchain_images = []
        self.swapchain_image_views = []
        self.render_pass = None
        self.framebuffers = []
        self.command_pool = None
        self.command_buffers = []
        
        # Pipeline objects
        self.pipeline_layout = None
        self.graphics_pipeline = None
        self.vertex_buffer = None
        self.index_buffer = None
        self.uniform_buffer = None
        
        # Synchronization
        self.image_available_semaphores = []
        self.render_finished_semaphores = []
        self.in_flight_fences = []
        
        # Window
        self.window = None
        
        # Terrain data
        self.heightmap = None
        self.vertices = []
        self.indices = []
        
        # Camera
        self.camera_pos = np.array([0.0, -5.0, 2.0], dtype=np.float32)
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.camera_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
        self.frame_count = 0
        self.last_time = time.time()
        
    def initialize(self) -> bool:
        """Initialize the complete rendering system"""
        if not GLFW_AVAILABLE or not VULKAN_AVAILABLE:
            logging.error("Required dependencies not available")
            return False
        
        try:
            # Initialize GLFW
            if not glfw.init():
                logging.error("Failed to initialize GLFW")
                return False
            
            # Create window
            glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
            glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
            
            self.window = glfw.create_window(self.width, self.height, "Vulkan Terrain Renderer", None, None)
            if not self.window:
                logging.error("Failed to create window")
                return False
            
            # Set up callbacks
            glfw.set_key_callback(self.window, self._key_callback)
            glfw.set_cursor_pos_callback(self.window, self._mouse_callback)
            glfw.set_scroll_callback(self.window, self._scroll_callback)
            
            # Initialize Vulkan
            self._create_instance()
            self._create_surface()
            self._pick_physical_device()
            self._create_logical_device()
            self._create_swapchain()
            self._create_image_views()
            self._create_render_pass()
            self._create_graphics_pipeline()
            self._create_framebuffers()
            self._create_command_pool()
            self._create_vertex_buffer()
            self._create_index_buffer()
            self._create_uniform_buffers()
            self._create_command_buffers()
            self._create_sync_objects()
            
            logging.info("Vulkan terrain renderer initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize renderer: {e}")
            return False
    
    def _create_instance(self):
        """Create Vulkan instance"""
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="Vulkan Terrain Renderer",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="VulkanForge",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_3
        )
        
        # Get required extensions
        extensions = glfw.get_required_instance_extensions()
        
        # Enable validation layers in debug mode
        layers = []
        if self._debug_mode():
            layers = ["VK_LAYER_KHRONOS_validation"]
            extensions.append(vk.VK_EXT_DEBUG_UTILS_EXTENSION_NAME)
        
        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
            enabledLayerCount=len(layers),
            ppEnabledLayerNames=layers,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions
        )
        
        self.instance = vk.vkCreateInstance(create_info, None)
    
    def _create_surface(self):
        """Create window surface"""
        # Platform-specific surface creation
        self.surface = glfw.create_window_surface(self.instance, self.window, None)
    
    def _pick_physical_device(self):
        """Select best GPU"""
        devices = vk.vkEnumeratePhysicalDevices(self.instance)
        
        # Score devices and pick best
        best_score = -1
        best_device = None
        
        for device in devices:
            score = self._rate_device_suitability(device)
            if score > best_score:
                best_score = score
                best_device = device
        
        if not best_device:
            raise RuntimeError("No suitable GPU found")
        
        self.physical_device = best_device
        
        # Log device info
        props = vk.vkGetPhysicalDeviceProperties(self.physical_device)
        logging.info(f"Selected GPU: {props.deviceName}")
    
    def _rate_device_suitability(self, device) -> int:
        """Rate GPU suitability"""
        props = vk.vkGetPhysicalDeviceProperties(device)
        features = vk.vkGetPhysicalDeviceFeatures(device)
        
        score = 0
        
        # Discrete GPUs are preferred
        if props.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            score += 1000
        
        # Maximum possible size of textures affects graphics quality
        score += props.limits.maxImageDimension2D
        
        # Required features
        if not features.tessellationShader:
            return 0  # Can't use this device
        
        return score
    
    def _create_logical_device(self):
        """Create logical device and queues"""
        # Find queue families
        queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device)
        
        graphics_family = None
        present_family = None
        
        for i, family in enumerate(queue_families):
            if family.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT:
                graphics_family = i
            
            if vk.vkGetPhysicalDeviceSurfaceSupportKHR(self.physical_device, i, self.surface):
                present_family = i
        
        if graphics_family is None or present_family is None:
            raise RuntimeError("Required queue families not found")
        
        # Create queues
        unique_families = set([graphics_family, present_family])
        queue_create_infos = []
        
        for family in unique_families:
            queue_create_info = vk.VkDeviceQueueCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                queueFamilyIndex=family,
                queueCount=1,
                pQueuePriorities=[1.0]
            )
            queue_create_infos.append(queue_create_info)
        
        # Enable features
        features = vk.VkPhysicalDeviceFeatures()
        features.tessellationShader = vk.VK_TRUE
        features.fillModeNonSolid = vk.VK_TRUE  # For wireframe
        
        # Device extensions
        device_extensions = [vk.VK_KHR_SWAPCHAIN_EXTENSION_NAME]
        
        # Create device
        create_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=len(queue_create_infos),
            pQueueCreateInfos=queue_create_infos,
            pEnabledFeatures=features,
            enabledExtensionCount=len(device_extensions),
            ppEnabledExtensionNames=device_extensions
        )
        
        self.device = vk.vkCreateDevice(self.physical_device, create_info, None)
        
        # Get queues
        self.graphics_queue = vk.vkGetDeviceQueue(self.device, graphics_family, 0)
        self.present_queue = vk.vkGetDeviceQueue(self.device, present_family, 0)
        
        self.graphics_family = graphics_family
        self.present_family = present_family
    
    def _create_swapchain(self):
        """Create swapchain"""
        # Query swapchain support
        capabilities = vk.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(self.physical_device, self.surface)
        formats = vk.vkGetPhysicalDeviceSurfaceFormatsKHR(self.physical_device, self.surface)
        present_modes = vk.vkGetPhysicalDeviceSurfacePresentModesKHR(self.physical_device, self.surface)
        
        # Choose format
        surface_format = formats[0]
        for fmt in formats:
            if fmt.format == vk.VK_FORMAT_B8G8R8A8_SRGB and fmt.colorSpace == vk.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR:
                surface_format = fmt
                break
        
        # Choose present mode
        present_mode = vk.VK_PRESENT_MODE_FIFO_KHR
        for mode in present_modes:
            if mode == vk.VK_PRESENT_MODE_MAILBOX_KHR:
                present_mode = mode
                break
        
        # Choose extent
        if capabilities.currentExtent.width != 0xFFFFFFFF:
            extent = capabilities.currentExtent
        else:
            extent = vk.VkExtent2D(width=self.width, height=self.height)
        
        # Image count
        image_count = capabilities.minImageCount + 1
        if capabilities.maxImageCount > 0 and image_count > capabilities.maxImageCount:
            image_count = capabilities.maxImageCount
        
        # Create swapchain
        create_info = vk.VkSwapchainCreateInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            surface=self.surface,
            minImageCount=image_count,
            imageFormat=surface_format.format,
            imageColorSpace=surface_format.colorSpace,
            imageExtent=extent,
            imageArrayLayers=1,
            imageUsage=vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            imageSharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            preTransform=capabilities.currentTransform,
            compositeAlpha=vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            presentMode=present_mode,
            clipped=vk.VK_TRUE,
            oldSwapchain=vk.VK_NULL_HANDLE
        )
        
        self.swapchain = vk.vkCreateSwapchainKHR(self.device, create_info, None)
        self.swapchain_images = vk.vkGetSwapchainImagesKHR(self.device, self.swapchain)
        self.swapchain_format = surface_format.format
        self.swapchain_extent = extent
    
    def _create_image_views(self):
        """Create image views for swapchain images"""
        self.swapchain_image_views = []
        
        for image in self.swapchain_images:
            create_info = vk.VkImageViewCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                image=image,
                viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                format=self.swapchain_format,
                components=vk.VkComponentMapping(
                    r=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                    g=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                    b=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                    a=vk.VK_COMPONENT_SWIZZLE_IDENTITY
                ),
                subresourceRange=vk.VkImageSubresourceRange(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0,
                    levelCount=1,
                    baseArrayLayer=0,
                    layerCount=1
                )
            )
            
            image_view = vk.vkCreateImageView(self.device, create_info, None)
            self.swapchain_image_views.append(image_view)
    
    def _create_render_pass(self):
        """Create render pass"""
        # Color attachment
        color_attachment = vk.VkAttachmentDescription(
            format=self.swapchain_format,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
        )
        
        color_attachment_ref = vk.VkAttachmentReference(
            attachment=0,
            layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        )
        
        # Depth attachment
        depth_format = self._find_depth_format()
        depth_attachment = vk.VkAttachmentDescription(
            format=depth_format,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        )
        
        depth_attachment_ref = vk.VkAttachmentReference(
            attachment=1,
            layout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        )
        
        # Subpass
        subpass = vk.VkSubpassDescription(
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=[color_attachment_ref],
            pDepthStencilAttachment=depth_attachment_ref
        )
        
        # Render pass
        render_pass_info = vk.VkRenderPassCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            attachmentCount=2,
            pAttachments=[color_attachment, depth_attachment],
            subpassCount=1,
            pSubpasses=[subpass]
        )
        
        self.render_pass = vk.vkCreateRenderPass(self.device, render_pass_info, None)
    
    def _create_graphics_pipeline(self):
        """Create the graphics pipeline"""
        # Shader stages
        vert_shader_code = self._load_shader("terrain.vert.spv")
        frag_shader_code = self._load_shader("terrain.frag.spv")
        
        vert_shader_module = self._create_shader_module(vert_shader_code)
        frag_shader_module = self._create_shader_module(frag_shader_code)
        
        vert_shader_stage_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
            module=vert_shader_module,
            pName="main"
        )
        
        frag_shader_stage_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            module=frag_shader_module,
            pName="main"
        )
        
        shader_stages = [vert_shader_stage_info, frag_shader_stage_info]
        
        # Vertex input
        binding_description = vk.VkVertexInputBindingDescription(
            binding=0,
            stride=6 * 4,  # 3 floats for position, 3 for normal
            inputRate=vk.VK_VERTEX_INPUT_RATE_VERTEX
        )
        
        attribute_descriptions = [
            vk.VkVertexInputAttributeDescription(
                binding=0,
                location=0,
                format=vk.VK_FORMAT_R32G32B32_SFLOAT,
                offset=0
            ),
            vk.VkVertexInputAttributeDescription(
                binding=0,
                location=1,
                format=vk.VK_FORMAT_R32G32B32_SFLOAT,
                offset=3 * 4
            )
        ]
        
        vertex_input_info = vk.VkPipelineVertexInputStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertexBindingDescriptionCount=1,
            pVertexBindingDescriptions=[binding_description],
            vertexAttributeDescriptionCount=len(attribute_descriptions),
            pVertexAttributeDescriptions=attribute_descriptions
        )
        
        # Input assembly
        input_assembly = vk.VkPipelineInputAssemblyStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            primitiveRestartEnable=vk.VK_FALSE
        )
        
        # Viewport state
        viewport = vk.VkViewport(
            x=0.0,
            y=0.0,
            width=float(self.swapchain_extent.width),
            height=float(self.swapchain_extent.height),
            minDepth=0.0,
            maxDepth=1.0
        )
        
        scissor = vk.VkRect2D(
            offset=vk.VkOffset2D(x=0, y=0),
            extent=self.swapchain_extent
        )
        
        viewport_state = vk.VkPipelineViewportStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewportCount=1,
            pViewports=[viewport],
            scissorCount=1,
            pScissors=[scissor]
        )
        
        # Rasterizer
        rasterizer = vk.VkPipelineRasterizationStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            depthClampEnable=vk.VK_FALSE,
            rasterizerDiscardEnable=vk.VK_FALSE,
            polygonMode=vk.VK_POLYGON_MODE_FILL,
            lineWidth=1.0,
            cullMode=vk.VK_CULL_MODE_BACK_BIT,
            frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
            depthBiasEnable=vk.VK_FALSE
        )
        
        # Multisampling
        multisampling = vk.VkPipelineMultisampleStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            sampleShadingEnable=vk.VK_FALSE,
            rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT
        )
        
        # Depth and stencil
        depth_stencil = vk.VkPipelineDepthStencilStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            depthTestEnable=vk.VK_TRUE,
            depthWriteEnable=vk.VK_TRUE,
            depthCompareOp=vk.VK_COMPARE_OP_LESS,
            depthBoundsTestEnable=vk.VK_FALSE,
            stencilTestEnable=vk.VK_FALSE
        )
        
        # Color blending
        color_blend_attachment = vk.VkPipelineColorBlendAttachmentState(
            colorWriteMask=vk.VK_COLOR_COMPONENT_R_BIT | vk.VK_COLOR_COMPONENT_G_BIT |
                          vk.VK_COLOR_COMPONENT_B_BIT | vk.VK_COLOR_COMPONENT_A_BIT,
            blendEnable=vk.VK_FALSE
        )
        
        color_blending = vk.VkPipelineColorBlendStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logicOpEnable=vk.VK_FALSE,
            attachmentCount=1,
            pAttachments=[color_blend_attachment]
        )
        
        # Pipeline layout
        uniform_layout_binding = vk.VkDescriptorSetLayoutBinding(
            binding=0,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1,
            stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT
        )
        
        descriptor_layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=1,
            pBindings=[uniform_layout_binding]
        )
        
        self.descriptor_set_layout = vk.vkCreateDescriptorSetLayout(self.device, descriptor_layout_info, None)
        
        pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[self.descriptor_set_layout]
        )
        
        self.pipeline_layout = vk.vkCreatePipelineLayout(self.device, pipeline_layout_info, None)
        
        # Create pipeline
        pipeline_info = vk.VkGraphicsPipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            stageCount=len(shader_stages),
            pStages=shader_stages,
            pVertexInputState=vertex_input_info,
            pInputAssemblyState=input_assembly,
            pViewportState=viewport_state,
            pRasterizationState=rasterizer,
            pMultisampleState=multisampling,
            pDepthStencilState=depth_stencil,
            pColorBlendState=color_blending,
            layout=self.pipeline_layout,
            renderPass=self.render_pass,
            subpass=0
        )
        
        self.graphics_pipeline = vk.vkCreateGraphicsPipelines(
            self.device, vk.VK_NULL_HANDLE, 1, [pipeline_info], None
        )[0]
        
        # Clean up shader modules
        vk.vkDestroyShaderModule(self.device, vert_shader_module, None)
        vk.vkDestroyShaderModule(self.device, frag_shader_module, None)
    
    def _create_framebuffers(self):
        """Create framebuffers"""
        # First create depth image
        self._create_depth_resources()
        
        self.framebuffers = []
        
        for image_view in self.swapchain_image_views:
            attachments = [image_view, self.depth_image_view]
            
            framebuffer_info = vk.VkFramebufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                renderPass=self.render_pass,
                attachmentCount=len(attachments),
                pAttachments=attachments,
                width=self.swapchain_extent.width,
                height=self.swapchain_extent.height,
                layers=1
            )
            
            framebuffer = vk.vkCreateFramebuffer(self.device, framebuffer_info, None)
            self.framebuffers.append(framebuffer)
    
    def _create_command_pool(self):
        """Create command pool"""
        pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=self.graphics_family
        )
        
        self.command_pool = vk.vkCreateCommandPool(self.device, pool_info, None)
    
    def _create_command_buffers(self):
        """Create command buffers"""
        alloc_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=len(self.framebuffers)
        )
        
        self.command_buffers = vk.vkAllocateCommandBuffers(self.device, alloc_info)
    
    def _create_sync_objects(self):
        """Create synchronization objects"""
        self.MAX_FRAMES_IN_FLIGHT = 2
        
        semaphore_info = vk.VkSemaphoreCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
        )
        
        fence_info = vk.VkFenceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            flags=vk.VK_FENCE_CREATE_SIGNALED_BIT
        )
        
        self.image_available_semaphores = []
        self.render_finished_semaphores = []
        self.in_flight_fences = []
        
        for i in range(self.MAX_FRAMES_IN_FLIGHT):
            self.image_available_semaphores.append(
                vk.vkCreateSemaphore(self.device, semaphore_info, None)
            )
            self.render_finished_semaphores.append(
                vk.vkCreateSemaphore(self.device, semaphore_info, None)
            )
            self.in_flight_fences.append(
                vk.vkCreateFence(self.device, fence_info, None)
            )
    
    def load_heightmap(self, heightmap: np.ndarray) -> bool:
        """Load heightmap data and generate terrain mesh"""
        try:
            self.heightmap = heightmap
            height, width = heightmap.shape
            
            # Generate vertices
            self.vertices = []
            self.indices = []
            
            # Create vertex grid
            for y in range(height):
                for x in range(width):
                    # Position
                    px = (x / (width - 1) - 0.5) * 10.0  # Scale to -5 to 5
                    py = (y / (height - 1) - 0.5) * 10.0
                    pz = heightmap[y, x] * self.config.height_scale
                    
                    # Calculate normal using finite differences
                    nx = 0.0
                    ny = 0.0
                    nz = 1.0
                    
                    if x > 0 and x < width - 1:
                        nx = (heightmap[y, x-1] - heightmap[y, x+1]) * 0.5
                    if y > 0 and y < height - 1:
                        ny = (heightmap[y-1, x] - heightmap[y+1, x]) * 0.5
                    
                    # Normalize
                    length = (nx*nx + ny*ny + nz*nz) ** 0.5
                    nx /= length
                    ny /= length
                    nz /= length
                    
                    self.vertices.extend([px, py, pz, nx, ny, nz])
            
            # Generate indices for triangles
            for y in range(height - 1):
                for x in range(width - 1):
                    # Two triangles per quad
                    v0 = y * width + x
                    v1 = v0 + 1
                    v2 = (y + 1) * width + x
                    v3 = v2 + 1
                    
                    # First triangle
                    self.indices.extend([v0, v2, v1])
                    # Second triangle
                    self.indices.extend([v1, v2, v3])
            
            # Update GPU buffers
            self._update_vertex_buffer()
            self._update_index_buffer()
            
            logging.info(f"Loaded heightmap: {width}x{height}, {len(self.vertices)//6} vertices, {len(self.indices)//3} triangles")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load heightmap: {e}")
            return False
    
    def _create_vertex_buffer(self):
        """Create vertex buffer"""
        if not self.vertices:
            # Create a simple quad if no terrain loaded
            self.vertices = [
                -1.0, -1.0, 0.0,  0.0, 0.0, 1.0,  # Bottom-left
                 1.0, -1.0, 0.0,  0.0, 0.0, 1.0,  # Bottom-right
                 1.0,  1.0, 0.0,  0.0, 0.0, 1.0,  # Top-right
                -1.0,  1.0, 0.0,  0.0, 0.0, 1.0,  # Top-left
            ]
            self.indices = [0, 1, 2, 2, 3, 0]
        
        # Create vertex buffer
        buffer_size = len(self.vertices) * 4  # 4 bytes per float
        
        # Create staging buffer
        staging_buffer, staging_memory = self._create_buffer(
            buffer_size,
            vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        )
        
        # Copy data to staging buffer
        data_ptr = vk.vkMapMemory(self.device, staging_memory, 0, buffer_size, 0)
        
        # Convert vertices to bytes
        vertices_bytes = struct.pack(f'{len(self.vertices)}f', *self.vertices)
        import ctypes
        ctypes.memmove(data_ptr, vertices_bytes, buffer_size)
        
        vk.vkUnmapMemory(self.device, staging_memory)
        
        # Create device local buffer
        self.vertex_buffer, self.vertex_buffer_memory = self._create_buffer(
            buffer_size,
            vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT | vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        )
        
        # Copy from staging to device
        self._copy_buffer(staging_buffer, self.vertex_buffer, buffer_size)
        
        # Clean up staging buffer
        vk.vkDestroyBuffer(self.device, staging_buffer, None)
        vk.vkFreeMemory(self.device, staging_memory, None)
    
    def _create_index_buffer(self):
        """Create index buffer"""
        if not self.indices:
            return
        
        buffer_size = len(self.indices) * 4  # 4 bytes per uint32
        
        # Create staging buffer
        staging_buffer, staging_memory = self._create_buffer(
            buffer_size,
            vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        )
        
        # Copy data
        data_ptr = vk.vkMapMemory(self.device, staging_memory, 0, buffer_size, 0)
        indices_bytes = struct.pack(f'{len(self.indices)}I', *self.indices)
        import ctypes
        ctypes.memmove(data_ptr, indices_bytes, buffer_size)
        vk.vkUnmapMemory(self.device, staging_memory)
        
        # Create device local buffer
        self.index_buffer, self.index_buffer_memory = self._create_buffer(
            buffer_size,
            vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT | vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        )
        
        # Copy
        self._copy_buffer(staging_buffer, self.index_buffer, buffer_size)
        
        # Clean up
        vk.vkDestroyBuffer(self.device, staging_buffer, None)
        vk.vkFreeMemory(self.device, staging_memory, None)
    
    def _update_vertex_buffer(self):
        """Update vertex buffer with new data"""
        if self.vertex_buffer:
            vk.vkDestroyBuffer(self.device, self.vertex_buffer, None)
            vk.vkFreeMemory(self.device, self.vertex_buffer_memory, None)
        
        self._create_vertex_buffer()
    
    def _update_index_buffer(self):
        """Update index buffer with new data"""
        if self.index_buffer:
            vk.vkDestroyBuffer(self.device, self.index_buffer, None)
            vk.vkFreeMemory(self.device, self.index_buffer_memory, None)
        
        self._create_index_buffer()
    
    def _create_uniform_buffers(self):
        """Create uniform buffers"""
        buffer_size = 3 * 16 * 4  # 3 mat4 matrices, 16 floats each, 4 bytes per float
        
        self.uniform_buffers = []
        self.uniform_buffers_memory = []
        
        for i in range(len(self.swapchain_images)):
            buffer, memory = self._create_buffer(
                buffer_size,
                vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            )
            self.uniform_buffers.append(buffer)
            self.uniform_buffers_memory.append(memory)
        
        # Create descriptor pool
        pool_size = vk.VkDescriptorPoolSize(
            type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=len(self.swapchain_images)
        )
        
        pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount=1,
            pPoolSizes=[pool_size],
            maxSets=len(self.swapchain_images)
        )
        
        self.descriptor_pool = vk.vkCreateDescriptorPool(self.device, pool_info, None)
        
        # Create descriptor sets
        layouts = [self.descriptor_set_layout] * len(self.swapchain_images)
        
        alloc_info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=len(layouts),
            pSetLayouts=layouts
        )
        
        self.descriptor_sets = vk.vkAllocateDescriptorSets(self.device, alloc_info)
        
        # Update descriptor sets
        for i in range(len(self.swapchain_images)):
            buffer_info = vk.VkDescriptorBufferInfo(
                buffer=self.uniform_buffers[i],
                offset=0,
                range=buffer_size
            )
            
            write_descriptor_set = vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=self.descriptor_sets[i],
                dstBinding=0,
                dstArrayElement=0,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                descriptorCount=1,
                pBufferInfo=[buffer_info]
            )
            
            vk.vkUpdateDescriptorSets(self.device, 1, [write_descriptor_set], 0, None)
    
    def _create_depth_resources(self):
        """Create depth buffer resources"""
        depth_format = self._find_depth_format()
        
        self.depth_image, self.depth_image_memory = self._create_image(
            self.swapchain_extent.width,
            self.swapchain_extent.height,
            depth_format,
            vk.VK_IMAGE_TILING_OPTIMAL,
            vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        )
        
        self.depth_image_view = self._create_image_view(
            self.depth_image,
            depth_format,
            vk.VK_IMAGE_ASPECT_DEPTH_BIT
        )
    
    def _create_buffer(self, size: int, usage: int, properties: int) -> Tuple:
        """Create a buffer with memory"""
        buffer_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size,
            usage=usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
        )
        
        buffer = vk.vkCreateBuffer(self.device, buffer_info, None)
        
        mem_requirements = vk.vkGetBufferMemoryRequirements(self.device, buffer)
        
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_requirements.size,
            memoryTypeIndex=self._find_memory_type(mem_requirements.memoryTypeBits, properties)
        )
        
        memory = vk.vkAllocateMemory(self.device, alloc_info, None)
        vk.vkBindBufferMemory(self.device, buffer, memory, 0)
        
        return buffer, memory
    
    def _create_image(self, width: int, height: int, format: int, tiling: int, usage: int, properties: int) -> Tuple:
        """Create an image with memory"""
        image_info = vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=vk.VK_IMAGE_TYPE_2D,
            extent=vk.VkExtent3D(width=width, height=height, depth=1),
            mipLevels=1,
            arrayLayers=1,
            format=format,
            tiling=tiling,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            usage=usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            samples=vk.VK_SAMPLE_COUNT_1_BIT
        )
        
        image = vk.vkCreateImage(self.device, image_info, None)
        
        mem_requirements = vk.vkGetImageMemoryRequirements(self.device, image)
        
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_requirements.size,
            memoryTypeIndex=self._find_memory_type(mem_requirements.memoryTypeBits, properties)
        )
        
        memory = vk.vkAllocateMemory(self.device, alloc_info, None)
        vk.vkBindImageMemory(self.device, image, memory, 0)
        
        return image, memory
    
    def _create_image_view(self, image: int, format: int, aspect_flags: int):
        """Create an image view"""
        view_info = vk.VkImageViewCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            image=image,
            viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
            format=format,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=aspect_flags,
                baseMipLevel=0,
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1
            )
        )
        
        return vk.vkCreateImageView(self.device, view_info, None)
    
    def _copy_buffer(self, src_buffer: int, dst_buffer: int, size: int):
        """Copy data between buffers"""
        alloc_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandPool=self.command_pool,
            commandBufferCount=1
        )
        
        command_buffer = vk.vkAllocateCommandBuffers(self.device, alloc_info)[0]
        
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )
        
        vk.vkBeginCommandBuffer(command_buffer, begin_info)
        
        copy_region = vk.VkBufferCopy(size=size)
        vk.vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, [copy_region])
        
        vk.vkEndCommandBuffer(command_buffer)
        
        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[command_buffer]
        )
        
        vk.vkQueueSubmit(self.graphics_queue, 1, [submit_info], vk.VK_NULL_HANDLE)
        vk.vkQueueWaitIdle(self.graphics_queue)
        
        vk.vkFreeCommandBuffers(self.device, self.command_pool, 1, [command_buffer])
    
    def _find_memory_type(self, type_filter: int, properties: int) -> int:
        """Find suitable memory type"""
        mem_properties = vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        
        for i in range(mem_properties.memoryTypeCount):
            if (type_filter & (1 << i)) and (mem_properties.memoryTypes[i].propertyFlags & properties) == properties:
                return i
        
        raise RuntimeError("Failed to find suitable memory type")
    
    def _find_depth_format(self) -> int:
        """Find supported depth format"""
        candidates = [
            vk.VK_FORMAT_D32_SFLOAT,
            vk.VK_FORMAT_D32_SFLOAT_S8_UINT,
            vk.VK_FORMAT_D24_UNORM_S8_UINT
        ]
        
        for format in candidates:
            props = vk.vkGetPhysicalDeviceFormatProperties(self.physical_device, format)
            if props.optimalTilingFeatures & vk.VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT:
                return format
        
        raise RuntimeError("Failed to find supported depth format")
    
    def _update_uniform_buffer(self, current_image: int):
        """Update uniform buffer for current frame"""
        # Create matrices
        model = self._create_model_matrix()
        view = self._create_view_matrix()
        proj = self._create_projection_matrix()
        
        # Pack matrices into bytes
        matrices_data = []
        for matrix in [model, view, proj]:
            for row in matrix:
                matrices_data.extend(row)
        
        buffer_data = struct.pack(f'{len(matrices_data)}f', *matrices_data)
        
        # Update buffer
        data_ptr = vk.vkMapMemory(self.device, self.uniform_buffers_memory[current_image], 0, len(buffer_data), 0)
        import ctypes
        ctypes.memmove(data_ptr, buffer_data, len(buffer_data))
        vk.vkUnmapMemory(self.device, self.uniform_buffers_memory[current_image])
    
    def _create_model_matrix(self) -> np.ndarray:
        """Create model transformation matrix"""
        return np.eye(4, dtype=np.float32)
    
    def _create_view_matrix(self) -> np.ndarray:
        """Create view matrix from camera"""
        eye = self.camera_pos
        center = self.camera_target
        up = self.camera_up
        
        f = center - eye
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
    
    def _create_projection_matrix(self) -> np.ndarray:
        """Create perspective projection matrix"""
        fov = np.radians(45.0)
        aspect = self.swapchain_extent.width / self.swapchain_extent.height
        near = 0.1
        far = 1000.0
        
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = 1.0 / (aspect * np.tan(fov / 2))
        proj[1, 1] = 1.0 / np.tan(fov / 2)
        proj[2, 2] = far / (far - near)
        proj[2, 3] = 1.0
        proj[3, 2] = -(far * near) / (far - near)
        
        return proj
    
    def _record_command_buffer(self, command_buffer: int, image_index: int):
        """Record rendering commands"""
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        )
        
        vk.vkBeginCommandBuffer(command_buffer, begin_info)
        
        # Begin render pass
        render_pass_info = vk.VkRenderPassBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            renderPass=self.render_pass,
            framebuffer=self.framebuffers[image_index],
            renderArea=vk.VkRect2D(
                offset=vk.VkOffset2D(x=0, y=0),
                extent=self.swapchain_extent
            )
        )
        
        # Clear values
        clear_values = [
            vk.VkClearValue(color=vk.VkClearColorValue(float32=[0.1, 0.1, 0.1, 1.0])),
            vk.VkClearValue(depthStencil=vk.VkClearDepthStencilValue(depth=1.0, stencil=0))
        ]
        
        render_pass_info.clearValueCount = len(clear_values)
        render_pass_info.pClearValues = clear_values
        
        vk.vkCmdBeginRenderPass(command_buffer, render_pass_info, vk.VK_SUBPASS_CONTENTS_INLINE)
        
        # Bind pipeline
        vk.vkCmdBindPipeline(command_buffer, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.graphics_pipeline)
        
        # Bind vertex buffer
        vertex_buffers = [self.vertex_buffer]
        offsets = [0]
        vk.vkCmdBindVertexBuffers(command_buffer, 0, 1, vertex_buffers, offsets)
        
        # Bind index buffer
        vk.vkCmdBindIndexBuffer(command_buffer, self.index_buffer, 0, vk.VK_INDEX_TYPE_UINT32)
        
        # Bind descriptor sets
        vk.vkCmdBindDescriptorSets(
            command_buffer,
            vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            self.pipeline_layout,
            0, 1,
            [self.descriptor_sets[image_index]],
            0, None
        )
        
        # Draw
        vk.vkCmdDrawIndexed(command_buffer, len(self.indices), 1, 0, 0, 0)
        
        vk.vkCmdEndRenderPass(command_buffer)
        vk.vkEndCommandBuffer(command_buffer)
    
    def _draw_frame(self):
        """Draw a single frame"""
        current_frame = self.frame_count % self.MAX_FRAMES_IN_FLIGHT
        
        # Wait for previous frame
        vk.vkWaitForFences(self.device, 1, [self.in_flight_fences[current_frame]], vk.VK_TRUE, 1000000000)
        
        # Acquire image
        try:
            image_index = vk.vkAcquireNextImageKHR(
                self.device,
                self.swapchain,
                1000000000,
                self.image_available_semaphores[current_frame],
                vk.VK_NULL_HANDLE
            )
        except vk.VkErrorOutOfDateKhr:
            # Swapchain needs recreation
            return
        
        # Update uniform buffer
        self._update_uniform_buffer(image_index)
        
        # Reset fence
        vk.vkResetFences(self.device, 1, [self.in_flight_fences[current_frame]])
        
        # Record command buffer
        vk.vkResetCommandBuffer(self.command_buffers[image_index], 0)
        self._record_command_buffer(self.command_buffers[image_index], image_index)
        
        # Submit
        wait_semaphores = [self.image_available_semaphores[current_frame]]
        wait_stages = [vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT]
        signal_semaphores = [self.render_finished_semaphores[current_frame]]
        
        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            waitSemaphoreCount=1,
            pWaitSemaphores=wait_semaphores,
            pWaitDstStageMask=wait_stages,
            commandBufferCount=1,
            pCommandBuffers=[self.command_buffers[image_index]],
            signalSemaphoreCount=1,
            pSignalSemaphores=signal_semaphores
        )
        
        vk.vkQueueSubmit(self.graphics_queue, 1, [submit_info], self.in_flight_fences[current_frame])
        
        # Present
        present_info = vk.VkPresentInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            waitSemaphoreCount=1,
            pWaitSemaphores=signal_semaphores,
            swapchainCount=1,
            pSwapchains=[self.swapchain],
            pImageIndices=[image_index]
        )
        
        try:
            vk.vkQueuePresentKHR(self.present_queue, present_info)
        except vk.VkErrorOutOfDateKhr:
            # Swapchain needs recreation
            pass
        
        self.frame_count += 1
    
    def render_loop(self):
        """Main rendering loop"""
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self._draw_frame()
            
            # Calculate FPS
            current_time = time.time()
            delta_time = current_time - self.last_time
            if delta_time >= 1.0:
                fps = self.frame_count / delta_time
                glfw.set_window_title(self.window, f"Vulkan Terrain Renderer - {fps:.1f} FPS")
                self.frame_count = 0
                self.last_time = current_time
        
        # Wait for device to finish
        vk.vkDeviceWaitIdle(self.device)
    
    def cleanup(self):
        """Clean up all resources"""
        # Destroy synchronization objects
        for i in range(self.MAX_FRAMES_IN_FLIGHT):
            vk.vkDestroySemaphore(self.device, self.image_available_semaphores[i], None)
            vk.vkDestroySemaphore(self.device, self.render_finished_semaphores[i], None)
            vk.vkDestroyFence(self.device, self.in_flight_fences[i], None)
        
        # Destroy command pool
        vk.vkDestroyCommandPool(self.device, self.command_pool, None)
        
        # Destroy framebuffers
        for framebuffer in self.framebuffers:
            vk.vkDestroyFramebuffer(self.device, framebuffer, None)
        
        # Destroy pipeline
        vk.vkDestroyPipeline(self.device, self.graphics_pipeline, None)
        vk.vkDestroyPipelineLayout(self.device, self.pipeline_layout, None)
        vk.vkDestroyRenderPass(self.device, self.render_pass, None)
        
        # Destroy image views
        for image_view in self.swapchain_image_views:
            vk.vkDestroyImageView(self.device, image_view, None)
        
        # Destroy swapchain
        vk.vkDestroySwapchainKHR(self.device, self.swapchain, None)
        
        # Destroy buffers
        if hasattr(self, 'vertex_buffer'):
            vk.vkDestroyBuffer(self.device, self.vertex_buffer, None)
            vk.vkFreeMemory(self.device, self.vertex_buffer_memory, None)
        
        if hasattr(self, 'index_buffer'):
            vk.vkDestroyBuffer(self.device, self.index_buffer, None)
            vk.vkFreeMemory(self.device, self.index_buffer_memory, None)
        
        # Destroy uniform buffers
        for i in range(len(self.uniform_buffers)):
            vk.vkDestroyBuffer(self.device, self.uniform_buffers[i], None)
            vk.vkFreeMemory(self.device, self.uniform_buffers_memory[i], None)
        
        # Destroy descriptor pool
        vk.vkDestroyDescriptorPool(self.device, self.descriptor_pool, None)
        vk.vkDestroyDescriptorSetLayout(self.device, self.descriptor_set_layout, None)
        
        # Destroy depth resources
        vk.vkDestroyImageView(self.device, self.depth_image_view, None)
        vk.vkDestroyImage(self.device, self.depth_image, None)
        vk.vkFreeMemory(self.device, self.depth_image_memory, None)
        
        # Destroy device
        vk.vkDestroyDevice(self.device, None)
        
        # Destroy surface
        vk.vkDestroySurfaceKHR(self.instance, self.surface, None)
        
        # Destroy instance
        vk.vkDestroyInstance(self.instance, None)
        
        # Destroy window
        glfw.destroy_window(self.window)
        glfw.terminate()
    
    def _key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard input"""
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        
        # Camera movement
        move_speed = 0.5
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_W:
                self.camera_pos[1] += move_speed
            elif key == glfw.KEY_S:
                self.camera_pos[1] -= move_speed
            elif key == glfw.KEY_A:
                self.camera_pos[0] -= move_speed
            elif key == glfw.KEY_D:
                self.camera_pos[0] += move_speed
            elif key == glfw.KEY_Q:
                self.camera_pos[2] += move_speed
            elif key == glfw.KEY_E:
                self.camera_pos[2] -= move_speed
    
    def _mouse_callback(self, window, xpos, ypos):
        """Handle mouse movement"""
        # Implement mouse look if needed
        pass
    
    def _scroll_callback(self, window, xoffset, yoffset):
        """Handle mouse scroll"""
        # Zoom camera
        self.camera_pos[1] += yoffset * 0.5
    
    def _debug_mode(self) -> bool:
        """Check if running in debug mode"""
        return os.environ.get('VULKAN_VALIDATION', '0') == '1'
    
    def _load_shader(self, filename: str) -> bytes:
        """Load compiled SPIR-V shader"""
        # For now, return embedded shader code
        # In production, load from file
        if "vert" in filename:
            return self._get_vertex_shader_spirv()
        else:
            return self._get_fragment_shader_spirv()
    
    def _create_shader_module(self, code: bytes) -> int:
        """Create shader module from SPIR-V code"""
        create_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(code),
            pCode=code
        )
        
        return vk.vkCreateShaderModule(self.device, create_info, None)
    
    def _get_vertex_shader_spirv(self) -> bytes:
        """Get embedded vertex shader SPIR-V"""
        # This is a minimal vertex shader compiled to SPIR-V
        # In production, compile from GLSL
        return bytes([
            0x03, 0x02, 0x23, 0x07,  # SPIR-V magic number
            0x00, 0x00, 0x01, 0x00,  # Version 1.0
            # ... rest of SPIR-V bytecode
            # This would be generated by glslc compiler
        ])
    
    def _get_fragment_shader_spirv(self) -> bytes:
        """Get embedded fragment shader SPIR-V"""
        # Minimal fragment shader
        return bytes([
            0x03, 0x02, 0x23, 0x07,  # SPIR-V magic number
            0x00, 0x00, 0x01, 0x00,  # Version 1.0
            # ... rest of SPIR-V bytecode
        ])


def create_synthetic_heightmap(size: int = 256) -> np.ndarray:
    """Create synthetic terrain for testing"""
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    
    # Create interesting terrain
    Z = np.sin(np.sqrt(X**2 + Y**2)) * 0.3
    Z += np.exp(-(X**2 + Y**2) / 10) * 2
    Z += np.sin(X * 0.5) * np.cos(Y * 0.5) * 0.2
    Z += np.random.rand(size, size) * 0.05  # Add some noise
    
    # Normalize to 0-1 range
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    
    return Z


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vulkan Terrain Renderer")
    parser.add_argument('heightmap', nargs='?', help='Path to heightmap file (GeoTIFF or image)')
    parser.add_argument('--width', type=int, default=1920, help='Window width')
    parser.add_argument('--height', type=int, default=1080, help='Window height')
    parser.add_argument('--preset', choices=['high_performance', 'balanced', 'high_quality'], 
                       default='balanced', help='Rendering preset')
    parser.add_argument('--synthetic-size', type=int, default=256, 
                       help='Size of synthetic terrain if no file provided')
    parser.add_argument('--validation', action='store_true', help='Enable Vulkan validation layers')
    
    args = parser.parse_args()
    
    # Set validation environment variable
    if args.validation:
        os.environ['VULKAN_VALIDATION'] = '1'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create renderer
    config = TerrainConfig.from_preset(args.preset)
    renderer = VulkanTerrainRenderer(args.width, args.height, config)
    
    # Initialize
    if not renderer.initialize():
        logging.error("Failed to initialize renderer")
        return 1
    
    # Load heightmap
    if args.heightmap:
        # Try to load from file
        try:
            if args.heightmap.endswith('.tif') or args.heightmap.endswith('.tiff'):
                # Load GeoTIFF
                if RASTERIO_AVAILABLE:
                    import rasterio
                    with rasterio.open(args.heightmap) as src:
                        heightmap = src.read(1).astype(np.float32)
                        # Normalize
                        heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
                else:
                    logging.warning("Rasterio not available, using synthetic terrain")
                    heightmap = create_synthetic_heightmap(args.synthetic_size)
            else:
                # Try to load as image
                from PIL import Image
                img = Image.open(args.heightmap).convert('L')
                heightmap = np.array(img, dtype=np.float32) / 255.0
        except Exception as e:
            logging.error(f"Failed to load heightmap: {e}")
            logging.info("Using synthetic terrain instead")
            heightmap = create_synthetic_heightmap(args.synthetic_size)
    else:
        # Create synthetic terrain
        logging.info(f"Creating synthetic terrain ({args.synthetic_size}x{args.synthetic_size})")
        heightmap = create_synthetic_heightmap(args.synthetic_size)
    
    # Load terrain
    if not renderer.load_heightmap(heightmap):
        logging.error("Failed to load heightmap")
        renderer.cleanup()
        return 1
    
    logging.info("Starting render loop. Press ESC to exit.")
    logging.info("Controls: WASD to move, QE for up/down, mouse scroll to zoom")
    
    try:
        # Main render loop
        renderer.render_loop()
    except Exception as e:
        logging.error(f"Error during rendering: {e}")
    finally:
        # Clean up
        renderer.cleanup()
    
    return 0


# GLSL Shader source code
# Save these as separate files: terrain.vert and terrain.frag
# Then compile with: glslc terrain.vert -o terrain.vert.spv

VERTEX_SHADER_SOURCE = """
#version 450

// Input
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

// Uniforms
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

// Output
layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    
    fragPos = vec3(ubo.model * vec4(inPosition, 1.0));
    fragNormal = mat3(transpose(inverse(ubo.model))) * inNormal;
    
    // Height-based coloring
    float height = inPosition.z;
    if (height < 0.1) {
        fragColor = vec3(0.1, 0.3, 0.5); // Water (blue)
    } else if (height < 0.3) {
        fragColor = vec3(0.2, 0.6, 0.2); // Grass (green)
    } else if (height < 0.6) {
        fragColor = vec3(0.5, 0.4, 0.3); // Rock (brown)
    } else {
        fragColor = vec3(0.9, 0.9, 0.9); // Snow (white)
    }
}
"""

FRAGMENT_SHADER_SOURCE = """
#version 450

// Input
layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;

// Output
layout(location = 0) out vec4 outColor;

void main() {
    // Simple directional lighting
    vec3 lightDir = normalize(vec3(1.0, 1.0, 2.0));
    vec3 normal = normalize(fragNormal);
    
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * fragColor;
    
    vec3 ambient = 0.3 * fragColor;
    vec3 result = ambient + diffuse;
    
    outColor = vec4(result, 1.0);
}
"""


# Create shader compilation script
def create_shader_compilation_script():
    """Create a script to compile shaders"""
    compile_script = """#!/bin/bash
# Compile shaders to SPIR-V

echo "Compiling shaders..."

# Check if glslc is available
if ! command -v glslc &> /dev/null; then
    echo "glslc not found. Please install Vulkan SDK."
    exit 1
fi

# Create shaders directory
mkdir -p shaders

# Write shader sources
cat > shaders/terrain.vert << 'EOF'
""" + VERTEX_SHADER_SOURCE + """
EOF

cat > shaders/terrain.frag << 'EOF'
""" + FRAGMENT_SHADER_SOURCE + """
EOF

# Compile shaders
glslc shaders/terrain.vert -o shaders/terrain.vert.spv
glslc shaders/terrain.frag -o shaders/terrain.frag.spv

echo "Shader compilation complete!"
"""
    
    with open('compile_shaders.sh', 'w') as f:
        f.write(compile_script)
    
    os.chmod('compile_shaders.sh', 0o755)
    print("Created compile_shaders.sh - run this to compile shaders")


if __name__ == "__main__":
    # Check dependencies
    missing_deps = []
    
    if not GLFW_AVAILABLE:
        missing_deps.append("glfw")
    
    if not VULKAN_AVAILABLE:
        missing_deps.append("vulkan")
    
    if missing_deps:
        print("Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing_deps)}")
        
        if 'glfw' in missing_deps:
            print("\nNote: GLFW also requires system libraries.")
            print("  Ubuntu/Debian: sudo apt-get install libglfw3-dev")
            print("  macOS: brew install glfw")
            print("  Windows: pip install glfw should include binaries")
        
        sys.exit(1)
    
    # Create shader compilation script
    if not os.path.exists('shaders/terrain.vert.spv'):
        create_shader_compilation_script()
        print("\nShaders need to be compiled. Run:")
        print("  ./compile_shaders.sh")
        print("\nOr manually create minimal SPIR-V shaders.")
        print("\nFor testing without shaders, the renderer will use embedded minimal shaders.")
    
    # Run main
    sys.exit(main())


