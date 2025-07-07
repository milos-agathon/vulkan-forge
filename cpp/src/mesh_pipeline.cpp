#include "vf/mesh_pipeline.hpp"
#include "vf/vk_common.hpp"

// Include embedded shader data (these would be generated from compiled shaders)
namespace vf {
namespace EmbeddedShaders {
    // Placeholder for embedded SPIR-V data - will be replaced with actual compiled shaders
    extern const uint32_t mesh_vert_spv[];
    extern const size_t mesh_vert_spv_size;
    extern const uint32_t mesh_frag_spv[];
    extern const size_t mesh_frag_spv_size;
    extern const uint32_t wireframe_frag_spv[];
    extern const size_t wireframe_frag_spv_size;
}
}

namespace vf {

// ============================================================================
// MeshPipeline Implementation
// ============================================================================

MeshPipeline::~MeshPipeline() {
    destroy();
}

MeshPipeline::MeshPipeline(MeshPipeline&& other) noexcept
    : device_(other.device_)
    , pipeline_(other.pipeline_)
    , pipelineLayout_(other.pipelineLayout_)
    , config_(std::move(other.config_)) {
    
    other.device_ = VK_NULL_HANDLE;
    other.pipeline_ = VK_NULL_HANDLE;
    other.pipelineLayout_ = VK_NULL_HANDLE;
}

MeshPipeline& MeshPipeline::operator=(MeshPipeline&& other) noexcept {
    if (this != &other) {
        destroy();
        
        device_ = other.device_;
        pipeline_ = other.pipeline_;
        pipelineLayout_ = other.pipelineLayout_;
        config_ = std::move(other.config_);
        
        other.device_ = VK_NULL_HANDLE;
        other.pipeline_ = VK_NULL_HANDLE;
        other.pipelineLayout_ = VK_NULL_HANDLE;
    }
    return *this;
}

VkResult MeshPipeline::create(VkDevice device,
                             VkRenderPass renderPass,
                             const MeshPipelineConfig& config) {
    if (isValid()) {
        destroy();
    }
    
    device_ = device;
    config_ = config;
    
    // Create shader modules
    std::vector<VkShaderModule> shaderModules;
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
    
    for (const auto& stage : config.shaderStages) {
        VkShaderModule module;
        VkResult result = createShaderModule(stage.spirvCode, module);
        if (result != VK_SUCCESS) {
            // Cleanup already created modules
            for (auto mod : shaderModules) {
                vkDestroyShaderModule(device_, mod, nullptr);
            }
            return result;
        }
        shaderModules.push_back(module);
        
        VkPipelineShaderStageCreateInfo stageInfo{};
        stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfo.stage = stage.stage;
        stageInfo.module = module;
        stageInfo.pName = stage.entryPoint.c_str();
        shaderStages.push_back(stageInfo);
    }
    
    // Vertex input state
    auto bindingDesc = config.vertexLayout.getBindingDescription();
    auto attributeDescs = config.vertexLayout.getAttributeDescriptions();
    
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescs.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescs.data();
    
    // Input assembly state
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = config.topology;
    inputAssembly.primitiveRestartEnable = VK_FALSE;
    
    // Viewport state (dynamic)
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    
    // Rasterization state
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = config.polygonMode;
    rasterizer.lineWidth = config.lineWidth;
    rasterizer.cullMode = config.cullMode;
    rasterizer.frontFace = config.frontFace;
    rasterizer.depthBiasEnable = VK_FALSE;
    
    // Multisampling state
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    
    // Depth and stencil state
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = config.depthTestEnable ? VK_TRUE : VK_FALSE;
    depthStencil.depthWriteEnable = config.depthWriteEnable ? VK_TRUE : VK_FALSE;
    depthStencil.depthCompareOp = config.depthCompareOp;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;
    
    // Color blending
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | 
                                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = config.blendEnable ? VK_TRUE : VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = config.srcColorBlendFactor;
    colorBlendAttachment.dstColorBlendFactor = config.dstColorBlendFactor;
    colorBlendAttachment.colorBlendOp = config.colorBlendOp;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;
    
    // Dynamic state
    VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;
    
    // Create pipeline layout
    std::vector<VkPushConstantRange> pushConstantRanges;
    for (const auto& range : config.pushConstants) {
        VkPushConstantRange vkRange{};
        vkRange.stageFlags = range.stageFlags;
        vkRange.offset = range.offset;
        vkRange.size = range.size;
        pushConstantRanges.push_back(vkRange);
    }
    
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0; // TODO: Add descriptor set layout support
    pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRanges.size());
    pipelineLayoutInfo.pPushConstantRanges = pushConstantRanges.data();
    
    VkResult result = vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout_);
    if (result != VK_SUCCESS) {
        for (auto module : shaderModules) {
            vkDestroyShaderModule(device_, module, nullptr);
        }
        return result;
    }
    
    // Create graphics pipeline
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout_;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    
    result = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_);
    
    // Cleanup shader modules
    for (auto module : shaderModules) {
        vkDestroyShaderModule(device_, module, nullptr);
    }
    
    return result;
}

VkResult MeshPipeline::createDefault(VkDevice device,
                                    VkRenderPass renderPass,
                                    const VertexLayout& vertexLayout,
                                    const std::string& debugName) {
    MeshPipelineConfig config(vertexLayout);
    config.debugName = debugName;
    
    // Add default shaders
    config.shaderStages.emplace_back(VK_SHADER_STAGE_VERTEX_BIT, getDefaultVertexShader());
    config.shaderStages.emplace_back(VK_SHADER_STAGE_FRAGMENT_BIT, getDefaultFragmentShader());
    
    // Add MVP push constant
    config.pushConstants.emplace_back(
        VK_SHADER_STAGE_VERTEX_BIT,
        0,
        sizeof(MeshUniforms::MVP)
    );
    
    return create(device, renderPass, config);
}

void MeshPipeline::bind(VkCommandBuffer commandBuffer) const {
    if (isValid()) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    }
}

void MeshPipeline::pushConstants(VkCommandBuffer commandBuffer,
                                VkShaderStageFlags stageFlags,
                                uint32_t offset,
                                uint32_t size,
                                const void* data) const {
    if (isValid()) {
        vkCmdPushConstants(commandBuffer, pipelineLayout_, stageFlags, offset, size, data);
    }
}

void MeshPipeline::bindDescriptorSets(VkCommandBuffer commandBuffer,
                                     uint32_t firstSet,
                                     const VkDescriptorSet* descriptorSets,
                                     uint32_t descriptorSetCount,
                                     const uint32_t* dynamicOffsets,
                                     uint32_t dynamicOffsetCount) const {
    if (isValid()) {
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                               pipelineLayout_, firstSet, descriptorSetCount,
                               descriptorSets, dynamicOffsetCount, dynamicOffsets);
    }
}

void MeshPipeline::destroy() {
    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    
    if (pipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        pipelineLayout_ = VK_NULL_HANDLE;
    }
    
    device_ = VK_NULL_HANDLE;
}

VkResult MeshPipeline::createShaderModule(const std::vector<uint32_t>& code, VkShaderModule& outModule) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size() * sizeof(uint32_t);
    createInfo.pCode = code.data();
    
    return vkCreateShaderModule(device_, &createInfo, nullptr, &outModule);
}

std::vector<uint32_t> MeshPipeline::getDefaultVertexShader() const {
    // Placeholder - will be replaced with actual embedded SPIR-V
    // For now, return empty vector (will cause pipeline creation to fail gracefully)
    return {};
}

std::vector<uint32_t> MeshPipeline::getDefaultFragmentShader() const {
    // Placeholder - will be replaced with actual embedded SPIR-V
    // For now, return empty vector (will cause pipeline creation to fail gracefully)
    return {};
}

// ============================================================================
// MeshPipelineFactory Implementation
// ============================================================================

VkResult MeshPipelineFactory::createBasic(VkDevice device,
                                         VkRenderPass renderPass,
                                         std::unique_ptr<MeshPipeline>& outPipeline) {
    auto pipeline = std::make_unique<MeshPipeline>();
    
    VertexLayout layout = VertexLayouts::positionColor();
    VkResult result = pipeline->createDefault(device, renderPass, layout, "basic_mesh");
    
    if (result == VK_SUCCESS) {
        outPipeline = std::move(pipeline);
    }
    
    return result;
}

VkResult MeshPipelineFactory::createTextured(VkDevice device,
                                            VkRenderPass renderPass,
                                            std::unique_ptr<MeshPipeline>& outPipeline) {
    auto pipeline = std::make_unique<MeshPipeline>();
    
    VertexLayout layout = VertexLayouts::positionNormalUV();
    VkResult result = pipeline->createDefault(device, renderPass, layout, "textured_mesh");
    
    if (result == VK_SUCCESS) {
        outPipeline = std::move(pipeline);
    }
    
    return result;
}

VkResult MeshPipelineFactory::createWireframe(VkDevice device,
                                             VkRenderPass renderPass,
                                             const VertexLayout& vertexLayout,
                                             std::unique_ptr<MeshPipeline>& outPipeline) {
    auto pipeline = std::make_unique<MeshPipeline>();
    
    MeshPipelineConfig config(vertexLayout);
    config.debugName = "wireframe_mesh";
    config.polygonMode = VK_POLYGON_MODE_LINE;
    config.cullMode = VK_CULL_MODE_NONE;
    
    // Add shaders (vertex + wireframe fragment)
    config.shaderStages.emplace_back(VK_SHADER_STAGE_VERTEX_BIT, pipeline->getDefaultVertexShader());
    config.shaderStages.emplace_back(VK_SHADER_STAGE_FRAGMENT_BIT, pipeline->getDefaultFragmentShader());
    
    // Add MVP push constant
    config.pushConstants.emplace_back(
        VK_SHADER_STAGE_VERTEX_BIT,
        0,
        sizeof(MeshUniforms::MVP)
    );
    
    VkResult result = pipeline->create(device, renderPass, config);
    
    if (result == VK_SUCCESS) {
        outPipeline = std::move(pipeline);
    }
    
    return result;
}

VkResult MeshPipelineFactory::createUnlit(VkDevice device,
                                         VkRenderPass renderPass,
                                         const VertexLayout& vertexLayout,
                                         std::unique_ptr<MeshPipeline>& outPipeline) {
    auto pipeline = std::make_unique<MeshPipeline>();
    
    MeshPipelineConfig config(vertexLayout);
    config.debugName = "unlit_mesh";
    
    // Add shaders
    config.shaderStages.emplace_back(VK_SHADER_STAGE_VERTEX_BIT, pipeline->getDefaultVertexShader());
    config.shaderStages.emplace_back(VK_SHADER_STAGE_FRAGMENT_BIT, pipeline->getDefaultFragmentShader());
    
    // Add push constants for MVP + material
    config.pushConstants.emplace_back(
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(MeshUniforms::MVPMaterial)
    );
    
    VkResult result = pipeline->create(device, renderPass, config);
    
    if (result == VK_SUCCESS) {
        outPipeline = std::move(pipeline);
    }
    
    return result;
}

} // namespace vf