/**
 * @file tessellation_pipeline.hpp
 * @brief GPU tessellation pipeline wrapper for high-performance terrain rendering
 * 
 * Provides comprehensive tessellation pipeline management including:
 * - Shader compilation and pipeline creation
 * - Push constants and descriptor set management
 * - Multiple pipeline variants (solid, wireframe, debug)
 * - Hot shader reloading for development
 * - Performance monitoring and statistics
 */

#pragma once

#include "vf/vk_common.hpp"
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <chrono>

namespace vf {

/**
 * @brief Tessellation pipeline configuration
 */
struct TessellationPipelineConfig {
    // Shader paths (optional - can use embedded shaders)
    std::string vertexShaderPath;
    std::string tessControlShaderPath;
    std::string tessEvalShaderPath;
    std::string fragmentShaderPath;
    
    // Pipeline state
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    uint32_t patchControlPoints = 4; // Quad patches
    VkPolygonMode polygonMode = VK_POLYGON_MODE_FILL;
    VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT;
    VkFrontFace frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    float lineWidth = 1.0f;
    
    // Depth testing
    bool depthTestEnable = true;
    bool depthWriteEnable = true;
    VkCompareOp depthCompareOp = VK_COMPARE_OP_LESS;
    
    // Blending
    bool blendEnable = false;
    VkBlendFactor srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    VkBlendFactor dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    VkBlendOp colorBlendOp = VK_BLEND_OP_ADD;
    
    // Multisampling
    VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT;
    
    // Specialization constants for shaders
    std::unordered_map<uint32_t, uint32_t> specializationConstants;
};

/**
 * @brief Vertex input description for terrain meshes
 */
struct TerrainVertexInput {
    static VkVertexInputBindingDescription getBindingDescription();
    static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions();
};

/**
 * @brief Descriptor set layout information
 */
struct DescriptorSetLayoutInfo {
    struct Binding {
        uint32_t binding;
        VkDescriptorType type;
        uint32_t count;
        VkShaderStageFlags stageFlags;
        
        Binding(uint32_t b, VkDescriptorType t, uint32_t c, VkShaderStageFlags s)
            : binding(b), type(t), count(c), stageFlags(s) {}
    };
    
    std::vector<Binding> bindings;
    
    void addBinding(uint32_t binding, VkDescriptorType type, 
                   uint32_t count, VkShaderStageFlags stages) {
        bindings.emplace_back(binding, type, count, stages);
    }
};

/**
 * @brief Push constants structure matching shader layout
 */
struct TessellationPushConstants {
    glm::mat4 modelMatrix;
    glm::mat4 viewMatrix;
    glm::mat4 projMatrix;
    glm::mat4 mvpMatrix;
    
    glm::vec3 cameraPosition;
    float tessellationScale;
    
    glm::vec2 heightmapSize;
    glm::vec2 terrainScale;
    float heightScale;
    float time;
    
    float nearDistance;
    float farDistance;
    float minTessLevel;
    float maxTessLevel;
    
    // Lighting parameters
    glm::vec3 sunDirection;
    float padding1;
    glm::vec3 sunColor;
    float padding2;
    glm::vec3 ambientColor;
    float shadowIntensity;
    
    // Fog parameters
    glm::vec3 fogColor;
    float fogDensity;
    float fogStart;
    float fogEnd;
    float padding3;
    float padding4;
    
    // Material parameters
    float roughness;
    float metallic;
    float specularPower;
    float padding5;
    
    // Wireframe parameters (for debug mode)
    glm::vec3 wireframeColor;
    float wireframeThickness;
    float wireframeOpacity;
    int visualizationMode;
    glm::vec3 lowTessColor;
    float padding6;
    glm::vec3 highTessColor;
    float padding7;
    glm::vec3 nearColor;
    float padding8;
    glm::vec3 farColor;
    float padding9;
};

static_assert(sizeof(TessellationPushConstants) <= 256, 
              "Push constants exceed maximum size");

/**
 * @brief Pipeline statistics for performance monitoring
 */
struct PipelineStats {
    uint64_t drawCalls = 0;
    uint64_t trianglesRendered = 0;
    uint64_t patchesSubmitted = 0;
    std::chrono::nanoseconds totalRenderTime{0};
    std::chrono::nanoseconds avgRenderTime{0};
    
    void reset() {
        drawCalls = 0;
        trianglesRendered = 0;
        patchesSubmitted = 0;
        totalRenderTime = std::chrono::nanoseconds{0};
        avgRenderTime = std::chrono::nanoseconds{0};
    }
    
    void addDrawCall(uint64_t triangles, uint64_t patches, 
                    std::chrono::nanoseconds renderTime) {
        drawCalls++;
        trianglesRendered += triangles;
        patchesSubmitted += patches;
        totalRenderTime += renderTime;
        avgRenderTime = totalRenderTime / drawCalls;
    }
};

/**
 * @brief High-performance tessellation pipeline for terrain rendering
 */
class TessellationPipeline {
public:
    TessellationPipeline();
    ~TessellationPipeline();
    
    // Initialization
    VkResult initialize(VkRenderPass renderPass, 
                       uint32_t subpass,
                       const TessellationPipelineConfig& config = {});
    void destroy();
    
    // Pipeline management
    VkResult createPipeline(VkRenderPass renderPass, 
                           uint32_t subpass,
                           const TessellationPipelineConfig& config);
    VkResult recreatePipeline(const TessellationPipelineConfig& config);
    
    // Shader management
    VkResult loadShaders(const TessellationPipelineConfig& config);
    VkResult reloadShaders(); // Hot reload for development
    VkResult compileShaderFromSource(VkShaderStageFlagBits stage, 
                                   const std::string& source,
                                   const std::string& entryPoint = "main");
    
    // Descriptor set management
    VkResult createDescriptorSetLayout(const DescriptorSetLayoutInfo& layoutInfo);
    VkDescriptorSetLayout getDescriptorSetLayout() const { return m_descriptorSetLayout; }
    VkPipelineLayout getPipelineLayout() const { return m_pipelineLayout; }
    
    // Rendering
    void bind(VkCommandBuffer commandBuffer);
    void updatePushConstants(VkCommandBuffer commandBuffer, 
                           const TessellationPushConstants& pushConstants);
    void bindDescriptorSets(VkCommandBuffer commandBuffer,
                          const std::vector<VkDescriptorSet>& descriptorSets,
                          uint32_t firstSet = 0);
    
    VkResult draw(VkCommandBuffer commandBuffer,
                 uint32_t patchCount,
                 uint32_t instanceCount = 1,
                 uint32_t firstPatch = 0,
                 uint32_t firstInstance = 0);
    
    VkResult drawIndexed(VkCommandBuffer commandBuffer,
                        uint32_t indexCount,
                        uint32_t instanceCount = 1,
                        uint32_t firstIndex = 0,
                        int32_t vertexOffset = 0,
                        uint32_t firstInstance = 0);
    
    // State queries
    bool isValid() const { return m_pipeline != VK_NULL_HANDLE; }
    bool isInitialized() const { return m_initialized; }
    const TessellationPipelineConfig& getConfig() const { return m_config; }
    
    // Statistics and debugging
    const PipelineStats& getStats() const { return m_stats; }
    void resetStats() { m_stats.reset(); }
    void enableProfiling(bool enable) { m_profilingEnabled = enable; }
    
    // Validation and debugging
    VkResult validateShaders() const;
    std::string getShaderInfoLog(VkShaderStageFlagBits stage) const;
    
private:
    // Internal shader management
    struct ShaderModule {
        VkShaderModule module = VK_NULL_HANDLE;
        std::vector<uint32_t> spirvCode;
        std::string source;
        std::string entryPoint = "main";
        std::chrono::file_time_type lastModified;
        
        void destroy();
    };
    
    // Internal pipeline creation helpers
    VkResult createShaderStages(std::vector<VkPipelineShaderStageCreateInfo>& stages);
    VkResult createPipelineLayout();
    VkResult createVertexInputState(VkPipelineVertexInputStateCreateInfo& vertexInput);
    VkResult createInputAssemblyState(VkPipelineInputAssemblyStateCreateInfo& inputAssembly);
    VkResult createTessellationState(VkPipelineTessellationStateCreateInfo& tessellation);
    VkResult createViewportState(VkPipelineViewportStateCreateInfo& viewport);
    VkResult createRasterizationState(VkPipelineRasterizationStateCreateInfo& rasterization);
    VkResult createMultisampleState(VkPipelineMultisampleStateCreateInfo& multisample);
    VkResult createDepthStencilState(VkPipelineDepthStencilStateCreateInfo& depthStencil);
    VkResult createColorBlendState(VkPipelineColorBlendStateCreateInfo& colorBlend);
    VkResult createDynamicState(VkPipelineDynamicStateCreateInfo& dynamic);
    
    // Shader compilation utilities
    VkResult compileGLSLToSPIRV(VkShaderStageFlagBits stage,
                               const std::string& source,
                               std::vector<uint32_t>& spirv);
    VkResult createShaderModule(const std::vector<uint32_t>& spirvCode,
                              VkShaderModule& shaderModule);
    std::string loadShaderSource(const std::string& filepath);
    bool checkShaderFileModified(VkShaderStageFlagBits stage);
    
    // Performance monitoring
    void beginProfiling(VkCommandBuffer commandBuffer);
    void endProfiling(VkCommandBuffer commandBuffer, uint64_t triangles, uint64_t patches);
    
private:
    // Core Vulkan objects
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    
    // Shaders
    std::unordered_map<VkShaderStageFlagBits, ShaderModule> m_shaders;
    
    // Configuration
    TessellationPipelineConfig m_config;
    bool m_initialized = false;
    
    // Performance monitoring
    PipelineStats m_stats;
    bool m_profilingEnabled = false;
    VkQueryPool m_queryPool = VK_NULL_HANDLE;
    uint32_t m_currentQuery = 0;
    
    // Hot reload support
    bool m_hotReloadEnabled = false;
    std::unordered_map<VkShaderStageFlagBits, std::string> m_shaderPaths;
    
    // Push constants range
    VkPushConstantRange m_pushConstantRange;
    
    // Default embedded shaders (fallback)
    static const std::unordered_map<VkShaderStageFlagBits, std::vector<uint32_t>> s_embeddedShaders;
};

/**
 * @brief Factory for creating specialized tessellation pipelines
 */
class TessellationPipelineFactory {
public:
    static std::unique_ptr<TessellationPipeline> createSolidPipeline(
        VkRenderPass renderPass, uint32_t subpass);
    
    static std::unique_ptr<TessellationPipeline> createWireframePipeline(
        VkRenderPass renderPass, uint32_t subpass);
    
    static std::unique_ptr<TessellationPipeline> createDebugPipeline(
        VkRenderPass renderPass, uint32_t subpass, uint32_t debugMode);
    
    static DescriptorSetLayoutInfo getDefaultDescriptorLayout();
    static TessellationPipelineConfig getDefaultConfig();
    static TessellationPipelineConfig getWireframeConfig();
    static TessellationPipelineConfig getDebugConfig(uint32_t debugMode);
};

/**
 * @brief RAII helper for pipeline binding and management
 */
class TessellationPipelineScope {
public:
    TessellationPipelineScope(TessellationPipeline& pipeline, 
                             VkCommandBuffer commandBuffer);
    ~TessellationPipelineScope();
    
    void updatePushConstants(const TessellationPushConstants& pushConstants);
    void bindDescriptorSets(const std::vector<VkDescriptorSet>& descriptorSets);
    VkResult draw(uint32_t patchCount, uint32_t instanceCount = 1);
    VkResult drawIndexed(uint32_t indexCount, uint32_t instanceCount = 1);
    
private:
    TessellationPipeline& m_pipeline;
    VkCommandBuffer m_commandBuffer;
    bool m_bound = false;
};

/**
 * @brief Multi-pipeline manager for handling different rendering modes
 */
class TessellationPipelineManager {
public:
    enum class PipelineType {
        Solid,
        Wireframe,
        Debug
    };
    
    TessellationPipelineManager();
    ~TessellationPipelineManager();
    
    VkResult initialize(VkRenderPass renderPass, uint32_t subpass);
    void destroy();
    
    TessellationPipeline* getPipeline(PipelineType type);
    VkResult switchPipeline(VkCommandBuffer commandBuffer, PipelineType type);
    
    void enableHotReload(bool enable);
    VkResult reloadAllShaders();
    
    const PipelineStats& getStats(PipelineType type) const;
    void resetAllStats();
    
private:
    std::unordered_map<PipelineType, std::unique_ptr<TessellationPipeline>> m_pipelines;
    PipelineType m_currentPipeline = PipelineType::Solid;
    bool m_initialized = false;
};

} // namespace vf