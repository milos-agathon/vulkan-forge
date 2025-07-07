#pragma once

#include "vertex_buffer.hpp"
#include "vk_common.hpp"
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <memory>
#include <array>

namespace vf {

/**
 * @brief Shader stage information for pipeline creation
 */
struct ShaderStage {
    VkShaderStageFlagBits stage;        ///< Shader stage (vertex, fragment, etc.)
    std::vector<uint32_t> spirvCode;    ///< SPIR-V bytecode
    std::string entryPoint;             ///< Entry point function name
    
    ShaderStage(VkShaderStageFlagBits stageFlags, 
               const std::vector<uint32_t>& code,
               const std::string& entry = "main")
        : stage(stageFlags), spirvCode(code), entryPoint(entry) {}
};

/**
 * @brief Push constant range for shader uniforms
 */
struct PushConstantRange {
    VkShaderStageFlags stageFlags;      ///< Which shader stages use this range
    uint32_t offset;                    ///< Offset in push constant block
    uint32_t size;                      ///< Size of the range in bytes
    
    PushConstantRange(VkShaderStageFlags stages, uint32_t off, uint32_t sz)
        : stageFlags(stages), offset(off), size(sz) {}
};

/**
 * @brief Mesh rendering pipeline configuration
 */
struct MeshPipelineConfig {
    // Shaders
    std::vector<ShaderStage> shaderStages;
    
    // Vertex input
    VertexLayout vertexLayout;
    
    // Rasterization
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
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
    VkBlendFactor srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    VkBlendFactor dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    VkBlendOp colorBlendOp = VK_BLEND_OP_ADD;
    
    // Push constants
    std::vector<PushConstantRange> pushConstants;
    
    // Debug
    std::string debugName = "mesh_pipeline";
    
    MeshPipelineConfig(const VertexLayout& layout) : vertexLayout(layout) {}
};

/**
 * @brief Graphics pipeline for mesh rendering
 * 
 * Encapsulates a complete Vulkan graphics pipeline optimized for
 * static mesh rendering with common features like MVP transforms,
 * texturing, and material support.
 */
class MeshPipeline {
public:
    MeshPipeline() = default;
    ~MeshPipeline();
    
    // Non-copyable but movable
    MeshPipeline(const MeshPipeline&) = delete;
    MeshPipeline& operator=(const MeshPipeline&) = delete;
    MeshPipeline(MeshPipeline&& other) noexcept;
    MeshPipeline& operator=(MeshPipeline&& other) noexcept;
    
    /**
     * @brief Create mesh rendering pipeline
     * 
     * @param device Vulkan logical device
     * @param renderPass Render pass the pipeline will be used with
     * @param config Pipeline configuration
     * @return VkResult creation result
     */
    VkResult create(VkDevice device,
                   VkRenderPass renderPass,
                   const MeshPipelineConfig& config);
    
    /**
     * @brief Create pipeline with embedded shaders
     * 
     * Uses default mesh vertex and fragment shaders embedded in the library.
     * 
     * @param device Vulkan logical device
     * @param renderPass Render pass
     * @param vertexLayout Vertex input layout
     * @param debugName Pipeline debug name
     * @return VkResult creation result
     */
    VkResult createDefault(VkDevice device,
                          VkRenderPass renderPass,
                          const VertexLayout& vertexLayout,
                          const std::string& debugName = "default_mesh");
    
    /**
     * @brief Bind pipeline for rendering
     * 
     * @param commandBuffer Command buffer to bind pipeline to
     */
    void bind(VkCommandBuffer commandBuffer) const;
    
    /**
     * @brief Update push constants
     * 
     * @param commandBuffer Command buffer
     * @param stageFlags Shader stages to update
     * @param offset Offset in push constant block
     * @param size Size of data to update
     * @param data Pointer to push constant data
     */
    void pushConstants(VkCommandBuffer commandBuffer,
                      VkShaderStageFlags stageFlags,
                      uint32_t offset,
                      uint32_t size,
                      const void* data) const;
    
    /**
     * @brief Bind descriptor sets
     * 
     * @param commandBuffer Command buffer
     * @param firstSet First descriptor set index
     * @param descriptorSets Array of descriptor sets to bind
     * @param descriptorSetCount Number of descriptor sets
     * @param dynamicOffsets Dynamic offsets (can be nullptr)
     * @param dynamicOffsetCount Number of dynamic offsets
     */
    void bindDescriptorSets(VkCommandBuffer commandBuffer,
                           uint32_t firstSet,
                           const VkDescriptorSet* descriptorSets,
                           uint32_t descriptorSetCount,
                           const uint32_t* dynamicOffsets = nullptr,
                           uint32_t dynamicOffsetCount = 0) const;
    
    /**
     * @brief Destroy pipeline and free resources
     */
    void destroy();
    
    // Getters
    VkPipeline getPipeline() const { return pipeline_; }
    VkPipelineLayout getLayout() const { return pipelineLayout_; }
    const MeshPipelineConfig& getConfig() const { return config_; }
    bool isValid() const { return pipeline_ != VK_NULL_HANDLE; }
    
private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    MeshPipelineConfig config_{VertexLayout(0)};
    
    /// Create shader module from SPIR-V code
    VkResult createShaderModule(const std::vector<uint32_t>& code, VkShaderModule& outModule);
    
    /// Get embedded default vertex shader
    std::vector<uint32_t> getDefaultVertexShader() const;
    
    /// Get embedded default fragment shader
    std::vector<uint32_t> getDefaultFragmentShader() const;
};

/**
 * @brief Mesh pipeline factory for common pipeline types
 */
class MeshPipelineFactory {
public:
    /**
     * @brief Create basic mesh pipeline (position + color)
     * 
     * @param device Vulkan logical device
     * @param renderPass Render pass
     * @param outPipeline Output pipeline
     * @return VkResult creation result
     */
    static VkResult createBasic(VkDevice device,
                               VkRenderPass renderPass,
                               std::unique_ptr<MeshPipeline>& outPipeline);
    
    /**
     * @brief Create textured mesh pipeline (position + normal + UV)
     * 
     * @param device Vulkan logical device
     * @param renderPass Render pass
     * @param outPipeline Output pipeline
     * @return VkResult creation result
     */
    static VkResult createTextured(VkDevice device,
                                  VkRenderPass renderPass,
                                  std::unique_ptr<MeshPipeline>& outPipeline);
    
    /**
     * @brief Create wireframe mesh pipeline
     * 
     * @param device Vulkan logical device
     * @param renderPass Render pass
     * @param vertexLayout Vertex input layout
     * @param outPipeline Output pipeline
     * @return VkResult creation result
     */
    static VkResult createWireframe(VkDevice device,
                                   VkRenderPass renderPass,
                                   const VertexLayout& vertexLayout,
                                   std::unique_ptr<MeshPipeline>& outPipeline);
    
    /**
     * @brief Create unlit mesh pipeline (no lighting calculations)
     * 
     * @param device Vulkan logical device
     * @param renderPass Render pass
     * @param vertexLayout Vertex input layout
     * @param outPipeline Output pipeline
     * @return VkResult creation result
     */
    static VkResult createUnlit(VkDevice device,
                               VkRenderPass renderPass,
                               const VertexLayout& vertexLayout,
                               std::unique_ptr<MeshPipeline>& outPipeline);
};

/**
 * @brief Common push constant structures for mesh rendering
 */
namespace MeshUniforms {
    /**
     * @brief Basic MVP matrix uniform
     */
    struct MVP {
        alignas(16) std::array<float, 16> model;       ///< Model matrix
        alignas(16) std::array<float, 16> view;        ///< View matrix  
        alignas(16) std::array<float, 16> projection;  ///< Projection matrix
        
        static constexpr uint32_t size() { return sizeof(MVP); }
    };
    
    /**
     * @brief Material properties
     */
    struct Material {
        alignas(16) std::array<float, 4> baseColor;    ///< Base color (RGBA)
        alignas(4) float metallic;                     ///< Metallic factor
        alignas(4) float roughness;                    ///< Roughness factor
        alignas(4) float emissive;                     ///< Emissive factor
        alignas(4) float padding;                      ///< Alignment padding
        
        static constexpr uint32_t size() { return sizeof(Material); }
    };
    
    /**
     * @brief Combined MVP + Material uniform
     */
    struct MVPMaterial {
        MVP mvp;
        Material material;
        
        static constexpr uint32_t size() { return sizeof(MVPMaterial); }
    };
}

} // namespace vf