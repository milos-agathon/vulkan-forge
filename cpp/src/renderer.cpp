#include "vf/renderer.hpp"
#include <array>
#include <stdexcept>
#include <fstream>

#define VK_CHECK(call)                                            \
    do{VkResult _r=call; if(_r!=VK_SUCCESS)                       \
        throw std::runtime_error("Vulkan error " #call);}while(0)

namespace {

/* ───── tiny, pre-compiled SPIR-V (fullscreen tri) ─────
   * vertex: pass through pos
   * fragment: outColor = vec4(inColor.rgb,1)
   compiled with:
   glslc -fshader-stage=vert passthrough.vert
----------------------------------------------------------------*/
static const uint32_t VTX_SPV[] = {
#include "shader_passthrough.vert.inc"
};
static const uint32_t FRG_SPV[] = {
#include "shader_color.frag.inc"
};

VkShaderModule makeShader(VkDevice dev, const uint32_t* code, size_t words)
{
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = words*4; ci.pCode = code;
    VkShaderModule m; VK_CHECK(vkCreateShaderModule(dev,&ci,nullptr,&m));
    return m;
}

} // anon

using namespace vf;

Renderer::Renderer(uint32_t w, uint32_t h):width(w),height(h)
{
    auto& gpu = ctx();
    /* ------- off-screen colour image + view --------------- */
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType=VK_IMAGE_TYPE_2D;
    ici.extent  ={w,h,1};
    ici.mipLevels=1; ici.arrayLayers=1;
    ici.format=VK_FORMAT_R8G8B8A8_UNORM;
    ici.tiling=VK_IMAGE_TILING_OPTIMAL;
    ici.usage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
              VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.samples=VK_SAMPLE_COUNT_1_BIT;
    VK_CHECK(vkCreateImage(gpu.device,&ici,nullptr,&color));

    VkMemoryRequirements req; vkGetImageMemoryRequirements(gpu.device,color,&req);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(gpu.phys,&mp);
    uint32_t type=0;
    for(uint32_t i=0;i<mp.memoryTypeCount;++i)
        if((req.memoryTypeBits&(1<<i)) &&
           (mp.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
        { type=i; break; }
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize=req.size; mai.memoryTypeIndex=type;
    VK_CHECK(vkAllocateMemory(gpu.device,&mai,nullptr,&colorMem));
    vkBindImageMemory(gpu.device,color,colorMem,0);

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image=color; vci.viewType=VK_IMAGE_VIEW_TYPE_2D;
    vci.format=VK_FORMAT_R8G8B8A8_UNORM;
    vci.subresourceRange.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.levelCount=1; vci.subresourceRange.layerCount=1;
    VK_CHECK(vkCreateImageView(gpu.device,&vci,nullptr,&colorView));

    /* ------- render pass ---------------------------------- */
    VkAttachmentDescription col{};
    col.format=VK_FORMAT_R8G8B8A8_UNORM;
    col.samples=VK_SAMPLE_COUNT_1_BIT;
    col.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR;
    col.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
    col.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;
    col.finalLayout  =VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    VkAttachmentReference colRef{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription sub{}; sub.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount=1; sub.pColorAttachments=&colRef;

    VkSubpassDependency dep{};
    dep.srcSubpass=VK_SUBPASS_EXTERNAL; dep.dstSubpass=0;
    dep.srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount=1; rpci.pAttachments=&col;
    rpci.subpassCount=1; rpci.pSubpasses=&sub;
    rpci.dependencyCount=1; rpci.pDependencies=&dep;
    VK_CHECK(vkCreateRenderPass(gpu.device,&rpci,nullptr,&pass));

    /* ------- framebuffer ---------------------------------- */
    VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fci.renderPass=pass; fci.attachmentCount=1; fci.pAttachments=&colorView;
    fci.width=w; fci.height=h; fci.layers=1;
    VK_CHECK(vkCreateFramebuffer(gpu.device,&fci,nullptr,&fb));

    /* ------- graphics pipeline ---------------------------- */
    VkShaderModule vtx = makeShader(gpu.device,VTX_SPV,sizeof(VTX_SPV)/4);
    VkShaderModule frg = makeShader(gpu.device,FRG_SPV,sizeof(FRG_SPV)/4);

    std::array<VkPipelineShaderStageCreateInfo,2> stages{};
    stages[0]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT;   stages[0].module=vtx; stages[0].pName="main";
    stages[1]=stages[0]; stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT;  stages[1].module=frg;

    /* vertex layout: vec3 pos + vec4 rgba packed tightly */
    VkVertexInputBindingDescription bind{0,sizeof(float)*7,VK_VERTEX_INPUT_RATE_VERTEX};
    std::array<VkVertexInputAttributeDescription,2> attr{};
    attr[0]={0,0,VK_FORMAT_R32G32B32_SFLOAT,0};
    attr[1]={1,0,VK_FORMAT_R32G32B32A32_SFLOAT,12};

    VkPipelineVertexInputStateCreateInfo vis{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vis.vertexBindingDescriptionCount=1; vis.pVertexBindingDescriptions=&bind;
    vis.vertexAttributeDescriptionCount=uint32_t(attr.size()); vis.pVertexAttributeDescriptions=attr.data();

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport vp{0,0,float(w),float(h),0.f,1.f};
    VkRect2D   sc{{0,0},{w,h}};
    VkPipelineViewportStateCreateInfo vpci{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vpci.viewportCount=1; vpci.pViewports=&vp;
    vpci.scissorCount =1; vpci.pScissors =&sc;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode=VK_POLYGON_MODE_FILL; rs.cullMode=VK_CULL_MODE_BACK_BIT; rs.lineWidth=1.f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount=1; cb.pAttachments=&cba;

    VkDynamicState dyns[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    ds.dynamicStateCount=2; ds.pDynamicStates=dyns;

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VK_CHECK(vkCreatePipelineLayout(gpu.device,&plci,nullptr,&pipeLayout));

    VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpci.stageCount=uint32_t(stages.size()); gpci.pStages=stages.data();
    gpci.pVertexInputState=&vis;
    gpci.pInputAssemblyState=&ia;
    gpci.pViewportState=&vpci;
    gpci.pRasterizationState=&rs;
    gpci.pMultisampleState=&ms;
    gpci.pColorBlendState=&cb;
    gpci.pDynamicState=&ds;
    gpci.layout=pipeLayout;
    gpci.renderPass=pass; gpci.subpass=0;
    VK_CHECK(vkCreateGraphicsPipelines(gpu.device,VK_NULL_HANDLE,1,&gpci,nullptr,&pipe));

    vkDestroyShaderModule(gpu.device,vtx,nullptr);
    vkDestroyShaderModule(gpu.device,frg,nullptr);
}

std::vector<uint8_t> Renderer::render(const HeightFieldScene& scene, uint32_t spp)
{
    auto& gpu = ctx();
    VkCommandBuffer cb = beginSingleTimeCmd();

    VkClearValue clear{{0.f,0.f,0.f,1.f}};
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass=pass; rbi.framebuffer=fb;
    rbi.renderArea={{0,0},{width,height}};
    rbi.clearValueCount=1; rbi.pClearValues=&clear;
    vkCmdBeginRenderPass(cb,&rbi,VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vp{0,0,float(width),float(height),0.f,1.f};
    VkRect2D sc{{0,0},{width,height}};
    vkCmdSetViewport(cb,0,1,&vp); vkCmdSetScissor(cb,0,1,&sc);

    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_GRAPHICS,pipe);
    VkDeviceSize offs=0;
    vkCmdBindVertexBuffers(cb,0,1,&scene.vbo,&offs);
    vkCmdBindIndexBuffer(cb,scene.ibo,0,VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cb,scene.nIdx,1,0,0,0);

    vkCmdEndRenderPass(cb);

    /* transition colour -> transfer src */
    VkImageMemoryBarrier bar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    bar.srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    bar.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
    bar.oldLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    bar.newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    bar.image=color;
    bar.subresourceRange.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;
    bar.subresourceRange.levelCount=1; bar.subresourceRange.layerCount=1;
    vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,0,nullptr,0,nullptr,1,&bar);

    endSingleTimeCmd(cb);

    /* host download ----------------------------------------------------- */
    VkDeviceSize bytes = width*height*4;
    VkBuffer dst; VkDeviceMemory dstMem;
    dst = allocHostVisible(bytes,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, dstMem);

    /* copy image -> buffer */
    cb = beginSingleTimeCmd();
    VkBufferImageCopy bic{};
    bic.imageSubresource.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;
    bic.imageSubresource.layerCount=1;
    bic.imageExtent = {width,height,1};
    vkCmdCopyImageToBuffer(cb, color,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           dst,1,&bic);
    endSingleTimeCmd(cb);

    void* map; vkMapMemory(gpu.device,dstMem,0,bytes,0,&map);
    std::vector<uint8_t> out(bytes);
    std::memcpy(out.data(),map,bytes);
    vkUnmapMemory(gpu.device,dstMem);

    vkDestroyBuffer(gpu.device,dst,nullptr);
    vkFreeMemory  (gpu.device,dstMem,nullptr);
    return out;
}

Renderer::~Renderer()
{
    auto& gpu = ctx();
    vkDestroyPipeline      (gpu.device,pipe,nullptr);
    vkDestroyPipelineLayout(gpu.device,pipeLayout,nullptr);
    vkDestroyFramebuffer   (gpu.device,fb,nullptr);
    vkDestroyRenderPass    (gpu.device,pass,nullptr);
    vkDestroyImageView     (gpu.device,colorView,nullptr);
    vkDestroyImage         (gpu.device,color,nullptr);
    vkFreeMemory           (gpu.device,colorMem,nullptr);
}
