#include "vf/vk_common.hpp"
#include <stdexcept>
#include <cstring>      // std::memcpy
#include <array>
#include <iostream>

#define VK_CHECK(call)                                           \
    do {                                                         \
        VkResult _res = call;                                    \
        if (_res != VK_SUCCESS)                                  \
            throw std::runtime_error("Vulkan error " #call);     \
    } while (0)

namespace vf {

static GpuContext g_ctx;

/* ─────────────────────────  helpers  ───────────────────────── */

static uint32_t pick_graphics_queue(VkPhysicalDevice dev)
{
    uint32_t nQ; vkGetPhysicalDeviceQueueFamilyProperties(dev, &nQ, nullptr);
    std::vector<VkQueueFamilyProperties> props(nQ);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &nQ, props.data());

    for (uint32_t i = 0; i < nQ; ++i)
        if (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) return i;
    throw std::runtime_error("No graphics queue");
}

static VkPhysicalDevice pick_device(VkInstance inst)
{
    uint32_t nDev; VK_CHECK(vkEnumeratePhysicalDevices(inst, &nDev, nullptr));
    if (!nDev) throw std::runtime_error("No Vulkan device found");
    std::vector<VkPhysicalDevice> devs(nDev);
    VK_CHECK(vkEnumeratePhysicalDevices(inst, &nDev, devs.data()));

    /* choose first discrete, else first */
    for (auto d: devs) {
        VkPhysicalDeviceProperties p{}; vkGetPhysicalDeviceProperties(d,&p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) return d;
    }
    return devs[0];
}

/* ─────────────────────────  GpuContext singleton  ───────────────────────── */

GpuContext& ctx()
{
    static bool first = true;
    if (!first) return g_ctx;
    first = false;

    /* 1. instance -------------------------------------------------------- */
    const std::array<const char*,1> inst_ext = { VK_KHR_SURFACE_EXTENSION_NAME };
    VkApplicationInfo ai { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    ai.pApplicationName = "vulkan-forge"; ai.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ci { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ci.pApplicationInfo  = &ai;
    ci.enabledExtensionCount   = uint32_t(inst_ext.size());
    ci.ppEnabledExtensionNames = inst_ext.data();
    VK_CHECK(vkCreateInstance(&ci,nullptr,&g_ctx.instance));

    /* 2. physical + logical device -------------------------------------- */
    g_ctx.phys = pick_device(g_ctx.instance);
    g_ctx.queueFamily = pick_graphics_queue(g_ctx.phys);

    float qprio = 1.f;
    VkDeviceQueueCreateInfo qci { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = g_ctx.queueFamily;
    qci.queueCount = 1; qci.pQueuePriorities = &qprio;

    VkPhysicalDeviceFeatures feats{};      // keep default
    VkDeviceCreateInfo dci { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.pEnabledFeatures = &feats;
    VK_CHECK(vkCreateDevice(g_ctx.phys,&dci,nullptr,&g_ctx.device));

    vkGetDeviceQueue(g_ctx.device,g_ctx.queueFamily,0,&g_ctx.queue);

    /* 3. one command-pool for short-lived commands ----------------------- */
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex=g_ctx.queueFamily;
    VK_CHECK(vkCreateCommandPool(g_ctx.device,&pci,nullptr,&g_ctx.cmdPool));

    return g_ctx;
}

/* ─────────────────────  single-time command helpers  ───────────────────── */

VkCommandBuffer beginSingleTimeCmd()
{
    auto& gpu = ctx();
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandPool = gpu.cmdPool; ai.commandBufferCount = 1;
    VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(gpu.device,&ai,&cb));

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb,&bi));
    return cb;
}

void endSingleTimeCmd(VkCommandBuffer cb)
{
    auto& gpu = ctx();
    VK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount=1; si.pCommandBuffers=&cb;
    VK_CHECK(vkQueueSubmit(gpu.queue,1,&si,VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle (gpu.queue));
    vkFreeCommandBuffers(gpu.device,gpu.cmdPool,1,&cb);
}

/* ─────────────────────  buffer helpers  ───────────────────── */

static void allocBuffer(VkDeviceSize bytes, VkBufferUsageFlags usage,
                        VkMemoryPropertyFlags memFlags,
                        VkBuffer& buf, VkDeviceMemory& mem)
{
    auto& gpu = ctx();
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes; bci.usage = usage; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(gpu.device,&bci,nullptr,&buf));

    VkMemoryRequirements req; vkGetBufferMemoryRequirements(gpu.device,buf,&req);

    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(gpu.phys,&mp);
    uint32_t type = 0;
    for(uint32_t i=0;i<mp.memoryTypeCount;++i)
        if ((req.memoryTypeBits&(1<<i)) && (mp.memoryTypes[i].propertyFlags&memFlags)==memFlags)
        { type=i; break; }

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size; mai.memoryTypeIndex = type;
    VK_CHECK(vkAllocateMemory(gpu.device,&mai,nullptr,&mem));
    vkBindBufferMemory(gpu.device,buf,mem,0);
}

VkBuffer allocHostVisible(size_t bytes, VkBufferUsageFlags usage, VkDeviceMemory& mem)
{
    VkBuffer buf;
    allocBuffer(bytes, usage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                buf, mem);
    return buf;
}

VkBuffer allocDeviceLocal(size_t bytes, VkBufferUsageFlags usage, VkDeviceMemory& mem)
{
    VkBuffer buf;
    allocBuffer(bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buf, mem);
    return buf;
}

void uploadToBuffer(VkBuffer dst, const void* src, size_t bytes)
{
    auto& gpu = ctx();
    VkDeviceMemory stagingMem;
    VkBuffer staging = allocHostVisible(bytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stagingMem);

    void* map; vkMapMemory(gpu.device,stagingMem,0,bytes,0,&map);
    std::memcpy(map,src,bytes);
    vkUnmapMemory(gpu.device,stagingMem);

    VkCommandBuffer cb = beginSingleTimeCmd();
    VkBufferCopy copy{0,0,bytes};
    vkCmdCopyBuffer(cb, staging,dst,1,&copy);
    endSingleTimeCmd(cb);

    vkDestroyBuffer(gpu.device, staging,nullptr);
    vkFreeMemory   (gpu.device, stagingMem,nullptr);
}

} // namespace vf
