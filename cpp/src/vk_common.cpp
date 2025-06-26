/* cpp/src/vk_common.cpp — global Vulkan helpers & utilities
 * -------------------------------------------------------- */
#include "vf/vk_common.hpp"

#include <vector>
#include <cstring>
#include <stdexcept>

#define VK_CHECK(call) \
    do { VkResult _r = (call); if (_r != VK_SUCCESS) throw std::runtime_error("Vulkan error"); } while (0)

namespace vf
{
/* ──────────────────────────────────────────────────────────── */
/*                          Global state                       */
/* ──────────────────────────────────────────────────────────── */
GpuContext g_ctx;

/* ----------------------------------------------------------- */
/*          Lazy one-time Vulkan initialisation                */
/* ----------------------------------------------------------- */
static void initOnce()
{
    static bool done = false;
    if (done) return;
    done = true;

    /* — Create instance — */
    VkApplicationInfo ai{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    ai.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &ai;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &g_ctx.instance));

    /* — Pick first physical device — */
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(g_ctx.instance, &count, nullptr);
    if (count == 0) throw std::runtime_error("No Vulkan devices found");

    std::vector<VkPhysicalDevice> phys(count);
    vkEnumeratePhysicalDevices(g_ctx.instance, &count, phys.data());
    g_ctx.phys = phys[0];

    /* — Find graphics-capable queue family — */
    vkGetPhysicalDeviceQueueFamilyProperties(g_ctx.phys, &count, nullptr);
    std::vector<VkQueueFamilyProperties> qprops(count);
    vkGetPhysicalDeviceQueueFamilyProperties(g_ctx.phys, &count, qprops.data());
    for (uint32_t i = 0; i < count; ++i)
        if (qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            g_ctx.gfxFamily = i;
            break;
        }

    /* — Create logical device & graphics queue — */
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = g_ctx.gfxFamily;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &prio;

    VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos    = &qci;
    VK_CHECK(vkCreateDevice(g_ctx.phys, &dci, nullptr, &g_ctx.device));

    vkGetDeviceQueue(g_ctx.device, g_ctx.gfxFamily, 0, &g_ctx.graphicsQ);
}

/* Make sure initOnce() runs before main() */
static struct _AutoInit { _AutoInit() { initOnce(); } } _auto;

/* ──────────────────────────────────────────────────────────── */
/*            One-shot command-buffer helpers                  */
/* ──────────────────────────────────────────────────────────── */
static VkCommandPool s_pool = VK_NULL_HANDLE;

static VkCommandPool getPool()
{
    if (s_pool) return s_pool;

    VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = g_ctx.gfxFamily;
    VK_CHECK(vkCreateCommandPool(g_ctx.device, &pci, nullptr, &s_pool));
    return s_pool;
}

VkCommandBuffer beginSingleTimeCmd()
{
    VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool        = getPool();
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(g_ctx.device, &ai, &cb));

    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    return cb;
}

void endSingleTimeCmd(VkCommandBuffer cb)
{
    vkEndCommandBuffer(cb);

    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cb;

    VK_CHECK(vkQueueSubmit(g_ctx.graphicsQ, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(g_ctx.graphicsQ));

    vkFreeCommandBuffers(g_ctx.device, getPool(), 1, &cb);
}

/* ──────────────────────────────────────────────────────────── */
/*               Memory-type chooser (exported)                */
/* ──────────────────────────────────────────────────────────── */
uint32_t chooseType(uint32_t bits, VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(g_ctx.phys, &mp);

    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;

    throw std::runtime_error("No compatible memory type found");
}

/* ──────────────────────────────────────────────────────────── */
/*                  Buffer helper utilities                    */
/* ──────────────────────────────────────────────────────────── */
static VkBuffer makeBuf(VkDeviceSize bytes,
                        VkBufferUsageFlags usage,
                        VkMemoryPropertyFlags props,
                        VkDeviceMemory& memOut)
{
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size  = bytes;
    bci.usage = usage;

    VkBuffer buf;
    VK_CHECK(vkCreateBuffer(g_ctx.device, &bci, nullptr, &buf));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g_ctx.device, buf, &req);

    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = chooseType(req.memoryTypeBits, props);
    VK_CHECK(vkAllocateMemory(g_ctx.device, &mai, nullptr, &memOut));

    vkBindBufferMemory(g_ctx.device, buf, memOut, 0);
    return buf;
}

VkBuffer allocDeviceLocal(VkDeviceSize bytes,
                          VkBufferUsageFlags usage,
                          VkDeviceMemory& mem)
{
    return makeBuf(bytes,
                   usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                   mem);
}

VkBuffer allocHostVisible(VkDeviceSize bytes,
                          VkBufferUsageFlags usage,
                          VkDeviceMemory& mem)
{
    return makeBuf(bytes,
                   usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   mem);
}

void uploadToBuffer(VkBuffer dst, const void* src, VkDeviceSize bytes)
{
    VkDeviceMemory stagingMem;
    VkBuffer staging = allocHostVisible(bytes,
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                        stagingMem);

    /* Copy user data into staging buffer */
    void* map = nullptr;
    vkMapMemory(g_ctx.device, stagingMem, 0, bytes, 0, &map);
    std::memcpy(map, src, static_cast<size_t>(bytes));
    vkUnmapMemory(g_ctx.device, stagingMem);

    /* Copy from staging into device-local buffer */
    VkCommandBuffer cb = beginSingleTimeCmd();
    VkBufferCopy region{ 0, 0, bytes };
    vkCmdCopyBuffer(cb, staging, dst, 1, &region);
    endSingleTimeCmd(cb);

    vkDestroyBuffer(g_ctx.device, staging, nullptr);
    vkFreeMemory  (g_ctx.device, stagingMem, nullptr);
}

} // namespace vf
