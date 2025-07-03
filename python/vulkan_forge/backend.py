# vulkan_forge/backend.py
"""Device enumeration and selection for Vulkan/CPU backends."""

import logging
import weakref
import ctypes
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

try:
    from . import vulkan_forge_native as native
except Exception:  # pragma: no cover - native module optional
    native = None

try:
    import vulkan as vk
    VULKAN_AVAILABLE = True
except ImportError:
    VULKAN_AVAILABLE = False
    # Mock Vulkan constants and functions for CPU fallback
    class MockVK:
        VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 0
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 0
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 0
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 0
        VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2 = 1000109001
        VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2 = 1000109002
        VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2 = 1000109003
        VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2 = 1000109004
        
        VK_MAKE_VERSION = staticmethod(lambda major, minor, patch: (major << 22) | (minor << 12) | patch)
        VK_API_VERSION_1_2 = 4202496
        VK_API_VERSION_1_0 = 4194304
        VK_SUCCESS = 0
        
        VK_KHR_SURFACE_EXTENSION_NAME = "VK_KHR_surface"
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME = "VK_EXT_debug_utils"
        VK_KHR_SWAPCHAIN_EXTENSION_NAME = "VK_KHR_swapchain"
        
        VK_PHYSICAL_DEVICE_TYPE_OTHER = 0
        VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1
        VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2
        VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU = 3
        VK_PHYSICAL_DEVICE_TYPE_CPU = 4
        
        VK_QUEUE_GRAPHICS_BIT = 1
        VK_SAMPLE_COUNT_1_BIT = 1
        VK_ATTACHMENT_LOAD_OP_CLEAR = 0
        VK_ATTACHMENT_STORE_OP_STORE = 0
        VK_ATTACHMENT_LOAD_OP_DONT_CARE = 1
        VK_ATTACHMENT_STORE_OP_DONT_CARE = 1
        VK_IMAGE_LAYOUT_UNDEFINED = 0
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR = 1000001002
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL = 1000001000
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL = 1000001001
        VK_FORMAT_D32_SFLOAT = 126
        VK_FORMAT_B8G8R8A8_UNORM = 44
        VK_IMAGE_ASPECT_COLOR_BIT = 1
        VK_IMAGE_ASPECT_DEPTH_BIT = 2
        VK_PIPELINE_BIND_POINT_GRAPHICS = 0
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 2

        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT = 0x00000001
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020
        
        # Mock classes for Vulkan structures
        class VkApplicationInfo:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class VkInstanceCreateInfo:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class VkDeviceQueueCreateInfo:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class VkDeviceCreateInfo:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class VkCommandPoolCreateInfo:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class VkAttachmentDescription2:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class VkAttachmentReference2:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class VkSubpassDescription2:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class VkRenderPassCreateInfo2:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class VkRenderPass:
            def __init__(self, handle=0):
                self.handle = handle
        
        # Mock functions
        @staticmethod
        def vkCreateInstance(*args):
            if VULKAN_AVAILABLE:
                raise ImportError("Vulkan not available - using CPU fallback")
            return "mock_instance"
        
        @staticmethod
        def vkEnumeratePhysicalDevices(*args):
            return []
        
        @staticmethod
        def vkGetPhysicalDeviceProperties(*args):
            return None
        
        @staticmethod
        def vkGetPhysicalDeviceFeatures(*args):
            return None
        
        @staticmethod
        def vkGetPhysicalDeviceMemoryProperties(*args):
            return None
        
        @staticmethod
        def vkGetPhysicalDeviceQueueFamilyProperties(*args):
            return []
        
        @staticmethod
        def vkCreateDevice(*args):
            return "mock_device"
        
        @staticmethod
        def vkGetDeviceQueue(*args):
            return "mock_queue"
        
        @staticmethod
        def vkCreateCommandPool(*args):
            return "mock_command_pool"
        
        @staticmethod
        def vkCreateRenderPass2(*args):
            return MockVK.VK_SUCCESS
        
        @staticmethod
        def vkDestroyCommandPool(*args):
            pass
        
        @staticmethod
        def vkDestroyDevice(*args):
            pass
        
        @staticmethod
        def vkDestroyInstance(*args):
            pass
    
    vk = MockVK()

BUFFER_USAGE_VERTEX = vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
BUFFER_USAGE_STORAGE = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

logger = logging.getLogger(__name__)

class VulkanForgeError(Exception):
    """Exception wrapper for Vulkan API failures."""
    
    def __init__(self, message: str, vk_result: Optional[int] = None):
        """Initialize with error message and optional vkResult code."""
        super().__init__(message)
        self.vk_result = vk_result
        if vk_result is not None:
            logger.error(f"Vulkan error: {message} (vkResult: {vk_result})")

if native is not None and hasattr(native, "VulkanForgeError"):
    VulkanForgeError = native.VulkanForgeError  # type: ignore[misc]

_capsule_get_ptr = ctypes.pythonapi.PyCapsule_GetPointer
_capsule_get_ptr.restype = ctypes.c_void_p
_capsule_get_ptr.argtypes = [ctypes.py_object, ctypes.c_char_p]

def _capsule_to_void_p(cap: object, name: str) -> ctypes.c_void_p:
    """Extract raw pointer from a PyCapsule."""
    return ctypes.c_void_p(_capsule_get_ptr(cap, name.encode()))


@dataclass
class PhysicalDeviceInfo:
    """Information about a Vulkan physical device."""
    
    device: Any  # VkPhysicalDevice
    properties: Any  # VkPhysicalDeviceProperties
    features: Any  # VkPhysicalDeviceFeatures
    memory_properties: Any  # VkPhysicalDeviceMemoryProperties
    queue_families: List[Any]  # List[VkQueueFamilyProperties]
    device_type: int  # VkPhysicalDeviceType
    
    @property
    def is_discrete_gpu(self) -> bool:
        """Check if this is a discrete GPU."""
        return self.device_type == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    
    @property
    def is_integrated_gpu(self) -> bool:
        """Check if this is an integrated GPU."""
        return self.device_type == vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
    
    @property
    def is_cpu(self) -> bool:
        """Check if this is a CPU device."""
        return self.device_type == vk.VK_PHYSICAL_DEVICE_TYPE_CPU


@dataclass
class LogicalDevice:
    """Wrapper for a Vulkan logical device and associated resources."""
    
    physical_device: PhysicalDeviceInfo
    device: Any  # VkDevice
    graphics_queue: Any  # VkQueue
    graphics_queue_family_index: int
    command_pool: Any  # VkCommandPool


class DeviceManager:
    """Manage Vulkan instance and device enumeration."""
    
    def __init__(self, app_name: str = "VulkanForge", enable_validation: bool = True):
        """Initialize the device manager."""
        self.app_name = app_name
        self.enable_validation = enable_validation
        self.instance: Optional[Any] = None  # VkInstance
        self.physical_devices: List[PhysicalDeviceInfo] = []
        self.logical_devices: List[LogicalDevice] = []
        
        if not VULKAN_AVAILABLE:
            logger.warning("Vulkan not available, creating CPU fallback device")
            self._create_cpu_fallback_device()
        else:
            try:
                self._create_instance()
                self._enumerate_devices()
            except Exception as e:
                logger.warning(f"Failed to initialize Vulkan: {e}, creating CPU fallback")
                self._create_cpu_fallback_device()
    
    def _create_cpu_fallback_device(self) -> None:
        """Create a CPU fallback device when Vulkan is not available."""
        # Create mock properties
        class MockProperties:
            def __init__(self):
                self.deviceName = "CPU Fallback Renderer"
                self.deviceType = vk.VK_PHYSICAL_DEVICE_TYPE_CPU
        
        class MockFeatures:
            pass
        
        class MockMemoryProperties:
            pass
        
        # Create a CPU device info
        cpu_device_info = PhysicalDeviceInfo(
            device="cpu_device",
            properties=MockProperties(),
            features=MockFeatures(),
            memory_properties=MockMemoryProperties(),
            queue_families=[],
            device_type=vk.VK_PHYSICAL_DEVICE_TYPE_CPU
        )
        
        self.physical_devices = [cpu_device_info]
        logger.info("Created CPU fallback device")
    
    def _create_instance(self) -> None:
        """Create Vulkan instance with validation layers if requested."""
        if not VULKAN_AVAILABLE:
            raise VulkanForgeError("Vulkan not available")
            
        # Use VK_API_VERSION_1_0 for compatibility
        api_version = getattr(vk, 'VK_API_VERSION_1_2', None)
        if api_version is None:
            api_version = getattr(vk, 'VK_API_VERSION_1_0', vk.VK_MAKE_VERSION(1, 0, 0))

        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName=self.app_name,
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="VulkanForge",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=api_version
        )
        
        extensions = [vk.VK_KHR_SURFACE_EXTENSION_NAME]
        layers = []
        
        if self.enable_validation:
            layers.append("VK_LAYER_KHRONOS_validation")
            extensions.append(vk.VK_EXT_DEBUG_UTILS_EXTENSION_NAME)
        
        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
            enabledLayerCount=len(layers),
            ppEnabledLayerNames=layers,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions
        )
        
        try:
            self.instance = vk.vkCreateInstance(create_info, None)
        except Exception as e:
            raise VulkanForgeError(f"Failed to create Vulkan instance: {e}")
    
    def _enumerate_devices(self) -> None:
        """Enumerate all available physical devices."""
        if not self.instance:
            raise VulkanForgeError("Instance not created")
        
        try:
            physical_devices = vk.vkEnumeratePhysicalDevices(self.instance)
        except Exception as e:
            raise VulkanForgeError(f"Failed to enumerate physical devices: {e}")
        
        for device in physical_devices:
            properties = vk.vkGetPhysicalDeviceProperties(device)
            features = vk.vkGetPhysicalDeviceFeatures(device)
            memory_properties = vk.vkGetPhysicalDeviceMemoryProperties(device)
            queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(device)
            
            device_info = PhysicalDeviceInfo(
                device=device,
                properties=properties,
                features=features,
                memory_properties=memory_properties,
                queue_families=queue_families,
                device_type=properties.deviceType
            )
            
            self.physical_devices.append(device_info)
            logger.info(f"Found device: {properties.deviceName} (Type: {properties.deviceType})")
    
    def create_logical_devices(self, prefer_discrete: bool = True) -> List[LogicalDevice]:
        """Create logical devices for all suitable physical devices."""
        if not self.physical_devices:
            raise VulkanForgeError("No physical devices found")
        
        # If we only have CPU devices, return them directly
        if all(d.is_cpu for d in self.physical_devices):
            logger.info("Only CPU devices available, using CPU renderer")
            return []
        
        # Sort devices by preference
        sorted_devices = sorted(
            self.physical_devices,
            key=lambda d: (d.is_discrete_gpu, d.is_integrated_gpu, not d.is_cpu),
            reverse=True
        )
        
        for physical_device in sorted_devices:
            if physical_device.is_cpu:
                continue  # Skip CPU devices for logical device creation
                
            try:
                logical_device = self._create_logical_device(physical_device)
                self.logical_devices.append(logical_device)
            except VulkanForgeError as e:
                logger.warning(f"Failed to create logical device: {e}")
                continue
        
        return self.logical_devices
    
    def _create_logical_device(self, physical_device: PhysicalDeviceInfo) -> LogicalDevice:
        """Create a logical device from a physical device."""
        if not VULKAN_AVAILABLE:
            raise VulkanForgeError("Vulkan not available")
            
        # Find graphics queue family
        graphics_queue_family = None
        for i, queue_family in enumerate(physical_device.queue_families):
            if queue_family.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT:
                graphics_queue_family = i
                break
        
        if graphics_queue_family is None:
            raise VulkanForgeError("No graphics queue family found")
        
        queue_priority = [1.0]
        queue_create_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=graphics_queue_family,
            queueCount=1,
            pQueuePriorities=queue_priority
        )
        
        device_extensions = [vk.VK_KHR_SWAPCHAIN_EXTENSION_NAME]
        
        device_create_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_create_info],
            enabledExtensionCount=len(device_extensions),
            ppEnabledExtensionNames=device_extensions,
            pEnabledFeatures=physical_device.features
        )
        
        try:
            device = vk.vkCreateDevice(physical_device.device, device_create_info, None)
        except Exception as e:
            raise VulkanForgeError(f"Failed to create logical device: {e}")
        
        # Get graphics queue
        graphics_queue = vk.vkGetDeviceQueue(device, graphics_queue_family, 0)
        
        # Create command pool
        command_pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=graphics_queue_family
        )
        
        try:
            command_pool = vk.vkCreateCommandPool(device, command_pool_info, None)
        except Exception as e:
            vk.vkDestroyDevice(device, None)
            raise VulkanForgeError(f"Failed to create command pool: {e}")
        
        return LogicalDevice(
            physical_device=physical_device,
            device=device,
            graphics_queue=graphics_queue,
            graphics_queue_family_index=graphics_queue_family,
            command_pool=command_pool
        )
    
    def cleanup(self) -> None:
        """Clean up all Vulkan resources."""
        if not VULKAN_AVAILABLE:
            return
            
        for logical_device in self.logical_devices:
            if logical_device.command_pool:
                vk.vkDestroyCommandPool(logical_device.device, logical_device.command_pool, None)
            if logical_device.device:
                vk.vkDestroyDevice(logical_device.device, None)
        
        if self.instance:
            vk.vkDestroyInstance(self.instance, None)

        self.logical_devices.clear()
        self.physical_devices.clear()


def create_allocator_native(instance: Any, physical_device: Any, device: Any) -> Optional[ctypes.c_void_p]:
    """Create a VMA allocator (native binding)."""
    if native is None or not hasattr(native, "create_allocator"):
        logger.warning("VMA allocator not available - native extension required")
        return None
    
    try:
        cap = native.create_allocator(int(instance), int(physical_device), int(device))
        ptr = _capsule_to_void_p(cap, "VmaAllocator")
        weakref.finalize(ptr, destroy_allocator, ptr)
        return ptr
    except Exception as e:
        logger.warning(f"Failed to create VMA allocator: {e}")
        return None


def create_allocator() -> Optional[ctypes.c_void_p]:
    """Convenience wrapper that creates a default allocator."""
    dm = DeviceManager(enable_validation=False)
    devices = dm.create_logical_devices()
    if not devices:
        return None
    dev = devices[0]
    return create_allocator_native(dm.instance, dev.physical_device.device, dev.device)


def allocate_buffer(allocator: Optional[ctypes.c_void_p], size: int, usage: int) -> Tuple[Optional[ctypes.c_void_p], Optional[ctypes.c_void_p]]:
    """Allocate a Vulkan buffer via VMA."""
    if native is None or not hasattr(native, "allocate_buffer") or allocator is None:
        logger.warning("Buffer allocation not available - native extension required")
        return None, None
    
    try:
        buf, alloc = native.allocate_buffer(int(allocator.value), size, usage)
        return ctypes.c_void_p(buf), ctypes.c_void_p(alloc)
    except Exception as e:
        logger.warning(f"Failed to allocate buffer: {e}")
        return None, None


def destroy_allocator(allocator: ctypes.c_void_p) -> None:
    """Destroy a VMA allocator."""
    if native is None or not hasattr(native, "destroy_allocator") or allocator is None:
        return
    
    try:
        native.destroy_allocator(int(allocator.value))
    except Exception as e:
        logger.warning(f"Failed to destroy allocator: {e}")


__all__ = [
    "DeviceManager",
    "VulkanForgeError",
    "LogicalDevice",
    "PhysicalDeviceInfo",
    "create_allocator_native",
    "create_allocator",
    "allocate_buffer",
    "destroy_allocator",
    "BUFFER_USAGE_VERTEX",
    "BUFFER_USAGE_STORAGE",
]
