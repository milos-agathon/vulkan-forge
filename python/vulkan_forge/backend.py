# vulkan_forge/backend.py
"""Device enumeration and selection for Vulkan/CPU backends."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

try:
    import vulkan as vk
except ImportError:
    class MockVK:
        VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 0
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 0
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 0
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 0
        VK_MAKE_VERSION = staticmethod(lambda major, minor, patch: (major << 22) | (minor << 12) | patch)
        VK_API_VERSION_1_2 = 4202496
        VK_API_VERSION_1_0 = 4194304
        VK_KHR_SURFACE_EXTENSION_NAME = ""
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME = ""
        VK_KHR_SWAPCHAIN_EXTENSION_NAME = ""
        VK_PHYSICAL_DEVICE_TYPE_OTHER = 0
        VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1
        VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2
        VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU = 3
        VK_PHYSICAL_DEVICE_TYPE_CPU = 4
        VK_QUEUE_GRAPHICS_BIT = 1
        
        # Mock functions
        @staticmethod
        def vkCreateInstance(*args):
            raise ImportError("Vulkan not available - using CPU fallback")
        @staticmethod
        def vkEnumeratePhysicalDevices(*args): 
            return []
        
         # Add other constants as needed
    vk = MockVK()

logger = logging.getLogger(__name__)

class VulkanForgeError(Exception):
    """Exception wrapper for Vulkan API failures."""
    
    def __init__(self, message: str, vk_result: Optional[int] = None):
        """Initialize with error message and optional vkResult code."""
        super().__init__(message)
        self.vk_result = vk_result
        if vk_result is not None:
            logger.error(f"Vulkan error: {message} (vkResult: {vk_result})")


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
        self._create_instance()
        self._enumerate_devices()
    
    def _create_instance(self) -> None:
        """Create Vulkan instance with validation layers if requested."""
        # Use VK_API_VERSION_1_0 for compatibility with older vulkan bindings
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
        
        # Sort devices by preference
        sorted_devices = sorted(
            self.physical_devices,
            key=lambda d: (d.is_discrete_gpu, d.is_integrated_gpu, not d.is_cpu),
            reverse=True
        )
        
        for physical_device in sorted_devices:
            try:
                logical_device = self._create_logical_device(physical_device)
                self.logical_devices.append(logical_device)
            except VulkanForgeError as e:
                logger.warning(f"Failed to create logical device: {e}")
                continue
        
        if not self.logical_devices:
            raise VulkanForgeError("Failed to create any logical devices")
        
        return self.logical_devices
    
    def _create_logical_device(self, physical_device: PhysicalDeviceInfo) -> LogicalDevice:
        """Create a logical device from a physical device."""
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
        for logical_device in self.logical_devices:
            if logical_device.command_pool:
                vk.vkDestroyCommandPool(logical_device.device, logical_device.command_pool, None)
            if logical_device.device:
                vk.vkDestroyDevice(logical_device.device, None)
        
        if self.instance:
            vk.vkDestroyInstance(self.instance, None)
        
        self.logical_devices.clear()
        self.physical_devices.clear()