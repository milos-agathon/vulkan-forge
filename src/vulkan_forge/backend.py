"""Device enumeration and selection utilities."""

from __future__ import annotations

import logging
import types
from dataclasses import dataclass
from typing import Any, List, Optional

try:
    import vulkan as vk  # type: ignore
except Exception:  # pragma: no cover - library not available
    class _VKStub:
        VK_PHYSICAL_DEVICE_TYPE_OTHER = 0
        VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1
        VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2
        VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU = 3
        VK_PHYSICAL_DEVICE_TYPE_CPU = 4

        VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 0

        VK_MAKE_VERSION = staticmethod(lambda major, minor, patch: 0)

        VK_SUCCESS = 0

        def __getattr__(self, name: str) -> Any:
            def _stub(*_args: Any, **kwargs: Any) -> Any:
                return types.SimpleNamespace(**kwargs)

            return _stub

    vk = _VKStub()

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Information about a Vulkan physical device."""

    handle: Optional[Any]
    properties: Optional[Any]
    device_type: int


def _create_instance() -> Any:
    """Create a minimal Vulkan instance."""
    app_info = vk.VkApplicationInfo(
        sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName="vulkan-forge",
        applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        pEngineName="vulkan-forge",
        engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        apiVersion=vk.VK_MAKE_VERSION(1, 0, 0),
    )
    info = vk.VkInstanceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=app_info,
        enabledExtensionCount=0,
        ppEnabledExtensionNames=None,
    )
    return vk.vkCreateInstance(info, None)


def _enumerate_devices(instance: Any) -> List[DeviceInfo]:
    """Enumerate available physical devices."""
    try:
        handles = vk.vkEnumeratePhysicalDevices(instance)
    except Exception as exc:  # pragma: no cover - depends on system
        logger.warning("Failed to enumerate physical devices: %s", exc)
        return []

    devices: List[DeviceInfo] = []
    for handle in handles:
        props = vk.vkGetPhysicalDeviceProperties(handle)
        devices.append(DeviceInfo(handle=handle, properties=props, device_type=props.deviceType))
    return devices


def select_device() -> DeviceInfo:
    """Select the best physical device or CPU fallback."""
    try:
        instance = _create_instance()
    except Exception as exc:  # pragma: no cover - depends on system
        logger.warning("Failed to create Vulkan instance: %s", exc)
        return DeviceInfo(handle=None, properties=None, device_type=vk.VK_PHYSICAL_DEVICE_TYPE_CPU)

    devices = _enumerate_devices(instance)
    if not devices:
        return DeviceInfo(handle=None, properties=None, device_type=vk.VK_PHYSICAL_DEVICE_TYPE_CPU)

    # Prefer discrete GPU
    gpu = next((d for d in devices if d.device_type == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU), None)
    if gpu:
        return gpu
    return devices[0]
