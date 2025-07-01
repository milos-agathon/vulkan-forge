# File: python/vulkan_forge/__init__.py
"""Python bindings for Vulkan Forge with optional native acceleration."""

from __future__ import annotations

import warnings
from importlib import import_module

try:  # Attempt to import the compiled extension first
    vulkan_forge_native = import_module("._vulkan_forge_native", __name__)
except Exception:  # pragma: no cover - optional native module missing
    warnings.warn(
        "vulkan_forge_native extension not found; using stubs.",
        RuntimeWarning,
    )
    vulkan_forge_native = import_module(".vulkan_forge_native", __name__)

# Use relative imports to avoid system package conflicts
from .backend import DeviceManager, VulkanForgeError
from .renderer import Renderer, create_renderer
from .matrices import Matrix4x4

# Re-export the main components
DeviceManager = DeviceManager
VulkanForgeError = VulkanForgeError
Renderer = Renderer
create_renderer = create_renderer
Matrix4x4 = Matrix4x4

__version__ = "0.1.0"
__all__ = [
    "DeviceManager",
    "VulkanForgeError",
    "Renderer",
    "create_renderer",
    "Matrix4x4",
]
