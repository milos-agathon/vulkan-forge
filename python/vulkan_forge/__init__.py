# File: python/vulkan_forge/__init__.py
"""Python bindings for Vulkan Forge with optional native acceleration."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import warnings

spec = importlib.util.find_spec(".vulkan_forge_native", __name__)
if spec and isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
    vulkan_forge_native = importlib.import_module(
        ".vulkan_forge_native", __name__
    )
else:
    warnings.warn(
        "vulkan_forge_native extension not found; using stubs.",
        RuntimeWarning,
    )
    from . import vulkan_forge_native

# Use relative imports to avoid system package conflicts
from .backend import DeviceManager, VulkanForgeError
from .renderer import Renderer, create_renderer
from .matrices import Matrix4x4
from . import backend
from . import renderer
from . import matrices

# Re-export the main components
DeviceManager = backend.DeviceManager
VulkanForgeError = backend.VulkanForgeError
Renderer = renderer.Renderer
create_renderer = renderer.create_renderer
Matrix4x4 = matrices.Matrix4x4
 
__version__ = "0.1.0"
__all__ = ["DeviceManager", "VulkanForgeError", "Renderer", "create_renderer", "Matrix4x4"]
