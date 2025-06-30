# File: python/vulkan_forge/__init__.py
# Use relative imports to avoid system package
from .backend import DeviceManager, VulkanForgeError
from .renderer import Renderer, create_renderer
from .matrices import Matrix4x4
# Use relative imports to avoid system package conflicts
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