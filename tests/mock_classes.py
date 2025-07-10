"""Mock classes for testing vulkan-forge without actual Vulkan implementation."""

import numpy as np
import os

class MockEngine:
    """Mock engine for testing."""
    def __init__(self):
        self.allocated_memory = 0
    
    def get_allocated_memory(self):
        return self.allocated_memory

class MockVertexBuffer:
    """Mock vertex buffer for testing."""
    def __init__(self, engine):
        self.engine = engine
        self.size = 0
    
    def upload_mesh_data(self, vertices, normals, tex_coords, indices):
        self.size = len(vertices) + len(normals) + len(tex_coords) + len(indices)
        self.engine.allocated_memory += self.size * 4  # 4 bytes per float
        return True
    
    def cleanup(self):
        self.engine.allocated_memory -= self.size * 4

class MockMeshLoader:
    """Mock mesh loader for testing."""
    def __init__(self):
        self.vertices = []
        self.indices = []
        self.groups = []
    
    def load_obj(self, filename):
        if not os.path.exists(filename):
            return False
        try:
            with open(filename, 'r') as f:
                content = f.read()
                # Simple parsing for testing
                lines = content.strip().split('\n')
                vertices = []
                indices = []
                for line in lines:
                    if line.startswith('v '):
                        parts = line.split()
                        if len(parts) >= 4:
                            vertices.extend([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif line.startswith('f '):
                        parts = line.split()
                        if len(parts) >= 4:
                            # Simple triangulation
                            indices.extend([int(parts[1])-1, int(parts[2])-1, int(parts[3])-1])
                self.vertices = vertices
                self.indices = indices
                return len(vertices) > 0
        except:
            return False
    
    def get_vertices(self):
        return self.vertices
    
    def get_indices(self):
        return self.indices
    
    def get_groups(self):
        return self.groups

class MockAllocator:
    """Mock allocator for testing."""
    def __init__(self):
        self.allocated = 0
    
    def __del__(self):
        pass

class NumpyBuffer:
    """Mock NumPy buffer for testing."""
    def __init__(self, allocator, array):
        self.allocator = allocator
        self.array = np.asarray(array)
        self.size = self.array.nbytes
        self.shape = self.array.shape
        
        # Validate array type
        if not isinstance(array, np.ndarray) and not hasattr(array, '__array__'):
            raise ValueError("Input must be array-like")
        
        # Check for unsupported dtypes
        if self.array.dtype == np.complex128:
            raise ValueError("Complex dtypes not supported")
    
    def update(self, new_data):
        new_array = np.asarray(new_data)
        if new_array.size > self.array.size:
            raise ValueError("New data too large for buffer")
        # Update would happen here
    
    def sync_to_gpu(self):
        # Mock GPU sync
        pass

class StructuredBuffer(NumpyBuffer):
    """Mock structured buffer."""
    pass

# Mock VMA classes
VULKAN_AVAILABLE = False  # Set to False to skip Vulkan tests

class MockDeviceManager:
    def __init__(self, enable_validation=False):
        self.enable_validation = enable_validation
    
    def create_logical_devices(self):
        return [MockDevice()]

class MockDevice:
    def __init__(self):
        self.physical_device = MockPhysicalDevice()
        self.device = "mock_device"

class MockPhysicalDevice:
    def __init__(self):
        self.device = "mock_physical_device"

def mock_create_allocator_native(instance, physical_device, device):
    return "mock_allocator"

def mock_allocate_buffer(allocator, size, usage):
    return ("mock_buffer", "mock_allocation")

class VertexBuffer:
    """Mock vertex buffer for testing."""
    def __init__(self, engine):
        self.engine = engine
        self.size = 0
        self.uploaded = False
        
        # Initialize allocated_memory if not present
        if not hasattr(self.engine, 'allocated_memory'):
            self.engine.allocated_memory = 0
        elif hasattr(self.engine.allocated_memory, '_mock_name'):
            # It's a Mock, replace with real value
            self.engine.allocated_memory = 0
    
    def upload_mesh_data(self, vertices, normals, tex_coords, indices):
        self.size = len(vertices) + len(normals) + len(tex_coords) + len(indices)
        self.engine.allocated_memory += self.size * 4  # 4 bytes per float
        self.uploaded = True
        return True
    
    def destroy(self):
        if self.uploaded:
            self.engine.allocated_memory = max(0, self.engine.allocated_memory - self.size * 4)
            self.uploaded = False
