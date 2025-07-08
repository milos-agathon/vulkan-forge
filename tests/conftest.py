"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import os
import sys
from pathlib import Path

# Add the python directory to the path for imports
test_dir = Path(__file__).parent
project_root = test_dir.parent
python_dir = project_root / "python"
sys.path.insert(0, str(python_dir))

# Mock vulkan_forge for testing when not built
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

@pytest.fixture
def engine():
    """Provide a mock engine for testing."""
    return MockEngine()

@pytest.fixture
def vertex_buffer(engine):
    """Provide a mock vertex buffer for testing."""
    return MockVertexBuffer(engine)

@pytest.fixture
def mesh_loader():
    """Provide a mock mesh loader for testing."""
    return MockMeshLoader()