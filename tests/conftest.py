"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Import all mock classes
from .test_mocks import *

# Add the python directory to the path for imports
test_dir = Path(__file__).parent
project_root = test_dir.parent
python_dir = project_root / "python"
sys.path.insert(0, str(python_dir))


@pytest.fixture
def engine():
    return MockEngine()


@pytest.fixture
def vertex_buffer(engine):
    return MockVertexBuffer(engine)


@pytest.fixture
def mesh_loader():
    return MockMeshLoader()


@pytest.fixture
def allocator():
    return MockAllocator()


@pytest.fixture
def sample_config():
    return TerrainConfig()


@pytest.fixture
def cache_config():
    # Fixed: Remove max_size_mb parameter
    return {"eviction_policy": "lru", "max_tiles": 64, "tile_size": 256}


@pytest.fixture
def vulkan_context():
    from unittest.mock import Mock

    return Mock()


@pytest.fixture
def mock_vulkan_context(vulkan_context):
    """Alias expected by legacy CLI tests."""
    return vulkan_context


@pytest.fixture
def shader_compiler():
    return ShaderCompiler()


@pytest.fixture
def shader_templates():
    return TerrainShaderTemplates()
