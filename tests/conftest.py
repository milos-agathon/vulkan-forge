"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import sys
from pathlib import Path
from .mock_classes import (
    MockEngine, MockVertexBuffer, MockMeshLoader, MockAllocator,
    NumpyBuffer, StructuredBuffer
)

# Add the python directory to the path for imports
test_dir = Path(__file__).parent
project_root = test_dir.parent
python_dir = project_root / "python"
sys.path.insert(0, str(python_dir))

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

@pytest.fixture
def allocator():
    """Provide a mock allocator for testing."""
    return MockAllocator()