"""CLI tests conftest."""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from test_mocks import *
except ImportError:
    # Create minimal inline mocks if test_mocks not available
    import numpy as np
    
    class GeoTiffLoader:
        def load(self, path): return False
        def get_data(self): return np.zeros((10,10)), {}
    
    class TerrainCache:
        def __init__(self, **kwargs): pass
        def get_statistics(self): return {}

@pytest.fixture
def cache_config():
    return {
        'eviction_policy': 'lru',
        'max_tiles': 64,
        'tile_size': 256
    }
