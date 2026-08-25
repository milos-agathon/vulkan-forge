"""P3: COG streaming tests.

Tests for Cloud Optimized GeoTIFF streaming functionality including:
- P3.1: Range reads
- P3.2: Overview detection  
- P3.3: Cache eviction
- Remote COG integration tests (network-dependent)
"""

from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path
from _cog_http_fixtures import _serve_range

PROJECT_ROOT = Path(__file__).parent.parent

AVAILABLE_DEMS = {
    "fuji": PROJECT_ROOT / "assets/tif/Mount_Fuji_30m.tif",
    "rainier": PROJECT_ROOT / "assets/tif/dem_rainier.tif",
}
REMOTE_COG_ASSET = PROJECT_ROOT / "assets/tif/moon_south_pole_lola.tif"

def get_test_dem() -> Path:
    """Get a local DEM for testing."""
    for _name, path in AVAILABLE_DEMS.items():
        if path.exists():
            return path
    raise AssertionError("required full-profile DEM asset is missing from assets/tif/")


@pytest.fixture
def local_dem_url():
    """Fixture providing a file:// URL to a local DEM."""
    dem_path = get_test_dem()
    return f"file://{dem_path.absolute()}"


@pytest.fixture
def remote_dem_url(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    if not REMOTE_COG_ASSET.is_file():
        raise AssertionError(f"required full-profile COG asset is missing: {REMOTE_COG_ASSET}")
    server, url, _served = _serve_range(REMOTE_COG_ASSET.read_bytes())
    try:
        yield url
    finally:
        server.shutdown()


def cog_available():
    """Check if COG streaming is available."""
    try:
        from forge3d.cog import is_cog_available
        return is_cog_available()
    except ImportError:
        return False


def rasterio_available():
    """Check if rasterio fallback is available."""
    try:
        import rasterio
        return not getattr(rasterio, "__forge3d_stub__", False)
    except Exception:
        return False


class TestCogApi:
    """Test COG Python API."""
    
    @pytest.mark.skipif(
        not cog_available() and not rasterio_available(),
        reason="Neither native COG nor rasterio available"
    )
    def test_open_cog(self, local_dem_url):
        """Test opening a COG dataset."""
        from forge3d.cog import open_cog
        
        ds = open_cog(local_dem_url, cache_size_mb=64)
        assert ds is not None
        assert ds.url == local_dem_url
    
    @pytest.mark.skipif(
        not cog_available() and not rasterio_available(),
        reason="Neither native COG nor rasterio available"
    )
    def test_bounds(self, local_dem_url):
        """Test bounds property."""
        from forge3d.cog import open_cog
        
        ds = open_cog(local_dem_url, cache_size_mb=64)
        bounds = ds.bounds
        
        assert isinstance(bounds, tuple)
        assert len(bounds) == 4
        minx, miny, maxx, maxy = bounds
        assert maxx > minx or maxx == minx == 0
        assert maxy > miny or maxy == miny == 0
    
    @pytest.mark.skipif(
        not cog_available() and not rasterio_available(),
        reason="Neither native COG nor rasterio available"
    )
    def test_overview_count(self, local_dem_url):
        """Test overview count."""
        from forge3d.cog import open_cog
        
        ds = open_cog(local_dem_url, cache_size_mb=64)
        count = ds.overview_count
        
        assert isinstance(count, int)
        assert count >= 1


class TestCogRangeRead:
    """P3.1: Test COG range read functionality."""
    
    @pytest.mark.skipif(
        not cog_available() and not rasterio_available(),
        reason="Neither native COG nor rasterio available"
    )
    def test_read_tile_basic(self, local_dem_url):
        """Test basic tile read."""
        from forge3d.cog import open_cog
        
        ds = open_cog(local_dem_url, cache_size_mb=64)
        
        tile = ds.read_tile(0, 0, lod=0)
        
        assert isinstance(tile, np.ndarray)
        assert tile.dtype == np.float32
        assert tile.ndim == 2
        assert tile.shape[0] > 0
        assert tile.shape[1] > 0
    
    @pytest.mark.skipif(
        not cog_available() and not rasterio_available(),
        reason="Neither native COG nor rasterio available"
    )
    def test_read_tile_deterministic(self, local_dem_url):
        """Test that reading the same tile twice returns identical data."""
        from forge3d.cog import open_cog
        
        ds = open_cog(local_dem_url, cache_size_mb=64)
        
        tile1 = ds.read_tile(0, 0, lod=0)
        tile2 = ds.read_tile(0, 0, lod=0)
        
        np.testing.assert_array_equal(tile1, tile2)
    
    @pytest.mark.skipif(
        not cog_available() and not rasterio_available(),
        reason="Neither native COG nor rasterio available"
    )
    def test_read_multiple_tiles(self, local_dem_url):
        """Test reading multiple tiles."""
        from forge3d.cog import open_cog
        
        ds = open_cog(local_dem_url, cache_size_mb=64)
        info = ds.ifd_info(0)
        
        tiles_to_read = min(4, info.tiles_across * info.tiles_down)
        tiles = []
        
        for i in range(tiles_to_read):
            x = i % max(1, info.tiles_across)
            y = i // max(1, info.tiles_across)
            if x < info.tiles_across and y < info.tiles_down:
                tile = ds.read_tile(x, y, lod=0)
                tiles.append(tile)
        
        assert len(tiles) > 0
        for tile in tiles:
            assert tile.dtype == np.float32

    def test_native_reads_deflate_predictor_tiled_tiff(self, tmp_path):
        """Test native COG path decodes horizontal-predictor tiled TIFF data."""
        if not cog_available():
            pytest.skip("Native COG unavailable")
        if not rasterio_available():
            pytest.skip("rasterio unavailable for predictor fixture generation")

        import rasterio
        from rasterio.transform import from_origin
        from forge3d.cog import CogDataset

        expected = (np.arange(16 * 16, dtype=np.uint16).reshape(16, 16) * 3) + 7
        path = tmp_path / "predictor_deflate_tiled.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=16,
            height=16,
            count=1,
            dtype="uint16",
            tiled=True,
            blockxsize=16,
            blockysize=16,
            compress="deflate",
            predictor=2,
            transform=from_origin(0, 16, 1, 1),
        ) as dst:
            dst.write(expected, 1)

        ds = CogDataset(f"file://{path}", cache_size_mb=4)
        actual = ds.read_tile(0, 0, lod=0)

        np.testing.assert_array_equal(actual, expected.astype(np.float32))


class TestCogOverviews:
    """P3.2: Test COG overview detection and selection."""
    
    @pytest.mark.skipif(
        not cog_available() and not rasterio_available(),
        reason="Neither native COG nor rasterio available"
    )
    def test_ifd_info(self, local_dem_url):
        """Test IFD info retrieval."""
        from forge3d.cog import open_cog, IfdInfo
        
        ds = open_cog(local_dem_url, cache_size_mb=64)
        info = ds.ifd_info(0)
        
        assert isinstance(info, IfdInfo)
        assert info.width > 0
        assert info.height > 0
        assert info.tile_width > 0
        assert info.tile_height > 0
    
    @pytest.mark.skipif(
        not cog_available() and not rasterio_available(),
        reason="Neither native COG nor rasterio available"
    )
    def test_overview_dimensions_decrease(self, tmp_path):
        """Test that overview dimensions decrease with level."""
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.transform import from_origin
        from forge3d.cog import open_cog

        path = tmp_path / "overviewed.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=64,
            height=64,
            count=1,
            dtype="float32",
            tiled=True,
            blockxsize=16,
            blockysize=16,
            transform=from_origin(0.0, 64.0, 1.0, 1.0),
        ) as dataset:
            dataset.write(np.arange(64 * 64, dtype=np.float32).reshape(64, 64), 1)
            dataset.build_overviews([2, 4], Resampling.average)

        ds = open_cog(f"file://{path}", cache_size_mb=64)
        assert ds.overview_count >= 2
        
        info0 = ds.ifd_info(0)
        info1 = ds.ifd_info(1)
        
        assert info1.width <= info0.width
        assert info1.height <= info0.height


class TestCogCache:
    """P3.3: Test COG cache functionality."""
    
    @pytest.mark.skipif(
        not cog_available() and not rasterio_available(),
        reason="Neither native COG nor rasterio available"
    )
    def test_cache_stats(self, local_dem_url):
        """Test cache statistics."""
        from forge3d.cog import open_cog, CogStats
        
        ds = open_cog(local_dem_url, cache_size_mb=64)
        
        _ = ds.read_tile(0, 0, lod=0)
        
        stats = ds.stats()
        
        assert isinstance(stats, CogStats)
        assert stats.cache_misses >= 1
        assert stats.memory_budget_bytes == 64 * 1024 * 1024
    
    @pytest.mark.skipif(
        not cog_available() and not rasterio_available(),
        reason="Neither native COG nor rasterio available"
    )
    def test_cache_hit(self, local_dem_url):
        """Test that repeated reads hit cache."""
        from forge3d.cog import open_cog
        
        ds = open_cog(local_dem_url, cache_size_mb=64)
        
        _ = ds.read_tile(0, 0, lod=0)
        stats_after_first = ds.stats()
        
        _ = ds.read_tile(0, 0, lod=0)
        stats_after_second = ds.stats()
        
        assert stats_after_second.cache_hits > stats_after_first.cache_hits
    
    @pytest.mark.skipif(
        not cog_available() and not rasterio_available(),
        reason="Neither native COG nor rasterio available"
    )
    def test_cache_memory_budget(self, local_dem_url):
        """Test that cache respects memory budget."""
        from forge3d.cog import open_cog
        
        budget_mb = 8
        ds = open_cog(local_dem_url, cache_size_mb=budget_mb)
        
        info = ds.ifd_info(0)
        
        tiles_to_read = min(100, info.tiles_across * info.tiles_down)
        for i in range(tiles_to_read):
            x = i % max(1, info.tiles_across)
            y = i // max(1, info.tiles_across)
            if x < info.tiles_across and y < info.tiles_down:
                try:
                    _ = ds.read_tile(x, y, lod=0)
                except Exception:
                    pass
        
        stats = ds.stats()
        
        tolerance = 1.2
        assert stats.memory_used_bytes <= budget_mb * 1024 * 1024 * tolerance


class TestCogDataclasses:
    """Test COG dataclass helpers."""
    
    def test_cog_stats_from_dict(self):
        """Test CogStats.from_dict."""
        from forge3d.cog import CogStats
        
        d = {
            "cache_hits": 100.0,
            "cache_misses": 50.0,
            "cache_evictions": 10.0,
            "memory_used_bytes": 1024000.0,
            "memory_budget_bytes": 2048000.0,
            "hit_rate_percent": 66.7,
        }
        
        stats = CogStats.from_dict(d)
        
        assert stats.cache_hits == 100
        assert stats.cache_misses == 50
        assert stats.cache_evictions == 10
        assert stats.memory_used_bytes == 1024000
        assert stats.memory_budget_bytes == 2048000
        assert stats.hit_rate_percent == 66.7
    
    def test_ifd_info_from_dict(self):
        """Test IfdInfo.from_dict."""
        from forge3d.cog import IfdInfo
        
        d = {
            "width": 1024,
            "height": 768,
            "tile_width": 256,
            "tile_height": 256,
            "tiles_across": 4,
            "tiles_down": 3,
            "bits_per_sample": 32,
            "compression": 8,
            "tile_count": 12,
        }
        
        info = IfdInfo.from_dict(d)
        
        assert info.width == 1024
        assert info.height == 768
        assert info.tile_width == 256
        assert info.tile_height == 256
        assert info.tiles_across == 4
        assert info.tiles_down == 3
        assert info.bits_per_sample == 32
        assert info.compression == 8
        assert info.tile_count == 12

    def test_cog_dataset_forwards_cache_dir_and_budget(self, monkeypatch, tmp_path):
        """Test native constructor receives disk-cache options."""
        from forge3d import cog

        captured = {}

        class FakeNativeCogDataset:
            def __init__(self, *args):
                captured["args"] = args

        monkeypatch.setattr(cog, "_COG_AVAILABLE", True)
        monkeypatch.setattr(cog, "_CogDatasetNative", FakeNativeCogDataset)

        ds = cog.CogDataset(
            "file:///tmp/test.tif",
            cache_size_mb=8,
            cache_dir=tmp_path,
            cache_budget_mb=4,
        )

        assert captured["args"] == ("file:///tmp/test.tif", 8, str(tmp_path), 4)
        assert ds._cache_dir == str(tmp_path)
        assert ds._cache_budget_mb == 4


class TestCogAvailability:
    """Test COG availability checking."""
    
    def test_is_cog_available(self):
        """Test is_cog_available function."""
        from forge3d.cog import is_cog_available
        
        result = is_cog_available()
        assert isinstance(result, bool)
    
    def test_cog_dataset_unavailable_error(self, monkeypatch):
        """Test error when COG not available and no fallback."""
        from forge3d import cog

        monkeypatch.setattr(cog, "_COG_AVAILABLE", False)
        with pytest.raises(RuntimeError, match="COG streaming is not available"):
            cog.CogDataset("file:///nonexistent.tif")


class TestRemoteCog:
    """Remote COG integration tests against a deterministic loopback server."""
    
    @pytest.mark.skipif(
        not cog_available() and not rasterio_available(),
        reason="Neither native COG nor rasterio available"
    )
    def test_remote_cog_open(self, remote_dem_url):
        """Test opening a remote COG via HTTP."""
        from forge3d.cog import open_cog
        
        ds = open_cog(remote_dem_url, cache_size_mb=32)
        
        assert ds is not None
        assert ds.overview_count >= 1
    
    @pytest.mark.skipif(
        not cog_available() and not rasterio_available(),
        reason="Neither native COG nor rasterio available"
    )
    def test_remote_cog_bounds(self, remote_dem_url):
        """Test bounds from remote COG."""
        from forge3d.cog import open_cog
        
        ds = open_cog(remote_dem_url, cache_size_mb=32)
        bounds = ds.bounds
        
        assert isinstance(bounds, tuple)
        assert len(bounds) == 4
    
    @pytest.mark.skipif(
        not cog_available() and not rasterio_available(),
        reason="Neither native COG nor rasterio available"
    )
    def test_remote_cog_tile_read(self, remote_dem_url):
        """Test reading a tile from remote COG."""
        from forge3d.cog import open_cog
        
        ds = open_cog(remote_dem_url, cache_size_mb=32)
        
        tile = ds.read_tile(0, 0, lod=0)
        
        assert isinstance(tile, np.ndarray)
        assert tile.ndim == 2
        assert tile.shape[0] > 0
        assert tile.shape[1] > 0
    
    @pytest.mark.skipif(
        not cog_available() and not rasterio_available(),
        reason="Neither native COG nor rasterio available"
    )
    def test_remote_cog_deterministic(self, remote_dem_url):
        """Test that remote tile reads are deterministic."""
        from forge3d.cog import open_cog
        
        ds = open_cog(remote_dem_url, cache_size_mb=32)
        
        tile1 = ds.read_tile(0, 0, lod=0)
        tile2 = ds.read_tile(0, 0, lod=0)
        
        np.testing.assert_array_equal(tile1, tile2)
    
    @pytest.mark.skipif(
        not cog_available() and not rasterio_available(),
        reason="Neither native COG nor rasterio available"
    )
    def test_remote_cog_cache_stats(self, remote_dem_url):
        """Test cache stats with remote COG."""
        from forge3d.cog import open_cog
        
        ds = open_cog(remote_dem_url, cache_size_mb=32)
        
        _ = ds.read_tile(0, 0, lod=0)
        stats1 = ds.stats()
        
        _ = ds.read_tile(0, 0, lod=0)
        stats2 = ds.stats()
        
        assert stats1.cache_misses >= 1
        assert stats2.cache_hits > stats1.cache_hits
