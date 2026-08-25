"""
Test DEM (Digital Elevation Model) loading functionality
MILESTONE 5 - Task 5.1
"""

import pytest
import numpy as np
import forge3d as f3d


def test_load_dem_from_array():
    """Test creating DEM from NumPy array."""
    # Create synthetic elevation data
    data = np.random.rand(256, 256).astype(np.float32) * 1000.0

    dem = f3d.io.load_dem_from_array(data)

    assert dem is not None
    assert dem.data.shape == (256, 256)
    assert dem.data.dtype == np.float32
    assert dem.domain[0] < dem.domain[1]
    assert dem.stats is not None
    assert "min" in dem.stats
    assert "max" in dem.stats
    assert "mean" in dem.stats


def test_load_dem_from_array_with_domain():
    """Test creating DEM with explicit domain."""
    data = np.random.rand(128, 128).astype(np.float32) * 500.0 + 200.0

    dem = f3d.io.load_dem_from_array(data, domain=(200.0, 700.0))

    assert dem.domain == (200.0, 700.0)


def test_load_dem_from_array_with_nodata():
    """Test creating DEM with nodata values."""
    data = np.random.rand(128, 128).astype(np.float32) * 1000.0

    # Set some pixels to nodata
    nodata_value = -9999.0
    data[::10, ::10] = nodata_value

    dem = f3d.io.load_dem_from_array(data, nodata=nodata_value)

    assert dem.nodata_value == nodata_value
    # Stats should exclude nodata
    assert dem.stats["min"] > nodata_value


def test_calculate_dem_stats():
    """Test DEM statistics calculation."""
    data = np.arange(100, dtype=np.float32).reshape(10, 10)

    stats = f3d.io.calculate_dem_stats(data)

    assert stats["min"] == 0.0
    assert stats["max"] == 99.0
    assert abs(stats["mean"] - 49.5) < 0.1
    assert stats["count"] == 100


def test_calculate_dem_stats_with_nodata():
    """Test DEM statistics with nodata values."""
    data = np.ones((10, 10), dtype=np.float32) * 100.0
    nodata = -9999.0
    data[0, :] = nodata  # First row is nodata

    stats = f3d.io.calculate_dem_stats(data, nodata)

    assert stats["min"] == 100.0
    assert stats["max"] == 100.0
    assert stats["count"] == 90  # 100 - 10 nodata pixels


def test_fill_nodata_nearest():
    """Test nodata filling with nearest neighbor."""
    data = np.ones((10, 10), dtype=np.float32) * 100.0
    nodata = -9999.0
    data[5, 5] = nodata

    filled = f3d.io.fill_nodata(data, nodata, method="nearest")

    assert not np.any(np.isclose(filled, nodata))
    assert filled[5, 5] == 100.0  # Should be filled with nearby value


def test_fill_nodata_no_missing():
    """Test fill_nodata when there are no missing values."""
    data = np.ones((10, 10), dtype=np.float32) * 100.0
    nodata = -9999.0

    filled = f3d.io.fill_nodata(data, nodata, method="nearest")

    np.testing.assert_array_equal(filled, data)


def test_dem_data_stats_auto_calculation():
    """Test that stats are automatically calculated on DEMData creation."""
    data = np.random.rand(64, 64).astype(np.float32) * 500.0

    dem = f3d.io.load_dem_from_array(data)

    # Stats should be auto-populated
    assert dem.stats is not None
    assert "mean" in dem.stats
    assert "std" in dem.stats
    assert "median" in dem.stats
    assert "p01" in dem.stats
    assert "p99" in dem.stats


def test_dem_data_invalid_shape():
    """Test that 1D or 3D arrays are rejected."""
    # 1D array
    with pytest.raises(ValueError, match="DEM data must be 2D"):
        f3d.io.load_dem_from_array(np.array([1, 2, 3]))

    # 3D array
    with pytest.raises(ValueError, match="DEM data must be 2D"):
        f3d.io.load_dem_from_array(np.random.rand(10, 10, 3))


def test_dem_synthetic_procedural():
    """Test creating procedural synthetic DEM."""
    # Create a simple terrain with a hill in the center
    size = 256
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    # Gaussian hill
    data = 1000.0 * np.exp(-(X**2 + Y**2) / 0.5)
    data = data.astype(np.float32)

    dem = f3d.io.load_dem_from_array(data)

    assert dem.data.shape == (size, size)
    assert dem.stats["max"] > dem.stats["min"]
    assert 0 <= dem.stats["mean"] <= 1000.0


def test_load_dem_from_geotiff(tmp_path):
    """Test loading DEM from GeoTIFF file."""
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / "elevation.tif"
    elevation = np.arange(16, dtype=np.float32).reshape(4, 4)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="float32",
        crs="EPSG:32610",
        transform=from_origin(500_000.0, 5_200_000.0, 30.0, 30.0),
    ) as dataset:
        dataset.write(elevation, 1)

    dem = f3d.io.load_dem(path, fill_nodata_values=True)

    assert dem is not None
    assert dem.data.ndim == 2
    assert dem.domain[0] < dem.domain[1]
    assert dem.crs is not None
    assert dem.transform is not None


def test_dem_data_repr():
    """Test DEMData string representation."""
    data = np.random.rand(64, 64).astype(np.float32) * 1000.0
    dem = f3d.io.load_dem_from_array(data)

    # Should not raise
    repr_str = repr(dem)
    assert "DEMData" in repr_str or "data=" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
