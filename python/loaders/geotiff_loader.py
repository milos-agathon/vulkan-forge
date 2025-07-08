"""
GeoTIFF loader for Vulkan-Forge terrain rendering.

Provides high-performance loading of geospatial height data with support for:
- Large GeoTIFF files (>4GB)
- Tile-based streaming
- Multiple resolution levels
- Memory-efficient processing
- Coordinate system transformations
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any, List
from dataclasses import dataclass
import threading
import time

try:
    from osgeo import gdal, gdalconst
    from osgeo import osr
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    gdal = None
    gdalconst = None
    osr = None

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    rasterio = None

logger = logging.getLogger(__name__)


@dataclass
class TerrainBounds:
    """Geographic bounds for terrain data."""
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float = 0.0
    max_z: float = 0.0
    
    @property
    def width(self) -> float:
        return self.max_x - self.min_x
    
    @property
    def height(self) -> float:
        return self.max_y - self.min_y
    
    @property
    def center(self) -> Tuple[float, float]:
        return (self.min_x + self.width / 2, self.min_y + self.height / 2)


@dataclass
class TerrainTileInfo:
    """Information about a terrain tile."""
    tile_x: int
    tile_y: int
    level: int
    bounds: TerrainBounds
    resolution: int  # pixels per side
    data_available: bool = True


@dataclass
class GeoTiffMetadata:
    """Metadata extracted from GeoTIFF file."""
    width: int
    height: int
    bounds: TerrainBounds
    crs: str
    nodata_value: Optional[float]
    data_type: str
    pixel_size_x: float
    pixel_size_y: float
    has_overviews: bool
    overview_levels: List[int]


class GeoTiffError(Exception):
    """Exception raised for GeoTIFF loading errors."""
    pass


class GeoTiffLoader:
    """
    High-performance GeoTIFF loader for terrain rendering.
    
    Supports both GDAL and rasterio backends with automatic fallback.
    Optimized for streaming large datasets with minimal memory usage.
    """
    
    def __init__(self, 
                 backend: str = "auto",
                 cache_size_mb: int = 512,
                 enable_overviews: bool = True):
        """
        Initialize GeoTIFF loader.
        
        Args:
            backend: "gdal", "rasterio", or "auto"
            cache_size_mb: GDAL cache size in megabytes
            enable_overviews: Use pyramid overviews for LOD
        """
        self.backend = self._select_backend(backend)
        self.cache_size_mb = cache_size_mb
        self.enable_overviews = enable_overviews
        self._dataset = None
        self._rasterio_dataset = None
        self._metadata: Optional[GeoTiffMetadata] = None
        self._lock = threading.RLock()
        
        # Configure GDAL if available
        if self.backend == "gdal" and GDAL_AVAILABLE:
            gdal.SetCacheMax(cache_size_mb * 1024 * 1024)
            gdal.UseExceptions()
    
    def _select_backend(self, backend: str) -> str:
        """Select the best available backend."""
        if backend == "auto":
            if RASTERIO_AVAILABLE:
                return "rasterio"
            elif GDAL_AVAILABLE:
                return "gdal"
            else:
                raise GeoTiffError("Neither GDAL nor rasterio available. Install one of them.")
        
        if backend == "gdal" and not GDAL_AVAILABLE:
            raise GeoTiffError("GDAL not available")
        if backend == "rasterio" and not RASTERIO_AVAILABLE:
            raise GeoTiffError("Rasterio not available")
        
        return backend
    
    def open(self, filepath: Union[str, Path]) -> GeoTiffMetadata:
        """
        Open GeoTIFF file and extract metadata.
        
        Args:
            filepath: Path to GeoTIFF file
            
        Returns:
            Metadata about the opened file
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise GeoTiffError(f"File not found: {filepath}")
        
        with self._lock:
            if self.backend == "rasterio":
                self._open_rasterio(filepath)
            else:
                self._open_gdal(filepath)
            
            self._metadata = self._extract_metadata()
            
        logger.info(f"Opened GeoTIFF: {filepath}")
        logger.info(f"Size: {self._metadata.width}x{self._metadata.height}")
        logger.info(f"Bounds: {self._metadata.bounds}")
        logger.info(f"CRS: {self._metadata.crs}")
        
        return self._metadata
    
    def _open_rasterio(self, filepath: Path):
        """Open file using rasterio backend."""
        try:
            self._rasterio_dataset = rasterio.open(str(filepath))
        except Exception as e:
            raise GeoTiffError(f"Failed to open with rasterio: {e}")
    
    def _open_gdal(self, filepath: Path):
        """Open file using GDAL backend."""
        try:
            self._dataset = gdal.Open(str(filepath), gdalconst.GA_ReadOnly)
            if self._dataset is None:
                raise GeoTiffError(f"GDAL failed to open: {filepath}")
        except Exception as e:
            raise GeoTiffError(f"Failed to open with GDAL: {e}")
    
    def _extract_metadata(self) -> GeoTiffMetadata:
        """Extract metadata from opened dataset."""
        if self.backend == "rasterio":
            return self._extract_metadata_rasterio()
        else:
            return self._extract_metadata_gdal()
    
    def _extract_metadata_rasterio(self) -> GeoTiffMetadata:
        """Extract metadata using rasterio."""
        ds = self._rasterio_dataset
        
        # Get bounds
        bounds = TerrainBounds(
            min_x=ds.bounds.left,
            max_x=ds.bounds.right,
            min_y=ds.bounds.bottom,
            max_y=ds.bounds.top
        )
        
        # Check for overviews
        overviews = []
        if ds.overviews(1):  # Band 1 overviews
            overviews = [ds.overviews(1)[i] for i in range(len(ds.overviews(1)))]
        
        return GeoTiffMetadata(
            width=ds.width,
            height=ds.height,
            bounds=bounds,
            crs=str(ds.crs) if ds.crs else "UNKNOWN",
            nodata_value=ds.nodata,
            data_type=str(ds.dtypes[0]),
            pixel_size_x=abs(ds.transform[0]),
            pixel_size_y=abs(ds.transform[4]),
            has_overviews=len(overviews) > 0,
            overview_levels=overviews
        )
    
    def _extract_metadata_gdal(self) -> GeoTiffMetadata:
        """Extract metadata using GDAL."""
        ds = self._dataset
        
        # Get geotransform
        gt = ds.GetGeoTransform()
        
        # Calculate bounds
        width, height = ds.RasterXSize, ds.RasterYSize
        min_x = gt[0]
        max_x = gt[0] + width * gt[1]
        max_y = gt[3]
        min_y = gt[3] + height * gt[5]
        
        bounds = TerrainBounds(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)
        
        # Get CRS
        srs = osr.SpatialReference()
        srs.ImportFromWkt(ds.GetProjection())
        
        # Get band info
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        data_type = gdal.GetDataTypeName(band.DataType)
        
        # Check overviews
        overview_count = band.GetOverviewCount()
        overviews = []
        for i in range(overview_count):
            overview = band.GetOverview(i)
            overviews.append(overview.XSize)
        
        return GeoTiffMetadata(
            width=width,
            height=height,
            bounds=bounds,
            crs=srs.ExportToProj4() if srs else "UNKNOWN",
            nodata_value=nodata,
            data_type=data_type,
            pixel_size_x=abs(gt[1]),
            pixel_size_y=abs(gt[5]),
            has_overviews=overview_count > 0,
            overview_levels=overviews
        )
    
    def read_tile(self,
                  tile_x: int,
                  tile_y: int,
                  tile_size: int = 512,
                  level: int = 0,
                  target_dtype: np.dtype = np.float32) -> Optional[np.ndarray]:
        """
        Read a specific tile from the dataset.
        
        Args:
            tile_x: Tile X coordinate
            tile_y: Tile Y coordinate  
            tile_size: Size of tile in pixels
            level: Overview level (0 = full resolution)
            target_dtype: Target data type for output
            
        Returns:
            Height data as numpy array, or None if outside bounds
        """
        if not self._metadata:
            raise GeoTiffError("No file opened")
        
        # Calculate pixel coordinates
        pixel_x = tile_x * tile_size
        pixel_y = tile_y * tile_size
        
        # Adjust for overview level
        if level > 0 and self._metadata.has_overviews:
            scale_factor = 2 ** level
            pixel_x //= scale_factor
            pixel_y //= scale_factor
            tile_size //= scale_factor
            effective_width = self._metadata.width // scale_factor
            effective_height = self._metadata.height // scale_factor
        else:
            effective_width = self._metadata.width
            effective_height = self._metadata.height
        
        # Check bounds
        if (pixel_x >= effective_width or 
            pixel_y >= effective_height or
            pixel_x + tile_size <= 0 or 
            pixel_y + tile_size <= 0):
            return None
        
        # Clamp to dataset bounds
        read_x = max(0, pixel_x)
        read_y = max(0, pixel_y)
        read_width = min(tile_size, effective_width - read_x)
        read_height = min(tile_size, effective_height - read_y)
        
        if read_width <= 0 or read_height <= 0:
            return None
        
        try:
            with self._lock:
                if self.backend == "rasterio":
                    data = self._read_tile_rasterio(
                        read_x, read_y, read_width, read_height, level
                    )
                else:
                    data = self._read_tile_gdal(
                        read_x, read_y, read_width, read_height, level
                    )
            
            # Handle nodata values
            if self._metadata.nodata_value is not None:
                data = np.where(data == self._metadata.nodata_value, 0.0, data)
            
            # Convert to target dtype
            if data.dtype != target_dtype:
                data = data.astype(target_dtype)
            
            # Pad if necessary (when at edges)
            if data.shape != (tile_size, tile_size):
                padded = np.zeros((tile_size, tile_size), dtype=target_dtype)
                y_offset = max(0, -pixel_y)
                x_offset = max(0, -pixel_x)
                padded[y_offset:y_offset+data.shape[0], 
                       x_offset:x_offset+data.shape[1]] = data
                data = padded
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to read tile ({tile_x}, {tile_y}): {e}")
            return None
    
    def _read_tile_rasterio(self, x: int, y: int, width: int, height: int, level: int) -> np.ndarray:
        """Read tile using rasterio."""
        if level > 0:
            # Use overview
            overview_idx = min(level - 1, len(self._rasterio_dataset.overviews(1)) - 1)
            overview_factor = self._rasterio_dataset.overviews(1)[overview_idx]
            
            # Read from overview
            window = Window(x, y, width, height)
            data = self._rasterio_dataset.read(
                1, 
                window=window, 
                out_shape=(height, width),
                resampling=rasterio.enums.Resampling.bilinear
            )
        else:
            # Read from full resolution
            window = Window(x, y, width, height)
            data = self._rasterio_dataset.read(1, window=window)
        
        return data
    
    def _read_tile_gdal(self, x: int, y: int, width: int, height: int, level: int) -> np.ndarray:
        """Read tile using GDAL."""
        if level > 0 and self._metadata.has_overviews:
            # Use overview
            band = self._dataset.GetRasterBand(1)
            overview_idx = min(level - 1, band.GetOverviewCount() - 1)
            band = band.GetOverview(overview_idx)
        else:
            band = self._dataset.GetRasterBand(1)
        
        data = band.ReadAsArray(x, y, width, height)
        if data is None:
            raise GeoTiffError(f"Failed to read data at ({x}, {y}, {width}, {height})")
        
        return data
    
    def read_region(self,
                   bounds: TerrainBounds,
                   max_pixels: int = 2048,
                   target_dtype: np.dtype = np.float32) -> Tuple[np.ndarray, TerrainBounds]:
        """
        Read a geographic region from the dataset.
        
        Args:
            bounds: Geographic bounds to read
            max_pixels: Maximum pixels in any dimension
            target_dtype: Target data type
            
        Returns:
            (height_data, actual_bounds)
        """
        if not self._metadata:
            raise GeoTiffError("No file opened")
        
        # Convert geographic bounds to pixel coordinates
        pixel_bounds = self._geo_to_pixel_bounds(bounds)
        
        # Calculate read size
        pixel_width = pixel_bounds.max_x - pixel_bounds.min_x
        pixel_height = pixel_bounds.max_y - pixel_bounds.min_y
        
        # Determine overview level
        level = 0
        while (pixel_width > max_pixels or pixel_height > max_pixels) and level < 10:
            level += 1
            pixel_width //= 2
            pixel_height //= 2
        
        # Read data
        try:
            with self._lock:
                if self.backend == "rasterio":
                    data = self._read_region_rasterio(bounds, max_pixels, level)
                else:
                    data = self._read_region_gdal(bounds, max_pixels, level)
            
            # Handle nodata
            if self._metadata.nodata_value is not None:
                data = np.where(data == self._metadata.nodata_value, 0.0, data)
            
            # Convert dtype
            if data.dtype != target_dtype:
                data = data.astype(target_dtype)
            
            # Calculate actual bounds
            actual_bounds = self._pixel_to_geo_bounds(
                TerrainBounds(
                    min_x=pixel_bounds.min_x,
                    max_x=pixel_bounds.min_x + data.shape[1],
                    min_y=pixel_bounds.min_y,
                    max_y=pixel_bounds.min_y + data.shape[0]
                )
            )
            
            # Update elevation bounds
            actual_bounds.min_z = float(np.min(data))
            actual_bounds.max_z = float(np.max(data))
            
            return data, actual_bounds
            
        except Exception as e:
            raise GeoTiffError(f"Failed to read region: {e}")
    
    def _read_region_rasterio(self, bounds: TerrainBounds, max_pixels: int, level: int) -> np.ndarray:
        """Read region using rasterio."""
        # Convert bounds to window
        transform = self._rasterio_dataset.transform
        
        # Calculate pixel coordinates
        left, bottom, right, top = bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y
        
        window = rasterio.windows.from_bounds(
            left, bottom, right, top, transform=transform
        )
        
        # Determine output size
        out_width = min(int(window.width), max_pixels)
        out_height = min(int(window.height), max_pixels)
        
        data = self._rasterio_dataset.read(
            1,
            window=window,
            out_shape=(out_height, out_width),
            resampling=rasterio.enums.Resampling.bilinear
        )
        
        return data
    
    def _read_region_gdal(self, bounds: TerrainBounds, max_pixels: int, level: int) -> np.ndarray:
        """Read region using GDAL."""
        pixel_bounds = self._geo_to_pixel_bounds(bounds)
        
        x = int(pixel_bounds.min_x)
        y = int(pixel_bounds.min_y)
        width = int(pixel_bounds.width)
        height = int(pixel_bounds.height)
        
        # Clamp to dataset bounds
        x = max(0, min(x, self._metadata.width))
        y = max(0, min(y, self._metadata.height))
        width = min(width, self._metadata.width - x)
        height = min(height, self._metadata.height - y)
        
        # Calculate output size
        out_width = min(width, max_pixels)
        out_height = min(height, max_pixels)
        
        band = self._dataset.GetRasterBand(1)
        data = band.ReadAsArray(x, y, width, height, out_width, out_height)
        
        if data is None:
            raise GeoTiffError("Failed to read region data")
        
        return data
    
    def _geo_to_pixel_bounds(self, geo_bounds: TerrainBounds) -> TerrainBounds:
        """Convert geographic bounds to pixel coordinates."""
        if self.backend == "rasterio":
            transform = self._rasterio_dataset.transform
            inv_transform = ~transform
            
            min_x, max_y = inv_transform * (geo_bounds.min_x, geo_bounds.max_y)
            max_x, min_y = inv_transform * (geo_bounds.max_x, geo_bounds.min_y)
        else:
            gt = self._dataset.GetGeoTransform()
            
            # Inverse geotransform
            det = gt[1] * gt[5] - gt[2] * gt[4]
            inv_gt = [
                -gt[3] * gt[2] / det + gt[0] * gt[5] / det,
                gt[5] / det,
                -gt[2] / det,
                gt[3] / det,
                -gt[1] / det,
                gt[1] * gt[4] / det - gt[0] * gt[1] / det
            ]
            
            min_x = inv_gt[0] + geo_bounds.min_x * inv_gt[1] + geo_bounds.max_y * inv_gt[2]
            max_y = inv_gt[3] + geo_bounds.min_x * inv_gt[4] + geo_bounds.max_y * inv_gt[5]
            max_x = inv_gt[0] + geo_bounds.max_x * inv_gt[1] + geo_bounds.min_y * inv_gt[2]
            min_y = inv_gt[3] + geo_bounds.max_x * inv_gt[4] + geo_bounds.min_y * inv_gt[5]
        
        return TerrainBounds(
            min_x=int(np.floor(min_x)),
            max_x=int(np.ceil(max_x)),
            min_y=int(np.floor(min_y)),
            max_y=int(np.ceil(max_y))
        )
    
    def _pixel_to_geo_bounds(self, pixel_bounds: TerrainBounds) -> TerrainBounds:
        """Convert pixel coordinates to geographic bounds."""
        if self.backend == "rasterio":
            transform = self._rasterio_dataset.transform
            
            min_x, max_y = transform * (pixel_bounds.min_x, pixel_bounds.min_y)
            max_x, min_y = transform * (pixel_bounds.max_x, pixel_bounds.max_y)
        else:
            gt = self._dataset.GetGeoTransform()
            
            min_x = gt[0] + pixel_bounds.min_x * gt[1] + pixel_bounds.min_y * gt[2]
            max_y = gt[3] + pixel_bounds.min_x * gt[4] + pixel_bounds.min_y * gt[5]
            max_x = gt[0] + pixel_bounds.max_x * gt[1] + pixel_bounds.max_y * gt[2]
            min_y = gt[3] + pixel_bounds.max_x * gt[4] + pixel_bounds.max_y * gt[5]
        
        return TerrainBounds(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)
    
    def get_tile_info(self, tile_size: int = 512, level: int = 0) -> List[TerrainTileInfo]:
        """
        Get information about all tiles in the dataset.
        
        Args:
            tile_size: Size of each tile in pixels
            level: Overview level
            
        Returns:
            List of tile information
        """
        if not self._metadata:
            raise GeoTiffError("No file opened")
        
        # Adjust for overview level
        if level > 0:
            scale_factor = 2 ** level
            width = self._metadata.width // scale_factor
            height = self._metadata.height // scale_factor
        else:
            width = self._metadata.width
            height = self._metadata.height
        
        tiles = []
        tiles_x = (width + tile_size - 1) // tile_size
        tiles_y = (height + tile_size - 1) // tile_size
        
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                # Calculate tile bounds in pixels
                pixel_x = tx * tile_size
                pixel_y = ty * tile_size
                pixel_width = min(tile_size, width - pixel_x)
                pixel_height = min(tile_size, height - pixel_y)
                
                # Convert to geographic bounds
                pixel_bounds = TerrainBounds(
                    min_x=pixel_x,
                    max_x=pixel_x + pixel_width,
                    min_y=pixel_y,
                    max_y=pixel_y + pixel_height
                )
                
                geo_bounds = self._pixel_to_geo_bounds(pixel_bounds)
                
                tiles.append(TerrainTileInfo(
                    tile_x=tx,
                    tile_y=ty,
                    level=level,
                    bounds=geo_bounds,
                    resolution=tile_size,
                    data_available=True
                ))
        
        return tiles
    
    def close(self):
        """Close the dataset and free resources."""
        with self._lock:
            if self._rasterio_dataset:
                self._rasterio_dataset.close()
                self._rasterio_dataset = None
            
            if self._dataset:
                self._dataset = None  # GDAL handles cleanup automatically
            
            self._metadata = None
        
        logger.info("GeoTIFF dataset closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    @property
    def metadata(self) -> Optional[GeoTiffMetadata]:
        """Get metadata for the opened dataset."""
        return self._metadata
    
    @property
    def is_open(self) -> bool:
        """Check if a dataset is currently open."""
        return self._metadata is not None


def create_test_geotiff(filepath: Union[str, Path], 
                       width: int = 1024, 
                       height: int = 1024,
                       bounds: Optional[TerrainBounds] = None) -> None:
    """
    Create a test GeoTIFF file for development and testing.
    
    Args:
        filepath: Output file path
        width: Image width in pixels
        height: Image height in pixels  
        bounds: Geographic bounds (default: 0-1000 in both X/Y)
    """
    if not RASTERIO_AVAILABLE:
        raise GeoTiffError("Rasterio required for creating test GeoTIFF")
    
    if bounds is None:
        bounds = TerrainBounds(min_x=0, max_x=1000, min_y=0, max_y=1000)
    
    # Create synthetic terrain data
    x = np.linspace(0, 10, width)
    y = np.linspace(0, 10, height)
    X, Y = np.meshgrid(x, y)
    
    # Generate interesting terrain: valleys, peaks, ridges
    terrain = (
        100 * np.sin(X * 0.5) * np.cos(Y * 0.3) +  # Rolling hills
        50 * np.sin(X * 2) * np.sin(Y * 2) +       # Smaller features
        200 * np.exp(-((X-5)**2 + (Y-5)**2) / 4) + # Central peak
        -100 * np.exp(-((X-2)**2 + (Y-8)**2) / 2)  # Valley
    )
    
    # Add some noise for realism
    terrain += np.random.normal(0, 5, terrain.shape)
    
    # Convert to appropriate data type
    terrain = terrain.astype(np.float32)
    
    # Create transform
    transform = from_bounds(
        bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y,
        width, height
    )
    
    # Write GeoTIFF
    with rasterio.open(
        filepath,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=terrain.dtype,
        crs='EPSG:3857',  # Web Mercator
        transform=transform,
        compress='lzw',
        tiled=True,
        blockxsize=512,
        blockysize=512
    ) as dst:
        dst.write(terrain, 1)
        
        # Build overviews for performance
        dst.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.average)
        dst.update_tags(ns='rio_overview', resampling='average')
    
    logger.info(f"Created test GeoTIFF: {filepath}")
    logger.info(f"Size: {width}x{height}, Bounds: {bounds}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create a test GeoTIFF
    test_file = "test_terrain.tif"
    create_test_geotiff(test_file, width=2048, height=2048)
    
    # Load and analyze
    with GeoTiffLoader() as loader:
        metadata = loader.open(test_file)
        print(f"Loaded: {metadata}")
        
        # Read a tile
        tile_data = loader.read_tile(0, 0, tile_size=512)
        if tile_data is not None:
            print(f"Tile shape: {tile_data.shape}")
            print(f"Height range: {tile_data.min():.1f} - {tile_data.max():.1f}")
        
        # Get tile info
        tiles = loader.get_tile_info(tile_size=512)
        print(f"Total tiles: {len(tiles)}")