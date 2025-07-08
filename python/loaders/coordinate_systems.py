"""
Coordinate system transformations and projections for terrain rendering.

Provides high-performance coordinate transformations with support for:
- Common geospatial coordinate reference systems (CRS)
- Efficient batch transformations
- Terrain-specific utilities (tile calculations, bounds)
- Web Mercator optimizations for performance
- UTM zone detection and handling
- Geographic coordinate utilities
"""

import numpy as np
import math
import logging
from typing import Tuple, List, Optional, Union, Dict, Any
from dataclasses import dataclass
from enum import Enum
import threading

try:
    import pyproj
    from pyproj import Transformer, CRS
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    pyproj = None
    Transformer = None
    CRS = None

from .geotiff_loader import TerrainBounds

logger = logging.getLogger(__name__)


class CoordinateSystemError(Exception):
    """Exception raised for coordinate system errors."""
    pass


@dataclass
class Point2D:
    """2D point with x, y coordinates."""
    x: float
    y: float
    
    def __iter__(self):
        yield self.x
        yield self.y
    
    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError("Point2D index out of range")


@dataclass
class Point3D:
    """3D point with x, y, z coordinates."""
    x: float
    y: float
    z: float = 0.0
    
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z
    
    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        else:
            raise IndexError("Point3D index out of range")


class CommonCRS(Enum):
    """Common coordinate reference systems."""
    WGS84 = "EPSG:4326"          # Geographic coordinates (lat/lon)
    WEB_MERCATOR = "EPSG:3857"   # Web Mercator (Google Maps, etc.)
    UTM_NORTH = "UTM_NORTH"      # UTM Northern hemisphere (zone auto-detected)
    UTM_SOUTH = "UTM_SOUTH"      # UTM Southern hemisphere (zone auto-detected)


@dataclass
class TileCoordinate:
    """Tile coordinate with level of detail."""
    x: int
    y: int
    z: int  # zoom level
    
    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z))


@dataclass
class UTMZone:
    """UTM zone information."""
    zone_number: int
    hemisphere: str  # 'N' or 'S'
    epsg_code: int
    
    @property
    def crs_string(self) -> str:
        return f"EPSG:{self.epsg_code}"


class CoordinateTransformer:
    """
    High-performance coordinate transformer with caching.
    
    Provides efficient transformations between coordinate reference systems
    with automatic caching of transformers for performance.
    """
    
    def __init__(self):
        self._transformers: Dict[Tuple[str, str], Any] = {}
        self._lock = threading.RLock()
        
        if not PYPROJ_AVAILABLE:
            logger.warning("pyproj not available - coordinate transformations will be limited")
    
    def get_transformer(self, src_crs: str, dst_crs: str) -> Optional[Any]:
        """
        Get or create a transformer for the given CRS pair.
        
        Args:
            src_crs: Source CRS (e.g., 'EPSG:4326')
            dst_crs: Destination CRS (e.g., 'EPSG:3857')
            
        Returns:
            Pyproj transformer or None if not available
        """
        if not PYPROJ_AVAILABLE:
            return None
        
        key = (src_crs, dst_crs)
        
        with self._lock:
            if key not in self._transformers:
                try:
                    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
                    self._transformers[key] = transformer
                    logger.debug(f"Created transformer: {src_crs} -> {dst_crs}")
                except Exception as e:
                    logger.error(f"Failed to create transformer {src_crs} -> {dst_crs}: {e}")
                    return None
            
            return self._transformers[key]
    
    def transform_point(self, 
                       x: float, 
                       y: float, 
                       src_crs: str, 
                       dst_crs: str) -> Tuple[float, float]:
        """
        Transform a single point between coordinate systems.
        
        Args:
            x: X coordinate (longitude for geographic)
            y: Y coordinate (latitude for geographic)
            src_crs: Source CRS
            dst_crs: Destination CRS
            
        Returns:
            Transformed (x, y) coordinates
        """
        if src_crs == dst_crs:
            return x, y
        
        transformer = self.get_transformer(src_crs, dst_crs)
        if transformer is None:
            raise CoordinateSystemError(f"Cannot transform from {src_crs} to {dst_crs}")
        
        try:
            return transformer.transform(x, y)
        except Exception as e:
            raise CoordinateSystemError(f"Transformation failed: {e}")
    
    def transform_points(self,
                        x_coords: np.ndarray,
                        y_coords: np.ndarray,
                        src_crs: str,
                        dst_crs: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform multiple points between coordinate systems.
        
        Args:
            x_coords: Array of X coordinates
            y_coords: Array of Y coordinates
            src_crs: Source CRS
            dst_crs: Destination CRS
            
        Returns:
            Transformed (x_coords, y_coords) arrays
        """
        if src_crs == dst_crs:
            return x_coords, y_coords
        
        transformer = self.get_transformer(src_crs, dst_crs)
        if transformer is None:
            raise CoordinateSystemError(f"Cannot transform from {src_crs} to {dst_crs}")
        
        try:
            return transformer.transform(x_coords, y_coords)
        except Exception as e:
            raise CoordinateSystemError(f"Batch transformation failed: {e}")
    
    def transform_bounds(self,
                        bounds: TerrainBounds,
                        src_crs: str,
                        dst_crs: str) -> TerrainBounds:
        """
        Transform terrain bounds between coordinate systems.
        
        Args:
            bounds: Source bounds
            src_crs: Source CRS
            dst_crs: Destination CRS
            
        Returns:
            Transformed bounds
        """
        if src_crs == dst_crs:
            return bounds
        
        # Transform corner points
        corners_x = np.array([bounds.min_x, bounds.max_x, bounds.min_x, bounds.max_x])
        corners_y = np.array([bounds.min_y, bounds.min_y, bounds.max_y, bounds.max_y])
        
        transformed_x, transformed_y = self.transform_points(
            corners_x, corners_y, src_crs, dst_crs
        )
        
        return TerrainBounds(
            min_x=float(np.min(transformed_x)),
            max_x=float(np.max(transformed_x)),
            min_y=float(np.min(transformed_y)),
            max_y=float(np.max(transformed_y)),
            min_z=bounds.min_z,
            max_z=bounds.max_z
        )


class WebMercatorUtils:
    """
    Optimized utilities for Web Mercator (EPSG:3857) coordinate system.
    
    Provides fast transformations and tile calculations for the most common
    web mapping coordinate system.
    """
    
    # Web Mercator constants
    EARTH_RADIUS = 6378137.0  # WGS84 semi-major axis
    ORIGIN_SHIFT = math.pi * EARTH_RADIUS
    MAX_LATITUDE = 85.0511287798
    
    @staticmethod
    def geographic_to_web_mercator(lon: float, lat: float) -> Tuple[float, float]:
        """
        Convert geographic coordinates (WGS84) to Web Mercator.
        
        Args:
            lon: Longitude in degrees
            lat: Latitude in degrees
            
        Returns:
            (x, y) in Web Mercator meters
        """
        # Clamp latitude to valid range
        lat = max(-WebMercatorUtils.MAX_LATITUDE, 
                  min(WebMercatorUtils.MAX_LATITUDE, lat))
        
        x = lon * WebMercatorUtils.ORIGIN_SHIFT / 180.0
        y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) * WebMercatorUtils.EARTH_RADIUS
        
        return x, y
    
    @staticmethod
    def web_mercator_to_geographic(x: float, y: float) -> Tuple[float, float]:
        """
        Convert Web Mercator coordinates to geographic (WGS84).
        
        Args:
            x: X coordinate in Web Mercator meters
            y: Y coordinate in Web Mercator meters
            
        Returns:
            (longitude, latitude) in degrees
        """
        lon = x / WebMercatorUtils.ORIGIN_SHIFT * 180.0
        lat = math.atan(math.exp(y / WebMercatorUtils.EARTH_RADIUS)) * 360.0 / math.pi - 90.0
        
        return lon, lat
    
    @staticmethod
    def geographic_bounds_to_web_mercator(bounds: TerrainBounds) -> TerrainBounds:
        """Convert geographic bounds to Web Mercator."""
        min_x, max_y = WebMercatorUtils.geographic_to_web_mercator(
            bounds.min_x, bounds.max_y
        )
        max_x, min_y = WebMercatorUtils.geographic_to_web_mercator(
            bounds.max_x, bounds.min_y
        )
        
        return TerrainBounds(
            min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
            min_z=bounds.min_z, max_z=bounds.max_z
        )
    
    @staticmethod
    def tile_bounds(tile_x: int, tile_y: int, zoom: int) -> TerrainBounds:
        """
        Calculate Web Mercator bounds for a tile.
        
        Args:
            tile_x: Tile X coordinate
            tile_y: Tile Y coordinate
            zoom: Zoom level
            
        Returns:
            Bounds in Web Mercator meters
        """
        n = 2.0 ** zoom
        
        # Convert tile coordinates to longitude/latitude
        lon_min = tile_x / n * 360.0 - 180.0
        lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n))))
        
        lon_max = (tile_x + 1) / n * 360.0 - 180.0
        lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (tile_y + 1) / n))))
        
        # Convert to Web Mercator
        min_x, max_y = WebMercatorUtils.geographic_to_web_mercator(lon_min, lat_max)
        max_x, min_y = WebMercatorUtils.geographic_to_web_mercator(lon_max, lat_min)
        
        return TerrainBounds(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)
    
    @staticmethod
    def geographic_to_tile(lon: float, lat: float, zoom: int) -> TileCoordinate:
        """
        Convert geographic coordinates to tile coordinates.
        
        Args:
            lon: Longitude in degrees
            lat: Latitude in degrees
            zoom: Zoom level
            
        Returns:
            Tile coordinate
        """
        n = 2.0 ** zoom
        
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
        
        # Clamp to valid range
        x = max(0, min(int(n) - 1, x))
        y = max(0, min(int(n) - 1, y))
        
        return TileCoordinate(x, y, zoom)
    
    @staticmethod
    def bounds_to_tiles(bounds: TerrainBounds, zoom: int) -> List[TileCoordinate]:
        """
        Get all tile coordinates that intersect with the given bounds.
        
        Args:
            bounds: Geographic bounds
            zoom: Zoom level
            
        Returns:
            List of tile coordinates
        """
        # Convert bounds to tile coordinates
        min_tile = WebMercatorUtils.geographic_to_tile(bounds.min_x, bounds.max_y, zoom)
        max_tile = WebMercatorUtils.geographic_to_tile(bounds.max_x, bounds.min_y, zoom)
        
        tiles = []
        for y in range(min_tile.y, max_tile.y + 1):
            for x in range(min_tile.x, max_tile.x + 1):
                tiles.append(TileCoordinate(x, y, zoom))
        
        return tiles


class UTMUtils:
    """
    Utilities for Universal Transverse Mercator (UTM) coordinate system.
    
    Provides UTM zone detection and coordinate transformations.
    """
    
    @staticmethod
    def get_utm_zone(lon: float, lat: float) -> UTMZone:
        """
        Determine UTM zone for given geographic coordinates.
        
        Args:
            lon: Longitude in degrees
            lat: Latitude in degrees
            
        Returns:
            UTM zone information
        """
        # Calculate zone number
        zone_number = int((lon + 180) / 6) + 1
        
        # Special cases for Norway and Svalbard
        if 56 <= lat < 64 and 3 <= lon < 12:
            zone_number = 32
        elif 72 <= lat < 84:
            if 0 <= lon < 9:
                zone_number = 31
            elif 9 <= lon < 21:
                zone_number = 33
            elif 21 <= lon < 33:
                zone_number = 35
            elif 33 <= lon < 42:
                zone_number = 37
        
        # Determine hemisphere
        hemisphere = 'N' if lat >= 0 else 'S'
        
        # Calculate EPSG code
        if hemisphere == 'N':
            epsg_code = 32600 + zone_number
        else:
            epsg_code = 32700 + zone_number
        
        return UTMZone(zone_number, hemisphere, epsg_code)
    
    @staticmethod
    def get_optimal_utm_crs(bounds: TerrainBounds) -> str:
        """
        Get the optimal UTM CRS for the given geographic bounds.
        
        Args:
            bounds: Geographic bounds in WGS84
            
        Returns:
            UTM CRS string (e.g., 'EPSG:32633')
        """
        # Use center point to determine zone
        center_lon = (bounds.min_x + bounds.max_x) / 2
        center_lat = (bounds.min_y + bounds.max_y) / 2
        
        utm_zone = UTMUtils.get_utm_zone(center_lon, center_lat)
        return utm_zone.crs_string
    
    @staticmethod
    def utm_bounds_for_zone(zone_number: int, hemisphere: str) -> TerrainBounds:
        """
        Get the geographic bounds for a UTM zone.
        
        Args:
            zone_number: UTM zone number (1-60)
            hemisphere: 'N' or 'S'
            
        Returns:
            Geographic bounds for the UTM zone
        """
        min_lon = (zone_number - 1) * 6 - 180
        max_lon = zone_number * 6 - 180
        
        if hemisphere == 'N':
            min_lat = 0
            max_lat = 84
        else:
            min_lat = -80
            max_lat = 0
        
        return TerrainBounds(
            min_x=min_lon, max_x=max_lon,
            min_y=min_lat, max_y=max_lat
        )


class TerrainCoordinateUtils:
    """
    Terrain-specific coordinate utilities.
    
    Provides functions for calculating terrain tile bounds, pixel coordinates,
    and other terrain-specific coordinate operations.
    """
    
    def __init__(self, transformer: Optional[CoordinateTransformer] = None):
        self.transformer = transformer or CoordinateTransformer()
    
    def calculate_tile_bounds(self,
                             tile_x: int,
                             tile_y: int,
                             tile_size: int,
                             pixel_size_x: float,
                             pixel_size_y: float,
                             origin_x: float,
                             origin_y: float,
                             level: int = 0) -> TerrainBounds:
        """
        Calculate geographic bounds for a terrain tile.
        
        Args:
            tile_x: Tile X coordinate
            tile_y: Tile Y coordinate
            tile_size: Tile size in pixels
            pixel_size_x: Pixel size in X direction (units per pixel)
            pixel_size_y: Pixel size in Y direction (units per pixel)
            origin_x: Dataset origin X coordinate
            origin_y: Dataset origin Y coordinate
            level: LOD level (0 = full resolution)
            
        Returns:
            Geographic bounds for the tile
        """
        # Adjust for LOD level
        effective_pixel_size_x = pixel_size_x * (2 ** level)
        effective_pixel_size_y = pixel_size_y * (2 ** level)
        
        # Calculate bounds
        min_x = origin_x + tile_x * tile_size * effective_pixel_size_x
        max_x = min_x + tile_size * effective_pixel_size_x
        max_y = origin_y - tile_y * tile_size * effective_pixel_size_y
        min_y = max_y - tile_size * effective_pixel_size_y
        
        return TerrainBounds(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)
    
    def geographic_to_pixel(self,
                           geo_x: float,
                           geo_y: float,
                           pixel_size_x: float,
                           pixel_size_y: float,
                           origin_x: float,
                           origin_y: float) -> Tuple[float, float]:
        """
        Convert geographic coordinates to pixel coordinates.
        
        Args:
            geo_x: Geographic X coordinate
            geo_y: Geographic Y coordinate
            pixel_size_x: Pixel size in X direction
            pixel_size_y: Pixel size in Y direction
            origin_x: Dataset origin X
            origin_y: Dataset origin Y
            
        Returns:
            (pixel_x, pixel_y) coordinates
        """
        pixel_x = (geo_x - origin_x) / pixel_size_x
        pixel_y = (origin_y - geo_y) / pixel_size_y
        
        return pixel_x, pixel_y
    
    def pixel_to_geographic(self,
                           pixel_x: float,
                           pixel_y: float,
                           pixel_size_x: float,
                           pixel_size_y: float,
                           origin_x: float,
                           origin_y: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to geographic coordinates.
        
        Args:
            pixel_x: Pixel X coordinate
            pixel_y: Pixel Y coordinate
            pixel_size_x: Pixel size in X direction
            pixel_size_y: Pixel size in Y direction
            origin_x: Dataset origin X
            origin_y: Dataset origin Y
            
        Returns:
            (geo_x, geo_y) coordinates
        """
        geo_x = origin_x + pixel_x * pixel_size_x
        geo_y = origin_y - pixel_y * pixel_size_y
        
        return geo_x, geo_y
    
    def calculate_resolution_for_scale(self,
                                      target_pixel_size: float,
                                      base_pixel_size: float) -> int:
        """
        Calculate the appropriate LOD level for a target pixel size.
        
        Args:
            target_pixel_size: Desired pixel size
            base_pixel_size: Base dataset pixel size
            
        Returns:
            LOD level (0 = full resolution)
        """
        if target_pixel_size <= base_pixel_size:
            return 0
        
        ratio = target_pixel_size / base_pixel_size
        level = int(math.log2(ratio))
        
        return max(0, level)
    
    def estimate_tile_count(self,
                           bounds: TerrainBounds,
                           tile_size: int,
                           pixel_size_x: float,
                           pixel_size_y: float,
                           level: int = 0) -> Tuple[int, int]:
        """
        Estimate the number of tiles needed to cover the given bounds.
        
        Args:
            bounds: Geographic bounds to cover
            tile_size: Tile size in pixels
            pixel_size_x: Pixel size in X direction
            pixel_size_y: Pixel size in Y direction
            level: LOD level
            
        Returns:
            (tiles_x, tiles_y) tuple
        """
        # Adjust for LOD level
        effective_pixel_size_x = pixel_size_x * (2 ** level)
        effective_pixel_size_y = pixel_size_y * (2 ** level)
        
        # Calculate pixel coverage
        pixels_x = bounds.width / effective_pixel_size_x
        pixels_y = bounds.height / effective_pixel_size_y
        
        # Calculate tile count
        tiles_x = int(math.ceil(pixels_x / tile_size))
        tiles_y = int(math.ceil(pixels_y / tile_size))
        
        return tiles_x, tiles_y


class CRSRegistry:
    """
    Registry of common coordinate reference systems.
    
    Provides easy access to common CRS definitions and utilities.
    """
    
    # Common EPSG codes
    WGS84 = "EPSG:4326"
    WEB_MERCATOR = "EPSG:3857"
    
    # Regional UTM zones (examples)
    UTM_ZONES = {
        # North America
        "utm_10n": "EPSG:32610",  # California
        "utm_11n": "EPSG:32611",  # Nevada, California
        "utm_12n": "EPSG:32612",  # Utah, Colorado
        "utm_13n": "EPSG:32613",  # Colorado, Wyoming
        "utm_14n": "EPSG:32614",  # Kansas, Nebraska
        "utm_15n": "EPSG:32615",  # Minnesota, Iowa
        "utm_16n": "EPSG:32616",  # Illinois, Wisconsin
        "utm_17n": "EPSG:32617",  # Michigan, Ohio
        "utm_18n": "EPSG:32618",  # Pennsylvania, New York
        "utm_19n": "EPSG:32619",  # Maine, Maritime Canada
        
        # Europe
        "utm_31n": "EPSG:32631",  # Norway, Denmark
        "utm_32n": "EPSG:32632",  # Germany, Poland
        "utm_33n": "EPSG:32633",  # Austria, Czech Republic
        "utm_34n": "EPSG:32634",  # Romania, Bulgaria
        "utm_35n": "EPSG:32635",  # Turkey, Greece
    }
    
    @classmethod
    def get_crs_info(cls, crs_code: str) -> Dict[str, Any]:
        """
        Get information about a coordinate reference system.
        
        Args:
            crs_code: CRS code (e.g., 'EPSG:4326')
            
        Returns:
            Dictionary with CRS information
        """
        info = {
            "code": crs_code,
            "type": "unknown",
            "units": "unknown",
            "description": "Unknown CRS"
        }
        
        if crs_code == cls.WGS84:
            info.update({
                "type": "geographic",
                "units": "degrees",
                "description": "WGS84 Geographic (latitude/longitude)"
            })
        elif crs_code == cls.WEB_MERCATOR:
            info.update({
                "type": "projected",
                "units": "meters",
                "description": "Web Mercator (Google Maps, OpenStreetMap)"
            })
        elif crs_code.startswith("EPSG:326") or crs_code.startswith("EPSG:327"):
            # UTM zones
            zone_num = int(crs_code[-2:])
            hemisphere = "North" if crs_code.startswith("EPSG:326") else "South"
            info.update({
                "type": "projected",
                "units": "meters",
                "description": f"UTM Zone {zone_num}{hemisphere[0]}"
            })
        
        return info
    
    @classmethod
    def is_geographic(cls, crs_code: str) -> bool:
        """Check if CRS uses geographic coordinates (degrees)."""
        return crs_code == cls.WGS84 or "4326" in crs_code
    
    @classmethod
    def is_projected(cls, crs_code: str) -> bool:
        """Check if CRS uses projected coordinates (meters)."""
        return not cls.is_geographic(crs_code)
    
    @classmethod
    def get_units(cls, crs_code: str) -> str:
        """Get the units for a CRS."""
        if cls.is_geographic(crs_code):
            return "degrees"
        else:
            return "meters"


# Global transformer instance
_global_transformer = CoordinateTransformer()


def transform_point(x: float, y: float, src_crs: str, dst_crs: str) -> Tuple[float, float]:
    """Transform a single point between coordinate systems."""
    return _global_transformer.transform_point(x, y, src_crs, dst_crs)


def transform_points(x_coords: np.ndarray, y_coords: np.ndarray, 
                    src_crs: str, dst_crs: str) -> Tuple[np.ndarray, np.ndarray]:
    """Transform multiple points between coordinate systems."""
    return _global_transformer.transform_points(x_coords, y_coords, src_crs, dst_crs)


def transform_bounds(bounds: TerrainBounds, src_crs: str, dst_crs: str) -> TerrainBounds:
    """Transform terrain bounds between coordinate systems."""
    return _global_transformer.transform_bounds(bounds, src_crs, dst_crs)


def detect_optimal_crs(bounds: TerrainBounds, crs: str = CRSRegistry.WGS84) -> str:
    """
    Detect the optimal projected CRS for the given bounds.
    
    Args:
        bounds: Geographic bounds
        crs: Current CRS (should be geographic)
        
    Returns:
        Optimal projected CRS string
    """
    if CRSRegistry.is_geographic(crs):
        # For geographic coordinates, suggest UTM
        center_lon = (bounds.min_x + bounds.max_x) / 2
        center_lat = (bounds.min_y + bounds.max_y) / 2
        
        utm_zone = UTMUtils.get_utm_zone(center_lon, center_lat)
        return utm_zone.crs_string
    else:
        # Already projected
        return crs


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test Web Mercator transformations
    print("Web Mercator Tests:")
    
    # San Francisco coordinates
    sf_lon, sf_lat = -122.4194, 37.7749
    print(f"San Francisco: {sf_lon}, {sf_lat}")
    
    # Convert to Web Mercator
    sf_x, sf_y = WebMercatorUtils.geographic_to_web_mercator(sf_lon, sf_lat)
    print(f"Web Mercator: {sf_x:.2f}, {sf_y:.2f}")
    
    # Convert back
    sf_lon2, sf_lat2 = WebMercatorUtils.web_mercator_to_geographic(sf_x, sf_y)
    print(f"Back to geographic: {sf_lon2:.6f}, {sf_lat2:.6f}")
    
    # Test tile calculations
    print(f"\nTile Calculations:")
    tile = WebMercatorUtils.geographic_to_tile(sf_lon, sf_lat, 10)
    print(f"Tile at zoom 10: {tile}")
    
    tile_bounds = WebMercatorUtils.tile_bounds(tile.x, tile.y, tile.z)
    print(f"Tile bounds: {tile_bounds}")
    
    # Test UTM detection
    print(f"\nUTM Detection:")
    utm_zone = UTMUtils.get_utm_zone(sf_lon, sf_lat)
    print(f"UTM Zone: {utm_zone}")
    
    # Test CRS registry
    print(f"\nCRS Registry:")
    for crs_name, crs_code in [("WGS84", CRSRegistry.WGS84), 
                               ("Web Mercator", CRSRegistry.WEB_MERCATOR)]:
        info = CRSRegistry.get_crs_info(crs_code)
        print(f"{crs_name}: {info}")
    
    # Test coordinate transformation (if pyproj available)
    if PYPROJ_AVAILABLE:
        print(f"\nCoordinate Transformation:")
        transformer = CoordinateTransformer()
        
        # Transform San Francisco from WGS84 to Web Mercator
        sf_merc_x, sf_merc_y = transformer.transform_point(
            sf_lon, sf_lat, CRSRegistry.WGS84, CRSRegistry.WEB_MERCATOR
        )
        print(f"Transformed SF: {sf_merc_x:.2f}, {sf_merc_y:.2f}")
        
        # Compare with direct calculation
        print(f"Direct calculation difference: "
              f"X: {abs(sf_merc_x - sf_x):.2f}, Y: {abs(sf_merc_y - sf_y):.2f}")
    else:
        print("\nPyproj not available - skipping transformation tests")