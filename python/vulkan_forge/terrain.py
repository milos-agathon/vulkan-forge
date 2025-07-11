"""
High-level terrain rendering API for Vulkan-Forge.

Provides a clean, Pythonic interface to the high-performance C++ terrain rendering system
with support for GeoTIFF data, real-time streaming, and GPU tessellation.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
import threading
import time
import weakref

# Import C++ bindings (would be available after compilation)
try:
    import vulkan_forge._vulkan_forge_native as _native
    NATIVE_AVAILABLE = True
except ImportError:
    _native = None
    NATIVE_AVAILABLE = False
    logging.warning("Native terrain renderer not available - using fallback implementation")

# Import our Python components
from .loaders.geotiff_loader import GeoTiffLoader, TerrainBounds, create_test_geotiff
from .loaders.terrain_cache import TerrainCache, get_default_cache
from .loaders.coordinate_systems import (
    CoordinateTransformer, WebMercatorUtils, UTMUtils, 
    CRSRegistry, transform_bounds
)
from .terrain_config import TerrainConfig, RenderConfig, CacheConfig, LODConfig

logger = logging.getLogger(__name__)


@dataclass
class TerrainStats:
    """Statistics for terrain rendering performance."""
    frame_time: float = 0.0           # Total frame time (ms)
    render_time: float = 0.0          # GPU render time (ms)
    culling_time: float = 0.0         # Culling time (ms)
    tiles_rendered: int = 0           # Number of tiles rendered
    tiles_culled: int = 0             # Number of tiles culled
    triangles_rendered: int = 0       # Total triangles rendered
    draw_calls: int = 0               # Number of draw calls
    memory_usage_mb: float = 0.0      # Memory usage in MB
    gpu_memory_mb: float = 0.0        # GPU memory usage in MB
    cache_hit_ratio: float = 0.0      # Cache hit ratio (0-1)
    
    @property
    def fps(self) -> float:
        """Calculate FPS from frame time."""
        return 1000.0 / self.frame_time if self.frame_time > 0 else 0.0
    
    @property
    def culling_efficiency(self) -> float:
        """Calculate culling efficiency (0-1)."""
        total_tiles = self.tiles_rendered + self.tiles_culled
        return self.tiles_culled / total_tiles if total_tiles > 0 else 0.0


@dataclass
class Camera:
    """Camera parameters for terrain rendering."""
    position: np.ndarray = None       # World position [x, y, z]
    direction: np.ndarray = None      # Look direction [x, y, z] 
    up: np.ndarray = None             # Up vector [x, y, z]
    fov: float = 45.0                 # Field of view in degrees
    near: float = 0.1                 # Near clipping plane
    far: float = 10000.0              # Far clipping plane
    aspect: float = 1.0               # Aspect ratio (width/height)
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.array([0.0, 100.0, 0.0], dtype=np.float32)
        if self.direction is None:
            self.direction = np.array([0.0, -0.7, -0.7], dtype=np.float32)
        if self.up is None:
            self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    def look_at(self, target: np.ndarray):
        """Point camera at target position."""
        direction = target - self.position
        direction = direction / np.linalg.norm(direction)
        self.direction = direction.astype(np.float32)
    
    def move(self, offset: np.ndarray):
        """Move camera by offset vector."""
        self.position += offset.astype(np.float32)
    
    def orbit(self, center: np.ndarray, azimuth_delta: float, elevation_delta: float):
        """Orbit camera around center point."""
        # Calculate current spherical coordinates
        offset = self.position - center
        radius = np.linalg.norm(offset)
        
        current_azimuth = np.arctan2(offset[2], offset[0])
        current_elevation = np.arcsin(offset[1] / radius)
        
        # Apply deltas
        new_azimuth = current_azimuth + azimuth_delta
        new_elevation = np.clip(current_elevation + elevation_delta, -np.pi/2 + 0.1, np.pi/2 - 0.1)
        
        # Convert back to Cartesian
        self.position[0] = center[0] + radius * np.cos(new_elevation) * np.cos(new_azimuth)
        self.position[1] = center[1] + radius * np.sin(new_elevation)
        self.position[2] = center[2] + radius * np.cos(new_elevation) * np.sin(new_azimuth)
        
        # Update direction to look at center
        self.look_at(center)
    
    def set_orbit_position(self, center: np.ndarray, angle_degrees: float, 
                          elevation_degrees: float, distance: float):
        """Position camera in orbit around center point with GL convention."""
        angle_rad = np.radians(angle_degrees)
        elevation_rad = np.radians(elevation_degrees)
        
        # Spherical to Cartesian with GL convention
        x = distance * np.cos(elevation_rad) * np.sin(angle_rad)  # East-west
        y = distance * np.sin(elevation_rad)                      # Height
        z = distance * np.cos(elevation_rad) * np.cos(angle_rad)  # North-south
        
        self.position = center + np.array([x, y, z], dtype=np.float32)
        self.look_at(center)
    
    def get_view_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get camera view vectors (forward, right, up) with GL convention."""
        # Forward vector (from position to target)
        target = self.position + self.direction
        forward = target - self.position
        forward_length = np.linalg.norm(forward)
        if forward_length > 1e-6:
            forward = forward / forward_length
        else:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        
        # Right vector = cross(forward, world_up)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right_length = np.linalg.norm(right)
        if right_length > 1e-6:
            right = right / right_length
        else:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        # Up vector = cross(right, forward)
        up = np.cross(right, forward)
        up_length = np.linalg.norm(up)
        if up_length > 1e-6:
            up = up / up_length
        else:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        return forward, right, up
    
    def project_perspective_gl(self, world_pos: np.ndarray, width: int, height: int) -> Optional[Tuple[int, int, float]]:
        """Project world position to screen coordinates using GL convention perspective projection."""
        forward, right, up = self.get_view_vectors()
        
        # Transform to camera space
        relative_pos = world_pos - self.position
        
        # Project onto camera plane using GL convention
        forward_dist = np.dot(relative_pos, forward)
        right_dist = np.dot(relative_pos, right)
        up_dist = np.dot(relative_pos, up)
        
        if forward_dist <= self.near:
            return None
        
        # Perspective projection
        tan_half_fov = np.tan(np.radians(self.fov / 2))
        x_proj = right_dist / (forward_dist * tan_half_fov * self.aspect)
        y_proj = up_dist / (forward_dist * tan_half_fov)
        
        # Convert to screen coordinates
        x_screen = int((x_proj + 1) * width / 2)
        y_screen = int((1 - y_proj) * height / 2)
        
        return (x_screen, y_screen, forward_dist)
    
    def project_orthographic_gl(self, world_pos: np.ndarray, width: int, height: int,
                               world_bounds: Tuple[float, float, float, float]) -> Tuple[int, int, float]:
        """Project world position to screen coordinates using GL convention orthographic projection."""
        x, y, z = world_pos
        world_x_min, world_x_max, world_z_min, world_z_max = world_bounds
        
        # Map world coordinates to screen coordinates using exact world bounds
        x_norm = (x - world_x_min) / (world_x_max - world_x_min)
        z_norm = (z - world_z_min) / (world_z_max - world_z_min)
        
        x_screen = int(x_norm * (width - 1))
        y_screen = int((1 - z_norm) * (height - 1))  # Flip Z for screen Y
        
        # Use height for depth testing (higher terrain = closer to camera)
        depth = 10.0 - y
        
        return (x_screen, y_screen, depth)


class TerrainDataset:
    """Represents a loaded terrain dataset."""
    
    def __init__(self, dataset_id: str, geotiff_path: Union[str, Path], 
                 config: Optional[TerrainConfig] = None):
        self.dataset_id = dataset_id
        self.geotiff_path = Path(geotiff_path)
        self.config = config or TerrainConfig()
        
        # Initialize loader
        self.loader = GeoTiffLoader(
            backend=self.config.loader_backend,
            cache_size_mb=self.config.gdal_cache_size_mb
        )
        
        # Load metadata
        self._metadata = None
        self._bounds = None
        self._crs = None
        self._loaded = False
        
    def load(self):
        """Load the dataset and extract metadata."""
        if self._loaded:
            return
        
        logger.info(f"Loading terrain dataset: {self.dataset_id}")
        start_time = time.time()
        
        try:
            self._metadata = self.loader.open(str(self.geotiff_path))
            self._bounds = self._metadata.bounds
            self._crs = self._metadata.crs
            self._loaded = True
            
            load_time = time.time() - start_time
            logger.info(f"Dataset loaded in {load_time:.2f}s: "
                       f"{self._metadata.width}x{self._metadata.height}, "
                       f"bounds: {self._bounds}")
            
        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset_id}: {e}")
            raise
    
    def unload(self):
        """Unload the dataset and free resources."""
        if not self._loaded:
            return
        
        self.loader.close()
        self._loaded = False
        logger.info(f"Unloaded dataset: {self.dataset_id}")
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    @property
    def metadata(self):
        if not self._loaded:
            self.load()
        return self._metadata
    
    @property
    def bounds(self) -> TerrainBounds:
        if not self._loaded:
            self.load()
        return self._bounds
    
    @property
    def crs(self) -> str:
        if not self._loaded:
            self.load()
        return self._crs
    
    def get_tile_data(self, tile_x: int, tile_y: int, level: int = 0, 
                     tile_size: int = 512) -> Optional[np.ndarray]:
        """Get height data for a specific tile."""
        if not self._loaded:
            self.load()
        
        return self.loader.read_tile(tile_x, tile_y, tile_size, level)
    
    def get_region_data(self, bounds: TerrainBounds, max_pixels: int = 2048) -> Tuple[np.ndarray, TerrainBounds]:
        """Get height data for a geographic region."""
        if not self._loaded:
            self.load()
        
        return self.loader.read_region(bounds, max_pixels)
    
    def transform_bounds_to_crs(self, target_crs: str) -> TerrainBounds:
        """Transform dataset bounds to target coordinate system."""
        if not self._loaded:
            self.load()
        
        transformer = CoordinateTransformer()
        return transformer.transform_bounds(self._bounds, self._crs, target_crs)
    
    def __enter__(self):
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()


class TerrainRenderer:
    """High-level terrain renderer with GPU tessellation."""
    
    def __init__(self, config: Optional[TerrainConfig] = None):
        self.config = config or TerrainConfig()
        
        # Core components
        self._native_renderer = None
        self._datasets: Dict[str, TerrainDataset] = {}
        self._active_dataset: Optional[str] = None
        self._cache = get_default_cache()
        
        # Rendering state
        self._initialized = False
        self._rendering = False
        self._render_thread: Optional[threading.Thread] = None
        self._stop_rendering = threading.Event()
        
        # Statistics
        self._stats = TerrainStats()
        self._frame_callbacks: List[Callable[[TerrainStats], None]] = []
        
        # Coordinate transformer
        self._transformer = CoordinateTransformer()
        
        logger.info("Terrain renderer created")
    
    def initialize(self, width: int = 1920, height: int = 1080):
        """Initialize the terrain renderer."""
        if self._initialized:
            return
        
        logger.info(f"Initializing terrain renderer: {width}x{height}")
        
        # Initialize native renderer if available
        if NATIVE_AVAILABLE:
            try:
                self._native_renderer = _native.TerrainRenderer()
                self._native_renderer.initialize(self.config.to_native_config())
                self._native_renderer.set_viewport(width, height)
                logger.info("Native terrain renderer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize native renderer: {e}")
                self._native_renderer = None
        
        # Configure cache
        cache_config = self.config.cache
        self._cache.set_max_tiles(cache_config.max_tiles)
        self._cache.set_max_memory_usage(cache_config.max_memory_mb * 1024 * 1024)
        
        self._initialized = True
        logger.info("Terrain renderer initialized")
    
    def destroy(self):
        """Destroy the renderer and free resources."""
        if not self._initialized:
            return
        
        # Stop rendering
        self.stop_rendering()
        
        # Unload all datasets
        for dataset in list(self._datasets.values()):
            dataset.unload()
        self._datasets.clear()
        
        # Destroy native renderer
        if self._native_renderer:
            self._native_renderer.destroy()
            self._native_renderer = None
        
        self._initialized = False
        logger.info("Terrain renderer destroyed")
    
    def load_dataset(self, dataset_id: str, geotiff_path: Union[str, Path], 
                    set_active: bool = True) -> TerrainDataset:
        """Load a terrain dataset from GeoTIFF file."""
        if not self._initialized:
            self.initialize()
        
        if dataset_id in self._datasets:
            logger.warning(f"Dataset {dataset_id} already loaded")
            return self._datasets[dataset_id]
        
        # Create and load dataset
        dataset = TerrainDataset(dataset_id, geotiff_path, self.config)
        dataset.load()
        
        # Register with cache
        self._cache.register_loader(dataset_id, dataset.loader)
        
        # Register with native renderer
        if self._native_renderer:
            try:
                self._native_renderer.load_dataset(dataset_id, str(geotiff_path))
            except Exception as e:
                logger.warning(f"Failed to load dataset in native renderer: {e}")
        
        self._datasets[dataset_id] = dataset
        
        if set_active or self._active_dataset is None:
            self.set_active_dataset(dataset_id)
        
        logger.info(f"Loaded terrain dataset: {dataset_id}")
        return dataset
    
    def unload_dataset(self, dataset_id: str):
        """Unload a terrain dataset."""
        if dataset_id not in self._datasets:
            logger.warning(f"Dataset {dataset_id} not found")
            return
        
        # Unregister from cache
        self._cache.unregister_loader(dataset_id)
        
        # Unregister from native renderer
        if self._native_renderer:
            try:
                self._native_renderer.unload_dataset(dataset_id)
            except Exception as e:
                logger.warning(f"Failed to unload dataset from native renderer: {e}")
        
        # Unload dataset
        dataset = self._datasets.pop(dataset_id)
        dataset.unload()
        
        # Update active dataset
        if self._active_dataset == dataset_id:
            self._active_dataset = next(iter(self._datasets.keys()), None)
            if self._active_dataset and self._native_renderer:
                self._native_renderer.set_active_dataset(self._active_dataset)
        
        logger.info(f"Unloaded terrain dataset: {dataset_id}")
    
    def set_active_dataset(self, dataset_id: str):
        """Set the active dataset for rendering."""
        if dataset_id not in self._datasets:
            raise ValueError(f"Dataset {dataset_id} not loaded")
        
        self._active_dataset = dataset_id
        
        if self._native_renderer:
            self._native_renderer.set_active_dataset(dataset_id)
        
        logger.info(f"Active dataset set to: {dataset_id}")
    
    def get_active_dataset(self) -> Optional[TerrainDataset]:
        """Get the currently active dataset."""
        if self._active_dataset and self._active_dataset in self._datasets:
            return self._datasets[self._active_dataset]
        return None
    
    def render_frame(self, camera: Camera) -> np.ndarray:
        """Render a single frame and return the image."""
        if not self._initialized:
            raise RuntimeError("Renderer not initialized")
        
        if not self._active_dataset:
            raise RuntimeError("No active dataset")
        
        start_time = time.time()
        
        # Use native renderer if available
        if self._native_renderer:
            try:
                # Convert camera to native format
                native_camera = self._camera_to_native(camera)
                
                # Render frame
                image_data = self._native_renderer.render(native_camera)
                
                # Update statistics
                self._update_stats_from_native()
                
                return np.array(image_data, dtype=np.uint8).reshape((camera.aspect * 1080, 1080, 4))
                
            except Exception as e:
                logger.error(f"Native rendering failed: {e}")
                # Fall through to software fallback
        
        # Software fallback rendering
        return self._render_frame_software(camera)
    
    def start_rendering(self, camera: Camera, target_fps: float = 60.0, 
                       frame_callback: Optional[Callable[[np.ndarray, TerrainStats], None]] = None):
        """Start continuous rendering in background thread."""
        if self._rendering:
            logger.warning("Already rendering")
            return
        
        self._rendering = True
        self._stop_rendering.clear()
        
        def render_loop():
            frame_time = 1.0 / target_fps
            
            while not self._stop_rendering.is_set():
                loop_start = time.time()
                
                try:
                    # Render frame
                    image = self.render_frame(camera)
                    
                    # Call frame callback
                    if frame_callback:
                        frame_callback(image, self._stats)
                    
                    # Call registered callbacks
                    for callback in self._frame_callbacks:
                        try:
                            callback(self._stats)
                        except Exception as e:
                            logger.error(f"Frame callback error: {e}")
                
                except Exception as e:
                    logger.error(f"Rendering error: {e}")
                
                # Sleep to maintain target FPS
                elapsed = time.time() - loop_start
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        self._render_thread = threading.Thread(target=render_loop, daemon=True)
        self._render_thread.start()
        
        logger.info(f"Started rendering at {target_fps} FPS")
    
    def stop_rendering(self):
        """Stop continuous rendering."""
        if not self._rendering:
            return
        
        self._stop_rendering.set()
        
        if self._render_thread and self._render_thread.is_alive():
            self._render_thread.join(timeout=2.0)
        
        self._rendering = False
        self._render_thread = None
        
        logger.info("Stopped rendering")
    
    def add_frame_callback(self, callback: Callable[[TerrainStats], None]):
        """Add a callback that gets called every frame."""
        self._frame_callbacks.append(callback)
    
    def remove_frame_callback(self, callback: Callable[[TerrainStats], None]):
        """Remove a frame callback."""
        if callback in self._frame_callbacks:
            self._frame_callbacks.remove(callback)
    
    def get_stats(self) -> TerrainStats:
        """Get current rendering statistics."""
        return self._stats
    
    def reset_stats(self):
        """Reset rendering statistics."""
        self._stats = TerrainStats()
        if self._native_renderer:
            self._native_renderer.reset_stats()
    
    def update_config(self, config: TerrainConfig):
        """Update renderer configuration."""
        self.config = config
        
        if self._native_renderer:
            self._native_renderer.update_config(config.to_native_config())
        
        # Update cache configuration
        cache_config = config.cache
        self._cache.set_max_tiles(cache_config.max_tiles)
        self._cache.set_max_memory_usage(cache_config.max_memory_mb * 1024 * 1024)
    
    def get_height_at_position(self, position: np.ndarray) -> float:
        """Get terrain height at world position."""
        if not self._active_dataset:
            return 0.0
        
        if self._native_renderer:
            try:
                return self._native_renderer.get_height_at_position(position[:2])
            except Exception as e:
                logger.debug(f"Native height query failed: {e}")
        
        # Fallback to dataset query
        dataset = self.get_active_dataset()
        if dataset and dataset.is_loaded:
            # Convert world position to dataset coordinates
            # This is simplified - would need proper coordinate transformation
            bounds = dataset.bounds
            x_ratio = (position[0] - bounds.min_x) / (bounds.max_x - bounds.min_x)
            y_ratio = (position[2] - bounds.min_y) / (bounds.max_y - bounds.min_y)
            
            if 0 <= x_ratio <= 1 and 0 <= y_ratio <= 1:
                # Sample height data at position
                tile_x = int(x_ratio * 10)  # Simplified tile calculation
                tile_y = int(y_ratio * 10)
                tile_data = dataset.get_tile_data(tile_x, tile_y)
                
                if tile_data is not None:
                    # Interpolate height from tile
                    h, w = tile_data.shape
                    sample_x = int((x_ratio * 10 - tile_x) * w)
                    sample_y = int((y_ratio * 10 - tile_y) * h)
                    sample_x = np.clip(sample_x, 0, w - 1)
                    sample_y = np.clip(sample_y, 0, h - 1)
                    return float(tile_data[sample_y, sample_x])
        
        return 0.0
    
    def get_terrain_bounds(self, crs: Optional[str] = None) -> Optional[TerrainBounds]:
        """Get bounds of active terrain dataset."""
        dataset = self.get_active_dataset()
        if not dataset:
            return None
        
        bounds = dataset.bounds
        
        if crs and crs != dataset.crs:
            bounds = dataset.transform_bounds_to_crs(crs)
        
        return bounds
    
    def create_optimal_camera(self) -> Camera:
        """Create a camera with optimal settings for the active dataset."""
        bounds = self.get_terrain_bounds()
        if not bounds:
            return Camera()
        
        # Position camera to view entire dataset
        center_x = (bounds.min_x + bounds.max_x) / 2
        center_z = (bounds.min_y + bounds.max_y) / 2
        center_y = (bounds.min_z + bounds.max_z) / 2
        
        # Calculate distance to see entire terrain
        terrain_size = max(bounds.max_x - bounds.min_x, bounds.max_y - bounds.min_y)
        distance = terrain_size * 0.8
        
        camera = Camera()
        camera.position = np.array([center_x, center_y + distance * 0.5, center_z + distance], dtype=np.float32)
        camera.look_at(np.array([center_x, center_y, center_z], dtype=np.float32))
        camera.far = distance * 3.0
        
        return camera
    
    def benchmark_performance(self, duration: float = 10.0, 
                            camera: Optional[Camera] = None) -> Dict[str, Any]:
        """Run performance benchmark and return results."""
        if camera is None:
            camera = self.create_optimal_camera()
        
        logger.info(f"Starting {duration}s performance benchmark")
        
        # Reset statistics
        self.reset_stats()
        
        # Render frames for specified duration
        start_time = time.time()
        frame_count = 0
        min_frame_time = float('inf')
        max_frame_time = 0.0
        total_frame_time = 0.0
        
        while time.time() - start_time < duration:
            frame_start = time.time()
            
            try:
                image = self.render_frame(camera)
                frame_count += 1
                
                frame_time = (time.time() - frame_start) * 1000.0  # Convert to ms
                min_frame_time = min(min_frame_time, frame_time)
                max_frame_time = max(max_frame_time, frame_time)
                total_frame_time += frame_time
                
            except Exception as e:
                logger.error(f"Benchmark frame failed: {e}")
        
        actual_duration = time.time() - start_time
        
        # Calculate statistics
        avg_frame_time = total_frame_time / frame_count if frame_count > 0 else 0
        avg_fps = frame_count / actual_duration if actual_duration > 0 else 0
        
        stats = self.get_stats()
        
        results = {
            'duration': actual_duration,
            'frame_count': frame_count,
            'avg_fps': avg_fps,
            'min_fps': 1000.0 / max_frame_time if max_frame_time > 0 else 0,
            'max_fps': 1000.0 / min_frame_time if min_frame_time < float('inf') else 0,
            'avg_frame_time_ms': avg_frame_time,
            'min_frame_time_ms': min_frame_time if min_frame_time < float('inf') else 0,
            'max_frame_time_ms': max_frame_time,
            'triangles_per_second': stats.triangles_rendered * avg_fps,
            'memory_usage_mb': stats.memory_usage_mb,
            'gpu_memory_mb': stats.gpu_memory_mb,
            'cache_hit_ratio': stats.cache_hit_ratio,
            'culling_efficiency': stats.culling_efficiency
        }
        
        logger.info(f"Benchmark complete: {avg_fps:.1f} FPS average, "
                   f"{results['triangles_per_second']:.0f} triangles/sec")
        
        return results
    
    def export_heightmap(self, bounds: TerrainBounds, width: int, height: int, 
                        output_path: Union[str, Path]) -> np.ndarray:
        """Export heightmap data to file."""
        dataset = self.get_active_dataset()
        if not dataset:
            raise RuntimeError("No active dataset")
        
        # Read height data for region
        height_data, actual_bounds = dataset.get_region_data(bounds, max(width, height))
        
        # Resize to target dimensions if needed
        if height_data.shape != (height, width):
            from scipy.ndimage import zoom
            scale_y = height / height_data.shape[0]
            scale_x = width / height_data.shape[1]
            height_data = zoom(height_data, (scale_y, scale_x), order=1)
        
        # Save to file
        output_path = Path(output_path)
        if output_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            # Save as image
            from PIL import Image
            
            # Normalize to 0-255 range
            normalized = ((height_data - height_data.min()) / 
                         (height_data.max() - height_data.min()) * 255).astype(np.uint8)
            
            Image.fromarray(normalized).save(output_path)
        else:
            # Save as numpy array
            np.save(output_path, height_data)
        
        logger.info(f"Exported heightmap: {output_path}")
        return height_data
    
    def _camera_to_native(self, camera: Camera):
        """Convert Python camera to native format."""
        if not self._native_renderer:
            return None
        
        # This would convert the camera to the native C++ format
        # For now, return a placeholder
        return {
            'position': camera.position.tolist(),
            'direction': camera.direction.tolist(),
            'up': camera.up.tolist(),
            'fov': camera.fov,
            'near': camera.near,
            'far': camera.far,
            'aspect': camera.aspect
        }
    
    def _update_stats_from_native(self):
        """Update statistics from native renderer."""
        if not self._native_renderer:
            return
        
        try:
            native_stats = self._native_renderer.get_stats()
            
            self._stats.frame_time = native_stats.get('frame_time', 0.0)
            self._stats.render_time = native_stats.get('render_time', 0.0)
            self._stats.culling_time = native_stats.get('culling_time', 0.0)
            self._stats.tiles_rendered = native_stats.get('tiles_rendered', 0)
            self._stats.tiles_culled = native_stats.get('tiles_culled', 0)
            self._stats.triangles_rendered = native_stats.get('triangles_rendered', 0)
            self._stats.draw_calls = native_stats.get('draw_calls', 0)
            self._stats.memory_usage_mb = native_stats.get('memory_usage', 0) / (1024 * 1024)
            self._stats.gpu_memory_mb = native_stats.get('gpu_memory_usage', 0) / (1024 * 1024)
            
            # Get cache statistics
            cache_stats = self._cache.get_stats()
            self._stats.cache_hit_ratio = cache_stats.hit_ratio
            
        except Exception as e:
            logger.debug(f"Failed to update stats from native renderer: {e}")
    
    def _render_frame_software(self, camera: Camera) -> np.ndarray:
        """Software fallback rendering (basic implementation)."""
        logger.debug("Using software fallback renderer")
        
        # Create a simple colored image as fallback
        height = int(1080)
        width = int(height * camera.aspect)
        
        # Generate a gradient based on camera position
        image = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Simple gradient based on camera height
        altitude_factor = camera.position[1] / 1000.0
        
        for y in range(height):
            for x in range(width):
                # Create a terrain-like pattern
                r = int(128 + 127 * np.sin(x * 0.01 + altitude_factor))
                g = int(128 + 127 * np.sin(y * 0.01 + altitude_factor))
                b = int(64 + 64 * altitude_factor)
                
                image[y, x] = [r, g, b, 255]
        
        # Update stats for fallback
        self._stats.frame_time = 16.7  # ~60 FPS
        self._stats.triangles_rendered = 1000  # Placeholder
        
        return image
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()


# Utility functions
def create_test_terrain(width: int = 2048, height: int = 2048, 
                       output_path: Optional[Union[str, Path]] = None) -> Path:
    """Create a test terrain GeoTIFF file for development and testing."""
    if output_path is None:
        output_path = Path("test_terrain.tif")
    else:
        output_path = Path(output_path)
    
    # Define terrain bounds (example: 10km x 10km area)
    bounds = TerrainBounds(
        min_x=0, max_x=10000,
        min_y=0, max_y=10000,
        min_z=0, max_z=500
    )
    
    create_test_geotiff(output_path, width, height, bounds)
    
    logger.info(f"Created test terrain: {output_path}")
    return output_path


def load_terrain_from_file(geotiff_path: Union[str, Path], 
                          config: Optional[TerrainConfig] = None) -> TerrainRenderer:
    """Convenience function to create renderer and load terrain from file."""
    renderer = TerrainRenderer(config)
    renderer.initialize()
    
    dataset_id = Path(geotiff_path).stem
    renderer.load_dataset(dataset_id, geotiff_path)
    
    return renderer


# Export public API
__all__ = [
    'TerrainRenderer',
    'TerrainDataset', 
    'TerrainStats',
    'Camera',
    'create_test_terrain',
    'load_terrain_from_file'
]