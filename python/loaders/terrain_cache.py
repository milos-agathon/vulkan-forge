"""
Tile-based caching system for massive terrain datasets.

Provides high-performance, memory-efficient caching of terrain tiles with:
- LRU eviction policy
- Multi-level detail (LOD) support
- Background prefetching
- Memory usage monitoring
- Thread-safe operations
- GPU buffer management integration
"""

import numpy as np
import threading
import time
import logging
import weakref
import gc
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Any, Callable, Set
from threading import RLock, Event, Thread
from queue import Queue, Empty
import psutil
import sys

from .geotiff_loader import TerrainBounds, TerrainTileInfo, GeoTiffLoader

logger = logging.getLogger(__name__)


@dataclass
class TileKey:
    """Unique identifier for a terrain tile."""
    x: int
    y: int
    level: int
    dataset_id: str
    
    def __hash__(self) -> int:
        return hash((self.x, self.y, self.level, self.dataset_id))
    
    def __str__(self) -> str:
        return f"Tile({self.x}, {self.y}, L{self.level}, {self.dataset_id})"


@dataclass
class CachedTile:
    """Cached terrain tile with metadata."""
    key: TileKey
    data: np.ndarray
    bounds: TerrainBounds
    size_bytes: int
    access_time: float = field(default_factory=time.time)
    access_count: int = 0
    gpu_buffer_id: Optional[int] = None
    
    def update_access(self):
        """Update access statistics."""
        self.access_time = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    prefetch_hits: int = 0
    total_tiles: int = 0
    memory_usage_bytes: int = 0
    gpu_memory_bytes: int = 0
    
    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def memory_usage_mb(self) -> float:
        return self.memory_usage_bytes / (1024 * 1024)
    
    @property
    def gpu_memory_mb(self) -> float:
        return self.gpu_memory_bytes / (1024 * 1024)


class TerrainCache:
    """
    High-performance tile cache for terrain data.
    
    Features:
    - LRU eviction with configurable memory limits
    - Background prefetching based on usage patterns
    - GPU buffer management integration
    - Memory monitoring and automatic GC
    - Thread-safe operations
    """
    
    def __init__(self,
                 max_memory_mb: int = 1024,
                 max_tiles: int = 1000,
                 prefetch_enabled: bool = True,
                 prefetch_radius: int = 2,
                 memory_pressure_threshold: float = 0.85,
                 cleanup_interval: float = 30.0):
        """
        Initialize terrain cache.
        
        Args:
            max_memory_mb: Maximum memory usage in megabytes
            max_tiles: Maximum number of cached tiles
            prefetch_enabled: Enable background prefetching
            prefetch_radius: Radius for prefetching neighboring tiles
            memory_pressure_threshold: Trigger cleanup when memory usage exceeds this ratio
            cleanup_interval: Interval between memory cleanup runs (seconds)
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_tiles = max_tiles
        self.prefetch_enabled = prefetch_enabled
        self.prefetch_radius = prefetch_radius
        self.memory_pressure_threshold = memory_pressure_threshold
        self.cleanup_interval = cleanup_interval
        
        # Thread safety
        self._lock = RLock()
        
        # Cache storage (LRU order)
        self._tiles: OrderedDict[TileKey, CachedTile] = OrderedDict()
        
        # Statistics
        self._stats = CacheStats()
        
        # Prefetching system
        self._prefetch_queue: Queue[TileKey] = Queue()
        self._prefetch_active: Set[TileKey] = set()
        self._prefetch_workers: List[Thread] = []
        self._shutdown_event = Event()
        
        # GPU buffer tracking
        self._gpu_buffers: Dict[TileKey, int] = {}
        self._gpu_buffer_callback: Optional[Callable] = None
        
        # Loaders registry
        self._loaders: Dict[str, GeoTiffLoader] = {}
        
        # Access pattern tracking for smart prefetching
        self._access_pattern: List[TileKey] = []
        self._pattern_lock = threading.Lock()
        
        # Background cleanup
        self._cleanup_thread: Optional[Thread] = None
        
        self._start_background_workers()
        
        logger.info(f"TerrainCache initialized: {max_memory_mb}MB, {max_tiles} tiles")
    
    def _start_background_workers(self):
        """Start background worker threads."""
        if self.prefetch_enabled:
            # Start prefetch workers
            num_workers = min(2, max(1, psutil.cpu_count() // 4))
            for i in range(num_workers):
                worker = Thread(
                    target=self._prefetch_worker,
                    name=f"TerrainCache-Prefetch-{i}",
                    daemon=True
                )
                worker.start()
                self._prefetch_workers.append(worker)
        
        # Start cleanup worker
        self._cleanup_thread = Thread(
            target=self._cleanup_worker,
            name="TerrainCache-Cleanup",
            daemon=True
        )
        self._cleanup_thread.start()
    
    def register_loader(self, dataset_id: str, loader: GeoTiffLoader):
        """Register a GeoTIFF loader for a dataset."""
        with self._lock:
            self._loaders[dataset_id] = loader
        logger.debug(f"Registered loader for dataset: {dataset_id}")
    
    def unregister_loader(self, dataset_id: str):
        """Unregister a loader and evict all its tiles."""
        with self._lock:
            if dataset_id in self._loaders:
                del self._loaders[dataset_id]
                
                # Evict all tiles from this dataset
                to_remove = [key for key in self._tiles.keys() if key.dataset_id == dataset_id]
                for key in to_remove:
                    self._evict_tile(key)
        
        logger.debug(f"Unregistered loader for dataset: {dataset_id}")
    
    def set_gpu_buffer_callback(self, callback: Callable[[TileKey, np.ndarray], int]):
        """
        Set callback for GPU buffer allocation.
        
        Args:
            callback: Function that takes (tile_key, data) and returns GPU buffer ID
        """
        self._gpu_buffer_callback = callback
    
    def get_tile(self, 
                 dataset_id: str,
                 x: int, 
                 y: int, 
                 level: int = 0,
                 prefetch_neighbors: bool = True) -> Optional[CachedTile]:
        """
        Get a terrain tile from cache or load if necessary.
        
        Args:
            dataset_id: Dataset identifier
            x: Tile X coordinate
            y: Tile Y coordinate
            level: LOD level
            prefetch_neighbors: Whether to prefetch neighboring tiles
            
        Returns:
            Cached tile or None if loading failed
        """
        key = TileKey(x, y, level, dataset_id)
        
        with self._lock:
            # Check cache first
            if key in self._tiles:
                tile = self._tiles[key]
                tile.update_access()
                
                # Move to end (most recently used)
                self._tiles.move_to_end(key)
                self._stats.hits += 1
                
                # Track access pattern
                self._track_access(key)
                
                # Schedule neighbor prefetching
                if prefetch_neighbors and self.prefetch_enabled:
                    self._schedule_neighbor_prefetch(key)
                
                logger.debug(f"Cache hit: {key}")
                return tile
            
            # Cache miss - try to load
            self._stats.misses += 1
            logger.debug(f"Cache miss: {key}")
            
            # Check if we have a loader for this dataset
            if dataset_id not in self._loaders:
                logger.warning(f"No loader registered for dataset: {dataset_id}")
                return None
            
            # Load tile
            tile = self._load_tile(key)
            if tile:
                self._track_access(key)
                if prefetch_neighbors and self.prefetch_enabled:
                    self._schedule_neighbor_prefetch(key)
            
            return tile
    
    def _load_tile(self, key: TileKey) -> Optional[CachedTile]:
        """Load a tile from storage."""
        loader = self._loaders.get(key.dataset_id)
        if not loader:
            return None
        
        try:
            # Load tile data
            data = loader.read_tile(key.x, key.y, tile_size=512, level=key.level)
            if data is None:
                logger.debug(f"No data available for tile: {key}")
                return None
            
            # Calculate size
            size_bytes = data.nbytes
            
            # Get bounds from loader metadata
            if loader.metadata:
                # Calculate tile bounds based on metadata
                pixel_size_x = loader.metadata.pixel_size_x * (2 ** key.level)
                pixel_size_y = loader.metadata.pixel_size_y * (2 ** key.level)
                tile_size = 512
                
                min_x = loader.metadata.bounds.min_x + key.x * tile_size * pixel_size_x
                max_x = min_x + tile_size * pixel_size_x
                max_y = loader.metadata.bounds.max_y - key.y * tile_size * pixel_size_y
                min_y = max_y - tile_size * pixel_size_y
                
                bounds = TerrainBounds(
                    min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
                    min_z=float(np.min(data)), max_z=float(np.max(data))
                )
            else:
                # Fallback bounds
                bounds = TerrainBounds(
                    min_x=key.x * 512, max_x=(key.x + 1) * 512,
                    min_y=key.y * 512, max_y=(key.y + 1) * 512,
                    min_z=float(np.min(data)), max_z=float(np.max(data))
                )
            
            # Create cached tile
            tile = CachedTile(
                key=key,
                data=data,
                bounds=bounds,
                size_bytes=size_bytes
            )
            
            # Add to cache
            self._add_tile(tile)
            
            # Create GPU buffer if callback is set
            if self._gpu_buffer_callback:
                try:
                    gpu_buffer_id = self._gpu_buffer_callback(key, data)
                    tile.gpu_buffer_id = gpu_buffer_id
                    self._gpu_buffers[key] = gpu_buffer_id
                    self._stats.gpu_memory_bytes += size_bytes
                except Exception as e:
                    logger.warning(f"Failed to create GPU buffer for {key}: {e}")
            
            logger.debug(f"Loaded tile: {key}, size: {size_bytes} bytes")
            return tile
            
        except Exception as e:
            logger.error(f"Failed to load tile {key}: {e}")
            return None
    
    def _add_tile(self, tile: CachedTile):
        """Add tile to cache with eviction if necessary."""
        # Add tile
        self._tiles[tile.key] = tile
        self._stats.total_tiles = len(self._tiles)
        self._stats.memory_usage_bytes += tile.size_bytes
        
        # Check if eviction is needed
        while (len(self._tiles) > self.max_tiles or 
               self._stats.memory_usage_bytes > self.max_memory_bytes):
            self._evict_lru_tile()
    
    def _evict_lru_tile(self):
        """Evict least recently used tile."""
        if not self._tiles:
            return
        
        # Get LRU tile (first in OrderedDict)
        key, tile = self._tiles.popitem(last=False)
        self._evict_tile_data(tile)
        
        self._stats.evictions += 1
        self._stats.total_tiles = len(self._tiles)
        
        logger.debug(f"Evicted LRU tile: {key}")
    
    def _evict_tile(self, key: TileKey):
        """Evict a specific tile."""
        if key in self._tiles:
            tile = self._tiles.pop(key)
            self._evict_tile_data(tile)
            self._stats.total_tiles = len(self._tiles)
    
    def _evict_tile_data(self, tile: CachedTile):
        """Clean up tile data and GPU resources."""
        self._stats.memory_usage_bytes -= tile.size_bytes
        
        # Clean up GPU buffer
        if tile.gpu_buffer_id is not None:
            # Note: GPU buffer cleanup would be handled by the callback system
            if tile.key in self._gpu_buffers:
                del self._gpu_buffers[tile.key]
            self._stats.gpu_memory_bytes -= tile.size_bytes
        
        # Explicit data cleanup
        del tile.data
    
    def _track_access(self, key: TileKey):
        """Track tile access for pattern analysis."""
        with self._pattern_lock:
            self._access_pattern.append(key)
            
            # Keep only recent accesses
            max_pattern_size = 100
            if len(self._access_pattern) > max_pattern_size:
                self._access_pattern = self._access_pattern[-max_pattern_size:]
    
    def _schedule_neighbor_prefetch(self, center_key: TileKey):
        """Schedule prefetching of neighboring tiles."""
        if not self.prefetch_enabled:
            return
        
        # Generate neighbor coordinates
        neighbors = []
        for dx in range(-self.prefetch_radius, self.prefetch_radius + 1):
            for dy in range(-self.prefetch_radius, self.prefetch_radius + 1):
                if dx == 0 and dy == 0:
                    continue  # Skip center tile
                
                neighbor_key = TileKey(
                    x=center_key.x + dx,
                    y=center_key.y + dy,
                    level=center_key.level,
                    dataset_id=center_key.dataset_id
                )
                
                # Check if not already cached or being prefetched
                if (neighbor_key not in self._tiles and 
                    neighbor_key not in self._prefetch_active):
                    neighbors.append(neighbor_key)
        
        # Add to prefetch queue
        for neighbor in neighbors:
            try:
                self._prefetch_queue.put_nowait(neighbor)
                self._prefetch_active.add(neighbor)
            except:
                break  # Queue full
    
    def _prefetch_worker(self):
        """Background worker for tile prefetching."""
        while not self._shutdown_event.is_set():
            try:
                # Get next tile to prefetch
                key = self._prefetch_queue.get(timeout=1.0)
                
                try:
                    # Remove from active set
                    self._prefetch_active.discard(key)
                    
                    # Check if still needed
                    with self._lock:
                        if key in self._tiles:
                            self._stats.prefetch_hits += 1
                            continue
                        
                        # Check memory pressure
                        memory_usage_ratio = self._stats.memory_usage_bytes / self.max_memory_bytes
                        if memory_usage_ratio > self.memory_pressure_threshold:
                            continue  # Skip prefetching under memory pressure
                        
                        # Load tile
                        self._load_tile(key)
                        
                except Exception as e:
                    logger.debug(f"Prefetch failed for {key}: {e}")
                finally:
                    self._prefetch_queue.task_done()
                    
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Prefetch worker error: {e}")
    
    def _cleanup_worker(self):
        """Background worker for memory cleanup."""
        while not self._shutdown_event.is_set():
            try:
                time.sleep(self.cleanup_interval)
                
                # Check system memory pressure
                system_memory = psutil.virtual_memory()
                if system_memory.percent > 80:  # System memory pressure
                    with self._lock:
                        # Aggressive cleanup - remove 25% of cached tiles
                        tiles_to_remove = len(self._tiles) // 4
                        for _ in range(tiles_to_remove):
                            if self._tiles:
                                self._evict_lru_tile()
                    
                    # Force garbage collection
                    gc.collect()
                    logger.info(f"Memory cleanup triggered: removed {tiles_to_remove} tiles")
                
                # Log statistics periodically
                self._log_stats()
                
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
    
    def _log_stats(self):
        """Log cache statistics."""
        stats = self.get_stats()
        logger.debug(
            f"Cache stats: {stats.total_tiles} tiles, "
            f"{stats.memory_usage_mb:.1f}MB RAM, "
            f"{stats.gpu_memory_mb:.1f}MB GPU, "
            f"hit ratio: {stats.hit_ratio:.2f}"
        )
    
    def prefetch_region(self, 
                       dataset_id: str,
                       bounds: TerrainBounds,
                       level: int = 0,
                       tile_size: int = 512):
        """
        Prefetch all tiles in a geographic region.
        
        Args:
            dataset_id: Dataset identifier
            bounds: Geographic bounds to prefetch
            level: LOD level
            tile_size: Tile size in pixels
        """
        if not self.prefetch_enabled:
            return
        
        loader = self._loaders.get(dataset_id)
        if not loader or not loader.metadata:
            return
        
        # Calculate tile coordinates for the region
        metadata = loader.metadata
        
        # Convert bounds to tile coordinates
        pixel_size_x = metadata.pixel_size_x * (2 ** level)
        pixel_size_y = metadata.pixel_size_y * (2 ** level)
        
        tile_x_min = int((bounds.min_x - metadata.bounds.min_x) / (tile_size * pixel_size_x))
        tile_x_max = int((bounds.max_x - metadata.bounds.min_x) / (tile_size * pixel_size_x))
        tile_y_min = int((metadata.bounds.max_y - bounds.max_y) / (tile_size * pixel_size_y))
        tile_y_max = int((metadata.bounds.max_y - bounds.min_y) / (tile_size * pixel_size_y))
        
        # Clamp to valid range
        tile_x_min = max(0, tile_x_min)
        tile_x_max = min(tile_x_max, (metadata.width // (2 ** level)) // tile_size)
        tile_y_min = max(0, tile_y_min)
        tile_y_max = min(tile_y_max, (metadata.height // (2 ** level)) // tile_size)
        
        # Schedule prefetching
        tiles_scheduled = 0
        for ty in range(tile_y_min, tile_y_max + 1):
            for tx in range(tile_x_min, tile_x_max + 1):
                key = TileKey(tx, ty, level, dataset_id)
                
                if (key not in self._tiles and 
                    key not in self._prefetch_active):
                    try:
                        self._prefetch_queue.put_nowait(key)
                        self._prefetch_active.add(key)
                        tiles_scheduled += 1
                    except:
                        break  # Queue full
        
        logger.info(f"Scheduled {tiles_scheduled} tiles for prefetching in region {bounds}")
    
    def clear(self, dataset_id: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            dataset_id: If specified, only clear tiles from this dataset
        """
        with self._lock:
            if dataset_id:
                # Clear specific dataset
                to_remove = [key for key in self._tiles.keys() if key.dataset_id == dataset_id]
                for key in to_remove:
                    self._evict_tile(key)
                logger.info(f"Cleared cache for dataset: {dataset_id}")
            else:
                # Clear all
                for tile in self._tiles.values():
                    self._evict_tile_data(tile)
                self._tiles.clear()
                self._gpu_buffers.clear()
                self._stats = CacheStats()
                logger.info("Cleared entire cache")
    
    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                prefetch_hits=self._stats.prefetch_hits,
                total_tiles=self._stats.total_tiles,
                memory_usage_bytes=self._stats.memory_usage_bytes,
                gpu_memory_bytes=self._stats.gpu_memory_bytes
            )
    
    def get_tile_keys(self) -> List[TileKey]:
        """Get list of all cached tile keys."""
        with self._lock:
            return list(self._tiles.keys())
    
    def is_tile_cached(self, dataset_id: str, x: int, y: int, level: int = 0) -> bool:
        """Check if a tile is in cache."""
        key = TileKey(x, y, level, dataset_id)
        with self._lock:
            return key in self._tiles
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get detailed memory usage information."""
        with self._lock:
            system_memory = psutil.virtual_memory()
            
            return {
                'cache_memory_mb': self._stats.memory_usage_bytes / (1024 * 1024),
                'cache_memory_ratio': self._stats.memory_usage_bytes / self.max_memory_bytes,
                'gpu_memory_mb': self._stats.gpu_memory_bytes / (1024 * 1024),
                'system_memory_percent': system_memory.percent,
                'system_memory_available_mb': system_memory.available / (1024 * 1024),
                'tile_count': self._stats.total_tiles,
                'max_tiles': self.max_tiles,
                'prefetch_queue_size': self._prefetch_queue.qsize(),
                'prefetch_active': len(self._prefetch_active)
            }
    
    def shutdown(self):
        """Shutdown cache and cleanup resources."""
        logger.info("Shutting down TerrainCache...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for workers to finish
        for worker in self._prefetch_workers:
            worker.join(timeout=5.0)
        
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
        
        # Clear cache
        self.clear()
        
        logger.info("TerrainCache shutdown complete")
    
    def __del__(self):
        """Destructor - ensure cleanup."""
        try:
            self.shutdown()
        except:
            pass


class TerrainCacheManager:
    """
    Global manager for terrain caches.
    
    Provides a singleton interface for managing multiple terrain caches
    across different datasets and use cases.
    """
    
    _instance: Optional['TerrainCacheManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'TerrainCacheManager':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._caches: Dict[str, TerrainCache] = {}
        self._default_cache: Optional[TerrainCache] = None
        self._cache_lock = RLock()
        self._initialized = True
        
        logger.info("TerrainCacheManager initialized")
    
    def get_cache(self, cache_id: str = "default") -> TerrainCache:
        """Get or create a terrain cache."""
        with self._cache_lock:
            if cache_id not in self._caches:
                # Create new cache with default settings
                cache = TerrainCache(
                    max_memory_mb=1024,
                    max_tiles=1000,
                    prefetch_enabled=True
                )
                self._caches[cache_id] = cache
                
                if cache_id == "default" or self._default_cache is None:
                    self._default_cache = cache
                
                logger.info(f"Created new terrain cache: {cache_id}")
            
            return self._caches[cache_id]
    
    def remove_cache(self, cache_id: str):
        """Remove and shutdown a cache."""
        with self._cache_lock:
            if cache_id in self._caches:
                cache = self._caches.pop(cache_id)
                cache.shutdown()
                
                if self._default_cache is cache:
                    self._default_cache = None
                
                logger.info(f"Removed terrain cache: {cache_id}")
    
    def get_default_cache(self) -> TerrainCache:
        """Get the default terrain cache."""
        return self.get_cache("default")
    
    def shutdown_all(self):
        """Shutdown all caches."""
        with self._cache_lock:
            for cache_id, cache in list(self._caches.items()):
                cache.shutdown()
            self._caches.clear()
            self._default_cache = None
        
        logger.info("All terrain caches shut down")


# Global cache manager instance
cache_manager = TerrainCacheManager()


def get_default_cache() -> TerrainCache:
    """Get the global default terrain cache."""
    return cache_manager.get_default_cache()


if __name__ == "__main__":
    # Example usage
    import tempfile
    from .geotiff_loader import create_test_geotiff
    
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        test_file = f.name
    
    create_test_geotiff(test_file, width=2048, height=2048)
    
    # Create cache and loader
    cache = TerrainCache(max_memory_mb=256, max_tiles=100)
    loader = GeoTiffLoader()
    
    try:
        # Register loader
        loader.open(test_file)
        cache.register_loader("test", loader)
        
        # Test tile loading
        print("Loading tiles...")
        start_time = time.time()
        
        for y in range(4):
            for x in range(4):
                tile = cache.get_tile("test", x, y, 0)
                if tile:
                    print(f"Loaded tile ({x}, {y}): {tile.data.shape}")
        
        load_time = time.time() - start_time
        print(f"Initial load time: {load_time:.2f}s")
        
        # Test cache hits
        print("\nTesting cache hits...")
        start_time = time.time()
        
        for y in range(4):
            for x in range(4):
                tile = cache.get_tile("test", x, y, 0)
        
        hit_time = time.time() - start_time
        print(f"Cache hit time: {hit_time:.2f}s")
        
        # Print statistics
        stats = cache.get_stats()
        print(f"\nCache statistics:")
        print(f"  Hits: {stats.hits}")
        print(f"  Misses: {stats.misses}")
        print(f"  Hit ratio: {stats.hit_ratio:.2f}")
        print(f"  Memory usage: {stats.memory_usage_mb:.1f}MB")
        print(f"  Total tiles: {stats.total_tiles}")
        
    finally:
        cache.shutdown()
        loader.close()
        
        # Cleanup
        import os
        os.unlink(test_file)