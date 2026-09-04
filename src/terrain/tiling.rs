//! B11: Tiled DEM pyramid & cache system
//!
//! This module provides a quad-tree based tiling system for large DEMs with LRU caching
//! to manage memory usage within the 512 MiB budget.

use crate::core::memory_tracker::global_tracker;
use glam::{Vec2, Vec3};
use std::collections::{HashMap, VecDeque};

/// Unique identifier for a tile in the quad-tree
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileId {
    /// Level of detail (0 = highest resolution)
    pub lod: u32,
    /// X coordinate at this LOD level  
    pub x: u32,
    /// Y coordinate at this LOD level
    pub y: u32,
}

impl TileId {
    pub fn new(lod: u32, x: u32, y: u32) -> Self {
        Self { lod, x, y }
    }

    /// Get the parent tile at the next lower resolution
    pub fn parent(self) -> Option<TileId> {
        if self.lod == 0 {
            None
        } else {
            Some(TileId::new(self.lod - 1, self.x / 2, self.y / 2))
        }
    }

    /// Get the four child tiles at the next higher resolution
    pub fn children(self) -> [TileId; 4] {
        let child_lod = self.lod + 1;
        let base_x = self.x * 2;
        let base_y = self.y * 2;
        [
            TileId::new(child_lod, base_x, base_y),
            TileId::new(child_lod, base_x + 1, base_y),
            TileId::new(child_lod, base_x, base_y + 1),
            TileId::new(child_lod, base_x + 1, base_y + 1),
        ]
    }
}

/// Spatial bounds for a tile in world coordinates
#[derive(Debug, Clone)]
pub struct TileBounds {
    pub min: Vec2,
    pub max: Vec2,
}

impl TileBounds {
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }

    pub fn center(&self) -> Vec2 {
        (self.min + self.max) * 0.5
    }

    pub fn size(&self) -> Vec2 {
        self.max - self.min
    }

    /// Test if point is inside bounds
    pub fn contains_point(&self, point: Vec2) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
    }

    /// Test if this bounds intersects with another
    pub fn intersects(&self, other: &TileBounds) -> bool {
        self.max.x >= other.min.x
            && self.min.x <= other.max.x
            && self.max.y >= other.min.y
            && self.min.y <= other.max.y
    }
}

/// A node in the quad-tree representing a tile
#[derive(Debug)]
pub struct QuadTreeNode {
    pub tile_id: TileId,
    pub bounds: TileBounds,
    pub is_loaded: bool,
}

impl QuadTreeNode {
    pub fn new(tile_id: TileId, bounds: TileBounds) -> Self {
        Self {
            tile_id,
            bounds,
            is_loaded: false,
        }
    }

    /// Calculate bounds for a tile given the root bounds and tile dimensions
    pub fn calculate_bounds(
        root_bounds: &TileBounds,
        tile_id: TileId,
        tile_size: Vec2,
    ) -> TileBounds {
        let lod_scale = 1.0 / (1 << tile_id.lod) as f32;
        let tile_world_size = tile_size * lod_scale;

        let min = root_bounds.min
            + Vec2::new(
                tile_id.x as f32 * tile_world_size.x,
                tile_id.y as f32 * tile_world_size.y,
            );
        let max = min + tile_world_size;

        TileBounds::new(min, max)
    }
}

/// Data for a cached tile
#[derive(Debug)]
pub struct TileData {
    pub tile_id: TileId,
    pub height_data: Vec<f32>, // Height values (width * height)
    pub width: u32,
    pub height: u32,
    pub host_memory_size: u64, // Size in bytes for memory tracking
}

impl TileData {
    pub fn new(tile_id: TileId, height_data: Vec<f32>, width: u32, height: u32) -> Self {
        let host_memory_size = (height_data.len() * std::mem::size_of::<f32>()) as u64;
        Self {
            tile_id,
            height_data,
            width,
            height,
            host_memory_size,
        }
    }
}

/// LRU cache for tile data with memory budget management
pub struct TileCache {
    capacity: usize,
    data: HashMap<TileId, TileData>,
    access_order: VecDeque<TileId>,
    total_memory_usage: u64,
}

impl TileCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            data: HashMap::new(),
            access_order: VecDeque::new(),
            total_memory_usage: 0,
        }
    }

    /// Insert a tile into the cache, evicting old entries if necessary
    pub fn insert(&mut self, tile_data: TileData) -> Result<(), String> {
        let tile_id = tile_data.tile_id;
        let memory_size = tile_data.host_memory_size;

        // Atomically reserve/register before mutating the cache. This keeps
        // tile-cache host memory in the same aggregate budget domain as
        // tracked GPU buffers and scoped CPU reservations.
        let tracker = global_tracker();
        tracker
            .track_buffer_allocation_labeled(memory_size, true, "terrain.tile-cache")
            .map_err(|error| error.to_string())?;

        // If tile already exists, remove it first
        if let Some(old_data) = self.data.remove(&tile_id) {
            self.access_order.retain(|&id| id != tile_id);
            self.total_memory_usage -= old_data.host_memory_size;
            tracker.free_buffer_allocation(old_data.host_memory_size, true);
        }

        // Evict oldest entries if at capacity
        while self.data.len() >= self.capacity && !self.access_order.is_empty() {
            self.evict_oldest()?;
        }

        // Insert new tile (the aggregate registration already succeeded).
        self.total_memory_usage += memory_size;
        self.data.insert(tile_id, tile_data);
        self.access_order.push_back(tile_id);

        Ok(())
    }

    /// Get a tile from the cache, updating access order
    pub fn get(&mut self, tile_id: &TileId) -> Option<&TileData> {
        if self.data.contains_key(tile_id) {
            // Move to back of queue (most recently used)
            self.access_order.retain(|&id| id != *tile_id);
            self.access_order.push_back(*tile_id);
            self.data.get(tile_id)
        } else {
            None
        }
    }

    /// Check if a tile is in the cache without affecting access order
    pub fn contains(&self, tile_id: &TileId) -> bool {
        self.data.contains_key(tile_id)
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        CacheStats {
            capacity: self.capacity,
            current_size: self.data.len(),
            memory_usage_bytes: self.total_memory_usage,
        }
    }

    /// Evict the oldest (least recently used) tile
    fn evict_oldest(&mut self) -> Result<(), String> {
        if let Some(oldest_id) = self.access_order.pop_front() {
            if let Some(old_data) = self.data.remove(&oldest_id) {
                let tracker = global_tracker();
                tracker.free_buffer_allocation(old_data.host_memory_size, true);
                self.total_memory_usage -= old_data.host_memory_size;
            }
        }
        Ok(())
    }

    /// Clear all cached data
    pub fn clear(&mut self) {
        let tracker = global_tracker();
        for (_, tile_data) in self.data.drain() {
            tracker.free_buffer_allocation(tile_data.host_memory_size, true);
        }
        self.access_order.clear();
        self.total_memory_usage = 0;
    }
}

impl Drop for TileCache {
    fn drop(&mut self) {
        self.clear();
    }
}

/// Cache statistics for monitoring
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub capacity: usize,
    pub current_size: usize,
    pub memory_usage_bytes: u64,
}

/// Simple frustum for visibility culling
#[derive(Debug)]
pub struct Frustum {
    pub position: Vec3,
    pub direction: Vec3,
    pub fov_y: f32,
    pub aspect_ratio: f32,
    pub near: f32,
    pub far: f32,
}

impl Frustum {
    pub fn new(
        position: Vec3,
        direction: Vec3,
        fov_y: f32,
        aspect_ratio: f32,
        near: f32,
        far: f32,
    ) -> Self {
        Self {
            position,
            direction,
            fov_y,
            aspect_ratio,
            near,
            far,
        }
    }

    /// Simple visibility test - check if tile bounds intersect with frustum projection
    /// This is a simplified implementation for the quad-tree demo
    pub fn intersects_bounds(&self, bounds: &TileBounds) -> bool {
        // For simplicity, use a distance-based test
        let camera_pos_2d = Vec2::new(self.position.x, self.position.z);
        let bounds_center = bounds.center();
        let bounds_radius = bounds.size().length() * 0.5;
        let distance = camera_pos_2d.distance(bounds_center);

        // Visible if within far distance plus bounds radius
        distance <= self.far + bounds_radius
    }
}

/// Main tiling system that manages the quad-tree and cache
pub struct TilingSystem {
    pub root_bounds: TileBounds,
    tile_cache: TileCache,
    max_lod: u32,
    pub tile_size: Vec2,
}

impl TilingSystem {
    pub fn new(
        root_bounds: TileBounds,
        cache_capacity: usize,
        max_lod: u32,
        tile_size: Vec2,
    ) -> Self {
        Self {
            root_bounds,
            tile_cache: TileCache::new(cache_capacity),
            max_lod,
            tile_size,
        }
    }

    /// Public accessor for the configured maximum LOD level
    pub fn max_lod(&self) -> u32 {
        self.max_lod
    }

    /// Generate list of visible tiles for a given frustum
    pub fn get_visible_tiles(&mut self, frustum: &Frustum) -> Vec<TileId> {
        let mut visible_tiles = Vec::new();
        self.collect_visible_tiles_recursive(TileId::new(0, 0, 0), &frustum, &mut visible_tiles);
        visible_tiles
    }

    /// Generate visible tiles at a fixed LOD level
    pub fn get_visible_tiles_at_lod(&self, frustum: &Frustum, lod: u32) -> Vec<TileId> {
        let n = 1u32 << lod;
        let mut out = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let id = TileId::new(lod, x, y);
                let bounds = QuadTreeNode::calculate_bounds(&self.root_bounds, id, self.tile_size);
                if frustum.intersects_bounds(&bounds) {
                    out.push(id);
                }
            }
        }
        out
    }

    /// Recursive function to collect visible tiles
    fn collect_visible_tiles_recursive(
        &self,
        tile_id: TileId,
        frustum: &Frustum,
        visible_tiles: &mut Vec<TileId>,
    ) {
        let bounds = QuadTreeNode::calculate_bounds(&self.root_bounds, tile_id, self.tile_size);

        // Test visibility
        if !frustum.intersects_bounds(&bounds) {
            return;
        }

        // If at max LOD or should stop subdivision, add this tile
        if tile_id.lod >= self.max_lod || self.should_stop_subdivision(tile_id, frustum) {
            visible_tiles.push(tile_id);
            return;
        }

        // Otherwise, recurse to children
        for child_id in tile_id.children().iter() {
            self.collect_visible_tiles_recursive(*child_id, frustum, visible_tiles);
        }
    }

    /// Determine if we should stop subdividing and use this tile
    fn should_stop_subdivision(&self, tile_id: TileId, frustum: &Frustum) -> bool {
        // Simple distance-based LOD: stop subdividing if tile is far from camera
        let bounds = QuadTreeNode::calculate_bounds(&self.root_bounds, tile_id, self.tile_size);
        let camera_pos_2d = Vec2::new(frustum.position.x, frustum.position.z);
        let tile_center = bounds.center();
        let distance = camera_pos_2d.distance(tile_center);
        let tile_size = bounds.size().length();

        // Stop if tile appears smaller than some threshold at this distance
        let pixel_size = tile_size / distance;
        pixel_size < 0.1 // Arbitrary threshold for demo
    }

    /// Load tile data using synthetic heights until a backing dataset is wired.
    pub fn load_tile(&mut self, tile_id: TileId) -> Result<(), String> {
        if self.tile_cache.contains(&tile_id) {
            return Ok(()); // Already loaded
        }

        // Generate synthetic height data for the tile
        let tile_resolution = 64; // Fixed resolution per tile for demo
        let height_data =
            self.generate_synthetic_heights(tile_id, tile_resolution, tile_resolution);

        let tile_data = TileData::new(tile_id, height_data, tile_resolution, tile_resolution);
        self.tile_cache.insert(tile_data)?;

        Ok(())
    }

    /// Generate synthetic heights to keep the demo path deterministic without a data source.
    fn generate_synthetic_heights(&self, tile_id: TileId, width: u32, height: u32) -> Vec<f32> {
        let mut heights = Vec::with_capacity((width * height) as usize);
        let bounds = QuadTreeNode::calculate_bounds(&self.root_bounds, tile_id, self.tile_size);

        for y in 0..height {
            for x in 0..width {
                let u = x as f32 / (width - 1) as f32;
                let v = y as f32 / (height - 1) as f32;
                let world_x = bounds.min.x + u * bounds.size().x;
                let world_y = bounds.min.y + v * bounds.size().y;

                // Simple synthetic height function
                let h = (world_x * 0.1).sin() * 10.0 + (world_y * 0.1).cos() * 10.0;
                heights.push(h);
            }
        }

        heights
    }

    /// Get tile data from cache
    pub fn get_tile_data(&mut self, tile_id: &TileId) -> Option<&TileData> {
        self.tile_cache.get(tile_id)
    }

    /// Insert externally prepared tile data into the cache
    pub fn insert_tile_data(&mut self, tile_data: TileData) -> Result<(), String> {
        self.tile_cache.insert(tile_data)
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> CacheStats {
        self.tile_cache.get_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_id_hierarchy() {
        let parent = TileId::new(0, 0, 0);
        let children = parent.children();

        assert_eq!(children[0], TileId::new(1, 0, 0));
        assert_eq!(children[1], TileId::new(1, 1, 0));
        assert_eq!(children[2], TileId::new(1, 0, 1));
        assert_eq!(children[3], TileId::new(1, 1, 1));

        // Test parent relationship
        assert_eq!(children[0].parent().unwrap(), parent);
    }

    #[test]
    fn test_tile_bounds() {
        let bounds = TileBounds::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0));

        assert_eq!(bounds.center(), Vec2::new(5.0, 5.0));
        assert_eq!(bounds.size(), Vec2::new(10.0, 10.0));
        assert!(bounds.contains_point(Vec2::new(5.0, 5.0)));
        assert!(!bounds.contains_point(Vec2::new(15.0, 15.0)));
    }

    #[test]
    fn test_tile_cache() {
        let mut cache = TileCache::new(2);

        let tile1 = TileData::new(TileId::new(0, 0, 0), vec![1.0; 64 * 64], 64, 64);
        let tile2 = TileData::new(TileId::new(0, 1, 0), vec![2.0; 64 * 64], 64, 64);
        let tile3 = TileData::new(TileId::new(0, 0, 1), vec![3.0; 64 * 64], 64, 64);

        // Insert first two tiles
        cache.insert(tile1).expect("Should insert successfully");
        cache.insert(tile2).expect("Should insert successfully");

        assert_eq!(cache.get_stats().current_size, 2);

        // Insert third tile should evict oldest
        cache.insert(tile3).expect("Should insert successfully");

        // Should still have 2 tiles but the first one should be evicted
        assert_eq!(cache.get_stats().current_size, 2);
        assert!(!cache.contains(&TileId::new(0, 0, 0)));
        assert!(cache.contains(&TileId::new(0, 1, 0)));
        assert!(cache.contains(&TileId::new(0, 0, 1)));
    }
}
