//! P2.1/M5: Complete clipmap level with center block and nested rings.

use super::ring::{make_center_block, make_ring, make_ring_skirts};
use super::vertex::ClipmapVertex;
use super::ClipmapConfig;
use crate::terrain::tiling::TileId;
use glam::Vec2;

fn snap_center_to_finest_grid(center: Vec2, base_cell_size: f32) -> Vec2 {
    if !base_cell_size.is_finite() || base_cell_size <= 0.0 {
        return center;
    }
    (center / base_cell_size).round() * base_cell_size
}

/// Bounds for a mesh region (start index, index count).
#[derive(Debug, Clone, Copy)]
pub struct MeshBounds {
    pub vertex_start: u32,
    pub vertex_count: u32,
    pub index_start: u32,
    pub index_count: u32,
}

/// Complete clipmap mesh data ready for GPU upload.
#[derive(Debug)]
pub struct ClipmapMesh {
    pub vertices: Vec<ClipmapVertex>,
    pub indices: Vec<u32>,
    pub center_bounds: MeshBounds,
    pub ring_bounds: Vec<MeshBounds>,
    pub triangle_count: u32,
}

impl ClipmapMesh {
    /// Get total vertex count.
    pub fn vertex_count(&self) -> u32 {
        self.vertices.len() as u32
    }

    /// Get total index count.
    pub fn index_count(&self) -> u32 {
        self.indices.len() as u32
    }

    /// Calculate triangle reduction percentage vs a full-resolution grid.
    pub fn triangle_reduction_percent(&self, full_res_triangles: u32) -> f32 {
        if full_res_triangles == 0 {
            return 0.0;
        }
        let reduction =
            (full_res_triangles as f32 - self.triangle_count as f32) / full_res_triangles as f32;
        (reduction * 100.0).max(0.0)
    }
}

/// Complete clipmap level managing center block and all LOD rings.
#[derive(Debug)]
pub struct ClipmapLevel {
    pub config: ClipmapConfig,
    pub center: Vec2,
    pub terrain_extent: f32,
    pub base_cell_size: f32,
    mesh: Option<ClipmapMesh>,
}

impl ClipmapLevel {
    /// Create a new clipmap level centered at the given world position.
    pub fn new(config: ClipmapConfig, center: Vec2, terrain_extent: f32) -> Self {
        let base_cell_size = terrain_extent / (config.center_resolution as f32 * 8.0);
        Self {
            config,
            center,
            terrain_extent,
            base_cell_size,
            mesh: None,
        }
    }

    /// Generate or regenerate the clipmap mesh.
    pub fn generate(&mut self) -> &ClipmapMesh {
        let mut all_vertices = Vec::new();
        let mut all_indices = Vec::new();
        let mut ring_bounds = Vec::new();

        // Generate center block
        let center_half = self.base_cell_size * self.config.center_resolution as f32 * 0.5;
        let (center_verts, center_indices) = make_center_block(
            self.config.center_resolution,
            self.center,
            center_half,
            self.terrain_extent,
        );

        let center_bounds = MeshBounds {
            vertex_start: 0,
            vertex_count: center_verts.len() as u32,
            index_start: 0,
            index_count: center_indices.len() as u32,
        };

        all_vertices.extend(center_verts);
        all_indices.extend(center_indices);

        // Generate rings from innermost (finest LOD) to outermost (coarsest)
        let mut current_inner = center_half;
        for ring_idx in 0..self.config.ring_count {
            let ring_extent = self.config.ring_extent(ring_idx, self.base_cell_size);
            let current_outer = current_inner + ring_extent;

            let vertex_start = all_vertices.len() as u32;
            let index_start = all_indices.len() as u32;

            let (mut ring_verts, mut ring_indices) = make_ring(
                ring_idx,
                current_inner,
                current_outer,
                self.config.ring_resolution,
                self.center,
                self.terrain_extent,
                self.config.morph_range,
            );
            let (skirt_verts, skirt_indices) = make_ring_skirts(
                &ring_verts,
                &ring_indices,
                self.config.skirt_depth,
                ring_idx,
                0,
            );
            ring_verts.extend(skirt_verts);
            ring_indices.extend(skirt_indices);

            // Offset indices by current vertex count
            let offset_indices: Vec<u32> = ring_indices.iter().map(|&i| i + vertex_start).collect();

            ring_bounds.push(MeshBounds {
                vertex_start,
                vertex_count: ring_verts.len() as u32,
                index_start,
                index_count: ring_indices.len() as u32,
            });

            all_vertices.extend(ring_verts);
            all_indices.extend(offset_indices);

            current_inner = current_outer;
        }

        let triangle_count = all_indices.len() as u32 / 3;

        self.mesh = Some(ClipmapMesh {
            vertices: all_vertices,
            indices: all_indices,
            center_bounds,
            ring_bounds,
            triangle_count,
        });

        self.mesh.as_ref().unwrap()
    }

    /// Get the generated mesh, generating if needed.
    pub fn mesh(&mut self) -> &ClipmapMesh {
        if self.mesh.is_none() {
            self.generate();
        }
        self.mesh.as_ref().unwrap()
    }

    /// Update the clipmap center position.
    /// Returns list of TileIds that should be requested for streaming.
    pub fn update_center(&mut self, new_center: Vec2) -> Vec<TileId> {
        let new_center = snap_center_to_finest_grid(new_center, self.base_cell_size);
        let delta = new_center - self.center;

        // Only regenerate if moved significantly (half a cell)
        if delta.length() < self.base_cell_size * 0.5 {
            return Vec::new();
        }

        self.center = new_center;
        self.mesh = None; // Force regeneration

        // Calculate which tiles are needed for each LOD level
        self.calculate_required_tiles()
    }

    /// Calculate tiles required for current clipmap position.
    pub fn calculate_required_tiles(&self) -> Vec<TileId> {
        let mut tiles = Vec::new();

        // Center block tiles (LOD 0)
        let center_tile = self.world_to_tile(self.center, 0);
        tiles.push(center_tile);

        // Ring tiles
        let mut current_inner = self.base_cell_size * self.config.center_resolution as f32 * 0.5;
        for ring_idx in 0..self.config.ring_count {
            let lod = self.config.ring_lod(ring_idx);
            let ring_extent = self.config.ring_extent(ring_idx, self.base_cell_size);
            let current_outer = current_inner + ring_extent;

            // Sample tiles at ring corners and edges
            let corners = [
                self.center + Vec2::new(-current_outer, -current_outer),
                self.center + Vec2::new(current_outer, -current_outer),
                self.center + Vec2::new(-current_outer, current_outer),
                self.center + Vec2::new(current_outer, current_outer),
            ];

            for corner in &corners {
                let tile = self.world_to_tile(*corner, lod);
                if !tiles.contains(&tile) {
                    tiles.push(tile);
                }
            }

            current_inner = current_outer;
        }

        tiles
    }

    /// Convert world position to tile ID at given LOD level.
    fn world_to_tile(&self, pos: Vec2, lod: u32) -> TileId {
        let tile_size = self.terrain_extent / (1 << lod) as f32;
        let normalized = (pos + Vec2::splat(self.terrain_extent * 0.5)) / tile_size;
        TileId::new(
            lod,
            normalized.x.floor().max(0.0) as u32,
            normalized.y.floor().max(0.0) as u32,
        )
    }

    /// Get LOD level for a given ring index.
    pub fn ring_lod(&self, ring_index: u32) -> u32 {
        self.config.ring_lod(ring_index)
    }

    /// Calculate triangle count for a full-resolution grid (for reduction comparison).
    pub fn full_resolution_triangle_count(&self) -> u32 {
        full_resolution_triangle_count(&self.config)
    }
}

/// Triangles required to cover the complete clipmap footprint at the finest
/// lattice. This is the meaningful comparator for a nested clipmap; the old
/// `center_resolution * 4` approximation covered only a small central square.
pub fn full_resolution_triangle_count(config: &ClipmapConfig) -> u32 {
    let ring_cells_per_side = (0..config.ring_count)
        .map(|ring| config.ring_resolution.checked_shl(ring).unwrap_or(u32::MAX))
        .fold(0u32, u32::saturating_add);
    let cells_per_side = config
        .center_resolution
        .saturating_add(ring_cells_per_side.saturating_mul(2));
    cells_per_side
        .saturating_mul(cells_per_side)
        .saturating_mul(2)
}

/// Generate a complete clipmap mesh from configuration.
pub fn clipmap_generate(config: &ClipmapConfig, center: Vec2, terrain_extent: f32) -> ClipmapMesh {
    let mut level = ClipmapLevel::new(config.clone(), center, terrain_extent);
    level.generate();
    level.mesh.take().unwrap()
}

/// Calculate triangle reduction percentage.
pub fn calculate_triangle_reduction(full_res_triangles: u32, clipmap_triangles: u32) -> f32 {
    if full_res_triangles == 0 {
        return 0.0;
    }
    ((full_res_triangles as f32 - clipmap_triangles as f32) / full_res_triangles as f32).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clipmap_level_creation() {
        let config = ClipmapConfig::new(4, 64);
        let level = ClipmapLevel::new(config, Vec2::ZERO, 1000.0);
        assert_eq!(level.config.ring_count, 4);
        assert_eq!(level.center, Vec2::ZERO);
    }

    #[test]
    fn test_clipmap_mesh_generation() {
        let config = ClipmapConfig::new(4, 32);
        let mut level = ClipmapLevel::new(config, Vec2::ZERO, 1000.0);
        let mesh = level.generate();

        assert!(mesh.vertex_count() > 0);
        assert!(mesh.index_count() > 0);
        assert_eq!(mesh.index_count() % 3, 0);
        assert_eq!(mesh.ring_bounds.len(), 4);
    }

    #[test]
    fn test_triangle_reduction_meets_40_percent() {
        let config = ClipmapConfig::new(4, 64);
        let mut level = ClipmapLevel::new(config, Vec2::ZERO, 1000.0);
        let full_res = level.full_resolution_triangle_count();
        let mesh = level.generate();

        // Compare against full-res grid
        let reduction = mesh.triangle_reduction_percent(full_res);

        // P2.1 exit criteria: ≥40% reduction
        assert!(
            reduction >= 40.0,
            "Triangle reduction {:.1}% should be >= 40%",
            reduction
        );
    }

    #[test]
    fn test_full_resolution_comparator_covers_outermost_ring() {
        let config = ClipmapConfig::new(4, 32);
        // 32 center cells plus two sides of 32*(1+2+4+8) ring cells.
        assert_eq!(full_resolution_triangle_count(&config), 1_968_128);
    }

    #[test]
    fn test_center_update_triggers_tile_requests() {
        let config = ClipmapConfig::new(4, 64);
        let mut level = ClipmapLevel::new(config, Vec2::ZERO, 1000.0);

        // Large movement should trigger tile requests
        let tiles = level.update_center(Vec2::new(100.0, 100.0));
        assert!(!tiles.is_empty());
    }

    #[test]
    fn test_small_center_update_no_regeneration() {
        let config = ClipmapConfig::new(4, 64);
        let mut level = ClipmapLevel::new(config, Vec2::ZERO, 1000.0);
        level.generate();

        // Small movement should not regenerate
        let tiles = level.update_center(Vec2::new(0.1, 0.1));
        assert!(tiles.is_empty());
    }

    #[test]
    fn test_center_updates_snap_to_the_finest_grid() {
        let config = ClipmapConfig::new(4, 64);
        let mut level = ClipmapLevel::new(config, Vec2::ZERO, 1000.0);
        let base_cell = level.base_cell_size;

        let requested = Vec2::new(base_cell * 0.6, -base_cell * 1.6);
        assert!(!level.update_center(requested).is_empty());
        assert_eq!(level.center, Vec2::new(base_cell, -base_cell * 2.0));

        // Another raw camera position inside the same snapped cell neither
        // changes the mesh lattice nor spuriously requests tiles.
        let same_cell = Vec2::new(base_cell * 0.7, -base_cell * 1.7);
        assert!(level.update_center(same_cell).is_empty());
        assert_eq!(level.center, Vec2::new(base_cell, -base_cell * 2.0));
    }

    #[test]
    fn test_initial_center_preserves_static_clipmap_semantics() {
        let config = ClipmapConfig::new(4, 64);
        let center = Vec2::new(12.25, -31.75);
        let level = ClipmapLevel::new(config, center, 1000.0);
        assert_eq!(level.center, center);
    }

    #[test]
    fn test_clipmap_generate_function() {
        let config = ClipmapConfig::new(4, 32);
        let mesh = clipmap_generate(&config, Vec2::ZERO, 1000.0);

        assert!(mesh.vertex_count() > 0);
        assert!(mesh.triangle_count > 0);
    }
}
