// src/picking/unified.rs
// Unified picking system with BVH-accelerated ray intersection
// Part of Plan 3: Premium - Unified Picking with BVH + Python Callbacks

use super::ray::Ray;
use super::selection::{SelectionManager, SelectionStyle};
use crate::accel::types::{Aabb as BvhAabb, BvhNode, Triangle};
use crate::core::resource_tracker::tracked_create_buffer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{Buffer, Device, Queue};

pub(crate) fn unpack_visibility_id(packed: u32) -> Option<(u32, u32)> {
    let value = packed.checked_sub(1)?;
    Some((value >> 16, value & 0xffff))
}

/// Rich pick result with full feature attributes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RichPickResult {
    /// Feature ID (0 = no feature/background)
    pub feature_id: u32,
    /// Layer name
    pub layer_name: String,
    /// World position of the hit
    pub world_pos: [f64; 3],
    /// Feature attributes as key-value pairs
    pub attributes: HashMap<String, String>,
    /// Terrain info if terrain was hit
    pub terrain_info: Option<TerrainHitInfo>,
    /// Hit distance along the ray
    pub hit_distance: f32,
}

/// Terrain hit information
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TerrainHitInfo {
    /// Elevation at the hit point
    pub elevation: f32,
    /// Slope angle in degrees
    pub slope: f32,
    /// Aspect angle in degrees
    pub aspect: f32,
    /// Surface normal at hit point
    pub normal: [f32; 3],
}

/// Pick event type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PickEventType {
    Click,
    DoubleClick,
    Hover,
    LassoComplete,
}

/// Pick event with context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PickEvent {
    /// Event type
    pub event_type: PickEventType,
    /// Screen position
    pub screen_pos: (u32, u32),
    /// Whether shift key was held
    pub shift_held: bool,
    /// Whether ctrl key was held
    pub ctrl_held: bool,
    /// Results associated with this event
    pub results: Vec<RichPickResult>,
}

/// Configuration for the unified picking system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedPickingConfig {
    /// Whether BVH picking is enabled
    pub bvh_enabled: bool,
    /// Whether terrain picking is enabled
    pub terrain_enabled: bool,
    /// Whether lasso selection is enabled
    pub lasso_enabled: bool,
    /// Maximum ray distance
    pub max_ray_distance: f32,
    /// BVH stack depth limit
    pub bvh_stack_depth: u32,
}

impl Default for UnifiedPickingConfig {
    fn default() -> Self {
        Self {
            bvh_enabled: false,
            terrain_enabled: true,
            lasso_enabled: false,
            max_ray_distance: 10000.0,
            bvh_stack_depth: 32,
        }
    }
}

/// BVH data for a layer
#[derive(Debug)]
pub struct LayerBvhData {
    /// Layer identifier
    pub layer_id: u32,
    /// Layer name
    pub name: String,
    /// BVH nodes buffer (GPU)
    pub nodes_buffer: Option<Buffer>,
    /// Triangles buffer (GPU)
    pub triangles_buffer: Option<Buffer>,
    /// Feature IDs buffer (GPU) - maps triangle to feature
    pub feature_ids_buffer: Option<Buffer>,
    /// CPU fallback data
    pub cpu_nodes: Vec<BvhNode>,
    pub cpu_triangles: Vec<Triangle>,
    pub cpu_feature_ids: Vec<u32>,
    /// World bounds
    pub world_aabb: BvhAabb,
    /// Whether data is on GPU
    pub is_gpu: bool,
}

impl LayerBvhData {
    /// Create empty layer BVH data
    pub fn new(layer_id: u32, name: String) -> Self {
        Self {
            layer_id,
            name,
            nodes_buffer: None,
            triangles_buffer: None,
            feature_ids_buffer: None,
            cpu_nodes: Vec::new(),
            cpu_triangles: Vec::new(),
            cpu_feature_ids: Vec::new(),
            world_aabb: BvhAabb::empty(),
            is_gpu: false,
        }
    }
}

/// Unified picking system with BVH acceleration
pub struct UnifiedPickingSystem {
    device: Arc<Device>,
    queue: Arc<Queue>,
    config: UnifiedPickingConfig,
    /// BVH data per layer
    layer_bvhs: HashMap<u32, LayerBvhData>,
    /// Selection manager
    selection_manager: SelectionManager,
    /// Attribute storage per feature
    feature_attributes: HashMap<(u32, u32), HashMap<String, String>>, // (layer_id, feature_id) -> attrs
}

impl UnifiedPickingSystem {
    /// Create a new unified picking system
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        Self {
            device,
            queue,
            config: UnifiedPickingConfig::default(),
            layer_bvhs: HashMap::new(),
            selection_manager: SelectionManager::new(),
            feature_attributes: HashMap::new(),
        }
    }

    /// Read terrain tile/triangle identities directly from TESSELLA's
    /// R32Uint visibility buffer. One aligned row is reserved per requested
    /// pixel so arbitrary batches, including the 10,000-pixel differential
    /// gate, use one submission and one map rather than a CPU ray loop.
    pub fn pick_visibility_pixels(
        &self,
        visibility: &wgpu::Texture,
        width: u32,
        height: u32,
        pixels: &[(u32, u32)],
    ) -> Result<Vec<Option<(u32, u32)>>, String> {
        if pixels.is_empty() {
            return Ok(Vec::new());
        }
        if pixels.iter().any(|&(x, y)| x >= width || y >= height) {
            return Err("visibility pick coordinate is outside the texture".to_string());
        }
        const ROW_BYTES: u64 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64;
        let readback = tracked_create_buffer(
            self.device.as_ref(),
            &wgpu::BufferDescriptor {
                label: Some("picking.visibility.readback"),
                size: ROW_BYTES * pixels.len() as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )
        .map_err(|error| error.to_string())?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("picking.visibility.encoder"),
            });
        for (index, &(x, y)) in pixels.iter().enumerate() {
            encoder.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    texture: visibility,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyBuffer {
                    buffer: &readback,
                    layout: wgpu::ImageDataLayout {
                        offset: index as u64 * ROW_BYTES,
                        bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }
        self.queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver.receive())
            .ok_or_else(|| "visibility picking callback dropped".to_string())?
            .map_err(|error| format!("visibility picking map failed: {error}"))?;
        let mapped = slice.get_mapped_range();
        let results = pixels
            .iter()
            .enumerate()
            .map(|(index, _)| {
                let offset = index * ROW_BYTES as usize;
                let packed =
                    u32::from_le_bytes(mapped[offset..offset + 4].try_into().expect("4 bytes"));
                unpack_visibility_id(packed)
            })
            .collect();
        drop(mapped);
        readback.unmap();
        Ok(results)
    }

    /// Get configuration
    pub fn config(&self) -> &UnifiedPickingConfig {
        &self.config
    }

    /// Get mutable configuration
    pub fn config_mut(&mut self) -> &mut UnifiedPickingConfig {
        &mut self.config
    }

    /// Enable or disable BVH picking
    pub fn set_bvh_enabled(&mut self, enabled: bool) {
        self.config.bvh_enabled = enabled;
    }

    /// Enable or disable terrain picking
    pub fn set_terrain_enabled(&mut self, enabled: bool) {
        self.config.terrain_enabled = enabled;
    }

    /// Enable or disable lasso selection
    pub fn set_lasso_enabled(&mut self, enabled: bool) {
        self.config.lasso_enabled = enabled;
    }

    /// Register a layer's BVH data
    pub fn register_layer_bvh(&mut self, data: LayerBvhData) {
        self.layer_bvhs.insert(data.layer_id, data);
    }

    /// Remove a layer's BVH data
    pub fn remove_layer_bvh(&mut self, layer_id: u32) {
        self.layer_bvhs.remove(&layer_id);
    }

    /// Get layer BVH data
    pub fn get_layer_bvh(&self, layer_id: u32) -> Option<&LayerBvhData> {
        self.layer_bvhs.get(&layer_id)
    }

    pub fn cpu_bvh_bytes(&self) -> u64 {
        self.layer_bvhs
            .values()
            .map(|layer| {
                layer.cpu_nodes.capacity() * std::mem::size_of::<BvhNode>()
                    + layer.cpu_triangles.capacity() * std::mem::size_of::<Triangle>()
                    + layer.cpu_feature_ids.capacity() * std::mem::size_of::<u32>()
            })
            .sum::<usize>() as u64
    }

    /// Set feature attributes
    pub fn set_feature_attributes(
        &mut self,
        layer_id: u32,
        feature_id: u32,
        attributes: HashMap<String, String>,
    ) {
        self.feature_attributes
            .insert((layer_id, feature_id), attributes);
    }

    /// Get feature attributes
    pub fn get_feature_attributes(
        &self,
        layer_id: u32,
        feature_id: u32,
    ) -> Option<&HashMap<String, String>> {
        self.feature_attributes.get(&(layer_id, feature_id))
    }

    /// Get selection manager
    pub fn selection_manager(&self) -> &SelectionManager {
        &self.selection_manager
    }

    /// Get mutable selection manager
    pub fn selection_manager_mut(&mut self) -> &mut SelectionManager {
        &mut self.selection_manager
    }

    /// Create a new selection set with style
    pub fn create_selection_set(&mut self, name: &str, style: SelectionStyle) {
        self.selection_manager.create_set_with_style(name, style);
    }

    /// Add features to a selection set
    pub fn add_to_selection(&mut self, set_name: &str, feature_ids: &[u32]) {
        self.selection_manager
            .add_many_to_set(set_name, feature_ids.iter().copied());
    }

    /// Remove features from a selection set
    pub fn remove_from_selection(&mut self, set_name: &str, feature_ids: &[u32]) {
        for &id in feature_ids {
            self.selection_manager.remove_from_set(set_name, id);
        }
    }

    /// Clear a selection set
    pub fn clear_selection(&mut self, set_name: &str) {
        self.selection_manager.clear_set(set_name);
    }

    /// CPU ray-BVH intersection for a layer
    pub fn ray_bvh_intersect_cpu(&self, ray: &Ray, layer_id: u32) -> Option<(u32, f32, [f32; 3])> {
        let layer = self.layer_bvhs.get(&layer_id)?;

        if layer.cpu_nodes.is_empty() || layer.cpu_triangles.is_empty() {
            return None;
        }

        let mut closest_t = self.config.max_ray_distance;
        let mut closest_feature_id = 0u32;
        let mut closest_pos = [0.0f32; 3];

        // Stack-based BVH traversal
        let mut stack = Vec::with_capacity(self.config.bvh_stack_depth as usize);
        stack.push(0u32); // Start at root

        while let Some(node_idx) = stack.pop() {
            if node_idx as usize >= layer.cpu_nodes.len() {
                continue;
            }

            let node = &layer.cpu_nodes[node_idx as usize];

            // Test ray-AABB intersection
            if !ray_aabb_intersect(ray, &node.aabb, closest_t) {
                continue;
            }

            if node.is_leaf() {
                // Test triangles in leaf
                let (first_prim, prim_count) = node.primitives().unwrap_or((0, 0));
                for i in 0..prim_count {
                    let tri_idx = (first_prim + i) as usize;
                    if tri_idx >= layer.cpu_triangles.len() {
                        continue;
                    }

                    let triangle = &layer.cpu_triangles[tri_idx];
                    if let Some(t) = ray_triangle_intersect(ray, triangle) {
                        if t > 0.0 && t < closest_t {
                            closest_t = t;
                            closest_pos = ray.point_at(t);
                            if tri_idx < layer.cpu_feature_ids.len() {
                                closest_feature_id = layer.cpu_feature_ids[tri_idx];
                            }
                        }
                    }
                }
            } else {
                // Push children onto stack
                let (left, right) = node.children().unwrap_or((0, 0));
                stack.push(right);
                stack.push(left);
            }
        }

        if closest_feature_id > 0 {
            Some((closest_feature_id, closest_t, closest_pos))
        } else {
            None
        }
    }

    /// Cast ray and find all hits (sorted by distance)
    pub fn ray_cast(&self, ray: &Ray) -> Vec<RichPickResult> {
        let mut results = Vec::new();

        if self.config.bvh_enabled {
            // Test against all layers
            for (layer_id, layer) in &self.layer_bvhs {
                if let Some((feature_id, distance, world_pos)) =
                    self.ray_bvh_intersect_cpu(ray, *layer_id)
                {
                    let attributes = self
                        .get_feature_attributes(*layer_id, feature_id)
                        .cloned()
                        .unwrap_or_default();

                    results.push(RichPickResult {
                        feature_id,
                        layer_name: layer.name.clone(),
                        world_pos: world_pos.map(f64::from),
                        attributes,
                        terrain_info: None,
                        hit_distance: distance,
                    });
                }
            }
        }

        // Sort by distance
        results.sort_by(|a, b| {
            a.hit_distance
                .partial_cmp(&b.hit_distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// Handle a pick event
    pub fn handle_pick_event(&mut self, ray: &Ray, event: &PickEvent) -> Vec<RichPickResult> {
        let results = self.ray_cast(ray);

        // Update selection based on event
        if !results.is_empty() && event.event_type == PickEventType::Click {
            let feature_id = results[0].feature_id;
            self.selection_manager
                .handle_pick(feature_id, event.shift_held);
        }

        results
    }
}

/// Ray-AABB intersection test
fn ray_aabb_intersect(ray: &Ray, aabb: &BvhAabb, max_t: f32) -> bool {
    let mut t_min = 0.0f32;
    let mut t_max = max_t;

    for i in 0..3 {
        if ray.direction[i].abs() < 1e-10 {
            // Ray parallel to slab
            if ray.origin[i] < aabb.min[i] || ray.origin[i] > aabb.max[i] {
                return false;
            }
        } else {
            let inv_d = 1.0 / ray.direction[i];
            let mut t1 = (aabb.min[i] - ray.origin[i]) * inv_d;
            let mut t2 = (aabb.max[i] - ray.origin[i]) * inv_d;

            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }

            t_min = t_min.max(t1);
            t_max = t_max.min(t2);

            if t_min > t_max {
                return false;
            }
        }
    }

    true
}

/// Moller-Trumbore ray-triangle intersection
fn ray_triangle_intersect(ray: &Ray, triangle: &Triangle) -> Option<f32> {
    const EPSILON: f32 = 1e-7;

    let edge1 = [
        triangle.v1[0] - triangle.v0[0],
        triangle.v1[1] - triangle.v0[1],
        triangle.v1[2] - triangle.v0[2],
    ];
    let edge2 = [
        triangle.v2[0] - triangle.v0[0],
        triangle.v2[1] - triangle.v0[1],
        triangle.v2[2] - triangle.v0[2],
    ];

    let h = cross(ray.direction, edge2);
    let a = dot(edge1, h);

    if a.abs() < EPSILON {
        return None; // Ray parallel to triangle
    }

    let f = 1.0 / a;
    let s = [
        ray.origin[0] - triangle.v0[0],
        ray.origin[1] - triangle.v0[1],
        ray.origin[2] - triangle.v0[2],
    ];
    let u = f * dot(s, h);

    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let q = cross(s, edge1);
    let v = f * dot(ray.direction, q);

    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = f * dot(edge2, q);

    if t > EPSILON {
        Some(t)
    } else {
        None
    }
}

/// Cross product
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ray_triangle_intersect() {
        let triangle = Triangle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);

        // Ray hitting triangle
        let ray = Ray::new([0.25, 0.25, 1.0], [0.0, 0.0, -1.0]);
        let t = ray_triangle_intersect(&ray, &triangle);
        assert!(t.is_some());
        assert!((t.unwrap() - 1.0).abs() < 1e-5);

        // Ray missing triangle
        let ray = Ray::new([2.0, 2.0, 1.0], [0.0, 0.0, -1.0]);
        let t = ray_triangle_intersect(&ray, &triangle);
        assert!(t.is_none());
    }

    #[test]
    fn visibility_id_reserves_zero_for_background() {
        assert_eq!(unpack_visibility_id(0), None);
        let packed = ((0x1234 << 16) | 0x56) + 1;
        assert_eq!(unpack_visibility_id(packed), Some((0x1234, 0x56)));
    }
}
