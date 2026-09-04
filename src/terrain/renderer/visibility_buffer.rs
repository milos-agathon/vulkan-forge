//! TESSELLA terrain visibility buffer.
//!
//! Zero is reserved for background. Visible primitives are encoded as
//! `1 + ((tile_lod_id & 0xffff) << 16) | (triangle_id & 0xffff)`. The extra one
//! keeps tile zero / triangle zero distinct from background. That packing is
//! defined by `src/shaders/terrain_visbuffer_write.wgsl` (pass 1) and decoded
//! by `src/shaders/terrain_visibility_fullscreen.wgsl` (pass 2); the CPU oracle
//! below is the third mirror of it. The material resolve is a full-screen
//! fragment pass. It reads primitive identity and depth, reconstructs the
//! visible surface, and invokes POM/material/feedback exactly once for every
//! non-background visibility pixel. The runtime resolve replays the original
//! clipmap geometry against the pass-1 depth/identity buffer; the fullscreen
//! helper remains in the module for shader-contract coverage. `terrain_visbuffer_resolve.wgsl`
//! is the compute pass that reads those counters back, not the resolve itself.

use crate::core::error::{RenderError, RenderResult};
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_texture, TrackedBuffer, TrackedTexture,
};
use bytemuck::{Pod, Zeroable};
use std::sync::{Mutex, OnceLock};

pub(in crate::terrain::renderer) struct CpuVisibilityOracle {
    mesh: crate::accel::cpu_bvh::MeshCPU,
    bvh: crate::accel::cpu_bvh::BvhCPU,
    identities: Vec<(u32, u32)>,
}

impl CpuVisibilityOracle {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::terrain::renderer) fn build(
        params: &crate::terrain::render_params::TerrainRenderParams,
        heightmap: &[f32],
        height_dims: (u32, u32),
        clipmap: &crate::terrain::clipmap::ClipmapMesh,
        tiles: &[crate::terrain::clipmap::gpu_lod::TileInfo],
        templates: &[crate::terrain::clipmap::gpu_lod::IndirectDrawTemplate],
        variant_count: u32,
        fallback_index_count: u32,
        lod_config: &crate::terrain::clipmap::gpu_lod::GpuLodConfig,
        submitted_tiles: Option<&[crate::terrain::clipmap::gpu_lod::TileInfo]>,
    ) -> anyhow::Result<Self> {
        use crate::terrain::clipmap::gpu_lod::cpu_lod_select;

        let decoded = params.decoded();
        let (h_min, h_max) = decoded.clamp.height_range;
        let h_center = (h_min + h_max) * 0.5;
        let skirt = super::core::clipmap_camera_config(&params.camera_mode)
            .map(|config| config.ring_resolution as f32 * 0.001)
            .unwrap_or(0.0);
        let vertices = clipmap
            .vertices
            .iter()
            .map(|vertex| {
                // Mirror `vs_clipmap_main`: the rendered surface blends the
                // fine sample toward a ring-dependent coarse reconstruction
                // before applying the configured height curve. Building the
                // BVH from fine samples alone moves triangle edges enough to
                // change primitive identity at grazing clipmap cameras.
                let height = sample_clipmap_height(heightmap, height_dims, vertex);
                let height = apply_height_curve(height, (h_min, h_max), params);
                let skirt_offset = if vertex.is_skirt() { skirt } else { 0.0 };
                [
                    vertex.position[0],
                    vertex.position[1],
                    (height - h_center - skirt_offset) * params.z_scale,
                ]
            })
            .collect::<Vec<_>>();
        let (eye, view, proj) = super::TerrainScene::build_camera_matrices(params);
        let tile_indices = tiles
            .iter()
            .enumerate()
            .map(|(index, tile)| (tile.tile_id, index))
            .collect::<std::collections::HashMap<_, _>>();
        let mut indices = Vec::new();
        let mut identities = Vec::new();
        if params.culling == "none" {
            // The renderer's pixel-correctness path submits the complete
            // clipmap once with the fallback instance (tile=0, lod=0).
            // Mirror that draw literally; applying CPU LOD selection here
            // would compare against geometry the GPU did not submit.
            let fallback_indices = clipmap
                .indices
                .get(..fallback_index_count as usize)
                .ok_or_else(|| anyhow::anyhow!("clipmap fallback draw exceeds its index buffer"))?;
            for (primitive, triangle) in fallback_indices.chunks_exact(3).enumerate() {
                indices.push([triangle[0], triangle[1], triangle[2]]);
                identities.push((0, primitive as u32 & 0xffff));
            }
        } else {
            let selected_tiles =
                crate::terrain::clipmap::gpu_lod::prefer_submitted_tiles(submitted_tiles, || {
                    cpu_lod_select(
                        tiles,
                        proj * view,
                        eye,
                        lod_config,
                        (
                            (h_min - h_center - skirt) * params.z_scale,
                            (h_max - h_center) * params.z_scale,
                        ),
                    )
                    .visible_tiles
                });
            for visible in selected_tiles {
                let Some(&tile_index) = tile_indices.get(&visible.tile_id) else {
                    continue;
                };
                let template_index =
                    tile_index * variant_count as usize + visible.selected_lod as usize;
                let Some(template) = templates.get(template_index) else {
                    continue;
                };
                let source = &clipmap.indices[template.first_index as usize
                    ..(template.first_index + template.index_count) as usize];
                for (primitive, triangle) in source.chunks_exact(3).enumerate() {
                    indices.push([triangle[0], triangle[1], triangle[2]]);
                    identities.push((
                        ((visible.selected_lod & 0xf) << 12) | (template.tile_id & 0xfff),
                        primitive as u32 & 0xffff,
                    ));
                }
            }
        }
        // GPU rasterisation clips in homogeneous coordinates before dividing
        // by w, then snaps the surviving vertices to a fixed-point subpixel
        // grid. Build the independent CPU BVH in that same screen/depth space:
        // projecting an unclipped eye/near-plane crossing triangle creates a
        // bogus screen-spanning primitive and a world-space ray is ambiguous
        // on shared edges.
        let (mesh, identities) = build_clipped_raster_mesh(
            &vertices,
            &indices,
            &identities,
            view,
            proj,
            params.size_px,
        )?;
        let bvh = crate::accel::cpu_bvh::build_bvh_cpu(
            &mesh,
            &crate::accel::cpu_bvh::BuildOptions::default(),
        )?;
        Ok(Self {
            mesh,
            bvh,
            identities,
        })
    }

    fn pick(&self, pixels: &[(u32, u32)]) -> Vec<Option<(u32, u32)>> {
        pixels
            .iter()
            .map(|&(x, y)| {
                let ray = crate::picking::Ray::new(
                    [x as f32 + 0.5, y as f32 + 0.5, -1.0],
                    [0.0, 0.0, 1.0],
                );
                self.intersect_raster(&ray, (x, y))
            })
            .collect()
    }

    fn intersect_raster(&self, ray: &crate::picking::Ray, pixel: (u32, u32)) -> Option<(u32, u32)> {
        let mut closest = f32::INFINITY;
        let mut identity = None;
        let root = self.bvh.nodes.len().checked_sub(1)? as u32;
        let mut stack = vec![root];
        while let Some(node_index) = stack.pop() {
            let node = self.bvh.nodes.get(node_index as usize)?;
            if !ray_aabb(ray, node.aabb_min, node.aabb_max, closest.min(2.0)) {
                continue;
            }
            if node.is_leaf() {
                for offset in 0..node.right {
                    let reordered = *self.bvh.tri_indices.get((node.left + offset) as usize)?;
                    let (v0, v1, v2) = self.mesh.get_triangle(reordered as usize)?;
                    // The visibility attachment is owned by fixed-point
                    // rasterisation, not by the floating-point ray/triangle
                    // predicate used to traverse the BVH.  At a shared edge
                    // the latter can reject a point that WebGPU's top-left
                    // rule accepted, producing a false CPU background pick.
                    // Derive depth from the same snapped screen triangle that
                    // decides coverage.
                    let Some(distance) = raster_distance_at_pixel(v0, v1, v2, pixel) else {
                        continue;
                    };
                    if distance < closest {
                        closest = distance;
                        identity = self.identities.get(reordered as usize).copied();
                    }
                }
            } else {
                stack.push(node.right);
                stack.push(node.left);
            }
        }
        identity
    }

    #[cfg(test)]
    fn intersect(&self, ray: &crate::picking::Ray, max_distance: f32) -> Option<(u32, u32)> {
        let mut closest = max_distance;
        let mut identity = None;
        // `build_recursive` appends each parent after its children, so the
        // root is the final node rather than node zero. Starting at node zero
        // only traverses the first leaf and made the CPU oracle report misses
        // for every pixel outside that leaf.
        let root = self.bvh.nodes.len().checked_sub(1)? as u32;
        let mut stack = vec![root];
        while let Some(node_index) = stack.pop() {
            let node = self.bvh.nodes.get(node_index as usize)?;
            if !ray_aabb(ray, node.aabb_min, node.aabb_max, closest) {
                continue;
            }
            if node.is_leaf() {
                for offset in 0..node.right {
                    let reordered = *self.bvh.tri_indices.get((node.left + offset) as usize)?;
                    let (v0, v1, v2) = self.mesh.get_triangle(reordered as usize)?;
                    if let Some(distance) = ray_triangle(ray, v0, v1, v2) {
                        if distance < closest {
                            closest = distance;
                            identity = self.identities.get(reordered as usize).copied();
                        }
                    }
                }
            } else {
                stack.push(node.right);
                stack.push(node.left);
            }
        }
        identity
    }
}

fn sample_height_bilinear(data: &[f32], dims: (u32, u32), uv: [f32; 2]) -> f32 {
    let fx = uv[0].clamp(0.0, 1.0) * dims.0.saturating_sub(1) as f32;
    let fy = uv[1].clamp(0.0, 1.0) * dims.1.saturating_sub(1) as f32;
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(dims.0.saturating_sub(1) as usize);
    let y1 = (y0 + 1).min(dims.1.saturating_sub(1) as usize);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let width = dims.0 as usize;
    let top = data[y0 * width + x0] * (1.0 - tx) + data[y0 * width + x1] * tx;
    let bottom = data[y1 * width + x0] * (1.0 - tx) + data[y1 * width + x1] * tx;
    top * (1.0 - ty) + bottom * ty
}

fn sample_clipmap_height(
    data: &[f32],
    dims: (u32, u32),
    vertex: &crate::terrain::clipmap::ClipmapVertex,
) -> f32 {
    let uv = [vertex.uv[0].clamp(0.0, 1.0), vertex.uv[1].clamp(0.0, 1.0)];
    let fine = sample_height_bilinear(data, dims, uv);
    let coarse_texels = 2.0_f32.powf((vertex.morph_data[1].max(0.0) + 1.0).min(16.0));
    let coarse_step = [
        coarse_texels / dims.0.saturating_sub(1).max(1) as f32,
        coarse_texels / dims.1.saturating_sub(1).max(1) as f32,
    ];
    let coarse_cell = [uv[0] / coarse_step[0], uv[1] / coarse_step[1]];
    let coarse_base = [
        coarse_cell[0].floor() * coarse_step[0],
        coarse_cell[1].floor() * coarse_step[1],
    ];
    let coarse_t = [coarse_cell[0].fract(), coarse_cell[1].fract()];
    let h00 = sample_height_bilinear(data, dims, coarse_base);
    let h10 = sample_height_bilinear(
        data,
        dims,
        [coarse_base[0] + coarse_step[0], coarse_base[1]],
    );
    let h01 = sample_height_bilinear(
        data,
        dims,
        [coarse_base[0], coarse_base[1] + coarse_step[1]],
    );
    let h11 = sample_height_bilinear(
        data,
        dims,
        [
            coarse_base[0] + coarse_step[0],
            coarse_base[1] + coarse_step[1],
        ],
    );
    let coarse_top = h00 + (h10 - h00) * coarse_t[0];
    let coarse_bottom = h01 + (h11 - h01) * coarse_t[0];
    let coarse = coarse_top + (coarse_bottom - coarse_top) * coarse_t[1];
    let morph = vertex.morph_data[0].clamp(0.0, 1.0);
    fine + (coarse - fine) * morph
}

fn apply_height_curve(
    raw: f32,
    height_range: (f32, f32),
    params: &crate::terrain::render_params::TerrainRenderParams,
) -> f32 {
    let range = (height_range.1 - height_range.0).max(1e-6);
    let t = ((raw - height_range.0) / range).clamp(0.0, 1.0);
    let strength = params.height_curve_strength.clamp(0.0, 1.0);
    let curved = match params.height_curve_mode.as_str() {
        "pow" => t.powf(params.height_curve_power.max(0.01)),
        "smoothstep" => t * t * (3.0 - 2.0 * t),
        "lut" => params
            .height_curve_lut
            .as_deref()
            .and_then(|lut| {
                let index = (t * lut.len().saturating_sub(1) as f32).round() as usize;
                lut.get(index).copied()
            })
            .unwrap_or(t),
        _ => t,
    };
    height_range.0 + (t + (curved - t) * strength) * range
}

fn build_clipped_raster_mesh(
    world_vertices: &[[f32; 3]],
    source_indices: &[[u32; 3]],
    source_identities: &[(u32, u32)],
    view: glam::Mat4,
    proj: glam::Mat4,
    viewport: (u32, u32),
) -> anyhow::Result<(crate::accel::cpu_bvh::MeshCPU, Vec<(u32, u32)>)> {
    const SUBPIXEL_SCALE: f32 = 256.0;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut identities = Vec::new();

    for (triangle, &identity) in source_indices.iter().zip(source_identities) {
        let mut clip = Vec::with_capacity(3);
        for &index in triangle {
            let position = world_vertices.get(index as usize).ok_or_else(|| {
                anyhow::anyhow!("visibility oracle index {index} is out of bounds")
            })?;
            // Mirror vs_clipmap_main exactly: it deliberately performs the
            // view and projection products as two ordered mat4*vec4 stages.
            // Collapsing them into `(proj * view) * position` moves rare
            // grazing-edge vertices across the rasterizer's subpixel grid.
            let view_position =
                deterministic_mat4_mul_vec4(view, glam::Vec3::from_array(*position).extend(1.0));
            clip.push(deterministic_mat4_mul_vec4(proj, view_position));
        }
        let polygon = clip_webgpu_polygon(clip);
        for fan in 1..polygon.len().saturating_sub(1) {
            let clip_triangle = [polygon[0], polygon[fan], polygon[fan + 1]];
            if clip_triangle.iter().any(|vertex| vertex.w <= 0.0) {
                continue;
            }
            let mut raster = [[0.0; 3]; 3];
            for (slot, vertex) in raster.iter_mut().zip(clip_triangle) {
                let ndc = vertex.truncate() / vertex.w;
                *slot = [
                    (((ndc.x * 0.5 + 0.5) * viewport.0 as f32) * SUBPIXEL_SCALE).round()
                        / SUBPIXEL_SCALE,
                    (((0.5 - ndc.y * 0.5) * viewport.1 as f32) * SUBPIXEL_SCALE).round()
                        / SUBPIXEL_SCALE,
                    ndc.z,
                ];
            }
            let area = (raster[1][0] - raster[0][0]) * (raster[2][1] - raster[0][1])
                - (raster[1][1] - raster[0][1]) * (raster[2][0] - raster[0][0]);
            if area == 0.0 {
                continue;
            }
            let base = vertices.len() as u32;
            vertices.extend(raster);
            indices.push([base, base + 1, base + 2]);
            // Clipping may split one source primitive into several raster
            // triangles, all of which retain the GPU's original primitive ID.
            identities.push(identity);
        }
    }

    if indices.is_empty() {
        anyhow::bail!("visibility oracle has no triangles inside the WebGPU clip volume");
    }
    Ok((
        crate::accel::cpu_bvh::MeshCPU::new(vertices, indices),
        identities,
    ))
}

/// CPU mirror of WGSL `det_mat4_mul_vec4`: column products followed by a
/// fixed left-to-right sum, with view and projection kept as separate calls.
fn deterministic_mat4_mul_vec4(matrix: glam::Mat4, vector: glam::Vec4) -> glam::Vec4 {
    let c0 = matrix.x_axis * vector.x;
    let c1 = matrix.y_axis * vector.y;
    let c2 = matrix.z_axis * vector.z;
    let c3 = matrix.w_axis * vector.w;
    let s01 = c0 + c1;
    let s012 = s01 + c2;
    s012 + c3
}

fn clip_webgpu_polygon(mut polygon: Vec<glam::Vec4>) -> Vec<glam::Vec4> {
    // WebGPU clip volume: -w <= x,y <= w and 0 <= z <= w.
    let planes: [fn(glam::Vec4) -> f32; 6] = [
        |v: glam::Vec4| v.x + v.w,
        |v: glam::Vec4| v.w - v.x,
        |v: glam::Vec4| v.y + v.w,
        |v: glam::Vec4| v.w - v.y,
        |v: glam::Vec4| v.z,
        |v: glam::Vec4| v.w - v.z,
    ];
    for plane in planes {
        polygon = clip_polygon_against_plane(&polygon, plane);
        if polygon.is_empty() {
            break;
        }
    }
    polygon
}

fn clip_polygon_against_plane(
    polygon: &[glam::Vec4],
    distance: fn(glam::Vec4) -> f32,
) -> Vec<glam::Vec4> {
    let Some(&last) = polygon.last() else {
        return Vec::new();
    };
    let mut output = Vec::with_capacity(polygon.len() + 1);
    let mut previous = last;
    let mut previous_distance = distance(previous);
    let mut previous_inside = previous_distance >= 0.0;

    for &current in polygon {
        let current_distance = distance(current);
        let current_inside = current_distance >= 0.0;
        if current_inside != previous_inside {
            let denominator = previous_distance - current_distance;
            if denominator != 0.0 {
                let t = (previous_distance / denominator).clamp(0.0, 1.0);
                output.push(previous + (current - previous) * t);
            }
        }
        if current_inside {
            output.push(current);
        }
        previous = current;
        previous_distance = current_distance;
        previous_inside = current_inside;
    }
    output
}

fn ray_aabb(ray: &crate::picking::Ray, min: [f32; 3], max: [f32; 3], limit: f32) -> bool {
    let mut near: f32 = 0.0;
    let mut far = limit;
    for axis in 0..3 {
        let inv = 1.0 / ray.direction[axis];
        let mut a = (min[axis] - ray.origin[axis]) * inv;
        let mut b = (max[axis] - ray.origin[axis]) * inv;
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        near = near.max(a);
        far = far.min(b);
        if near > far {
            return false;
        }
    }
    true
}

#[cfg(test)]
fn ray_triangle(
    ray: &crate::picking::Ray,
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
) -> Option<f32> {
    let origin = glam::Vec3::from_array(ray.origin);
    let direction = glam::Vec3::from_array(ray.direction);
    let a = glam::Vec3::from_array(v0);
    let edge1 = glam::Vec3::from_array(v1) - a;
    let edge2 = glam::Vec3::from_array(v2) - a;
    let p = direction.cross(edge2);
    let det = edge1.dot(p);
    if det.abs() < 1e-8 {
        return None;
    }
    let inv_det = det.recip();
    let t = origin - a;
    let u = t.dot(p) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = t.cross(edge1);
    let v = direction.dot(q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let distance = edge2.dot(q) * inv_det;
    (distance > 0.0).then_some(distance)
}

fn raster_top_left_covers(
    v0: [f32; 3],
    mut v1: [f32; 3],
    mut v2: [f32; 3],
    pixel: (u32, u32),
) -> bool {
    const SCALE: f32 = 256.0;
    let point = (
        i64::from(pixel.0) * 256 + 128,
        i64::from(pixel.1) * 256 + 128,
    );
    let signed_area = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0]);
    if signed_area == 0.0 {
        return false;
    }
    // Normalize to positive area in the framebuffer's Y-down coordinates.
    if signed_area < 0.0 {
        std::mem::swap(&mut v1, &mut v2);
    }
    let p = |v: [f32; 3]| ((v[0] * SCALE).round() as i64, (v[1] * SCALE).round() as i64);
    let a = p(v0);
    let b = p(v1);
    let c = p(v2);
    let covered_edge = |start: (i64, i64), end: (i64, i64)| {
        let dx = end.0 - start.0;
        let dy = end.1 - start.1;
        let edge = dx * (point.1 - start.1) - dy * (point.0 - start.0);
        edge > 0 || (edge == 0 && (dy < 0 || (dy == 0 && dx > 0)))
    };
    covered_edge(a, b) && covered_edge(b, c) && covered_edge(c, a)
}

/// Return WebGPU raster depth as the synthetic ray distance used by the CPU
/// BVH. Coverage and interpolation both use the already subpixel-snapped
/// screen vertices emitted by `build_clipped_raster_mesh`.
fn raster_distance_at_pixel(
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    pixel: (u32, u32),
) -> Option<f32> {
    if !raster_top_left_covers(v0, v1, v2, pixel) {
        return None;
    }
    let point = glam::vec2(pixel.0 as f32 + 0.5, pixel.1 as f32 + 0.5);
    let a = glam::Vec2::from_array([v0[0], v0[1]]);
    let b = glam::Vec2::from_array([v1[0], v1[1]]);
    let c = glam::Vec2::from_array([v2[0], v2[1]]);
    let area = (b - a).perp_dot(c - a);
    if area == 0.0 {
        return None;
    }
    let w0 = (b - point).perp_dot(c - point) / area;
    let w1 = (c - point).perp_dot(a - point) / area;
    let depth = w0 * v0[2] + w1 * v1[2] + (1.0 - w0 - w1) * v2[2];
    (0.0..=1.0).contains(&depth).then_some(1.0 + depth)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_oracle_traverses_the_postorder_bvh_root() {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for triangle in 0..6u32 {
            let x = triangle as f32 * 2.0;
            let first = vertices.len() as u32;
            vertices.extend([[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0], [x, 1.0, 0.0]]);
            indices.push([first, first + 1, first + 2]);
        }
        let mesh = crate::accel::cpu_bvh::MeshCPU::new(vertices, indices);
        let bvh = crate::accel::cpu_bvh::build_bvh_cpu(
            &mesh,
            &crate::accel::cpu_bvh::BuildOptions::default(),
        )
        .expect("build test BVH");
        assert!(bvh.nodes.len() > 1);
        assert!(bvh.nodes[0].is_leaf());
        let oracle = CpuVisibilityOracle {
            mesh,
            bvh,
            identities: (0..6).map(|triangle| (triangle, 0)).collect(),
        };

        assert_eq!(
            oracle.intersect(
                &crate::picking::Ray::new([10.25, 0.25, 1.0], [0.0, 0.0, -1.0]),
                f32::INFINITY,
            ),
            Some((5, 0)),
        );
    }

    #[test]
    fn cpu_oracle_rejects_hits_beyond_the_far_clip_plane() {
        let mesh = crate::accel::cpu_bvh::MeshCPU::new(
            vec![[0.0, 0.0, -2.0], [1.0, 0.0, -2.0], [0.0, 1.0, -2.0]],
            vec![[0, 1, 2]],
        );
        let bvh = crate::accel::cpu_bvh::build_bvh_cpu(
            &mesh,
            &crate::accel::cpu_bvh::BuildOptions::default(),
        )
        .expect("build test BVH");
        let oracle = CpuVisibilityOracle {
            mesh,
            bvh,
            identities: vec![(7, 9)],
        };
        let ray = crate::picking::Ray::new([0.25, 0.25, 0.0], [0.0, 0.0, -1.0]);

        assert_eq!(oracle.intersect(&ray, 1.0), None);
        assert_eq!(oracle.intersect(&ray, 3.0), Some((7, 9)));
    }

    #[test]
    fn raster_top_left_rule_assigns_a_shared_diagonal_once() {
        let first = [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [0.0, 1.0, 0.5]];
        let second = [[1.0, 0.0, 0.5], [1.0, 1.0, 0.5], [0.0, 1.0, 0.5]];

        assert!(!raster_top_left_covers(
            first[0],
            first[1],
            first[2],
            (0, 0),
        ));
        assert!(raster_top_left_covers(
            second[0],
            second[1],
            second[2],
            (0, 0),
        ));
        assert_eq!(
            raster_distance_at_pixel(second[0], second[1], second[2], (0, 0)),
            Some(1.5),
            "a GPU-owned edge pixel must retain a depth for CPU picking"
        );
    }

    #[test]
    fn homogeneous_clipping_bounds_a_near_plane_crossing_triangle() {
        let polygon = clip_webgpu_polygon(vec![
            glam::vec4(-0.5, -0.5, -0.5, 1.0),
            glam::vec4(0.5, -0.5, 0.5, 1.0),
            glam::vec4(0.0, 0.5, 0.5, 1.0),
        ]);

        assert_eq!(polygon.len(), 4);
        for vertex in polygon {
            assert!(vertex.x >= -vertex.w && vertex.x <= vertex.w);
            assert!(vertex.y >= -vertex.w && vertex.y <= vertex.w);
            assert!(vertex.z >= 0.0 && vertex.z <= vertex.w);
        }
    }

    #[test]
    fn clipped_raster_mesh_preserves_source_primitive_identity() {
        let (mesh, identities) = build_clipped_raster_mesh(
            &[[-0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.0, 0.5, 0.5]],
            &[[0, 1, 2]],
            &[(7, 11)],
            glam::Mat4::IDENTITY,
            glam::Mat4::IDENTITY,
            (64, 64),
        )
        .expect("clip one near-plane crossing triangle");

        assert_eq!(mesh.indices.len(), 2);
        assert_eq!(identities, vec![(7, 11), (7, 11)]);
        assert!(mesh
            .vertices
            .iter()
            .all(|vertex| (0.0..=1.0).contains(&vertex[2])));
    }

    #[test]
    fn raster_projection_keeps_shader_view_then_projection_order() {
        let view = glam::Mat4::from_cols(
            glam::vec4(1.0, 0.0, 0.0, 0.0),
            glam::vec4(0.0, 1.0, 0.0, 0.0),
            glam::vec4(0.0, 0.0, 1.0, 0.0),
            glam::vec4(0.25, -0.5, 0.75, 1.0),
        );
        let proj = glam::Mat4::from_cols(
            glam::vec4(1.5, 0.0, 0.0, 0.0),
            glam::vec4(0.0, 2.0, 0.0, 0.0),
            glam::vec4(0.0, 0.0, 0.5, 1.0),
            glam::vec4(0.0, 0.0, 0.25, 0.0),
        );
        let position = glam::vec4(0.125, -0.25, 0.5, 1.0);

        let staged = deterministic_mat4_mul_vec4(proj, deterministic_mat4_mul_vec4(view, position));
        let expected = glam::vec4(0.5625, -1.5, 0.875, 1.25);
        assert_eq!(staged, expected);
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct VisibilityCounters {
    visible_pixels: u32,
    feedback_records: u32,
    material_invocations: u32,
    background_pixels: u32,
    fallback_texels: u32,
    forward_material_invocations: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VisibilityStats {
    pub visible_pixels: u32,
    /// Feedback records emitted by the most recent frame.
    pub feedback_records: u32,
    /// Feedback records emitted by the visibility-resolve path.
    pub visibility_feedback_records: u32,
    /// Feedback records emitted by the forward path, including overdraw.
    pub forward_feedback_records: u32,
    pub material_invocations: u32,
    pub background_pixels: u32,
    pub fallback_texels: u32,
    pub forward_material_invocations: u32,
}

static LAST_STATS: OnceLock<Mutex<VisibilityStats>> = OnceLock::new();

pub fn publish_stats(stats: VisibilityStats) {
    if let Ok(mut current) = LAST_STATS
        .get_or_init(|| Mutex::new(VisibilityStats::default()))
        .lock()
    {
        let mut stats = stats;
        if stats.forward_material_invocations > 0 {
            stats.forward_feedback_records = stats.feedback_records;
            stats.visibility_feedback_records = current.visibility_feedback_records;
        } else if stats.material_invocations > 0 {
            stats.visibility_feedback_records = stats.feedback_records;
            stats.forward_feedback_records = current.forward_feedback_records;
            stats.forward_material_invocations = current.forward_material_invocations;
        }
        *current = stats;
    }
}

pub fn latest_stats() -> VisibilityStats {
    LAST_STATS
        .get_or_init(|| Mutex::new(VisibilityStats::default()))
        .lock()
        .map(|stats| *stats)
        .unwrap_or_default()
}

pub struct TerrainVisibilityBuffer {
    width: u32,
    height: u32,
    _texture: TrackedTexture,
    view: wgpu::TextureView,
    stats_buffer: TrackedBuffer,
    stats_readback: TrackedBuffer,
    stats_bind_group: wgpu::BindGroup,
    stats_pipeline: wgpu::ComputePipeline,
    staged: bool,
}

impl TerrainVisibilityBuffer {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        frame_counters: &wgpu::Buffer,
    ) -> RenderResult<Self> {
        let width = width.max(1);
        let height = height.max(1);
        let texture = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("terrain.visibility.ids"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Uint,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )?;
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let stats_buffer = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("terrain.visibility.stats"),
                size: std::mem::size_of::<VisibilityCounters>() as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        let stats_readback = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("terrain.visibility.stats_readback"),
                size: std::mem::size_of::<VisibilityCounters>() as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain.visibility.stats.layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let stats_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.visibility.stats.bind_group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: stats_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: frame_counters.as_entire_binding(),
                },
            ],
        });
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "terrain_visbuffer_resolve",
            include_str!("../../shaders/terrain_visbuffer_resolve.wgsl"),
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain.visibility.stats.pipeline_layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let stats_pipeline = crate::core::shader_registry::with_error_scope(
            device,
            "terrain.visibility.stats.pipeline",
            || {
                crate::core::shader_registry::create_compute_pipeline_scoped(
                    device,
                    &wgpu::ComputePipelineDescriptor {
                        label: Some("terrain.visibility.stats.pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &shader,
                        entry_point: "cs_main",
                    },
                )
            },
        );
        Ok(Self {
            width,
            height,
            _texture: texture,
            view,
            stats_buffer,
            stats_readback,
            stats_bind_group,
            stats_pipeline,
            staged: false,
        })
    }

    pub fn matches(&self, width: u32, height: u32) -> bool {
        self.width == width.max(1) && self.height == height.max(1)
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self._texture
    }

    pub fn stage_stats(&mut self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.stats_buffer, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain.visibility.stats.pass"),
                timestamp_writes: None,
            });
            crate::core::shader_registry::record_shader_use("terrain_visbuffer_resolve");
            pass.set_pipeline(&self.stats_pipeline);
            pass.set_bind_group(0, &self.stats_bind_group, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.stats_buffer,
            0,
            &self.stats_readback,
            0,
            std::mem::size_of::<VisibilityCounters>() as u64,
        );
        self.staged = true;
    }

    pub fn finish_frame(&mut self, device: &wgpu::Device) -> RenderResult<VisibilityStats> {
        if !self.staged {
            return Ok(VisibilityStats::default());
        }
        let slice = self.stats_readback.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver.receive())
            .ok_or_else(|| RenderError::render("visibility stats callback dropped"))?
            .map_err(|error| {
                RenderError::render(format!("visibility stats map failed: {error}"))
            })?;
        let mapped = slice.get_mapped_range();
        let counters = bytemuck::pod_read_unaligned::<VisibilityCounters>(&mapped);
        drop(mapped);
        self.stats_readback.unmap();
        self.staged = false;
        Ok(VisibilityStats {
            visible_pixels: counters.visible_pixels,
            feedback_records: counters.feedback_records,
            visibility_feedback_records: 0,
            forward_feedback_records: 0,
            material_invocations: counters.material_invocations,
            background_pixels: counters.background_pixels,
            fallback_texels: counters.fallback_texels,
            forward_material_invocations: counters.forward_material_invocations,
        })
    }
}

#[cfg(feature = "extension-module")]
impl super::TerrainScene {
    pub(super) fn create_visibility_resolve_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        // The runtime resolve entry point is `fs_visibility_geometry`. It only
        // reads the visibility ID texture from group 7; vertices, indices,
        // templates, meta, and sampled depth belong exclusively to the static
        // fullscreen reconstruction helper. Do not put those resources in the
        // geometry bind group: binding the live depth attachment as a sampled
        // texture in this same depth-equal pass is a WebGPU usage conflict,
        // even on backends that prune unused entry-point resources.
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain.visibility.geometry_resolve.layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        })
    }

    pub(super) fn visibility_resolve_bind_group(&self) -> anyhow::Result<wgpu::BindGroup> {
        let visibility = self
            .visibility_buffer
            .lock()
            .map_err(|_| anyhow::anyhow!("terrain visibility buffer mutex poisoned"))?;
        let buffer = visibility
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("terrain visibility buffer not initialized"))?;
        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.visibility.geometry_resolve.bind_group"),
            layout: &self.visibility_resolve_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(buffer.view()),
            }],
        }))
    }

    pub(super) fn ensure_visibility_buffer(&self, width: u32, height: u32) -> anyhow::Result<()> {
        let mut visibility = self
            .visibility_buffer
            .lock()
            .map_err(|_| anyhow::anyhow!("terrain visibility buffer mutex poisoned"))?;
        if visibility
            .as_ref()
            .is_none_or(|buffer| !buffer.matches(width, height))
        {
            *visibility = Some(
                TerrainVisibilityBuffer::new(
                    self.device.as_ref(),
                    width,
                    height,
                    &self.vt_frame_counters_buffer,
                )
                .map_err(anyhow::Error::msg)?,
            );
        }
        Ok(())
    }

    pub(super) fn stage_visibility_stats(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> anyhow::Result<()> {
        let mut visibility = self
            .visibility_buffer
            .lock()
            .map_err(|_| anyhow::anyhow!("terrain visibility buffer mutex poisoned"))?;
        let buffer = visibility
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("terrain visibility buffer not initialized"))?;
        buffer.stage_stats(encoder);
        Ok(())
    }

    pub(super) fn finish_visibility_frame(&self) -> anyhow::Result<VisibilityStats> {
        let mut visibility = self
            .visibility_buffer
            .lock()
            .map_err(|_| anyhow::anyhow!("terrain visibility buffer mutex poisoned"))?;
        let Some(buffer) = visibility.as_mut() else {
            let stats = VisibilityStats::default();
            publish_stats(stats);
            return Ok(stats);
        };
        let stats = buffer
            .finish_frame(self.device.as_ref())
            .map_err(anyhow::Error::msg)?;
        publish_stats(stats);
        Ok(stats)
    }

    pub(super) fn pick_visibility_pixels(
        &self,
        pixels: &[(u32, u32)],
    ) -> anyhow::Result<Vec<Option<(u32, u32)>>> {
        let visibility = self
            .visibility_buffer
            .lock()
            .map_err(|_| anyhow::anyhow!("terrain visibility buffer mutex poisoned"))?;
        let buffer = visibility
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no completed visibility render is available"))?;
        let picking =
            crate::picking::UnifiedPickingSystem::new(self.device.clone(), self.queue.clone());
        picking
            .pick_visibility_pixels(buffer.texture(), buffer.width, buffer.height, pixels)
            .map_err(anyhow::Error::msg)
    }

    pub(super) fn pick_visibility_pixels_cpu(
        &self,
        pixels: &[(u32, u32)],
    ) -> anyhow::Result<Vec<Option<(u32, u32)>>> {
        let oracle = self
            .cpu_visibility_oracle
            .lock()
            .map_err(|_| anyhow::anyhow!("terrain CPU visibility oracle mutex poisoned"))?;
        let oracle = oracle
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no CPU BVH visibility oracle is available"))?;
        Ok(oracle.pick(pixels))
    }
}
