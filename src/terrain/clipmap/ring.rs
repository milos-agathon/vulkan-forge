//! P2.1/M5: Clipmap ring mesh generation with skirts.
//!
//! Generates hollow ring meshes (donut shapes) for each LOD level,
//! plus the solid center block at finest resolution.

use super::vertex::ClipmapVertex;
use glam::Vec2;
use std::collections::BTreeMap;

/// Generate the center block mesh (solid grid at finest LOD).
pub fn make_center_block(
    resolution: u32,
    center: Vec2,
    half_extent: f32,
    terrain_extent: f32,
) -> (Vec<ClipmapVertex>, Vec<u32>) {
    let n = resolution as usize;
    let cell_size = (half_extent * 2.0) / resolution as f32;
    let mut vertices = Vec::with_capacity((n + 1) * (n + 1));
    let mut indices = Vec::with_capacity(n * n * 6);

    for y in 0..=n {
        for x in 0..=n {
            let wx = center.x - half_extent + x as f32 * cell_size;
            let wz = center.y - half_extent + y as f32 * cell_size;
            let u = (wx + terrain_extent * 0.5) / terrain_extent;
            let v = (wz + terrain_extent * 0.5) / terrain_extent;
            vertices.push(ClipmapVertex::center(
                wx,
                wz,
                u.clamp(0.0, 1.0),
                v.clamp(0.0, 1.0),
            ));
        }
    }

    let stride = n + 1;
    for y in 0..n {
        for x in 0..n {
            let i0 = (y * stride + x) as u32;
            let i1 = i0 + 1;
            let i2 = i0 + stride as u32;
            let i3 = i2 + 1;
            // CCW winding
            indices.extend_from_slice(&[i0, i1, i2, i1, i3, i2]);
        }
    }

    (vertices, indices)
}

/// Generate a single clipmap ring (hollow donut shape).
///
/// The ring is a non-overlapping annular grid. The grid spacing is the
/// next-coarser, world-anchored lattice (twice the previous ring's spacing),
/// while the central square is omitted. Exact inner and outer boundary
/// coordinates are added to that lattice, so arbitrary/odd extents close
/// exactly without shifting the interior lattice phase when the center moves.
pub fn make_ring(
    ring_index: u32,
    inner_extent: f32,
    outer_extent: f32,
    resolution: u32,
    center: Vec2,
    terrain_extent: f32,
    morph_range: f32,
) -> (Vec<ClipmapVertex>, Vec<u32>) {
    assert!(outer_extent > inner_extent);
    assert!(resolution > 0);
    let ring_width = outer_extent - inner_extent;
    // Ring vertices are intentionally one LOD coarser than the region inside.
    let cell_size = 2.0 * ring_width / resolution as f32;
    assert!(cell_size.is_finite() && cell_size > 0.0);

    // Never derive the lattice from `center - outer_extent`: that changes its
    // phase on every clipmap recenter. Instead retain every world-lattice line
    // inside the footprint and splice in the moving boundaries as needed.
    let x_coords = anchored_coordinates(
        center.x - outer_extent,
        center.x + outer_extent,
        center.x - inner_extent,
        center.x + inner_extent,
        cell_size,
    );
    let y_coords = anchored_coordinates(
        center.y - outer_extent,
        center.y + outer_extent,
        center.y - inner_extent,
        center.y + inner_extent,
        cell_size,
    );
    let mut vertices = Vec::with_capacity(x_coords.len() * y_coords.len());
    let mut indices = Vec::new();

    let morph_range = morph_range.clamp(0.0, 1.0);
    let morph_weight = |position: Vec2| {
        let radial = (position - center).abs().max_element();
        let t = ((radial - inner_extent) / ring_width).clamp(0.0, 1.0);
        if morph_range <= f32::EPSILON {
            return 0.0;
        }
        let edge_t = ((t - (1.0 - morph_range)) / morph_range).clamp(0.0, 1.0);
        edge_t * edge_t * (3.0 - 2.0 * edge_t)
    };

    let to_uv = |wx: f32, wz: f32| -> (f32, f32) {
        let u = (wx + terrain_extent * 0.5) / terrain_extent;
        let v = (wz + terrain_extent * 0.5) / terrain_extent;
        (u.clamp(0.0, 1.0), v.clamp(0.0, 1.0))
    };

    for &wz in &y_coords {
        for &wx in &x_coords {
            let position = Vec2::new(wx, wz);
            let (u, v) = to_uv(position.x, position.y);
            vertices.push(ClipmapVertex::new(
                position.x,
                position.y,
                u,
                v,
                morph_weight(position),
                ring_index,
            ));
        }
    }

    let stride = x_coords.len();
    for y in 0..y_coords.len() - 1 {
        for x in 0..x_coords.len() - 1 {
            // The exact inner-boundary coordinates ensure no cell straddles the
            // hole. Midpoint ownership therefore emits each annular cell once.
            let midpoint = Vec2::new(
                (x_coords[x] + x_coords[x + 1]) * 0.5,
                (y_coords[y] + y_coords[y + 1]) * 0.5,
            );
            if (midpoint.x - center.x).abs() < inner_extent
                && (midpoint.y - center.y).abs() < inner_extent
            {
                continue;
            }
            let i0 = (y * stride + x) as u32;
            let i1 = i0 + 1;
            let i2 = i0 + stride as u32;
            let i3 = i2 + 1;
            indices.extend_from_slice(&[i0, i1, i2, i1, i3, i2]);
        }
    }

    (vertices, indices)
}

/// Coordinate lines for one dimension of an annular grid.
///
/// The interior lines are all multiples of `spacing` in world space. Boundary
/// lines are inserted separately so changing a clipmap extent does not require
/// rounding that extent to the lattice (which would leave a gap or overlap).
fn anchored_coordinates(
    min: f32,
    max: f32,
    inner_min: f32,
    inner_max: f32,
    spacing: f32,
) -> Vec<f32> {
    debug_assert!(min < max);
    debug_assert!(spacing.is_finite() && spacing > 0.0);

    let first = (min / spacing).ceil() as i64;
    let last = (max / spacing).floor() as i64;
    let mut coordinates = Vec::with_capacity((last - first + 5).max(5) as usize);
    coordinates.extend([min, inner_min, inner_max, max]);
    for index in first..=last {
        coordinates.push(index as f32 * spacing);
    }
    coordinates.sort_by(|a, b| a.total_cmp(b));
    coordinates.dedup_by(|a, b| (*a - *b).abs() <= spacing.abs() * 1e-6);
    coordinates
}

/// Generate skirt vertices for a ring to hide seams.
///
/// Boundary edges are derived from the indexed annulus itself. This covers both
/// its outer edge and the central hole, without assuming the producer's row
/// layout or accidentally joining unrelated strips.
pub fn make_ring_skirts(
    vertices: &[ClipmapVertex],
    indices: &[u32],
    skirt_depth: f32,
    ring_index: u32,
    _row_width: usize,
) -> (Vec<ClipmapVertex>, Vec<u32>) {
    let mut skirt_verts = Vec::new();
    let mut skirt_indices = Vec::new();

    // Ordered keys make skirt vertex/index emission byte-stable for render
    // certificates and golden comparisons.
    let mut edges: BTreeMap<(u32, u32), (u32, u32)> = BTreeMap::new();
    for triangle in indices.chunks_exact(3) {
        for (a, b) in [
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ] {
            let key = (a.min(b), a.max(b));
            if edges.remove(&key).is_none() {
                edges.insert(key, (a, b));
            }
        }
    }

    let base_idx = vertices.len() as u32;
    let mut skirt_for = BTreeMap::new();
    let mut skirt_index = |source: u32, skirt_verts: &mut Vec<ClipmapVertex>| {
        *skirt_for.entry(source).or_insert_with(|| {
            let v = vertices[source as usize];
            let index = base_idx + skirt_verts.len() as u32;
            skirt_verts.push(ClipmapVertex::skirt(
                v.position[0],
                v.position[1],
                v.uv[0],
                v.uv[1],
                ring_index,
            ));
            index
        })
    };
    for (_key, (a, b)) in edges {
        let skirt_a = skirt_index(a, &mut skirt_verts);
        let skirt_b = skirt_index(b, &mut skirt_verts);
        skirt_indices.extend_from_slice(&[a, b, skirt_a, b, skirt_b, skirt_a]);
    }

    let _ = skirt_depth; // Used in shader for the vertical offset.
    (skirt_verts, skirt_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_center_block_vertex_count() {
        let (verts, indices) = make_center_block(4, Vec2::ZERO, 10.0, 100.0);
        assert_eq!(verts.len(), 25); // 5x5 vertices for 4x4 cells
        assert_eq!(indices.len(), 4 * 4 * 6); // 16 quads * 6 indices
    }

    #[test]
    fn test_center_block_ccw_winding() {
        let (verts, indices) = make_center_block(2, Vec2::ZERO, 10.0, 100.0);
        // First triangle
        let i0 = indices[0] as usize;
        let i1 = indices[1] as usize;
        let i2 = indices[2] as usize;
        let v0 = Vec2::from(verts[i0].position);
        let v1 = Vec2::from(verts[i1].position);
        let v2 = Vec2::from(verts[i2].position);
        // CCW check: cross product should be positive
        let cross = (v1 - v0).perp_dot(v2 - v0);
        assert!(cross > 0.0, "First triangle should be CCW");
    }

    #[test]
    fn test_ring_generation() {
        let (verts, indices) = make_ring(1, 10.0, 20.0, 8, Vec2::ZERO, 100.0, 0.3);
        // 16x16 coarse cells with an 8x8 center hole: every emitted cell is
        // owned exactly once, unlike the former overlapping four strips.
        assert_eq!(verts.len(), 17 * 17);
        assert_eq!(indices.len(), (16 * 16 - 8 * 8) * 6);
        assert_eq!(indices.len() % 3, 0);
    }

    #[test]
    fn test_morph_weights_in_range() {
        let (verts, _) = make_ring(1, 10.0, 20.0, 8, Vec2::ZERO, 100.0, 0.3);
        for v in &verts {
            assert!(v.morph_weight() >= 0.0 && v.morph_weight() <= 1.0);
        }
        assert!(verts.iter().any(|v| {
            let weight = v.morph_weight();
            weight > 0.0 && weight < 1.0
        }));
    }

    #[test]
    fn test_ring_keeps_world_lattice_phase_when_center_moves() {
        let (initial, _) = make_ring(1, 10.0, 20.0, 8, Vec2::ZERO, 100.0, 0.3);
        let (recentered, _) = make_ring(1, 10.0, 20.0, 8, Vec2::new(0.5, 0.5), 100.0, 0.3);

        // (0, 15) is an unchanged world-lattice vertex inside both annuli.
        // A center-shifted lattice would move this vertex to (0.5, 15.5).
        let retained = |vertices: &[ClipmapVertex]| {
            vertices.iter().any(|vertex| {
                (vertex.position[0] - 0.0).abs() < 1e-6 && (vertex.position[1] - 15.0).abs() < 1e-6
            })
        };
        assert!(retained(&initial));
        assert!(retained(&recentered));
    }

    #[test]
    fn test_odd_mismatched_extents_have_exact_annular_coverage() {
        let center = Vec2::new(0.35, -0.6);
        let inner = 7.25;
        let outer = 19.8;
        let (vertices, indices) = make_ring(2, inner, outer, 7, center, 100.0, 0.3);

        for (coordinate, expected) in [
            (0, center.x - outer),
            (0, center.x - inner),
            (0, center.x + inner),
            (0, center.x + outer),
            (1, center.y - outer),
            (1, center.y - inner),
            (1, center.y + inner),
            (1, center.y + outer),
        ] {
            assert!(vertices
                .iter()
                .any(|vertex| { (vertex.position[coordinate] - expected).abs() < 1e-5 }));
        }

        let covered_area: f32 = indices
            .chunks_exact(3)
            .map(|triangle| {
                let a = Vec2::from(vertices[triangle[0] as usize].position);
                let b = Vec2::from(vertices[triangle[1] as usize].position);
                let c = Vec2::from(vertices[triangle[2] as usize].position);
                (b - a).perp_dot(c - a).abs() * 0.5
            })
            .sum();
        let expected_area = 4.0 * (outer * outer - inner * inner);
        assert!(
            (covered_area - expected_area).abs() < 1e-2,
            "covered area {covered_area} did not match exact annulus area {expected_area}"
        );
    }

    #[test]
    fn test_single_cell_resolution_is_safe() {
        let (vertices, indices) = make_ring(0, 1.0, 2.0, 1, Vec2::ZERO, 100.0, 0.0);
        assert!(!vertices.is_empty());
        assert!(!indices.is_empty());
    }

    #[test]
    fn test_skirts_follow_indexed_annulus_boundaries() {
        let resolution = 8u32;
        let (inner, outer) = (10.0f32, 20.0f32);
        let (verts, indices) = make_ring(1, inner, outer, resolution, Vec2::ZERO, 100.0, 0.3);
        let (skirt_verts, skirt_indices) = make_ring_skirts(&verts, &indices, 5.0, 1, 0);

        let all: Vec<ClipmapVertex> = verts
            .iter()
            .copied()
            .chain(skirt_verts.iter().copied())
            .collect();
        let boundary_edges = 4 * 16 + 4 * 8; // outer plus central-hole perimeter
        let adjacent_spacing = (outer - inner) / resolution as f32 * 2.0;
        let mut max_edge = 0.0f32;
        for tri in skirt_indices.chunks(3) {
            for k in 0..3 {
                let a = Vec2::from(all[tri[k] as usize].position);
                let b = Vec2::from(all[tri[(k + 1) % 3] as usize].position);
                max_edge = max_edge.max(a.distance(b));
            }
        }
        assert_eq!(skirt_verts.len(), boundary_edges);
        assert_eq!(skirt_indices.len(), boundary_edges * 6);
        assert!(
            max_edge <= adjacent_spacing + 1e-4,
            "skirt edge {} exceeds adjacent vertex spacing {}",
            max_edge,
            adjacent_spacing
        );
    }
}
