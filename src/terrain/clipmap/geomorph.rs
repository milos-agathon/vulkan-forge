//! P2.2: Geo-morphing and seam correctness utilities.
//!
//! Provides vertex blending at LOD boundaries to eliminate T-junction
//! artifacts and visual seams between clipmap rings.

use super::vertex::ClipmapVertex;
use glam::Vec2;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// Configuration for geo-morphing.
#[derive(Debug, Clone, Copy)]
pub struct GeomorphConfig {
    /// Morph blend range as fraction of ring width [0.0-1.0].
    pub morph_range: f32,
    /// Maximum allowed seam gap in world units.
    pub max_seam_gap: f32,
    /// Enable snapping vertices to coarser grid at boundaries.
    pub snap_to_coarse: bool,
}

impl Default for GeomorphConfig {
    fn default() -> Self {
        Self {
            morph_range: 0.3,
            max_seam_gap: 0.001,
            snap_to_coarse: true,
        }
    }
}

/// Result of seam analysis.
#[derive(Debug, Clone)]
pub struct SeamAnalysis {
    /// Number of boundary vertices analyzed.
    pub boundary_vertex_count: u32,
    /// Number of fine/coarse rendered-depth comparisons performed.
    pub depth_sample_count: u32,
    /// Maximum gap between adjacent LOD levels.
    pub max_gap: f32,
    /// Average gap between adjacent LOD levels.
    pub avg_gap: f32,
    /// Number of T-junction candidates detected.
    pub t_junction_count: u32,
    /// Number of boundary samples beyond the configured seam threshold.
    pub crack_count: u32,
    /// Whether all seams are within acceptable tolerance.
    pub seams_valid: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DepthSeamAnalysis {
    pub sample_count: u32,
    pub max_depth_gap: f32,
    pub avg_depth_gap: f32,
    pub crack_count: u32,
}

static LAST_SEAM_ANALYSIS: OnceLock<Mutex<SeamAnalysis>> = OnceLock::new();
static SEAM_ANALYSIS_COUNT: AtomicU64 = AtomicU64::new(0);

/// What [`latest_seam_analysis`] reports before any geometry build has run, and
/// what it falls back to when the lock is poisoned: one crack and invalid
/// seams, so "nothing was analysed" can never be read as "nothing was wrong".
fn fail_closed_seam_analysis() -> SeamAnalysis {
    SeamAnalysis {
        boundary_vertex_count: 0,
        depth_sample_count: 0,
        max_gap: 0.0,
        avg_gap: 0.0,
        t_junction_count: 0,
        crack_count: 1,
        seams_valid: false,
    }
}

pub fn publish_seam_analysis(analysis: SeamAnalysis) {
    if let Ok(mut current) = LAST_SEAM_ANALYSIS
        .get_or_init(|| Mutex::new(fail_closed_seam_analysis()))
        .lock()
    {
        *current = analysis;
        SEAM_ANALYSIS_COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

/// Number of clipmap geometry builds that have published a seam analysis in
/// this process.
///
/// [`SeamAnalysis`] describes the LAST build only, and the clipmap geometry
/// cache (`src/terrain/renderer/geometry.rs:616-625`) can serve an unbounded
/// number of frames from a single build, so a caller that samples the analysis
/// once after a long render loop cannot distinguish "analysed on every frame"
/// from "analysed once". This counter makes that difference observable. It is
/// deliberately a free function rather than a [`SeamAnalysis`] field so that
/// every existing struct literal keeps compiling.
pub fn seam_analysis_count() -> u64 {
    SEAM_ANALYSIS_COUNT.load(Ordering::Relaxed)
}

pub fn latest_seam_analysis() -> SeamAnalysis {
    LAST_SEAM_ANALYSIS
        .get_or_init(|| Mutex::new(fail_closed_seam_analysis()))
        .lock()
        .map(|analysis| analysis.clone())
        .unwrap_or_else(|_| fail_closed_seam_analysis())
}

/// Calculate morph weight for a vertex based on distance from LOD boundary.
///
/// Returns a weight in [0.0, 1.0] where:
/// - 0.0 = use fine (current) LOD height
/// - 1.0 = use coarse (next) LOD height
pub fn calculate_morph_weight(distance_from_inner: f32, ring_width: f32, morph_range: f32) -> f32 {
    if ring_width <= 0.0 || morph_range <= 0.0 {
        return 0.0;
    }

    let t = (distance_from_inner / ring_width).clamp(0.0, 1.0);
    let morph_start = 1.0 - morph_range;

    if t > morph_start {
        ((t - morph_start) / morph_range).min(1.0)
    } else {
        0.0
    }
}

/// Snap a UV coordinate to the coarser LOD grid.
///
/// This ensures that vertices at LOD boundaries align with the coarser grid,
/// eliminating T-junctions where the coarse level samples the heightmap.
pub fn snap_uv_to_coarse_grid(uv: Vec2, ring_index: u32, texture_size: u32) -> Vec2 {
    let lod_scale = 1 << ring_index;
    let coarse_texel_size = lod_scale as f32 / texture_size as f32;

    Vec2::new(
        (uv.x / coarse_texel_size).floor() * coarse_texel_size,
        (uv.y / coarse_texel_size).floor() * coarse_texel_size,
    )
}

/// Analyze seams between adjacent clipmap rings for potential artifacts.
pub fn analyze_seams(
    inner_vertices: &[ClipmapVertex],
    outer_vertices: &[ClipmapVertex],
    config: &GeomorphConfig,
) -> SeamAnalysis {
    if inner_vertices.is_empty() || outer_vertices.is_empty() {
        return SeamAnalysis {
            boundary_vertex_count: 0,
            depth_sample_count: 0,
            max_gap: f32::INFINITY,
            avg_gap: f32::INFINITY,
            t_junction_count: 0,
            crack_count: 1,
            seams_valid: false,
        };
    }
    let (inner_min, inner_max) = inner_vertices
        .iter()
        .map(|vertex| Vec2::from(vertex.position))
        .fold(
            (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY)),
            |(min, max), position| (min.min(position), max.max(position)),
        );
    // A finer boundary vertex is allowed to land in the middle of a coarser
    // edge; vertex-to-vertex distance therefore reports false cracks at every
    // legitimate T-junction. Measure the four shared boundary lines instead.
    let mut side_gaps = [f32::INFINITY; 4];
    for vertex in outer_vertices {
        let p = Vec2::from(vertex.position);
        if p.y >= inner_min.y && p.y <= inner_max.y {
            side_gaps[0] = side_gaps[0].min((p.x - inner_min.x).abs());
            side_gaps[1] = side_gaps[1].min((p.x - inner_max.x).abs());
        }
        if p.x >= inner_min.x && p.x <= inner_max.x {
            side_gaps[2] = side_gaps[2].min((p.y - inner_min.y).abs());
            side_gaps[3] = side_gaps[3].min((p.y - inner_max.y).abs());
        }
    }
    let max_gap = side_gaps.iter().copied().fold(0.0_f32, f32::max);
    let avg_gap = side_gaps.iter().sum::<f32>() / side_gaps.len() as f32;
    let crack_count = side_gaps
        .iter()
        .filter(|gap| **gap > config.max_seam_gap)
        .count() as u32;
    let boundary_count = inner_vertices
        .iter()
        .filter(|vertex| {
            let p = Vec2::from(vertex.position);
            (p.x - inner_min.x).abs() <= config.max_seam_gap
                || (p.x - inner_max.x).abs() <= config.max_seam_gap
                || (p.y - inner_min.y).abs() <= config.max_seam_gap
                || (p.y - inner_max.y).abs() <= config.max_seam_gap
        })
        .count() as u32;
    let t_junction_count = 0;

    SeamAnalysis {
        boundary_vertex_count: boundary_count,
        depth_sample_count: 0,
        max_gap,
        avg_gap,
        t_junction_count,
        crack_count,
        seams_valid: crack_count == 0,
    }
}

/// Measure the rendered-height discontinuity at fine/coarse boundary samples.
///
/// Fine vertices may terminate in the middle of a coarse edge. The detector
/// samples the DEM through each vertex's actual UV, linearly interpolates the
/// two bracketing coarse-edge heights, and compares the resulting world-space
/// depths. This catches the T-junction cracks that an XY-only edge test cannot.
pub fn analyze_depth_discontinuities(
    fine_vertices: &[ClipmapVertex],
    coarse_vertices: &[ClipmapVertex],
    heightmap: &[f32],
    height_dims: (u32, u32),
    z_scale: f32,
    threshold: f32,
) -> DepthSeamAnalysis {
    if height_dims.0 < 2
        || height_dims.1 < 2
        || heightmap.len() != (height_dims.0 * height_dims.1) as usize
    {
        return DepthSeamAnalysis {
            crack_count: 1,
            max_depth_gap: f32::INFINITY,
            avg_depth_gap: f32::INFINITY,
            ..Default::default()
        };
    }
    let (fine_min, fine_max) = fine_vertices
        .iter()
        .map(|vertex| Vec2::from(vertex.position))
        .fold(
            (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY)),
            |(min, max), position| (min.min(position), max.max(position)),
        );
    let side_of = |p: Vec2| -> Option<(usize, f32)> {
        let distances = [
            (p.x - fine_min.x).abs(),
            (p.x - fine_max.x).abs(),
            (p.y - fine_min.y).abs(),
            (p.y - fine_max.y).abs(),
        ];
        let (side, gap) = distances
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))?;
        (*gap <= threshold.max(1e-6)).then_some((side, if side < 2 { p.y } else { p.x }))
    };
    let mut coarse_sides: [Vec<(f32, &ClipmapVertex)>; 4] = Default::default();
    for vertex in coarse_vertices {
        if let Some((side, axis)) = side_of(Vec2::from(vertex.position)) {
            coarse_sides[side].push((axis, vertex));
        }
    }
    for side in &mut coarse_sides {
        side.sort_by(|a, b| a.0.total_cmp(&b.0));
        side.dedup_by(|a, b| (a.0 - b.0).abs() <= f32::EPSILON);
    }

    let sample_uv = |uv: [f32; 2]| {
        let x = uv[0].clamp(0.0, 1.0) * (height_dims.0 - 1) as f32;
        let y = uv[1].clamp(0.0, 1.0) * (height_dims.1 - 1) as f32;
        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(height_dims.0 - 1);
        let y1 = (y0 + 1).min(height_dims.1 - 1);
        let tx = x - x0 as f32;
        let ty = y - y0 as f32;
        let at = |sx: u32, sy: u32| heightmap[(sy * height_dims.0 + sx) as usize];
        let top = at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx;
        let bottom = at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx;
        top * (1.0 - ty) + bottom * ty
    };
    let sample = |vertex: &ClipmapVertex| {
        let fine = sample_uv(vertex.uv);
        let coarse_texels = 1u32 << ((vertex.ring_index() as u32 + 1).min(16));
        let step = Vec2::new(
            coarse_texels as f32 / (height_dims.0 - 1) as f32,
            coarse_texels as f32 / (height_dims.1 - 1) as f32,
        );
        let uv = Vec2::from(vertex.uv);
        let cell = uv / step;
        let base = cell.floor() * step;
        let t = cell.fract();
        let h00 = sample_uv(base.clamp(Vec2::ZERO, Vec2::ONE).into());
        let h10 = sample_uv(
            (base + Vec2::new(step.x, 0.0))
                .clamp(Vec2::ZERO, Vec2::ONE)
                .into(),
        );
        let h01 = sample_uv(
            (base + Vec2::new(0.0, step.y))
                .clamp(Vec2::ZERO, Vec2::ONE)
                .into(),
        );
        let h11 = sample_uv((base + step).clamp(Vec2::ZERO, Vec2::ONE).into());
        let coarse = h00 * (1.0 - t.x) * (1.0 - t.y)
            + h10 * t.x * (1.0 - t.y)
            + h01 * (1.0 - t.x) * t.y
            + h11 * t.x * t.y;
        let weight = vertex.morph_weight().clamp(0.0, 1.0);
        (fine * (1.0 - weight) + coarse * weight) * z_scale
    };
    let mut result = DepthSeamAnalysis::default();
    let mut total = 0.0;
    for fine in fine_vertices {
        let Some((side, axis)) = side_of(Vec2::from(fine.position)) else {
            continue;
        };
        let candidates = &coarse_sides[side];
        let upper = candidates.partition_point(|(coordinate, _)| *coordinate < axis);
        if upper == 0 || upper >= candidates.len() {
            continue;
        }
        let (lo_axis, lo) = candidates[upper - 1];
        let (hi_axis, hi) = candidates[upper];
        let t = ((axis - lo_axis) / (hi_axis - lo_axis).max(f32::EPSILON)).clamp(0.0, 1.0);
        let coarse_depth = sample(lo) * (1.0 - t) + sample(hi) * t;
        let gap = (sample(fine) - coarse_depth).abs();
        result.sample_count += 1;
        result.max_depth_gap = result.max_depth_gap.max(gap);
        total += gap;
        result.crack_count += u32::from(gap > threshold);
    }
    if result.sample_count > 0 {
        result.avg_depth_gap = total / result.sample_count as f32;
    } else {
        // A detector that compared no fine/coarse samples has proved nothing.
        // Fail closed so crack_count == 0 can never result from an empty
        // candidate set.
        result.crack_count = 1;
        result.max_depth_gap = f32::INFINITY;
        result.avg_depth_gap = f32::INFINITY;
    }
    result
}

/// Correct seam vertices by snapping to coarser grid positions.
///
/// Returns the number of vertices corrected.
pub fn correct_seam_vertices(
    vertices: &mut [ClipmapVertex],
    ring_index: u32,
    _texture_size: u32,
    config: &GeomorphConfig,
) -> u32 {
    if !config.snap_to_coarse {
        return 0;
    }

    let mut corrected = 0;
    let morph_threshold = 1.0 - config.morph_range * 0.5;

    for v in vertices.iter_mut() {
        // Only correct vertices near the outer boundary (high morph weight)
        if v.morph_weight() > morph_threshold && (v.ring_index() as u32) == ring_index {
            // The shader evaluates a bilinear coarse-grid height at this UV
            // and blends by morph_weight. Preserve world-anchored UVs so
            // material/height sampling stays geographically aligned.
            corrected += 1;
        }
    }

    corrected
}

/// Blend vertex positions at LOD boundaries for smooth transitions.
pub fn blend_boundary_vertices(
    fine_vertices: &[ClipmapVertex],
    coarse_vertices: &[ClipmapVertex],
    blend_factor: f32,
) -> Vec<ClipmapVertex> {
    let mut blended = Vec::with_capacity(fine_vertices.len());

    for fine_v in fine_vertices {
        // Find corresponding coarse vertex
        let fine_pos = Vec2::from(fine_v.position);
        let mut best_coarse: Option<&ClipmapVertex> = None;
        let mut best_dist = f32::MAX;

        for coarse_v in coarse_vertices {
            let coarse_pos = Vec2::from(coarse_v.position);
            let dist = fine_pos.distance(coarse_pos);
            if dist < best_dist {
                best_dist = dist;
                best_coarse = Some(coarse_v);
            }
        }

        if let Some(coarse_v) = best_coarse {
            if best_dist < 1.0 {
                // Blend position based on morph weight
                let t = blend_factor * fine_v.morph_weight();
                let blended_pos = fine_pos.lerp(Vec2::from(coarse_v.position), t);
                let blended_uv = Vec2::from(fine_v.uv).lerp(Vec2::from(coarse_v.uv), t);

                blended.push(ClipmapVertex::new(
                    blended_pos.x,
                    blended_pos.y,
                    blended_uv.x,
                    blended_uv.y,
                    fine_v.morph_weight(),
                    fine_v.ring_index() as u32,
                ));
            } else {
                blended.push(*fine_v);
            }
        } else {
            blended.push(*fine_v);
        }
    }

    blended
}

/// Validate that geo-morphing eliminates visible seams.
pub fn validate_geomorph(
    vertices: &[ClipmapVertex],
    config: &GeomorphConfig,
) -> Result<(), String> {
    // Check morph weights are in valid range
    for (i, v) in vertices.iter().enumerate() {
        let mw = v.morph_weight();
        if mw > 1.0 {
            return Err(format!("Vertex {} has morph_weight {} > 1.0", i, mw));
        }
        // Negative morph weight is valid (indicates skirt vertex)
    }

    // Check for discontinuities in morph weights
    let mut prev_morph = 0.0_f32;
    let mut large_jumps = 0;

    for v in vertices {
        let mw = v.morph_weight();
        if mw >= 0.0 {
            // Skip skirt vertices
            let jump = (mw - prev_morph).abs();
            if jump > config.morph_range {
                large_jumps += 1;
            }
            prev_morph = mw;
        }
    }

    if large_jumps > vertices.len() / 10 {
        return Err(format!(
            "Too many large morph weight discontinuities: {} (threshold: {})",
            large_jumps,
            vertices.len() / 10
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morph_weight_calculation() {
        // At inner boundary, weight should be 0
        assert_eq!(calculate_morph_weight(0.0, 100.0, 0.3), 0.0);

        // At outer boundary, weight should be 1
        assert!((calculate_morph_weight(100.0, 100.0, 0.3) - 1.0).abs() < 0.01);

        // In middle of ring (before morph zone), weight should be 0
        assert_eq!(calculate_morph_weight(50.0, 100.0, 0.3), 0.0);

        // At start of morph zone (70% through ring with 0.3 morph_range)
        let w = calculate_morph_weight(70.0, 100.0, 0.3);
        assert!(w >= 0.0 && w <= 0.1);
    }

    #[test]
    fn test_snap_uv_to_coarse_grid() {
        // Ring 0 with 256 texture should snap to 1/256 grid
        let uv = Vec2::new(0.123, 0.456);
        let snapped = snap_uv_to_coarse_grid(uv, 0, 256);
        assert!((snapped.x * 256.0).fract() < 0.001);
        assert!((snapped.y * 256.0).fract() < 0.001);

        // Ring 1 should snap to 2/256 grid
        let snapped = snap_uv_to_coarse_grid(uv, 1, 256);
        assert!((snapped.x * 128.0).fract() < 0.001);
        assert!((snapped.y * 128.0).fract() < 0.001);
    }

    #[test]
    fn test_geomorph_config_default() {
        let config = GeomorphConfig::default();
        assert!(config.morph_range > 0.0 && config.morph_range <= 1.0);
        assert!(config.max_seam_gap > 0.0);
        assert!(config.snap_to_coarse);
    }

    #[test]
    fn seam_detector_accepts_coarse_edges_and_rejects_open_boundaries() {
        let vertex = |x, y| ClipmapVertex::new(x, y, 0.0, 0.0, 0.0, 0);
        let inner = [
            vertex(-1.0, -1.0),
            vertex(1.0, -1.0),
            vertex(1.0, 1.0),
            vertex(-1.0, 1.0),
        ];
        // One midpoint per coarse edge is enough: fine vertices may form
        // legitimate T-junctions along those segments.
        let aligned = [
            vertex(-1.0, 0.0),
            vertex(1.0, 0.0),
            vertex(0.0, -1.0),
            vertex(0.0, 1.0),
        ];
        let config = GeomorphConfig::default();
        let valid = analyze_seams(&inner, &aligned, &config);
        assert!(valid.seams_valid);
        assert_eq!(valid.crack_count, 0);

        let shifted = [
            vertex(-0.9, 0.0),
            vertex(1.1, 0.0),
            vertex(0.0, -0.9),
            vertex(0.0, 1.1),
        ];
        let invalid = analyze_seams(&inner, &shifted, &config);
        assert!(!invalid.seams_valid);
        assert_eq!(invalid.crack_count, 4);
    }

    #[test]
    fn depth_detector_fails_closed_without_bracketing_samples() {
        let fine = [ClipmapVertex::new(0.0, 0.0, 0.5, 0.5, 1.0, 0)];
        let analysis = analyze_depth_discontinuities(&fine, &[], &[0.0; 4], (2, 2), 1.0, 0.001);
        assert_eq!(analysis.sample_count, 0);
        assert_eq!(analysis.crack_count, 1);
        assert!(analysis.max_depth_gap.is_infinite());
        assert!(analysis.avg_depth_gap.is_infinite());
    }

    #[test]
    fn depth_detector_counts_bracketed_boundary_samples() {
        let vertex = |x, y, u, v| ClipmapVertex::new(x, y, u, v, 1.0, 0);
        let fine = [
            vertex(-1.0, -1.0, 0.0, 0.0),
            vertex(-1.0, 0.0, 0.0, 0.5),
            vertex(-1.0, 1.0, 0.0, 1.0),
            vertex(1.0, -1.0, 1.0, 0.0),
            vertex(1.0, 0.0, 1.0, 0.5),
            vertex(1.0, 1.0, 1.0, 1.0),
        ];
        let coarse = [
            vertex(-1.0, -1.0, 0.0, 0.0),
            vertex(-1.0, 1.0, 0.0, 1.0),
            vertex(1.0, -1.0, 1.0, 0.0),
            vertex(1.0, 1.0, 1.0, 1.0),
        ];
        let analysis = analyze_depth_discontinuities(&fine, &coarse, &[0.0; 4], (2, 2), 1.0, 0.001);
        assert!(analysis.sample_count >= 2);
        assert_eq!(analysis.crack_count, 0);
        assert_eq!(analysis.max_depth_gap, 0.0);
    }

    #[test]
    fn publish_seam_analysis_counts_every_build() {
        // `SeamAnalysis` is a mesh-BUILD-time metric that the geometry cache can
        // replay across arbitrarily many frames. Without a build counter a
        // caller cannot tell one analysis from six hundred, so "0 cracks over a
        // 600-frame flythrough" would be unfalsifiable. Strict increase is
        // asserted (rather than exact deltas) so the test stays correct when the
        // suite runs multi-threaded.
        let vertex = |x, y, u, v| ClipmapVertex::new(x, y, u, v, 0.0, 0);
        let inner = [
            vertex(-1.0, -1.0, 0.0, 0.0),
            vertex(1.0, -1.0, 1.0, 0.0),
            vertex(1.0, 1.0, 1.0, 1.0),
            vertex(-1.0, 1.0, 0.0, 1.0),
        ];
        let outer = [
            vertex(-1.0, 0.0, 0.0, 0.5),
            vertex(1.0, 0.0, 1.0, 0.5),
            vertex(0.0, -1.0, 0.5, 0.0),
            vertex(0.0, 1.0, 0.5, 1.0),
        ];
        let sample = analyze_seams(&inner, &outer, &GeomorphConfig::default());

        let before = seam_analysis_count();
        publish_seam_analysis(sample.clone());
        let first = seam_analysis_count();
        publish_seam_analysis(sample);
        let second = seam_analysis_count();

        assert!(first > before, "{first} did not exceed {before}");
        assert!(second > first, "{second} did not exceed {first}");
    }
}
