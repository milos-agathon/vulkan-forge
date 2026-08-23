// src/path_tracing/hybrid_compute/terrain_heightfield.rs
// PROMETHEUS: 2.5D min-max acceleration structure over a DEM heightfield for
// the hybrid path tracer. Each texel of mip L stores (min_height, max_height)
// over its 2x2 children in mip L-1; mip 0 stores the min/max of the four
// corner samples of each bilinear DEM cell. Built on the CPU from the same
// heightfield that feeds the rasterizer (the in-tree HZB producer in
// hzb_build.wgsl is shader-only and not integrated, so `from_heightfield` is
// the supported constructor), uploaded once as an RG32Float mip chain.
// RELEVANT FILES: src/shaders/hybrid_terrain_traversal.wgsl, src/path_tracing/hybrid_compute/render.rs

use crate::core::error::RenderError;
use crate::core::resource_tracker::{tracked_create_texture, TrackedTexture};
use bytemuck::{Pod, Zeroable};
use wgpu::{Device, Queue, TextureFormat};

/// World -> texel transform and traversal constants consumed by
/// hybrid_terrain_traversal.wgsl. Deliberately packed as six vec4 rows
/// (96 bytes) so the WGSL uniform layout is alignment-trivial:
///   row 0 origin_spacing: origin_x, origin_z (world xz of DEM texel (0,0)),
///                         spacing_x, spacing_z (world units per texel)
///   row 1 h_params:       h_min, h_max (raw DEM range), exaggeration
///                         (world y = height * exaggeration), env intensity
///   row 2 albedo_pad:     terrain albedo rgb, unused
///   row 3 dims:           width_texels, height_texels, cell_w, cell_h
///   row 4 mips:           mip_count, flags (bit0 = terrain enabled),
///                         env_width, env_height (0 = constant env fallback)
///   row 5 extra:          spp (camera samples per frame), Welford window
///                         (frames per convergence window), unused, unused
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TerrainPtUniforms {
    pub origin_spacing: [f32; 4],
    pub h_params: [f32; 4],
    pub albedo_pad: [f32; 4],
    pub dims: [u32; 4],
    pub mips: [u32; 4],
    pub extra: [u32; 4],
}

/// Curvature parameters consumed by the shared terrain traversal. The two
/// explicit pads match WGSL uniform layout (vec2 alignment = 8 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct EarthCurvatureUniforms {
    pub inv_two_r_prime: f32,
    pub _pad0: f32,
    pub ray_origin_geodetic: [f32; 2],
    pub enabled: u32,
    pub _pad1: u32,
}

impl EarthCurvatureUniforms {
    pub fn new(
        earth: crate::geo::refraction::EarthModel,
        refraction: crate::geo::refraction::RefractionModel,
        ray_origin_geodetic: [f64; 2],
        azimuth_deg: f64,
    ) -> Result<Self, RenderError> {
        if !ray_origin_geodetic[0].is_finite()
            || !(-90.0..=90.0).contains(&ray_origin_geodetic[0])
            || !ray_origin_geodetic[1].is_finite()
            || !(-180.0..=180.0).contains(&ray_origin_geodetic[1])
        {
            return Err(RenderError::render(
                "ray-origin latitude/longitude must be finite and in [-90,90]/[-180,180]",
            ));
        }
        let effective_radius =
            crate::geo::refraction::effective_radius_m(earth, refraction, azimuth_deg)
                .map_err(RenderError::render)?;
        let enabled = effective_radius.is_finite();
        Ok(Self {
            inv_two_r_prime: if enabled {
                (0.5 / effective_radius) as f32
            } else {
                0.0
            },
            _pad0: 0.0,
            ray_origin_geodetic: [ray_origin_geodetic[0] as f32, ray_origin_geodetic[1] as f32],
            enabled: u32::from(enabled),
            _pad1: 0,
        })
    }
}

/// Exact range of y(t) + d(t)^2/(2R') over a node ray span.
#[cfg(test)]
fn curved_ray_height_range(
    origin_y: f32,
    direction_y: f32,
    horizontal_direction_sq: f32,
    t0: f32,
    t1: f32,
    inv_two_r_prime: f32,
) -> (f32, f32) {
    let a = horizontal_direction_sq * inv_two_r_prime;
    let height = |t: f32| origin_y + direction_y * t + a * t * t;
    let y0 = height(t0);
    let y1 = height(t1);
    let mut minimum = y0.min(y1);
    let maximum = y0.max(y1);
    if a > 0.0 {
        let vertex = -direction_y / (2.0 * a);
        if (t0..=t1).contains(&vertex) {
            minimum = minimum.min(height(vertex));
        }
    }
    (minimum, maximum)
}

/// CPU-side min-max mip chain (kept for unit tests and re-upload).
pub struct MinMaxMips {
    /// levels[0] is the finest (per-cell) level; each entry is [min, max].
    /// Levels are padded to power-of-two dims with (+inf, -inf) sentinel
    /// texels so the wgpu floor-division mip chain and the shader's pure
    /// shift-based node->cell math agree exactly; sentinel nodes always fail
    /// the traversal band test.
    pub levels: Vec<Vec<[f32; 2]>>,
    /// Padded (width, height) per level, same order as `levels`.
    pub dims: Vec<(u32, u32)>,
    /// Logical (unpadded) cell counts of level 0.
    pub cell_w: u32,
    pub cell_h: u32,
}

/// Build the min-max cell pyramid on the CPU.
///
/// Level 0 covers the (w-1) x (h-1) bilinear DEM cells (cell (x, y) stores
/// the min/max of its four corner samples, which bounds the bilinear surface
/// over the cell), padded to power-of-two dims. Every coarser mip reduces
/// 2x2 children, so parents always cover all children.
pub fn build_minmax_mips(heights: &[f32], w: u32, h: u32) -> Result<MinMaxMips, RenderError> {
    if w < 2 || h < 2 {
        return Err(RenderError::Upload(format!(
            "terrain heightfield must be at least 2x2 texels, got {w}x{h}"
        )));
    }
    if heights.len() != (w as usize) * (h as usize) {
        return Err(RenderError::Upload(format!(
            "heightfield length {} does not match {w}x{h}",
            heights.len()
        )));
    }
    if heights.iter().any(|v| !v.is_finite()) {
        return Err(RenderError::Upload(
            "terrain heightfield contains non-finite samples".into(),
        ));
    }

    let cw = w - 1;
    let ch = h - 1;
    let pw = cw.next_power_of_two();
    let ph = ch.next_power_of_two();
    const EMPTY: [f32; 2] = [f32::INFINITY, f32::NEG_INFINITY];
    let mut level0 = vec![EMPTY; (pw as usize) * (ph as usize)];
    for y in 0..ch as usize {
        for x in 0..cw as usize {
            let i00 = y * w as usize + x;
            let i10 = i00 + 1;
            let i01 = i00 + w as usize;
            let i11 = i01 + 1;
            let (a, b, c, d) = (heights[i00], heights[i10], heights[i01], heights[i11]);
            level0[y * pw as usize + x] = [a.min(b).min(c).min(d), a.max(b).max(c).max(d)];
        }
    }

    let mut levels = vec![level0];
    let mut dims = vec![(pw, ph)];
    while dims.last().unwrap().0 > 1 || dims.last().unwrap().1 > 1 {
        let (lw, lh) = *dims.last().unwrap();
        let (nw, nh) = ((lw / 2).max(1), (lh / 2).max(1));
        let prev = levels.last().unwrap();
        let mut next = vec![EMPTY; (nw as usize) * (nh as usize)];
        for y in 0..nh as usize {
            for x in 0..nw as usize {
                let mut mn = f32::INFINITY;
                let mut mx = f32::NEG_INFINITY;
                for dy in 0..2usize {
                    for dx in 0..2usize {
                        // Non-square pot dims collapse one axis early; clamp
                        // keeps full coverage in that case.
                        let sx = (2 * x + dx).min(lw as usize - 1);
                        let sy = (2 * y + dy).min(lh as usize - 1);
                        let v = prev[sy * lw as usize + sx];
                        mn = mn.min(v[0]);
                        mx = mx.max(v[1]);
                    }
                }
                next[y * nw as usize + x] = [mn, mx];
            }
        }
        levels.push(next);
        dims.push((nw, nh));
    }

    Ok(MinMaxMips {
        levels,
        dims,
        cell_w: cw,
        cell_h: ch,
    })
}

/// GPU min-max pyramid + the DEM height texture the leaf test samples.
pub struct TerrainMinMaxPyramid {
    pub height_texture: TrackedTexture,
    pub minmax_texture: TrackedTexture,
    pub mip_count: u32,
    pub cell_w: u32,
    pub cell_h: u32,
    pub h_min: f32,
    pub h_max: f32,
    pub byte_size: u64,
    width: u32,
    height: u32,
    cpu_mips: MinMaxMips,
}

impl TerrainMinMaxPyramid {
    /// Upload the DEM (R32Float, 1 mip) and its min-max pyramid (RG32Float,
    /// full chain) built from the same heightfield. Both allocations are
    /// created through `tracked_create_texture`, so the global memory tracker
    /// and allocation ledger record and release them automatically.
    pub fn from_heightfield(
        device: &Device,
        queue: &Queue,
        heights: &[f32],
        w: u32,
        h: u32,
    ) -> Result<Self, RenderError> {
        let mips = build_minmax_mips(heights, w, h)?;
        let (pot_w, pot_h) = mips.dims[0];
        let (cell_w, cell_h) = (mips.cell_w, mips.cell_h);
        let mip_count = mips.levels.len() as u32;
        let h_min = heights.iter().copied().fold(f32::INFINITY, f32::min);
        let h_max = heights.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let height_texture = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("hybrid-pt-terrain-height"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )?;
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &height_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(heights),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        let minmax_texture = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("hybrid-pt-terrain-minmax"),
                size: wgpu::Extent3d {
                    width: pot_w,
                    height: pot_h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: mip_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TextureFormat::Rg32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )?;
        let mut byte_size = (w as u64) * (h as u64) * 4;
        for (level, ((lw, lh), data)) in mips.dims.iter().zip(mips.levels.iter()).enumerate() {
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &minmax_texture,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(data),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(lw * 8),
                    rows_per_image: Some(*lh),
                },
                wgpu::Extent3d {
                    width: *lw,
                    height: *lh,
                    depth_or_array_layers: 1,
                },
            );
            byte_size += (*lw as u64) * (*lh as u64) * 8;
        }
        log::info!(
            "hybrid-pt-terrain-minmax: {}x{} DEM -> {} mips, {:.2} MiB total",
            w,
            h,
            mip_count,
            byte_size as f64 / (1024.0 * 1024.0)
        );

        Ok(Self {
            height_texture,
            minmax_texture,
            mip_count,
            cell_w,
            cell_h,
            h_min,
            h_max,
            byte_size,
            width: w,
            height: h,
            cpu_mips: mips,
        })
    }

    /// CPU mirror used to derive conservative tile AABBs for raster HZB
    /// culling. This is the same PROMETHEUS pyramid uploaded above, not a
    /// separately constructed terrain min-max hierarchy.
    pub(crate) fn cpu_mips(&self) -> &MinMaxMips {
        &self.cpu_mips
    }

    /// Uniform block for the traversal kernel; terrain is centered on the
    /// world origin: texel (0,0) sits at (-(w-1)/2*sx, -(h-1)/2*sz).
    #[allow(clippy::too_many_arguments)]
    pub fn uniforms(
        &self,
        spacing_x: f32,
        spacing_z: f32,
        exaggeration: f32,
        albedo: [f32; 3],
        env_intensity: f32,
        env_dims: (u32, u32),
        spp: u32,
        welford_window: u32,
    ) -> TerrainPtUniforms {
        let origin_x = -0.5 * (self.width as f32 - 1.0) * spacing_x;
        let origin_z = -0.5 * (self.height as f32 - 1.0) * spacing_z;
        TerrainPtUniforms {
            origin_spacing: [origin_x, origin_z, spacing_x, spacing_z],
            h_params: [self.h_min, self.h_max, exaggeration, env_intensity],
            albedo_pad: [albedo[0], albedo[1], albedo[2], 0.0],
            dims: [self.width, self.height, self.cell_w, self.cell_h],
            mips: [self.mip_count, 1, env_dims.0, env_dims.1],
            extra: [spp.max(1), welford_window.max(2), 0, 0],
        }
    }
}

/// Complete GPU terrain scene for the hybrid tracer: the min-max pyramid plus
/// the environment map and the shading constants the kernel needs. This is
/// the seam `HybridPathTracer::render` accepts to make terrain a first-class
/// primitive alongside mesh/SDF geometry.
pub struct TerrainPtScene {
    pub pyramid: TerrainMinMaxPyramid,
    pub env_texture: TrackedTexture,
    /// (0, 0) selects the constant-white env fallback in the kernel.
    pub env_dims: (u32, u32),
    spacing: (f32, f32),
    exaggeration: f32,
    albedo: [f32; 3],
    env_intensity: f32,
    env_tracked: (u32, u32),
}

impl TerrainPtScene {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        queue: &Queue,
        heights: &[f32],
        dem_width: u32,
        dem_height: u32,
        spacing: (f32, f32),
        exaggeration: f32,
        albedo: [f32; 3],
        env_map: Option<(&[f32], u32, u32)>,
        env_intensity: f32,
    ) -> Result<Self, RenderError> {
        if !(spacing.0.is_finite() && spacing.0 > 0.0 && spacing.1.is_finite() && spacing.1 > 0.0) {
            return Err(RenderError::Upload(format!(
                "terrain spacing must be finite and > 0, got {spacing:?}"
            )));
        }
        if !(exaggeration.is_finite() && exaggeration > 0.0) {
            return Err(RenderError::Upload(
                "terrain exaggeration must be finite and > 0".into(),
            ));
        }
        if albedo.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(RenderError::Upload(
                "terrain albedo must be finite and >= 0".into(),
            ));
        }
        if !(env_intensity.is_finite() && env_intensity >= 0.0) {
            return Err(RenderError::Upload(
                "env intensity must be finite and >= 0".into(),
            ));
        }
        let pyramid =
            TerrainMinMaxPyramid::from_heightfield(device, queue, heights, dem_width, dem_height)?;

        let (env_data, env_w, env_h, env_dims): (Vec<f32>, u32, u32, (u32, u32)) = match env_map {
            Some((data, w, h)) => {
                if w == 0 || h == 0 || data.len() != (w as usize) * (h as usize) * 3 {
                    return Err(RenderError::Upload(
                        "env map dims do not match data length".into(),
                    ));
                }
                if data.iter().any(|v| !v.is_finite()) {
                    return Err(RenderError::Upload(
                        "env map contains non-finite samples".into(),
                    ));
                }
                (data.to_vec(), w, h, (w, h))
            }
            // 1x1 white placeholder; env_dims (0,0) routes the kernel through
            // the constant fallback so both configurations share one code path.
            None => (vec![1.0, 1.0, 1.0], 1, 1, (0, 0)),
        };
        let env_rgba: Vec<f32> = env_data
            .chunks_exact(3)
            .flat_map(|c| [c[0], c[1], c[2], 1.0])
            .collect();
        let env_texture = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("hybrid-pt-terrain-env"),
                size: wgpu::Extent3d {
                    width: env_w,
                    height: env_h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )?;
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &env_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&env_rgba),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(env_w * 16),
                rows_per_image: Some(env_h),
            },
            wgpu::Extent3d {
                width: env_w,
                height: env_h,
                depth_or_array_layers: 1,
            },
        );

        Ok(Self {
            pyramid,
            env_texture,
            env_dims,
            spacing,
            exaggeration,
            albedo,
            env_intensity,
            env_tracked: (env_w, env_h),
        })
    }

    /// Total tracked GPU bytes (pyramid mips + DEM texture + env map).
    pub fn byte_size(&self) -> u64 {
        let (ew, eh) = self.env_tracked;
        self.pyramid.byte_size + (ew as u64) * (eh as u64) * 16
    }

    pub fn uniforms(&self, spp: u32, welford_window: u32) -> TerrainPtUniforms {
        self.pyramid.uniforms(
            self.spacing.0,
            self.spacing.1,
            self.exaggeration,
            self.albedo,
            self.env_intensity,
            self.env_dims,
            spp,
            welford_window,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gpu::try_ctx;
    use crate::core::resource_tracker::{tracked_create_buffer, tracked_create_buffer_init};

    fn ramp(w: u32, h: u32) -> Vec<f32> {
        (0..w * h)
            .map(|i| (i % w) as f32 * 0.5 + (i / w) as f32 * 0.25)
            .collect()
    }

    #[test]
    fn minmax_invariant_per_node() {
        let mips = build_minmax_mips(&ramp(256, 256), 256, 256).unwrap();
        for level in &mips.levels {
            for v in level {
                // Sentinel padding is (+inf, -inf); real nodes are ordered.
                if v[0].is_finite() || v[1].is_finite() {
                    assert!(v[0] <= v[1], "min must be <= max");
                }
            }
        }
    }

    #[test]
    fn mip_count_and_dims() {
        let mips = build_minmax_mips(&ramp(256, 256), 256, 256).unwrap();
        assert_eq!((mips.cell_w, mips.cell_h), (255, 255));
        assert_eq!(mips.dims[0], (256, 256)); // padded to power of two
        assert_eq!(*mips.dims.last().unwrap(), (1, 1));
        assert_eq!(mips.levels.len(), 9); // 256 -> 128 -> ... -> 1
                                          // Odd, non-square input pads each axis independently.
        let mips = build_minmax_mips(&ramp(100, 37), 100, 37).unwrap();
        assert_eq!((mips.cell_w, mips.cell_h), (99, 36));
        assert_eq!(mips.dims[0], (128, 64));
        assert_eq!(*mips.dims.last().unwrap(), (1, 1));
        assert_eq!(mips.levels.len(), 8); // max(128, 64) -> 8 levels
    }

    #[test]
    fn parent_covers_children() {
        let mips = build_minmax_mips(&ramp(64, 64), 64, 64).unwrap();
        for l in 1..mips.levels.len() {
            let (pw, ph) = mips.dims[l];
            let (cw, ch) = mips.dims[l - 1];
            for y in 0..ph as usize {
                for x in 0..pw as usize {
                    let p = mips.levels[l][y * pw as usize + x];
                    for dy in 0..2usize {
                        for dx in 0..2usize {
                            let sx = (2 * x + dx).min(cw as usize - 1);
                            let sy = (2 * y + dy).min(ch as usize - 1);
                            let c = mips.levels[l - 1][sy * cw as usize + sx];
                            assert!(p[0] <= c[0] && p[1] >= c[1], "parent must cover child");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn root_covers_full_range() {
        let heights = ramp(33, 17);
        let mips = build_minmax_mips(&heights, 33, 17).unwrap();
        let root = mips.levels.last().unwrap()[0];
        let mn = heights.iter().copied().fold(f32::INFINITY, f32::min);
        let mx = heights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(root[0], mn);
        assert_eq!(root[1], mx);
    }

    #[test]
    fn flat_dem_is_valid() {
        let mips = build_minmax_mips(&vec![5.0; 16 * 16], 16, 16).unwrap();
        for level in &mips.levels {
            for v in level {
                // Real nodes are exactly flat; padding sentinels are
                // (+inf, -inf) and are skipped by the traversal band test.
                if v[0].is_finite() {
                    assert_eq!(v[0], 5.0);
                    assert_eq!(v[1], 5.0);
                }
            }
        }
        // The root must be real (it covers the whole DEM).
        let root = mips.levels.last().unwrap()[0];
        assert_eq!(root, [5.0, 5.0]);
    }

    #[test]
    fn degenerate_dems_error() {
        assert!(build_minmax_mips(&[1.0], 1, 1).is_err());
        assert!(build_minmax_mips(&[f32::NAN; 4], 2, 2).is_err());
        assert!(build_minmax_mips(&[1.0; 5], 2, 2).is_err());
    }

    const PROOF_DEM_SIDE: usize = 256;
    const PROOF_CELL_COUNT: usize = PROOF_DEM_SIDE - 1;
    const PROOF_SPACING_M: f32 = 500.0;

    fn curvature_fixture() -> Vec<f32> {
        (0..PROOF_DEM_SIDE * PROOF_DEM_SIDE)
            .map(|i| {
                let x = (i % PROOF_DEM_SIDE) as f32;
                let y = (i / PROOF_DEM_SIDE) as f32;
                900.0
                    + 180.0 * (x * 0.071).sin()
                    + 120.0 * (y * 0.047).cos()
                    + 650.0 * (-((x - 150.0).powi(2) + (y - 126.0).powi(2)) / 900.0).exp()
            })
            .collect()
    }

    #[derive(Clone, Copy)]
    struct ProofRay {
        origin: [f32; 3],
        direction: [f32; 3],
        inv_two_r_prime: f32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct ProofGpuRay {
        origin_tmin: [f32; 4],
        direction_tmax: [f32; 4],
    }

    impl From<ProofRay> for ProofGpuRay {
        fn from(ray: ProofRay) -> Self {
            Self {
                origin_tmin: [ray.origin[0], ray.origin[1], ray.origin[2], 1e-3],
                direction_tmax: [
                    ray.direction[0],
                    ray.direction[1],
                    ray.direction[2],
                    200_000.0,
                ],
            }
        }
    }

    fn proof_height(heights: &[f32], cell_x: usize, cell_z: usize, x: f64, z: f64) -> f64 {
        let u = x / f64::from(PROOF_SPACING_M) - cell_x as f64;
        let v = z / f64::from(PROOF_SPACING_M) - cell_z as f64;
        let at = |x: usize, z: usize| f64::from(heights[z * PROOF_DEM_SIDE + x]);
        let h00 = at(cell_x, cell_z);
        let h10 = at(cell_x + 1, cell_z);
        let h01 = at(cell_x, cell_z + 1);
        let h11 = at(cell_x + 1, cell_z + 1);
        (h00 * (1.0 - u) + h10 * u) * (1.0 - v) + (h01 * (1.0 - u) + h11 * u) * v
    }

    fn root_in_span(a: f64, b: f64, c: f64, t0: f64, t1: f64) -> bool {
        if a.abs() < 1e-15 {
            return b.abs() >= 1e-15 && (-c / b) >= t0 && (-c / b) <= t1;
        }
        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return false;
        }
        let square_root = discriminant.sqrt();
        let q = -0.5 * (b + square_root.copysign(b));
        let first = q / a;
        let second = if q.abs() < 1e-30 {
            f64::INFINITY
        } else {
            c / q
        };
        (first >= t0 && first <= t1) || (second >= t0 && second <= t1)
    }

    // Independent f64 brute-force oracle: expand the bilinear patch and the
    // curved ray analytically, then solve their quadratic intersection.
    fn brute_cell_hit(
        heights: &[f32],
        ray: ProofRay,
        cell_x: usize,
        cell_z: usize,
        t0: f64,
        t1: f64,
    ) -> bool {
        let at = |x: usize, z: usize| f64::from(heights[z * PROOF_DEM_SIDE + x]);
        let h00 = at(cell_x, cell_z);
        let hx = at(cell_x + 1, cell_z) - h00;
        let hz = at(cell_x, cell_z + 1) - h00;
        let hxz = at(cell_x + 1, cell_z + 1) - h00 - hx - hz;
        let u0 = f64::from(ray.origin[0]) / f64::from(PROOF_SPACING_M) - cell_x as f64;
        let v0 = f64::from(ray.origin[2]) / f64::from(PROOF_SPACING_M) - cell_z as f64;
        let du = f64::from(ray.direction[0]) / f64::from(PROOF_SPACING_M);
        let dv = f64::from(ray.direction[2]) / f64::from(PROOF_SPACING_M);
        let terrain_a = hxz * du * dv;
        let terrain_b = hx * du + hz * dv + hxz * (u0 * dv + v0 * du);
        let terrain_c = h00 + hx * u0 + hz * v0 + hxz * u0 * v0;
        let horizontal_sq =
            f64::from(ray.direction[0]).powi(2) + f64::from(ray.direction[2]).powi(2);
        root_in_span(
            horizontal_sq * f64::from(ray.inv_two_r_prime) - terrain_a,
            f64::from(ray.direction[1]) - terrain_b,
            f64::from(ray.origin[1]) - terrain_c,
            t0,
            t1,
        )
    }

    fn deviation_span_hit(d0: f32, dm: f32, d1: f32, any_hit: bool) -> bool {
        if any_hit && d0 <= 0.0 {
            return true;
        }
        let a = 2.0 * d1 + 2.0 * d0 - 4.0 * dm;
        let b = d1 - d0 - a;
        root_in_span(f64::from(a), f64::from(b), f64::from(d0), 0.0, 1.0)
    }

    // f32 mirror of terrain_leaf_intersect() in the production WGSL.
    fn descent_cell_hit(
        heights: &[f32],
        ray: ProofRay,
        cell_x: usize,
        cell_z: usize,
        t0: f32,
        t1: f32,
    ) -> bool {
        let deviation = |t: f32| {
            let x = ray.origin[0] + t * ray.direction[0];
            let z = ray.origin[2] + t * ray.direction[2];
            let terrain = proof_height(heights, cell_x, cell_z, f64::from(x), f64::from(z)) as f32;
            let horizontal_sq =
                ray.direction[0] * ray.direction[0] + ray.direction[2] * ray.direction[2];
            ray.origin[1] + t * ray.direction[1] + horizontal_sq * ray.inv_two_r_prime * t * t
                - terrain
        };
        let d0 = deviation(t0);
        let dm = deviation(0.5 * (t0 + t1));
        let d1 = deviation(t1);
        deviation_span_hit(d0, dm, d1, true)
    }

    fn slab_axis(origin: f64, direction: f64, low: f64, high: f64) -> Option<(f64, f64)> {
        if direction.abs() < 1e-12 {
            return (origin >= low && origin <= high).then_some((f64::NEG_INFINITY, f64::INFINITY));
        }
        let a = (low - origin) / direction;
        let b = (high - origin) / direction;
        Some((a.min(b), a.max(b)))
    }

    fn slab_xz(ray: ProofRay, x0: f64, x1: f64, z0: f64, z1: f64) -> Option<(f64, f64)> {
        let x = slab_axis(
            f64::from(ray.origin[0]),
            f64::from(ray.direction[0]),
            x0,
            x1,
        )?;
        let z = slab_axis(
            f64::from(ray.origin[2]),
            f64::from(ray.direction[2]),
            z0,
            z1,
        )?;
        (x.0.max(z.0) <= x.1.min(z.1)).then_some((x.0.max(z.0), x.1.min(z.1)))
    }

    fn brute_2d_hit(heights: &[f32], ray: ProofRay) -> bool {
        let extent = PROOF_CELL_COUNT as f64 * f64::from(PROOF_SPACING_M);
        let Some((enter, exit)) = slab_xz(ray, 0.0, extent, 0.0, extent) else {
            return false;
        };
        let mut t = enter.max(1e-3);
        let end = exit.min(200_000.0);
        while t <= end {
            let probe = (t + 1e-5).min(end);
            let x = f64::from(ray.origin[0]) + probe * f64::from(ray.direction[0]);
            let z = f64::from(ray.origin[2]) + probe * f64::from(ray.direction[2]);
            let cell_x = (x / f64::from(PROOF_SPACING_M)).floor().clamp(0.0, 254.0) as usize;
            let cell_z = (z / f64::from(PROOF_SPACING_M)).floor().clamp(0.0, 254.0) as usize;
            let next_x = if ray.direction[0] > 0.0 {
                ((cell_x + 1) as f64 * f64::from(PROOF_SPACING_M) - f64::from(ray.origin[0]))
                    / f64::from(ray.direction[0])
            } else if ray.direction[0] < 0.0 {
                (cell_x as f64 * f64::from(PROOF_SPACING_M) - f64::from(ray.origin[0]))
                    / f64::from(ray.direction[0])
            } else {
                f64::INFINITY
            };
            let next_z = if ray.direction[2] > 0.0 {
                ((cell_z + 1) as f64 * f64::from(PROOF_SPACING_M) - f64::from(ray.origin[2]))
                    / f64::from(ray.direction[2])
            } else if ray.direction[2] < 0.0 {
                (cell_z as f64 * f64::from(PROOF_SPACING_M) - f64::from(ray.origin[2]))
                    / f64::from(ray.direction[2])
            } else {
                f64::INFINITY
            };
            let next = next_x.min(next_z).min(end);
            if brute_cell_hit(heights, ray, cell_x, cell_z, t, next) {
                return true;
            }
            if next >= end {
                break;
            }
            t = next + 1e-7;
        }
        false
    }

    fn descent_2d_hit(heights: &[f32], mips: &MinMaxMips, ray: ProofRay) -> bool {
        let mut stack = vec![(mips.levels.len() - 1, 0u32, 0u32)];
        while let Some((level, nx, nz)) = stack.pop() {
            let cx0 = nx << level;
            let cz0 = nz << level;
            if cx0 >= PROOF_CELL_COUNT as u32 || cz0 >= PROOF_CELL_COUNT as u32 {
                continue;
            }
            let cx1 = ((nx + 1) << level).min(PROOF_CELL_COUNT as u32);
            let cz1 = ((nz + 1) << level).min(PROOF_CELL_COUNT as u32);
            let Some((enter, exit)) = slab_xz(
                ray,
                f64::from(cx0) * f64::from(PROOF_SPACING_M),
                f64::from(cx1) * f64::from(PROOF_SPACING_M),
                f64::from(cz0) * f64::from(PROOF_SPACING_M),
                f64::from(cz1) * f64::from(PROOF_SPACING_M),
            ) else {
                continue;
            };
            let t0 = enter.max(1e-3) as f32;
            let t1 = exit.min(200_000.0) as f32;
            if t0 > t1 {
                continue;
            }
            let (lw, _) = mips.dims[level];
            let mm = mips.levels[level][(nz * lw + nx) as usize];
            let (ray_min, ray_max) = curved_ray_height_range(
                ray.origin[1],
                ray.direction[1],
                ray.direction[0] * ray.direction[0] + ray.direction[2] * ray.direction[2],
                t0,
                t1,
                ray.inv_two_r_prime,
            );
            if ray_min > mm[1] || ray_max < mm[0] {
                continue;
            }
            if level == 0 {
                if descent_cell_hit(heights, ray, cx0 as usize, cz0 as usize, t0, t1) {
                    return true;
                }
            } else {
                for child_z in 0..2 {
                    for child_x in 0..2 {
                        stack.push((level - 1, nx * 2 + child_x, nz * 2 + child_z));
                    }
                }
            }
        }
        false
    }

    fn next_u32(state: &mut u32) -> u32 {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        *state
    }

    fn proof_ray(heights: &[f32], x: f32, z: f32, azimuth: f32, elevation: f32) -> ProofRay {
        let cell_x = x.floor() as usize;
        let cell_z = z.floor() as usize;
        let origin_x = x * PROOF_SPACING_M;
        let origin_z = z * PROOF_SPACING_M;
        let surface = proof_height(
            heights,
            cell_x,
            cell_z,
            f64::from(origin_x),
            f64::from(origin_z),
        ) as f32;
        let horizontal = elevation.cos();
        ProofRay {
            origin: [origin_x, surface + 1.7, origin_z],
            direction: [
                horizontal * azimuth.cos(),
                elevation.sin(),
                horizontal * azimuth.sin(),
            ],
            inv_two_r_prime: 1.0 / 14_650_000.0,
        }
    }

    fn visit_statements(block: &naga::Block, visit: &mut impl FnMut(&naga::Statement)) {
        for statement in block.iter() {
            visit(statement);
            match statement {
                naga::Statement::Block(inner) => visit_statements(inner, visit),
                naga::Statement::If { accept, reject, .. } => {
                    visit_statements(accept, visit);
                    visit_statements(reject, visit);
                }
                naga::Statement::Switch { cases, .. } => {
                    for case in cases {
                        visit_statements(&case.body, visit);
                    }
                }
                naga::Statement::Loop {
                    body, continuing, ..
                } => {
                    visit_statements(body, visit);
                    visit_statements(continuing, visit);
                }
                _ => {}
            }
        }
    }

    fn expression_depends_on(
        function: &naga::Function,
        expression: naga::Handle<naga::Expression>,
        predicate: &impl Fn(naga::Handle<naga::Expression>, &naga::Expression) -> bool,
    ) -> bool {
        let value = &function.expressions[expression];
        if predicate(expression, value) {
            return true;
        }
        let recurse = |child| expression_depends_on(function, child, predicate);
        match value {
            naga::Expression::Compose { components, .. } => components.iter().copied().any(recurse),
            naga::Expression::Access { base, index } => recurse(*base) || recurse(*index),
            naga::Expression::AccessIndex { base, .. }
            | naga::Expression::Splat { value: base, .. }
            | naga::Expression::Swizzle { vector: base, .. }
            | naga::Expression::Load { pointer: base }
            | naga::Expression::Unary { expr: base, .. }
            | naga::Expression::Derivative { expr: base, .. }
            | naga::Expression::Relational { argument: base, .. }
            | naga::Expression::As { expr: base, .. }
            | naga::Expression::ArrayLength(base) => recurse(*base),
            naga::Expression::Binary { left, right, .. } => recurse(*left) || recurse(*right),
            naga::Expression::Select {
                condition,
                accept,
                reject,
            } => recurse(*condition) || recurse(*accept) || recurse(*reject),
            naga::Expression::Math {
                arg,
                arg1,
                arg2,
                arg3,
                ..
            } => {
                recurse(*arg)
                    || arg1.is_some_and(recurse)
                    || arg2.is_some_and(recurse)
                    || arg3.is_some_and(recurse)
            }
            naga::Expression::ImageSample {
                image,
                sampler,
                coordinate,
                array_index,
                depth_ref,
                ..
            } => {
                recurse(*image)
                    || recurse(*sampler)
                    || recurse(*coordinate)
                    || array_index.is_some_and(recurse)
                    || depth_ref.is_some_and(recurse)
            }
            naga::Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => {
                recurse(*image)
                    || recurse(*coordinate)
                    || array_index.is_some_and(recurse)
                    || sample.is_some_and(recurse)
                    || level.is_some_and(recurse)
            }
            naga::Expression::ImageQuery { image, .. } => recurse(*image),
            naga::Expression::RayQueryGetIntersection { query, .. } => recurse(*query),
            _ => false,
        }
    }

    fn find_function(
        module: &naga::Module,
        name: &str,
    ) -> Result<naga::Handle<naga::Function>, String> {
        module
            .functions
            .iter()
            .find_map(|(handle, function)| {
                (function.name.as_deref() == Some(name)).then_some(handle)
            })
            .ok_or_else(|| format!("assembled hybrid shader is missing {name}"))
    }

    fn call_contract(
        function: &naga::Function,
        callee: naga::Handle<naga::Function>,
        curvature_argument: usize,
        caller_argument: u32,
    ) -> Option<naga::Handle<naga::Expression>> {
        let mut result = None;
        visit_statements(&function.body, &mut |statement| {
            if let naga::Statement::Call {
                function: target,
                arguments,
                result: call_result,
            } = statement
            {
                if *target == callee
                    && arguments.get(curvature_argument).is_some_and(|argument| {
                        expression_depends_on(function, *argument, &|_, expression| {
                            matches!(expression, naga::Expression::FunctionArgument(index) if *index == caller_argument)
                        })
                    })
                {
                    result = *call_result;
                }
            }
        });
        result
    }

    fn call_with_bool_literal(
        function: &naga::Function,
        callee: naga::Handle<naga::Function>,
        argument_index: usize,
        expected: bool,
    ) -> Option<naga::Handle<naga::Expression>> {
        let mut result = None;
        visit_statements(&function.body, &mut |statement| {
            if let naga::Statement::Call {
                function: target,
                arguments,
                result: call_result,
            } = statement
            {
                if *target == callee && arguments.get(argument_index).is_some_and(|argument| {
                    matches!(
                        function.expressions[*argument],
                        naga::Expression::Literal(naga::Literal::Bool(value)) if value == expected
                    )
                }) {
                    result = *call_result;
                }
            }
        });
        result
    }

    fn call_result(
        function: &naga::Function,
        callee: naga::Handle<naga::Function>,
    ) -> Option<naga::Handle<naga::Expression>> {
        let mut result = None;
        visit_statements(&function.body, &mut |statement| {
            if let naga::Statement::Call {
                function: target,
                result: call_result,
                ..
            } = statement
            {
                if *target == callee {
                    result = *call_result;
                }
            }
        });
        result
    }

    fn result_feeds_control(
        function: &naga::Function,
        result: naga::Handle<naga::Expression>,
    ) -> bool {
        let mut feeds = false;
        visit_statements(&function.body, &mut |statement| {
            let control = match statement {
                naga::Statement::If { condition, .. } => Some(*condition),
                naga::Statement::Switch { selector, .. } => Some(*selector),
                _ => None,
            };
            feeds |= control.is_some_and(|expression| {
                expression_depends_on(function, expression, &|handle, _| handle == result)
            });
        });
        feeds
    }

    fn expression_is_tainted(
        function: &naga::Function,
        expression: naga::Handle<naga::Expression>,
        result: naga::Handle<naga::Expression>,
        locals: &std::collections::HashSet<naga::Handle<naga::LocalVariable>>,
    ) -> bool {
        expression_depends_on(function, expression, &|handle, value| {
            handle == result
                || matches!(value, naga::Expression::LocalVariable(local) if locals.contains(local))
        })
    }

    fn pointer_local(
        function: &naga::Function,
        expression: naga::Handle<naga::Expression>,
    ) -> Option<naga::Handle<naga::LocalVariable>> {
        match function.expressions[expression] {
            naga::Expression::LocalVariable(local) => Some(local),
            naga::Expression::Access { base, .. } | naga::Expression::AccessIndex { base, .. } => {
                pointer_local(function, base)
            }
            _ => None,
        }
    }

    fn pointer_global(
        function: &naga::Function,
        expression: naga::Handle<naga::Expression>,
    ) -> Option<naga::Handle<naga::GlobalVariable>> {
        match function.expressions[expression] {
            naga::Expression::GlobalVariable(global) => Some(global),
            naga::Expression::Access { base, .. } | naga::Expression::AccessIndex { base, .. } => {
                pointer_global(function, base)
            }
            _ => None,
        }
    }

    fn argument_controls_return(
        function: &naga::Function,
        block: &naga::Block,
        argument: u32,
        control_tainted: bool,
    ) -> bool {
        for statement in block.iter() {
            match statement {
                naga::Statement::If {
                    condition,
                    accept,
                    reject,
                } => {
                    let tainted = control_tainted
                        || expression_depends_on(
                            function,
                            *condition,
                            &|_, expression| matches!(expression, naga::Expression::FunctionArgument(index) if *index == argument),
                        );
                    if argument_controls_return(function, accept, argument, tainted)
                        || argument_controls_return(function, reject, argument, tainted)
                    {
                        return true;
                    }
                }
                naga::Statement::Switch { selector, cases } => {
                    let tainted = control_tainted
                        || expression_depends_on(
                            function,
                            *selector,
                            &|_, expression| matches!(expression, naga::Expression::FunctionArgument(index) if *index == argument),
                        );
                    if cases.iter().any(|case| {
                        argument_controls_return(function, &case.body, argument, tainted)
                    }) {
                        return true;
                    }
                }
                naga::Statement::Loop {
                    body, continuing, ..
                } => {
                    if argument_controls_return(function, body, argument, control_tainted)
                        || argument_controls_return(function, continuing, argument, control_tainted)
                    {
                        return true;
                    }
                }
                naga::Statement::Block(inner) => {
                    if argument_controls_return(function, inner, argument, control_tainted) {
                        return true;
                    }
                }
                naga::Statement::Return { .. } if control_tainted => return true,
                _ => {}
            }
        }
        false
    }

    fn controlled_effect_local(
        function: &naga::Function,
        block: &naga::Block,
        result: naga::Handle<naga::Expression>,
        control_tainted: bool,
    ) -> Option<naga::Handle<naga::LocalVariable>> {
        for statement in block.iter() {
            match statement {
                naga::Statement::Store { pointer, value } if control_tainted => {
                    let local = pointer_local(function, *pointer)?;
                    if !expression_depends_on(
                        function,
                        *value,
                        &|_, expression| matches!(expression, naga::Expression::LocalVariable(candidate) if *candidate == local),
                    ) {
                        return Some(local);
                    }
                }
                naga::Statement::If {
                    condition,
                    accept,
                    reject,
                } => {
                    let tainted = control_tainted
                        || expression_depends_on(function, *condition, &|handle, _| {
                            handle == result
                        });
                    if let Some(local) = controlled_effect_local(function, accept, result, tainted)
                        .or_else(|| controlled_effect_local(function, reject, result, tainted))
                    {
                        return Some(local);
                    }
                }
                naga::Statement::Switch { selector, cases } => {
                    let tainted = control_tainted
                        || expression_depends_on(function, *selector, &|handle, _| {
                            handle == result
                        });
                    for case in cases {
                        if let Some(local) =
                            controlled_effect_local(function, &case.body, result, tainted)
                        {
                            return Some(local);
                        }
                    }
                }
                naga::Statement::Loop {
                    body, continuing, ..
                } => {
                    if let Some(local) = controlled_effect_local(
                        function,
                        body,
                        result,
                        control_tainted,
                    )
                    .or_else(|| {
                        controlled_effect_local(function, continuing, result, control_tainted)
                    }) {
                        return Some(local);
                    }
                }
                naga::Statement::Block(inner) => {
                    if let Some(local) =
                        controlled_effect_local(function, inner, result, control_tainted)
                    {
                        return Some(local);
                    }
                }
                _ => {}
            }
        }
        None
    }

    fn block_stores_zero(
        function: &naga::Function,
        block: &naga::Block,
        target: naga::Handle<naga::LocalVariable>,
    ) -> bool {
        block.iter().any(|statement| {
            matches!(
                statement,
                naga::Statement::Store { pointer, value }
                    if pointer_local(function, *pointer) == Some(target)
                        && matches!(
                            function.expressions[*value],
                            naga::Expression::Literal(naga::Literal::F32(value)) if value == 0.0
                        )
            )
        })
    }

    fn expression_is_any_hit_and_entry_nonpositive(
        function: &naga::Function,
        expression: naga::Handle<naga::Expression>,
        entry_deviation: naga::Handle<naga::Expression>,
    ) -> bool {
        let naga::Expression::Binary {
            op: naga::BinaryOperator::LogicalAnd,
            left,
            right,
        } = function.expressions[expression]
        else {
            return false;
        };
        let is_any_hit = |operand| {
            matches!(
                function.expressions[operand],
                naga::Expression::FunctionArgument(6)
            )
        };
        let is_entry_nonpositive = |operand| {
            matches!(
                function.expressions[operand],
                naga::Expression::Binary {
                    op: naga::BinaryOperator::LessEqual,
                    left,
                    right,
                } if left == entry_deviation
                    && matches!(
                        function.expressions[right],
                        naga::Expression::Literal(naga::Literal::F32(value)) if value == 0.0
                    )
            )
        };
        (is_any_hit(left) && is_entry_nonpositive(right))
            || (is_entry_nonpositive(left) && is_any_hit(right))
    }

    fn local_reaches_global_store(
        module: &naga::Module,
        function: &naga::Function,
        seed: naga::Handle<naga::LocalVariable>,
        global_name: &str,
    ) -> bool {
        let mut locals = std::collections::HashSet::from([seed]);
        loop {
            let before = locals.len();
            let mut reaches = false;
            visit_statements(&function.body, &mut |statement| {
                if let naga::Statement::Store { pointer, value } = statement {
                    let tainted = expression_depends_on(
                        function,
                        *value,
                        &|_, expression| matches!(expression, naga::Expression::LocalVariable(local) if locals.contains(local)),
                    );
                    if tainted {
                        if let Some(local) = pointer_local(function, *pointer) {
                            locals.insert(local);
                        }
                        if pointer_global(function, *pointer).is_some_and(|global| {
                            module.global_variables[global].name.as_deref() == Some(global_name)
                        }) {
                            reaches = true;
                        }
                    }
                }
            });
            if reaches {
                return true;
            }
            if locals.len() == before {
                return false;
            }
        }
    }

    fn taint_block(
        function: &naga::Function,
        block: &naga::Block,
        result: naga::Handle<naga::Expression>,
        locals: &mut std::collections::HashSet<naga::Handle<naga::LocalVariable>>,
        control_tainted: bool,
        return_tainted: &mut bool,
    ) -> bool {
        let mut changed = false;
        for statement in block.iter() {
            match statement {
                naga::Statement::Store { pointer, value } => {
                    if control_tainted || expression_is_tainted(function, *value, result, locals) {
                        if let Some(local) = pointer_local(function, *pointer) {
                            changed |= locals.insert(local);
                        }
                    }
                }
                naga::Statement::If {
                    condition,
                    accept,
                    reject,
                } => {
                    let branch_tainted = control_tainted
                        || expression_is_tainted(function, *condition, result, locals);
                    changed |= taint_block(
                        function,
                        accept,
                        result,
                        locals,
                        branch_tainted,
                        return_tainted,
                    );
                    changed |= taint_block(
                        function,
                        reject,
                        result,
                        locals,
                        branch_tainted,
                        return_tainted,
                    );
                }
                naga::Statement::Switch { selector, cases } => {
                    let branch_tainted = control_tainted
                        || expression_is_tainted(function, *selector, result, locals);
                    for case in cases {
                        changed |= taint_block(
                            function,
                            &case.body,
                            result,
                            locals,
                            branch_tainted,
                            return_tainted,
                        );
                    }
                }
                naga::Statement::Loop {
                    body,
                    continuing,
                    break_if,
                } => {
                    changed |= taint_block(
                        function,
                        body,
                        result,
                        locals,
                        control_tainted,
                        return_tainted,
                    );
                    let continuing_tainted = control_tainted
                        || break_if.is_some_and(|condition| {
                            expression_is_tainted(function, condition, result, locals)
                        });
                    changed |= taint_block(
                        function,
                        continuing,
                        result,
                        locals,
                        continuing_tainted,
                        return_tainted,
                    );
                }
                naga::Statement::Block(inner) => {
                    changed |= taint_block(
                        function,
                        inner,
                        result,
                        locals,
                        control_tainted,
                        return_tainted,
                    );
                }
                naga::Statement::Return { value } => {
                    *return_tainted |= control_tainted
                        || value.is_some_and(|returned| {
                            expression_is_tainted(function, returned, result, locals)
                        });
                }
                _ => {}
            }
        }
        changed
    }

    fn result_reaches_return(
        function: &naga::Function,
        result: naga::Handle<naga::Expression>,
    ) -> bool {
        let mut locals = std::collections::HashSet::new();
        loop {
            let mut changed = false;
            for (local, variable) in function.local_variables.iter() {
                if variable.init.is_some_and(|initial| {
                    expression_is_tainted(function, initial, result, &locals)
                }) {
                    changed |= locals.insert(local);
                }
            }
            let mut return_tainted = false;
            changed |= taint_block(
                function,
                &function.body,
                result,
                &mut locals,
                false,
                &mut return_tainted,
            );
            if return_tainted {
                return true;
            }
            if !changed {
                return false;
            }
        }
    }

    fn assert_curvature_shader_dataflow(source: &str) -> Result<(), String> {
        let module = naga::front::wgsl::parse_str(source)
            .map_err(|error| format!("assembled hybrid shader does not parse: {error}"))?;
        let trace_handle = find_function(&module, "terrain_trace")?;
        let range_handle = find_function(&module, "terrain_curved_height_range")?;
        let leaf_handle = find_function(&module, "terrain_leaf_intersect")?;
        let height_handle = find_function(&module, "terrain_curved_height")?;
        let optimized_handle = find_function(&module, "intersect_hybrid_optimized")?;
        let shadow_handle = find_function(&module, "intersect_shadow_ray")?;
        let ibl_handle = find_function(&module, "intersect_ibl_occlusion_ray")?;
        let main = &module
            .entry_points
            .iter()
            .find(|entry| entry.name == "main_terrain")
            .ok_or("assembled hybrid shader is missing main_terrain")?
            .function;
        let trace = &module.functions[trace_handle];
        let range = &module.functions[range_handle];
        let leaf = &module.functions[leaf_handle];
        let height = &module.functions[height_handle];
        let optimized = &module.functions[optimized_handle];
        let shadow = &module.functions[shadow_handle];
        let ibl = &module.functions[ibl_handle];

        let range_result = call_contract(trace, range_handle, 3, 2)
            .ok_or("terrain_trace does not forward its explicit curvature policy to node tests")?;
        if !result_feeds_control(trace, range_result) {
            return Err("curvature-aware node range no longer controls descent rejection".into());
        }
        let leaf_result = call_contract(trace, leaf_handle, 5, 2)
            .ok_or("terrain_trace does not forward its explicit curvature policy to leaf tests")?;
        call_contract(trace, leaf_handle, 6, 1)
            .ok_or("terrain_trace does not forward its any-hit policy to leaf tests")?;
        if !result_feeds_control(trace, leaf_result) {
            return Err("curvature-aware leaf result no longer controls terrain hits".into());
        }
        let range_height = call_contract(range, height_handle, 2, 3)
            .ok_or("node range no longer evaluates the curved ray height")?;
        if !result_reaches_return(range, range_height) {
            return Err("curved node height no longer reaches the node range return".into());
        }
        let leaf_height = call_contract(leaf, height_handle, 2, 5)
            .ok_or("leaf solve no longer evaluates the curved ray height")?;
        if !result_reaches_return(leaf, leaf_height) {
            return Err("curved leaf height no longer reaches TerrainLeafHit".into());
        }
        let leaf_entry_deviation = leaf
            .named_expressions
            .iter()
            .find_map(|(handle, name)| (name == "c").then_some(handle))
            .ok_or("leaf solve no longer names its entry deviation")?;
        let leaf_hit_parameter = leaf
            .local_variables
            .iter()
            .find_map(|(handle, variable)| {
                (variable.name.as_deref() == Some("s_hit")).then_some(handle)
            })
            .ok_or("leaf solve no longer exposes its hit parameter")?;
        let closes_boundary_crack = leaf.body.iter().any(|statement| {
            matches!(statement, naga::Statement::If { condition, accept, .. }
                if expression_is_any_hit_and_entry_nonpositive(
                    leaf,
                    *condition,
                    *leaf_entry_deviation,
                )
                    && block_stores_zero(leaf, accept, leaf_hit_parameter))
        });
        if !closes_boundary_crack {
            return Err("below-terrain leaf entry no longer closes shared-boundary cracks".into());
        }

        let optimized_result = call_contract(optimized, trace_handle, 2, 2)
            .ok_or("hybrid any-hit traversal does not forward its curvature policy")?;
        if !result_feeds_control(optimized, optimized_result) {
            return Err("terrain any-hit result no longer controls hybrid occlusion".into());
        }
        if !argument_controls_return(trace, &trace.body, 1, false) {
            return Err("terrain_trace any_hit no longer controls early termination".into());
        }
        let sun_result = call_with_bool_literal(shadow, optimized_handle, 2, true)
            .ok_or("sun shadow traversal is not explicitly curvature-enabled")?;
        if !result_reaches_return(shadow, sun_result) {
            return Err("curvature-enabled sun traversal no longer reaches visibility".into());
        }
        let ibl_result = call_with_bool_literal(ibl, optimized_handle, 2, false)
            .ok_or("IBL traversal is not explicitly curvature-disabled")?;
        if !result_reaches_return(ibl, ibl_result) {
            return Err("curvature-independent IBL traversal no longer reaches visibility".into());
        }
        for (callee, label) in [(shadow_handle, "sun"), (ibl_handle, "IBL")] {
            let result = call_result(main, callee).ok_or_else(|| {
                format!("main_terrain no longer calls the {label} visibility path")
            })?;
            let effect = controlled_effect_local(main, &main.body, result, false)
                .ok_or_else(|| format!("{label} visibility no longer changes a shading value"))?;
            if !local_reaches_global_store(&module, main, effect, "accum_hdr") {
                return Err(format!(
                    "{label} visibility no longer reaches accumulated terrain radiance"
                ));
            }
        }

        let curvature = module
            .global_variables
            .iter()
            .find_map(|(handle, variable)| {
                (variable.name.as_deref() == Some("earth_curvature")).then_some(handle)
            })
            .ok_or("assembled hybrid shader is missing earth_curvature")?;
        let mut return_uses_inv_two_r_prime = false;
        visit_statements(&height.body, &mut |statement| {
            if let naga::Statement::Return { value: Some(value) } = statement {
                return_uses_inv_two_r_prime |= expression_depends_on(
                    height,
                    *value,
                    &|handle, expression| {
                        if let naga::Expression::AccessIndex { base, index: 0 } = expression {
                            expression_depends_on(
                                height,
                                *base,
                                &|_, base_expression| matches!(base_expression, naga::Expression::GlobalVariable(global) if *global == curvature),
                            )
                        } else {
                            let _ = handle;
                            false
                        }
                    },
                );
            }
        });
        if !return_uses_inv_two_r_prime {
            return Err("curved ray height no longer depends on inv_two_r_prime".into());
        }
        Ok(())
    }

    const PRODUCTION_GPU_PROOF_ENTRY: &str = r#"
struct HeliosProofRay {
    origin_tmin: vec4<f32>,
    direction_tmax: vec4<f32>,
}

@group(3) @binding(8) var<storage, read_write> helios_proof_hits: array<u32>;
@group(3) @binding(9) var<storage, read> helios_proof_rays: array<HeliosProofRay>;

// Test-only dispatch seam. The call resolves to the exact production
// terrain_trace function assembled above; no traversal or curvature math is
// copied into this entry point.
@compute @workgroup_size(64, 1, 1)
fn main_helios_production_terrain_trace_proof(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= arrayLength(&helios_proof_rays)) { return; }
    let input = helios_proof_rays[index];
    var ray: Ray;
    ray.origin = input.origin_tmin.xyz;
    ray.tmin = input.origin_tmin.w;
    ray.direction = input.direction_tmax.xyz;
    ray.tmax = input.direction_tmax.w;
    let hit = terrain_trace(ray, true, true);
    helios_proof_hits[index] = select(0u, 1u, hit.hit != 0u);
}
"#;

    fn assert_production_gpu_proof_calls_exact_trace(source: &str) -> Result<(), String> {
        let module = naga::front::wgsl::parse_str(source)
            .map_err(|error| format!("production GPU proof does not parse: {error}"))?;
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|error| format!("production GPU proof does not validate: {error}"))?;
        let trace = find_function(&module, "terrain_trace")?;
        let entry = &module
            .entry_points
            .iter()
            .find(|entry| entry.name == "main_helios_production_terrain_trace_proof")
            .ok_or("production GPU proof entry is missing")?
            .function;
        let result = call_result(entry, trace)
            .ok_or("production GPU proof entry does not call exact production terrain_trace")?;
        let output = module
            .global_variables
            .iter()
            .find_map(|(handle, variable)| {
                (variable.name.as_deref() == Some("helios_proof_hits")).then_some(handle)
            })
            .ok_or("production GPU proof hit output is missing")?;
        let mut reaches_output = false;
        visit_statements(&entry.body, &mut |statement| {
            if let naga::Statement::Store { pointer, value } = statement {
                reaches_output |= pointer_global(entry, *pointer) == Some(output)
                    && expression_depends_on(entry, *value, &|handle, _| handle == result);
            }
        });
        reaches_output
            .then_some(())
            .ok_or_else(|| "production terrain_trace hit bit does not reach proof output".into())
    }

    fn production_gpu_hits(
        heights: &[f32],
        rays: &[ProofRay],
    ) -> Result<(Vec<u32>, wgpu::AdapterInfo, bool), String> {
        let context = try_ctx().map_err(|error| error.to_string())?;
        let device = &context.device;
        let queue = &context.queue;
        let adapter_info = context.adapter.get_info();
        let pyramid = TerrainMinMaxPyramid::from_heightfield(
            device,
            queue,
            heights,
            PROOF_DEM_SIDE as u32,
            PROOF_DEM_SIDE as u32,
        )
        .map_err(|error| error.to_string())?;
        let terrain_uniform = TerrainPtUniforms {
            origin_spacing: [0.0, 0.0, PROOF_SPACING_M, PROOF_SPACING_M],
            h_params: [pyramid.h_min, pyramid.h_max, 1.0, 0.0],
            albedo_pad: [0.0; 4],
            dims: [
                PROOF_DEM_SIDE as u32,
                PROOF_DEM_SIDE as u32,
                PROOF_CELL_COUNT as u32,
                PROOF_CELL_COUNT as u32,
            ],
            mips: [pyramid.mip_count, 1, 0, 0],
            extra: [0; 4],
        };
        let curvature_uniform = EarthCurvatureUniforms {
            inv_two_r_prime: 1.0 / 14_650_000.0,
            _pad0: 0.0,
            ray_origin_geodetic: [0.0; 2],
            enabled: 1,
            _pad1: 0,
        };
        let terrain_buffer = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("helios-production-proof-terrain-uniform"),
                contents: bytemuck::bytes_of(&terrain_uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )
        .map_err(|error| error.to_string())?;
        let curvature_buffer = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("helios-production-proof-curvature-uniform"),
                contents: bytemuck::bytes_of(&curvature_uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )
        .map_err(|error| error.to_string())?;
        let gpu_rays: Vec<ProofGpuRay> = rays.iter().copied().map(Into::into).collect();
        let ray_buffer = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("helios-production-proof-rays"),
                contents: bytemuck::cast_slice(&gpu_rays),
                usage: wgpu::BufferUsages::STORAGE,
            },
        )
        .map_err(|error| error.to_string())?;
        let output_size = (rays.len() * std::mem::size_of::<u32>()) as u64;
        let output = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("helios-production-proof-hit-bits"),
                size: output_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )
        .map_err(|error| error.to_string())?;
        let readback = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("helios-production-proof-readback"),
                size: output_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )
        .map_err(|error| error.to_string())?;

        let source = format!(
            "{}\n{}",
            crate::shader_sources::hybrid_kernel(),
            PRODUCTION_GPU_PROOF_ENTRY
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("helios-production-terrain-trace-proof"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("helios-production-terrain-trace-proof"),
                layout: None,
                module: &shader,
                entry_point: "main_helios_production_terrain_trace_proof",
            },
        );
        let empty0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("helios-production-proof-empty-0"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[],
        });
        let empty1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("helios-production-proof-empty-1"),
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[],
        });
        let height_view = pyramid
            .height_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let minmax_view = pyramid
            .minmax_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let terrain_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("helios-production-proof-terrain"),
            layout: &pipeline.get_bind_group_layout(2),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&height_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&minmax_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: terrain_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: curvature_buffer.as_entire_binding(),
                },
            ],
        });
        let proof_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("helios-production-proof-io"),
            layout: &pipeline.get_bind_group_layout(3),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: ray_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("helios-production-proof-dispatch"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("helios-production-proof-dispatch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &empty0, &[]);
            pass.set_bind_group(1, &empty1, &[]);
            pass.set_bind_group(2, &terrain_group, &[]);
            pass.set_bind_group(3, &proof_group, &[]);
            pass.dispatch_workgroups((rays.len() as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_size);
        queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|error| error.to_string())?
            .map_err(|error| error.to_string())?;
        let mapped = slice.get_mapped_range();
        let hits = bytemuck::cast_slice::<u8, u32>(&mapped).to_vec();
        drop(mapped);
        readback.unmap();
        Ok((hits, adapter_info, context.software_fallback))
    }

    fn validate_helios_candidate_sha(
        candidate: Option<String>,
        require_physical: bool,
    ) -> Option<String> {
        let Some(candidate) = candidate else {
            assert!(
                !require_physical,
                "FORGE3D_HELIOS_CANDIDATE_SHA must bind the production GPU proof"
            );
            return None;
        };
        assert!(
            candidate.len() == 40
                && candidate
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
            "FORGE3D_HELIOS_CANDIDATE_SHA must be a full lowercase Git SHA"
        );
        Some(candidate)
    }

    fn helios_candidate_sha(require_physical: bool) -> Option<String> {
        let candidate = match std::env::var("FORGE3D_HELIOS_CANDIDATE_SHA") {
            Ok(candidate) => Some(candidate),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => {
                panic!("FORGE3D_HELIOS_CANDIDATE_SHA must be a full lowercase Git SHA")
            }
        };
        validate_helios_candidate_sha(candidate, require_physical)
    }

    fn helios_gpu_proof_status(
        require_physical: bool,
        physical_nvidia_vulkan: bool,
    ) -> &'static str {
        if require_physical && physical_nvidia_vulkan {
            "PASS"
        } else {
            "NOT_PROVEN"
        }
    }

    #[test]
    fn production_gpu_status_requires_strict_physical_lane() {
        assert_eq!(helios_gpu_proof_status(false, false), "NOT_PROVEN");
        assert_eq!(helios_gpu_proof_status(false, true), "NOT_PROVEN");
        assert_eq!(helios_gpu_proof_status(true, false), "NOT_PROVEN");
        assert_eq!(helios_gpu_proof_status(true, true), "PASS");
    }

    #[test]
    fn candidate_sha_is_optional_only_outside_strict_physical_lane() {
        let valid = "1".repeat(40);
        assert_eq!(validate_helios_candidate_sha(None, false), None);
        assert_eq!(
            validate_helios_candidate_sha(Some(valid.clone()), false),
            Some(valid.clone())
        );
        assert_eq!(
            validate_helios_candidate_sha(Some(valid.clone()), true),
            Some(valid)
        );
        assert!(std::panic::catch_unwind(|| validate_helios_candidate_sha(None, true)).is_err());
        assert!(std::panic::catch_unwind(|| {
            validate_helios_candidate_sha(Some("INVALID".to_owned()), false)
        })
        .is_err());
    }

    #[test]
    fn captured_physical_mask_miss_survives_leaf_boundary_rounding() {
        let heights = curvature_fixture();
        let ray = ProofRay {
            origin: [125_750.0, 870.546_14, 67_750.0],
            direction: [0.798_591_73, 0.010_471_784, 0.601_781_96],
            inv_two_r_prime: 6.825_938_2e-8,
        };
        assert!(brute_2d_hit(&heights, ray));

        // The independent oracle crosses into leaf (254,137) within 0.061 mm
        // of its shared boundary. NVIDIA Vulkan rounded the entry to the
        // below-terrain side while the adjacent leaf remained above it.
        assert!(deviation_span_hit(
            -0.000_061_035,
            -0.663_696_3,
            -1.318_298_3,
            true,
        ));
    }

    #[test]
    fn primary_below_entry_without_crossing_is_not_a_hit() {
        assert!(!deviation_span_hit(-1.0, -2.0, -3.0, false));
    }

    #[test]
    fn primary_below_entry_with_upward_exit_keeps_exact_root() {
        assert!(deviation_span_hit(-1.0, 0.0, 1.0, false));
    }

    #[test]
    fn curvature_descent_is_conservative() {
        assert_eq!(std::mem::size_of::<EarthCurvatureUniforms>(), 24);
        let shader = crate::shader_sources::hybrid_kernel();
        assert_curvature_shader_dataflow(&shader).unwrap();
        let production_proof = format!("{shader}\n{PRODUCTION_GPU_PROOF_ENTRY}");
        assert_production_gpu_proof_calls_exact_trace(&production_proof).unwrap();
        let severed_proof = production_proof.replace(
            "let hit = terrain_trace(ray, true, true);",
            "var hit: HybridHitResult; hit.hit = 0u;",
        );
        assert_ne!(
            severed_proof, production_proof,
            "GPU proof mutation target drifted"
        );
        assert!(assert_production_gpu_proof_calls_exact_trace(&severed_proof).is_err());
        for mutant in [
            shader.replace(
                "let ray_height = terrain_curved_height_range(ray, t_lo, t_hi, apply_curvature);",
                "let ray_height = vec2<f32>(ray.origin.y + t_lo * ray.direction.y, ray.origin.y + t_hi * ray.direction.y);",
            ),
            shader.replace(
                "terrain_leaf_intersect(ray, cx0, cz0, t_lo, t_hi, apply_curvature, any_hit)",
                "terrain_leaf_intersect(ray, cx0, cz0, t_lo, t_hi, false, any_hit)",
            ),
            shader.replace(
                "terrain_leaf_intersect(ray, cx0, cz0, t_lo, t_hi, apply_curvature, any_hit)",
                "terrain_leaf_intersect(ray, cx0, cz0, t_lo, t_hi, apply_curvature, false)",
            ),
            shader.replace(
                "let hit = intersect_hybrid_optimized(ray, 0.01, false);",
                "let hit = intersect_hybrid_optimized(ray, 0.01, true);",
            ),
            shader.replace(
                "if (intersect_ibl_occlusion_ray(eray, 1e30))",
                "if (intersect_shadow_ray(eray, 1e30))",
            ),
            shader.replace("if (any_hit) { return res; }", "if (apply_curvature) { return res; }"),
            shader.replace("env_vis = 0.0;", "env_vis = env_vis;"),
            shader.replace(
                "horizontal_d2 * earth_curvature.inv_two_r_prime",
                "0.0",
            ),
            shader.replace(
                "return vec2<f32>(minimum, max(y0, y1));",
                "return vec2<f32>(ray.origin.y + t0 * ray.direction.y, ray.origin.y + t1 * ray.direction.y);",
            ),
            shader.replace(
                "let c = d3.x;\n    let a = 2.0 * d3.z + 2.0 * d3.x - 4.0 * d3.y;\n    let b = d3.z - d3.x - a;",
                "let c = 1.0;\n    let a = 0.0;\n    let b = 1.0;",
            ),
            shader.replace(
                "if (any_hit && c <= 0.0) {\n        s_hit = 0.0;",
                "if (any_hit && c >= 0.0) {\n        s_hit = 0.0;",
            ),
            shader.replace(
                "if (any_hit && c <= 0.0) {\n        s_hit = 0.0;",
                "if (any_hit && d3.y <= 0.0) {\n        s_hit = 0.0;",
            ),
            shader.replace(
                "if (any_hit && c <= 0.0) {\n        s_hit = 0.0;",
                "if (any_hit || c <= 0.0) {\n        s_hit = 0.0;",
            ),
            shader.replace(
                "if (any_hit && c <= 0.0) {\n        s_hit = 0.0;",
                "if (!any_hit && c <= 0.0) {\n        s_hit = 0.0;",
            ),
        ] {
            assert_ne!(mutant, shader, "HELIOS mutation target drifted");
            assert!(assert_curvature_shader_dataflow(&mutant).is_err());
        }
        let distance = 100_000.0f64;
        let inv_two_r = 1.0 / 14_650_000.0;
        let reference_drop = distance * distance * inv_two_r;
        let gpu_drop = (distance as f32).powi(2) * inv_two_r as f32;
        assert!((reference_drop - f64::from(gpu_drop)).abs() < 1.0);

        let heights = curvature_fixture();
        let mips =
            build_minmax_mips(&heights, PROOF_DEM_SIDE as u32, PROOF_DEM_SIDE as u32).unwrap();
        let mut state = 0x4845_4c49u32;
        let mut false_misses = 0usize;
        let mut false_hits = 0usize;
        for _ in 0..10_000 {
            let x = 1.0
                + (next_u32(&mut state) % 253) as f32
                + (next_u32(&mut state) as f32 / u32::MAX as f32) * 0.999;
            let z = 1.0
                + (next_u32(&mut state) % 253) as f32
                + (next_u32(&mut state) as f32 / u32::MAX as f32) * 0.999;
            let azimuth = (next_u32(&mut state) as f32 / u32::MAX as f32) * std::f32::consts::TAU;
            let elevation = (0.1 + (next_u32(&mut state) % 790) as f32 / 100.0).to_radians();
            let ray = proof_ray(&heights, x, z, azimuth, elevation);
            let brute = brute_2d_hit(&heights, ray);
            let descent = descent_2d_hit(&heights, &mips, ray);
            false_misses += usize::from(brute && !descent);
            false_hits += usize::from(!brute && descent);
        }

        let mut mask_matches = 0usize;
        let mut mask_pixels = 0usize;
        let mut mask_false_misses = 0usize;
        let mut mask_false_hits = 0usize;
        let azimuth = 37.0f32.to_radians();
        let elevation = 0.6f32.to_radians();
        for z in 0..PROOF_CELL_COUNT {
            for x in 0..PROOF_CELL_COUNT {
                let ray = proof_ray(&heights, x as f32 + 0.5, z as f32 + 0.5, azimuth, elevation);
                let brute = brute_2d_hit(&heights, ray);
                let descent = descent_2d_hit(&heights, &mips, ray);
                mask_matches += usize::from(brute == descent);
                mask_false_misses += usize::from(brute && !descent);
                mask_false_hits += usize::from(!brute && descent);
                mask_pixels += 1;
            }
        }
        let false_hit_rate = false_hits as f64 / 10_000.0;
        let agreement = mask_matches as f64 / mask_pixels as f64;
        println!(
            "HELIOS 2D 256x256 conservative descent: rays=10000, false_misses={false_misses}, \
             false_hit_rate={false_hit_rate:.6}, shadow_mask_agreement={agreement:.6}, \
             mask_false_misses={mask_false_misses}, mask_false_hits={mask_false_hits}"
        );
        assert_eq!(false_misses, 0);
        assert!(false_hit_rate < 0.001);
        assert!(agreement >= 0.999);
    }

    #[test]
    fn curvature_descent_production_gpu_is_conservative() {
        let require_physical = std::env::var_os("FORGE3D_HELIOS_REQUIRE_PHYSICAL_GPU").is_some();
        let candidate_sha = helios_candidate_sha(require_physical);
        let heights = curvature_fixture();
        let mut state = 0x4845_4c49u32;
        let mut rays = Vec::with_capacity(10_000 + PROOF_CELL_COUNT * PROOF_CELL_COUNT);
        let mut oracle = Vec::with_capacity(rays.capacity());
        for _ in 0..10_000 {
            let x = 1.0
                + (next_u32(&mut state) % 253) as f32
                + (next_u32(&mut state) as f32 / u32::MAX as f32) * 0.999;
            let z = 1.0
                + (next_u32(&mut state) % 253) as f32
                + (next_u32(&mut state) as f32 / u32::MAX as f32) * 0.999;
            let azimuth = (next_u32(&mut state) as f32 / u32::MAX as f32) * std::f32::consts::TAU;
            let elevation = (0.1 + (next_u32(&mut state) % 790) as f32 / 100.0).to_radians();
            let ray = proof_ray(&heights, x, z, azimuth, elevation);
            oracle.push(brute_2d_hit(&heights, ray));
            rays.push(ray);
        }
        let azimuth = 37.0f32.to_radians();
        let elevation = 0.6f32.to_radians();
        for z in 0..PROOF_CELL_COUNT {
            for x in 0..PROOF_CELL_COUNT {
                let ray = proof_ray(&heights, x as f32 + 0.5, z as f32 + 0.5, azimuth, elevation);
                oracle.push(brute_2d_hit(&heights, ray));
                rays.push(ray);
            }
        }

        let (gpu, adapter, software_fallback) = match production_gpu_hits(&heights, &rays) {
            Ok(result) => result,
            Err(error) if !require_physical => {
                println!(
                    "HELIOS_PRODUCTION_TERRAIN_TRACE_GPU_JSON {}",
                    serde_json::json!({
                        "schema": "forge3d.helios_production_terrain_trace_gpu/1",
                        "status": "NOT_PROVEN",
                        "candidate_sha": candidate_sha,
                        "strict_physical_required": false,
                        "adapter": serde_json::Value::Null,
                        "reason": error,
                        "metrics": {
                            "rays": 10_000,
                            "shadow_mask_pixels": PROOF_CELL_COUNT * PROOF_CELL_COUNT,
                        },
                    })
                );
                return;
            }
            Err(error) => panic!("required HELIOS physical GPU proof could not run: {error}"),
        };
        let physical_nvidia_vulkan = !software_fallback
            && adapter.vendor == 0x10de
            && adapter.device_type == wgpu::DeviceType::DiscreteGpu
            && adapter.backend == wgpu::Backend::Vulkan;
        assert_eq!(gpu.len(), oracle.len());
        let gpu: Vec<bool> = gpu
            .into_iter()
            .map(|hit| {
                assert!(hit <= 1, "production proof emitted non-bit value {hit}");
                hit != 0
            })
            .collect();
        let arbitrary = &gpu[..10_000];
        let arbitrary_oracle = &oracle[..10_000];
        let false_misses = arbitrary_oracle
            .iter()
            .zip(arbitrary)
            .filter(|(expected, actual)| **expected && !**actual)
            .count();
        let false_hits = arbitrary_oracle
            .iter()
            .zip(arbitrary)
            .filter(|(expected, actual)| !**expected && **actual)
            .count();
        let mask = &gpu[10_000..];
        let mask_oracle = &oracle[10_000..];
        let mask_false_miss_diagnostics: Vec<_> = mask_oracle
            .iter()
            .zip(mask)
            .enumerate()
            .filter_map(|(index, (expected, actual))| {
                if *expected && !*actual {
                    let ray = rays[10_000 + index];
                    Some(serde_json::json!({
                        "index": index,
                        "x": index % PROOF_CELL_COUNT,
                        "z": index / PROOF_CELL_COUNT,
                        "origin": ray.origin,
                        "direction": ray.direction,
                        "inv_two_r_prime": ray.inv_two_r_prime,
                    }))
                } else {
                    None
                }
            })
            .collect();
        if !mask_false_miss_diagnostics.is_empty() {
            println!(
                "HELIOS_PRODUCTION_TERRAIN_TRACE_GPU_MISSES_JSON {}",
                serde_json::json!({
                    "misses": mask_false_miss_diagnostics,
                })
            );
        }
        let mask_false_misses = mask_false_miss_diagnostics.len();
        let mask_false_hits = mask_oracle
            .iter()
            .zip(mask)
            .filter(|(expected, actual)| !**expected && **actual)
            .count();
        let mask_matches = mask.len() - mask_false_misses - mask_false_hits;
        let false_hit_rate = false_hits as f64 / 10_000.0;
        let agreement = mask_matches as f64 / mask.len() as f64;
        let status = helios_gpu_proof_status(require_physical, physical_nvidia_vulkan);
        println!(
            "HELIOS_PRODUCTION_TERRAIN_TRACE_GPU_JSON {}",
            serde_json::json!({
                "schema": "forge3d.helios_production_terrain_trace_gpu/1",
                "status": status,
                "candidate_sha": candidate_sha,
                "strict_physical_required": require_physical,
                "adapter": {
                    "status": "ok",
                    "name": adapter.name.clone(),
                    "vendor": adapter.vendor,
                    "device": adapter.device,
                    "device_type": format!("{:?}", adapter.device_type),
                    "backend": format!("{:?}", adapter.backend),
                    "driver": adapter.driver.clone(),
                    "driver_info": adapter.driver_info.clone(),
                    "software_fallback": software_fallback,
                    "software_classification": if software_fallback { "software" } else { "hardware" },
                },
                "metrics": {
                    "rays": 10_000,
                    "false_misses": false_misses,
                    "false_hit_rate": false_hit_rate,
                    "shadow_mask_pixels": mask.len(),
                    "shadow_mask_agreement": agreement,
                    "mask_false_misses": mask_false_misses,
                    "mask_false_hits": mask_false_hits,
                },
            })
        );
        if require_physical {
            assert!(
                physical_nvidia_vulkan,
                "HELIOS proof requires physical NVIDIA Vulkan, got {adapter:?}, software_fallback={software_fallback}"
            );
        }
        assert_eq!(false_misses, 0);
        assert!(false_hit_rate < 0.001);
        assert!(agreement >= 0.999);
    }
}
