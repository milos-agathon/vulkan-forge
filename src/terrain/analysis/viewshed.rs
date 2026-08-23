use crate::core::error::{RenderError, RenderResult};
use crate::core::resource_tracker::{tracked_create_buffer, tracked_create_buffer_init};
use crate::geo::refraction::{principal_radii_m, EarthModel, RefractionModel};
use crate::path_tracing::hybrid_compute::terrain_heightfield::TerrainMinMaxPyramid;
use bytemuck::{Pod, Zeroable};

#[derive(Clone, Copy, Debug)]
pub struct ViewshedOptions {
    pub width: u32,
    pub height: u32,
    pub observer_x: f32,
    pub observer_y: f32,
    pub observer_height_m: f32,
    pub target_height_m: f32,
    pub max_distance_m: f32,
    pub observer_latitude_rad: f32,
    pub observer_longitude_rad: f32,
    pub left_unwrapped_deg: f32,
    pub top_deg: f32,
    pub longitude_step_deg: f32,
    pub latitude_step_deg: f32,
    pub geodesic_sphere_radius_m: f32,
    pub earth_model: EarthModel,
    pub refraction_model: RefractionModel,
}

#[derive(Debug)]
pub struct ViewshedOutput {
    pub visibility: Vec<bool>,
    pub curvature_drop_m: Vec<f32>,
    pub refraction_gain_m: Vec<f32>,
    pub horizon_distance_m: Vec<f32>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ViewshedUniforms {
    dimensions: [u32; 4],
    observer: [f32; 4],
    metric: [f32; 4],
    physics: [f32; 4],
    geodetic: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ViewshedCell {
    visible: u32,
    curvature_drop_m: f32,
    refraction_gain_m: f32,
    horizon_distance_m: f32,
}

fn physics_terms(options: &ViewshedOptions) -> Result<[f32; 4], String> {
    if matches!(options.earth_model, EarthModel::Flat)
        && !matches!(options.refraction_model, RefractionModel::None)
    {
        return Err("flat earth only supports refraction_model='none'".into());
    }
    let one_minus_k = 1.0 - options.refraction_model.k()?;
    let (inv_meridional, inv_prime_vertical) = match options.earth_model {
        EarthModel::Flat => (0.0, 0.0),
        EarthModel::Sphere { radius_m } if radius_m.is_finite() && radius_m > 0.0 => {
            (radius_m.recip(), radius_m.recip())
        }
        EarthModel::Sphere { .. } => return Err("sphere radius must be finite and positive".into()),
        EarthModel::Ellipsoid { latitude_deg } => {
            let (meridional, prime_vertical) = principal_radii_m(latitude_deg)?;
            (meridional.recip(), prime_vertical.recip())
        }
    };
    Ok([
        inv_meridional as f32,
        inv_prime_vertical as f32,
        one_minus_k as f32,
        f32::from(!matches!(options.earth_model, EarthModel::Flat)),
    ])
}

fn validate_common(
    heights: &[f32],
    position_count: usize,
    positions_are_finite: bool,
    options: &ViewshedOptions,
) -> Result<(), String> {
    let expected = options.width as usize * options.height as usize;
    // terrain_pack_node reserves 13 bits per cell coordinate, matching the
    // production PROMETHEUS traversal contract used by the shared shader.
    const MAX_PACKED_CELL_COUNT: u32 = 1 << 13;
    if options.width < 2
        || options.height < 2
        || options.width - 1 > MAX_PACKED_CELL_COUNT
        || options.height - 1 > MAX_PACKED_CELL_COUNT
        || heights.len() != expected
        || position_count != expected
    {
        return Err(format!(
            "DEM/position lengths do not match supported dimensions {}x{} (both dimensions must be at least 2 and packed traversal supports at most 8192 cells per axis)",
            options.width,
            options.height
        ));
    }
    if heights.iter().any(|height| !height.is_finite()) || !positions_are_finite {
        return Err(format!(
            "DEM heights and geodesic positions must be finite ({} heights)",
            heights.len(),
        ));
    }
    let finite = [
        options.observer_x,
        options.observer_y,
        options.observer_height_m,
        options.target_height_m,
        options.max_distance_m,
        options.observer_latitude_rad,
        options.observer_longitude_rad,
        options.left_unwrapped_deg,
        options.top_deg,
        options.longitude_step_deg,
        options.latitude_step_deg,
        options.geodesic_sphere_radius_m,
    ]
    .iter()
    .all(|value| value.is_finite());
    if !finite
        || options.observer_x < -0.5
        || options.observer_x > options.width as f32 - 0.5
        || options.observer_y < -0.5
        || options.observer_y > options.height as f32 - 0.5
        || options.observer_height_m < 0.0
        || options.target_height_m < 0.0
        || options.max_distance_m <= 0.0
        || options.longitude_step_deg <= 0.0
        || options.latitude_step_deg <= 0.0
        || options.geodesic_sphere_radius_m < 0.0
    {
        return Err(
            "viewshed dimensions, observer, heights, spacing, and distance are invalid".into(),
        );
    }
    physics_terms(options)?;
    Ok(())
}

fn validate(
    heights: &[f32],
    positions_m: &[[f32; 2]],
    options: &ViewshedOptions,
) -> Result<(), String> {
    validate_common(
        heights,
        positions_m.len(),
        positions_m
            .iter()
            .flatten()
            .all(|coordinate| coordinate.is_finite()),
        options,
    )
}

fn compute_visibility(
    heights: &[f32],
    positions_m: &[[f32; 2]],
    options: &ViewshedOptions,
) -> RenderResult<ViewshedOutput> {
    validate(heights, positions_m, options).map_err(RenderError::render)?;
    let physics = physics_terms(options).map_err(RenderError::render)?;
    let context = crate::core::gpu::try_ctx()?;
    let device = &context.device;
    let queue = &context.queue;
    let uniforms = ViewshedUniforms {
        dimensions: [options.width, options.height, 0, 0],
        observer: [
            options.observer_x,
            options.observer_y,
            options.observer_height_m,
            options.target_height_m,
        ],
        metric: [
            options.max_distance_m,
            options.longitude_step_deg,
            options.latitude_step_deg,
            options.geodesic_sphere_radius_m,
        ],
        physics,
        geodetic: [
            options.observer_latitude_rad,
            options.observer_longitude_rad,
            options.left_unwrapped_deg,
            options.top_deg,
        ],
    };
    // HELIOS reuses PROMETHEUS' tracked DEM/min-max textures. This is the only
    // acceleration structure for the call; there is no flat shadow copy or
    // curvature-specific pyramid.
    let pyramid = TerrainMinMaxPyramid::from_heightfield(
        device,
        queue,
        heights,
        options.width,
        options.height,
    )?;
    let height_view = pyramid
        .height_texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    let minmax_view = pyramid
        .minmax_texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    let uniform_buffer = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("helios.viewshed.uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        },
    )?;
    let position_buffer = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("helios.viewshed.geodesic_positions"),
            contents: bytemuck::cast_slice(positions_m),
            usage: wgpu::BufferUsages::STORAGE,
        },
    )?;
    let output_size = (heights.len() * std::mem::size_of::<ViewshedCell>()) as u64;
    let output = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("helios.viewshed.output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        },
    )?;
    let readback = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("helios.viewshed.readback"),
            size: output_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        },
    )?;

    let shader_source = shader_source();
    let module = crate::core::shader_registry::create_labeled_shader_module(
        device,
        "helios.viewshed.shader",
        &shader_source,
    );
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("helios.viewshed.bind_group_layout"),
        entries: &analysis_bind_group_layout_entries(2, 3),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("helios.viewshed.pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = crate::core::shader_registry::try_create_compute_pipeline_scoped(
        device,
        &wgpu::ComputePipelineDescriptor {
            label: Some("helios.viewshed.pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "main",
        },
    )
    .map_err(|error| RenderError::render(format!("viewshed pipeline creation failed: {error}")))?;
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("helios.viewshed.bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: position_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&height_view),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(&minmax_view),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("helios.viewshed.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("helios.viewshed.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(options.width.div_ceil(8), options.height.div_ceil(8), 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_size);
    queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|error| RenderError::readback(format!("viewshed callback failed: {error}")))?
        .map_err(|error| RenderError::readback(format!("viewshed map failed: {error}")))?;
    let mapped = slice.get_mapped_range();
    let cells = bytemuck::cast_slice::<u8, ViewshedCell>(&mapped);
    if cells.iter().any(|cell| cell.visible > 1) {
        drop(mapped);
        readback.unmap();
        return Err(RenderError::render(
            "viewshed geodesic leaves the DEM footprint",
        ));
    }
    let result = ViewshedOutput {
        visibility: cells.iter().map(|cell| cell.visible != 0).collect(),
        curvature_drop_m: cells.iter().map(|cell| cell.curvature_drop_m).collect(),
        refraction_gain_m: cells.iter().map(|cell| cell.refraction_gain_m).collect(),
        horizon_distance_m: cells.iter().map(|cell| cell.horizon_distance_m).collect(),
    };
    drop(mapped);
    readback.unmap();
    Ok(result)
}

pub fn compute_viewshed(
    heights: &[f32],
    positions_m: &[[f32; 2]],
    options: &ViewshedOptions,
) -> RenderResult<ViewshedOutput> {
    compute_visibility(heights, positions_m, options)
}

fn shader_source() -> String {
    format!(
        "{}\n{}",
        include_str!("../../shaders/includes/determinism.wgsl"),
        include_str!("../../shaders/terrain_viewshed.wgsl")
    )
}

fn analysis_bind_group_layout_entries(
    input_binding: u32,
    output_binding: u32,
) -> [wgpu::BindGroupLayoutEntry; 5] {
    let buffer = |binding, ty| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let texture = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    [
        buffer(0, wgpu::BufferBindingType::Uniform),
        buffer(
            input_binding,
            wgpu::BufferBindingType::Storage { read_only: true },
        ),
        buffer(
            output_binding,
            wgpu::BufferBindingType::Storage { read_only: false },
        ),
        texture(6),
        texture(7),
    ]
}

pub fn compute_shadow_mask(
    heights: &[f32],
    geodetic_positions_and_sun_rad: &[[f32; 4]],
    options: &ViewshedOptions,
) -> RenderResult<Vec<bool>> {
    let expected = options.width as usize * options.height as usize;
    if geodetic_positions_and_sun_rad.len() != expected
        || geodetic_positions_and_sun_rad
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
    {
        return Err(RenderError::render(
            "shadow-mask geodetic/solar inputs do not match the DEM",
        ));
    }
    validate_common(heights, geodetic_positions_and_sun_rad.len(), true, options)
        .map_err(RenderError::render)?;
    let physics = physics_terms(options).map_err(RenderError::render)?;
    let context = crate::core::gpu::try_ctx()?;
    let device = &context.device;
    let queue = &context.queue;
    let uniforms = ViewshedUniforms {
        dimensions: [options.width, options.height, 0, 0],
        observer: [
            options.observer_x,
            options.observer_y,
            options.observer_height_m,
            options.target_height_m,
        ],
        metric: [
            options.max_distance_m,
            options.longitude_step_deg,
            options.latitude_step_deg,
            options.geodesic_sphere_radius_m,
        ],
        physics,
        geodetic: [
            options.observer_latitude_rad,
            options.observer_longitude_rad,
            options.left_unwrapped_deg,
            options.top_deg,
        ],
    };
    let pyramid = TerrainMinMaxPyramid::from_heightfield(
        device,
        queue,
        heights,
        options.width,
        options.height,
    )?;
    let height_view = pyramid
        .height_texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    let minmax_view = pyramid
        .minmax_texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    let uniform_buffer = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("helios.shadow_mask.uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        },
    )?;
    let input_buffer = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("helios.shadow_mask.inputs"),
            contents: bytemuck::cast_slice(geodetic_positions_and_sun_rad),
            usage: wgpu::BufferUsages::STORAGE,
        },
    )?;
    let word_count = expected.div_ceil(32);
    let zero_words = vec![0u32; word_count];
    let output_size = (word_count * std::mem::size_of::<u32>()) as u64;
    let output = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("helios.shadow_mask.output"),
            contents: bytemuck::cast_slice(&zero_words),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        },
    )?;
    let readback = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("helios.shadow_mask.readback"),
            size: output_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        },
    )?;
    let shader_source = shader_source();
    let module = crate::core::shader_registry::create_labeled_shader_module(
        device,
        "helios.shadow_mask.shader",
        &shader_source,
    );
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("helios.shadow_mask.bind_group_layout"),
        entries: &analysis_bind_group_layout_entries(4, 5),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("helios.shadow_mask.pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = crate::core::shader_registry::try_create_compute_pipeline_scoped(
        device,
        &wgpu::ComputePipelineDescriptor {
            label: Some("helios.shadow_mask.pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "shadow_mask_main",
        },
    )
    .map_err(|error| {
        RenderError::render(format!("shadow-mask pipeline creation failed: {error}"))
    })?;
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("helios.shadow_mask.bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&height_view),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(&minmax_view),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("helios.shadow_mask.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("helios.shadow_mask.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(options.width.div_ceil(8), options.height.div_ceil(8), 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_size);
    queue.submit(Some(encoder.finish()));
    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|error| RenderError::readback(format!("shadow-mask callback failed: {error}")))?
        .map_err(|error| RenderError::readback(format!("shadow-mask map failed: {error}")))?;
    let mapped = slice.get_mapped_range();
    let words = bytemuck::cast_slice::<u8, u32>(&mapped);
    let visibility = (0..expected)
        .map(|index| words[index / 32] & (1u32 << (index % 32)) != 0)
        .collect();
    drop(mapped);
    readback.unmap();
    Ok(visibility)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_fxc_compatible_traversal(module: &naga::Module) {
        for (_, function) in module.functions.iter().filter(|(_, function)| {
            matches!(
                function.name.as_deref(),
                Some("terrain_leaf_occluded" | "terrain_select_child" | "terrain_trace_segment")
            )
        }) {
            for (_, local) in function.local_variables.iter() {
                assert!(
                    !matches!(module.types[local.ty].inner, naga::TypeInner::Array { .. }),
                    "FXC cannot compile dynamically indexed function arrays in HELIOS traversal"
                );
            }
        }
    }

    fn leaf_segment_occluded(
        heights: [f32; 4],
        origin: [f32; 2],
        direction: [f32; 2],
        height_limit: f32,
    ) -> bool {
        let mut deviation = [0.0; 3];
        for (index, parameter) in [0.0_f32, 0.5, 1.0].into_iter().enumerate() {
            let u = (origin[0] + parameter * direction[0]).clamp(0.0, 1.0);
            let v = (origin[1] + parameter * direction[1]).clamp(0.0, 1.0);
            let low = heights[0] + (heights[1] - heights[0]) * u;
            let high = heights[2] + (heights[3] - heights[2]) * u;
            deviation[index] = low + (high - low) * v - height_limit;
        }
        let quadratic = 2.0 * deviation[2] + 2.0 * deviation[0] - 4.0 * deviation[1];
        let linear = deviation[2] - deviation[0] - quadratic;
        let mut maximum = deviation[0].max(deviation[2]);
        if quadratic.abs() > 1e-12 {
            let vertex = -linear / (2.0 * quadratic);
            if (0.0..1.0).contains(&vertex) {
                maximum = maximum.max(quadratic * vertex * vertex + linear * vertex + deviation[0]);
            }
        }
        maximum > 0.0
    }

    #[test]
    fn viewshed_validation_rejects_bad_models_and_dimensions() {
        let mut options = ViewshedOptions {
            width: 2,
            height: 2,
            observer_x: 0.0,
            observer_y: 0.0,
            observer_height_m: 1.7,
            target_height_m: 0.0,
            max_distance_m: 10.0,
            observer_latitude_rad: 0.0,
            observer_longitude_rad: 0.0,
            left_unwrapped_deg: 0.0,
            top_deg: 1.0,
            longitude_step_deg: 1.0,
            latitude_step_deg: 1.0,
            geodesic_sphere_radius_m: 0.0,
            earth_model: EarthModel::Flat,
            refraction_model: RefractionModel::None,
        };
        assert!(validate(&[0.0; 4], &[[0.0; 2]; 4], &options).is_ok());
        options.width = 1;
        assert!(validate(&[0.0; 2], &[[0.0; 2]; 2], &options).is_err());
        options.width = 2;
        options.refraction_model = RefractionModel::EffectiveRadius { k: 1.0 };
        assert!(validate(&[0.0; 4], &[[0.0; 2]; 4], &options).is_err());
        options.refraction_model = RefractionModel::Bennett {
            pressure_mbar: 1013.25,
            temperature_c: 15.0,
        };
        assert_eq!(
            physics_terms(&options).unwrap_err(),
            "flat earth only supports refraction_model='none'"
        );
    }

    #[test]
    fn continuous_leaf_test_detects_between_endpoint_blocker() {
        // Along the diagonal this bilinear saddle is zero at both endpoints
        // and 50 m at the midpoint. Endpoint sampling misses the 25 m ray;
        // the exact production leaf polynomial must report an occluder.
        let heights = [0.0, 100.0, 100.0, 0.0];
        assert!(heights[0] < 25.0 && heights[3] < 25.0);
        assert!(leaf_segment_occluded(heights, [0.0, 0.0], [1.0, 1.0], 25.0));
    }

    #[test]
    fn viewshed_shader_is_valid_wgsl() {
        let source = shader_source();
        for production_primitive in [
            "const TERRAIN_INVALID_NODE: u32 = 0xFFFFFFFFu",
            "fn terrain_slab_xz(",
            "fn terrain_leaf_occluded(",
            "fn terrain_select_child(",
            "fn terrain_trace_segment(",
            "textureNumLevels(minmax_texture)",
        ] {
            assert!(
                source.contains(production_primitive),
                "viewshed shader must retain production traversal primitive {production_primitive}"
            );
        }
        assert!(!source.contains("fn minmax_may_exceed("));
        assert_eq!(source.matches("terrain_trace_segment(").count(), 3);
        let module = naga::front::wgsl::parse_str(&source)
            .expect("assembled HELIOS viewshed WGSL must parse");
        assert_fxc_compatible_traversal(&module);
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("assembled HELIOS viewshed WGSL must validate");
    }

    #[test]
    #[should_panic(
        expected = "FXC cannot compile dynamically indexed function arrays in HELIOS traversal"
    )]
    fn fxc_traversal_shape_gate_rejects_child_selection_array_mutant() {
        let source = shader_source().replace(
            "let cell_width = uniforms.dimensions.x - 1u;\n    let cell_height = uniforms.dimensions.y - 1u;\n    let child_level = parent_level - 1u;",
            "let cell_width = uniforms.dimensions.x - 1u;\n    let cell_height = uniforms.dimensions.y - 1u;\n    var fxc_mutant: array<u32, 4u>;\n    let child_level = parent_level - 1u;",
        );
        let module = naga::front::wgsl::parse_str(&source).expect("mutant must remain valid WGSL");
        assert_fxc_compatible_traversal(&module);
    }

    #[test]
    fn analysis_entry_points_use_explicit_resource_layouts() {
        let viewshed_entries = analysis_bind_group_layout_entries(2, 3);
        assert_eq!(
            viewshed_entries.map(|entry| entry.binding),
            [0, 2, 3, 6, 7],
            "the viewshed layout must contain exactly the bindings used by main"
        );
        let entries = analysis_bind_group_layout_entries(4, 5);
        assert_eq!(
            entries.map(|entry| entry.binding),
            [0, 4, 5, 6, 7],
            "the shadow-mask layout must contain exactly the bindings used by shadow_mask_main"
        );
        assert!(matches!(
            entries[0].ty,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                ..
            }
        ));
        assert!(matches!(
            entries[1].ty,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                ..
            }
        ));
        assert!(matches!(
            entries[2].ty,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                ..
            }
        ));
        for entry in &entries[3..] {
            assert!(matches!(
                entry.ty,
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }
            ));
        }
    }
}
