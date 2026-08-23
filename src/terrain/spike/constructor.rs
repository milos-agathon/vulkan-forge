use super::*;

#[pymethods]
impl TerrainSpike {
    #[new]
    #[pyo3(text_signature = "(width, height, grid=128, colormap='viridis')")]
    pub fn new(
        width: u32,
        height: u32,
        grid: Option<u32>,
        colormap: Option<String>,
    ) -> PyResult<Self> {
        let grid = grid.unwrap_or(128).max(2);

        let colormap_name = colormap.as_deref().unwrap_or("viridis");

        // Validate colormap against central SUPPORTED list
        if !SUPPORTED.contains(&colormap_name) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Unknown colormap '{}'. Supported: {}",
                colormap_name,
                SUPPORTED.join(", ")
            )));
        }

        let which = map_name_to_type(colormap_name).map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Unknown colormap '{}'. Supported: {}",
                colormap_name,
                SUPPORTED.join(", ")
            ))
        })?;

        // Instance/adapter/device
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No suitable GPU adapter"))?;

        let mut capabilities = crate::core::capabilities::CapabilitySet::negotiate(
            adapter.features(),
            adapter.get_info().backend,
        );
        let requested_limits = adapter.limits();
        let first_request = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("terrain-device"),
                required_features: capabilities.granted,
                required_limits: requested_limits.clone(),
            },
            None,
        ));
        let (device, queue) = match first_request {
            Ok(pair) => pair,
            Err(error) => {
                capabilities.downgrade_after_request_failure(&error.to_string());
                pollster::block_on(adapter.request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("terrain-device"),
                        required_features: capabilities.granted,
                        required_limits: requested_limits,
                    },
                    None,
                ))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            }
        };
        let device = std::sync::Arc::new(device);
        let queue = std::sync::Arc::new(queue);

        // Offscreen color + depth
        let color = tracked_create_texture(
            &device,
            &wgpu::TextureDescriptor {
                label: Some("terrain-color"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TEXTURE_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )?;
        let color_view = color.create_view(&Default::default());
        let normal = tracked_create_texture(
            &device,
            &wgpu::TextureDescriptor {
                label: Some("terrain-normal"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: NORMAL_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        )?;
        let normal_view = normal.create_view(&Default::default());

        // Shader + pipeline - using T33 shared pipeline

        // T33-BEGIN:remove-local-ubo-layout
        // Removed local, conflated layout; using shared tp.{bgl_globals,bgl_height,bgl_lut}
        // T33-END:remove-local-ubo-layout

        // T33-BEGIN:terrainspike-use-t33
        // Use shared T33 pipeline
        let init_height_filterable = device
            .features()
            .contains(wgpu::Features::FLOAT32_FILTERABLE);
        let tp = crate::terrain::pipeline::TerrainPipeline::create(
            &device,
            TEXTURE_FORMAT,
            NORMAL_FORMAT,
            1,
            None,
            init_height_filterable,
        );
        // T33-END:terrainspike-use-t33

        // Mesh + uniforms
        let (vbuf, ibuf, nidx) = build_grid_xyuv(&device, grid)?;
        let (view, proj, light) = build_view_matrices(width, height);

        let mut globals = Globals::default();
        // R4: Seed globals.sun_dir from computed light
        globals.sun_dir = light;
        // Use globals (with h_min/h_max) -> h_range is computed inside to_uniforms()
        let uniforms = globals.to_uniforms(view, proj);

        let ubo_usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let uniform_size = std::mem::size_of::<TerrainUniforms>() as u64;

        // Runtime debug assertion to ensure uniform buffer matches WGSL expectations
        debug_assert_eq!(
            uniform_size, 176,
            "Uniform buffer size {} doesn't match WGSL expectation {}",
            uniform_size, 176
        );

        let ubo = tracked_create_buffer_init(
            &device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain-ubo"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: ubo_usage,
            },
        )?;

        // E2: Create a default tile uniforms buffer
        let tile_init = TileUniformsCPU {
            world_remap: [globals.spacing, globals.spacing, 0.0, 0.0],
        };
        let tile_ubo = tracked_create_buffer_init(
            &device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain-tile-ubo"),
                contents: bytemuck::bytes_of(&tile_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;

        // E1b: Create default tile slot and mosaic params UBOs
        let tile_slot_init = TileSlotCPU {
            lod: 0,
            x: 0,
            y: 0,
            slot: 0,
        };
        let tile_slot_ubo = tracked_create_buffer_init(
            &device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain-tile-slot-ubo"),
                contents: bytemuck::bytes_of(&tile_slot_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;
        let mosaic_params_init = MosaicParamsCPU {
            inv_tiles_x: 1.0,
            inv_tiles_y: 1.0,
            tiles_x: 1,
            tiles_y: 1,
        };
        let mosaic_params_ubo = tracked_create_buffer_init(
            &device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain-mosaic-params-ubo"),
                contents: bytemuck::bytes_of(&mosaic_params_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;

        // tracked_create_buffer_init already accounts for the UBO allocation;
        // no manual track_buffer_allocation here (it would double-count).

        let (lut, lut_format) = ColormapLUT::new(&device, &queue, &adapter, which)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // T33-BEGIN:bg1-height-dummy
        // Provide a tiny dummy height if the spike has none yet (keeps validation clean)
        let (hview, hsamp) = {
            let tex = tracked_create_texture(
                &device,
                &wgpu::TextureDescriptor {
                    label: Some("dummy-height-r32f"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
            )?;
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::bytes_of(&0.0f32),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(std::num::NonZeroU32::new(4).unwrap().into()),
                    rows_per_image: Some(std::num::NonZeroU32::new(1).unwrap().into()),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            let view = tex.create_view(&Default::default());
            // Sampler type matches current pipeline expectation (filterable vs non-filterable)
            let samp = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("dummy-height-sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: if init_height_filterable {
                    wgpu::FilterMode::Linear
                } else {
                    wgpu::FilterMode::Nearest
                },
                min_filter: if init_height_filterable {
                    wgpu::FilterMode::Linear
                } else {
                    wgpu::FilterMode::Nearest
                },
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });
            (view, samp)
        };

        // Create bind groups
        let bg0_globals = tp.make_bg_globals(&device, &ubo);
        let bg1_height = tp.make_bg_height(&device, &hview, &hsamp);
        let bg2_lut = tp.make_bg_lut(&device, &lut.view, &lut.sampler);
        let bg5_tile =
            tp.make_bg_tile(&device, &tile_ubo, None, &tile_slot_ubo, &mosaic_params_ubo)?;

        return Ok(Self {
            width,
            height,
            _grid: grid,
            device,
            queue,
            tp,
            bg0_globals,
            bg1_height,
            bg2_lut,
            vbuf,
            ibuf,
            nidx,
            ubo,
            tile_ubo,
            tile_slot_ubo,
            mosaic_params_ubo,
            colormap_lut: lut,
            lut_format,
            color,
            color_view,
            _normal: normal,
            normal_view,
            globals,
            last_uniforms: uniforms,
            height_view: Some(hview),
            height_sampler: Some(hsamp),
            height_filterable: init_height_filterable,
            tiling_system: None,
            height_mosaic: None,
            overlay_mosaic: None,
            overlay_renderer: None,
            bg5_tile,
            page_table: None,
            async_loader: None,
            async_overlay_loader: None,
            prev_visible_height: HashSet::new(),
            prev_visible_overlay: HashSet::new(),
        });
    }
}
