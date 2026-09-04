impl Scene {
    pub(super) fn new_impl(
        width: u32,
        height: u32,
        grid: Option<u32>,
        colormap: Option<String>,
    ) -> PyResult<Self> {
        let allocation_owner = crate::core::resource_tracker::AllocationOwner::new();
        let _allocation_scope = allocation_owner.activate();
        let grid = grid.unwrap_or(128).max(2);
        let g = crate::core::gpu::try_ctx()?;

        let sample_count = 1;
        let (color, color_view) = create_color_texture(&g.device, width, height)?;
        let (normal, normal_view) = create_normal_texture(&g.device, width, height)?;
        let (msaa_color, msaa_view) = create_msaa_targets(&g.device, width, height, sample_count)?;
        let (msaa_normal, msaa_normal_view) =
            create_msaa_normal_targets(&g.device, width, height, sample_count)?;
        let (depth, depth_view) = create_depth_target(&g.device, width, height, sample_count)?;

        let depth_format = if sample_count > 1 {
            Some(wgpu::TextureFormat::Depth32Float)
        } else {
            None
        };

        let height_filterable = g
            .device
            .features()
            .contains(wgpu::Features::FLOAT32_FILTERABLE);
        let tp = crate::terrain::pipeline::TerrainPipeline::create(
            &g.device,
            TEXTURE_FORMAT,
            NORMAL_FORMAT,
            sample_count,
            depth_format,
            height_filterable,
        );

        let ssao = match SsaoResources::new(&g.device, &g.queue, width, height, &color, &normal) {
            Ok(ssao) => ssao,
            Err(_) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "SSAO system temporarily disabled due to pipeline validation issues",
                ));
            }
        };

        let (vbuf, ibuf, nidx) = create_grid_buffers(&g.device, grid)?;

        let mut scene = SceneGlobals::default();
        scene.proj = crate::camera::perspective_wgpu(
            45f32.to_radians(),
            width as f32 / height as f32,
            0.1,
            100.0,
        );
        let uniforms = scene.globals.to_uniforms(scene.view, scene.proj);
        let ubo = tracked_create_buffer_init(
            &g.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("scene-ubo"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;

        let cmap_name = colormap.as_deref().unwrap_or("viridis");
        if !crate::colormap::SUPPORTED.contains(&cmap_name) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Unknown colormap '{}'. Supported: {}",
                cmap_name,
                crate::colormap::SUPPORTED.join(", ")
            )));
        }
        let which = crate::colormap::map_name_to_type(cmap_name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let (lut, lut_format) =
            crate::terrain::ColormapLUT::new(&g.device, &g.queue, &g.adapter, which)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let (hview, hsamp) = create_dummy_height_texture(&g.device, &g.queue)?;

        let bg0_globals = tp.make_bg_globals(&g.device, &ubo);
        let bg1_height = tp.make_bg_height(&g.device, &hview, &hsamp);
        let bg2_lut = tp.make_bg_lut(&g.device, &lut.view, &lut.sampler);

        let tile_world_remap: [f32; 4] = [1.0, 1.0, 0.0, 0.0];
        let tile_ubo = tracked_create_buffer_init(
            &g.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("scene.tile_ubo"),
                contents: bytemuck::cast_slice(&tile_world_remap),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;
        let zero16 = [0u8; 16];
        let tile_slot_ubo = tracked_create_buffer_init(
            &g.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("scene.tile_slot_ubo"),
                contents: &zero16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;
        let mosaic_params_ubo = tracked_create_buffer_init(
            &g.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("scene.mosaic_params_ubo"),
                contents: &zero16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;
        let bg3_tile = tp.make_bg_tile(
            &g.device,
            &tile_ubo,
            None,
            &tile_slot_ubo,
            &mosaic_params_ubo,
        )?;

        let bg4_dummy_cloud_shadows = create_dummy_cloud_shadow_bind_group(&tp, &g.device)?;

        let mut reflection_renderer = crate::core::reflections::PlanarReflectionRenderer::new(
            &g.device,
            crate::core::reflections::ReflectionQuality::Low,
        )?;
        reflection_renderer.set_enabled(false);
        reflection_renderer.create_bind_group(&g.device, &tp.bgl_reflection);
        reflection_renderer.upload_uniforms(&g.queue);

        let mut overlay_renderer = crate::core::overlays::OverlayRenderer::new(
            &g.device,
            TEXTURE_FORMAT,
            height_filterable,
        )?;
        overlay_renderer.recreate_bind_group(&g.device, None, Some(&hview), None, None)?;
        overlay_renderer.upload_uniforms(&g.queue);

        let mut text_renderer =
            crate::core::text_overlay::TextOverlayRenderer::new(&g.device, TEXTURE_FORMAT)?;
        text_renderer.set_resolution(width, height);
        text_renderer.set_alpha(1.0);
        text_renderer.set_enabled(false);
        text_renderer.upload_uniforms(&g.queue);

        let mut text3d_renderer =
            crate::core::text_mesh::TextMeshRenderer::new(&g.device, TEXTURE_FORMAT, depth_format)?;
        text3d_renderer.set_view_proj(scene.view, scene.proj);
        text3d_renderer.set_color(1.0, 1.0, 1.0, 1.0);
        text3d_renderer.set_light_dir([0.5, 1.0, 0.3]);
        text3d_renderer.upload_uniforms(&g.queue);

        Ok(Self {
            width,
            height,
            grid,
            tp,
            bg0_globals,
            bg1_height,
            bg2_lut,
            bg3_tile,
            _tile_ubo: tile_ubo,
            _tile_slot_ubo: tile_slot_ubo,
            _mosaic_params_ubo: mosaic_params_ubo,
            vbuf,
            ibuf,
            nidx,
            ubo,
            colormap: lut,
            lut_format,
            color,
            color_view,
            normal,
            normal_view,
            sample_count,
            msaa_color,
            msaa_view,
            msaa_normal,
            msaa_normal_view,
            depth,
            depth_view,
            height_view: Some(hview),
            height_sampler: Some(hsamp),
            scene,
            camera_anchor: crate::camera::Anchor::new(),
            last_uniforms: uniforms,
            ssao,
            ssao_enabled: false,
            ssgi_enabled: false,
            ssgi_settings: crate::lighting::screen_space::SSGISettings::default(),
            ssr_enabled: false,
            ssr_settings: crate::lighting::screen_space::SSRSettings::default(),
            #[cfg(feature = "extension-module")]
            atmosphere_state: None,
            bloom_enabled: false,
            bloom_config: crate::core::bloom::BloomConfig::default(),
            terrain_enabled: true,
            reflection_renderer: Some(reflection_renderer),
            reflections_enabled: false,
            dof_renderer: None,
            dof_enabled: false,
            dof_params: crate::core::dof::CameraDofParams::default(),
            cloud_shadow_renderer: None,
            cloud_shadows_enabled: false,
            bg3_cloud_shadows: None,
            bg4_dummy_cloud_shadows,
            cloud_renderer: None,
            clouds_enabled: false,
            ground_plane_renderer: None,
            ground_plane_enabled: false,
            water_surface_renderer: None,
            water_surface_enabled: false,
            soft_light_radius_renderer: None,
            soft_light_radius_enabled: false,
            point_spot_lights_renderer: None,
            point_spot_lights_enabled: false,
            ltc_area_lights_renderer: None,
            ltc_area_lights_enabled: false,
            ibl_renderer: None,
            ibl_enabled: false,
            dual_source_oit_renderer: None,
            dual_source_oit_enabled: false,
            overlay_renderer: Some(overlay_renderer),
            overlay_enabled: false,
            text_overlay_renderer: Some(text_renderer),
            text_overlay_enabled: false,
            text_overlay_alpha: 1.0,
            text_instances: Vec::new(),
            text3d_renderer: Some(text3d_renderer),
            text3d_enabled: false,
            text3d_instances: Vec::new(),
            render_timing: std::sync::Mutex::new(None),
            allocation_owner,
            #[cfg(feature = "enable-gpu-instancing")]
            mesh_instanced_renderer: Some(
                crate::render::mesh_instanced::MeshInstancedRenderer::new(
                    &g.device,
                    TEXTURE_FORMAT,
                    depth_format,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
            ),
            #[cfg(feature = "enable-gpu-instancing")]
            instanced_batches: Vec::new(),
        })
    }
}
