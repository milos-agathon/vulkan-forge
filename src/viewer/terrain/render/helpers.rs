use super::*;
use crate::core::ibl::{IBLQuality, IBLRenderer};
use crate::core::resource_tracker::{tracked_create_buffer, tracked_create_texture};
use crate::viewer::terrain::overlay::OverlayStack;
use half::f16;

impl ViewerTerrainScene {
    fn clear_terrain_ibl(&mut self) {
        self.terrain_ibl_renderer = None;
        self.terrain_ibl_hdr_path = None;
        self.terrain_ibl_specular_view = None;
        self.terrain_ibl_irradiance_view = None;
        self.terrain_ibl_brdf_view = None;
        self.terrain_ibl_specular_mip_count = 1;
    }

    fn ensure_terrain_ibl_sampler(&mut self) {
        if self.terrain_ibl_sampler.is_none() {
            self.terrain_ibl_sampler = Some(self.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("terrain_viewer_pbr.ibl_sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }));
        }
    }

    fn ensure_terrain_ibl_fallback_resources(&mut self) -> anyhow::Result<()> {
        self.ensure_terrain_ibl_sampler();

        if self.terrain_ibl_fallback_cube.is_none() {
            let cube = tracked_create_texture(
                &self.device,
                &wgpu::TextureDescriptor {
                    label: Some("terrain_viewer_pbr.ibl_fallback_cube"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 6,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
            )?;
            let zero = f16::from_f32(0.0).to_bits();
            let one = f16::from_f32(1.0).to_bits();
            let mut texels = Vec::with_capacity(24);
            for _ in 0..6 {
                texels.extend_from_slice(&[zero, zero, zero, one]);
            }
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &cube,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&texels),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(8),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 6,
                },
            );
            self.terrain_ibl_fallback_cube_view =
                Some(cube.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("terrain_viewer_pbr.ibl_fallback_cube_view"),
                    format: Some(wgpu::TextureFormat::Rgba16Float),
                    dimension: Some(wgpu::TextureViewDimension::Cube),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(6),
                }));
            self.terrain_ibl_fallback_cube = Some(cube);
        }

        if self.terrain_ibl_fallback_brdf.is_none() {
            let brdf = tracked_create_texture(
                &self.device,
                &wgpu::TextureDescriptor {
                    label: Some("terrain_viewer_pbr.ibl_fallback_brdf"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
            )?;
            let zero = f16::from_f32(0.0).to_bits();
            let one = f16::from_f32(1.0).to_bits();
            let texels = [zero, zero, zero, one];
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &brdf,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&texels),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(8),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            self.terrain_ibl_fallback_brdf_view =
                Some(brdf.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("terrain_viewer_pbr.ibl_fallback_brdf_view"),
                    ..Default::default()
                }));
            self.terrain_ibl_fallback_brdf = Some(brdf);
        }
        Ok(())
    }

    fn load_terrain_ibl(&mut self, path: &std::path::Path) -> anyhow::Result<()> {
        let hdr_img = crate::formats::hdr::load_hdr(path)
            .map_err(|e| anyhow::anyhow!("failed to load HDR '{}': {}", path.display(), e))?;
        let quality = IBLQuality::Low;
        let mut ibl = IBLRenderer::new(&self.device, quality)?;
        ibl.set_base_resolution(quality.base_environment_size());
        ibl.load_environment_map(
            &self.device,
            &self.queue,
            &hdr_img.data,
            hdr_img.width,
            hdr_img.height,
        )
        .map_err(anyhow::Error::msg)?;
        ibl.initialize(&self.device, &self.queue)
            .map_err(anyhow::Error::msg)?;

        let (irr_tex, spec_tex, brdf_tex) = ibl.textures();
        let irr_tex = irr_tex.ok_or_else(|| anyhow::anyhow!("missing irradiance cube map"))?;
        let spec_tex = spec_tex.ok_or_else(|| anyhow::anyhow!("missing specular cube map"))?;
        let brdf_tex = brdf_tex.ok_or_else(|| anyhow::anyhow!("missing BRDF LUT"))?;
        let specular_mip_count = quality.specular_mip_levels();

        self.terrain_ibl_specular_view = Some(spec_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain_viewer_pbr.ibl_specular_view"),
            format: Some(wgpu::TextureFormat::Rgba16Float),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(specular_mip_count),
            base_array_layer: 0,
            array_layer_count: Some(6),
        }));
        self.terrain_ibl_irradiance_view =
            Some(irr_tex.create_view(&wgpu::TextureViewDescriptor {
                label: Some("terrain_viewer_pbr.ibl_irradiance_view"),
                format: Some(wgpu::TextureFormat::Rgba16Float),
                dimension: Some(wgpu::TextureViewDimension::Cube),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(6),
            }));
        self.terrain_ibl_brdf_view = Some(brdf_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain_viewer_pbr.ibl_brdf_view"),
            ..Default::default()
        }));
        self.terrain_ibl_specular_mip_count = specular_mip_count;
        self.terrain_ibl_hdr_path = Some(path.to_path_buf());
        self.terrain_ibl_renderer = Some(ibl);
        Ok(())
    }

    pub(super) fn ensure_terrain_ibl_resources(&mut self) -> anyhow::Result<()> {
        self.ensure_terrain_ibl_fallback_resources()?;

        let desired_path = self.pbr_config.hdr_path.clone();
        match desired_path {
            Some(path) => {
                let needs_reload = self.terrain_ibl_hdr_path.as_ref() != Some(&path)
                    || self.terrain_ibl_renderer.is_none()
                    || self.terrain_ibl_specular_view.is_none()
                    || self.terrain_ibl_irradiance_view.is_none()
                    || self.terrain_ibl_brdf_view.is_none();
                if needs_reload {
                    if let Err(err) = self.load_terrain_ibl(&path) {
                        eprintln!(
                            "[terrain_pbr] Failed to load HDRI '{}': {}",
                            path.display(),
                            err
                        );
                        self.clear_terrain_ibl();
                    } else {
                        println!("[terrain_pbr] Loaded HDRI '{}'", path.display());
                    }
                }
            }
            None => {
                if self.terrain_ibl_renderer.is_some() || self.terrain_ibl_hdr_path.is_some() {
                    self.clear_terrain_ibl();
                }
            }
        }
        Ok(())
    }

    pub(super) fn terrain_ibl_uniform_params(&self) -> [f32; 4] {
        let theta = self.pbr_config.hdr_rotate_deg.to_radians();
        [
            if self.terrain_ibl_specular_view.is_some()
                && self.terrain_ibl_irradiance_view.is_some()
                && self.terrain_ibl_brdf_view.is_some()
            {
                1.0
            } else {
                0.0
            },
            self.terrain_ibl_specular_mip_count.saturating_sub(1) as f32,
            theta.sin(),
            theta.cos(),
        ]
    }

    /// Prepare PBR bind group with current uniforms (called before render pass)
    /// Gets heightmap_view internally from self.terrain to avoid borrow issues
    pub(super) fn prepare_pbr_bind_group_internal(
        &mut self,
        uniforms: &TerrainPbrUniforms,
    ) -> anyhow::Result<()> {
        // Ensure fallback texture exists first (before any borrows)
        self.ensure_fallback_texture()?;
        self.ensure_terrain_ibl_resources()?;

        // Early return checks
        if self.pbr_bind_group_layout.is_none() || self.terrain.is_none() {
            return Ok(());
        }

        // Create or update uniform buffer
        if self.pbr_uniform_buffer.is_none() {
            self.pbr_uniform_buffer = Some(tracked_create_buffer(
                &self.device,
                &wgpu::BufferDescriptor {
                    label: Some("terrain_viewer_pbr.uniform_buffer"),
                    size: std::mem::size_of::<TerrainPbrUniforms>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            )?);
        }

        // Write uniforms
        if let Some(ref buf) = self.pbr_uniform_buffer {
            self.queue
                .write_buffer(buf, 0, bytemuck::cast_slice(&[*uniforms]));
        }

        // Recreate bind group
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain_viewer_pbr.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Ensure overlay stack exists with fallback texture BEFORE borrowing other fields
        if self.overlay_stack.is_none() {
            self.overlay_stack = Some(OverlayStack::new(self.device.clone(), self.queue.clone()));
        }
        // Rebuild overlay composite if dirty, then ensure fallback exists
        if let Some(ref mut stack) = self.overlay_stack {
            if stack.is_dirty() {
                if let Some(ref terrain) = self.terrain {
                    stack.build_composite(terrain.dimensions.0, terrain.dimensions.1)?;
                }
            }
            stack.ensure_fallback_texture()?;
        }

        // Now borrow everything we need
        let layout = self.pbr_bind_group_layout.as_ref().unwrap();
        let terrain = self.terrain.as_ref().unwrap();
        let fallback_view = self.fallback_texture_view.as_ref().unwrap();
        let ao_view = self.height_ao_view.as_ref().unwrap_or(fallback_view);
        let sv_view = self.sun_vis_view.as_ref().unwrap_or(fallback_view);
        let ibl_specular_view = self
            .terrain_ibl_specular_view
            .as_ref()
            .or(self.terrain_ibl_fallback_cube_view.as_ref())
            .unwrap();
        let ibl_irradiance_view = self
            .terrain_ibl_irradiance_view
            .as_ref()
            .or(self.terrain_ibl_fallback_cube_view.as_ref())
            .unwrap();
        let ibl_brdf_view = self
            .terrain_ibl_brdf_view
            .as_ref()
            .or(self.terrain_ibl_fallback_brdf_view.as_ref())
            .unwrap();
        let ibl_sampler = self.terrain_ibl_sampler.as_ref().unwrap();

        // Get overlay view and sampler from stack
        // ensure_fallback_texture() guarantees composite_view is Some (either actual composite or RGBA fallback)
        let overlay_stack = self.overlay_stack.as_ref().unwrap();
        let overlay_view = overlay_stack
            .composite_view()
            .expect("overlay composite_view should exist after ensure_fallback_texture");
        let overlay_sampler = if self.pbr_config.overlay.preserve_colors {
            &sampler
        } else {
            overlay_stack.sampler()
        };

        // Get CSM shadow resources - create fallbacks if they don't exist
        let (shadow_view, moment_view, shadow_sampler) =
            if let Some(csm) = self.csm_renderer.as_ref() {
                (
                    csm.shadow_texture_view(),
                    csm.moment_binding_view(),
                    &csm.shadow_sampler,
                )
            } else {
                eprintln!("[ERROR] CSM renderer not initialized - cannot create PBR bind group");
                return Ok(());
            };

        // Moment sampler (Filtering)
        // We can use the existing pbr sampler (linear) or create a new one.
        // csm_renderer doesn't expose a dedicated moment sampler, but it uses Linear/Linear.
        // Let's use the one we created above 'sampler' which is Linear/Nearest/Clamp.
        // Or better, creating a dedicated one matching CSM requirements.

        let moment_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("csm.moment.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let csm_buffer = if let Some(buf) = &self.csm_uniform_buffer {
            buf
        } else {
            return Ok(());
        };

        // P6.2: Write CSM uniforms with technique value from pbr_config
        let shadows_enabled = !self
            .pbr_config
            .shadow_technique
            .eq_ignore_ascii_case("none");
        let technique = ShadowTechnique::from_name(&self.pbr_config.shadow_technique)
            .filter(|technique| *technique != ShadowTechnique::CSM)
            .unwrap_or(ShadowTechnique::PCF);

        // Build CSM uniforms with current technique
        // Use cascade data from CSM renderer if shadow depth passes have been rendered
        let debug_mode = self.pbr_config.debug_mode;
        let technique_params = super::super::shader_pbr::default_viewer_pcss_technique_params();
        let csm_uniforms = if let Some(ref csm) = self.csm_renderer {
            // Copy uniforms from CSM renderer (populated by render_shadow_passes)
            let mut u = csm.uniforms;
            u.technique = technique.as_u32();
            if !shadows_enabled {
                u.cascade_count = 0;
            }
            u.pcf_kernel_size = match technique {
                ShadowTechnique::Hard => 1,
                ShadowTechnique::PCSS => 5,
                _ => 3,
            };
            u.technique_params = technique_params;
            u.debug_mode = debug_mode; // P6.2: Debug visualization from pbr_config
            u
        } else {
            // Fallback: no shadow depth passes, cascade_count=0 triggers soft shadow fallback
            let mut u = CsmUniforms::default();
            u.technique = technique.as_u32();
            u.cascade_count = 0;
            u.shadow_map_size = self.pbr_config.shadow_map_res as f32;
            u.pcf_kernel_size = match technique {
                ShadowTechnique::Hard => 1,
                ShadowTechnique::PCSS => 5,
                _ => 3,
            };
            u.depth_bias = 0.005;
            u.slope_bias = 0.01;
            u.peter_panning_offset = 0.001;
            u.evsm_positive_exp = crate::shadows::EVSM_MAX_EXPONENT_RGBA16F;
            u.evsm_negative_exp = crate::shadows::EVSM_MAX_EXPONENT_RGBA16F;
            u.technique_params = technique_params;
            u.debug_mode = debug_mode; // P6.2: Debug visualization from pbr_config

            // Set up default cascade far distances for cascade selection
            let terrain_scale = terrain.dimensions.0.max(terrain.dimensions.1) as f32;
            let base_distance = terrain_scale * 0.1;
            for (i, cascade) in u.cascades.iter_mut().enumerate() {
                cascade.far_distance = base_distance * (2.0_f32).powi(i as i32 + 1);
            }
            u
        };

        // Write CSM uniforms to buffer
        // Debug: log uniform values
        static CSM_ONCE: std::sync::Once = std::sync::Once::new();
        CSM_ONCE.call_once(|| {
            println!(
                "[csm_uniforms] cascade_count={}, technique={}, shadow_map_size={}",
                csm_uniforms.cascade_count, csm_uniforms.technique, csm_uniforms.shadow_map_size
            );
            for (i, c) in csm_uniforms
                .cascades
                .iter()
                .enumerate()
                .take(csm_uniforms.cascade_count as usize)
            {
                println!(
                    "[csm_uniforms] cascade[{}] far_distance={:.1}",
                    i, c.far_distance
                );
            }
        });
        self.queue
            .write_buffer(csm_buffer, 0, bytemuck::cast_slice(&[csm_uniforms]));

        if let Some(ref buf) = self.pbr_uniform_buffer {
            self.pbr_bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("terrain_viewer_pbr.bind_group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&terrain.heightmap_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(ao_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(sv_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(overlay_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(overlay_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(&shadow_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::Sampler(shadow_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::TextureView(&moment_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::Sampler(&moment_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: csm_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: wgpu::BindingResource::TextureView(ibl_specular_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: wgpu::BindingResource::TextureView(ibl_irradiance_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 14,
                        resource: wgpu::BindingResource::Sampler(ibl_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 15,
                        resource: wgpu::BindingResource::TextureView(ibl_brdf_view),
                    },
                ],
            }));
        }
        Ok(())
    }

    /// Ensure fallback 1x1 white texture exists for when AO/sun_vis are disabled
    pub(super) fn ensure_fallback_texture(&mut self) -> anyhow::Result<()> {
        if self.fallback_texture.is_some() {
            return Ok(());
        }

        let texture = tracked_create_texture(
            &self.device,
            &wgpu::TextureDescriptor {
                label: Some("terrain_viewer.fallback_texture"),
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

        // Write 1.0 (fully lit / no occlusion)
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&[1.0f32]),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        self.fallback_texture_view =
            Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.fallback_texture = Some(texture);
        Ok(())
    }

    /// Dispatch compute passes for heightfield AO and sun visibility
    pub(super) fn dispatch_heightfield_compute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        terrain_span: [f32; 2],
        sun_dir: glam::Vec3,
    ) {
        let terrain = match self.terrain.as_ref() {
            Some(t) => t,
            None => return,
        };
        let (width, height) = terrain.dimensions;
        let terrain_width = terrain_span[0];
        let terrain_depth = terrain_span[1];
        let z_scale = terrain.z_scale;

        // Height AO compute pass
        if self.pbr_config.height_ao.enabled {
            if let (
                Some(ref pipeline),
                Some(ref layout),
                Some(ref uniform_buf),
                Some(ref ao_view),
                Some(ref sampler),
            ) = (
                &self.height_ao_pipeline,
                &self.height_ao_bind_group_layout,
                &self.height_ao_uniform_buffer,
                &self.height_ao_view,
                &self.sampler_nearest,
            ) {
                let ao_width = (width as f32 * self.pbr_config.height_ao.resolution_scale) as u32;
                let ao_height = (height as f32 * self.pbr_config.height_ao.resolution_scale) as u32;

                // Update uniforms
                let uniforms: [f32; 16] = [
                    self.pbr_config.height_ao.directions as f32,
                    self.pbr_config.height_ao.steps as f32,
                    self.pbr_config.height_ao.max_distance,
                    self.pbr_config.height_ao.strength,
                    terrain_width / width as f32,
                    terrain_depth / height as f32,
                    z_scale,
                    terrain.domain.0,
                    ao_width as f32,
                    ao_height as f32,
                    width as f32,
                    height as f32,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ];
                self.queue
                    .write_buffer(uniform_buf, 0, bytemuck::cast_slice(&uniforms));

                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("terrain_viewer.height_ao_bind_group"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&terrain.heightmap_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(ao_view),
                        },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("terrain_viewer.height_ao_compute"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((ao_width + 7) / 8, (ao_height + 7) / 8, 1);
            }
        }

        // Sun visibility compute pass
        if self.pbr_config.sun_visibility.enabled {
            if let (
                Some(ref pipeline),
                Some(ref layout),
                Some(ref uniform_buf),
                Some(ref sv_view),
                Some(ref sampler),
            ) = (
                &self.sun_vis_pipeline,
                &self.sun_vis_bind_group_layout,
                &self.sun_vis_uniform_buffer,
                &self.sun_vis_view,
                &self.sampler_nearest,
            ) {
                let sv_width =
                    (width as f32 * self.pbr_config.sun_visibility.resolution_scale) as u32;
                let sv_height =
                    (height as f32 * self.pbr_config.sun_visibility.resolution_scale) as u32;
                let hard_mode = self
                    .pbr_config
                    .sun_visibility
                    .mode
                    .eq_ignore_ascii_case("hard");
                let effective_samples = if hard_mode {
                    1.0
                } else {
                    self.pbr_config.sun_visibility.samples as f32
                };
                let effective_softness = if hard_mode {
                    0.0
                } else {
                    self.pbr_config.sun_visibility.softness
                };

                // Update uniforms - sun_dir should point toward sun (negate light direction)
                let uniforms: [f32; 16] = [
                    effective_samples,
                    self.pbr_config.sun_visibility.steps as f32,
                    self.pbr_config.sun_visibility.max_distance,
                    effective_softness,
                    terrain_width / width as f32,
                    terrain_depth / height as f32,
                    z_scale,
                    terrain.domain.0,
                    sv_width as f32,
                    sv_height as f32,
                    width as f32,
                    height as f32,
                    sun_dir.x,
                    sun_dir.y,
                    sun_dir.z,
                    self.pbr_config.sun_visibility.bias,
                ];
                self.queue
                    .write_buffer(uniform_buf, 0, bytemuck::cast_slice(&uniforms));

                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("terrain_viewer.sun_vis_bind_group"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&terrain.heightmap_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(sv_view),
                        },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("terrain_viewer.sun_vis_compute"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((sv_width + 7) / 8, (sv_height + 7) / 8, 1);
            }
        }
    }
}
