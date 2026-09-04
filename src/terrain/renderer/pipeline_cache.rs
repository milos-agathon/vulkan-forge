use super::*;
use crate::terrain::renderer::core::TERRAIN_DEPTH_FORMAT;

impl TerrainScene {
    fn create_fullscreen_blit_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        sample_count: u32,
        pipeline_label: &'static str,
        shader_label: &'static str,
        shader_source: &str,
        depth_stencil: Option<wgpu::DepthStencilState>,
    ) -> wgpu::RenderPipeline {
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            shader_label,
            shader_source,
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain.blit.pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        crate::core::shader_registry::with_error_scope(device, pipeline_label, || {
            crate::core::shader_registry::create_render_pipeline_scoped(
                device,
                &wgpu::RenderPipelineDescriptor {
                    label: Some(pipeline_label),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: "vs_main",
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: "fs_main",
                        targets: &[Some(wgpu::ColorTargetState {
                            format: color_format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil,
                    multisample: wgpu::MultisampleState {
                        count: sample_count,
                        ..Default::default()
                    },
                    multiview: None,
                },
            )
        })
    }

    /// Assemble the forward terrain module. `shader_sources` owns the include
    /// expansion (WGSL has no preprocessor) and the two virtual-texture atlas
    /// variants; this layer only picks the variant the device can execute.
    pub(super) fn preprocess_terrain_shader(
        device: &wgpu::Device,
        include_aov_depth: bool,
    ) -> String {
        let mut source = if Self::terrain_atlas_is_bindless(device) {
            crate::shader_sources::terrain_bindless()
        } else {
            crate::shader_sources::terrain()
        };

        if !include_aov_depth {
            const BEGIN: &str = "// TERRAIN_AOV_DEPTH_BEGIN";
            const END: &str = "// TERRAIN_AOV_DEPTH_END";
            let begin = source.find(BEGIN);
            assert!(
                begin.is_some(),
                "terrain shader must contain the AOV depth begin marker"
            );
            let begin = begin.unwrap_or(0);
            let end_offset = source[begin..].find(END);
            assert!(
                end_offset.is_some(),
                "terrain shader must contain the AOV depth end marker"
            );
            let end_offset = end_offset.unwrap_or(0);
            let end = begin + end_offset + END.len();
            source.replace_range(
                begin..end,
                "// AOV depth is source-isolated from ordinary beauty modules.",
            );

            const ENTRY_BEGIN: &str = "// TERRAIN_AOV_ENTRY_BEGIN";
            const ENTRY_END: &str = "// TERRAIN_AOV_ENTRY_END";
            let entry_begin = source.find(ENTRY_BEGIN);
            assert!(
                entry_begin.is_some(),
                "terrain shader must contain the AOV entry begin marker"
            );
            let entry_begin = entry_begin.unwrap_or(0);
            let entry_end_offset = source[entry_begin..].find(ENTRY_END);
            assert!(
                entry_end_offset.is_some(),
                "terrain shader must contain the AOV entry end marker"
            );
            let entry_end = entry_begin + entry_end_offset.unwrap_or(0) + ENTRY_END.len();
            source.replace_range(
                entry_begin..entry_end,
                "// AOV entry is source-isolated from ordinary beauty modules.",
            );
        }

        source
    }

    /// True when the compiled module indexes the atlas `binding_array`. The
    /// compatibility path is a capability fallback, so the pipeline that
    /// compiled it records one — this covers renders that never build a
    /// virtual-texture runtime, which is what the other two sites
    /// (`core::capabilities::record_bindless_bc_fallbacks` and the VT runtime
    /// constructor) key off. `record_degradation` dedups on `(kind, name)`, so
    /// the identical triple collapses to a single certificate entry.
    fn terrain_atlas_is_bindless(device: &wgpu::Device) -> bool {
        let bindless = super::virtual_texture::bindless_bc_supported(device);
        if !bindless {
            crate::core::degradation::record_degradation(
                "rendering_fallback",
                "terrain_vt_bindless_atlas",
                "adapter lacks descriptor indexing; using the single-atlas compatibility path",
            );
        }
        bindless
    }

    /// TESSELLA pass 1 module: terrain + `terrain_visbuffer_write.wgsl`.
    fn preprocess_visibility_write_shader(device: &wgpu::Device) -> String {
        crate::shader_sources::terrain_visbuffer_write(Self::terrain_atlas_is_bindless(device))
    }

    /// TESSELLA pass 2 module: terrain + visibility resolve helpers. The
    /// runtime entry point replays clipmap geometry at equal depth; keeping a
    /// distinct source from pass 1 gives the certificate two hashes instead of
    /// aliasing one module.
    fn preprocess_visibility_resolve_shader(device: &wgpu::Device) -> String {
        crate::shader_sources::terrain_visbuffer_resolve(Self::terrain_atlas_is_bindless(device))
    }

    pub(super) fn create_render_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        light_buffer_layout: &wgpu::BindGroupLayout,
        ibl_bind_group_layout: &wgpu::BindGroupLayout,
        shadow_bind_group_layout: &wgpu::BindGroupLayout,
        fog_bind_group_layout: &wgpu::BindGroupLayout,
        water_reflection_bind_group_layout: &wgpu::BindGroupLayout,
        material_layer_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> wgpu::RenderPipeline {
        let shader_source = Self::preprocess_terrain_shader(device, false);
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "terrain_pbr_pom.shader",
            &shader_source,
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain_pbr_pom.pipeline_layout"),
            bind_group_layouts: &[
                bind_group_layout,         // @group(0): terrain uniforms/textures (bindings 0-11)
                light_buffer_layout,       // @group(1): lights (bindings 3-5)
                ibl_bind_group_layout,     // @group(2): IBL (bindings 0-4)
                &shadow_bind_group_layout, // @group(3): shadows (bindings 0-4)
                fog_bind_group_layout,     // @group(4): dedicated shared atmosphere (bindings 0-2)
                water_reflection_bind_group_layout, // @group(5): water reflections (bindings 0-2)
                material_layer_bind_group_layout, // @group(6): material layers + probes
            ],
            push_constant_ranges: &[],
        });

        crate::core::shader_registry::with_error_scope(device, "terrain_pbr_pom.pipeline", || {
            crate::core::shader_registry::create_render_pipeline_scoped(
                device,
                &wgpu::RenderPipelineDescriptor {
                    label: Some("terrain_pbr_pom.pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: "vs_main",
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: "fs_main",
                        targets: &[Some(wgpu::ColorTargetState {
                            format: color_format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: TERRAIN_DEPTH_FORMAT,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: sample_count,
                        ..Default::default()
                    },
                    multiview: None,
                },
            )
        })
    }

    pub(super) fn create_clipmap_render_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        light_buffer_layout: &wgpu::BindGroupLayout,
        ibl_bind_group_layout: &wgpu::BindGroupLayout,
        shadow_bind_group_layout: &wgpu::BindGroupLayout,
        fog_bind_group_layout: &wgpu::BindGroupLayout,
        water_reflection_bind_group_layout: &wgpu::BindGroupLayout,
        material_layer_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> wgpu::RenderPipeline {
        // `vs_clipmap_main` lives in the shared terrain_pbr_pom.wgsl module, so
        // the clipmap geometry path shades through the exact same PBR fragment
        // stage as the procedural-grid path.
        let shader_source = Self::preprocess_terrain_shader(device, false);
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "terrain_pbr_pom.clipmap.shader",
            &shader_source,
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain_pbr_pom.clipmap.pipeline_layout"),
            bind_group_layouts: &[
                bind_group_layout,
                light_buffer_layout,
                ibl_bind_group_layout,
                &shadow_bind_group_layout,
                fog_bind_group_layout,
                water_reflection_bind_group_layout,
                material_layer_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        crate::core::shader_registry::with_error_scope(
            device,
            "terrain_pbr_pom.clipmap.pipeline",
            || {
                crate::core::shader_registry::create_render_pipeline_scoped(
                    device,
                    &wgpu::RenderPipelineDescriptor {
                        label: Some("terrain_pbr_pom.clipmap.pipeline"),
                        layout: Some(&pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &shader,
                            entry_point: "vs_clipmap_main",
                            buffers: &[
                                crate::terrain::clipmap::ClipmapVertex::desc(),
                                crate::terrain::clipmap::gpu_lod::ClipmapDrawInstance::desc(),
                            ],
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &shader,
                            entry_point: "fs_main",
                            targets: &[Some(wgpu::ColorTargetState {
                                format: color_format,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                        }),
                        primitive: wgpu::PrimitiveState::default(),
                        depth_stencil: Some(wgpu::DepthStencilState {
                            format: TERRAIN_DEPTH_FORMAT,
                            depth_write_enabled: true,
                            depth_compare: wgpu::CompareFunction::Less,
                            stencil: wgpu::StencilState::default(),
                            bias: wgpu::DepthBiasState::default(),
                        }),
                        multisample: wgpu::MultisampleState {
                            count: sample_count,
                            ..Default::default()
                        },
                        multiview: None,
                    },
                )
            },
        )
    }

    pub(super) fn create_clipmap_visibility_write_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader_source = Self::preprocess_visibility_write_shader(device);
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "terrain_visbuffer_write.shader",
            &shader_source,
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain.visbuffer.write.pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });
        crate::core::shader_registry::with_error_scope(
            device,
            "terrain.visbuffer.write.pipeline",
            || {
                crate::core::shader_registry::create_render_pipeline_scoped(
                    device,
                    &wgpu::RenderPipelineDescriptor {
                        label: Some("terrain.visbuffer.write.pipeline"),
                        layout: Some(&pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &shader,
                            entry_point: "vs_clipmap_main",
                            buffers: &[
                                crate::terrain::clipmap::ClipmapVertex::desc(),
                                crate::terrain::clipmap::gpu_lod::ClipmapDrawInstance::desc(),
                            ],
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &shader,
                            entry_point: "fs_visibility",
                            targets: &[Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::R32Uint,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                        }),
                        primitive: wgpu::PrimitiveState::default(),
                        depth_stencil: Some(wgpu::DepthStencilState {
                            format: TERRAIN_DEPTH_FORMAT,
                            depth_write_enabled: true,
                            depth_compare: wgpu::CompareFunction::Less,
                            stencil: wgpu::StencilState::default(),
                            bias: wgpu::DepthBiasState::default(),
                        }),
                        multisample: wgpu::MultisampleState::default(),
                        multiview: None,
                    },
                )
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn create_clipmap_visibility_resolve_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        light_buffer_layout: &wgpu::BindGroupLayout,
        ibl_bind_group_layout: &wgpu::BindGroupLayout,
        shadow_bind_group_layout: &wgpu::BindGroupLayout,
        fog_bind_group_layout: &wgpu::BindGroupLayout,
        water_reflection_bind_group_layout: &wgpu::BindGroupLayout,
        material_layer_bind_group_layout: &wgpu::BindGroupLayout,
        visibility_resolve_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let shader_source = Self::preprocess_visibility_resolve_shader(device);
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "terrain_visbuffer_resolve.shader",
            &shader_source,
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain.visbuffer.resolve.pipeline_layout"),
            bind_group_layouts: &[
                bind_group_layout,
                light_buffer_layout,
                ibl_bind_group_layout,
                shadow_bind_group_layout,
                fog_bind_group_layout,
                water_reflection_bind_group_layout,
                material_layer_bind_group_layout,
                visibility_resolve_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        crate::core::shader_registry::with_error_scope(
            device,
            "terrain.visbuffer.resolve.pipeline",
            || {
                crate::core::shader_registry::create_render_pipeline_scoped(
                    device,
                    &wgpu::RenderPipelineDescriptor {
                        label: Some("terrain.visbuffer.resolve.pipeline"),
                        layout: Some(&pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &shader,
                            entry_point: "vs_clipmap_main",
                            buffers: &[
                                crate::terrain::clipmap::ClipmapVertex::desc(),
                                crate::terrain::clipmap::gpu_lod::ClipmapDrawInstance::desc(),
                            ],
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &shader,
                            entry_point: "fs_visibility_geometry",
                            targets: &[Some(wgpu::ColorTargetState {
                                format: color_format,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                        }),
                        primitive: wgpu::PrimitiveState::default(),
                        depth_stencil: Some(wgpu::DepthStencilState {
                            format: TERRAIN_DEPTH_FORMAT,
                            depth_write_enabled: false,
                            depth_compare: wgpu::CompareFunction::Equal,
                            stencil: wgpu::StencilState::default(),
                            bias: wgpu::DepthBiasState::default(),
                        }),
                        multisample: wgpu::MultisampleState::default(),
                        multiview: None,
                    },
                )
            },
        )
    }

    pub(super) fn create_blit_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> wgpu::RenderPipeline {
        Self::create_fullscreen_blit_pipeline(
            device,
            bind_group_layout,
            color_format,
            sample_count,
            "terrain.blit.pipeline",
            "terrain.blit.shader",
            include_str!("../../shaders/terrain_blit.wgsl"),
            None,
        )
    }

    pub(super) fn create_depth_blit_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> wgpu::RenderPipeline {
        // This pass clears depth for the terrain scene and draws only color, so it needs a
        // depth-compatible pipeline that never overwrites the cleared depth buffer.
        Self::create_fullscreen_blit_pipeline(
            device,
            bind_group_layout,
            color_format,
            sample_count,
            "terrain.blit.depth.pipeline",
            "terrain.blit.depth.shader",
            include_str!("../../shaders/terrain_blit.wgsl"),
            Some(wgpu::DepthStencilState {
                format: TERRAIN_DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
        )
    }

    pub(super) fn create_aether_depth_blit_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> wgpu::RenderPipeline {
        let source = crate::shader_sources::aether_blit();
        Self::create_fullscreen_blit_pipeline(
            device,
            bind_group_layout,
            color_format,
            sample_count,
            "terrain.aether.blit.depth.pipeline",
            "terrain.aether.blit.shader",
            &source,
            Some(wgpu::DepthStencilState {
                format: TERRAIN_DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
        )
    }

    pub(super) fn create_normal_blit_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> wgpu::RenderPipeline {
        Self::create_fullscreen_blit_pipeline(
            device,
            bind_group_layout,
            color_format,
            sample_count,
            "terrain.blit.normal.pipeline",
            "terrain.blit.normal.shader",
            include_str!("../../shaders/terrain_normal_blit.wgsl"),
            None,
        )
    }

    /// M1: Create AOV-enabled render pipeline with multiple render targets
    /// This pipeline outputs to 4 color targets: beauty, albedo, normal, depth.
    /// VERITAS: with `include_source_id` a 5th `R32Uint` target carries the
    /// per-pixel VT source-id map (requires sample_count == 1 — R32Uint is
    /// not multisample-capable).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn create_aov_render_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        light_buffer_layout: &wgpu::BindGroupLayout,
        ibl_bind_group_layout: &wgpu::BindGroupLayout,
        shadow_bind_group_layout: &wgpu::BindGroupLayout,
        fog_bind_group_layout: &wgpu::BindGroupLayout,
        water_reflection_bind_group_layout: &wgpu::BindGroupLayout,
        material_layer_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        sample_count: u32,
        output_mask: u8,
        include_source_id: bool,
        clipmap_geometry: bool,
    ) -> wgpu::RenderPipeline {
        let shader_source = Self::preprocess_terrain_shader(device, true);
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "terrain_pbr_pom.aov.shader",
            &shader_source,
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain_pbr_pom.aov.pipeline_layout"),
            bind_group_layouts: &[
                bind_group_layout,
                light_buffer_layout,
                ibl_bind_group_layout,
                shadow_bind_group_layout,
                fog_bind_group_layout,
                water_reflection_bind_group_layout,
                material_layer_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        // M1: AOV pipeline with beauty plus only the requested auxiliary
        // targets. `None` preserves WGSL location numbering without allocating,
        // clearing, writing, or resolving an unused RGBA16F texture.
        // Target 0: Beauty (tonemapped color)
        // Target 1: Albedo (base color before lighting)
        // Target 2: Normal (normalized world-space normal, signed float)
        // Target 3: Depth (linear depth normalized)
        // Target 4 (optional, VERITAS): per-pixel VT source id (R32Uint)
        let mut targets = vec![
            // Target 0: Beauty
            Some(wgpu::ColorTargetState {
                format: color_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            // Target 1: Albedo
            (output_mask & super::aov::AOV_ALBEDO_BIT != 0).then_some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            // Target 2: Normal
            (output_mask & super::aov::AOV_NORMAL_BIT != 0).then_some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            // Target 3: Depth
            (output_mask & super::aov::AOV_DEPTH_BIT != 0).then_some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
        ];
        if include_source_id {
            targets.push(Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R32Uint,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }));
        }
        let clipmap_vertex_buffers = [
            crate::terrain::clipmap::ClipmapVertex::desc(),
            crate::terrain::clipmap::gpu_lod::ClipmapDrawInstance::desc(),
        ];
        let (vertex_entry, vertex_buffers): (&str, &[wgpu::VertexBufferLayout]) =
            if clipmap_geometry {
                ("vs_clipmap_main", &clipmap_vertex_buffers)
            } else {
                ("vs_main", &[])
            };
        let pipeline_label = if clipmap_geometry {
            "terrain_pbr_pom.aov.clipmap.pipeline"
        } else {
            "terrain_pbr_pom.aov.pipeline"
        };
        crate::core::shader_registry::with_error_scope(device, pipeline_label, || {
            crate::core::shader_registry::create_render_pipeline_scoped(
                device,
                &wgpu::RenderPipelineDescriptor {
                    label: Some(pipeline_label),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: vertex_entry,
                        buffers: vertex_buffers,
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: "fs_aov_main",
                        targets: &targets,
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: TERRAIN_DEPTH_FORMAT,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: sample_count,
                        ..Default::default()
                    },
                    multiview: None,
                },
            )
        })
    }
}
