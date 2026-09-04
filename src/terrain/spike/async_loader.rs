use super::*;

#[pymethods]
impl TerrainSpike {
    // E1c/E1e: Enable async tile loader with dedup/backpressure
    #[pyo3(
        text_signature = "($self, tile_resolution=64, max_in_flight=32, pool_size=4, template=None, scale=None, offset=None, coalesce_policy='coarse')"
    )]
    pub fn enable_async_loader(
        &mut self,
        tile_resolution: Option<u32>,
        max_in_flight: Option<usize>,
        pool_size: Option<usize>,
        template: Option<String>,
        scale: Option<f32>,
        offset: Option<f32>,
        coalesce_policy: Option<String>,
    ) -> PyResult<()> {
        let tiling_system = self.tiling_system.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Tiling system not enabled. Call enable_tiling() first.",
            )
        })?;
        let res = tile_resolution.unwrap_or(64).max(1);
        let inflight = max_in_flight.unwrap_or(32).max(1);
        let pool = pool_size.unwrap_or(4).max(1);
        let policy = match coalesce_policy.as_deref() {
            None | Some("coarse") => crate::terrain::page_table::CoalescePolicy::PreferCoarse,
            Some("fine") => crate::terrain::page_table::CoalescePolicy::PreferFine,
            Some(other) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Invalid coalesce_policy '{}'. Expected 'coarse' or 'fine'",
                    other
                )));
            }
        };
        let tmpl = template.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "template is required; synthetic height I/O is test-only",
            )
        })?;
        let s = scale.unwrap_or(1.0);
        let o = offset.unwrap_or(0.0);
        let rdr = std::sync::Arc::new(crate::terrain::page_table::FileHeightReader::new(
            tmpl, s, o,
        ));
        let loader = crate::terrain::page_table::AsyncTileLoader::new_with_reader(
            tiling_system.root_bounds.clone(),
            tiling_system.tile_size,
            res,
            inflight,
            pool,
            rdr,
            policy,
        );
        self.async_loader = Some(loader);
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn debug_async_loader_stats(&self) -> PyResult<(usize, usize, usize)> {
        if let Some(loader) = &self.async_loader {
            let stats = loader.stats();
            Ok(stats)
        } else {
            Ok((0, 0, 0))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn debug_async_loader_counters(
        &self,
    ) -> PyResult<(usize, usize, usize, usize, usize, usize)> {
        if let Some(loader) = &self.async_loader {
            let c = loader.counters();
            Ok(c)
        } else {
            Ok((0, 0, 0, 0, 0, 0))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn debug_async_overlay_loader_stats(&self) -> PyResult<(usize, usize, usize)> {
        if let Some(loader) = &self.async_overlay_loader {
            let stats = loader.stats();
            Ok(stats)
        } else {
            Ok((0, 0, 0))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn debug_async_overlay_loader_counters(
        &self,
    ) -> PyResult<(usize, usize, usize, usize, usize, usize)> {
        if let Some(loader) = &self.async_overlay_loader {
            let c = loader.counters();
            Ok(c)
        } else {
            Ok((0, 0, 0, 0, 0, 0))
        }
    }

    // E3: Enable GPU overlay mosaic (RGBA8 atlas) and wire overlay compositor
    #[pyo3(text_signature = "($self, tile_px, tiles_x, tiles_y, srgb=True, filter_linear=True)")]
    pub fn enable_overlay_mosaic(
        &mut self,
        tile_px: u32,
        tiles_x: u32,
        tiles_y: u32,
        srgb: Option<bool>,
        filter_linear: Option<bool>,
    ) -> PyResult<()> {
        let srgb = srgb.unwrap_or(true);
        let filter_linear = filter_linear.unwrap_or(true);
        let cfg = crate::terrain::stream::MosaicConfig {
            tile_size_px: tile_px,
            tiles_x,
            tiles_y,
            fixed_lod: None,
        };
        let mosaic =
            crate::terrain::stream::ColorMosaic::new(&self.device, cfg, srgb, filter_linear)?;

        // Lazy-create overlay renderer if needed
        if self.overlay_renderer.is_none() {
            let ov = crate::core::overlays::OverlayRenderer::new(
                &self.device,
                TEXTURE_FORMAT,
                self.height_filterable,
            )?;
            self.overlay_renderer = Some(ov);
        }
        // Bind overlay view; use height mosaic if available for altitude/contour paths
        if let Some(ref mut ov) = self.overlay_renderer {
            let height_view_opt = self.height_mosaic.as_ref().map(|m| &m.view);
            let pt_buf_opt = self.page_table.as_ref().map(|pt| pt.buffer.inner());
            ov.recreate_bind_group(
                &self.device,
                Some(&mosaic.view),
                height_view_opt,
                pt_buf_opt,
                None,
            )?;
            // Enable overlay by default with alpha 1.0
            ov.set_enabled(true);
            ov.set_overlay_alpha(1.0);
            ov.upload_uniforms(&self.queue);
        }
        self.overlay_mosaic = Some(mosaic);
        Ok(())
    }

    // E3/E1 parity: enable async overlay loader (RGBA8)
    #[pyo3(
        text_signature = "($self, tile_resolution=64, max_in_flight=32, pool_size=4, template=None, coalesce_policy='coarse')"
    )]
    pub fn enable_async_overlay_loader(
        &mut self,
        tile_resolution: Option<u32>,
        max_in_flight: Option<usize>,
        pool_size: Option<usize>,
        template: Option<String>,
        coalesce_policy: Option<String>,
    ) -> PyResult<()> {
        let tiling_system = self.tiling_system.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Tiling system not enabled. Call enable_tiling() first.",
            )
        })?;
        let res = tile_resolution.unwrap_or(64).max(1);
        let inflight = max_in_flight.unwrap_or(32).max(1);
        let pool = pool_size.unwrap_or(4).max(1);
        let policy = match coalesce_policy.as_deref() {
            None | Some("coarse") => crate::terrain::page_table::CoalescePolicy::PreferCoarse,
            Some("fine") => crate::terrain::page_table::CoalescePolicy::PreferFine,
            Some(other) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Invalid coalesce_policy '{}'. Expected 'coarse' or 'fine'",
                    other
                )));
            }
        };
        let tmpl = template.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "template is required; synthetic overlay I/O is test-only",
            )
        })?;
        let rdr = std::sync::Arc::new(crate::terrain::page_table::FileOverlayReader::new(tmpl));
        let loader = crate::terrain::page_table::AsyncOverlayLoader::new_with_reader(
            tiling_system.root_bounds.clone(),
            tiling_system.tile_size,
            res,
            inflight,
            pool,
            rdr,
            policy,
        );
        self.async_overlay_loader = Some(loader);
        Ok(())
    }
}
