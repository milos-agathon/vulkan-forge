use super::super::*;

#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "Frame")]
pub struct Frame {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    texture: Arc<crate::core::resource_tracker::TrackedTexture>,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
}

impl Frame {
    pub(crate) fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        texture: impl Into<Arc<crate::core::resource_tracker::TrackedTexture>>,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            device,
            queue,
            texture: texture.into(),
            width,
            height,
            format,
        }
    }

    pub(crate) fn read_tight_bytes(&self) -> anyhow::Result<Vec<u8>> {
        read_texture_tight(
            &self.device,
            &self.queue,
            &self.texture,
            (self.width, self.height),
            self.format,
        )
    }

    pub(crate) fn read_rgb_u8(&self) -> anyhow::Result<ndarray::Array3<u8>> {
        let rgba = self.read_tight_bytes()?;
        let mut rgb = Vec::with_capacity((self.width * self.height * 3) as usize);
        for pixel in rgba.chunks_exact(4) {
            match self.format {
                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
                    rgb.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
                }
                wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
                    rgb.extend_from_slice(&pixel[..3]);
                }
                other => {
                    return Err(anyhow::anyhow!(
                        "acceptance beauty readback requires an 8-bit RGBA target, got {other:?}"
                    ));
                }
            }
        }
        ndarray::Array3::from_shape_vec((self.height as usize, self.width as usize, 3), rgb)
            .map_err(|_| anyhow::anyhow!("failed to reshape beauty readback"))
    }

    #[cfg(feature = "images")]
    pub(crate) fn read_rgba_f32(&self) -> anyhow::Result<Vec<f32>> {
        match self.format {
            wgpu::TextureFormat::Rgba8Unorm
            | wgpu::TextureFormat::Rgba8UnormSrgb
            | wgpu::TextureFormat::Bgra8Unorm
            | wgpu::TextureFormat::Bgra8UnormSrgb => {
                let data = self.read_tight_bytes()?;
                let mut rgba = Vec::with_capacity(data.len());
                for px in data.chunks_exact(4) {
                    match self.format {
                        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
                            rgba.push(px[2] as f32 / 255.0);
                            rgba.push(px[1] as f32 / 255.0);
                            rgba.push(px[0] as f32 / 255.0);
                            rgba.push(px[3] as f32 / 255.0);
                        }
                        _ => {
                            rgba.push(px[0] as f32 / 255.0);
                            rgba.push(px[1] as f32 / 255.0);
                            rgba.push(px[2] as f32 / 255.0);
                            rgba.push(px[3] as f32 / 255.0);
                        }
                    }
                }
                Ok(rgba)
            }
            wgpu::TextureFormat::Rgba16Float => crate::core::hdr::read_hdr_texture(
                &self.device,
                &self.queue,
                &self.texture,
                self.width,
                self.height,
                self.format,
            )
            .map_err(anyhow::Error::msg),
            other => Err(anyhow::anyhow!(
                "unsupported texture format for float readback: {:?}",
                other
            )),
        }
    }

    #[cfg(feature = "images")]
    pub(crate) fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl Frame {
    #[new]
    fn py_new() -> PyResult<Self> {
        Err(PyRuntimeError::new_err(
            "Frame objects are constructed internally by forge3d",
        ))
    }

    #[getter]
    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn format(&self) -> String {
        format!("{:?}", self.format)
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let path_obj = Path::new(path);
        match self.format {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
                if let Some(ext) = path_obj.extension().and_then(|ext| ext.to_str()) {
                    if !ext.eq_ignore_ascii_case("png") {
                        return Err(PyValueError::new_err(format!(
                            "expected .png extension for RGBA8 frame save, got .{}",
                            ext
                        )));
                    }
                }
                let mut data = self
                    .read_tight_bytes()
                    .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
                for px in data.chunks_exact_mut(4) {
                    px[3] = 255;
                }
                image_write::write_png_rgba8(path_obj, &data, self.width, self.height).map_err(
                    |err| PyRuntimeError::new_err(format!("failed to write PNG: {err:#}")),
                )?;
                Ok(())
            }
            wgpu::TextureFormat::Rgba16Float => {
                if let Some(ext) = path_obj.extension().and_then(|ext| ext.to_str()) {
                    if !ext.eq_ignore_ascii_case("exr") {
                        return Err(PyValueError::new_err(format!(
                            "expected .exr extension for RGBA16F frame save, got .{}",
                            ext
                        )));
                    }
                }
                #[cfg(feature = "images")]
                {
                    let data = crate::core::hdr::read_hdr_texture(
                        &self.device,
                        &self.queue,
                        &self.texture,
                        self.width,
                        self.height,
                        self.format,
                    )
                    .map_err(|err| {
                        PyRuntimeError::new_err(format!("HDR readback failed: {err}"))
                    })?;

                    exr_write::write_exr_rgba_f32(
                        path_obj,
                        self.width,
                        self.height,
                        &data,
                        "beauty",
                    )
                    .map_err(|err| {
                        PyRuntimeError::new_err(format!("failed to write EXR: {err:#}"))
                    })?;
                    Ok(())
                }
                #[cfg(not(feature = "images"))]
                {
                    Err(PyRuntimeError::new_err(
                        "saving RGBA16F frames requires the 'images' feature",
                    ))
                }
            }
            other => Err(PyValueError::new_err(format!(
                "unsupported texture format for save: {:?}",
                other
            ))),
        }
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray3<u8>> {
        match self.format {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
                let data = self
                    .read_tight_bytes()
                    .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
                let arr = ndarray::Array3::from_shape_vec(
                    (self.height as usize, self.width as usize, 4),
                    data,
                )
                .map_err(|_| {
                    PyRuntimeError::new_err("failed to reshape RGBA buffer into numpy array")
                })?;
                Ok(arr.into_pyarray_bound(py).into_gil_ref())
            }
            wgpu::TextureFormat::Rgba16Float => Err(PyRuntimeError::new_err(
                "to_numpy for RGBA16F frames is not implemented yet",
            )),
            other => Err(PyValueError::new_err(format!(
                "unsupported texture format for numpy conversion: {:?}",
                other
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Frame(width={}, height={}, format={:?})",
            self.width, self.height, self.format
        )
    }
}
