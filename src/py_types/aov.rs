use super::super::*;

fn decode_unorm_normal(pixel: &mut [f32]) {
    let x = pixel[0] * 2.0 - 1.0;
    let y = pixel[1] * 2.0 - 1.0;
    let z = pixel[2] * 2.0 - 1.0;
    let length = (x * x + y * y + z * z).sqrt().max(1e-8);
    pixel[0] = x / length;
    pixel[1] = y / length;
    pixel[2] = z / length;
}

fn resize_f32(
    source: &[f32],
    source_width: u32,
    source_height: u32,
    width: u32,
    height: u32,
    channels: usize,
) -> Vec<f32> {
    if (source_width, source_height) == (width, height) {
        return source.to_vec();
    }
    let mut output = vec![0.0; width as usize * height as usize * channels];
    for y in 0..height {
        let source_y = ((y as f32 + 0.5) * source_height as f32 / height as f32 - 0.5)
            .clamp(0.0, source_height.saturating_sub(1) as f32);
        let y0 = source_y.floor() as u32;
        let y1 = (y0 + 1).min(source_height - 1);
        let ty = source_y - y0 as f32;
        for x in 0..width {
            let source_x = ((x as f32 + 0.5) * source_width as f32 / width as f32 - 0.5)
                .clamp(0.0, source_width.saturating_sub(1) as f32);
            let x0 = source_x.floor() as u32;
            let x1 = (x0 + 1).min(source_width - 1);
            let tx = source_x - x0 as f32;
            for channel in 0..channels {
                let sample = |sx: u32, sy: u32| {
                    source[((sy * source_width + sx) as usize * channels) + channel]
                };
                let top = sample(x0, y0) * (1.0 - tx) + sample(x1, y0) * tx;
                let bottom = sample(x0, y1) * (1.0 - tx) + sample(x1, y1) * tx;
                output[((y * width + x) as usize * channels) + channel] =
                    top * (1.0 - ty) + bottom * ty;
            }
        }
    }
    output
}

#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "AovFrame")]
pub struct AovFrame {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    albedo_texture: Option<crate::core::resource_tracker::TrackedTexture>,
    normal_texture: Option<crate::core::resource_tracker::TrackedTexture>,
    depth_texture: Option<crate::core::resource_tracker::TrackedTexture>,
    /// VERITAS: per-pixel VT source-id map (R32Uint, 0 == SOURCE_ID_NONE).
    /// Registry/ledger accounting is owned by the `TrackedTexture` wrapper.
    source_id_texture: Option<crate::core::resource_tracker::TrackedTexture>,
    width: u32,
    height: u32,
    color_format: wgpu::TextureFormat,
    texture_width: u32,
    texture_height: u32,
    normal_encoded_unorm: bool,
}

#[cfg(feature = "images")]
fn extend_rgba_channels(
    channels: &mut Vec<exr_write::ExrChannelData>,
    prefix: &str,
    data: &[f32],
) -> anyhow::Result<()> {
    let expected_pixels = data.len() / 4;
    anyhow::ensure!(
        data.len() == expected_pixels * 4,
        "expected RGBA buffer for EXR export"
    );
    let mut r = Vec::with_capacity(expected_pixels);
    let mut g = Vec::with_capacity(expected_pixels);
    let mut b = Vec::with_capacity(expected_pixels);
    let mut a = Vec::with_capacity(expected_pixels);
    for px in data.chunks_exact(4) {
        r.push(px[0]);
        g.push(px[1]);
        b.push(px[2]);
        a.push(px[3]);
    }
    channels.extend([
        exr_write::ExrChannelData {
            name: format!("{prefix}.R"),
            data: r,
            quantize_linearly: false,
        },
        exr_write::ExrChannelData {
            name: format!("{prefix}.G"),
            data: g,
            quantize_linearly: false,
        },
        exr_write::ExrChannelData {
            name: format!("{prefix}.B"),
            data: b,
            quantize_linearly: false,
        },
        exr_write::ExrChannelData {
            name: format!("{prefix}.A"),
            data: a,
            quantize_linearly: true,
        },
    ]);
    Ok(())
}

#[cfg(feature = "images")]
fn extend_rgb_channels(
    channels: &mut Vec<exr_write::ExrChannelData>,
    prefix: &str,
    data: &[f32],
    suffixes: [&str; 3],
    quantize_linearly: bool,
) -> anyhow::Result<()> {
    let expected_pixels = data.len() / 4;
    anyhow::ensure!(
        data.len() == expected_pixels * 4,
        "expected RGBA buffer for EXR export"
    );
    let mut c0 = Vec::with_capacity(expected_pixels);
    let mut c1 = Vec::with_capacity(expected_pixels);
    let mut c2 = Vec::with_capacity(expected_pixels);
    for px in data.chunks_exact(4) {
        c0.push(px[0]);
        c1.push(px[1]);
        c2.push(px[2]);
    }
    channels.extend([
        exr_write::ExrChannelData {
            name: format!("{prefix}.{}", suffixes[0]),
            data: c0,
            quantize_linearly,
        },
        exr_write::ExrChannelData {
            name: format!("{prefix}.{}", suffixes[1]),
            data: c1,
            quantize_linearly,
        },
        exr_write::ExrChannelData {
            name: format!("{prefix}.{}", suffixes[2]),
            data: c2,
            quantize_linearly,
        },
    ]);
    Ok(())
}

#[cfg(feature = "images")]
fn extend_scalar_channel(
    channels: &mut Vec<exr_write::ExrChannelData>,
    prefix: &str,
    data: &[f32],
    suffix: &str,
) {
    channels.push(exr_write::ExrChannelData {
        name: format!("{prefix}.{suffix}"),
        data: data.to_vec(),
        quantize_linearly: true,
    });
}

#[cfg(feature = "images")]
fn build_terrain_exr_channels(
    beauty: &[f32],
    albedo: Option<&[f32]>,
    normal: Option<&[f32]>,
    depth: Option<&[f32]>,
) -> anyhow::Result<Vec<exr_write::ExrChannelData>> {
    let mut channels = Vec::new();
    extend_rgba_channels(&mut channels, "beauty", beauty)?;

    if let Some(rgba) = albedo {
        extend_rgb_channels(&mut channels, "albedo", rgba, ["R", "G", "B"], false)?;
    }
    if let Some(rgba) = normal {
        extend_rgb_channels(&mut channels, "normal", rgba, ["X", "Y", "Z"], true)?;
    }
    if let Some(values) = depth {
        extend_scalar_channel(&mut channels, "depth", values, "Z");
    }

    Ok(channels)
}

#[cfg(feature = "extension-module")]
impl AovFrame {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        albedo_texture: Option<crate::core::resource_tracker::TrackedTexture>,
        normal_texture: Option<crate::core::resource_tracker::TrackedTexture>,
        depth_texture: Option<crate::core::resource_tracker::TrackedTexture>,
        source_id_texture: Option<crate::core::resource_tracker::TrackedTexture>,
        width: u32,
        height: u32,
        color_format: wgpu::TextureFormat,
        texture_width: u32,
        texture_height: u32,
        normal_encoded_unorm: bool,
    ) -> Self {
        Self {
            device,
            queue,
            albedo_texture,
            normal_texture,
            depth_texture,
            source_id_texture,
            width,
            height,
            color_format,
            texture_width,
            texture_height,
            normal_encoded_unorm,
        }
    }

    fn read_texture_rgba_f32(
        &self,
        texture: &wgpu::Texture,
        format: wgpu::TextureFormat,
    ) -> anyhow::Result<Vec<f32>> {
        let rgba = if format == wgpu::TextureFormat::Rgba8Unorm {
            crate::core::hdr::read_ldr_texture(
                &self.device,
                &self.queue,
                texture,
                self.texture_width,
                self.texture_height,
            )
            .map(|bytes| {
                bytes
                    .into_iter()
                    .map(|value| value as f32 / 255.0)
                    .collect()
            })
            .map_err(anyhow::Error::msg)?
        } else {
            crate::core::hdr::read_hdr_texture(
                &self.device,
                &self.queue,
                texture,
                self.texture_width,
                self.texture_height,
                format,
            )
            .map_err(anyhow::Error::msg)?
        };
        Ok(resize_f32(
            &rgba,
            self.texture_width,
            self.texture_height,
            self.width,
            self.height,
            4,
        ))
    }

    fn read_albedo_data(&self) -> anyhow::Result<Vec<f32>> {
        let texture = self
            .albedo_texture
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Albedo AOV not available"))?;
        self.read_texture_rgba_f32(texture, self.color_format)
    }

    fn read_normal_data(&self) -> anyhow::Result<Vec<f32>> {
        let texture = self
            .normal_texture
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Normal AOV not available"))?;
        let mut rgba = self.read_texture_rgba_f32(texture, self.color_format)?;
        if self.normal_encoded_unorm {
            for pixel in rgba.as_chunks_mut::<4>().0 {
                decode_unorm_normal(pixel);
            }
        }
        Ok(rgba)
    }

    fn read_depth_data(&self) -> anyhow::Result<Vec<f32>> {
        let texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Depth AOV not available"))?;
        let rgba = self.read_texture_rgba_f32(texture, wgpu::TextureFormat::Rgba16Float)?;
        Ok(rgba.chunks_exact(4).map(|px| px[0]).collect())
    }

    fn rgba_to_rgb_array(&self, rgba: &[f32]) -> anyhow::Result<ndarray::Array3<f32>> {
        let mut rgb = Vec::with_capacity((self.width * self.height * 3) as usize);
        for px in rgba.chunks_exact(4) {
            rgb.extend_from_slice(&px[..3]);
        }
        ndarray::Array3::from_shape_vec((self.height as usize, self.width as usize, 3), rgb)
            .map_err(|_| anyhow::anyhow!("failed to reshape RGBA buffer into RGB array"))
    }

    fn encode_rgb_png(&self, rgb: impl Iterator<Item = [f32; 3]>) -> Vec<u8> {
        let mut bytes = Vec::with_capacity((self.width * self.height * 4) as usize);
        for [r, g, b] in rgb {
            bytes.push((r.clamp(0.0, 1.0) * 255.0).round() as u8);
            bytes.push((g.clamp(0.0, 1.0) * 255.0).round() as u8);
            bytes.push((b.clamp(0.0, 1.0) * 255.0).round() as u8);
            bytes.push(255);
        }
        bytes
    }

    fn write_png_bytes(&self, path: &str, data: &[u8]) -> PyResult<()> {
        image_write::write_png_rgba8(Path::new(path), data, self.width, self.height)
            .map_err(|err| PyRuntimeError::new_err(format!("failed to write PNG: {err:#}")))
    }
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl AovFrame {
    #[new]
    fn py_new() -> PyResult<Self> {
        Err(PyRuntimeError::new_err(
            "AovFrame objects are constructed internally by forge3d",
        ))
    }

    #[getter]
    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    #[getter]
    fn has_albedo(&self) -> bool {
        self.albedo_texture.is_some()
    }

    #[getter]
    fn has_normal(&self) -> bool {
        self.normal_texture.is_some()
    }

    #[getter]
    fn has_depth(&self) -> bool {
        self.depth_texture.is_some()
    }

    #[getter]
    fn has_source_id(&self) -> bool {
        self.source_id_texture.is_some()
    }

    /// VERITAS: per-pixel VT source-id map as a `(H, W)` uint32 array.
    /// `0` is `SOURCE_ID_NONE` (no attribution); nonzero values are the
    /// stable ids recorded in the provenance manifest's `source_table`.
    fn source_id<'py>(&self, py: Python<'py>) -> PyResult<&'py numpy::PyArray2<u32>> {
        let texture = self.source_id_texture.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "Source-id AOV not available (enable AovSettings.source_id before rendering)",
            )
        })?;
        let data = py
            .allow_threads(|| {
                crate::core::hdr::read_r32uint_texture(
                    &self.device,
                    &self.queue,
                    texture,
                    self.width,
                    self.height,
                )
            })
            .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err}")))?;
        let arr =
            ndarray::Array2::from_shape_vec((self.height as usize, self.width as usize), data)
                .map_err(|_| {
                    PyRuntimeError::new_err("failed to reshape source-id buffer into numpy array")
                })?;
        Ok(arr.into_pyarray_bound(py).into_gil_ref())
    }

    fn albedo<'py>(&self, py: Python<'py>) -> PyResult<&'py numpy::PyArray3<f32>> {
        let rgba = py
            .allow_threads(|| self.read_albedo_data())
            .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
        let arr = self
            .rgba_to_rgb_array(&rgba)
            .map_err(|err| PyRuntimeError::new_err(format!("reshape failed: {err:#}")))?;
        Ok(arr.into_pyarray_bound(py).into_gil_ref())
    }

    fn normal<'py>(&self, py: Python<'py>) -> PyResult<&'py numpy::PyArray3<f32>> {
        let rgba = py
            .allow_threads(|| self.read_normal_data())
            .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
        let arr = self
            .rgba_to_rgb_array(&rgba)
            .map_err(|err| PyRuntimeError::new_err(format!("reshape failed: {err:#}")))?;
        Ok(arr.into_pyarray_bound(py).into_gil_ref())
    }

    fn depth<'py>(&self, py: Python<'py>) -> PyResult<&'py numpy::PyArray2<f32>> {
        let data = py
            .allow_threads(|| self.read_depth_data())
            .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
        let arr =
            ndarray::Array2::from_shape_vec((self.height as usize, self.width as usize), data)
                .map_err(|_| {
                    PyRuntimeError::new_err("failed to reshape depth buffer into numpy array")
                })?;
        Ok(arr.into_pyarray_bound(py).into_gil_ref())
    }

    fn save_albedo(&self, path: &str) -> PyResult<()> {
        let rgba = self
            .read_albedo_data()
            .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
        let png = self.encode_rgb_png(rgba.chunks_exact(4).map(|px| [px[0], px[1], px[2]]));
        self.write_png_bytes(path, &png)
    }

    fn save_normal(&self, path: &str) -> PyResult<()> {
        let rgba = self
            .read_normal_data()
            .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
        let png = self.encode_rgb_png(
            rgba.chunks_exact(4)
                .map(|px| [px[0] * 0.5 + 0.5, px[1] * 0.5 + 0.5, px[2] * 0.5 + 0.5]),
        );
        self.write_png_bytes(path, &png)
    }

    fn save_depth(&self, path: &str) -> PyResult<()> {
        let depth = self
            .read_depth_data()
            .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
        let png = self.encode_rgb_png(depth.iter().copied().map(|value| [value, value, value]));
        self.write_png_bytes(path, &png)
    }

    fn save_all(&self, output_dir: &str, base_name: &str) -> PyResult<()> {
        let dir = Path::new(output_dir);
        std::fs::create_dir_all(dir)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to create directory: {e}")))?;

        if self.albedo_texture.is_some() {
            let path = dir.join(format!("{}_albedo.png", base_name));
            self.save_albedo(path.to_str().unwrap())?;
        }
        if self.normal_texture.is_some() {
            let path = dir.join(format!("{}_normal.png", base_name));
            self.save_normal(path.to_str().unwrap())?;
        }
        if self.depth_texture.is_some() {
            let path = dir.join(format!("{}_depth.png", base_name));
            self.save_depth(path.to_str().unwrap())?;
        }
        Ok(())
    }

    #[cfg(feature = "images")]
    fn save_exr(&self, path: &str, beauty_frame: &crate::Frame) -> PyResult<()> {
        if beauty_frame.dimensions() != (self.width, self.height) {
            return Err(PyValueError::new_err(format!(
                "beauty frame size {:?} does not match AOV size {:?}",
                beauty_frame.dimensions(),
                (self.width, self.height)
            )));
        }

        let beauty = beauty_frame
            .read_rgba_f32()
            .map_err(|err| PyRuntimeError::new_err(format!("beauty readback failed: {err:#}")))?;

        let albedo = self
            .albedo_texture
            .as_ref()
            .map(|_| {
                self.read_albedo_data().map_err(|err| {
                    PyRuntimeError::new_err(format!("albedo readback failed: {err:#}"))
                })
            })
            .transpose()?;
        let normal = self
            .normal_texture
            .as_ref()
            .map(|_| {
                self.read_normal_data().map_err(|err| {
                    PyRuntimeError::new_err(format!("normal readback failed: {err:#}"))
                })
            })
            .transpose()?;
        let depth = self
            .depth_texture
            .as_ref()
            .map(|_| {
                self.read_depth_data().map_err(|err| {
                    PyRuntimeError::new_err(format!("depth readback failed: {err:#}"))
                })
            })
            .transpose()?;
        let channels = build_terrain_exr_channels(
            &beauty,
            albedo.as_deref(),
            normal.as_deref(),
            depth.as_deref(),
        )
        .map_err(|err| PyRuntimeError::new_err(format!("EXR channel build failed: {err:#}")))?;

        exr_write::write_exr_f32_channels(Path::new(path), self.width, self.height, channels)
            .map_err(|err| PyRuntimeError::new_err(format!("failed to write EXR: {err:#}")))?;
        Ok(())
    }

    #[cfg(not(feature = "images"))]
    fn save_exr(&self, _path: &str, _beauty_frame: &crate::Frame) -> PyResult<()> {
        Err(PyRuntimeError::new_err(
            "saving EXR files requires the 'images' feature",
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "AovFrame(width={}, height={}, albedo={}, normal={}, depth={}, source_id={})",
            self.width,
            self.height,
            self.albedo_texture.is_some(),
            self.normal_texture.is_some(),
            self.depth_texture.is_some(),
            self.source_id_texture.is_some()
        )
    }
}

#[cfg(all(test, feature = "images"))]
mod tests {
    use super::*;
    use exr::prelude::{read_first_flat_layer_from_file, FlatSamples};
    use std::collections::BTreeMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_path(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        path.push(format!(
            "forge3d_{name}_{}_{}.exr",
            std::process::id(),
            nonce
        ));
        path
    }

    fn flat_f32_channels(path: &Path) -> (usize, usize, BTreeMap<String, Vec<f32>>) {
        let image = read_first_flat_layer_from_file(path).expect("round-trip EXR should load");
        let width = image.layer_data.size.width();
        let height = image.layer_data.size.height();
        let channels = image
            .layer_data
            .channel_data
            .list
            .into_iter()
            .map(|channel| {
                let samples = match channel.sample_data {
                    FlatSamples::F32(samples) => samples,
                    other => panic!("expected f32 samples, got {other:?}"),
                };
                (channel.name.to_string(), samples)
            })
            .collect();
        (width, height, channels)
    }

    #[test]
    fn terrain_exr_channel_round_trip_preserves_names_and_values() {
        let beauty = vec![
            0.1, 0.2, 0.3, 1.0, 0.4, 0.5, 0.6, 0.75, 0.7, 0.8, 0.9, 0.5, 1.0, 0.0, 0.25, 0.125,
        ];
        let albedo = vec![
            0.9, 0.1, 0.2, 0.0, 0.8, 0.2, 0.3, 0.0, 0.7, 0.3, 0.4, 0.0, 0.6, 0.4, 0.5, 0.0,
        ];
        let normal = vec![
            0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0,
        ];
        let depth = vec![0.1, 0.3, 0.6, 0.9];

        let channels =
            build_terrain_exr_channels(&beauty, Some(&albedo), Some(&normal), Some(&depth))
                .expect("channel assembly should succeed");
        let expected_channels: BTreeMap<String, Vec<f32>> = channels
            .iter()
            .map(|channel| (channel.name.clone(), channel.data.clone()))
            .collect();

        let path = unique_temp_path("terrain_aov_channels");
        exr_write::write_exr_f32_channels(&path, 2, 2, channels).expect("EXR write should succeed");

        let (width, height, actual_channels) = flat_f32_channels(&path);
        std::fs::remove_file(&path).ok();

        assert_eq!((width, height), (2, 2));
        assert_eq!(actual_channels, expected_channels);
    }
}
