use glam::Vec2;

use crate::terrain::tiling::{TileBounds, TileId};

/// Simple file-backed overlay reader that expands a template like
/// "/data/tiles/{lod}/{x}/{y}.png" and returns RGBA8 bytes.
pub struct FileOverlayReader {
    template: String,
}

impl FileOverlayReader {
    pub fn new(template: String) -> Self {
        Self { template }
    }

    fn expand(&self, id: TileId) -> String {
        self.template
            .replace("{lod}", &id.lod.to_string())
            .replace("{x}", &id.x.to_string())
            .replace("{y}", &id.y.to_string())
    }
}

impl OverlayReader for FileOverlayReader {
    fn read(
        &self,
        _root_bounds: &TileBounds,
        _tile_size: Vec2,
        tile_id: TileId,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let path = self.expand(tile_id);
        match image::open(&path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                if rgba.width() != width || rgba.height() != height {
                    let resized = image::imageops::resize(
                        &rgba,
                        width,
                        height,
                        image::imageops::FilterType::Triangle,
                    );
                    resized.into_raw()
                } else {
                    rgba.into_raw()
                }
            }
            Err(_) => vec![0u8; (width * height * 4) as usize],
        }
    }
}

/// Simple file-backed height reader using PNG grayscale (8/16-bit) to f32 with scale/offset
pub struct FileHeightReader {
    template: String,
    scale: f32,
    offset: f32,
}

impl FileHeightReader {
    pub fn new(template: String, scale: f32, offset: f32) -> Self {
        Self {
            template,
            scale,
            offset,
        }
    }

    fn expand(&self, id: TileId) -> String {
        self.template
            .replace("{lod}", &id.lod.to_string())
            .replace("{x}", &id.x.to_string())
            .replace("{y}", &id.y.to_string())
    }
}

impl HeightReader for FileHeightReader {
    fn read(
        &self,
        _root_bounds: &TileBounds,
        _tile_size: Vec2,
        tile_id: TileId,
        width: u32,
        height: u32,
    ) -> Vec<f32> {
        let path = self.expand(tile_id);
        let expected = (width * height) as usize;
        match image::open(&path) {
            Ok(img) => {
                let gray = img.to_luma16();
                let (w, h) = gray.dimensions();
                if w != width || h != height {
                    let resized = image::imageops::resize(
                        &gray,
                        width,
                        height,
                        image::imageops::FilterType::Triangle,
                    );
                    let mut out = Vec::with_capacity(expected);
                    for &v16 in resized.as_raw() {
                        let v = (v16 as f32) / 65535.0;
                        out.push(v * self.scale + self.offset);
                    }
                    out
                } else {
                    let mut out = Vec::with_capacity(expected);
                    for &v16 in gray.as_raw() {
                        let v = (v16 as f32) / 65535.0;
                        out.push(v * self.scale + self.offset);
                    }
                    out
                }
            }
            Err(_) => vec![0.0f32; expected],
        }
    }
}

pub trait HeightReader: Send + Sync + 'static {
    fn read(
        &self,
        root_bounds: &TileBounds,
        tile_size: Vec2,
        tile_id: TileId,
        width: u32,
        height: u32,
    ) -> Vec<f32>;
}

pub trait OverlayReader: Send + Sync + 'static {
    fn read(
        &self,
        root_bounds: &TileBounds,
        tile_size: Vec2,
        tile_id: TileId,
        width: u32,
        height: u32,
    ) -> Vec<u8>;
}
