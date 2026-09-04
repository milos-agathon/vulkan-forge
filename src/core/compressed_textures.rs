mod bc4;
pub mod bc5;
pub mod bc7;
mod compression;
mod load;
mod parsing;
mod types;
mod upload;

pub use bc5::{decode_bc5_rg8, encode_bc5_rg8};
pub use bc7::{decode_bc7_rgba8, encode_bc7_rgba8};
pub use types::{CompressedImage, CompressionOptions, CompressionStats};

#[cfg(test)]
mod tests;
