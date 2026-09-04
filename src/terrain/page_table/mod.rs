// E1: Page table & async IO scaffolding
// Lightweight GPU buffer storing tile->slot mappings for the current height mosaic.
// Also includes a simple async tile request queue scaffold (main-thread drained).

mod common;
mod gpu;
mod height_loader;
mod overlay_loader;
mod queue;
mod readers;

pub use common::CoalescePolicy;
pub use gpu::{PageTable, PageTableEntry};
pub use height_loader::AsyncTileLoader;
pub use overlay_loader::{AsyncOverlayLoader, OverlayTileData};
pub use queue::AsyncTileQueue;
pub use readers::{FileHeightReader, FileOverlayReader, HeightReader, OverlayReader};
