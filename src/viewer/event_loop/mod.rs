// src/viewer/event_loop/mod.rs
// Event loop handling for the interactive viewer
// Extracted from mod.rs as part of the viewer refactoring

mod cmd_parse_init;
mod command_batch;
mod command_preflight;
mod command_preflight_helpers;
mod ipc_state;
mod runner;
mod stdin_reader;

pub use cmd_parse_init::parse_initial_commands;
pub(crate) use command_batch::{command_priority, order_command_batch};
pub use ipc_state::{
    get_ipc_queue, get_ipc_stats, get_lasso_state, get_pick_events, get_scene_review_state,
    get_terrain_volumetrics_report, set_pending_bundle_load, set_pending_bundle_save,
    take_pending_bundle_load, take_pending_bundle_save, update_ipc_frame_stats,
    update_ipc_media_diagnostics, update_ipc_revision_stats, update_ipc_stats,
    update_ipc_transform_stats, update_scene_review_state, update_terrain_volumetrics_report,
    QueuedIpcCommand,
};
pub use runner::{run_viewer, run_viewer_with_ipc};
pub use stdin_reader::spawn_stdin_reader;
