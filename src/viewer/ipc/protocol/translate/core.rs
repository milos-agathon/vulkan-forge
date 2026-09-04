use crate::viewer::viewer_enums::ViewerCmd;

use super::super::request::IpcRequest;

pub(super) fn to_viewer_cmd(req: &IpcRequest) -> Option<ViewerCmd> {
    match req {
        IpcRequest::PickAt { x, y, shift, ctrl } => Some(ViewerCmd::PickAt {
            x: *x,
            y: *y,
            shift: *shift,
            ctrl: *ctrl,
        }),
        IpcRequest::UpdateLabels => Some(ViewerCmd::UpdateLabels),
        IpcRequest::SetLassoMode { enabled } => Some(ViewerCmd::SetLassoMode { enabled: *enabled }),
        IpcRequest::ClearSelection => Some(ViewerCmd::ClearSelection),
        IpcRequest::SetOitEnabled { enabled, mode } => Some(ViewerCmd::SetOitEnabled {
            enabled: *enabled,
            mode: mode.clone(),
        }),
        IpcRequest::GetOitMode => Some(ViewerCmd::GetOitMode),
        IpcRequest::SetTaaEnabled { enabled } => {
            Some(ViewerCmd::SetTaaEnabled { enabled: *enabled })
        }
        IpcRequest::GetTaaStatus => Some(ViewerCmd::GetTaaStatus),
        IpcRequest::SetTaaParams {
            history_weight,
            jitter_scale,
            enable_jitter,
        } => Some(ViewerCmd::SetTaaParams {
            history_weight: *history_weight,
            jitter_scale: *jitter_scale,
            enable_jitter: *enable_jitter,
        }),
        IpcRequest::LoadObj { path } => Some(ViewerCmd::LoadObj(path.clone())),
        IpcRequest::LoadGltf { path } => Some(ViewerCmd::LoadGltf(path.clone())),
        IpcRequest::SetTransform {
            translation,
            rotation_quat,
            scale,
        } => Some(ViewerCmd::SetTransform {
            translation: *translation,
            rotation_quat: *rotation_quat,
            scale: *scale,
        }),
        IpcRequest::CamLookat { eye, target, up } => Some(ViewerCmd::SetCamLookAt {
            eye: *eye,
            target: *target,
            up: *up,
        }),
        IpcRequest::SetFov { deg } => Some(ViewerCmd::SetFov(*deg)),
        IpcRequest::LitSun {
            azimuth_deg,
            elevation_deg,
        } => Some(ViewerCmd::SetSunDirection {
            azimuth_deg: *azimuth_deg,
            elevation_deg: *elevation_deg,
        }),
        IpcRequest::SetObservation {
            year,
            month,
            day,
            hour,
            minute,
            second,
            latitude_deg,
            longitude_deg,
            height_m,
        } => Some(ViewerCmd::SetObservation {
            year: *year,
            month: *month,
            day: *day,
            hour: *hour,
            minute: *minute,
            second: *second,
            latitude_deg: *latitude_deg,
            longitude_deg: *longitude_deg,
            height_m: *height_m,
        }),
        IpcRequest::LitIbl { path, intensity } => Some(ViewerCmd::SetIbl {
            path: path.clone(),
            intensity: *intensity,
        }),
        IpcRequest::SetZScale { value } => Some(ViewerCmd::SetZScale(*value)),
        IpcRequest::Snapshot {
            path,
            width,
            height,
        } => Some(ViewerCmd::SnapshotWithSize {
            path: path.clone(),
            width: *width,
            height: *height,
        }),
        IpcRequest::Close => Some(ViewerCmd::Quit),
        IpcRequest::SaveBundle { path, name } => Some(ViewerCmd::SaveBundle {
            path: path.clone(),
            name: name.clone(),
        }),
        IpcRequest::LoadBundle { path } => Some(ViewerCmd::LoadBundle { path: path.clone() }),
        _ => None,
    }
}
