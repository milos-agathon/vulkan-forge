// src/viewer/ipc_split/mod.rs
// IPC module for non-blocking viewer control via TCP + NDJSON
// Split from ipc.rs for <= 300 LOC per file

mod protocol;
mod server;

pub use protocol::{
    ipc_request_to_viewer_cmd, parse_ipc_request, IpcRequest, IpcResponse,
    TerrainVolumetricsReport, TerrainVolumetricsVolumeReport, ViewerStats,
};
pub use server::{start_ipc_server, IpcServerConfig, IpcServerHandle};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_load_obj() {
        let json = r#"{"cmd":"load_obj","path":"model.obj"}"#;
        let req = parse_ipc_request(json).unwrap();
        match req {
            IpcRequest::LoadObj { path } => assert_eq!(path, "model.obj"),
            _ => panic!("Expected LoadObj"),
        }
    }

    #[test]
    fn test_parse_cam_lookat() {
        let json = r#"{"cmd":"cam_lookat","eye":[0,5,10],"target":[0,0,0],"up":[0,1,0]}"#;
        let req = parse_ipc_request(json).unwrap();
        match req {
            IpcRequest::CamLookat { eye, target, up } => {
                assert_eq!(eye, [0.0, 5.0, 10.0]);
                assert_eq!(target, [0.0, 0.0, 0.0]);
                assert_eq!(up, [0.0, 1.0, 0.0]);
            }
            _ => panic!("Expected CamLookat"),
        }
    }

    #[test]
    fn test_parse_set_transform() {
        let json = r#"{"cmd":"set_transform","translation":[1,2,3]}"#;
        let req = parse_ipc_request(json).unwrap();
        match req {
            IpcRequest::SetTransform { translation, .. } => {
                assert_eq!(translation, Some([1.0, 2.0, 3.0]));
            }
            _ => panic!("Expected SetTransform"),
        }
    }

    #[test]
    fn test_parse_snapshot() {
        let json = r#"{"cmd":"snapshot","path":"out.png","width":1920,"height":1080}"#;
        let req = parse_ipc_request(json).unwrap();
        match req {
            IpcRequest::Snapshot {
                path,
                width,
                height,
            } => {
                assert_eq!(path, "out.png");
                assert_eq!(width, Some(1920));
                assert_eq!(height, Some(1080));
            }
            _ => panic!("Expected Snapshot"),
        }
    }

    #[test]
    fn canonical_media_attachment_and_removal_translate_without_clamping() {
        let attach = parse_ipc_request(
            r#"{"cmd":"set_media","media":{"sigma_a":[0.1,0.2,0.3],"sigma_s":[0.4,0.5,0.6],"phase":"Isotropic","density":{"Homogeneous":{"authored_density":0.75,"mapping":{"physical_density_per_authored_unit":2.0}}},"version":9}}"#,
        )
        .unwrap();
        match ipc_request_to_viewer_cmd(&attach).unwrap().unwrap() {
            crate::viewer::viewer_enums::ViewerCmd::SetMedia {
                medium: Some(medium),
                version,
            } => {
                assert_eq!(version, 9);
                assert!(medium
                    .sigma_t()
                    .components()
                    .into_iter()
                    .zip([0.5, 0.7, 0.9])
                    .all(|(actual, expected)| (actual - expected).abs() < 1.0e-6));
            }
            _ => panic!("expected canonical media attachment"),
        }

        let remove = parse_ipc_request(r#"{"cmd":"set_media","media":null}"#).unwrap();
        assert!(matches!(
            ipc_request_to_viewer_cmd(&remove).unwrap(),
            Some(crate::viewer::viewer_enums::ViewerCmd::SetMedia {
                medium: None,
                version: 0
            })
        ));
    }

    #[test]
    fn canonical_media_rejects_a_present_malformed_version_but_defaults_absent() {
        let malformed: IpcRequest = serde_json::from_str(
            r#"{"cmd":"set_media","media":{"sigma_a":[0.1,0.1,0.1],"sigma_s":[0.2,0.2,0.2],"phase":"Isotropic","density":{"Homogeneous":{"authored_density":1.0,"mapping":{"physical_density_per_authored_unit":1.0}}},"version":"bad"}}"#,
        )
        .unwrap();
        assert!(ipc_request_to_viewer_cmd(&malformed)
            .unwrap_err()
            .contains("version must be a non-negative integer"));

        let absent: IpcRequest = serde_json::from_str(
            r#"{"cmd":"set_media","media":{"sigma_a":[0.1,0.1,0.1],"sigma_s":[0.2,0.2,0.2],"phase":"Isotropic","density":{"Homogeneous":{"authored_density":1.0,"mapping":{"physical_density_per_authored_unit":1.0}}}}}"#,
        )
        .unwrap();
        assert!(matches!(
            ipc_request_to_viewer_cmd(&absent).unwrap(),
            Some(crate::viewer::viewer_enums::ViewerCmd::SetMedia { version: 0, .. })
        ));
    }

    #[test]
    fn pick_and_manual_label_update_translate_to_execution_commands() {
        let pick = parse_ipc_request(r#"{"cmd":"pick_at","x":320,"y":200,"shift":true}"#).unwrap();
        assert!(matches!(
            ipc_request_to_viewer_cmd(&pick).unwrap(),
            Some(crate::viewer::viewer_enums::ViewerCmd::PickAt {
                x: 320,
                y: 200,
                shift: true,
                ctrl: false
            })
        ));

        let update = parse_ipc_request(r#"{"cmd":"update_labels"}"#).unwrap();
        assert!(matches!(
            ipc_request_to_viewer_cmd(&update).unwrap(),
            Some(crate::viewer::viewer_enums::ViewerCmd::UpdateLabels)
        ));
    }

    #[test]
    fn test_parse_set_terrain_scatter() {
        let json = r#"{
            "cmd":"set_terrain_scatter",
            "batches":[
                {
                    "name":"trees",
                    "color":[0.2,0.6,0.3,1.0],
                    "max_draw_distance":180.0,
                    "terrain_blend":{"enabled":true,"bury_depth":0.5,"fade_distance":2.0},
                    "terrain_contact":{"enabled":true,"distance":1.5,"strength":0.3,"vertical_weight":0.75},
                    "transforms":[[1.0,0.0,0.0,3.0,0.0,1.0,0.0,4.0,0.0,0.0,1.0,5.0,0.0,0.0,0.0,1.0]],
                    "levels":[
                        {
                            "positions":[[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]],
                            "normals":[[0.0,1.0,0.0],[0.0,1.0,0.0],[0.0,1.0,0.0]],
                            "indices":[0,1,2],
                            "max_distance":90.0
                        }
                    ]
                }
            ]
        }"#;
        let req = parse_ipc_request(json).unwrap();
        match &req {
            IpcRequest::SetTerrainScatter { batches } => {
                assert_eq!(batches.len(), 1);
                assert_eq!(batches[0].name.as_deref(), Some("trees"));
                assert_eq!(
                    batches[0].terrain_blend.as_ref().unwrap().enabled,
                    Some(true)
                );
                assert_eq!(batches[0].transforms.len(), 1);
                assert_eq!(batches[0].levels.len(), 1);
                assert_eq!(batches[0].levels[0].indices, vec![0, 1, 2]);
            }
            _ => panic!("Expected SetTerrainScatter"),
        }

        let cmd = ipc_request_to_viewer_cmd(&req).unwrap().unwrap();
        match cmd {
            crate::viewer::viewer_enums::ViewerCmd::SetTerrainScatter { batches } => {
                assert_eq!(batches.len(), 1);
                assert_eq!(batches[0].name.as_deref(), Some("trees"));
                assert!(batches[0].terrain_blend.enabled);
                assert_eq!(batches[0].terrain_contact.vertical_weight, 0.75);
                assert_eq!(batches[0].transforms.len(), 1);
                assert_eq!(batches[0].levels.len(), 1);
                assert_eq!(batches[0].levels[0].mesh.indices, vec![0, 1, 2]);
            }
            _ => panic!("Expected ViewerCmd::SetTerrainScatter"),
        }
    }

    #[cfg(feature = "enable-gpu-instancing")]
    #[test]
    fn test_parse_set_terrain_scatter_preserves_valid_wind() {
        let json = r#"{
            "cmd":"set_terrain_scatter",
            "batches":[
                {
                    "name":"trees",
                    "transforms":[[1.0,0.0,0.0,3.0,0.0,1.0,0.0,4.0,0.0,0.0,1.0,5.0,0.0,0.0,0.0,1.0]],
                    "levels":[
                        {
                            "positions":[[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]],
                            "normals":[[0.0,1.0,0.0],[0.0,1.0,0.0],[0.0,1.0,0.0]],
                            "indices":[0,1,2]
                        }
                    ],
                    "wind":{
                        "enabled":true,
                        "direction_deg":45.0,
                        "speed":1.25,
                        "amplitude":2.0,
                        "rigidity":0.3,
                        "bend_start":0.2,
                        "bend_extent":0.8,
                        "gust_strength":0.5,
                        "gust_frequency":0.4,
                        "fade_start":20.0,
                        "fade_end":80.0
                    }
                }
            ]
        }"#;
        let req = parse_ipc_request(json).unwrap();
        let cmd = ipc_request_to_viewer_cmd(&req).unwrap().unwrap();
        match cmd {
            crate::viewer::viewer_enums::ViewerCmd::SetTerrainScatter { batches } => {
                assert_eq!(batches.len(), 1);
                assert!(batches[0].wind.enabled);
                assert_eq!(batches[0].wind.direction_deg, 45.0);
                assert_eq!(batches[0].wind.speed, 1.25);
                assert_eq!(batches[0].wind.amplitude, 2.0);
                assert_eq!(batches[0].wind.rigidity, 0.3);
                assert_eq!(batches[0].wind.bend_start, 0.2);
                assert_eq!(batches[0].wind.bend_extent, 0.8);
                assert_eq!(batches[0].wind.gust_strength, 0.5);
                assert_eq!(batches[0].wind.gust_frequency, 0.4);
                assert_eq!(batches[0].wind.fade_start, 20.0);
                assert_eq!(batches[0].wind.fade_end, 80.0);
            }
            _ => panic!("Expected ViewerCmd::SetTerrainScatter"),
        }
    }

    #[cfg(feature = "enable-gpu-instancing")]
    #[test]
    fn test_parse_set_terrain_scatter_rejects_invalid_wind() {
        let json = r#"{
            "cmd":"set_terrain_scatter",
            "batches":[
                {
                    "name":"trees",
                    "transforms":[[1.0,0.0,0.0,3.0,0.0,1.0,0.0,4.0,0.0,0.0,1.0,5.0,0.0,0.0,0.0,1.0]],
                    "levels":[
                        {
                            "positions":[[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]],
                            "normals":[[0.0,1.0,0.0],[0.0,1.0,0.0],[0.0,1.0,0.0]],
                            "indices":[0,1,2]
                        }
                    ],
                    "wind":{
                        "enabled":true,
                        "amplitude":2.0,
                        "bend_extent":0.0
                    }
                }
            ]
        }"#;
        let req = parse_ipc_request(json).unwrap();
        let err = ipc_request_to_viewer_cmd(&req).unwrap_err();
        assert!(err.contains("scatter batch 0"));
        assert!(err.contains("bend_extent"));
    }

    #[test]
    fn test_parse_clear_terrain_scatter() {
        let json = r#"{"cmd":"clear_terrain_scatter"}"#;
        let req = parse_ipc_request(json).unwrap();
        match &req {
            IpcRequest::ClearTerrainScatter => {}
            _ => panic!("Expected ClearTerrainScatter"),
        }

        let cmd = ipc_request_to_viewer_cmd(&req).unwrap().unwrap();
        match cmd {
            crate::viewer::viewer_enums::ViewerCmd::ClearTerrainScatter => {}
            _ => panic!("Expected ViewerCmd::ClearTerrainScatter"),
        }
    }

    #[test]
    fn test_parse_set_point_cloud_camera_params() {
        let json = r#"{"cmd":"set_point_cloud_params","phi":0.6,"theta":0.5,"radius":1.4}"#;
        let req = parse_ipc_request(json).unwrap();
        match req {
            IpcRequest::SetPointCloudParams {
                phi, theta, radius, ..
            } => {
                assert_eq!(phi, Some(0.6));
                assert_eq!(theta, Some(0.5));
                assert_eq!(radius, Some(1.4));
            }
            _ => panic!("Expected SetPointCloudParams"),
        }
    }

    #[test]
    fn test_response_serialization() {
        let resp = IpcResponse::success();
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains(r#""ok":true"#));
        assert!(!json.contains("error"));
        assert!(!json.contains("stats"));

        let resp = IpcResponse::error("test error");
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains(r#""ok":false"#));
        assert!(json.contains("test error"));

        let resp = IpcResponse::with_active_scene_variant(None);
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains(r#""active_scene_variant":null"#));
    }

    #[test]
    fn test_response_serialization_with_created_id() {
        let resp = IpcResponse::with_id(42);
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains(r#""ok":true"#));
        assert!(json.contains(r#""id":42"#));
    }

    #[test]
    fn test_parse_label_and_overlay_create_ids() {
        let label_json = r#"{"cmd":"add_label","id":42,"text":"City","world_pos":[1.0,2.0,3.0]}"#;
        let req = parse_ipc_request(label_json).unwrap();
        match &req {
            IpcRequest::AddLabel { id, text, .. } => {
                assert_eq!(*id, Some(42));
                assert_eq!(text, "City");
            }
            _ => panic!("Expected AddLabel"),
        }

        let cmd = ipc_request_to_viewer_cmd(&req).unwrap().unwrap();
        match cmd {
            crate::viewer::viewer_enums::ViewerCmd::AddLabel { id, text, .. } => {
                assert_eq!(id, Some(42));
                assert_eq!(text, "City");
            }
            _ => panic!("Expected ViewerCmd::AddLabel"),
        }

        let overlay_json = r#"{
            "cmd":"add_vector_overlay",
            "id":7,
            "name":"label-halo",
            "vertices":[[0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0]],
            "indices":[0],
            "primitive":"points"
        }"#;
        let req = parse_ipc_request(overlay_json).unwrap();
        match &req {
            IpcRequest::AddVectorOverlay { id, name, .. } => {
                assert_eq!(*id, Some(7));
                assert_eq!(name, "label-halo");
            }
            _ => panic!("Expected AddVectorOverlay"),
        }

        let cmd = ipc_request_to_viewer_cmd(&req).unwrap().unwrap();
        match cmd {
            crate::viewer::viewer_enums::ViewerCmd::AddVectorOverlay { id, name, .. } => {
                assert_eq!(id, Some(7));
                assert_eq!(name, "label-halo");
            }
            _ => panic!("Expected ViewerCmd::AddVectorOverlay"),
        }
    }

    #[test]
    fn test_parse_set_scene_review_state() {
        let json = r#"{
            "cmd":"set_scene_review_state",
            "state":{
                "base_state":{
                    "labels":[{"kind":"point","text":"Base","world_pos":[0.0,0.0,0.0]}]
                },
                "review_layers":[
                    {
                        "id":"notes",
                        "name":"Notes",
                        "labels":[{"kind":"point","text":"Note","world_pos":[1.0,0.0,0.0]}]
                    }
                ],
                "variants":[
                    {
                        "id":"review",
                        "active_layer_ids":["notes"],
                        "preset":{"exposure":2.0}
                    }
                ],
                "active_variant_id":"review"
            }
        }"#;
        let req = parse_ipc_request(json).unwrap();
        match &req {
            IpcRequest::SetSceneReviewState { state } => {
                assert_eq!(state.review_layers.len(), 1);
                assert_eq!(state.variants.len(), 1);
                assert_eq!(state.active_variant_id.as_deref(), Some("review"));
            }
            _ => panic!("Expected SetSceneReviewState"),
        }

        let cmd = ipc_request_to_viewer_cmd(&req).unwrap().unwrap();
        match cmd {
            crate::viewer::viewer_enums::ViewerCmd::SetSceneReviewState { state } => {
                assert_eq!(state.review_layers.len(), 1);
                assert_eq!(
                    state.snapshot().active_scene_variant.as_deref(),
                    Some("review")
                );
                assert_eq!(state.snapshot().scene_variants[0].id, "review");
            }
            _ => panic!("Expected ViewerCmd::SetSceneReviewState"),
        }
    }

    #[test]
    fn test_parse_set_scene_review_state_rejects_duplicate_layer_ids() {
        let json = r#"{
            "cmd":"set_scene_review_state",
            "state":{
                "review_layers":[{"id":"dup"},{"id":"dup"}],
                "variants":[]
            }
        }"#;
        let req = parse_ipc_request(json).unwrap();
        let err = ipc_request_to_viewer_cmd(&req).unwrap_err();
        assert!(err.contains("Duplicate review layer ID"));
    }

    #[test]
    fn test_parse_set_scene_review_state_rejects_missing_variant_layers() {
        let json = r#"{
            "cmd":"set_scene_review_state",
            "state":{
                "review_layers":[{"id":"known"}],
                "variants":[{"id":"review","active_layer_ids":["missing"]}]
            }
        }"#;
        let req = parse_ipc_request(json).unwrap();
        let err = ipc_request_to_viewer_cmd(&req).unwrap_err();
        assert!(err.contains("unknown review layer ID"));
    }

    #[test]
    fn test_scene_review_query_requests_parse_as_special() {
        for json in [
            r#"{"cmd":"list_scene_variants"}"#,
            r#"{"cmd":"list_review_layers"}"#,
            r#"{"cmd":"get_active_scene_variant"}"#,
        ] {
            let req = parse_ipc_request(json).unwrap();
            assert!(ipc_request_to_viewer_cmd(&req).unwrap().is_none());
        }

        let apply =
            parse_ipc_request(r#"{"cmd":"apply_scene_variant","variant_id":"review"}"#).unwrap();
        let toggle = parse_ipc_request(
            r#"{"cmd":"set_review_layer_visible","layer_id":"notes","visible":true}"#,
        )
        .unwrap();

        assert!(matches!(
            ipc_request_to_viewer_cmd(&apply).unwrap().unwrap(),
            crate::viewer::viewer_enums::ViewerCmd::ApplySceneVariant { .. }
        ));
        assert!(matches!(
            ipc_request_to_viewer_cmd(&toggle).unwrap().unwrap(),
            crate::viewer::viewer_enums::ViewerCmd::SetReviewLayerVisible { .. }
        ));
    }

    #[cfg(feature = "enable-gpu-instancing")]
    #[test]
    fn test_parse_set_terrain_scatter_with_hlod() {
        let json = r#"{
            "cmd":"set_terrain_scatter",
            "batches":[
                {
                    "name":"trees",
                    "color":[0.2,0.6,0.3,1.0],
                    "max_draw_distance":180.0,
                    "transforms":[[1.0,0.0,0.0,3.0,0.0,1.0,0.0,4.0,0.0,0.0,1.0,5.0,0.0,0.0,0.0,1.0]],
                    "levels":[
                        {
                            "positions":[[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]],
                            "normals":[[0.0,1.0,0.0],[0.0,1.0,0.0],[0.0,1.0,0.0]],
                            "indices":[0,1,2],
                            "max_distance":90.0
                        }
                    ],
                    "hlod":{
                        "hlod_distance":100.0,
                        "cluster_radius":25.0,
                        "simplify_ratio":0.5
                    }
                }
            ]
        }"#;
        let req = parse_ipc_request(json).unwrap();
        match &req {
            IpcRequest::SetTerrainScatter { batches } => {
                assert_eq!(batches.len(), 1);
                let hlod = batches[0].hlod.as_ref().expect("hlod should be present");
                assert_eq!(hlod.hlod_distance, 100.0);
                assert_eq!(hlod.cluster_radius, 25.0);
                assert_eq!(hlod.simplify_ratio, 0.5);
            }
            _ => panic!("Expected SetTerrainScatter"),
        }

        let cmd = ipc_request_to_viewer_cmd(&req).unwrap().unwrap();
        match cmd {
            crate::viewer::viewer_enums::ViewerCmd::SetTerrainScatter { batches } => {
                assert_eq!(batches.len(), 1);
                let hlod_config = batches[0]
                    .hlod_config
                    .as_ref()
                    .expect("hlod_config should be present");
                assert_eq!(hlod_config.hlod_distance, 100.0);
                assert_eq!(hlod_config.cluster_radius, 25.0);
                assert_eq!(hlod_config.simplify_ratio, 0.5);
            }
            _ => panic!("Expected ViewerCmd::SetTerrainScatter"),
        }
    }

    #[cfg(feature = "enable-gpu-instancing")]
    #[test]
    fn test_parse_set_terrain_scatter_without_hlod_backward_compat() {
        let json = r#"{
            "cmd":"set_terrain_scatter",
            "batches":[
                {
                    "name":"rocks",
                    "transforms":[[1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0]],
                    "levels":[
                        {
                            "positions":[[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]],
                            "indices":[0,1,2]
                        }
                    ]
                }
            ]
        }"#;
        let req = parse_ipc_request(json).unwrap();
        match &req {
            IpcRequest::SetTerrainScatter { batches } => {
                assert_eq!(batches.len(), 1);
                assert!(
                    batches[0].hlod.is_none(),
                    "hlod should be None when omitted"
                );
            }
            _ => panic!("Expected SetTerrainScatter"),
        }

        let cmd = ipc_request_to_viewer_cmd(&req).unwrap().unwrap();
        match cmd {
            crate::viewer::viewer_enums::ViewerCmd::SetTerrainScatter { batches } => {
                assert!(
                    batches[0].hlod_config.is_none(),
                    "hlod_config should be None when omitted"
                );
            }
            _ => panic!("Expected ViewerCmd::SetTerrainScatter"),
        }
    }
}
