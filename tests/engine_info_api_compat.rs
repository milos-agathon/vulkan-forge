use forge3d::core::context::EngineInfo;

#[test]
fn engine_info_public_layout_remains_the_original_seven_fields() {
    let info = EngineInfo {
        backend: "metal".to_string(),
        adapter_name: "Apple M4".to_string(),
        device_name: "Apple M4".to_string(),
        max_texture_dimension_2d: 16_384,
        max_buffer_size: 1,
        device_type: "integratedgpu".to_string(),
        software_fallback: false,
    };

    let EngineInfo {
        backend,
        adapter_name,
        device_name,
        max_texture_dimension_2d,
        max_buffer_size,
        device_type,
        software_fallback,
    } = info;
    let _: (String, String, String, u32, u64, String, bool) = (
        backend,
        adapter_name,
        device_name,
        max_texture_dimension_2d,
        max_buffer_size,
        device_type,
        software_fallback,
    );
}
