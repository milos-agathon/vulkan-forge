use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct Uniforms {
    pub width: u32,
    pub height: u32,
    pub frame_index: u32,
    pub aov_flags: u32,
    pub cam_origin: [f32; 3],
    pub cam_fov_y: f32,
    pub cam_right: [f32; 3],
    pub cam_aspect: f32,
    pub cam_up: [f32; 3],
    pub cam_exposure: f32,
    pub cam_forward: [f32; 3],
    pub seed_hi: u32,
    pub seed_lo: u32,
    pub camera_model: u32,
    pub full_width: u32,
    pub full_height: u32,
    pub pixel_offset_x: u32,
    pub pixel_offset_y: u32,
    pub ortho_half_height: f32,
    pub camera_flags: u32,
    pub sensor_rect: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct Sphere {
    pub center: [f32; 3],
    pub radius: f32,
    pub albedo: [f32; 3],
    pub metallic: f32,
    pub emissive: [f32; 3],
    pub roughness: f32,
    pub ior: f32,
    pub ax: f32,
    pub ay: f32,
    pub _pad1: f32,
}
