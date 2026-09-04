use super::*;

fn mip_extent(extent: u32, mip: u32) -> u32 {
    (extent >> mip).max(1)
}

fn proportional_bounds(index: u32, src: u32, dst: u32) -> std::ops::Range<u32> {
    let lo = index * src / dst;
    let hi = ((index + 1) * src).div_ceil(dst).min(src);
    lo..hi
}

fn max_reduce_to(
    source: &[f32],
    width: u32,
    height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Vec<f32> {
    let mut result = vec![0.0; (dst_width * dst_height) as usize];
    for y in 0..dst_height {
        for x in 0..dst_width {
            let mut reduced = 0.0_f32;
            for sy in proportional_bounds(y, height, dst_height) {
                for sx in proportional_bounds(x, width, dst_width) {
                    reduced = reduced.max(source[(sy * width + sx) as usize]);
                }
            }
            result[(y * dst_width + x) as usize] = reduced;
        }
    }
    result
}

fn max_reduce(source: &[f32], width: u32, height: u32) -> Vec<f32> {
    max_reduce_to(
        source,
        width,
        height,
        mip_extent(width, 1),
        mip_extent(height, 1),
    )
}

#[test]
fn same_size_initial_pass_remains_an_exact_copy() {
    let source: Vec<f32> = (0u16..35).map(|index| f32::from(index) / 34.0).collect();
    assert_eq!(max_reduce_to(&source, 7, 5, 7, 5), source);
}

#[test]
fn fused_half_resolution_max_is_proportional_and_conservative() {
    assert_eq!(
        max_reduce(&(0u16..16).map(f32::from).collect::<Vec<_>>(), 4, 4),
        [5.0, 7.0, 13.0, 15.0],
    );
    assert_eq!(
        max_reduce(&(0u16..15).map(f32::from).collect::<Vec<_>>(), 5, 3),
        [12.0, 14.0],
    );

    for (width, height) in [(8, 6), (7, 5), (2, 9)] {
        let source: Vec<f32> = (0..width * height)
            .map(|index| {
                f32::from(u16::try_from(index * 37 % 101).expect("value is below 101")) / 100.0
            })
            .collect();
        let fused = max_reduce(&source, width, height);

        let dst_width = mip_extent(width, 1);
        let dst_height = mip_extent(height, 1);
        for y in 0..dst_height {
            for x in 0..dst_width {
                let reduced = fused[(y * dst_width + x) as usize];
                for sy in proportional_bounds(y, height, dst_height) {
                    for sx in proportional_bounds(x, width, dst_width) {
                        assert!(reduced >= source[(sy * width + sx) as usize]);
                    }
                }
            }
        }
    }
}

#[test]
fn fused_initial_write_is_one_quarter_at_win2_resolution() {
    let full = 3840_u64 * 2160;
    let half = u64::from(mip_extent(3840, 1)) * u64::from(mip_extent(2160, 1));
    assert_eq!(half * 4, full);
}

#[test]
fn initial_shader_uses_proportional_bounds_and_max_reduction() {
    naga::front::wgsl::parse_str(HZB_BUILD_SOURCE).expect("valid HZB build WGSL");
    let generic = HZB_BUILD_SOURCE
        .split_once("fn cs_copy")
        .map(|(_, entry)| entry)
        .expect("generic HZB copy entry point");
    let generic = generic
        .split_once("fn cs_copy_max_reduce")
        .map(|(entry, _)| entry)
        .expect("generic HZB copy body");
    assert!(generic.contains("let depth = textureLoad(depth_in, gid.xy, 0);"));

    let entry = HZB_BUILD_SOURCE
        .split_once("fn cs_copy_max_reduce")
        .map(|(_, entry)| entry)
        .expect("terrain HZB MAX-seed entry point");
    let entry = entry
        .split_once("fn cs_downsample")
        .map(|(entry, _)| entry)
        .expect("initial HZB entry body");
    assert!(entry.contains("let src_lo = gid.xy * src_dims / dst_dims;"));
    assert!(entry.contains("(gid.xy + 1u) * src_dims + dst_dims - 1u"));
    assert!(entry.contains("reduced = max(reduced, textureLoad(depth_in"));
}
