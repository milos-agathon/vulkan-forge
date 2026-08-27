/*!
# Metal API internals.

## Pipeline Layout

In Metal, push constants, vertex buffers, and resources in the bind groups
are all placed together in the native resource bindings, which work similarly to D3D11:
there are tables of textures, buffers, and samplers.

We put push constants first (if any) in the table, followed by bind group 0
resources, followed by other bind groups. The vertex buffers are bound at the very
end of the VS buffer table.

!*/

// `MTLFeatureSet` is superseded by `MTLGpuFamily`.
// However, `MTLGpuFamily` is only supported starting MacOS 10.15, whereas our minimum target is MacOS 10.13,
// See https://github.com/gpuweb/gpuweb/issues/1069 for minimum spec.
// TODO: Eventually all deprecated features should be abstracted and use new api when available.
#[allow(deprecated)]
mod adapter;
mod command;
mod conv;
mod device;
mod surface;
mod time;

use std::{
    fmt, iter, ops,
    ptr::NonNull,
    sync::{atomic, Arc},
    thread,
};

use arrayvec::ArrayVec;
use bitflags::bitflags;
use metal::foreign_types::ForeignTypeRef as _;
use objc::{sel, sel_impl};
use parking_lot::{Mutex, RwLock};

#[derive(Clone, Debug)]
pub struct Api;

type ResourceIndex = u32;

impl crate::Api for Api {
    type Instance = Instance;
    type Surface = Surface;
    type Adapter = Adapter;
    type Device = Device;

    type Queue = Queue;
    type CommandEncoder = CommandEncoder;
    type CommandBuffer = CommandBuffer;

    type Buffer = Buffer;
    type Texture = Texture;
    type SurfaceTexture = SurfaceTexture;
    type TextureView = TextureView;
    type Sampler = Sampler;
    type QuerySet = QuerySet;
    type Fence = Fence;

    type BindGroupLayout = BindGroupLayout;
    type BindGroup = BindGroup;
    type PipelineLayout = PipelineLayout;
    type ShaderModule = ShaderModule;
    type RenderPipeline = RenderPipeline;
    type ComputePipeline = ComputePipeline;

    type AccelerationStructure = AccelerationStructure;
}

pub struct Instance {
    managed_metal_layer_delegate: surface::HalManagedMetalLayerDelegate,
}

impl Instance {
    pub fn create_surface_from_layer(&self, layer: &metal::MetalLayerRef) -> Surface {
        unsafe { Surface::from_layer(layer) }
    }
}

impl crate::Instance<Api> for Instance {
    unsafe fn init(_desc: &crate::InstanceDescriptor) -> Result<Self, crate::InstanceError> {
        profiling::scope!("Init Metal Backend");
        // We do not enable metal validation based on the validation flags as it affects the entire
        // process. Instead, we enable the validation inside the test harness itself in tests/src/native.rs.
        Ok(Instance {
            managed_metal_layer_delegate: surface::HalManagedMetalLayerDelegate::new(),
        })
    }

    unsafe fn create_surface(
        &self,
        _display_handle: raw_window_handle::RawDisplayHandle,
        window_handle: raw_window_handle::RawWindowHandle,
    ) -> Result<Surface, crate::InstanceError> {
        match window_handle {
            #[cfg(target_os = "ios")]
            raw_window_handle::RawWindowHandle::UiKit(handle) => {
                let _ = &self.managed_metal_layer_delegate;
                Ok(unsafe { Surface::from_view(handle.ui_view.as_ptr(), None) })
            }
            #[cfg(target_os = "macos")]
            raw_window_handle::RawWindowHandle::AppKit(handle) => Ok(unsafe {
                Surface::from_view(
                    handle.ns_view.as_ptr(),
                    Some(&self.managed_metal_layer_delegate),
                )
            }),
            _ => Err(crate::InstanceError::new(format!(
                "window handle {window_handle:?} is not a Metal-compatible handle"
            ))),
        }
    }

    unsafe fn destroy_surface(&self, surface: Surface) {
        unsafe { surface.dispose() };
    }

    unsafe fn enumerate_adapters(&self) -> Vec<crate::ExposedAdapter<Api>> {
        let devices = metal::Device::all();
        let mut adapters: Vec<crate::ExposedAdapter<Api>> = devices
            .into_iter()
            .map(|dev| {
                let name = dev.name().into();
                let shared = AdapterShared::new(dev);
                crate::ExposedAdapter {
                    info: wgt::AdapterInfo {
                        name,
                        vendor: 0,
                        device: 0,
                        device_type: shared.private_caps.device_type(),
                        driver: String::new(),
                        driver_info: String::new(),
                        backend: wgt::Backend::Metal,
                    },
                    features: shared.private_caps.features(),
                    capabilities: shared.private_caps.capabilities(),
                    adapter: Adapter::new(Arc::new(shared)),
                }
            })
            .collect();
        adapters.sort_by_key(|ad| {
            (
                ad.adapter.shared.private_caps.low_power,
                ad.adapter.shared.private_caps.headless,
            )
        });
        adapters
    }
}

bitflags!(
    /// Similar to `MTLCounterSamplingPoint`, but a bit higher abstracted for our purposes.
    #[derive(Debug, Copy, Clone)]
    pub struct TimestampQuerySupport: u32 {
        /// On creating Metal encoders.
        const STAGE_BOUNDARIES = 1 << 1;
        /// Within existing draw encoders.
        const ON_RENDER_ENCODER = Self::STAGE_BOUNDARIES.bits() | (1 << 2);
        /// Within existing dispatch encoders.
        const ON_COMPUTE_ENCODER = Self::STAGE_BOUNDARIES.bits() | (1 << 3);
        /// Within existing blit encoders.
        const ON_BLIT_ENCODER = Self::STAGE_BOUNDARIES.bits() | (1 << 4);

        /// Within any wgpu render/compute pass.
        const INSIDE_WGPU_PASSES = Self::ON_RENDER_ENCODER.bits() | Self::ON_COMPUTE_ENCODER.bits();
    }
);

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct PrivateCapabilities {
    family_check: bool,
    msl_version: metal::MTLLanguageVersion,
    fragment_rw_storage: bool,
    read_write_texture_tier: metal::MTLReadWriteTextureTier,
    msaa_desktop: bool,
    msaa_apple3: bool,
    msaa_apple7: bool,
    resource_heaps: bool,
    argument_buffers: bool,
    shared_textures: bool,
    mutable_comparison_samplers: bool,
    sampler_clamp_to_border: bool,
    indirect_draw_dispatch: bool,
    base_vertex_first_instance_drawing: bool,
    dual_source_blending: bool,
    low_power: bool,
    headless: bool,
    layered_rendering: bool,
    function_specialization: bool,
    depth_clip_mode: bool,
    texture_cube_array: bool,
    supports_float_filtering: bool,
    format_depth24_stencil8: bool,
    format_depth32_stencil8_filter: bool,
    format_depth32_stencil8_none: bool,
    format_min_srgb_channels: u8,
    format_b5: bool,
    format_bc: bool,
    format_eac_etc: bool,
    format_astc: bool,
    format_astc_hdr: bool,
    format_any8_unorm_srgb_all: bool,
    format_any8_unorm_srgb_no_write: bool,
    format_any8_snorm_all: bool,
    format_r16_norm_all: bool,
    format_r32_all: bool,
    format_r32_no_write: bool,
    format_r32float_no_write_no_filter: bool,
    format_r32float_no_filter: bool,
    format_r32float_all: bool,
    format_rgba8_srgb_all: bool,
    format_rgba8_srgb_no_write: bool,
    format_rgb10a2_unorm_all: bool,
    format_rgb10a2_unorm_no_write: bool,
    format_rgb10a2_uint_write: bool,
    format_rg11b10_all: bool,
    format_rg11b10_no_write: bool,
    format_rgb9e5_all: bool,
    format_rgb9e5_no_write: bool,
    format_rgb9e5_filter_only: bool,
    format_rg32_color: bool,
    format_rg32_color_write: bool,
    format_rg32float_all: bool,
    format_rg32float_color_blend: bool,
    format_rg32float_no_filter: bool,
    format_rgba32int_color: bool,
    format_rgba32int_color_write: bool,
    format_rgba32float_color: bool,
    format_rgba32float_color_write: bool,
    format_rgba32float_all: bool,
    format_depth16unorm: bool,
    format_depth32float_filter: bool,
    format_depth32float_none: bool,
    format_bgr10a2_all: bool,
    format_bgr10a2_no_write: bool,
    max_buffers_per_stage: ResourceIndex,
    max_vertex_buffers: ResourceIndex,
    max_textures_per_stage: ResourceIndex,
    max_samplers_per_stage: ResourceIndex,
    buffer_alignment: u64,
    max_buffer_size: u64,
    max_texture_size: u64,
    max_texture_3d_size: u64,
    max_texture_layers: u64,
    max_fragment_input_components: u64,
    max_color_render_targets: u8,
    max_varying_components: u32,
    max_threads_per_group: u32,
    max_total_threadgroup_memory: u32,
    sample_count_mask: crate::TextureFormatCapabilities,
    supports_debug_markers: bool,
    supports_binary_archives: bool,
    supports_capture_manager: bool,
    can_set_maximum_drawables_count: bool,
    can_set_display_sync: bool,
    can_set_next_drawable_timeout: bool,
    supports_arrays_of_textures: bool,
    supports_arrays_of_textures_write: bool,
    supports_mutability: bool,
    supports_depth_clip_control: bool,
    supports_preserve_invariance: bool,
    supports_shader_primitive_index: bool,
    has_unified_memory: Option<bool>,
    timestamp_query_support: TimestampQuerySupport,
    /// macOS 26 reports legacy counter capabilities but only the render-pass
    /// start-of-vertex boundary has been observed to produce timestamp ticks.
    timestamp_query_requires_render_boundary: bool,
}

#[derive(Clone, Debug)]
struct PrivateDisabilities {
    /// Near depth is not respected properly on some Intel GPUs.
    broken_viewport_near_depth: bool,
    /// Multi-target clears don't appear to work properly on Intel GPUs.
    #[allow(dead_code)]
    broken_layered_clear_image: bool,
}

#[derive(Debug, Default)]
struct Settings {
    retain_command_buffer_references: bool,
}

struct AdapterShared {
    device: Mutex<metal::Device>,
    disabilities: PrivateDisabilities,
    private_caps: PrivateCapabilities,
    settings: Settings,
    presentation_timer: time::PresentationTimer,
}

unsafe impl Send for AdapterShared {}
unsafe impl Sync for AdapterShared {}

/// State shared by every Metal command encoder and segmented command buffer
/// created from one queue. The counter covers each encoder-owned native buffer
/// segment, so deferred timestamp resolves cannot leak buffers when an encoder
/// is discarded. Internal buffers that are committed immediately (for example,
/// an empty-submit fence signal) do not enter this outstanding set.
#[derive(Debug)]
struct QueueShared {
    raw: Arc<Mutex<metal::CommandQueue>>,
    command_buffer_created_not_submitted: atomic::AtomicUsize,
    async_device_error: Arc<atomic::AtomicU8>,
}

impl QueueShared {
    fn new(raw: metal::CommandQueue) -> Arc<Self> {
        Arc::new(Self {
            raw: Arc::new(Mutex::new(raw)),
            command_buffer_created_not_submitted: atomic::AtomicUsize::new(0),
            async_device_error: Arc::new(atomic::AtomicU8::new(ASYNC_DEVICE_ERROR_NONE)),
        })
    }
}

unsafe impl Send for QueueShared {}
unsafe impl Sync for QueueShared {}

impl AdapterShared {
    fn new(device: metal::Device) -> Self {
        let private_caps = PrivateCapabilities::new(&device);
        log::debug!("{:#?}", private_caps);

        Self {
            disabilities: PrivateDisabilities::new(&device),
            private_caps,
            device: Mutex::new(device),
            settings: Settings::default(),
            presentation_timer: time::PresentationTimer::new(),
        }
    }
}

pub struct Adapter {
    shared: Arc<AdapterShared>,
}

pub struct Queue {
    shared: Arc<QueueShared>,
    timestamp_period: f32,
}

unsafe impl Send for Queue {}
unsafe impl Sync for Queue {}

impl Queue {
    pub unsafe fn queue_from_raw(raw: metal::CommandQueue, timestamp_period: f32) -> Self {
        Self {
            shared: QueueShared::new(raw),
            timestamp_period,
        }
    }
}

pub struct Device {
    shared: Arc<AdapterShared>,
    features: wgt::Features,
}

pub struct Surface {
    view: Option<NonNull<objc::runtime::Object>>,
    render_layer: Mutex<metal::MetalLayer>,
    swapchain_format: RwLock<Option<wgt::TextureFormat>>,
    extent: RwLock<wgt::Extent3d>,
    main_thread_id: thread::ThreadId,
    // Useful for UI-intensive applications that are sensitive to
    // window resizing.
    pub present_with_transaction: bool,
}

unsafe impl Send for Surface {}
unsafe impl Sync for Surface {}

#[derive(Debug)]
pub struct SurfaceTexture {
    texture: Texture,
    drawable: metal::MetalDrawable,
    present_with_transaction: bool,
}

impl std::borrow::Borrow<Texture> for SurfaceTexture {
    fn borrow(&self) -> &Texture {
        &self.texture
    }
}

unsafe impl Send for SurfaceTexture {}
unsafe impl Sync for SurfaceTexture {}

impl crate::Queue<Api> for Queue {
    unsafe fn submit(
        &self,
        command_buffers: &[&CommandBuffer],
        signal_fence: Option<(&mut Fence, crate::FenceValue)>,
    ) -> Result<(), crate::DeviceError> {
        objc::rc::autoreleasepool(|| {
            if let Some((fence, _)) = signal_fence.as_ref() {
                if let Some(error) = fence.device_error() {
                    return Err(error);
                }
            }
            if let Some(error) = decode_device_error(
                self.shared
                    .async_device_error
                    .load(atomic::Ordering::Acquire),
            ) {
                return Err(error);
            }
            if command_buffers
                .iter()
                .any(|command_buffer| command_buffer.submitted.load(atomic::Ordering::Acquire))
            {
                return Err(crate::DeviceError::Lost);
            }

            let submission_buffers: Vec<_> = command_buffers
                .iter()
                .flat_map(|command_buffer| {
                    command_buffer
                        .segments
                        .iter()
                        .map(|segment| segment.raw().to_owned())
                })
                .collect();
            let has_deferred_timestamp_resolve = command_buffers.iter().any(|command_buffer| {
                command_buffer.segments.iter().any(|segment| {
                    matches!(segment, CommandBufferSegment::DeferredTimestampResolve(_))
                })
            });
            let submission_error = signal_fence
                .as_ref()
                .map(|(fence, _)| Arc::clone(&fence.async_device_error))
                .unwrap_or_else(|| Arc::clone(&self.shared.async_device_error));

            if has_deferred_timestamp_resolve {
                // Reserve the complete execution order before committing any
                // predecessor.  The pre-created resolve and continuation are
                // therefore ordered behind the sampling command buffer even
                // though the resolve is encoded by its completion handler.
                for command_buffer in &submission_buffers {
                    command_buffer.enqueue();
                }
            }

            let extra_command_buffer = match signal_fence {
                Some((fence, value)) => {
                    let completed_value = Arc::clone(&fence.completed_value);
                    let async_device_error = Arc::clone(&submission_error);
                    let queue_error = Arc::clone(&self.shared.async_device_error);
                    let status_buffers = submission_buffers.clone();
                    let block = block::ConcreteBlock::new(
                        move |signal_buffer: &metal::CommandBufferRef| {
                            let mut succeeded = true;
                            if status_buffers.is_empty() {
                                if signal_buffer.status()
                                    != metal::MTLCommandBufferStatus::Completed
                                {
                                    succeeded = false;
                                    let error = if signal_buffer.status()
                                        == metal::MTLCommandBufferStatus::Error
                                    {
                                        command_buffer_device_error(signal_buffer)
                                    } else {
                                        crate::DeviceError::Lost
                                    };
                                    record_async_device_error(&async_device_error, &error);
                                    record_async_device_error(&queue_error, &error);
                                }
                            } else {
                                for command_buffer in &status_buffers {
                                    match command_buffer.status() {
                                        metal::MTLCommandBufferStatus::Completed => {}
                                        metal::MTLCommandBufferStatus::Error => {
                                            succeeded = false;
                                            let error = command_buffer_device_error(command_buffer);
                                            record_async_device_error(&async_device_error, &error);
                                            record_async_device_error(&queue_error, &error);
                                        }
                                        _ => {
                                            succeeded = false;
                                            let error = crate::DeviceError::Lost;
                                            record_async_device_error(&async_device_error, &error);
                                            record_async_device_error(&queue_error, &error);
                                        }
                                    }
                                }
                            }
                            if succeeded {
                                completed_value.store(value, atomic::Ordering::Release);
                            }
                        },
                    )
                    .copy();

                    let raw = match command_buffers.last() {
                        Some(command_buffer) => command_buffer.final_raw().to_owned(),
                        None => {
                            let queue = self.shared.raw.lock();
                            queue
                                .new_command_buffer_with_unretained_references()
                                .to_owned()
                        }
                    };
                    raw.set_label("(wgpu internal) Signal");
                    raw.add_completed_handler(&block);

                    fence.maintain();
                    fence.pending_command_buffers.push((value, raw.to_owned()));
                    if command_buffers.is_empty() {
                        Some(raw)
                    } else {
                        None
                    }
                }
                None => None,
            };

            // Every predecessor handler must be attached before the first
            // predecessor is committed.  Otherwise a very short predecessor
            // can complete before its resolve gate exists.
            for command_buffer in command_buffers {
                for segment in &command_buffer.segments {
                    if let CommandBufferSegment::DeferredTimestampResolve(resolve) = segment {
                        let resolve = Arc::clone(resolve);
                        let resolve_for_handler = Arc::clone(&resolve);
                        let queue_shared = Arc::clone(&self.shared);
                        let async_device_error = Arc::clone(&submission_error);
                        let block = block::ConcreteBlock::new(move |_predecessor| {
                            commit_deferred_timestamp_resolve(
                                &resolve_for_handler,
                                Arc::clone(&queue_shared),
                                &async_device_error,
                            );
                        })
                        .copy();
                        resolve.predecessor.add_completed_handler(&block);
                    }
                }
            }

            for command_buffer in command_buffers {
                let was_submitted = command_buffer
                    .submitted
                    .swap(true, atomic::Ordering::AcqRel);
                debug_assert!(!was_submitted);
                for segment in &command_buffer.segments {
                    if let CommandBufferSegment::Commands { raw, deferred } = segment {
                        if !deferred {
                            raw.commit();
                            decrement_unsubmitted(&command_buffer.queue_shared);
                        }
                    }
                }
            }

            if let Some(raw) = extra_command_buffer {
                raw.commit();
            }
            Ok(())
        })
    }
    unsafe fn present(
        &self,
        _surface: &Surface,
        texture: SurfaceTexture,
    ) -> Result<(), crate::SurfaceError> {
        let queue = &self.shared.raw.lock();
        objc::rc::autoreleasepool(|| {
            let command_buffer = queue.new_command_buffer();
            command_buffer.set_label("(wgpu internal) Present");

            // https://developer.apple.com/documentation/quartzcore/cametallayer/1478157-presentswithtransaction?language=objc
            if !texture.present_with_transaction {
                command_buffer.present_drawable(&texture.drawable);
            }

            command_buffer.commit();

            if texture.present_with_transaction {
                command_buffer.wait_until_scheduled();
                texture.drawable.present();
            }
        });
        Ok(())
    }

    unsafe fn get_timestamp_period(&self) -> f32 {
        self.timestamp_period
    }
}

#[derive(Debug)]
pub struct Buffer {
    raw: metal::Buffer,
    size: wgt::BufferAddress,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    fn as_raw(&self) -> BufferPtr {
        unsafe { NonNull::new_unchecked(self.raw.as_ptr()) }
    }
}

#[derive(Debug)]
pub struct Texture {
    raw: metal::Texture,
    format: wgt::TextureFormat,
    raw_type: metal::MTLTextureType,
    array_layers: u32,
    mip_levels: u32,
    copy_size: crate::CopyExtent,
}

unsafe impl Send for Texture {}
unsafe impl Sync for Texture {}

#[derive(Debug)]
pub struct TextureView {
    raw: metal::Texture,
    aspects: crate::FormatAspects,
}

unsafe impl Send for TextureView {}
unsafe impl Sync for TextureView {}

impl TextureView {
    fn as_raw(&self) -> TexturePtr {
        unsafe { NonNull::new_unchecked(self.raw.as_ptr()) }
    }
}

#[derive(Debug)]
pub struct Sampler {
    raw: metal::SamplerState,
}

unsafe impl Send for Sampler {}
unsafe impl Sync for Sampler {}

impl Sampler {
    fn as_raw(&self) -> SamplerPtr {
        unsafe { NonNull::new_unchecked(self.raw.as_ptr()) }
    }
}

#[derive(Debug)]
pub struct BindGroupLayout {
    /// Sorted list of BGL entries.
    entries: Arc<[wgt::BindGroupLayoutEntry]>,
}

#[derive(Clone, Debug, Default)]
struct ResourceData<T> {
    buffers: T,
    textures: T,
    samplers: T,
}

#[derive(Clone, Debug, Default)]
struct MultiStageData<T> {
    vs: T,
    fs: T,
    cs: T,
}

const NAGA_STAGES: MultiStageData<naga::ShaderStage> = MultiStageData {
    vs: naga::ShaderStage::Vertex,
    fs: naga::ShaderStage::Fragment,
    cs: naga::ShaderStage::Compute,
};

impl<T> ops::Index<naga::ShaderStage> for MultiStageData<T> {
    type Output = T;
    fn index(&self, stage: naga::ShaderStage) -> &T {
        match stage {
            naga::ShaderStage::Vertex => &self.vs,
            naga::ShaderStage::Fragment => &self.fs,
            naga::ShaderStage::Compute => &self.cs,
        }
    }
}

impl<T> MultiStageData<T> {
    fn map_ref<Y>(&self, fun: impl Fn(&T) -> Y) -> MultiStageData<Y> {
        MultiStageData {
            vs: fun(&self.vs),
            fs: fun(&self.fs),
            cs: fun(&self.cs),
        }
    }
    fn map<Y>(self, fun: impl Fn(T) -> Y) -> MultiStageData<Y> {
        MultiStageData {
            vs: fun(self.vs),
            fs: fun(self.fs),
            cs: fun(self.cs),
        }
    }
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T> {
        iter::once(&self.vs)
            .chain(iter::once(&self.fs))
            .chain(iter::once(&self.cs))
    }
    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut T> {
        iter::once(&mut self.vs)
            .chain(iter::once(&mut self.fs))
            .chain(iter::once(&mut self.cs))
    }
}

type MultiStageResourceCounters = MultiStageData<ResourceData<ResourceIndex>>;
type MultiStageResources = MultiStageData<naga::back::msl::EntryPointResources>;

#[derive(Debug)]
struct BindGroupLayoutInfo {
    base_resource_indices: MultiStageResourceCounters,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct PushConstantsInfo {
    count: u32,
    buffer_index: ResourceIndex,
}

#[derive(Debug)]
pub struct PipelineLayout {
    bind_group_infos: ArrayVec<BindGroupLayoutInfo, { crate::MAX_BIND_GROUPS }>,
    push_constants_infos: MultiStageData<Option<PushConstantsInfo>>,
    total_counters: MultiStageResourceCounters,
    total_push_constants: u32,
    per_stage_map: MultiStageResources,
}

trait AsNative {
    type Native;
    fn from(native: &Self::Native) -> Self;
    fn as_native(&self) -> &Self::Native;
}

type BufferPtr = NonNull<metal::MTLBuffer>;
type TexturePtr = NonNull<metal::MTLTexture>;
type SamplerPtr = NonNull<metal::MTLSamplerState>;

impl AsNative for BufferPtr {
    type Native = metal::BufferRef;
    #[inline]
    fn from(native: &Self::Native) -> Self {
        unsafe { NonNull::new_unchecked(native.as_ptr()) }
    }
    #[inline]
    fn as_native(&self) -> &Self::Native {
        unsafe { Self::Native::from_ptr(self.as_ptr()) }
    }
}

impl AsNative for TexturePtr {
    type Native = metal::TextureRef;
    #[inline]
    fn from(native: &Self::Native) -> Self {
        unsafe { NonNull::new_unchecked(native.as_ptr()) }
    }
    #[inline]
    fn as_native(&self) -> &Self::Native {
        unsafe { Self::Native::from_ptr(self.as_ptr()) }
    }
}

impl AsNative for SamplerPtr {
    type Native = metal::SamplerStateRef;
    #[inline]
    fn from(native: &Self::Native) -> Self {
        unsafe { NonNull::new_unchecked(native.as_ptr()) }
    }
    #[inline]
    fn as_native(&self) -> &Self::Native {
        unsafe { Self::Native::from_ptr(self.as_ptr()) }
    }
}

#[derive(Debug)]
struct BufferResource {
    ptr: BufferPtr,
    offset: wgt::BufferAddress,
    dynamic_index: Option<u32>,

    /// The buffer's size, if it is a [`Storage`] binding. Otherwise `None`.
    ///
    /// Buffers with the [`wgt::BufferBindingType::Storage`] binding type can
    /// hold WGSL runtime-sized arrays. When one does, we must pass its size to
    /// shader entry points to implement bounds checks and WGSL's `arrayLength`
    /// function. See [`device::CompiledShader::sized_bindings`] for details.
    ///
    /// [`Storage`]: wgt::BufferBindingType::Storage
    binding_size: Option<wgt::BufferSize>,

    binding_location: u32,
}

#[derive(Debug, Default)]
pub struct BindGroup {
    counters: MultiStageResourceCounters,
    buffers: Vec<BufferResource>,
    samplers: Vec<SamplerPtr>,
    textures: Vec<TexturePtr>,
}

unsafe impl Send for BindGroup {}
unsafe impl Sync for BindGroup {}

#[derive(Debug)]
pub struct ShaderModule {
    naga: crate::NagaShader,
    runtime_checks: bool,
}

#[derive(Debug, Default)]
struct PipelineStageInfo {
    push_constants: Option<PushConstantsInfo>,

    /// The buffer argument table index at which we pass runtime-sized arrays' buffer sizes.
    ///
    /// See [`device::CompiledShader::sized_bindings`] for more details.
    sizes_slot: Option<naga::back::msl::Slot>,

    /// Bindings of all WGSL `storage` globals that contain runtime-sized arrays.
    ///
    /// See [`device::CompiledShader::sized_bindings`] for more details.
    sized_bindings: Vec<naga::ResourceBinding>,
}

impl PipelineStageInfo {
    fn clear(&mut self) {
        self.push_constants = None;
        self.sizes_slot = None;
        self.sized_bindings.clear();
    }

    fn assign_from(&mut self, other: &Self) {
        self.push_constants = other.push_constants;
        self.sizes_slot = other.sizes_slot;
        self.sized_bindings.clear();
        self.sized_bindings.extend_from_slice(&other.sized_bindings);
    }
}

#[derive(Debug)]
pub struct RenderPipeline {
    raw: metal::RenderPipelineState,
    #[allow(dead_code)]
    vs_lib: metal::Library,
    #[allow(dead_code)]
    fs_lib: Option<metal::Library>,
    vs_info: PipelineStageInfo,
    fs_info: Option<PipelineStageInfo>,
    raw_primitive_type: metal::MTLPrimitiveType,
    raw_triangle_fill_mode: metal::MTLTriangleFillMode,
    raw_front_winding: metal::MTLWinding,
    raw_cull_mode: metal::MTLCullMode,
    raw_depth_clip_mode: Option<metal::MTLDepthClipMode>,
    depth_stencil: Option<(metal::DepthStencilState, wgt::DepthBiasState)>,
}

unsafe impl Send for RenderPipeline {}
unsafe impl Sync for RenderPipeline {}

#[derive(Debug)]
pub struct ComputePipeline {
    raw: metal::ComputePipelineState,
    #[allow(dead_code)]
    cs_lib: metal::Library,
    cs_info: PipelineStageInfo,
    work_group_size: metal::MTLSize,
    work_group_memory_sizes: Vec<u32>,
}

unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

#[derive(Debug, Clone)]
pub struct QuerySet {
    raw_buffer: metal::Buffer,
    //Metal has a custom buffer for counters.
    counter_sample_buffer: Option<metal::CounterSampleBuffer>,
    /// A private 1x1 render target used only for command-boundary timestamp
    /// writes on macOS 26, where the legacy imperative sampling methods return
    /// zero even though render-pass sample attachments remain operational.
    timestamp_dummy_target: Option<metal::Texture>,
    ty: wgt::QueryType,
}

unsafe impl Send for QuerySet {}
unsafe impl Sync for QuerySet {}

const ASYNC_DEVICE_ERROR_NONE: u8 = 0;
const ASYNC_DEVICE_ERROR_OUT_OF_MEMORY: u8 = 1;
const ASYNC_DEVICE_ERROR_LOST: u8 = 2;
const ASYNC_DEVICE_ERROR_RESOURCE_CREATION: u8 = 3;

fn encode_device_error(error: &crate::DeviceError) -> u8 {
    match error {
        crate::DeviceError::OutOfMemory => ASYNC_DEVICE_ERROR_OUT_OF_MEMORY,
        crate::DeviceError::Lost => ASYNC_DEVICE_ERROR_LOST,
        crate::DeviceError::ResourceCreationFailed => ASYNC_DEVICE_ERROR_RESOURCE_CREATION,
    }
}

fn decode_device_error(error: u8) -> Option<crate::DeviceError> {
    match error {
        ASYNC_DEVICE_ERROR_NONE => None,
        ASYNC_DEVICE_ERROR_OUT_OF_MEMORY => Some(crate::DeviceError::OutOfMemory),
        ASYNC_DEVICE_ERROR_LOST => Some(crate::DeviceError::Lost),
        ASYNC_DEVICE_ERROR_RESOURCE_CREATION => Some(crate::DeviceError::ResourceCreationFailed),
        _ => Some(crate::DeviceError::Lost),
    }
}

fn record_async_device_error(slot: &atomic::AtomicU8, error: &crate::DeviceError) {
    let _ = slot.compare_exchange(
        ASYNC_DEVICE_ERROR_NONE,
        encode_device_error(error),
        atomic::Ordering::AcqRel,
        atomic::Ordering::Acquire,
    );
}

#[cfg(test)]
mod timestamp_error_tests {
    use super::*;

    #[test]
    fn asynchronous_device_error_is_sticky_and_decodable() {
        for error in [
            crate::DeviceError::OutOfMemory,
            crate::DeviceError::Lost,
            crate::DeviceError::ResourceCreationFailed,
        ] {
            let slot = atomic::AtomicU8::new(ASYNC_DEVICE_ERROR_NONE);
            record_async_device_error(&slot, &error);
            let encoded = slot.load(atomic::Ordering::Acquire);
            assert_eq!(decode_device_error(encoded), Some(error));
        }

        let slot = atomic::AtomicU8::new(ASYNC_DEVICE_ERROR_NONE);
        record_async_device_error(&slot, &crate::DeviceError::Lost);
        record_async_device_error(&slot, &crate::DeviceError::OutOfMemory);
        assert_eq!(
            decode_device_error(slot.load(atomic::Ordering::Acquire)),
            Some(crate::DeviceError::Lost)
        );
    }
}

#[cfg(test)]
mod command_buffer_count_tests {
    use super::*;

    #[test]
    fn dropping_unsubmitted_segmented_command_buffer_restores_baseline() {
        let Some(device) = metal::Device::system_default() else {
            return;
        };
        let queue = QueueShared::new(device.new_command_queue());
        let baseline = queue
            .command_buffer_created_not_submitted
            .load(atomic::Ordering::Acquire);

        let segments = {
            let raw_queue = queue.raw.lock();
            (0..3)
                .map(|_| CommandBufferSegment::Commands {
                    raw: raw_queue.new_command_buffer().to_owned(),
                    deferred: false,
                })
                .collect::<Vec<_>>()
        };
        queue
            .command_buffer_created_not_submitted
            .fetch_add(segments.len(), atomic::Ordering::AcqRel);
        {
            let _command_buffer = CommandBuffer {
                segments,
                queue_shared: Arc::clone(&queue),
                submitted: atomic::AtomicBool::new(false),
            };
            assert_eq!(
                queue
                    .command_buffer_created_not_submitted
                    .load(atomic::Ordering::Acquire),
                baseline + 3,
                "segmented command buffer must account for every native segment"
            );
        }

        assert_eq!(
            queue
                .command_buffer_created_not_submitted
                .load(atomic::Ordering::Acquire),
            baseline,
            "dropping an unsubmitted segmented command buffer must restore its baseline"
        );
    }
}

fn command_buffer_device_error(command_buffer: &metal::CommandBufferRef) -> crate::DeviceError {
    // metal 0.27 exposes the command-buffer status but not its NSError
    // accessor.  Query the documented Objective-C property directly so an
    // asynchronous error is still classified without adding a dependency or
    // changing the public HAL API.
    let error: *mut objc::runtime::Object = unsafe { objc::msg_send![command_buffer, error] };
    if error.is_null() {
        return crate::DeviceError::Lost;
    }
    let code: isize = unsafe { objc::msg_send![error, code] };
    match code as u32 {
        x if x == metal::MTLCommandBufferError::OutOfMemory as u32 => {
            crate::DeviceError::OutOfMemory
        }
        x if x == metal::MTLCommandBufferError::Timeout as u32
            || x == metal::MTLCommandBufferError::PageFault as u32
            || x == metal::MTLCommandBufferError::NotPermitted as u32
            || x == metal::MTLCommandBufferError::DeviceRemoved as u32 =>
        {
            crate::DeviceError::Lost
        }
        _ => crate::DeviceError::ResourceCreationFailed,
    }
}

fn decrement_unsubmitted(queue_shared: &QueueShared) {
    let previous = queue_shared
        .command_buffer_created_not_submitted
        .fetch_sub(1, atomic::Ordering::AcqRel);
    debug_assert!(previous > 0);
}

fn commit_deferred_timestamp_resolve(
    resolve: &DeferredTimestampResolve,
    queue_shared: Arc<QueueShared>,
    async_device_error: &atomic::AtomicU8,
) {
    objc::rc::autoreleasepool(|| {
        if resolve.predecessor.status() != metal::MTLCommandBufferStatus::Completed {
            let error = if resolve.predecessor.status() == metal::MTLCommandBufferStatus::Error {
                command_buffer_device_error(&resolve.predecessor)
            } else {
                crate::DeviceError::Lost
            };
            log::error!(
                "metal: timestamp sampling predecessor failed: {:?}",
                resolve.predecessor.status()
            );
            record_async_device_error(async_device_error, &error);
            resolve.blit.end_encoding();
            resolve.ended.store(true, atomic::Ordering::Release);
            let continuation = resolve.continuation.clone();
            let continuation_queue_shared = Arc::clone(&queue_shared);
            let block = block::ConcreteBlock::new(move |_| {
                continuation.commit();
                decrement_unsubmitted(&continuation_queue_shared);
            })
            .copy();
            resolve.raw.add_completed_handler(&block);
            resolve.raw.commit();
            decrement_unsubmitted(&queue_shared);
            return;
        }

        for group in &resolve.debug_groups {
            resolve.blit.push_debug_group(group);
        }
        resolve.blit.resolve_counters(
            &resolve.sample_buffer,
            metal::NSRange {
                location: resolve.range.start as u64,
                length: (resolve.range.end - resolve.range.start) as u64,
            },
            &resolve.destination,
            resolve.offset,
        );
        for _ in &resolve.debug_groups {
            resolve.blit.pop_debug_group();
        }
        resolve.blit.end_encoding();
        resolve.ended.store(true, atomic::Ordering::Release);
        let continuation = resolve.continuation.clone();
        let continuation_queue_shared = Arc::clone(&queue_shared);
        let block = block::ConcreteBlock::new(move |_| {
            continuation.commit();
            decrement_unsubmitted(&continuation_queue_shared);
        })
        .copy();
        resolve.raw.add_completed_handler(&block);
        resolve.raw.commit();
        decrement_unsubmitted(&queue_shared);
    });
}

#[derive(Debug)]
pub struct Fence {
    completed_value: Arc<atomic::AtomicU64>,
    async_device_error: Arc<atomic::AtomicU8>,
    /// The pending fence values have to be ascending.
    pending_command_buffers: Vec<(crate::FenceValue, metal::CommandBuffer)>,
}

unsafe impl Send for Fence {}
unsafe impl Sync for Fence {}

impl Fence {
    fn device_error(&self) -> Option<crate::DeviceError> {
        decode_device_error(self.async_device_error.load(atomic::Ordering::Acquire))
    }

    fn get_latest(&self) -> crate::FenceValue {
        self.completed_value.load(atomic::Ordering::Acquire)
    }

    fn maintain(&mut self) {
        let latest = self.get_latest();
        self.pending_command_buffers
            .retain(|&(value, _)| value > latest);
    }
}

struct IndexState {
    buffer_ptr: BufferPtr,
    offset: wgt::BufferAddress,
    stride: wgt::BufferAddress,
    raw_type: metal::MTLIndexType,
}

#[derive(Default)]
struct Temp {
    binding_sizes: Vec<u32>,
}

struct CommandState {
    blit: Option<metal::BlitCommandEncoder>,
    render: Option<metal::RenderCommandEncoder>,
    compute: Option<metal::ComputeCommandEncoder>,
    raw_primitive_type: metal::MTLPrimitiveType,
    index: Option<IndexState>,
    raw_wg_size: metal::MTLSize,
    stage_infos: MultiStageData<PipelineStageInfo>,

    /// Sizes of currently bound [`wgt::BufferBindingType::Storage`] buffers.
    ///
    /// Specifically:
    ///
    /// - The keys are ['ResourceBinding`] values (that is, the WGSL `@group`
    ///   and `@binding` attributes) for `var<storage>` global variables in the
    ///   current module that contain runtime-sized arrays.
    ///
    /// - The values are the actual sizes of the buffers currently bound to
    ///   provide those globals' contents, which are needed to implement bounds
    ///   checks and the WGSL `arrayLength` function.
    ///
    /// For each stage `S` in `stage_infos`, we consult this to find the sizes
    /// of the buffers listed in [`stage_infos.S.sized_bindings`], which we must
    /// pass to the entry point.
    ///
    /// See [`device::CompiledShader::sized_bindings`] for more details.
    ///
    /// [`ResourceBinding`]: naga::ResourceBinding
    storage_buffer_length_map: rustc_hash::FxHashMap<naga::ResourceBinding, wgt::BufferSize>,

    work_group_memory_sizes: Vec<u32>,
    push_constants: Vec<u32>,

    /// Timer query that should be executed when the next pass starts.
    pending_timer_queries: Vec<(QuerySet, u32)>,
}

pub struct CommandEncoder {
    shared: Arc<AdapterShared>,
    queue_shared: Arc<QueueShared>,
    raw_cmd_buf: Option<metal::CommandBuffer>,
    raw_cmd_buf_deferred: bool,
    completed_segments: Vec<CommandBufferSegment>,
    recording_error: Option<crate::DeviceError>,
    debug_marker_stack: Vec<String>,
    state: CommandState,
    temp: Temp,
}

impl fmt::Debug for CommandEncoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommandEncoder")
            .field("queue_shared", &self.queue_shared)
            .field("raw_cmd_buf", &self.raw_cmd_buf)
            .finish()
    }
}

unsafe impl Send for CommandEncoder {}
unsafe impl Sync for CommandEncoder {}

#[derive(Debug)]
pub struct CommandBuffer {
    segments: Vec<CommandBufferSegment>,
    queue_shared: Arc<QueueShared>,
    submitted: atomic::AtomicBool,
}

unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}

#[derive(Debug)]
enum CommandBufferSegment {
    Commands {
        raw: metal::CommandBuffer,
        /// A continuation is committed by the resolve completion handler so
        /// commands recorded after a deferred resolve cannot overtake it.
        deferred: bool,
    },
    DeferredTimestampResolve(Arc<DeferredTimestampResolve>),
}

#[derive(Debug)]
struct DeferredTimestampResolve {
    predecessor: metal::CommandBuffer,
    raw: metal::CommandBuffer,
    blit: metal::BlitCommandEncoder,
    continuation: metal::CommandBuffer,
    sample_buffer: metal::CounterSampleBuffer,
    range: ops::Range<u32>,
    destination: metal::Buffer,
    offset: wgt::BufferAddress,
    debug_groups: Vec<String>,
    ended: atomic::AtomicBool,
}

impl Drop for DeferredTimestampResolve {
    fn drop(&mut self) {
        if !self.ended.load(atomic::Ordering::Acquire) {
            self.blit.end_encoding();
        }
    }
}

impl CommandBufferSegment {
    fn raw(&self) -> &metal::CommandBuffer {
        match self {
            Self::Commands { raw, .. } => raw,
            Self::DeferredTimestampResolve(resolve) => &resolve.raw,
        }
    }
}

impl CommandBuffer {
    fn final_raw(&self) -> &metal::CommandBuffer {
        match self.segments.last() {
            Some(CommandBufferSegment::Commands { raw, .. }) => raw,
            _ => unreachable!("a Metal command buffer must end in a commands segment"),
        }
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        if !self.submitted.load(atomic::Ordering::Acquire) {
            for _ in &self.segments {
                let previous = self
                    .queue_shared
                    .command_buffer_created_not_submitted
                    .fetch_sub(1, atomic::Ordering::AcqRel);
                debug_assert!(previous > 0);
            }
        }
    }
}

#[derive(Debug)]
pub struct AccelerationStructure;
