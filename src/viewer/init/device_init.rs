// src/viewer/init/device_init.rs
// Device and surface initialization for the Viewer

use std::sync::Arc;
use wgpu::{Adapter, Device, Instance, Queue, Surface, SurfaceConfiguration};
use winit::window::Window;

/// Resources created during device initialization
pub struct DeviceResources {
    pub surface: Surface<'static>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub adapter: Arc<Adapter>,
    pub config: SurfaceConfiguration,
    pub adapter_info: wgpu::AdapterInfo,
}

fn validate_m06_adapter(info: &wgpu::AdapterInfo) -> Result<(), String> {
    let is_nvidia = info.vendor == 0x10de || info.name.to_ascii_lowercase().contains("nvidia");
    if info.backend == wgpu::Backend::Vulkan
        && is_nvidia
        && info.device_type == wgpu::DeviceType::DiscreteGpu
    {
        return Ok(());
    }
    Err(format!(
        "M-06 requires a physical NVIDIA Vulkan viewer adapter; got name={:?} vendor={:#06x} device={:#06x} backend={:?} type={:?} driver={:?} driver_info={:?}",
        info.name,
        info.vendor,
        info.device,
        info.backend,
        info.device_type,
        info.driver,
        info.driver_info,
    ))
}

/// Create wgpu device, queue, adapter, and surface
pub async fn create_device_and_surface(
    window: Arc<Window>,
    vsync: bool,
) -> Result<DeviceResources, Box<dyn std::error::Error>> {
    let size = window.inner_size();

    // Create wgpu instance
    let requested_backend = crate::core::gpu::requested_backend_from_env()?;
    let backend_mask = requested_backend
        .as_ref()
        .map_or(wgpu::Backends::all(), |(_, mask, _)| *mask);
    let instance = Instance::new(wgpu::InstanceDescriptor {
        backends: backend_mask,
        flags: crate::core::gpu::instance_flags(),
        ..Default::default()
    });

    // Create surface
    let surface = instance.create_surface(Arc::clone(&window))?;

    // Request adapter
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .ok_or("Failed to find suitable adapter")?;

    let adapter = Arc::new(adapter);
    let adapter_info = adapter.get_info();
    if let Some((raw, _, expected)) = requested_backend.as_ref() {
        if adapter_info.backend != *expected {
            return Err(format!(
                "viewer adapter backend {:?} does not match requested backend '{raw}'",
                adapter_info.backend
            )
            .into());
        }
    }
    if std::env::var("RUN_M06_VIEWER_CI").as_deref() == Ok("1") {
        validate_m06_adapter(&adapter_info)?;
    }
    println!(
        "FORGE3D_VIEWER_ADAPTER name={:?} vendor={:#06x} device={:#06x} backend={:?} device_type={:?} driver={:?} driver_info={:?}",
        adapter_info.name,
        adapter_info.vendor,
        adapter_info.device,
        adapter_info.backend,
        adapter_info.device_type,
        adapter_info.driver,
        adapter_info.driver_info,
    );

    // Request every optional capability the adapter advertises. A driver may
    // still reject that set, so retry once without optional features and
    // record the downgrade instead of failing the viewer.
    let mut capabilities = crate::core::capabilities::CapabilitySet::negotiate(adapter.features());
    let requested_limits = adapter.limits();
    let (device, queue) = match adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Viewer Device"),
                required_features: capabilities.granted,
                required_limits: requested_limits.clone(),
            },
            None,
        )
        .await
    {
        Ok(pair) => pair,
        Err(error) => {
            capabilities.downgrade_after_request_failure(&error.to_string());
            adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("Viewer Device"),
                        required_features: capabilities.granted,
                        required_limits: requested_limits,
                    },
                    None,
                )
                .await?
        }
    };

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    // Configure surface
    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(surface_caps.formats[0]);

    let config = SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        format: surface_format,
        width: size.width,
        height: size.height,
        present_mode: if vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        },
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };

    surface.configure(&device, &config);

    Ok(DeviceResources {
        surface,
        device,
        queue,
        adapter,
        config,
        adapter_info,
    })
}

#[cfg(test)]
mod tests {
    use super::validate_m06_adapter;

    fn adapter(
        vendor: u32,
        backend: wgpu::Backend,
        device_type: wgpu::DeviceType,
    ) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: if vendor == 0x10de {
                "NVIDIA RTX"
            } else {
                "AMD Radeon"
            }
            .to_string(),
            vendor,
            device: 1,
            device_type,
            driver: "test".to_string(),
            driver_info: "test".to_string(),
            backend,
        }
    }

    #[test]
    fn m06_adapter_gate_rejects_wrong_vendor_backend_and_software() {
        assert!(validate_m06_adapter(&adapter(
            0x10de,
            wgpu::Backend::Vulkan,
            wgpu::DeviceType::DiscreteGpu
        ))
        .is_ok());
        assert!(validate_m06_adapter(&adapter(
            0x1002,
            wgpu::Backend::Vulkan,
            wgpu::DeviceType::DiscreteGpu
        ))
        .is_err());
        assert!(validate_m06_adapter(&adapter(
            0x10de,
            wgpu::Backend::Dx12,
            wgpu::DeviceType::DiscreteGpu
        ))
        .is_err());
        assert!(validate_m06_adapter(&adapter(
            0x10de,
            wgpu::Backend::Vulkan,
            wgpu::DeviceType::Cpu
        ))
        .is_err());
        assert!(validate_m06_adapter(&adapter(
            0x10de,
            wgpu::Backend::Vulkan,
            wgpu::DeviceType::VirtualGpu
        ))
        .is_err());
    }
}
