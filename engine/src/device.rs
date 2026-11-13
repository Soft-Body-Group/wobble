//! wgpu device bootstrap. Native limits (8K textures need > the 8192 default),
//! timestamp features requested when available, gracefully absent when not.

use std::sync::Arc;

pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    /// TIMESTAMP_QUERY granted.
    pub timestamps: bool,
    /// TIMESTAMP_QUERY_INSIDE_ENCODERS granted (precise copy timing).
    pub timestamps_inside_encoders: bool,
    pub backend: wgpu::Backend,
}

impl GpuContext {
    pub fn new() -> anyhow::Result<Self> {
        pollster::block_on(Self::new_async())
    }

    /// The same bootstrap as a future. A browser cannot block on the adapter
    /// and device requests (they resolve on the JS event loop), so the web
    /// build awaits this; native blocks on it through `new`.
    pub async fn new_async() -> anyhow::Result<Self> {
        // `WGPU_BACKEND=dx12` / `vulkan` / `metal` picks the backend (and
        // the other `WGPU_*` switches apply). `Instance::default()` reads no
        // environment, so on Windows there was no way to A/B DX12 against
        // the Vulkan adapter wgpu prefers — and a crash that moves with the
        // backend is a finding.
        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
        );
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await?;

        let info = adapter.get_info();
        log::info!("adapter: {} ({:?}, {:?})", info.name, info.device_type, info.backend);

        let adapter_features = adapter.features();
        let mut features = wgpu::Features::empty();
        let timestamps = adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY);
        if timestamps {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        let timestamps_inside_encoders =
            adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        if timestamps_inside_encoders {
            features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        }
        // A Blender node's frames arrive as float32 (straight off Blender's
        // readback, no conversion); sampling them in the fused pass needs
        // the float32 textures to be filterable. Taken where the adapter
        // offers it (every Apple GPU, every desktop GPU); without it the
        // node converts to half on the way in instead.
        if adapter_features.contains(wgpu::Features::FLOAT32_FILTERABLE) {
            features |= wgpu::Features::FLOAT32_FILTERABLE;
        }
        // f16 arithmetic in shaders: the neural nets (`nn`) store their
        // feature maps as f16 where it is offered — half the memory and
        // half the bandwidth — and fall back to f32 where it is not.
        if adapter_features.contains(wgpu::Features::SHADER_F16) {
            features |= wgpu::Features::SHADER_F16;
        }
        // Single-channel half textures written from compute: a matte's
        // filter chain runs on one channel instead of four
        // (`nodes::stage::one_channel`). Native adapters offer it through
        // their own format table; where they do not (a browser), mattes take
        // the four-channel chain.
        let specific = adapter_features.contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES);
        if specific {
            features |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        }
        let r16 = adapter.get_texture_format_features(wgpu::TextureFormat::R16Float);
        crate::nodes::stage::set_one_channel(
            specific
                && r16.allowed_usages.contains(wgpu::TextureUsages::STORAGE_BINDING)
                && r16.flags.contains(wgpu::TextureFormatFeatureFlags::STORAGE_WRITE_ONLY),
        );

        // The brief's rationale: default limits cap texture dims at 8192 (fails
        // 8K DCI width) and storage buffers far too small. Take what the
        // hardware actually has.
        let limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("compositor-device"),
                required_features: features,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: Default::default(),
            })
            .await?;

        Ok(Self {
            instance,
            backend: info.backend,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            timestamps,
            timestamps_inside_encoders,
        })
    }
}
