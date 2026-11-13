//! Hand the pixel ring's mapping to the GPU, so a frame from Blender goes
//! from the shared pages into the node's layer as one GPU blit and no CPU
//! copy at all.
//!
//! macOS / Apple silicon: memory is unified, so a `MTLBuffer` created
//! with `newBufferWithBytesNoCopy` over the page-aligned mapping IS the
//! ring — Blender's writes land in it, and a `copy_buffer_to_texture`
//! from it is the GPU reading the file's pages. wgpu takes the raw Metal
//! buffer through its hal (`create_buffer_from_hal`). Elsewhere this
//! returns an error and the session keeps the CPU path (one memcpy on the
//! socket thread, then `write_texture`).
//!
//! The mapping is opened read-write because Metal may refuse a read-only
//! range; nothing here ever writes to it. The ring's slots start on page
//! boundaries and its rows are 256-byte aligned (`shm`), which is exactly
//! what a buffer-to-texture copy needs.

#[cfg(target_os = "macos")]
use anyhow::Context;
use anyhow::Result;

/// A ring the GPU can read.
pub struct SharedRing {
    pub buffer: wgpu::Buffer,
    pub len: u64,
    // Keeps the pages mapped for as long as the buffer exists. Dropped
    // after `buffer` (declaration order), which is the order that matters.
    _map: memmap2::MmapMut,
}

#[cfg(target_os = "macos")]
pub fn import(device: &wgpu::Device, src: &super::shm::Source) -> Result<SharedRing> {
    use objc2_metal::{MTLDevice, MTLResourceOptions};
    use wgpu::hal;

    // Blender's ring, mapped again for the GPU; only ever read here, and
    // the mapping outlives the Metal buffer wrapping it (field order).
    let map = src.map(None)?;
    let len = map.len() as u64;
    if len == 0 || len % super::shm::PAGE as u64 != 0 {
        anyhow::bail!("pixel ring of {len} bytes is not whole pages");
    }
    let ptr = std::ptr::NonNull::new(map.as_ptr() as *mut std::ffi::c_void)
        .context("null mapping")?;
    let hal_device = unsafe { device.as_hal::<hal::api::Metal>() }
        .context("the GPU is not Metal; the ring stays on the CPU path")?;
    let raw = unsafe {
        hal_device.raw_device().newBufferWithBytesNoCopy_length_options_deallocator(
            ptr,
            len as usize,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }
    .context("Metal refused the mapping (not page-aligned?)")?;
    drop(hal_device);
    let hal_buffer = unsafe { hal::metal::Device::buffer_from_raw(raw, len) };
    let buffer = unsafe {
        device.create_buffer_from_hal::<hal::api::Metal>(
            hal_buffer,
            &wgpu::BufferDescriptor {
                label: Some("blender-ring"),
                size: len,
                usage: wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )
    };
    Ok(SharedRing { buffer, len, _map: map })
}

#[cfg(not(target_os = "macos"))]
pub fn import(_device: &wgpu::Device, _src: &super::shm::Source) -> Result<SharedRing> {
    anyhow::bail!("GPU import of the pixel ring is macOS-only for now; the CPU path serves")
}

/// One of Blender's frames as a texture of ours: an IOSurface the native
/// module blitted Blender's picture into, looked up by id and wrapped for
/// wgpu. The pixels never left the GPU. `read_rows` maps the surface for
/// the CPU cache (a settled frame kept for a scrub back).
pub struct SurfaceTexture {
    pub texture: wgpu::Texture,
    pub w: u32,
    pub h: u32,
    #[cfg(target_os = "macos")]
    surface: objc2_core_foundation::CFRetained<objc2_io_surface::IOSurfaceRef>,
}

// SAFETY: an IOSurface is a CoreFoundation object (retain/release and
// lock/unlock are thread-safe by contract), and the wgpu texture is Send +
// Sync on its own. The cache copy maps it from a helper thread.
unsafe impl Send for SurfaceTexture {}
unsafe impl Sync for SurfaceTexture {}

#[cfg(target_os = "macos")]
pub fn import_surfaces(device: &wgpu::Device, ids: &[u32], w: u32, h: u32) -> Result<Vec<SurfaceTexture>> {
    use objc2_metal::{MTLDevice, MTLPixelFormat, MTLStorageMode, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage};
    use wgpu::hal;

    let hal_device = unsafe { device.as_hal::<hal::api::Metal>() }
        .context("the GPU is not Metal; Blender's surfaces cannot be shared")?;
    let mut out = Vec::with_capacity(ids.len());
    for id in ids {
        let surface = objc2_io_surface::IOSurfaceRef::lookup(*id)
            .with_context(|| format!("IOSurface {id} is not visible from here"))?;
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::RGBA32Float,
                w as usize,
                h as usize,
                false,
            )
        };
        desc.setStorageMode(MTLStorageMode::Shared);
        desc.setUsage(MTLTextureUsage::ShaderRead);
        let raw = hal_device
            .raw_device()
            .newTextureWithDescriptor_iosurface_plane(&desc, &surface, 0)
            .context("Metal refused Blender's IOSurface")?;
        let hal_tex = unsafe {
            hal::metal::Device::texture_from_raw(
                raw,
                wgpu::TextureFormat::Rgba32Float,
                MTLTextureType::Type2D,
                1,
                1,
                hal::CopyExtent { width: w, height: h, depth: 1 },
                None,
            )
        };
        let texture = unsafe {
            device.create_texture_from_hal::<hal::api::Metal>(
                hal_tex,
                &wgpu::TextureDescriptor {
                    label: Some("blender-surface"),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba32Float,
                    usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::TextureUses::RESOURCE,
            )
        };
        out.push(SurfaceTexture { texture, w, h, surface });
    }
    drop(hal_device);
    Ok(out)
}

#[cfg(not(target_os = "macos"))]
pub fn import_surfaces(_device: &wgpu::Device, _ids: &[u32], _w: u32, _h: u32) -> Result<Vec<SurfaceTexture>> {
    anyhow::bail!("shared GPU surfaces are macOS-only for now")
}

impl SurfaceTexture {
    /// The top-left `w` x `h` of the surface as rgba32f bytes, rows
    /// un-padded, as they sit (bottom-up). A CPU map of the surface: for
    /// the cache copy, never the display path.
    #[cfg(target_os = "macos")]
    pub fn read_rows(&self, w: u32, h: u32) -> Option<Vec<u8>> {
        use objc2_io_surface::IOSurfaceLockOptions;
        if w > self.w || h > self.h || w == 0 || h == 0 {
            return None;
        }
        let s = &self.surface;
        if unsafe { s.lock(IOSurfaceLockOptions::ReadOnly, std::ptr::null_mut()) } != 0 {
            return None;
        }
        let stride = s.bytes_per_row();
        let base = s.base_address().as_ptr() as *const u8;
        let row = w as usize * 16;
        let mut out = Vec::with_capacity(row * h as usize);
        for y in 0..h as usize {
            // SAFETY: locked, inside the surface's own rows.
            let src = unsafe { std::slice::from_raw_parts(base.add(y * stride), row) };
            out.extend_from_slice(src);
        }
        unsafe { s.unlock(IOSurfaceLockOptions::ReadOnly, std::ptr::null_mut()) };
        Some(out)
    }

    #[cfg(not(target_os = "macos"))]
    pub fn read_rows(&self, _w: u32, _h: u32) -> Option<Vec<u8>> {
        None
    }
}
