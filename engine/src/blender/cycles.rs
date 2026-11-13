//! Live Cycles straight into a Blender layer: no readback, no ring.
//!
//! A Blender built from the wobble fork (`_cycles.wobble_target`, see the
//! fork's `intern/cycles/blender/wobble.h`) swaps Cycles' display driver for
//! one that writes every display update into a buffer WOBBLE owns. Cycles
//! already imports Vulkan buffers into CUDA for its own viewport (graphics
//! interop); the fork only points that import at this buffer.
//!
//! So here: an exportable `VkBuffer` of half4 pixels on wgpu's own Vulkan
//! device, its Win32 handle duplicated into Blender's process, the GPU's
//! UUID (Cycles refuses interop with another GPU) and a small named stamp
//! block Cycles bumps after each update. A landed update is copied row by
//! row into a 256-aligned staging buffer (Cycles packs rows at `w * 8`),
//! which then takes the ring's GPU path into the layer.
//!
//! macOS: nothing to import. Cycles' Metal interop takes a plain MTLBuffer
//! on unified memory, so the target is ONE POSIX shared memory object: the
//! stamp in its first page (`shm::HEADER_BYTES`), the pixels after it. Both
//! processes wrap the pixel pages in an MTLBuffer (`newBufferWithBytesNoCopy`,
//! the ring's `gpuimport` precedent); the fork pads rows to 256 bytes (stamp
//! version 2), so an update is ONE buffer-to-texture copy and no staging.
//! Elsewhere `CyclesTarget::new` says why not.

use anyhow::Result;

/// Bytes per pixel Cycles writes (half4).
pub const BPP: u64 = 8;

/// The stamp block the fork writes (`WobbleStamp` in the fork's wobble.h).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Stamp {
    pub magic: u32,
    pub version: u32,
    /// Odd while Cycles is writing an update, even once it has landed.
    pub seq: u64,
    pub width: u32,
    pub height: u32,
    pub full_width: u32,
    pub full_height: u32,
    pub updates: u64,
    pub buffer_id: u64,
}

pub const STAMP_MAGIC: u32 = 0x4359_4357; // 'WCYC'
/// Stamp version 1: rows packed at `w * 8` (Windows). 2: rows padded to
/// `padded_stride` (macOS), straight into a texture.
pub const STAMP_PADDED: u32 = 2;

/// A landed update: `w` x `h` half4 pixels, rows bottom-up at `w * 8`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Update {
    pub seq: u64,
    pub w: u32,
    pub h: u32,
    pub updates: u64,
}

/// Row stride of the staging copy: a buffer-to-texture copy wants 256.
pub fn padded_stride(w: u32) -> u32 {
    (w * BPP as u32).div_ceil(256) * 256
}

#[cfg(windows)]
pub use win::CyclesTarget;
#[cfg(target_os = "macos")]
pub use mac::CyclesTarget;

/// Elsewhere there is no target: `new` always fails, so no value exists,
/// but the API matches the Windows one field for field, method for method,
/// so the app compiles the same on every OS (the Mac build broke on the
/// unit-struct stub this replaces).
#[cfg(not(any(windows, target_os = "macos")))]
pub struct CyclesTarget {
    pub shared: wgpu::Buffer,
    pub staging: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
    pub size: u64,
    pub uuid: [u8; 16],
    pub stamp_name: String,
    never: std::convert::Infallible,
}

#[cfg(not(any(windows, target_os = "macos")))]
impl CyclesTarget {
    pub fn new(_device: &wgpu::Device, _w: u32, _h: u32, _tag: &str) -> Result<Self> {
        anyhow::bail!("live Cycles needs Windows + Vulkan or macOS + Metal (a wobble-fork Blender)")
    }
    pub fn handle_for(&self, _pid: u32) -> Result<u64> {
        match self.never {}
    }
    pub fn uuid_hex(&self) -> String {
        match self.never {}
    }
    pub fn stamp(&self) -> Stamp {
        match self.never {}
    }
    pub fn poll(&mut self) -> Option<Update> {
        match self.never {}
    }
    pub fn encode_rows(&self, _enc: &mut wgpu::CommandEncoder, _u: &Update) {
        match self.never {}
    }
}

#[cfg(windows)]
mod win {
    use super::*;
    use anyhow::Context;
    use ash::vk;
    use wgpu::hal;
    use windows_sys::Win32::Foundation::{CloseHandle, DuplicateHandle, DUPLICATE_SAME_ACCESS, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, MapViewOfFile, UnmapViewOfFile, FILE_MAP_ALL_ACCESS, MEMORY_MAPPED_VIEW_ADDRESS, PAGE_READWRITE,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcess, PROCESS_DUP_HANDLE};

    pub struct CyclesTarget {
        /// The buffer Cycles writes, as wgpu sees it. Declared first so it
        /// drops before the Vulkan objects under it.
        pub shared: wgpu::Buffer,
        /// 256-aligned rows for the copy into the layer.
        pub staging: wgpu::Buffer,
        pub width: u32,
        pub height: u32,
        pub size: u64,
        pub uuid: [u8; 16],
        pub stamp_name: String,
        handle: HANDLE,
        mapping: HANDLE,
        stamp: *mut Stamp,
        last_seq: u64,
    }

    // The stamp pointer is a view of a mapping this struct owns.
    unsafe impl Send for CyclesTarget {}
    unsafe impl Sync for CyclesTarget {}

    impl CyclesTarget {
        /// A target for renders up to `w` x `h`. `tag` makes the stamp
        /// block's name unique per session.
        pub fn new(device: &wgpu::Device, w: u32, h: u32, tag: &str) -> Result<Self> {
            anyhow::ensure!(w > 0 && h > 0, "empty Cycles target");
            let size = w as u64 * h as u64 * BPP;
            let hal_device = unsafe { device.as_hal::<hal::api::Vulkan>() }
                .context("wobble is not on Vulkan (set WGPU_BACKEND=vulkan); Cycles can only share Vulkan memory")?;
            let raw = hal_device.raw_device().clone();
            let phys = hal_device.raw_physical_device();
            let instance = hal_device.shared_instance().raw_instance().clone();

            // The GPU's UUID: Cycles matches it against its CUDA device.
            let mut id = vk::PhysicalDeviceIDProperties::default();
            let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
            unsafe { instance.get_physical_device_properties2(phys, &mut props) };
            let uuid = id.device_uuid;

            let mut ext_buf = vk::ExternalMemoryBufferCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);
            let info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(
                    vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::STORAGE_BUFFER,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .push_next(&mut ext_buf);
            let buffer = unsafe { raw.create_buffer(&info, None) }.context("creating the shared buffer")?;
            let req = unsafe { raw.get_buffer_memory_requirements(buffer) };
            let mem_props = unsafe { instance.get_physical_device_memory_properties(phys) };
            let Some(type_index) = (0..mem_props.memory_type_count).find(|&i| {
                req.memory_type_bits & (1 << i) != 0
                    && mem_props.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            }) else {
                unsafe { raw.destroy_buffer(buffer, None) };
                anyhow::bail!("no device-local memory for the shared buffer");
            };
            let mut export = vk::ExportMemoryAllocateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(type_index)
                .push_next(&mut export)
                .push_next(&mut dedicated);
            let memory = match unsafe { raw.allocate_memory(&alloc, None) } {
                Ok(m) => m,
                Err(e) => {
                    unsafe { raw.destroy_buffer(buffer, None) };
                    anyhow::bail!("allocating exportable memory: {e}");
                }
            };
            let cleanup = |raw: &ash::Device| unsafe {
                raw.destroy_buffer(buffer, None);
                raw.free_memory(memory, None);
            };
            if let Err(e) = unsafe { raw.bind_buffer_memory(buffer, memory, 0) } {
                cleanup(&raw);
                anyhow::bail!("binding the shared buffer: {e}");
            }
            let win32 = ash::khr::external_memory_win32::Device::new(&instance, &raw);
            let get = vk::MemoryGetWin32HandleInfoKHR::default()
                .memory(memory)
                .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);
            let handle = match unsafe { win32.get_memory_win32_handle(&get) } {
                Ok(h) => h as HANDLE,
                Err(e) => {
                    cleanup(&raw);
                    anyhow::bail!("exporting the shared buffer: {e}");
                }
            };
            drop(hal_device);

            // wgpu frees nothing of ours: the callback runs when it drops
            // its buffer (after the GPU is done with it), and only then do
            // the Vulkan buffer and its memory go.
            let owner = raw.clone();
            let hal_buffer = unsafe {
                hal::vulkan::Buffer::from_raw_externally_owned(
                    buffer,
                    Box::new(move || {
                        owner.destroy_buffer(buffer, None);
                        owner.free_memory(memory, None);
                    }),
                )
            };
            let shared = unsafe {
                device.create_buffer_from_hal::<hal::api::Vulkan>(
                    hal_buffer,
                    &wgpu::BufferDescriptor {
                        label: Some("cycles-shared"),
                        size,
                        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    },
                )
            };
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cycles-staging"),
                size: padded_stride(w) as u64 * h as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // The stamp block, by name: Blender opens it from the name alone.
            let stamp_name = format!("Local\\wobble_cycles_{}_{tag}", std::process::id());
            let wide: Vec<u16> = stamp_name.encode_utf16().chain(Some(0)).collect();
            let bytes = std::mem::size_of::<Stamp>() as u32;
            let mapping = unsafe {
                CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, 0, bytes, wide.as_ptr())
            };
            anyhow::ensure!(!mapping.is_null(), "CreateFileMapping: {}", std::io::Error::last_os_error());
            let view = unsafe { MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, bytes as usize) };
            if view.Value.is_null() {
                unsafe { CloseHandle(mapping) };
                anyhow::bail!("MapViewOfFile: {}", std::io::Error::last_os_error());
            }
            let stamp = view.Value as *mut Stamp;
            unsafe { stamp.write_volatile(Stamp::default()) };

            Ok(Self {
                shared,
                staging,
                width: w,
                height: h,
                size,
                uuid,
                stamp_name,
                handle,
                mapping,
                stamp,
                last_seq: 0,
            })
        }

        /// The buffer's handle as Blender (process `pid`) will see it.
        pub fn handle_for(&self, pid: u32) -> Result<u64> {
            unsafe {
                let target = OpenProcess(PROCESS_DUP_HANDLE, 0, pid);
                anyhow::ensure!(!target.is_null(), "OpenProcess({pid}): {}", std::io::Error::last_os_error());
                let mut out: HANDLE = std::ptr::null_mut();
                let ok = DuplicateHandle(GetCurrentProcess(), self.handle, target, &mut out, 0, 0, DUPLICATE_SAME_ACCESS);
                CloseHandle(target);
                anyhow::ensure!(ok != 0, "DuplicateHandle: {}", std::io::Error::last_os_error());
                Ok(out as u64)
            }
        }

        pub fn uuid_hex(&self) -> String {
            self.uuid.iter().map(|b| format!("{b:02x}")).collect()
        }

        /// What the stamp says right now.
        pub fn stamp(&self) -> Stamp {
            unsafe { self.stamp.read_volatile() }
        }

        /// A new, complete update since the last call, or `None`.
        pub fn poll(&mut self) -> Option<Update> {
            let s = self.stamp();
            if s.magic != STAMP_MAGIC || s.seq & 1 == 1 || s.seq == self.last_seq {
                return None;
            }
            if s.width == 0 || s.height == 0 || s.width as u64 * s.height as u64 * BPP > self.size {
                return None;
            }
            self.last_seq = s.seq;
            Some(Update { seq: s.seq, w: s.width, h: s.height, updates: s.updates })
        }

        /// Record the row copies that put update `u` into `staging` at
        /// the padded stride.
        pub fn encode_rows(&self, enc: &mut wgpu::CommandEncoder, u: &Update) {
            let row = u.w as u64 * BPP;
            let stride = padded_stride(u.w) as u64;
            for y in 0..u.h as u64 {
                enc.copy_buffer_to_buffer(&self.shared, y * row, &self.staging, y * stride, row);
            }
        }
    }

    impl Drop for CyclesTarget {
        fn drop(&mut self) {
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: self.stamp as *mut _ });
                CloseHandle(self.mapping);
                CloseHandle(self.handle);
            }
        }
    }
}

#[cfg(target_os = "macos")]
mod mac {
    use super::*;
    use crate::blender::shm;
    use anyhow::Context;
    use wgpu::hal;

    pub struct CyclesTarget {
        /// The pixel pages as wgpu sees them. Cycles writes them padded, so
        /// this IS the staging buffer too (`staging` is the same buffer).
        /// Declared before the mapping so it drops first.
        pub shared: wgpu::Buffer,
        pub staging: wgpu::Buffer,
        pub width: u32,
        pub height: u32,
        /// Bytes the pixels can hold.
        pub size: u64,
        /// One GPU on Apple silicon: nothing to match.
        pub uuid: [u8; 16],
        /// The shared memory object's name, as `shm_open` takes it (the
        /// fork opens it with this exact string).
        pub stamp_name: String,
        source: shm::Source,
        map: memmap2::MmapMut,
        last_seq: u64,
    }

    // The stamp is read through the mapping this struct owns.
    unsafe impl Send for CyclesTarget {}
    unsafe impl Sync for CyclesTarget {}

    impl CyclesTarget {
        /// A target for renders up to `w` x `h`. `tag` makes the name
        /// unique per session; macOS caps it at 31 characters.
        pub fn new(device: &wgpu::Device, w: u32, h: u32, tag: &str) -> Result<Self> {
            use objc2_metal::{MTLDevice, MTLResourceOptions};
            anyhow::ensure!(w > 0 && h > 0, "empty Cycles target");
            let size = (padded_stride(w) as u64 * h as u64).div_ceil(shm::PAGE as u64) * shm::PAGE as u64;
            let name = format!("wcy_{tag}");
            anyhow::ensure!(name.len() < 31, "shared memory name {name} is too long for macOS");
            let bytes = shm::HEADER_BYTES + size as usize;
            let source = shm::Source::Posix { name: name.clone(), bytes };
            let mut map = source.map(Some(bytes))?;
            map[..shm::HEADER_BYTES].fill(0);

            let ptr = std::ptr::NonNull::new(unsafe { map.as_mut_ptr().add(shm::HEADER_BYTES) } as *mut std::ffi::c_void)
                .context("null mapping")?;
            let hal_device = unsafe { device.as_hal::<hal::api::Metal>() }
                .context("wobble is not on Metal; Cycles can only share Metal memory here")?;
            let raw = unsafe {
                hal_device.raw_device().newBufferWithBytesNoCopy_length_options_deallocator(
                    ptr,
                    size as usize,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
            }
            .context("Metal refused the shared pages")?;
            drop(hal_device);
            let hal_buffer = unsafe { hal::metal::Device::buffer_from_raw(raw, size) };
            let shared = unsafe {
                device.create_buffer_from_hal::<hal::api::Metal>(
                    hal_buffer,
                    &wgpu::BufferDescriptor {
                        label: Some("cycles-shared"),
                        size,
                        usage: wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    },
                )
            };
            Ok(Self {
                staging: shared.clone(),
                shared,
                width: w,
                height: h,
                size,
                uuid: [0; 16],
                stamp_name: format!("/{name}"),
                source,
                map,
                last_seq: 0,
            })
        }

        /// No handle to pass: the name is the whole introduction. Non-zero,
        /// because zero is what detaches.
        pub fn handle_for(&self, _pid: u32) -> Result<u64> {
            Ok(1)
        }

        pub fn uuid_hex(&self) -> String {
            String::new()
        }

        /// What the stamp says right now.
        pub fn stamp(&self) -> Stamp {
            let s = unsafe { (self.map.as_ptr() as *const Stamp).read_volatile() };
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            s
        }

        /// A new, complete update since the last call, or `None`.
        pub fn poll(&mut self) -> Option<Update> {
            let s = self.stamp();
            if s.magic != STAMP_MAGIC || s.version != STAMP_PADDED || s.seq & 1 == 1 || s.seq == self.last_seq {
                return None;
            }
            if s.width == 0 || s.height == 0 || padded_stride(s.width) as u64 * s.height as u64 > self.size {
                return None;
            }
            self.last_seq = s.seq;
            Some(Update { seq: s.seq, w: s.width, h: s.height, updates: s.updates })
        }

        /// Nothing to record: Cycles already wrote the rows padded.
        pub fn encode_rows(&self, _enc: &mut wgpu::CommandEncoder, _u: &Update) {}
    }

    impl Drop for CyclesTarget {
        fn drop(&mut self) {
            // The fork keeps its own mapping; the name just stops resolving.
            self.source.unlink();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_padded_to_256() {
        assert_eq!(padded_stride(32), 256);
        assert_eq!(padded_stride(33), 512);
        assert_eq!(padded_stride(1920), 1920 * 8);
        assert_eq!(std::mem::size_of::<Stamp>(), 48, "mirrors WobbleStamp in the fork");
    }

    /// The real target on the real GPU: the stamp sits in the first page,
    /// a padded update written into the pixel pages polls once, and an
    /// odd (in-progress) seq never does.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_metal_target_polls_padded_updates() {
        let ctx = crate::GpuContext::new().unwrap();
        let mut t = CyclesTarget::new(&ctx.device, 64, 32, "test_a").unwrap();
        assert_eq!(t.size % crate::blender::shm::PAGE as u64, 0);
        let name = t.stamp_name.clone();
        // What the fork does, from a second mapping of the same object.
        let src = crate::blender::shm::Source::Posix {
            name: name.trim_start_matches('/').to_string(),
            bytes: crate::blender::shm::HEADER_BYTES + t.size as usize,
        };
        let mut other = src.map(None).unwrap();
        let stamp = other.as_mut_ptr() as *mut Stamp;
        let write = |seq: u64| unsafe {
            stamp.write_volatile(Stamp {
                magic: STAMP_MAGIC,
                version: STAMP_PADDED,
                seq,
                width: 40,
                height: 20,
                full_width: 40,
                full_height: 20,
                updates: seq / 2,
                buffer_id: 1,
            })
        };
        write(1);
        assert_eq!(t.poll(), None, "an update still being written is not taken");
        write(2);
        assert_eq!(t.poll(), Some(Update { seq: 2, w: 40, h: 20, updates: 1 }));
        assert_eq!(t.poll(), None, "taken once");
    }

    /// The real export, on a real Vulkan device (Windows + Vulkan only).
    #[cfg(windows)]
    #[test]
    #[ignore]
    fn a_shared_buffer_exports_with_a_handle_and_a_uuid() {
        let ctx = crate::GpuContext::new().unwrap();
        let t = CyclesTarget::new(&ctx.device, 64, 32, "test").unwrap();
        assert_ne!(t.uuid, [0u8; 16]);
        let h = t.handle_for(std::process::id()).unwrap();
        assert_ne!(h, 0);
    }
}
