// SPDX-License-Identifier: GPL-3.0-or-later
// Part of wobble; see LICENSING.md.

//! `wobble_native`: the compiled half of wobble's Blender bridge, loaded by
//! the bridge script inside Blender's own process.
//!
//! What it is for: handing Blender's rendered picture to wobble WITHOUT a
//! readback. The script draws the viewport into a `GPUOffScreen`; this
//! module reaches that offscreen's Metal texture (through symbols Blender's
//! binary exports — see `Syms`), blits it on the GPU into an IOSurface,
//! and wobble maps the same IOSurface as a texture of its own. Nothing is
//! read back, nothing is copied by a CPU.
//!
//! The symbols are Blender's internals, not an API: they are resolved at
//! load with `dlsym`, and `available()` is false when any is missing, so a
//! Blender that renames them costs the fallback (the shared-memory ring),
//! never a crash. macOS only today; the module builds elsewhere as a stub
//! that reports itself unavailable.

use pyo3::prelude::*;

#[cfg(target_os = "macos")]
mod mac {
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_core_foundation::{CFBoolean, CFDictionary, CFNumber, CFRetained, CFString, CFType};
    // Global surfaces are what `IOSurfaceLookup` finds by id from another
    // process, which is the whole point here (the alternative is passing
    // a mach port, and the bridge has no mach channel). Deprecated by
    // Apple as insecure: any process of this user could read the frames.
    #[allow(deprecated)]
    use objc2_io_surface::kIOSurfaceIsGlobal;
    use objc2_io_surface::{
        kIOSurfaceBytesPerElement, kIOSurfaceHeight, kIOSurfacePixelFormat, kIOSurfaceWidth,
        IOSurfaceRef,
    };
    use objc2_metal::{
        MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDevice,
        MTLOrigin, MTLPixelFormat, MTLResource, MTLSize, MTLStorageMode, MTLTexture,
        MTLTextureDescriptor, MTLTextureUsage,
    };
    use pyo3::prelude::*;
    use std::ffi::c_void;
    use std::sync::OnceLock;

    /// `'RGfA'`: 32-bit float RGBA, what a Metal `RGBA32Float` texture
    /// wants on an IOSurface.
    const PIXEL_FORMAT_RGBA32F: i64 = 0x5247_6641;

    type OffscreenColorTexture = unsafe extern "C" fn(*const c_void) -> *mut c_void;
    type GetMetalHandle = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
    type GpuFinish = unsafe extern "C" fn();

    /// Blender's exported symbols, resolved once. The mangled names are
    /// `blender::GPU_offscreen_color_texture(const GPUOffScreen*)`,
    /// `blender::gpu::MTLTexture::get_metal_handle()` and
    /// `blender::GPU_finish()` as Blender 5.2 spells them.
    struct Syms {
        offscreen_color_texture: OffscreenColorTexture,
        get_metal_handle: GetMetalHandle,
        gpu_finish: GpuFinish,
    }

    fn sym(name: &str) -> Option<*mut c_void> {
        let c = std::ffi::CString::new(name).ok()?;
        let p = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c.as_ptr()) };
        (!p.is_null()).then_some(p)
    }

    fn syms() -> Option<&'static Syms> {
        static SYMS: OnceLock<Option<Syms>> = OnceLock::new();
        SYMS.get_or_init(|| {
            let a = sym("_ZN7blender27GPU_offscreen_color_textureEPKNS_12GPUOffScreenE")?;
            let b = sym("_ZN7blender3gpu10MTLTexture16get_metal_handleEv")?;
            let c = sym("_ZN7blender10GPU_finishEv")?;
            // SAFETY: the signatures above are those functions' C++ ABI
            // shapes on arm64/x86_64 (a pointer in, a pointer out; `this`
            // first for the member function).
            Some(unsafe {
                Syms {
                    offscreen_color_texture: std::mem::transmute::<*mut c_void, OffscreenColorTexture>(a),
                    get_metal_handle: std::mem::transmute::<*mut c_void, GetMetalHandle>(b),
                    gpu_finish: std::mem::transmute::<*mut c_void, GpuFinish>(c),
                }
            })
        })
        .as_ref()
    }

    pub fn available() -> bool {
        syms().is_some()
    }

    /// The `GPUOffScreen*` inside a `gpu.types.GPUOffScreen` Python object:
    /// the one pointer field right after the object header
    /// (`BPyGPUOffScreen { PyObject_HEAD; GPUOffScreen *offscreen; }`).
    fn offscreen_ptr(obj: &Bound<'_, PyAny>) -> PyResult<*const c_void> {
        let ty = obj.get_type();
        let name = ty.name()?;
        if name != "GPUOffScreen" {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "expected a gpu.types.GPUOffScreen, got {name}"
            )));
        }
        let base = obj.as_ptr() as *const u8;
        // SAFETY: the type was checked; the field follows PyObject_HEAD in
        // every CPython Blender ships (no free-threaded builds).
        let p = unsafe { *(base.add(std::mem::size_of::<pyo3::ffi::PyObject>()) as *const *const c_void) };
        if p.is_null() {
            return Err(pyo3::exceptions::PyValueError::new_err("the offscreen has been freed"));
        }
        Ok(p)
    }

    fn make_surface(w: u32, h: u32) -> Option<CFRetained<IOSurfaceRef>> {
        let width = CFNumber::new_i64(w as i64);
        let height = CFNumber::new_i64(h as i64);
        let bpe = CFNumber::new_i64(16);
        let fmt = CFNumber::new_i64(PIXEL_FORMAT_RGBA32F);
        let yes: &CFBoolean = unsafe { objc2_core_foundation::kCFBooleanTrue }?;
        #[allow(deprecated)]
        let keys: [&CFString; 5] = unsafe {
            [kIOSurfaceWidth, kIOSurfaceHeight, kIOSurfaceBytesPerElement, kIOSurfacePixelFormat, kIOSurfaceIsGlobal]
        };
        let values: [&CFType; 5] = [&width, &height, &bpe, &fmt, yes];
        let dict = CFDictionary::from_slices(&keys, &values);
        unsafe { IOSurfaceRef::new(dict.as_opaque()) }
    }

    /// N IOSurfaces of one size, each wrapped as a Metal texture on
    /// Blender's device once the first blit reveals which device that is.
    #[pyclass(unsendable)]
    pub struct SurfaceRing {
        surfaces: Vec<CFRetained<IOSurfaceRef>>,
        textures: Vec<Retained<ProtocolObject<dyn MTLTexture>>>,
        queue: Option<Retained<ProtocolObject<dyn MTLCommandQueue>>>,
        w: u32,
        h: u32,
    }

    #[pymethods]
    impl SurfaceRing {
        #[new]
        fn new(slots: usize, w: u32, h: u32) -> PyResult<Self> {
            if !available() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "this Blender does not export the GPU symbols the bridge needs",
                ));
            }
            if slots == 0 || w == 0 || h == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err("empty ring"));
            }
            let mut surfaces = Vec::with_capacity(slots);
            for _ in 0..slots {
                surfaces.push(make_surface(w, h).ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("IOSurfaceCreate failed")
                })?);
            }
            Ok(SurfaceRing { surfaces, textures: Vec::new(), queue: None, w, h })
        }

        /// The surfaces' global ids, what wobble looks them up by.
        fn ids(&self) -> Vec<u32> {
            self.surfaces.iter().map(|s| s.id()).collect()
        }

        fn width(&self) -> u32 {
            self.w
        }

        fn height(&self) -> u32 {
            self.h
        }

        /// Copy the offscreen's colour texture (its top-left `w` x `h`)
        /// into slot `slot`, on the GPU, and wait for the copy: when this
        /// returns the surface holds the frame.
        fn blit(&mut self, offscreen: &Bound<'_, PyAny>, slot: usize, w: u32, h: u32) -> PyResult<()> {
            let s = syms().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("symbols missing"))?;
            if slot >= self.surfaces.len() || w == 0 || h == 0 || w > self.w || h > self.h {
                return Err(pyo3::exceptions::PyValueError::new_err("slot or size out of range"));
            }
            let off = offscreen_ptr(offscreen)?;
            let tex = unsafe { (s.offscreen_color_texture)(off) };
            if tex.is_null() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err("the offscreen has no colour texture"));
            }
            // Blender's draw into that texture is queued on its own command
            // buffers; finish them before another queue reads the texture.
            unsafe { (s.gpu_finish)() };
            let raw = unsafe { (s.get_metal_handle)(tex) } as *mut ProtocolObject<dyn MTLTexture>;
            let src: Retained<ProtocolObject<dyn MTLTexture>> = unsafe { Retained::retain(raw) }
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("no Metal texture behind the offscreen"))?;
            if src.width() < w as usize || src.height() < h as usize {
                return Err(pyo3::exceptions::PyValueError::new_err("the offscreen is smaller than the copy"));
            }
            let device = src.device();
            if self.textures.is_empty() {
                let desc = unsafe {
                    MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                        MTLPixelFormat::RGBA32Float,
                        self.w as usize,
                        self.h as usize,
                        false,
                    )
                };
                desc.setStorageMode(MTLStorageMode::Shared);
                desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
                for surf in &self.surfaces {
                    let t = device.newTextureWithDescriptor_iosurface_plane(&desc, surf, 0)
                        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Metal refused the IOSurface"))?;
                    self.textures.push(t);
                }
                self.queue = device.newCommandQueue();
            }
            let queue = self.queue.as_ref().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("no command queue"))?;
            let cb = queue.commandBuffer().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("no command buffer"))?;
            let enc = cb.blitCommandEncoder().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("no blit encoder"))?;
            unsafe {
                enc.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    &src,
                    0,
                    0,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                    MTLSize { width: w as usize, height: h as usize, depth: 1 },
                    &self.textures[slot],
                    0,
                    0,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                );
            }
            enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
            Ok(())
        }
    }
}

#[cfg(target_os = "macos")]
use mac::SurfaceRing;

/// Whether this Blender, on this platform, can hand textures over.
#[pyfunction]
fn available() -> bool {
    #[cfg(target_os = "macos")]
    {
        mac::available()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

#[pymodule]
fn wobble_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(available, m)?)?;
    #[cfg(target_os = "macos")]
    m.add_class::<SurfaceRing>()?;
    Ok(())
}
