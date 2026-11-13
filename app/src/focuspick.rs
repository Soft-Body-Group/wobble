//! Defocus focus pick: read the depth pass under the pointer and make it the
//! focal plane.
//!
//! The panel's pick button ARMS it for one Defocus node; a press on the
//! viewer then samples, a drag keeps sampling (a live focus pull), and the
//! release disarms. What is read is the node's OWN depth texture — the Z
//! port as its stage resolved it this frame — not the pixels on screen, so
//! the node need not be the view target and the value is exactly what the
//! CoC maths sees. `focus` is a raw pass value, so it lands unconverted.
//!
//! Same readback shape as the pixel picker: one texel, one read in flight,
//! landing a frame or two later.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

struct Read {
    node: u64,
    buffer: wgpu::Buffer,
    ready: Arc<AtomicBool>,
    mapped: bool,
}

#[derive(Default)]
pub struct FocusPick {
    /// The Defocus node the next viewer press samples for.
    pub armed: Option<u64>,
    /// Pointer in normalised canvas coordinates while a press is held.
    pub cursor: Option<[f32; 2]>,
    /// The viewer saw the press: its release is what disarms.
    pub held: bool,
    read: Option<Read>,
}

impl FocusPick {
    pub fn toggle(&mut self, node: u64) {
        self.armed = if self.armed == Some(node) { None } else { Some(node) };
        self.cursor = None;
        self.held = false;
    }

    /// Queue a one-texel copy of `tex` under the cursor.
    pub fn encode(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, tex: &wgpu::Texture) {
        let (Some(node), Some([u, v])) = (self.armed, self.cursor) else { return };
        if self.read.is_some() || !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
            return;
        }
        let (w, h) = (tex.width(), tex.height());
        let x = ((u * w as f32) as u32).min(w.saturating_sub(1));
        let y = ((v * h as f32) as u32).min(h.saturating_sub(1));
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("focus-pick"),
            size: 256,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        self.read = Some(Read { node, buffer, ready: Arc::new(AtomicBool::new(false)), mapped: false });
    }

    /// A landed sample: (node, raw depth value). Call once a frame. The
    /// depth port is a Data pipe — one channel broadcast to rgb — so R is it.
    pub fn poll(&mut self) -> Option<(u64, f32)> {
        let r = self.read.as_mut()?;
        if !r.mapped {
            r.mapped = true;
            let flag = r.ready.clone();
            r.buffer.slice(..).map_async(wgpu::MapMode::Read, move |res| {
                if res.is_ok() {
                    flag.store(true, Ordering::Release);
                }
            });
        }
        if !r.ready.load(Ordering::Acquire) {
            return None;
        }
        let r = self.read.take()?;
        let z = {
            let view = r.buffer.slice(..).get_mapped_range().expect("a mapped buffer");
            let px: &[half::f16] = bytemuck::cast_slice(&view[..8]);
            px[0].to_f32()
        };
        r.buffer.unmap();
        z.is_finite().then_some((r.node, z))
    }
}
