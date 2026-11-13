//! `--shot out.png`: the designer's frame rendered with no window, through
//! egui-wgpu on a headless device. A window's own screenshot only arrives
//! once the window is actually painted, and one that opens behind other
//! windows or on another Space is not. Wobble's `uishot` does the same.

use std::path::Path;

/// Run `frames` passes of `build` on a `w` x `h` point screen and save the
/// last one at `ppp` pixels per point. `events(frame)` is that frame's input.
#[allow(clippy::too_many_arguments)]
pub fn render(
    path: &Path,
    ctx: &egui::Context,
    (w_pt, h_pt): (f32, f32),
    ppp: f32,
    frames: usize,
    events: impl Fn(usize) -> Vec<egui::Event>,
    mut build: impl FnMut(&mut egui::Ui),
) -> Result<(), String> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .map_err(|e| format!("no GPU adapter: {e}"))?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
        .map_err(|e| format!("no GPU device: {e}"))?;

    let mut deltas = Vec::new();
    let mut last = None;
    for frame in 0..frames.max(1) {
        let mut input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w_pt, h_pt))),
            time: Some(frame as f64 / 60.0),
            events: events(frame),
            ..Default::default()
        };
        input.viewports.values_mut().for_each(|v| v.native_pixels_per_point = Some(ppp));
        let out = ctx.run_ui(input, &mut build);
        deltas.extend(out.textures_delta.set.clone());
        last = Some(out);
    }
    let out = last.ok_or("no frame")?;
    let prims = ctx.tessellate(out.shapes, out.pixels_per_point);
    let (w, h) = ((w_pt * ppp) as u32, (h_pt * ppp) as u32);
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shot"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    let mut renderer = egui_wgpu::Renderer::new(&device, format, egui_wgpu::RendererOptions::default());
    for (id, ds) in &deltas {
        for d in ds {
            renderer.update_texture(&device, &queue, *id, d);
        }
    }
    let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [w, h], pixels_per_point: ppp };
    let mut enc = device.create_command_encoder(&Default::default());
    let user = renderer.update_buffers(&device, &queue, &mut enc, &prims, &screen);
    {
        let mut rp = enc
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shot"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            })
            .forget_lifetime();
        renderer.render(&mut rp, &prims, &screen);
    }
    let unpadded = w as usize * 4;
    let padded = unpadded.div_ceil(256) * 256;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shot-readback"),
        size: (padded * h as usize) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded as u32),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit(user.into_iter().chain([enc.finish()]));
    let (tx, rx) = std::sync::mpsc::channel();
    buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
    let mut rgb = Vec::with_capacity(w as usize * h as usize * 3);
    {
        let data = buf.slice(..).get_mapped_range().expect("a mapped readback");
        for y in 0..h as usize {
            for px in data[y * padded..y * padded + unpadded].chunks_exact(4) {
                rgb.extend_from_slice(&px[..3]);
            }
        }
    }
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut png = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    png.set_color(png::ColorType::Rgb);
    png.set_depth(png::BitDepth::Eight);
    png.write_header()
        .and_then(|mut wr| wr.write_image_data(&rgb))
        .map_err(|e| e.to_string())
}

/// A left click at `p` on frame `at`, as egui wants it: move, press, release
/// on the frames after.
pub fn click(p: egui::Pos2, at: usize, frame: usize) -> Vec<egui::Event> {
    let button = |pressed| egui::Event::PointerButton {
        pos: p,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: Default::default(),
    };
    match frame.checked_sub(at) {
        Some(0) => vec![egui::Event::PointerMoved(p)],
        Some(1) => vec![button(true)],
        Some(2) => vec![button(false)],
        _ => Vec::new(),
    }
}
