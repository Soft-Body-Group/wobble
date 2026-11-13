//! Per-pixel adaptive "ink" for viewer overlays: gizmo/HUD vector marks are
//! tessellated by epaint but rendered by our own pipeline inside the egui
//! render pass — the fragment shader samples the comp under each fragment
//! and paints pure black over bright pixels, pure white over dark ones.

use egui::epaint;

const INK_WGSL: &str = r#"
// viewport = the CALLBACK rect in points (egui-wgpu sets the GPU viewport
// to the callback rect, so NDC spans that rect, not the window).
// frost: x = blur radius (px), y = darken 0..1.
// palette: the colours nodes asked their marks to wear this frame, display
// encoded; a mark names one by index (see `fs`).
struct U { viewport: vec4f, dest: vec4f, frost: vec4f, palette: array<vec4f, 16> }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var comp: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VOut { @builtin(position) pos: vec4f, @location(0) misc: vec4f }

@vertex
fn vs(@location(0) p: vec2f, @location(1) misc: vec4f) -> VOut {
    var o: VOut;
    let ndc = (p - u.viewport.xy) / u.viewport.zw * 2.0 - 1.0;
    o.pos = vec4f(ndc.x, -ndc.y, 0.0, 1.0);
    o.misc = misc;
    return o;
}

// Clamp-to-edge: off-image taps repeat the border instead of averaging in
// letterbox black, so panels near (or past) the image edge never darken.
fn bg_clamped(uv: vec2f) -> vec3f {
    return textureSampleLevel(comp, samp, clamp(uv, vec2f(0.0), vec2f(1.0)), 0.0).rgb;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
    let uv = (in.pos.xy - u.dest.xy) / u.dest.zw;
    let cov0 = in.misc.w;
    // Every flag rides the vertex colour as a RATIO of its coverage: epaint
    // feathers a shape by interpolating its colour towards transparent,
    // which scales all four channels together, so a ratio holds across the
    // fringe where an absolute value would not — at exactly the half-covered
    // pixels a rim is made of, a lane would fall back to plain ink and a
    // palette index would slide onto a neighbour's colour.
    let n = in.misc / max(cov0, 1e-4);
    // Hover snaps to white before anything else is read: a highlighted mark
    // is (a, a, a, a), which would otherwise decode as the frost sentinel.
    let hi = n.x > 0.5;
    let idx0 = i32(n.y * 3.984375 + 0.5);
    if (!hi && idx0 >= 4) {
        // Frost material: heavy disc blur of whatever sits under the panel
        // (old-iOS frosted glass), lightly desaturated, never darkening
        // (edge-clamped taps; luminance preserved).
        var acc = bg_clamped(uv);
        var wsum = 1.0;
        for (var ring = 1; ring <= 6; ring++) {
            let rr = f32(ring) / 6.0 * u.frost.x; // radius in dest px
            let n = 6 + ring * 6;
            for (var k = 0; k < n; k++) {
                let a = f32(k) / f32(n) * 6.2831853 + f32(ring) * 0.7;
                let off = vec2f(cos(a), sin(a)) * rr / u.dest.zw;
                acc += bg_clamped(uv + off);
                wsum += 1.0;
            }
        }
        // The comp texture is scene-linear (the view transform lives at
        // the blit), so tone it before showing it as frost — otherwise
        // anything above 1.0 clips to white through the panel.
        let b = vt_display_tone(acc / wsum);
        let g = dot(b, vec3f(0.2126, 0.7152, 0.0722));
        let frost = mix(b, vec3f(g), 0.35) * (1.0 - u.frost.y);
        return vec4f(frost * cov0, cov0); // premultiplied
    }
    var luma = 0.0;
    if (all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0))) {
        // Wide blurred lookup (weighted 3x3, ~16px apart) so the black/white
        // pick keys off neighborhood brightness, not single-texel noise.
        let o = vec2f(16.0) / u.dest.zw;
        var sum = vec3f(0.0);
        var wsum = 0.0;
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let w = select(1.0, 2.0, dx == 0) * select(1.0, 2.0, dy == 0);
                let q = clamp(uv + vec2f(f32(dx), f32(dy)) * o, vec2f(0.0), vec2f(1.0));
                sum += textureSampleLevel(comp, samp, q, 0.0).rgb * w;
                wsum += w;
            }
        }
        luma = vt_display_luma(sum / wsum);
    }
    // Soft monochrome ink: light grey over dark image, near-black over
    // bright — the pick dissolves across a wide luma band instead of
    // flipping at a threshold; hovered marks snap to pure white.
    let base = mix(vec3f(0.78), vec3f(0.12), smoothstep(0.28, 0.62, luma));
    let cov = in.misc.w;
    var ink = select(base, vec3f(1.0), hi);
    if (!hi && idx0 > 0) {
        // Subtle channel tint: lane 1/2/3 = r/g/b (a quarter of coverage
        // each), misc.z = strength. Applied after the luma pick.
        var tint = vec3f(0.79, 0.35, 0.33);
        if (idx0 == 2) { tint = vec3f(0.39, 0.70, 0.36); }
        if (idx0 == 3) { tint = vec3f(0.37, 0.51, 0.83); }
        ink = mix(ink, tint, in.misc.z);
    } else if (!hi) {
        // A node's own HUD colour: the blue channel names a palette slot in
        // sixteenths of coverage. It replaces the adaptive pick outright —
        // the point of choosing a colour is that it stays that colour.
        let pal = i32(n.z * 16.0 + 0.5);
        if (pal > 0) {
            ink = u.palette[min(pal, 16) - 1].rgb;
        }
    }
    return vec4f(ink * cov, cov); // premultiplied
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 2],
    /// (highlight-red, tint-lane, tint-strength, coverage-alpha) from the
    /// tessellated vertex color.
    misc: [f32; 4],
}

/// How many distinct colours nodes can ask their marks to wear in one
/// frame: the index is four bits of the vertex colour (`tint_color`), and
/// zero means the adaptive ink.
pub const PALETTE: usize = 15;

/// Shape collector with the small painter-like API the gizmos use. Colors
/// are ink-decided per pixel; callers only supply coverage alpha — or, for
/// a node that chose its own HUD colour, `set_tint` before its marks.
#[derive(Default)]
pub struct InkShapes {
    shapes: Vec<epaint::Shape>,
    /// The colours handed a slot this frame, display-encoded rgb.
    palette: Vec<[f32; 3]>,
    /// The slot (1-based) plain marks are currently drawn in; 0 = adaptive.
    tint: usize,
}

/// The ink shader decides the tone; here color only encodes coverage alpha
/// and, via the red channel, the hover-highlight flag.
fn ink_color(alpha: f32, hi: bool) -> egui::Color32 {
    let a = alpha.clamp(0.0, 1.0);
    if hi {
        egui::Color32::WHITE.gamma_multiply(a)
    } else {
        egui::Color32::from_black_alpha((a * 255.0) as u8)
    }
}

/// Like ink_color, but requesting a subtle per-channel tint (lane 0/1/2 =
/// r/g/b) mixed in by the shader after the luma pick. Green carries the
/// lane as a quarter of coverage per lane (a ratio, so the feathered fringe
/// decodes as the same lane), blue the strength; hover marks fall back to
/// the white snap.
fn lane_color(alpha: f32, lane: usize) -> egui::Color32 {
    let a = (alpha.clamp(0.0, 1.0) * 255.0) as u8;
    let g = (a as f32 * (lane.min(2) + 1) as f32 / 4.0).round() as u8;
    egui::Color32::from_rgba_premultiplied(0, g, (0.35f32 * 255.0) as u8, a)
}

/// A mark in palette slot `slot` (1-based): the index rides the blue
/// channel in sixteenths of coverage. A byte cannot hold that ratio to
/// within half a step under an alpha of ~16/255 (the error is 8/a), so a
/// tinted mark is never fainter than 20/255 — the alternative is a mark in
/// somebody else's colour.
fn tint_color(alpha: f32, slot: usize) -> egui::Color32 {
    let a = ((alpha.clamp(0.0, 1.0) * 255.0).round() as u8).max(20);
    let b = (a as f32 * slot.min(PALETTE) as f32 / 16.0).round() as u8;
    egui::Color32::from_rgba_premultiplied(0, 0, b, a)
}

impl InkShapes {
    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }

    /// The colour plain marks are drawn in from here on: `Some` display-
    /// encoded rgb for a node that chose one, `None` for the adaptive ink.
    /// Lane-tinted marks and frost are unaffected. A colour beyond the
    /// `PALETTE` distinct ones a frame can carry falls back to adaptive.
    pub fn set_tint(&mut self, col: Option<[f32; 3]>) {
        self.tint = match col {
            None => 0,
            Some(c) => {
                let c = c.map(|x| if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 });
                match self.palette.iter().position(|p| *p == c) {
                    Some(i) => i + 1,
                    None if self.palette.len() < PALETTE => {
                        self.palette.push(c);
                        self.palette.len()
                    }
                    None => 0,
                }
            }
        }
    }

    /// The frame's palette as the shader's uniform array (unused slots
    /// black; the sixteenth is never named).
    pub fn palette(&self) -> [[f32; 4]; 16] {
        let mut out = [[0.0; 4]; 16];
        for (o, c) in out.iter_mut().zip(&self.palette) {
            *o = [c[0], c[1], c[2], 1.0];
        }
        out
    }

    /// Plain ink in the current tint (hover still snaps white).
    fn ink(&self, alpha: f32, hi: bool) -> egui::Color32 {
        if hi || self.tint == 0 {
            ink_color(alpha, hi)
        } else {
            tint_color(alpha, self.tint)
        }
    }

    /// Frosted-glass panel: the shader replaces this rect with a heavily
    /// blurred sample of the comp underneath (lane sentinel 4 = material,
    /// green at full coverage).
    pub fn frost(&mut self, rect: egui::Rect, rounding: f32, alpha: f32) {
        let a = (alpha.clamp(0.0, 1.0) * 255.0) as u8;
        self.shapes.push(epaint::Shape::rect_filled(
            rect,
            rounding,
            egui::Color32::from_rgba_premultiplied(0, a, 0, a),
        ));
    }

    /// seg with a subtle channel tint (hover still snaps white).
    pub fn seg_lane(
        &mut self,
        a: egui::Pos2,
        b: egui::Pos2,
        w: f32,
        alpha: f32,
        hi: bool,
        lane: usize,
    ) {
        let col = if hi { ink_color(alpha, true) } else { lane_color(alpha, lane) };
        self.shapes.push(epaint::Shape::line_segment([a, b], egui::Stroke::new(w, col)));
    }

    /// One polyline as ONE shape: a 500-vertex outline is one path to
    /// tessellate, not 500 segments. Plain ink, or a lane tint.
    pub fn path(
        &mut self,
        pts: Vec<egui::Pos2>,
        closed: bool,
        w: f32,
        alpha: f32,
        lane: Option<usize>,
    ) {
        if pts.len() < 2 {
            return;
        }
        let col = match lane {
            Some(l) => lane_color(alpha, l),
            None => self.ink(alpha, false),
        };
        let stroke = egui::Stroke::new(w, col);
        self.shapes.push(if closed {
            epaint::Shape::closed_line(pts, stroke)
        } else {
            epaint::Shape::line(pts, stroke)
        });
    }

    /// Many small marks as ONE mesh: a square dot per position and, when
    /// given, a thin quad from the dot to `tick`. Thousands of points cost
    /// one shape, not thousands.
    pub fn dots(
        &mut self,
        marks: impl Iterator<Item = (egui::Pos2, Option<egui::Pos2>)>,
        r: f32,
        alpha: f32,
        tick_w: f32,
    ) {
        let col = self.ink(alpha, false);
        let mut mesh = epaint::Mesh::default();
        let quad = |mesh: &mut epaint::Mesh, p: [egui::Pos2; 4]| {
            let i = mesh.vertices.len() as u32;
            for q in p {
                mesh.vertices.push(epaint::Vertex {
                    pos: q,
                    uv: epaint::WHITE_UV,
                    color: col,
                });
            }
            mesh.indices.extend_from_slice(&[i, i + 1, i + 2, i, i + 2, i + 3]);
        };
        for (c, tick) in marks {
            quad(
                &mut mesh,
                [
                    egui::pos2(c.x - r, c.y - r),
                    egui::pos2(c.x + r, c.y - r),
                    egui::pos2(c.x + r, c.y + r),
                    egui::pos2(c.x - r, c.y + r),
                ],
            );
            if let Some(t) = tick {
                let d = t - c;
                if d.length_sq() > 1e-6 {
                    let n = egui::vec2(-d.y, d.x).normalized() * (tick_w * 0.5);
                    quad(&mut mesh, [c + n, t + n, t - n, c - n]);
                }
            }
        }
        if !mesh.indices.is_empty() {
            self.shapes.push(epaint::Shape::mesh(mesh));
        }
    }

    /// Many short segments as ONE mesh in a lane tint (normals, hairs).
    pub fn segs_lane(
        &mut self,
        segs: impl Iterator<Item = (egui::Pos2, egui::Pos2)>,
        w: f32,
        alpha: f32,
        lane: usize,
    ) {
        let col = lane_color(alpha, lane);
        let mut mesh = epaint::Mesh::default();
        for (a, b) in segs {
            let d = b - a;
            if d.length_sq() < 1e-6 {
                continue;
            }
            let n = egui::vec2(-d.y, d.x).normalized() * (w * 0.5);
            let i = mesh.vertices.len() as u32;
            for q in [a + n, b + n, b - n, a - n] {
                mesh.vertices.push(epaint::Vertex { pos: q, uv: epaint::WHITE_UV, color: col });
            }
            mesh.indices.extend_from_slice(&[i, i + 1, i + 2, i, i + 2, i + 3]);
        }
        if !mesh.indices.is_empty() {
            self.shapes.push(epaint::Shape::mesh(mesh));
        }
    }

    /// circle_fill with a subtle channel tint (hover still snaps white).
    pub fn circle_fill_lane(
        &mut self,
        c: egui::Pos2,
        r: f32,
        alpha: f32,
        hi: bool,
        lane: usize,
    ) {
        let col = if hi { ink_color(alpha, true) } else { lane_color(alpha, lane) };
        self.shapes.push(epaint::Shape::circle_filled(c, r, col));
    }

    pub fn seg(&mut self, a: egui::Pos2, b: egui::Pos2, w: f32, alpha: f32, hi: bool) {
        let col = self.ink(alpha, hi);
        self.shapes.push(epaint::Shape::line_segment([a, b], egui::Stroke::new(w, col)));
    }

    pub fn circle_stroke(&mut self, c: egui::Pos2, r: f32, w: f32, alpha: f32, hi: bool) {
        let col = self.ink(alpha, hi);
        self.shapes.push(epaint::Shape::circle_stroke(c, r, egui::Stroke::new(w, col)));
    }

    pub fn circle_fill(&mut self, c: egui::Pos2, r: f32, alpha: f32, hi: bool) {
        let col = self.ink(alpha, hi);
        self.shapes.push(epaint::Shape::circle_filled(c, r, col));
    }

    pub fn rect(
        &mut self,
        rect: egui::Rect,
        rounding: f32,
        fill: bool,
        w: f32,
        alpha: f32,
        hi: bool,
    ) {
        let col = self.ink(alpha, hi);
        let (f, s) = if fill {
            (col, egui::Stroke::NONE)
        } else {
            (egui::Color32::TRANSPARENT, egui::Stroke::new(w, col))
        };
        self.shapes.push(epaint::Shape::rect_stroke(
            rect,
            rounding,
            s,
            egui::StrokeKind::Outside,
        ));
        if fill {
            self.shapes.push(epaint::Shape::rect_filled(rect, rounding, f));
        }
    }

    /// Tessellate everything into one mesh (positions in points).
    pub fn tessellate(self, pixels_per_point: f32) -> epaint::Mesh {
        let palette = self.palette();
        let mesh = self.tessellate_marks(pixels_per_point);
        CAPTURE.with(|c| {
            if let Some(slot) = c.borrow_mut().as_mut() {
                *slot = Some((mesh.clone(), palette));
            }
        });
        mesh
    }

    fn tessellate_marks(self, pixels_per_point: f32) -> epaint::Mesh {
        let mut tess = epaint::Tessellator::new(
            pixels_per_point,
            epaint::TessellationOptions::default(),
            [1, 1],
            Vec::new(),
        );
        let mut mesh = epaint::Mesh::default();
        for s in self.shapes {
            tess.tessellate_shape(s, &mut mesh);
        }
        mesh
    }
}

thread_local! {
    /// While armed, the last ink mesh tessellated on this thread and its
    /// palette: a pass drawn for ANOTHER screen (`apiserve::remote`) has no
    /// wgpu to run the ink callback on, so it takes the marks and sends
    /// them, and the screen runs the same per-pixel pick over its picture.
    static CAPTURE: std::cell::RefCell<Option<Option<(epaint::Mesh, [[f32; 4]; 16])>>> =
        const { std::cell::RefCell::new(None) };
}

/// Start keeping the ink this thread tessellates (`capture_take` ends it).
pub fn capture_begin() {
    CAPTURE.with(|c| *c.borrow_mut() = Some(None));
}

/// The ink tessellated since `capture_begin`, if any; disarms.
pub fn capture_take() -> Option<(epaint::Mesh, [[f32; 4]; 16])> {
    CAPTURE.with(|c| c.borrow_mut().take().flatten())
}

/// `U` in the shader: three vec4 + the 16-entry palette.
const UNIFORM_BYTES: u64 = 48 + 16 * 16;

/// Persistent GPU state, stored in the egui renderer's callback resources.
pub struct InkRender {
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    vbuf: Option<wgpu::Buffer>,
    ibuf: Option<wgpu::Buffer>,
    index_count: u32,
    bind: Option<wgpu::BindGroup>,
}

impl InkRender {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ink"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{}{}", engine::viewtf::DISPLAY_ESTIMATE_WGSL, INK_WGSL).into(),
            ),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ink"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ink-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ink-uniform"),
            size: UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { pipeline, sampler, uniform, vbuf: None, ibuf: None, index_count: 0, bind: None }
    }
}

/// One frame's ink draw: the tessellated mesh + comp view + image dest rect.
pub struct InkCallback {
    pub mesh: epaint::Mesh,
    /// The callback rect (viewer panel) in POINTS: [x, y, w, h].
    pub clip: [f32; 4],
    /// Where the comp image lands on screen, physical px: [x, y, w, h].
    pub dest: [f32; 4],
    /// [blur radius px, darken 0..1, 0, 0] for the frost material.
    pub frost: [f32; 4],
    /// The node HUD colours marks name by slot (`InkShapes::palette`).
    pub palette: [[f32; 4]; 16],
    pub comp_view: wgpu::TextureView,
}

impl egui_wgpu::CallbackTrait for InkCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(r) = resources.get_mut::<InkRender>() else { return Vec::new() };
        let _ = screen;
        let mut u: Vec<f32> = Vec::with_capacity(UNIFORM_BYTES as usize / 4);
        u.extend_from_slice(&self.clip);
        u.extend_from_slice(&self.dest);
        u.extend_from_slice(&self.frost);
        for c in &self.palette {
            u.extend_from_slice(c);
        }
        debug_assert_eq!(u.len() * 4, UNIFORM_BYTES as usize);
        queue.write_buffer(&r.uniform, 0, bytemuck::cast_slice(&u));

        let verts: Vec<Vertex> = self
            .mesh
            .vertices
            .iter()
            .map(|v| Vertex {
                pos: [v.pos.x, v.pos.y],
                misc: [
                    v.color.r() as f32 / 255.0,
                    v.color.g() as f32 / 255.0,
                    v.color.b() as f32 / 255.0,
                    v.color.a() as f32 / 255.0,
                ],
            })
            .collect();
        use wgpu::util::DeviceExt;
        r.vbuf = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ink-vb"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        }));
        r.ibuf = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ink-ib"),
            contents: bytemuck::cast_slice(&self.mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        }));
        r.index_count = self.mesh.indices.len() as u32;
        r.bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ink-bg"),
            layout: &r.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: r.uniform.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.comp_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&r.sampler),
                },
            ],
        }));
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(r) = resources.get::<InkRender>() else { return };
        let (Some(vb), Some(ib), Some(bg)) = (&r.vbuf, &r.ibuf, &r.bind) else { return };
        if r.index_count == 0 {
            return;
        }
        pass.set_pipeline(&r.pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..r.index_count, 0, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What `fs` reads off a fragment's `misc`: (highlight, lane/frost
    /// index, palette slot). Kept in step with the shader by hand — it IS
    /// the contract the encoders above are written against.
    fn decode_misc([r, g, b, a]: [f32; 4]) -> (bool, i32, usize) {
        let n = [r, g, b].map(|x| x / a.max(1e-4));
        let hi = n[0] > 0.5;
        let idx = (n[1] * 3.984_375 + 0.5) as i32;
        let pal = if hi || idx > 0 { 0 } else { (n[2] * 16.0 + 0.5) as usize };
        (hi, idx.min(4), pal)
    }

    /// A vertex colour as `prepare` hands it to the shader.
    fn misc(c: egui::Color32) -> [f32; 4] {
        [c.r(), c.g(), c.b(), c.a()].map(|x| x as f32 / 255.0)
    }

    fn decode(c: egui::Color32) -> (bool, i32, usize) {
        decode_misc(misc(c))
    }

    /// The GPU interpolates towards epaint's transparent outer vertex in
    /// float, all four channels together — no second byte rounding.
    fn feathered(c: egui::Color32, t: f32) -> [f32; 4] {
        misc(c).map(|x| x * t)
    }

    const ALPHAS: [f32; 8] = [0.05, 0.125, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0];

    #[test]
    fn a_palette_slot_survives_the_byte_and_the_fringe() {
        for slot in 1..=PALETTE {
            for a in ALPHAS {
                let c = tint_color(a, slot);
                assert_eq!(decode(c), (false, 0, slot), "slot {slot} at alpha {a}");
                for t in [0.9, 0.6, 0.35] {
                    assert_eq!(
                        decode_misc(feathered(c, t)),
                        (false, 0, slot),
                        "slot {slot} alpha {a} feathered {t}"
                    );
                }
            }
        }
    }

    #[test]
    fn plain_lane_frost_and_highlight_decode_as_themselves() {
        for a in ALPHAS {
            assert_eq!(decode(ink_color(a, false)), (false, 0, 0), "plain at {a}");
            assert_eq!(decode(ink_color(a, true)).0, true, "hover at {a}");
            for lane in 0..3 {
                let c = lane_color(a, lane);
                assert_eq!(decode(c), (false, lane as i32 + 1, 0), "lane {lane} at {a}");
                assert_eq!(decode_misc(feathered(c, 0.5)), (false, lane as i32 + 1, 0));
            }
        }
        let mut ink = InkShapes::default();
        ink.frost(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(10.0, 10.0)), 0.0, 0.5);
        let epaint::Shape::Rect(r) = &ink.shapes[0] else { panic!("frost is a rect") };
        assert_eq!(decode(r.fill), (false, 4, 0));
        assert_eq!(decode_misc(feathered(r.fill, 0.5)), (false, 4, 0), "frost rim stays frost");
    }

    #[test]
    fn slots_are_handed_out_once_per_colour_and_run_out_gracefully() {
        let mut ink = InkShapes::default();
        assert_eq!(ink.tint, 0);
        ink.set_tint(Some([1.0, 0.5, 0.0]));
        assert_eq!(ink.tint, 1);
        ink.set_tint(Some([0.0, 0.5, 1.0]));
        assert_eq!(ink.tint, 2);
        ink.set_tint(Some([1.0, 0.5, 0.0]));
        assert_eq!(ink.tint, 1, "the same colour reuses its slot");
        ink.set_tint(None);
        assert_eq!(ink.tint, 0);
        for i in 0..PALETTE {
            ink.set_tint(Some([i as f32 / 32.0, 0.1, 0.2]));
        }
        assert_eq!(ink.palette.len(), PALETTE);
        ink.set_tint(Some([0.9, 0.9, 0.9]));
        assert_eq!(ink.tint, 0, "a sixteenth colour draws adaptive, not in another's slot");
        let pal = ink.palette();
        assert_eq!(pal[0], [1.0, 0.5, 0.0, 1.0]);
        assert_eq!(pal[15], [0.0; 4]);
        // The tinted marks themselves carry the slot.
        ink.set_tint(Some([0.0, 0.5, 1.0]));
        ink.seg(egui::Pos2::ZERO, egui::pos2(1.0, 1.0), 1.0, 0.8, false);
        let epaint::Shape::LineSegment { stroke, .. } = &ink.shapes[0] else { panic!("a segment") };
        assert_eq!(decode(stroke.color), (false, 0, 2));
        ink.seg(egui::Pos2::ZERO, egui::pos2(1.0, 1.0), 1.0, 0.8, true);
        let epaint::Shape::LineSegment { stroke, .. } = &ink.shapes[1] else { panic!("a segment") };
        assert!(decode(stroke.color).0, "hover over a tinted node still snaps white");
    }
}
