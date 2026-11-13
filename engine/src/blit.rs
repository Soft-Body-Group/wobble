//! Blit: fullscreen triangle mapping the fused output into an arbitrary
//! destination rect (viewer pan/zoom); pixels outside the rect are the
//! backdrop colour ([`Blit::set_backdrop`], black until set).
//!
//! This is also where the **view transform** lives. The fused pass leaves the
//! output texture scene-linear, so the blit does exposure, the display
//! rendering transform (see [`crate::viewtf`]) and the inverse display EOTF,
//! in that order. Keeping it here rather than in the fused tail means comp
//! output, the NVMe render cache and any exported frames all stay
//! scene-referred, and auditioning a transform costs one pipeline rebuild
//! rather than a re-render.
//!
//! If the target format is sRGB the hardware does the final encode; otherwise
//! the shader does it. A non-sRGB EOTF on an sRGB target cannot be honoured —
//! [`crate::viewtf::ViewSettings::hardware_encode_ok`] reports that case.
//!
//! Sampling is bilinear when the comp is minified and NEAREST once it is
//! magnified past 1:1, so zooming in shows hard pixel edges rather than an
//! interpolated smear. That switch is made in the shader from the ratio of
//! the destination rect to the texture size, not by swapping samplers.

use crate::viewtf::{BlitUniform, ViewSettings};
use web_time::Instant;

/// Bindings, vertex stage and the fragment tail. `TRANSFORM` is replaced with
/// the selected view transform's WGSL (which defines `view_transform`) and
/// `ENCODE` with the inverse-EOTF expression over `c`.
const BLIT_WGSL: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> u: U;
// A 3D LUT for transforms built on one (AgX). Always declared AND touched in
// `fs`, so every pipeline's auto-derived layout has the same four bindings
// and one bind-group shape serves them all; a 1^3 placeholder otherwise.
@group(0) @binding(3) var lut3d: texture_3d<f32>;

struct VOut { @builtin(position) pos: vec4f }

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    var out: VOut;
    let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
    out.pos = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

TRANSFORM

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
    _ = textureDimensions(lut3d);
    let uv = (in.pos.xy - u.dest.xy) / u.dest.zw;
    if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) {
        // An overlay blit (a second viewer pane) leaves everything outside
        // its rect alone; its backdrop alpha is the negative sentinel.
        if (u.backdrop.a < 0.0) {
            discard;
        }
        return u.backdrop;
    }
    // Past 1:1 you are looking at PIXELS, not at an image: snap to the texel
    // centre so each one reads as a hard square instead of a bilinear smear.
    // Sampling a linear sampler exactly at a texel centre returns that texel
    // untouched, so this needs no second sampler and no bind-group swap.
    //
    // Only when magnifying. Minification still filters, because there the
    // blur is doing real work — snapping would alias the image to bits.
    let ts = vec2f(textureDimensions(src));
    var suv = uv;
    if (u.dest.z > ts.x) {
        suv = (min(floor(uv * ts), ts - vec2f(1.0)) + vec2f(0.5)) / ts;
    }
    let s = textureSampleLevel(src, samp, suv, 0.0);
    // Alpha view shows coverage as grey and skips the transform; it is a
    // display-referred value already.
    // A single colour channel (mode.w = 1/2/3 for r/g/b) is shown as grey
    // and DOES go through exposure, the transform and gamma: it is a scene
    // value like any other, just one of them.
    var c: vec3f;
    var rgb = s.rgb;
    let ch = i32(u.mode.w + 0.5);
    if (ch >= 1 && ch <= 3) {
        rgb = vec3f(s[ch - 1]);
    }
    if (u.mode.x > 0.5) {
        c = vec3f(clamp(s.a, 0.0, 1.0));
    } else {
        c = pow(max(view_transform(rgb * u.mode.y), vec3f(0.0)), vec3f(u.mode.z));
    }
    return vec4f(ENCODE, select(s.a, 1.0, u.mode.x > 0.5 || ch >= 1));
}
"#;

pub struct Blit {
    target_format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    /// `ViewSettings::shader_key` the pipeline was built for.
    shader_key: u64,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    bind_group: Option<(wgpu::BindGroup, usize)>,
    generation: usize,
    /// What `fs` writes outside the destination rect, ready for the target.
    backdrop: [f32; 4],
    /// Bound at `lut3d`: the cube the current transform needs, or 1^3.
    lut_view: wgpu::TextureView,
    /// Which display's AgX cube is uploaded (None = the placeholder).
    lut_loaded: Option<crate::viewtf::Gamut>,
    /// Which one the current view wants (set by `set_view`).
    lut_want: Option<crate::viewtf::Gamut>,
    /// Single-channel view: 0 = rgb, 1/2/3 = r/g/b as grey (`set_channel`).
    channel: u8,
    /// Draw OVER what the target holds, touching only `dest` (`set_overlay`).
    overlay: bool,
}

/// An rgba32float 3D texture holding `data` (`size` w x h x d, red
/// fastest).
fn lut_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    size: [u32; 3],
    data: &[[f32; 3]],
) -> wgpu::TextureView {
    let extent = wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: size[2] };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("blit-lut3d"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let rgba: Vec<[f32; 4]> = data.iter().map(|c| [c[0], c[1], c[2], 1.0]).collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&rgba),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size[0] * 16),
            rows_per_image: Some(size[1]),
        },
        extent,
    );
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

impl Blit {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let view = ViewSettings::default();
        let pipeline = build_pipeline(device, target_format, &view);
        // The placeholder never needs data: nothing samples it.
        let lut_view = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("blit-lut3d-none"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blit-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blit-uniform"),
            size: std::mem::size_of::<BlitUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            target_format,
            pipeline,
            shader_key: view.shader_key(),
            sampler,
            uniform,
            bind_group: None,
            generation: 0,
            backdrop: [0.0, 0.0, 0.0, 1.0],
            lut_view,
            lut_loaded: None,
            lut_want: None,
            channel: 0,
            overlay: false,
        }
    }

    /// The colour around the image, as display-encoded sRGB bytes. It is a UI
    /// colour, not a scene one, so it skips exposure and the view transform;
    /// an sRGB target encodes in hardware and is handed the linear value.
    /// Show one colour channel as grey (`Some(0..=2)` = r/g/b), or all.
    /// Rides the uniform: no pipeline rebuild.
    pub fn set_channel(&mut self, channel: Option<usize>) {
        self.channel = channel.map_or(0, |c| c.min(2) as u8 + 1);
    }

    /// Overlay mode: load the target instead of clearing it and leave every
    /// pixel outside the destination rect untouched — for a SECOND image in
    /// the same target (dual view's other pane). Needs its own `Blit`: one
    /// instance's uniform and bind group serve one draw per submit.
    pub fn set_overlay(&mut self, on: bool) {
        self.overlay = on;
    }

    pub fn set_backdrop(&mut self, rgb: [u8; 3]) {
        let srgb = self.target_format.is_srgb();
        let ch = |b: u8| {
            let x = b as f32 / 255.0;
            if !srgb {
                x
            } else if x <= 0.04045 {
                x / 12.92
            } else {
                ((x + 0.055) / 1.055).powf(2.4)
            }
        };
        self.backdrop = [ch(rgb[0]), ch(rgb[1]), ch(rgb[2]), 1.0];
    }

    /// Call when the source texture was recreated so the bind group refreshes.
    pub fn invalidate(&mut self) {
        self.generation += 1;
    }

    /// Rebuild the pipeline if `view` selects a different shader. Parameter
    /// tweaks (exposure, every OpenDRT knob) ride the uniform and land here as
    /// a no-op; only the transform, display gamut or EOTF force a compile.
    /// Returns true when a rebuild happened.
    pub fn set_view(&mut self, device: &wgpu::Device, view: &ViewSettings) -> bool {
        self.lut_want = view.lut_display();
        let key = view.shader_key();
        if key == self.shader_key {
            return false;
        }
        let t0 = Instant::now();
        self.pipeline = build_pipeline(device, self.target_format, view);
        self.shader_key = key;
        // The pipeline is built with `layout: None`, so every pipeline owns an
        // exclusive auto-derived bind group layout — a bind group made against
        // the old pipeline is rejected by the new one. Force a rebuild.
        self.invalidate();
        log::info!(
            "blit view transform -> {} ({:.1}ms)",
            view.effective().label(),
            t0.elapsed().as_secs_f32() * 1e3
        );
        true
    }

    /// `dest_rect`: [x, y, w, h] where the image lands, in target pixels.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::TextureView,
        dst: &wgpu::TextureView,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites>,
        dest_rect: [f32; 4],
        show_alpha: bool,
        view: &ViewSettings,
    ) {
        let mut u = BlitUniform {
            dest: dest_rect,
            mode: [
                if show_alpha { 1.0 } else { 0.0 },
                view.exposure.exp2(),
                view.inv_gamma(),
                self.channel as f32,
            ],
            backdrop: if self.overlay { [0.0, 0.0, 0.0, -1.0] } else { self.backdrop },
            ..Default::default()
        };
        view.fill_uniform(&mut u);
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
        // Upload the cubes the transform wants (once per display; the parsed
        // cubes are cached process-wide by `agx::lut` / `agx::lumcomp`).
        if let Some(d) = self.lut_want.filter(|d| Some(*d) != self.lut_loaded) {
            if let Some(l) = crate::viewtf::agx::lut_stack(d) {
                self.lut_view = lut_texture(device, queue, l.extent, &l.data);
                self.lut_loaded = Some(d);
                self.invalidate();
            }
        }
        let stale = match &self.bind_group {
            Some((_, gen)) => *gen != self.generation,
            None => true,
        };
        if stale {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blit-bg"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.lut_view),
                    },
                ],
            });
            self.bind_group = Some((bg, self.generation));
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if self.overlay {
                        wgpu::LoadOp::Load
                    } else {
                        wgpu::LoadOp::Clear(wgpu::Color::BLACK)
                    },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group.as_ref().unwrap().0, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Assemble the blit shader for one view transform. Kept free-standing so the
/// headless shader-validation test can call it without a surface.
pub fn wgsl_for(view: &ViewSettings, target_format: wgpu::TextureFormat) -> String {
    let body = BLIT_WGSL
        .replace("TRANSFORM", &view.wgsl())
        .replace("ENCODE", view.encode_expr(target_format.is_srgb()));
    format!("{}{}", crate::viewtf::uniform_wgsl(), body)
}

fn build_pipeline(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
    view: &ViewSettings,
) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("blit"),
        source: wgpu::ShaderSource::Wgsl(wgsl_for(view, target_format).into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("blit"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(target_format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viewtf::{Eotf, Gamut, ViewTransform};

    /// Every transform x display x EOTF combination must produce WGSL naga
    /// accepts — this is the cheap headless guard against a codegen typo.
    #[test]
    fn every_view_transform_compiles() {
        for transform in ViewTransform::ALL {
            for display in Gamut::DISPLAYS {
                for eotf in Eotf::ALL {
                    let view = ViewSettings {
                        transform,
                        display,
                        eotf,
                        ..Default::default()
                    };
                    for srgb_target in [false, true] {
                        let fmt = if srgb_target {
                            wgpu::TextureFormat::Bgra8UnormSrgb
                        } else {
                            wgpu::TextureFormat::Bgra8Unorm
                        };
                        let src = wgsl_for(&view, fmt);
                        let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
                            panic!("{transform:?}/{display:?}/{eotf:?} parse: {e}\n{src}")
                        });
                        naga::valid::Validator::new(
                            naga::valid::ValidationFlags::all(),
                            naga::valid::Capabilities::all(),
                        )
                        .validate(&module)
                        .unwrap_or_else(|e| {
                            panic!("{transform:?}/{display:?}/{eotf:?} validate: {e:?}")
                        });
                    }
                }
            }
        }
    }

    #[test]
    fn srgb_target_defers_to_hardware() {
        let view = ViewSettings {
            eotf: Eotf::Srgb,
            ..Default::default()
        };
        assert!(!wgsl_for(&view, wgpu::TextureFormat::Bgra8UnormSrgb).contains("vt_srgb_encode(c)"));
        assert!(wgsl_for(&view, wgpu::TextureFormat::Bgra8Unorm).contains("vt_srgb_encode(c)"));
    }

    /// Switching transform rebuilds the pipeline, and the auto-derived bind
    /// group layout is exclusive to it — so the cached bind group has to be
    /// rebuilt too, or the next draw fails validation. Skipped when no
    /// adapter is available.
    #[test]
    fn switching_transform_rebuilds_the_bind_group() {
        let Ok(ctx) = crate::GpuContext::new() else {
            eprintln!("no adapter; skipping");
            return;
        };
        const FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
        let (w, h) = (64u32, 64u32);
        let make = |usage, format| {
            ctx.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let src = make(wgpu::TextureUsages::TEXTURE_BINDING, wgpu::TextureFormat::Rgba16Float);
        let dst = make(wgpu::TextureUsages::RENDER_ATTACHMENT, FMT);

        let mut blit = Blit::new(&ctx.device, FMT);
        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        for transform in [ViewTransform::Filmic, ViewTransform::None, ViewTransform::Filmic] {
            let view = ViewSettings { transform, ..Default::default() };
            blit.set_view(&ctx.device, &view);
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            blit.encode(
                &ctx.device,
                &ctx.queue,
                &mut enc,
                &src,
                &dst,
                None,
                [0.0, 0.0, w as f32, h as f32],
                false,
                &view,
            );
            ctx.queue.submit([enc.finish()]);
        }
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "validation error after switching: {err:?}");
    }

    #[test]
    fn parameter_edits_do_not_change_the_shader() {
        let a = ViewSettings::default();
        let mut b = a;
        b.exposure = 2.0;
        assert_eq!(a.shader_key(), b.shader_key(), "exposure must ride uniform");
    }
}
