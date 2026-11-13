//! WGSL execution of the IFNet forward pass.
//!
//! The whole inference is planned once per (model, resolution): every
//! dispatch's uniform buffer and bind group is prebuilt, so `run` is
//! upload → one command encoder replaying ~60 dispatches → readback.
//! Kernels live in `shaders.wgsl`, one module per `//---kernel:` marker.
//! Compute-pass dispatches are implicitly ordered on their storage buffers
//! (per-dispatch usage scopes), so the chain needs no manual barriers.
//!
//! This is a *bake* engine: readback at the end is deliberate and the
//! blocking map is fine. Nothing here runs per interactive frame.

use super::*;
use std::sync::Arc;

const SHADERS: &str = include_str!("shaders.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackU {
    w: u32,
    h: u32,
    mode: u32,
    _p: u32,
    t: f32,
    _p1: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ResizeU {
    sw: u32,
    sh: u32,
    dw: u32,
    dh: u32,
    sc0: u32,
    dc0: u32,
    nch: u32,
    _p: u32,
    post_mul: f32,
    _p1: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvU {
    in_c: u32,
    out_c: u32,
    ih: u32,
    iw: u32,
    oh: u32,
    ow: u32,
    stride: u32,
    _p: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Chw {
    c: u32,
    h: u32,
    w: u32,
    _p: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WarpU {
    w: u32,
    h: u32,
    flow_c0: u32,
    clamp01: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AddU {
    w: u32,
    h: u32,
    nch: u32,
    _p: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MergeU {
    w: u32,
    h: u32,
    _p: [u32; 2],
}

struct Pass {
    /// Pass name for the parity-bisection tests (`run_prefix`).
    #[cfg_attr(not(test), allow(dead_code))]
    label: String,
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    groups: [u32; 3],
}

struct GpuConv {
    w: wgpu::Buffer,
    b: wgpu::Buffer,
    slope: wgpu::Buffer,
}

struct GpuResConv {
    w: wgpu::Buffer,
    b: wgpu::Buffer,
    beta: wgpu::Buffer,
    slope: wgpu::Buffer,
}

/// A planned IFNet inference at one resolution. Rebuild on size change.
pub struct RifeGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    w: usize,
    h: usize,
    pw: usize,
    ph: usize,
    passes: Vec<Pass>,
    /// (buffer, template) for the four pack dispatches — `t` is poked per run.
    pack_uniforms: Vec<(wgpu::Buffer, PackU)>,
    img0: wgpu::Buffer,
    img1: wgpu::Buffer,
    merged: wgpu::Buffer,
    merged_hdr: wgpu::Buffer,
    flow: wgpu::Buffer,
    mask: wgpu::Buffer,
    staging: wgpu::Buffer,
    /// Named intermediates for the parity-bisection tests.
    #[allow(dead_code)]
    dbg: Vec<(String, wgpu::Buffer)>,
}

fn groups2(w: usize, h: usize, z: usize) -> [u32; 3] {
    [(w as u32).div_ceil(8), (h as u32).div_ceil(8), z as u32]
}

impl RifeGpu {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        model: &RifeModel,
        w: usize,
        h: usize,
    ) -> Result<Self> {
        let pw = pad32(w);
        let ph = pad32(h);
        let full = pw * ph;

        // --- pipelines, one module per kernel chunk -----------------------
        let mut pipelines: HashMap<String, wgpu::ComputePipeline> = HashMap::new();
        for chunk in SHADERS.split("//---kernel:").skip(1) {
            let (name, body) = chunk.split_once('\n').context("bad shader chunk")?;
            let name = name.trim().to_string();
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("rife-{name}")),
                source: wgpu::ShaderSource::Wgsl(body.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("rife-{name}")),
                layout: None,
                module: &module,
                entry_point: Some(&name),
                compilation_options: Default::default(),
                cache: None,
            });
            pipelines.insert(name, pipeline);
        }
        let pipe = |n: &str| pipelines.get(n).cloned().unwrap_or_else(|| panic!("missing kernel {n}"));

        // --- buffers ------------------------------------------------------
        let storage = |label: &str, n_f32: usize, _readback: bool| {
            // COPY_SRC unconditionally: the parity-bisection tests read
            // intermediates back, and it costs nothing.
            let usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC;
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (n_f32 * 4) as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        let weights = |label: &str, data: &[f32]| {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (data.len() * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: true,
            });
            buf.slice(..)
                .get_mapped_range_mut().expect("a mapped buffer")
                .copy_from_slice(bytemuck::cast_slice(data));
            buf.unmap();
            buf
        };

        let mut max_h1 = 0usize;
        let mut max_feat = 0usize;
        for (i, &s) in SCALES.iter().enumerate() {
            let (rw, rh) = (pw / s, ph / s);
            max_h1 = max_h1.max((BLOCK_C[i] / 2) * (rw / 2) * (rh / 2));
            max_feat = max_feat.max(BLOCK_C[i] * (rw / 4) * (rh / 4));
        }

        let img0 = storage("rife-img0", 3 * full, false);
        let img1 = storage("rife-img1", 3 * full, false);
        let xfull = storage("rife-xfull", 8 * full, false);
        let xs = storage("rife-xs", 12 * full, false);
        let h1 = storage("rife-h1", max_h1, false);
        let feat_a = storage("rife-feat-a", max_feat, false);
        let feat_b = storage("rife-feat-b", max_feat, false);
        let tmp6 = storage("rife-tmp6", HEAD_C * full, false);
        let flow = storage("rife-flow", 4 * full, true);
        let mask = storage("rife-mask", full, true);
        let fdelta = storage("rife-fdelta", 4 * full, false);
        let mdelta = storage("rife-mdelta", full, false);
        let warped0 = storage("rife-warped0", 3 * full, false);
        let warped1 = storage("rife-warped1", 3 * full, false);
        let hw0 = storage("rife-hw0", 3 * full, false);
        let hw1 = storage("rife-hw1", 3 * full, false);
        let merged = storage("rife-merged", 3 * full, true);
        let merged_hdr = storage("rife-merged-hdr", 3 * full, true);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rife-staging"),
            size: (11 * full * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- weight buffers -----------------------------------------------
        let gpu_blocks: Vec<_> = model
            .blocks
            .iter()
            .enumerate()
            .map(|(i, blk)| {
                let conv0: Vec<GpuConv> = blk
                    .conv0
                    .iter()
                    .enumerate()
                    .map(|(k, l)| GpuConv {
                        w: weights(&format!("rife-b{i}c{k}w"), &l.w.data),
                        b: weights(&format!("rife-b{i}c{k}b"), &l.b.data),
                        slope: weights(&format!("rife-b{i}c{k}s"), &l.slope),
                    })
                    .collect();
                let convblock: Vec<GpuResConv> = blk
                    .convblock
                    .iter()
                    .enumerate()
                    .map(|(j, l)| GpuResConv {
                        w: weights(&format!("rife-b{i}r{j}w"), &l.w.data),
                        b: weights(&format!("rife-b{i}r{j}b"), &l.b.data),
                        beta: weights(&format!("rife-b{i}r{j}beta"), &l.beta),
                        slope: weights(&format!("rife-b{i}r{j}s"), &l.slope),
                    })
                    .collect();
                let last_w = weights(&format!("rife-b{i}lw"), &blk.lastconv.w.data);
                let last_b = weights(&format!("rife-b{i}lb"), &blk.lastconv.b.data);
                (conv0, convblock, last_w, last_b)
            })
            .collect();

        // --- plan ---------------------------------------------------------
        let uniform = |label: &str, data: &[u8]| {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: data.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            buf.slice(..).get_mapped_range_mut().expect("a mapped buffer").copy_from_slice(data);
            buf.unmap();
            buf
        };
        let mut passes: Vec<Pass> = Vec::new();
        let mut pack_uniforms: Vec<(wgpu::Buffer, PackU)> = Vec::new();
        let mut push = |label: &str,
                        pipeline: wgpu::ComputePipeline,
                        u: &wgpu::Buffer,
                        bufs: &[&wgpu::Buffer],
                        groups: [u32; 3]| {
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: u.as_entire_binding(),
            }];
            for (k, b) in bufs.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry {
                    binding: (k + 1) as u32,
                    resource: b.as_entire_binding(),
                });
            }
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rife-pass"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
            passes.push(Pass { label: label.to_string(), pipeline, bind, groups });
        };

        for (i, &s) in SCALES.iter().enumerate() {
            let blk = &model.blocks[i];
            let (rw, rh) = (pw / s, ph / s);
            let (conv0, convblock, last_w, last_b) = &gpu_blocks[i];

            // pack full-res input stack
            let pu = PackU {
                w: pw as u32,
                h: ph as u32,
                mode: if i == 0 { 0 } else { 1 },
                _p: 0,
                t: 0.5,
                _p1: [0.0; 3],
            };
            let pu_buf = uniform(&format!("rife-u-pack{i}"), bytemuck::bytes_of(&pu));
            let (pa, pb) = if i == 0 { (&img0, &img1) } else { (&warped0, &warped1) };
            push(&format!("pack{i}"), pipe("pack"), &pu_buf, &[pa, pb, &mask, &xfull], groups2(pw, ph, 1));
            pack_uniforms.push((pu_buf, pu));

            // downsample stack (+ flow, scaled) into xs
            let base = if i == 0 { 7 } else { 8 };
            let ru = ResizeU {
                sw: pw as u32,
                sh: ph as u32,
                dw: rw as u32,
                dh: rh as u32,
                sc0: 0,
                dc0: 0,
                nch: base as u32,
                _p: 0,
                post_mul: 1.0,
                _p1: [0.0; 3],
            };
            push(
                &format!("down{i}"),
                pipe("resize"),
                &uniform(&format!("rife-u-down{i}"), bytemuck::bytes_of(&ru)),
                &[&xfull, &xs],
                groups2(rw, rh, base),
            );
            if i > 0 {
                let fu = ResizeU {
                    nch: 4,
                    dc0: 8,
                    post_mul: 1.0 / s as f32,
                    ..ru
                };
                push(
                    &format!("downf{i}"),
                    pipe("resize"),
                    &uniform(&format!("rife-u-downf{i}"), bytemuck::bytes_of(&fu)),
                    &[&flow, &xs],
                    groups2(rw, rh, 4),
                );
            }

            // conv0: two stride-2 convs
            let c = blk.c;
            let (h1w, h1h) = (rw / 2, rh / 2);
            let cu = ConvU {
                in_c: blk.in_planes as u32,
                out_c: (c / 2) as u32,
                ih: rh as u32,
                iw: rw as u32,
                oh: h1h as u32,
                ow: h1w as u32,
                stride: 2,
                _p: 0,
            };
            push(
                &format!("c0a{i}"),
                pipe("conv"),
                &uniform(&format!("rife-u-c0a{i}"), bytemuck::bytes_of(&cu)),
                &[&xs, &conv0[0].w, &conv0[0].b, &conv0[0].slope, &h1],
                groups2(h1w, h1h, c / 2),
            );
            let (fw, fh) = (rw / 4, rh / 4);
            let cu2 = ConvU {
                in_c: (c / 2) as u32,
                out_c: c as u32,
                ih: h1h as u32,
                iw: h1w as u32,
                oh: fh as u32,
                ow: fw as u32,
                stride: 2,
                _p: 0,
            };
            push(
                &format!("c0b{i}"),
                pipe("conv"),
                &uniform(&format!("rife-u-c0b{i}"), bytemuck::bytes_of(&cu2)),
                &[&h1, &conv0[1].w, &conv0[1].b, &conv0[1].slope, &feat_a],
                groups2(fw, fh, c),
            );

            // eight residual blocks, ping-pong a<->b (even count ends in a)
            let rcu = Chw { c: c as u32, h: fh as u32, w: fw as u32, _p: 0 };
            for (j, rc) in convblock.iter().enumerate() {
                let (src, dst) = if j % 2 == 0 { (&feat_a, &feat_b) } else { (&feat_b, &feat_a) };
                push(
                    &format!("rc{i}-{j}"),
                    pipe("resconv"),
                    &uniform(&format!("rife-u-rc{i}-{j}"), bytemuck::bytes_of(&rcu)),
                    &[src, &rc.w, &rc.b, &rc.beta, &rc.slope, dst],
                    groups2(fw, fh, c),
                );
            }

            // head: deconv + pixelshuffle -> [6, rh, rw]
            push(
                &format!("dc{i}"),
                pipe("deconv_ps"),
                &uniform(&format!("rife-u-dc{i}"), bytemuck::bytes_of(&rcu)),
                &[&feat_a, last_w, last_b, &tmp6],
                groups2(rw, rh, HEAD_C),
            );

            // upsample head output to full res; flow channels gain *scale
            let (flow_dst, mask_dst) = if i == 0 { (&flow, &mask) } else { (&fdelta, &mdelta) };
            let upf = ResizeU {
                sw: rw as u32,
                sh: rh as u32,
                dw: pw as u32,
                dh: ph as u32,
                sc0: 0,
                dc0: 0,
                nch: 4,
                _p: 0,
                post_mul: s as f32,
                _p1: [0.0; 3],
            };
            push(
                &format!("upf{i}"),
                pipe("resize"),
                &uniform(&format!("rife-u-upf{i}"), bytemuck::bytes_of(&upf)),
                &[&tmp6, flow_dst],
                groups2(pw, ph, 4),
            );
            let upm = ResizeU { sc0: 4, nch: 1, post_mul: 1.0, ..upf };
            push(
                &format!("upm{i}"),
                pipe("resize"),
                &uniform(&format!("rife-u-upm{i}"), bytemuck::bytes_of(&upm)),
                &[&tmp6, mask_dst],
                groups2(pw, ph, 1),
            );
            if i > 0 {
                let af = AddU { w: pw as u32, h: ph as u32, nch: 4, _p: 0 };
                push(
                    &format!("addf{i}"),
                    pipe("add"),
                    &uniform(&format!("rife-u-addf{i}"), bytemuck::bytes_of(&af)),
                    &[&fdelta, &flow],
                    groups2(pw, ph, 4),
                );
                let am = AddU { nch: 1, ..af };
                push(
                    &format!("addm{i}"),
                    pipe("add"),
                    &uniform(&format!("rife-u-addm{i}"), bytemuck::bytes_of(&am)),
                    &[&mdelta, &mask],
                    groups2(pw, ph, 1),
                );
            }

            // warp both sources by the accumulated flow (net-clamped variant)
            for (img, c0, dst) in [(&img0, 0u32, &warped0), (&img1, 2u32, &warped1)] {
                let wu = WarpU { w: pw as u32, h: ph as u32, flow_c0: c0, clamp01: 1 };
                push(
                    &format!("warp{i}-{c0}"),
                    pipe("warp"),
                    &uniform(&format!("rife-u-w{i}-{c0}"), bytemuck::bytes_of(&wu)),
                    &[img, &flow, dst],
                    groups2(pw, ph, 1),
                );
            }
        }

        // final fusion + the HDR-honest variant on unclamped sources
        let mu = MergeU { w: pw as u32, h: ph as u32, _p: [0; 2] };
        push(
            "merge",
            pipe("merge"),
            &uniform("rife-u-merge", bytemuck::bytes_of(&mu)),
            &[&warped0, &warped1, &mask, &merged],
            groups2(pw, ph, 1),
        );
        for (img, c0, dst) in [(&img0, 0u32, &hw0), (&img1, 2u32, &hw1)] {
            let wu = WarpU { w: pw as u32, h: ph as u32, flow_c0: c0, clamp01: 0 };
            push(
                &format!("whdr{c0}"),
                pipe("warp"),
                &uniform(&format!("rife-u-whdr{c0}"), bytemuck::bytes_of(&wu)),
                &[img, &flow, dst],
                groups2(pw, ph, 1),
            );
        }
        push(
            "merge-hdr",
            pipe("merge"),
            &uniform("rife-u-merge-hdr", bytemuck::bytes_of(&mu)),
            &[&hw0, &hw1, &mask, &merged_hdr],
            groups2(pw, ph, 1),
        );

        let dbg = [
            ("xfull", &xfull),
            ("xs", &xs),
            ("h1", &h1),
            ("feat_a", &feat_a),
            ("feat_b", &feat_b),
            ("tmp6", &tmp6),
            ("fdelta", &fdelta),
            ("mdelta", &mdelta),
            ("warped0", &warped0),
            ("warped1", &warped1),
        ]
        .into_iter()
        .map(|(n, b)| (n.to_string(), (*b).clone()))
        .collect();

        Ok(Self {
            device,
            queue,
            w,
            h,
            pw,
            ph,
            passes,
            pack_uniforms,
            img0,
            img1,
            merged,
            merged_hdr,
            flow,
            mask,
            staging,
            dbg,
        })
    }

    /// Run one inference. `img0`/`img1` are `[3][h][w]` planes at the
    /// constructor's size; `t` in 0..1 is the interpolation position.
    pub fn run(&self, img0: &[f32], img1: &[f32], t: f32) -> Result<RifeOutput> {
        let (w, h, pw, ph) = (self.w, self.h, self.pw, self.ph);
        let full = pw * ph;
        assert_eq!(img0.len(), 3 * w * h);
        assert_eq!(img1.len(), 3 * w * h);

        let p0 = pad_planes(img0, 3, w, h, pw, ph);
        let p1 = pad_planes(img1, 3, w, h, pw, ph);
        self.queue.write_buffer(&self.img0, 0, bytemuck::cast_slice(&p0));
        self.queue.write_buffer(&self.img1, 0, bytemuck::cast_slice(&p1));
        for (buf, template) in &self.pack_uniforms {
            let u = PackU { t, ..*template };
            self.queue.write_buffer(buf, 0, bytemuck::bytes_of(&u));
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rife") });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rife"),
                timestamp_writes: None,
            });
            for pass in &self.passes {
                cpass.set_pipeline(&pass.pipeline);
                cpass.set_bind_group(0, &pass.bind, &[]);
                cpass.dispatch_workgroups(pass.groups[0], pass.groups[1], pass.groups[2]);
            }
        }
        let fb = (full * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.merged, 0, &self.staging, 0, 3 * fb);
        encoder.copy_buffer_to_buffer(&self.merged_hdr, 0, &self.staging, 3 * fb, 3 * fb);
        encoder.copy_buffer_to_buffer(&self.flow, 0, &self.staging, 6 * fb, 4 * fb);
        encoder.copy_buffer_to_buffer(&self.mask, 0, &self.staging, 10 * fb, fb);
        self.queue.submit([encoder.finish()]);

        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().context("map_async dropped")?.context("staging map failed")?;
        let data = slice.get_mapped_range().expect("a mapped buffer");
        let all: &[f32] = bytemuck::cast_slice(&data);
        let merged = crop_planes(&all[..3 * full], 3, pw, ph, w, h);
        let merged_hdr = crop_planes(&all[3 * full..6 * full], 3, pw, ph, w, h);
        let flow = crop_planes(&all[6 * full..10 * full], 4, pw, ph, w, h);
        let mut mask = crop_planes(&all[10 * full..11 * full], 1, pw, ph, w, h);
        drop(data);
        self.staging.unmap();
        for m in &mut mask {
            *m = sigmoid(*m);
        }
        Ok(RifeOutput { merged, merged_hdr, flow, mask })
    }
}

#[cfg(test)]
impl RifeGpu {
    /// Submit only the passes up to and including `upto` (a pass label).
    fn run_prefix(&self, img0: &[f32], img1: &[f32], t: f32, upto: &str) {
        let (w, h, pw, ph) = (self.w, self.h, self.pw, self.ph);
        let p0 = pad_planes(img0, 3, w, h, pw, ph);
        let p1 = pad_planes(img1, 3, w, h, pw, ph);
        self.queue.write_buffer(&self.img0, 0, bytemuck::cast_slice(&p0));
        self.queue.write_buffer(&self.img1, 0, bytemuck::cast_slice(&p1));
        for (buf, template) in &self.pack_uniforms {
            let u = PackU { t, ..*template };
            self.queue.write_buffer(buf, 0, bytemuck::bytes_of(&u));
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            for pass in &self.passes {
                cpass.set_pipeline(&pass.pipeline);
                cpass.set_bind_group(0, &pass.bind, &[]);
                cpass.dispatch_workgroups(pass.groups[0], pass.groups[1], pass.groups[2]);
                if pass.label == upto {
                    break;
                }
            }
        }
        self.queue.submit([encoder.finish()]);
    }

    fn read_named(&self, name: &str, n: usize) -> Vec<f32> {
        let buf = match name {
            "flow" => &self.flow,
            "mask" => &self.mask,
            other => &self.dbg.iter().find(|(k, _)| k == other).unwrap().1,
        };
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, (n * 4) as u64);
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let out = bytemuck::cast_slice(&slice.get_mapped_range().expect("a mapped buffer")).to_vec();
        staging.unmap();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::super::reference;
    use super::*;

    fn ctx() -> Option<crate::GpuContext> {
        match crate::GpuContext::new() {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("SKIP rife gpu test: no adapter ({e})");
                None
            }
        }
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    /// Run one kernel once: uploads every buffer, dispatches, reads all back.
    fn exec_kernel(
        ctx: &crate::GpuContext,
        name: &str,
        ubytes: &[u8],
        bufs: &mut [Vec<f32>],
        groups: [u32; 3],
    ) {
        let device = &ctx.device;
        let chunk = SHADERS
            .split("//---kernel:")
            .skip(1)
            .find(|c| c.split_once('\n').unwrap().0.trim() == name)
            .unwrap();
        let body = chunk.split_once('\n').unwrap().1;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(body.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: None,
            module: &module,
            entry_point: Some(name),
            compilation_options: Default::default(),
            cache: None,
        });
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: ubytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        ubuf.slice(..).get_mapped_range_mut().expect("a mapped buffer").copy_from_slice(ubytes);
        ubuf.unmap();
        let gbufs: Vec<wgpu::Buffer> = bufs
            .iter()
            .map(|b| {
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (b.len() * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: true,
                });
                buf.slice(..)
                    .get_mapped_range_mut().expect("a mapped buffer")
                    .copy_from_slice(bytemuck::cast_slice(b));
                buf.unmap();
                buf
            })
            .collect();
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() }];
        for (k, b) in gbufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (k + 1) as u32,
                resource: b.as_entire_binding(),
            });
        }
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let staging: Vec<wgpu::Buffer> = bufs
            .iter()
            .map(|b| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (b.len() * 4) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&pipeline);
            cp.set_bind_group(0, &bind, &[]);
            cp.dispatch_workgroups(groups[0], groups[1], groups[2]);
        }
        for (g, s) in gbufs.iter().zip(&staging) {
            enc.copy_buffer_to_buffer(g, 0, s, 0, g.size());
        }
        ctx.queue.submit([enc.finish()]);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        for (b, s) in bufs.iter_mut().zip(&staging) {
            let slice = s.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            b.copy_from_slice(bytemuck::cast_slice(&slice.get_mapped_range().expect("a mapped buffer")));
            s.unmap();
        }
    }

    /// Headless naga guard, so CI (no GPU) still catches a kernel typo —
    /// same pattern as blit's view-transform test.
    #[test]
    fn every_kernel_validates() {
        let mut n = 0;
        for chunk in SHADERS.split("//---kernel:").skip(1) {
            let (name, body) = chunk.split_once('\n').unwrap();
            let module = naga::front::wgsl::parse_str(body)
                .unwrap_or_else(|e| panic!("{name} parse: {e}"));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name} validate: {e:?}"));
            n += 1;
        }
        assert_eq!(n, 8, "kernel count drifted from the plan builder's expectations");
    }

    #[test]
    fn kernel_resize_parity() {
        let Some(ctx) = ctx() else { return };
        let mut rng = Lcg(1);
        let (sw, sh, dw, dh) = (16usize, 8usize, 6usize, 10usize);
        let src: Vec<f32> = (0..3 * sw * sh).map(|_| rng.signed(2.0)).collect();
        let mut dst_ref = vec![0f32; 3 * dw * dh];
        reference::resize_into(&src, sw, sh, 1, &mut dst_ref, dw, dh, 0, 2, 1.5);
        let u = ResizeU {
            sw: sw as u32, sh: sh as u32, dw: dw as u32, dh: dh as u32,
            sc0: 1, dc0: 0, nch: 2, _p: 0, post_mul: 1.5, _p1: [0.0; 3],
        };
        let mut bufs = vec![src, vec![0f32; 3 * dw * dh]];
        exec_kernel(&ctx, "resize", bytemuck::bytes_of(&u), &mut bufs, groups2(dw, dh, 2));
        let d = max_abs_diff(&bufs[1][..2 * dw * dh], &dst_ref[..2 * dw * dh]);
        assert!(d < 1e-5, "resize diverges: {d}");
    }

    #[test]
    fn kernel_conv_parity() {
        let Some(ctx) = ctx() else { return };
        let mut rng = Lcg(2);
        let (in_c, out_c, ih, iw) = (5usize, 7usize, 12usize, 10usize);
        let src: Vec<f32> = (0..in_c * ih * iw).map(|_| rng.signed(1.0)).collect();
        let layer = ConvLayer {
            w: rng.tensor(vec![out_c, in_c, 3, 3], 0.3),
            b: rng.tensor(vec![out_c], 0.2),
            slope: (0..out_c).map(|_| rng.uniform()).collect(),
        };
        for stride in [1usize, 2] {
            let (dref, oh, ow) = reference::conv3x3(&src, in_c, ih, iw, &layer, stride);
            let u = ConvU {
                in_c: in_c as u32, out_c: out_c as u32, ih: ih as u32, iw: iw as u32,
                oh: oh as u32, ow: ow as u32, stride: stride as u32, _p: 0,
            };
            let mut bufs = vec![
                src.clone(), layer.w.data.clone(), layer.b.data.clone(),
                layer.slope.clone(), vec![0f32; out_c * oh * ow],
            ];
            exec_kernel(&ctx, "conv", bytemuck::bytes_of(&u), &mut bufs, groups2(ow, oh, out_c));
            let d = max_abs_diff(&bufs[4], &dref);
            assert!(d < 1e-4, "conv s{stride} diverges: {d}");
        }
    }

    #[test]
    fn kernel_resconv_parity() {
        let Some(ctx) = ctx() else { return };
        let mut rng = Lcg(3);
        let (c, h, w) = (6usize, 9usize, 7usize);
        let src: Vec<f32> = (0..c * h * w).map(|_| rng.signed(1.0)).collect();
        let layer = ResConvLayer {
            w: rng.tensor(vec![c, c, 3, 3], 0.3),
            b: rng.tensor(vec![c], 0.2),
            beta: (0..c).map(|_| 0.5 + rng.uniform()).collect(),
            slope: (0..c).map(|_| rng.uniform()).collect(),
        };
        let dref = reference::resconv(&src, c, h, w, &layer);
        let u = Chw { c: c as u32, h: h as u32, w: w as u32, _p: 0 };
        let mut bufs = vec![
            src, layer.w.data.clone(), layer.b.data.clone(),
            layer.beta.clone(), layer.slope.clone(), vec![0f32; c * h * w],
        ];
        exec_kernel(&ctx, "resconv", bytemuck::bytes_of(&u), &mut bufs, groups2(w, h, c));
        let d = max_abs_diff(&bufs[5], &dref);
        assert!(d < 1e-4, "resconv diverges: {d}");
    }

    #[test]
    fn kernel_deconv_ps_parity() {
        let Some(ctx) = ctx() else { return };
        let mut rng = Lcg(4);
        let (c, h, w) = (5usize, 6usize, 4usize);
        let src: Vec<f32> = (0..c * h * w).map(|_| rng.signed(1.0)).collect();
        let layer = DeconvLayer {
            w: rng.tensor(vec![c, 4 * HEAD_C, 4, 4], 0.3),
            b: rng.tensor(vec![4 * HEAD_C], 0.2),
        };
        let dref = reference::deconv_ps(&src, c, h, w, &layer);
        let u = Chw { c: c as u32, h: h as u32, w: w as u32, _p: 0 };
        let mut bufs = vec![src, layer.w.data.clone(), layer.b.data.clone(), vec![0f32; HEAD_C * 16 * h * w]];
        exec_kernel(&ctx, "deconv_ps", bytemuck::bytes_of(&u), &mut bufs, groups2(4 * w, 4 * h, HEAD_C));
        let d = max_abs_diff(&bufs[3], &dref);
        assert!(d < 1e-4, "deconv_ps diverges: {d}");
    }

    #[test]
    fn kernel_warp_parity() {
        let Some(ctx) = ctx() else { return };
        let mut rng = Lcg(5);
        let (h, w) = (9usize, 11usize);
        let img: Vec<f32> = (0..3 * h * w).map(|_| rng.signed(2.0)).collect();
        let flow: Vec<f32> = (0..4 * h * w).map(|_| rng.signed(3.0)).collect();
        for (c0, clamp) in [(0u32, 0u32), (2, 1)] {
            let dref = reference::warp(&img, 3, h, w, &flow, c0 as usize, clamp == 1);
            let u = WarpU { w: w as u32, h: h as u32, flow_c0: c0, clamp01: clamp };
            let mut bufs = vec![img.clone(), flow.clone(), vec![0f32; 3 * h * w]];
            exec_kernel(&ctx, "warp", bytemuck::bytes_of(&u), &mut bufs, groups2(w, h, 1));
            let d = max_abs_diff(&bufs[2], &dref);
            assert!(d < 1e-4, "warp c0={c0} clamp={clamp} diverges: {d}");
        }
    }

    #[test]
    fn gpu_matches_reference() {
        let Some(ctx) = ctx() else { return };
        let model = RifeModel::random(42);
        let (w, h) = (64, 32);
        let mut rng = Lcg(9);
        let img0: Vec<f32> = (0..3 * w * h).map(|_| rng.uniform()).collect();
        // img1: shifted-ish copy of img0 plus noise, so flow has something to find
        let img1: Vec<f32> = (0..3 * w * h).map(|_| rng.uniform()).collect();

        let cpu = reference::forward(&model, &img0, &img1, w, h, 0.5);
        let gpu = RifeGpu::new(ctx.device.clone(), ctx.queue.clone(), &model, w, h).unwrap();
        let out = gpu.run(&img0, &img1, 0.5).unwrap();

        let dm = max_abs_diff(&cpu.merged, &out.merged);
        let dh = max_abs_diff(&cpu.merged_hdr, &out.merged_hdr);
        let df = max_abs_diff(&cpu.flow, &out.flow);
        let dk = max_abs_diff(&cpu.mask, &out.mask);
        eprintln!("rife parity: merged {dm:.2e} hdr {dh:.2e} flow {df:.2e} mask {dk:.2e}");
        assert!(dm < 2e-3, "merged diverges: {dm}");
        assert!(dh < 2e-3, "merged_hdr diverges: {dh}");
        assert!(df < 5e-3, "flow diverges: {df}");
        assert!(dk < 2e-3, "mask diverges: {dk}");
    }

    #[test]
    fn gpu_non_multiple_of_32_pads() {
        let Some(ctx) = ctx() else { return };
        let model = RifeModel::random(3);
        let (w, h) = (50, 30);
        let mut rng = Lcg(2);
        let img0: Vec<f32> = (0..3 * w * h).map(|_| rng.uniform()).collect();
        let img1: Vec<f32> = (0..3 * w * h).map(|_| rng.uniform()).collect();
        let gpu = RifeGpu::new(ctx.device.clone(), ctx.queue.clone(), &model, w, h).unwrap();
        let out = gpu.run(&img0, &img1, 0.25).unwrap();
        assert_eq!(out.flow.len(), 4 * w * h);
        assert!(out.flow.iter().all(|v| v.is_finite()));
        let cpu = reference::forward(&model, &img0, &img1, w, h, 0.25);
        assert!(max_abs_diff(&cpu.flow, &out.flow) < 5e-3);
    }

    /// Manual bisection aid: prints the max divergence at every checkpoint.
    #[test]
    #[ignore]
    fn trace_divergence() {
        let Some(ctx) = ctx() else { return };
        let model = RifeModel::random(42);
        let (w, h) = (64usize, 32usize);
        let mut rng = Lcg(9);
        let img0: Vec<f32> = (0..3 * w * h).map(|_| rng.uniform()).collect();
        let img1: Vec<f32> = (0..3 * w * h).map(|_| rng.uniform()).collect();
        let (pw, ph) = (pad32(w), pad32(h));

        let mut refmap: HashMap<String, Vec<f32>> = HashMap::new();
        reference::forward_traced(&model, &img0, &img1, w, h, 0.5, |n, d| {
            refmap.insert(n.to_string(), d.to_vec());
        });
        let gpu = RifeGpu::new(ctx.device.clone(), ctx.queue.clone(), &model, w, h).unwrap();

        for i in 0..4usize {
            let s = SCALES[i];
            let c = BLOCK_C[i];
            let (rw, rh) = (pw / s, ph / s);
            let in_c = BLOCK_IN[i];
            let xs_label = if i == 0 { "down0".to_string() } else { format!("downf{i}") };
            let fm_label = if i == 0 { "upm0".to_string() } else { format!("addm{i}") };
            let checks: Vec<(String, String, &str, usize)> = vec![
                (format!("xs{i}"), xs_label, "xs", in_c * rw * rh),
                (format!("c0a{i}"), format!("c0a{i}"), "h1", (c / 2) * (rh / 2) * (rw / 2)),
                (format!("c0b{i}"), format!("c0b{i}"), "feat_a", c * (rh / 4) * (rw / 4)),
                (format!("rc{i}"), format!("rc{i}-7"), "feat_a", c * (rh / 4) * (rw / 4)),
                (format!("dc{i}"), format!("dc{i}"), "tmp6", HEAD_C * rh * rw),
                (format!("flow{i}"), fm_label.clone(), "flow", 4 * pw * ph),
                (format!("mask{i}"), fm_label, "mask", pw * ph),
                (format!("warp{i}"), format!("warp{i}-0"), "warped0", 3 * pw * ph),
            ];
            for (ref_key, gpu_label, buf, n) in checks {
                gpu.run_prefix(&img0, &img1, 0.5, &gpu_label);
                let got = gpu.read_named(buf, n);
                let want = &refmap[&ref_key];
                assert_eq!(want.len(), n, "{ref_key} length");
                let d = max_abs_diff(&got, want);
                eprintln!("block{i} {ref_key:>8} (after {gpu_label:>8}): {d:.3e}");
            }
        }
    }

    /// End-to-end against real PyTorch outputs, when the user has run
    /// `tools/rife_convert.py --golden`. Self-skips otherwise, so CI and
    /// fresh clones stay green.
    #[test]
    fn golden_bundle_matches_torch() {
        let weights = std::path::Path::new("../data/rife/rife46.wifw");
        let golden = std::path::Path::new("../data/rife/golden.wifw");
        if !weights.exists() || !golden.exists() {
            eprintln!("SKIP rife golden test: run tools/rife_convert.py --golden first");
            return;
        }
        let model = RifeModel::load(weights).unwrap();
        let g = read_wifw(golden).unwrap();
        let img0 = &g["golden.img0"];
        let img1 = &g["golden.img1"];
        let t = g["golden.t"].data[0];
        let (h, w) = (img0.dims[1], img0.dims[2]);

        let cpu = reference::forward(&model, &img0.data, &img1.data, w, h, t);
        let dm = max_abs_diff(&cpu.merged, &g["golden.merged"].data);
        let df = max_abs_diff(&cpu.flow, &g["golden.flow"].data);
        let dk = max_abs_diff(&cpu.mask, &g["golden.mask"].data);
        eprintln!("rife golden (cpu): merged {dm:.2e} flow {df:.2e} mask {dk:.2e}");
        assert!(dm < 5e-3, "merged vs torch: {dm}");
        assert!(df < 5e-2, "flow vs torch: {df}");
        assert!(dk < 5e-3, "mask vs torch: {dk}");

        if let Some(ctx) = ctx() {
            let gpu = RifeGpu::new(ctx.device.clone(), ctx.queue.clone(), &model, w, h).unwrap();
            let out = gpu.run(&img0.data, &img1.data, t).unwrap();
            let dm = max_abs_diff(&out.merged, &g["golden.merged"].data);
            let df = max_abs_diff(&out.flow, &g["golden.flow"].data);
            eprintln!("rife golden (gpu): merged {dm:.2e} flow {df:.2e}");
            assert!(dm < 5e-3, "gpu merged vs torch: {dm}");
            assert!(df < 5e-2, "gpu flow vs torch: {df}");
        }
    }
}
