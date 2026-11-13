//! SEA-RAFT (Wang, Lipson, Deng — ECCV 2024, BSD-3) in WGSL: real
//! two-frame optical flow, the Flow node's quality engine.
//!
//! RIFE is a frame INTERPOLATOR whose flow is whatever makes a good blend;
//! SEA-RAFT is trained and judged on flow itself. On the golden pair (a
//! texture moved (3, −2)) torch reads (3.01, −1.99).
//!
//! The network, as `princeton-vl/SEA-RAFT` builds it for the M models:
//!
//! - two ResNet-34-style trunks to 1/8 resolution (64/128/256 wide, 3/4/6
//!   blocks): `fnet` per frame for features, `cnet` on both frames stacked
//!   for the context, then a 3×3 `init_conv` splitting into the hidden
//!   state and the context;
//! - an initial flow straight off the context (`flow_head`), then
//!   `iters` refinements, each: a correlation lookup (4 levels, radius 4,
//!   324 channels), a motion encoder, two ConvNeXt blocks over
//!   [hidden | context | motion | flow], a new flow delta;
//! - RAFT's convex upsampling to full resolution.
//!
//! BatchNorm is folded into the convolutions on load. The all-pairs
//! correlation volume is NEVER built (at 2K it is 4.8 GB): bilinear
//! sampling of the volume equals the dot product with bilinearly sampled
//! features, so the lookup computes exactly that, on the fly.
//!
//! Weights: `MemorySlices/Tartan-C-T432x960-M` (`model.safetensors`),
//! the checkpoint trained on TartanAir, FlyingChairs and FlyingThings only
//! — none of the KITTI-tuned ones, whose training data is non-commercial.
//! `tools/searaft_golden.py` runs the real torch model for the tests.

use crate::npy::Tensor;
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

const SHADERS: &str = include_str!("searaft/shaders.wgsl");

/// The M architecture's hyperparameters.
const DIM: usize = 128;
const RADIUS: usize = 4;
const LEVELS: usize = 4;
const CORR_C: usize = LEVELS * (2 * RADIUS + 1) * (2 * RADIUS + 1);
pub const ITERS: usize = 4;
/// Blocks per stage: ResNet-34's first three.
const BLOCKS: [usize; 3] = [3, 4, 6];
const WIDTHS: [usize; 3] = [64, 128, 256];

/// A convolution with BatchNorm folded in: `[oc][ic][k][k]` and a bias.
#[derive(Clone)]
pub struct Conv {
    pub oc: usize,
    pub ic: usize,
    pub k: usize,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

/// One ResNet basic block.
pub struct Block {
    pub conv1: Conv,
    pub conv2: Conv,
    pub down: Option<Conv>,
    pub stride: usize,
}

pub struct Trunk {
    pub stem: Conv,
    pub blocks: Vec<Block>,
    pub out: Conv,
}

pub struct ConvNext {
    pub dw: Conv,
    pub ln_w: Vec<f32>,
    pub ln_b: Vec<f32>,
    pub pw1: Conv,
    pub pw2: Conv,
    pub gamma: Vec<f32>,
    pub fin: Conv,
}

pub struct Model {
    pub fnet: Trunk,
    pub cnet: Trunk,
    pub init_conv: Conv,
    pub head1: Conv,
    pub head2: Conv,
    pub up1: Conv,
    pub up2: Conv,
    pub convc1: Conv,
    pub convc2: Conv,
    pub convf1: Conv,
    pub convf2: Conv,
    pub enc: Conv,
    pub refine: Vec<ConvNext>,
}

struct Weights(HashMap<String, Tensor>);

impl Weights {
    fn take(&mut self, k: &str) -> Result<Tensor> {
        self.0.remove(k).with_context(|| format!("missing tensor {k}"))
    }

    fn conv(&mut self, name: &str) -> Result<Conv> {
        let w = self.take(&format!("{name}.weight"))?;
        let (oc, ic, k) = match w.dims.as_slice() {
            [o, i, k, k2] if k == k2 => (*o, *i, *k),
            [o, i] => (*o, *i, 1),
            d => bail!("{name}.weight: shape {d:?}"),
        };
        let b = match self.0.remove(&format!("{name}.bias")) {
            Some(b) => b.data,
            None => vec![0.0; oc],
        };
        anyhow::ensure!(b.len() == oc, "{name}.bias: {} for {oc} channels", b.len());
        Ok(Conv { oc, ic, k, w: w.data, b })
    }

    /// A convolution followed by an eval-mode BatchNorm, as one conv.
    fn conv_bn(&mut self, conv: &str, bn: &str) -> Result<Conv> {
        let mut c = self.conv(conv)?;
        let g = self.take(&format!("{bn}.weight"))?.data;
        let beta = self.take(&format!("{bn}.bias"))?.data;
        let mean = self.take(&format!("{bn}.running_mean"))?.data;
        let var = self.take(&format!("{bn}.running_var"))?.data;
        let per = c.ic * c.k * c.k;
        for o in 0..c.oc {
            let s = g[o] / (var[o] + 1e-5).sqrt();
            for v in &mut c.w[o * per..(o + 1) * per] {
                *v *= s;
            }
            c.b[o] = (c.b[o] - mean[o]) * s + beta[o];
        }
        Ok(c)
    }

    fn trunk(&mut self, p: &str) -> Result<Trunk> {
        let stem = self.conv_bn(&format!("{p}.conv1"), &format!("{p}.bn1"))?;
        let mut blocks = vec![];
        for (s, (&n, _)) in BLOCKS.iter().zip(WIDTHS).enumerate() {
            for i in 0..n {
                let b = format!("{p}.layer{}.{i}", s + 1);
                let stride = if s > 0 && i == 0 { 2 } else { 1 };
                let down = if stride == 2 {
                    Some(self.conv_bn(&format!("{b}.downsample.0"), &format!("{b}.bn3"))?)
                } else {
                    None
                };
                blocks.push(Block {
                    conv1: self.conv_bn(&format!("{b}.conv1"), &format!("{b}.bn1"))?,
                    conv2: self.conv_bn(&format!("{b}.conv2"), &format!("{b}.bn2"))?,
                    down,
                    stride,
                });
            }
        }
        Ok(Trunk { stem, blocks, out: self.conv(&format!("{p}.final_conv"))? })
    }
}

impl Model {
    pub fn load(path: &Path) -> Result<Self> {
        Self::from_tensors(crate::safetensors::read(path)?).context("not a SEA-RAFT M checkpoint")
    }

    pub fn from_tensors(t: HashMap<String, Tensor>) -> Result<Self> {
        let mut w = Weights(t);
        let fnet = w.trunk("fnet")?;
        let cnet = w.trunk("cnet")?;
        anyhow::ensure!(cnet.stem.ic == 6 && fnet.stem.ic == 3, "trunk inputs are not 6 / 3 channels");
        let mut refine = vec![];
        for i in 0..2 {
            let p = format!("update_block.refine.{i}");
            refine.push(ConvNext {
                dw: w.conv(&format!("{p}.dwconv"))?,
                ln_w: w.take(&format!("{p}.norm.weight"))?.data,
                ln_b: w.take(&format!("{p}.norm.bias"))?.data,
                pw1: w.conv(&format!("{p}.pwconv1"))?,
                pw2: w.conv(&format!("{p}.pwconv2"))?,
                gamma: w.take(&format!("{p}.gamma"))?.data,
                fin: w.conv(&format!("{p}.final"))?,
            });
        }
        let m = Model {
            fnet,
            cnet,
            init_conv: w.conv("init_conv")?,
            head1: w.conv("flow_head.0")?,
            head2: w.conv("flow_head.2")?,
            up1: w.conv("upsample_weight.0")?,
            up2: w.conv("upsample_weight.2")?,
            convc1: w.conv("update_block.encoder.convc1")?,
            convc2: w.conv("update_block.encoder.convc2")?,
            convf1: w.conv("update_block.encoder.convf1")?,
            convf2: w.conv("update_block.encoder.convf2")?,
            enc: w.conv("update_block.encoder.conv")?,
            refine,
        };
        anyhow::ensure!(m.convc1.ic == CORR_C, "correlation is {} channels, this build expects {CORR_C}", m.convc1.ic);
        anyhow::ensure!(m.refine[0].dw.oc == 3 * DIM, "the update block is not {DIM} wide");
        Ok(m)
    }
}

// --------------------------------------------------------------- the GPU

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct ConvU {
    ic: u32,
    oc: u32,
    ih: u32,
    iw: u32,
    oh: u32,
    ow: u32,
    k: u32,
    stride: u32,
    pad: u32,
    in_off: u32,
    out_off: u32,
    res_off: u32,
    flags: u32,
    _p: [u32; 3],
}

const RELU: u32 = 1;
const GELU: u32 = 2;
const SCALE: u32 = 4;
const RES: u32 = 8;
const RELU_AFTER: u32 = 16;

struct Pass {
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    groups: [u32; 3],
}

struct GConv {
    w: wgpu::Buffer,
    b: wgpu::Buffer,
}

/// A planned inference at one (padded) resolution.
pub struct Gpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    /// The caller's size, and the padded one the network runs at.
    w: usize,
    h: usize,
    pw: usize,
    ph: usize,
    pad: [usize; 2],
    passes: Vec<Pass>,
    img: wgpu::Buffer,
    out: wgpu::Buffer,
    staging: wgpu::Buffer,
    #[allow(dead_code)]
    taps: Vec<(&'static str, wgpu::Buffer, usize)>,
}

/// The largest storage buffer a plan at `w × h` needs, in bytes: the
/// first stage's activations at half size.
pub fn largest_buffer(w: usize, h: usize) -> u64 {
    let (pw, ph) = (w.div_ceil(8) * 8, h.div_ceil(8) * 8);
    (WIDTHS[0] * (pw / 2) * (ph / 2) * 4) as u64
}

impl Gpu {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, m: &Model, w: usize, h: usize) -> Result<Self> {
        // InputPadder, sintel mode: replicate, split evenly.
        let (pw, ph) = (w.div_ceil(8) * 8, h.div_ceil(8) * 8);
        let pad = [(pw - w) / 2, (ph - h) / 2];
        let (w8, h8) = (pw / 8, ph / 8);
        anyhow::ensure!(w8 >= 16 && h8 >= 16, "{w}x{h} is too small for SEA-RAFT (128 px each way at least)");
        let max = device.limits().max_storage_buffer_binding_size as u64;
        anyhow::ensure!(
            largest_buffer(w, h) <= max,
            "{w}x{h} needs {} MB buffers, this GPU allows {} MB: analyse at a lower detail",
            largest_buffer(w, h) >> 20,
            max >> 20
        );

        let mut pipelines: HashMap<String, wgpu::ComputePipeline> = HashMap::new();
        for chunk in SHADERS.split("//---kernel:").skip(1) {
            let (name, body) = chunk.split_once('\n').context("bad shader chunk")?;
            let name = name.trim().to_string();
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("searaft-{name}")),
                source: wgpu::ShaderSource::Wgsl(body.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("searaft-{name}")),
                layout: None,
                module: &module,
                entry_point: Some(&name),
                compilation_options: Default::default(),
                cache: None,
            });
            pipelines.insert(name, pipeline);
        }
        let pipe = |n: &str| pipelines.get(n).cloned().unwrap_or_else(|| panic!("missing kernel {n}"));

        let storage = |label: &str, floats: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (floats.max(1) * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let upload = |label: &str, data: &[f32]| {
            let b = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (data.len().max(1) * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: true,
            });
            if !data.is_empty() {
                b.slice(..).get_mapped_range_mut().expect("a mapped buffer").copy_from_slice(bytemuck::cast_slice(data));
            }
            b.unmap();
            b
        };
        let uniform = |data: &[u8]| {
            let b = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("searaft-u"),
                size: data.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: true,
            });
            b.slice(..).get_mapped_range_mut().expect("a mapped buffer").copy_from_slice(data);
            b.unmap();
            b
        };
        let gconv = |c: &Conv| GConv { w: upload("searaft-w", &c.w), b: upload("searaft-b", &c.b) };
        // The tiled conv's weights: [oc/16][ic][k][k][16], zero-padded.
        let gconv16 = |c: &Conv| {
            let groups = c.oc.div_ceil(16);
            let kk = c.k * c.k;
            let mut w = vec![0f32; groups * c.ic * kk * 16];
            for o in 0..c.oc {
                for i in 0..c.ic {
                    for t in 0..kk {
                        w[(((o / 16) * c.ic + i) * kk + t) * 16 + o % 16] = c.w[(o * c.ic + i) * kk + t];
                    }
                }
            }
            GConv { w: upload("searaft-w16", &w), b: upload("searaft-b", &c.b) }
        };

        let n2 = (pw / 2) * (ph / 2);
        let n8 = w8 * h8;
        let big = WIDTHS[0] * n2;
        // Activations.
        let img = storage("searaft-img", 6 * pw * ph);
        let a = storage("searaft-a", big);
        let b = storage("searaft-b", big);
        let c = storage("searaft-c", big);
        let d = storage("searaft-d", big);
        let fmap1 = storage("searaft-fmap1", 256 * n8);
        // The second frame's features and their pooled levels, end to end.
        let mut lvl = vec![(w8, h8)];
        for _ in 1..LEVELS {
            let (lw, lh) = *lvl.last().unwrap();
            lvl.push((lw / 2, lh / 2));
        }
        let mut offs = vec![0usize];
        for l in 1..LEVELS {
            offs.push(offs[l - 1] + 256 * lvl[l - 1].0 * lvl[l - 1].1);
        }
        let fmap2 = storage("searaft-fmap2", offs[LEVELS - 1] + 256 * lvl[LEVELS - 1].0 * lvl[LEVELS - 1].1);
        let x = storage("searaft-x", 3 * DIM * n8);
        let t1 = storage("searaft-t1", 3 * DIM * n8);
        let t2 = storage("searaft-t2", 3 * DIM * n8);
        let hid = storage("searaft-hid", 4 * DIM * n8);
        let corr = storage("searaft-corr", CORR_C * n8);
        let m1 = storage("searaft-m1", 256 * n8);
        let cf = storage("searaft-cf", 256 * n8);
        let upd = storage("searaft-upd", 6 * n8);
        let flow = storage("searaft-flow", 2 * n8);
        let mask = storage("searaft-mask", 576 * n8);
        let out = storage("searaft-out", 6 * pw * ph);
        let dummy = storage("searaft-dummy", 4);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("searaft-staging"),
            size: (6 * pw * ph * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut passes: Vec<Pass> = vec![];
        let mut keep: Vec<GConv> = vec![];
        let mut keep_b: Vec<wgpu::Buffer> = vec![];
        let bind = |pipeline: &wgpu::ComputePipeline, u: &wgpu::Buffer, bufs: &[&wgpu::Buffer]| {
            let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() }];
            for (k, b) in bufs.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry { binding: (k + 1) as u32, resource: b.as_entire_binding() });
            }
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("searaft"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            })
        };
        let g8 = |w: usize, h: usize, z: usize| [(w as u32).div_ceil(8), (h as u32).div_ceil(8), z as u32];

        // One convolution dispatch.
        #[allow(clippy::too_many_arguments)]
        let conv = |passes: &mut Vec<Pass>,
                        keep: &mut Vec<GConv>,
                        cv: &Conv,
                        src: &wgpu::Buffer,
                        in_off: usize,
                        (iw, ih): (usize, usize),
                        stride: usize,
                        dst: &wgpu::Buffer,
                        out_off: usize,
                        flags: u32,
                        scale: Option<&wgpu::Buffer>,
                        res: Option<(&wgpu::Buffer, usize)>| {
            let g = gconv16(cv);
            let pad = cv.k / 2;
            let (ow, oh) = ((iw + 2 * pad - cv.k) / stride + 1, (ih + 2 * pad - cv.k) / stride + 1);
            let u = ConvU {
                ic: cv.ic as u32,
                oc: cv.oc as u32,
                ih: ih as u32,
                iw: iw as u32,
                oh: oh as u32,
                ow: ow as u32,
                k: cv.k as u32,
                stride: stride as u32,
                pad: pad as u32,
                in_off: in_off as u32,
                out_off: out_off as u32,
                res_off: res.map(|r| r.1).unwrap_or(0) as u32,
                flags,
                ..Default::default()
            };
            let p = pipe("conv");
            let ub = uniform(bytemuck::bytes_of(&u));
            let bg = bind(&p, &ub, &[src, &g.w, &g.b, scale.unwrap_or(&dummy), res.map(|r| r.0).unwrap_or(&dummy), dst]);
            passes.push(Pass { pipeline: p, bind: bg, groups: g8(ow, oh, cv.oc.div_ceil(16)) });
            keep.push(g);
            (ow, oh)
        };

        // A trunk from `src` (channels `in_off..`) to 256 channels at 1/8
        // in `dst`. Blocks ping-pong through a..d.
        let trunk = |passes: &mut Vec<Pass>, keep: &mut Vec<GConv>, t: &Trunk, in_off: usize, dst: &wgpu::Buffer| {
            let mut size = conv(passes, keep, &t.stem, &img, in_off, (pw, ph), 2, &a, 0, RELU, None, None);
            let bufs = [&a, &b, &c, &d];
            let mut cur = 0usize;
            for blk in &t.blocks {
                let others: Vec<usize> = (0..4).filter(|i| *i != cur).collect();
                let (y, z, sk) = (others[0], others[1], others[2]);
                let s1 = conv(passes, keep, &blk.conv1, bufs[cur], 0, size, blk.stride, bufs[y], 0, RELU, None, None);
                let res = match &blk.down {
                    Some(dn) => {
                        conv(passes, keep, dn, bufs[cur], 0, size, blk.stride, bufs[sk], 0, 0, None, None);
                        sk
                    }
                    None => cur,
                };
                conv(passes, keep, &blk.conv2, bufs[y], 0, s1, 1, bufs[z], 0, RELU | RES | RELU_AFTER, None, Some((bufs[res], 0)));
                size = s1;
                cur = z;
            }
            conv(passes, keep, &t.out, bufs[cur], 0, size, 1, dst, 0, 0, None, None);
        };

        // Features: frame 1 into fmap1, frame 2 into fmap2's level 0.
        trunk(&mut passes, &mut keep, &m.fnet, 0, &fmap1);
        trunk(&mut passes, &mut keep, &m.fnet, 3, &fmap2);
        for l in 1..LEVELS {
            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct PoolU {
                c: u32,
                ih: u32,
                iw: u32,
                oh: u32,
                ow: u32,
                src_off: u32,
                dst_off: u32,
                _p: u32,
            }
            let u = PoolU {
                c: 256,
                ih: lvl[l - 1].1 as u32,
                iw: lvl[l - 1].0 as u32,
                oh: lvl[l].1 as u32,
                ow: lvl[l].0 as u32,
                src_off: offs[l - 1] as u32,
                dst_off: 0,
                _p: 0,
            };
            // Pooling reads and writes the same buffer at disjoint ranges,
            // which a bind group cannot express: bounce through `a`.
            let ub = uniform(bytemuck::bytes_of(&u));
            let p = pipe("pool2");
            let bg = bind(&p, &ub, &[&fmap2, &a]);
            passes.push(Pass { pipeline: p, bind: bg, groups: g8(lvl[l].0, lvl[l].1, 256) });
            keep_b.push(ub);
            let n = 256 * lvl[l].0 * lvl[l].1;
            passes.push(Pass {
                pipeline: pipe("copy"),
                bind: {
                    let cu = [offs[l] as u32, n as u32, 0, 0];
                    let ub = uniform(bytemuck::cast_slice(&cu));
                    let p = pipe("copy");
                    let bg = bind(&p, &ub, &[&a, &fmap2]);
                    keep_b.push(ub);
                    bg
                },
                groups: [(n as u32).div_ceil(256), 1, 1],
            });
        }

        // Context: both frames through cnet, then init_conv straight into
        // x's first 256 channels (hidden | context).
        trunk(&mut passes, &mut keep, &m.cnet, 0, &t1);
        conv(&mut passes, &mut keep, &m.init_conv, &t1, 0, (w8, h8), 1, &x, 0, 0, None, None);

        let step = |passes: &mut Vec<Pass>, keep_b: &mut Vec<wgpu::Buffer>, reset: bool| {
            let su = [h8 as u32, w8 as u32, (3 * DIM - 2) as u32, reset as u32];
            let ub = uniform(bytemuck::cast_slice(&su));
            let p = pipe("flowstep");
            let bg = bind(&p, &ub, &[&upd, &flow, &x]);
            passes.push(Pass { pipeline: p, bind: bg, groups: g8(w8, h8, 1) });
            keep_b.push(ub);
        };
        // The initial flow, off the hidden state.
        conv(&mut passes, &mut keep, &m.head1, &x, 0, (w8, h8), 1, &hid, 0, RELU, None, None);
        conv(&mut passes, &mut keep, &m.head2, &hid, 0, (w8, h8), 1, &upd, 0, 0, None, None);
        step(&mut passes, &mut keep_b, true);

        let gammas: Vec<wgpu::Buffer> = m.refine.iter().map(|r| upload("searaft-gamma", &r.gamma)).collect();
        let lns: Vec<(wgpu::Buffer, wgpu::Buffer)> =
            m.refine.iter().map(|r| (upload("searaft-ln-w", &r.ln_w), upload("searaft-ln-b", &r.ln_b))).collect();
        for _ in 0..ITERS {
            // The correlation lookup.
            let cu = [
                256u32,
                h8 as u32,
                w8 as u32,
                RADIUS as u32,
                lvl[0].0 as u32,
                lvl[0].1 as u32,
                lvl[1].0 as u32,
                lvl[1].1 as u32,
                lvl[2].0 as u32,
                lvl[2].1 as u32,
                lvl[3].0 as u32,
                lvl[3].1 as u32,
                offs[1] as u32,
                offs[2] as u32,
                offs[3] as u32,
                0,
            ];
            let ub = uniform(bytemuck::cast_slice(&cu));
            let p = pipe("corr");
            let bg = bind(&p, &ub, &[&fmap1, &fmap2, &flow, &corr]);
            passes.push(Pass { pipeline: p, bind: bg, groups: g8(w8, h8, LEVELS * (2 * RADIUS + 1)) });
            keep_b.push(ub);
            // The motion encoder: [corr 192 | flow 64] -> 126, into x.
            let s8 = (w8, h8);
            conv(&mut passes, &mut keep, &m.convc1, &corr, 0, s8, 1, &m1, 0, RELU, None, None);
            conv(&mut passes, &mut keep, &m.convc2, &m1, 0, s8, 1, &cf, 0, RELU, None, None);
            conv(&mut passes, &mut keep, &m.convf1, &flow, 0, s8, 1, &m1, 0, RELU, None, None);
            conv(&mut passes, &mut keep, &m.convf2, &m1, 0, s8, 1, &cf, 192, RELU, None, None);
            conv(&mut passes, &mut keep, &m.enc, &cf, 0, s8, 1, &x, 2 * DIM, RELU, None, None);
            // Two ConvNeXt blocks over x = [hidden | context | motion | flow].
            for (r, (gam, (lnw, lnb))) in m.refine.iter().zip(gammas.iter().zip(&lns)) {
                let dwg = gconv(&r.dw);
                let du = [(3 * DIM) as u32, h8 as u32, w8 as u32, r.dw.k as u32];
                let ub = uniform(bytemuck::cast_slice(&du));
                let p = pipe("dwconv");
                let bg = bind(&p, &ub, &[&x, &dwg.w, &dwg.b, &t1]);
                passes.push(Pass { pipeline: p, bind: bg, groups: g8(w8, h8, 3 * DIM) });
                keep_b.push(ub);
                keep.push(dwg);
                let lu = [(3 * DIM) as u32, h8 as u32, w8 as u32, 0];
                let ub = uniform(bytemuck::cast_slice(&lu));
                let p = pipe("layernorm");
                let bg = bind(&p, &ub, &[&t1, lnw, lnb, &t2]);
                passes.push(Pass { pipeline: p, bind: bg, groups: g8(w8, h8, 1) });
                keep_b.push(ub);
                conv(&mut passes, &mut keep, &r.pw1, &t2, 0, s8, 1, &hid, 0, GELU, None, None);
                conv(&mut passes, &mut keep, &r.pw2, &hid, 0, s8, 1, &t1, 0, SCALE | RES, Some(gam), Some((&x, 0)));
                conv(&mut passes, &mut keep, &r.fin, &t1, 0, s8, 1, &x, 0, 0, None, None);
            }
            conv(&mut passes, &mut keep, &m.head1, &x, 0, s8, 1, &hid, 0, RELU, None, None);
            conv(&mut passes, &mut keep, &m.head2, &hid, 0, s8, 1, &upd, 0, 0, None, None);
            step(&mut passes, &mut keep_b, false);
        }
        // The upsampling mask, once, off the last hidden state.
        conv(&mut passes, &mut keep, &m.up1, &x, 0, (w8, h8), 1, &hid, 0, RELU, None, None);
        conv(&mut passes, &mut keep, &m.up2, &hid, 0, (w8, h8), 1, &mask, 0, 0, None, None);
        let uu = [h8 as u32, w8 as u32, 0, 0];
        let ub = uniform(bytemuck::cast_slice(&uu));
        let p = pipe("upsample");
        let bg = bind(&p, &ub, &[&mask, &flow, &upd, &out]);
        passes.push(Pass { pipeline: p, bind: bg, groups: g8(pw, ph, 1) });
        keep_b.push(ub);
        drop(keep);
        drop(keep_b);

        Ok(Self {
            device,
            queue,
            w,
            h,
            pw,
            ph,
            pad,
            passes,
            img,
            out,
            staging,
            taps: vec![("fmap1", fmap1, 256 * n8), ("x", x, 3 * DIM * n8)],
        })
    }

    /// Flow from `img1` to `img2` (each `[3][h][w]`, 0..1 sRGB-encoded),
    /// in pixels: `[2][h][w]` flow then `[4][h][w]` uncertainty.
    pub fn run(&self, img1: &[f32], img2: &[f32]) -> Result<Vec<f32>> {
        self.run_upto(img1, img2, self.passes.len())
    }

    fn run_upto(&self, img1: &[f32], img2: &[f32], upto: usize) -> Result<Vec<f32>> {
        let (w, h, pw, ph) = (self.w, self.h, self.pw, self.ph);
        assert_eq!(img1.len(), 3 * w * h);
        assert_eq!(img2.len(), 3 * w * h);
        // 2x - 1, replicate-padded into [img1 | img2].
        let mut stack = vec![0f32; 6 * pw * ph];
        for (f, src) in [img1, img2].iter().enumerate() {
            for c in 0..3 {
                for y in 0..ph {
                    let sy = y.saturating_sub(self.pad[1]).min(h - 1);
                    for x in 0..pw {
                        let sx = x.saturating_sub(self.pad[0]).min(w - 1);
                        stack[((f * 3 + c) * ph + y) * pw + x] = 2.0 * src[(c * h + sy) * w + sx] - 1.0;
                    }
                }
            }
        }
        self.queue.write_buffer(&self.img, 0, bytemuck::cast_slice(&stack));
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("searaft") });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("searaft"), timestamp_writes: None });
            for p in self.passes.iter().take(upto) {
                cp.set_pipeline(&p.pipeline);
                cp.set_bind_group(0, &p.bind, &[]);
                cp.dispatch_workgroups(p.groups[0], p.groups[1], p.groups[2]);
            }
        }
        enc.copy_buffer_to_buffer(&self.out, 0, &self.staging, 0, (6 * pw * ph * 4) as u64);
        self.queue.submit([enc.finish()]);
        let all = self.read(&self.staging)?;
        // Unpad.
        let mut out = vec![0f32; 6 * w * h];
        for c in 0..6 {
            for y in 0..h {
                let s = (c * ph + y + self.pad[1]) * pw + self.pad[0];
                out[(c * h + y) * w..][..w].copy_from_slice(&all[s..s + w]);
            }
        }
        Ok(out)
    }

    fn read(&self, staging: &wgpu::Buffer) -> Result<Vec<f32>> {
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().context("map_async dropped")?.context("staging map failed")?;
        let v = bytemuck::cast_slice(&slice.get_mapped_range().expect("a mapped buffer")).to_vec();
        staging.unmap();
        Ok(v)
    }

    /// A buffer the plan wrote, read back after a full run (tests).
    #[cfg(test)]
    fn tap(&self, name: &str) -> Vec<f32> {
        let (_, b, n) = self.taps.iter().find(|(t, _, _)| *t == name).expect("a tap");
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (*n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(b, 0, &staging, 0, (*n * 4) as u64);
        self.queue.submit([enc.finish()]);
        self.read(&staging).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    /// End to end against the real torch model (`tools/searaft_golden.py`):
    /// the features, the context, and the final flow. Self-skips without
    /// `data/searaft/` or a GPU.
    #[test]
    fn golden_pair_matches_torch() {
        let weights = Path::new("../data/searaft/C-T-M.safetensors");
        let golden = Path::new("../data/searaft/golden.wifw");
        if !weights.exists() || !golden.exists() {
            eprintln!("SKIP: no data/searaft/C-T-M.safetensors + golden.wifw");
            return;
        }
        let Ok(ctx) = crate::GpuContext::new() else {
            eprintln!("SKIP: no GPU");
            return;
        };
        let m = Model::load(weights).unwrap();
        let g = crate::rife::read_wifw(golden).unwrap();
        let (img1, img2) = (&g["golden.img1"], &g["golden.img2"]);
        let (h, w) = (img1.dims[1], img1.dims[2]);
        let gpu = Gpu::new(ctx.device.clone(), ctx.queue.clone(), &m, w, h).unwrap();
        let out = gpu.run(&img1.data, &img2.data).unwrap();
        let df = max_abs(&gpu.tap("fmap1"), &g["golden.fmap1"].data);
        // The hidden half of x has been refined since; the context half is
        // init_conv's output as it was.
        let n8 = (w / 8) * (h / 8);
        let dc = max_abs(&gpu.tap("x")[DIM * n8..2 * DIM * n8], &g["golden.cnet"].data[DIM * n8..2 * DIM * n8]);
        let want = &g["golden.flow"].data;
        let dflow = max_abs(&out[..2 * w * h], want);
        let n = (w * h) as f32;
        let mean = |v: &[f32], c: usize| v[c * w * h..(c + 1) * w * h].iter().sum::<f32>() / n;
        eprintln!(
            "searaft vs torch: fmap1 {df:.2e}, cnet {dc:.2e}, flow {dflow:.2e}; mean flow ({:.3}, {:.3}) vs ({:.3}, {:.3})",
            mean(&out, 0),
            mean(&out, 1),
            mean(want, 0),
            mean(want, 1)
        );
        assert!(df < 1e-2, "features diverge: {df}");
        assert!(dc < 1e-2, "context diverges: {dc}");
        assert!(dflow < 5e-2, "flow diverges: {dflow}");
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    /// `cargo test --release -p engine searaft::bench -- --ignored
    /// --nocapture`: where a frame's time goes, pass group by pass group.
    #[test]
    #[ignore]
    fn timing() {
        let m = Model::load(Path::new("../data/searaft/C-T-M.safetensors")).unwrap();
        let ctx = crate::GpuContext::new().unwrap();
        let (w, h) = (960, 544);
        let img: Vec<f32> = (0..3 * w * h).map(|i| ((i * 7919) % 1000) as f32 / 1000.0).collect();
        let gpu = Gpu::new(ctx.device.clone(), ctx.queue.clone(), &m, w, h).unwrap();
        let _ = gpu.run(&img, &img).unwrap();
        let best = (0..4)
            .map(|_| {
                let t = std::time::Instant::now();
                let _ = gpu.run(&img, &img).unwrap();
                t.elapsed().as_secs_f64()
            })
            .fold(f64::MAX, f64::min);
        eprintln!("SEA-RAFT {w}x{h}: {best:.3}s a pair (best of 4)");
        let total = gpu.passes.len();
        // Each pass alone (its inputs are whatever the last run left).
        let mut times: Vec<(f64, usize)> = (0..total)
            .map(|i| {
                let t = std::time::Instant::now();
                let mut enc = gpu.device.create_command_encoder(&Default::default());
                {
                    let mut cp = enc.begin_compute_pass(&Default::default());
                    let p = &gpu.passes[i];
                    cp.set_pipeline(&p.pipeline);
                    cp.set_bind_group(0, &p.bind, &[]);
                    cp.dispatch_workgroups(p.groups[0], p.groups[1], p.groups[2]);
                }
                gpu.queue.submit([enc.finish()]);
                let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
                (t.elapsed().as_secs_f64(), i)
            })
            .collect();
        times.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for (s, i) in times.iter().take(15) {
            eprintln!("pass {i:3} groups {:?}: {:.3}s", gpu.passes[*i].groups, s);
        }
        let mut last = 0.0;
        for upto in [1usize, 20, 40, 80, 100, 120, total] {
            let upto = upto.min(total);
            let t = std::time::Instant::now();
            let _ = gpu.run_upto(&img, &img, upto).unwrap();
            let s = t.elapsed().as_secs_f64();
            eprintln!("first {upto:3} of {total} passes: {:.3}s (+{:.3})", s, s - last);
            last = s;
        }
    }
}
