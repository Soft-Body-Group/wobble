//! Running a [`Net`] on the GPU, tile by tile.
//!
//! The picture is cut into overlapping tiles of the input; each tile is
//! loaded into the arena, run through every op, and the middle of its
//! answer (the halo trimmed) written into the output texture. Tiles are all
//! the same size — the last row and column slide inwards to stay inside the
//! picture — so a conv's uniform is the same for every tile and the whole
//! frame is one compute pass of dispatches against two bind groups. The
//! arena is sized for one tile, which is what lets the memory scale down
//! with the card rather than the picture (manifesto rule 3).

use super::wgsl::{self, ConvKey, ConvU, CopyU, EmitU, LoadU, MapU};
use super::{pad4, Epi, Io, Net, Op, Out, Slice};
use std::collections::HashMap;
use std::sync::Arc;

/// Compiled kernels and their layouts, shared by every runner on a device:
/// a Metal compile is ~100 ms, and a canvas change rebuilds a runner.
pub struct Kernels {
    device: Arc<wgpu::Device>,
    /// The f16 arena where the device can do f16 arithmetic in shaders.
    pub half: bool,
    conv_layout: wgpu::BindGroupLayout,
    conv_pl: wgpu::PipelineLayout,
    load_layout: wgpu::BindGroupLayout,
    load_pl: wgpu::PipelineLayout,
    emit_layout: wgpu::BindGroupLayout,
    emit_pl: wgpu::PipelineLayout,
    convs: HashMap<ConvKey, wgpu::ComputePipeline>,
    deconvs: HashMap<ConvKey, wgpu::ComputePipeline>,
    pool: Option<wgpu::ComputePipeline>,
    logskip: Option<wgpu::ComputePipeline>,
    copy: Option<wgpu::ComputePipeline>,
    load: Option<wgpu::ComputePipeline>,
    emit: Option<wgpu::ComputePipeline>,
}

fn entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None }
}

fn uniform_dyn(size: u64) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: true,
        min_binding_size: wgpu::BufferSize::new(size),
    }
}

fn storage(read_only: bool) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

/// Per-dispatch uniform slots are this far apart (the dynamic-offset
/// alignment every adapter accepts).
const SLOT: u64 = 256;

impl Kernels {
    /// `half`: store features as f16. Asked for, and granted only where the
    /// device has `SHADER_F16`.
    pub fn new(device: Arc<wgpu::Device>, half: bool) -> Self {
        let half = half && device.features().contains(wgpu::Features::SHADER_F16);
        let conv_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nn-conv"),
            entries: &[
                entry(0, uniform_dyn(64)),
                entry(1, storage(false)),
                entry(2, storage(true)),
            ],
        });
        let load_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nn-load"),
            entries: &[
                entry(0, uniform_dyn(std::mem::size_of::<LoadU>() as u64)),
                entry(1, storage(false)),
                entry(
                    2,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
            ],
        });
        let emit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nn-emit"),
            entries: &[
                entry(0, uniform_dyn(std::mem::size_of::<EmitU>() as u64)),
                entry(1, storage(false)),
                entry(
                    2,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                entry(3, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
                entry(
                    4,
                    wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                ),
            ],
        });
        let pl = |l: &wgpu::BindGroupLayout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("nn"),
                bind_group_layouts: &[Some(l)],
                immediate_size: 0,
            })
        };
        let (conv_pl, load_pl, emit_pl) = (pl(&conv_layout), pl(&load_layout), pl(&emit_layout));
        Self {
            device,
            half,
            conv_layout,
            conv_pl,
            load_layout,
            load_pl,
            emit_layout,
            emit_pl,
            convs: HashMap::new(),
            deconvs: HashMap::new(),
            pool: None,
            logskip: None,
            copy: None,
            load: None,
            emit: None,
        }
    }

    fn pipeline(&self, label: &str, wgsl: &str, layout: &wgpu::PipelineLayout) -> wgpu::ComputePipeline {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    fn ensure(&mut self, net: &Net) {
        let half = self.half;
        for op in &net.ops {
            match op {
                Op::Conv(c) => {
                    let key = conv_key(c, half);
                    if !self.convs.contains_key(&key) {
                        let p = self.pipeline("nn-conv", &wgsl::conv(key), &self.conv_pl);
                        self.convs.insert(key, p);
                    }
                }
                Op::Deconv(c) => {
                    let key = conv_key(c, half);
                    if !self.deconvs.contains_key(&key) {
                        let p = self.pipeline("nn-deconv", &wgsl::deconv(key), &self.conv_pl);
                        self.deconvs.insert(key, p);
                    }
                }
                Op::Pool { .. } if self.pool.is_none() => {
                    self.pool = Some(self.pipeline("nn-pool", &wgsl::pool(half), &self.conv_pl));
                }
                Op::LogSkip { .. } if self.logskip.is_none() => {
                    self.logskip = Some(self.pipeline("nn-logskip", &wgsl::logskip(half), &self.conv_pl));
                }
                _ => {}
            }
        }
        if self.copy.is_none() {
            self.copy = Some(self.pipeline("nn-copy", &wgsl::copy(half), &self.conv_pl));
        }
        if self.load.is_none() {
            self.load = Some(self.pipeline("nn-load", &wgsl::load(half), &self.load_pl));
        }
        if self.emit.is_none() {
            self.emit = Some(self.pipeline("nn-emit", &wgsl::emit(half), &self.emit_pl));
        }
    }
}

/// Output groups one thread computes: the largest of 4..1 that divides
/// the conv's groups, so no thread computes padding.
fn ocb_for(cout: u32) -> u32 {
    let g = pad4(cout) / 4;
    [4, 3, 2, 1].into_iter().find(|b| g % b == 0).unwrap()
}

fn conv_key(c: &super::Conv, half: bool) -> ConvKey {
    let (epi, nres) = match &c.epi {
        Epi::Plain { res, slope, .. } => (slope.is_some() as u32, res.len() as u32),
        Epi::Gate { .. } => (2, 1),
        Epi::Silu => (3, 0),
    };
    assert!(nres <= 2, "at most two residual slices");
    ConvKey { k: c.k, ocb: ocb_for(c.cout), up: c.up, silu_in: c.silu_in, epi, nres, half }
}

/// A transposed conv's weights as `wgsl::deconv` reads them: per output
/// group, per input group, per tap (16), per input lane, a vec4 over the
/// group's four output channels.
fn repack_deconv(c: &super::Conv) -> Vec<[f32; 4]> {
    let (cin, cout) = (c.cin as usize, c.cout as usize);
    let (ic4, oc4) = (pad4(c.cin) as usize / 4, pad4(c.cout) as usize / 4);
    let mut out = Vec::with_capacity(oc4 * ic4 * 64);
    for og in 0..oc4 {
        for ic in 0..ic4 {
            for t in 0..16 {
                for j in 0..4 {
                    let i = ic * 4 + j;
                    out.push(std::array::from_fn(|l| {
                        let o = og * 4 + l;
                        if o < cout && i < cin {
                            c.weight[(o * cin + i) * 16 + t]
                        } else {
                            0.0
                        }
                    }));
                }
            }
        }
    }
    out
}

/// The weights in the order the kernel reads them: per output block, per
/// input group, per tap, per output vec4, four input channels — each a
/// vec4 over four output channels. Padding is zero.
fn repack(c: &super::Conv) -> Vec<[f32; 4]> {
    let (cin, cout, k) = (c.cin as usize, c.cout as usize, c.k as usize);
    let ocb = ocb_for(c.cout) as usize;
    let (ic4, oc4) = (pad4(c.cin) as usize / 4, pad4(c.cout) as usize / 4);
    let mut out = Vec::with_capacity(oc4 * ic4 * k * k * 4);
    for og in 0..oc4 / ocb {
        for ic in 0..ic4 {
            for t in 0..k * k {
                for ob in 0..ocb {
                    for j in 0..4 {
                        let i = ic * 4 + j;
                        let mut v = [0.0f32; 4];
                        for (l, slot) in v.iter_mut().enumerate() {
                            let o = (og * ocb + ob) * 4 + l;
                            if o < cout && i < cin {
                                *slot = c.weight[(o * cin + i) * k * k + t];
                            }
                        }
                        out.push(v);
                    }
                }
            }
        }
    }
    out
}

fn vec4s(v: &[f32], n4: usize) -> Vec<[f32; 4]> {
    (0..n4).map(|g| std::array::from_fn(|l| v.get(g * 4 + l).copied().unwrap_or(0.0))).collect()
}

/// How the output and its colour are handled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Colour {
    /// Apply the working <-> display bijection at all (off for tests that
    /// compare raw network values).
    pub encode: bool,
    /// Working -> linear Rec.709, and its inverse.
    pub to_709: [[f32; 3]; 3],
    pub from_709: [[f32; 3]; 3],
    /// An HDR net's clip level, in linear Rec.709: the input is divided by
    /// it on the way in and the prediction multiplied by it on the way out.
    pub clip: f32,
}

impl Colour {
    pub const RAW: Colour = Colour {
        encode: false,
        to_709: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        from_709: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        clip: 1.0,
    };
}

struct Tile {
    ox: u32,
    oy: u32,
    /// Output region this tile writes.
    v: [u32; 4],
}

struct Dispatch {
    pipeline: Pipe,
    offset: u32,
    groups: (u32, u32, u32),
    /// Multiply-adds, for splitting the frame into submissions.
    macs: u64,
}

/// Multiply-adds one submission may carry. Windows resets a GPU whose
/// submission runs past ~2 s (TDR) and the whole frame of an RRDB is a
/// minute on a laptop GPU; at this size a submission is ~0.1 s on an M4
/// Pro and stays under the limit on a GPU ten times slower.
pub const SUBMIT_MACS: u64 = 100_000_000_000;

#[derive(Clone, Copy)]
enum Pipe {
    Conv(ConvKey),
    Deconv(ConvKey),
    Pool,
    LogSkip,
    Copy,
}

/// One net, built for one input size and output scale.
pub struct Runner {
    tw: u32,
    th: u32,
    tiles: Vec<Tile>,
    ops: Vec<Dispatch>,
    _arena: wgpu::Buffer,
    _weights: wgpu::Buffer,
    _op_uniforms: wgpu::Buffer,
    tile_uniforms: wgpu::Buffer,
    conv_bg: wgpu::BindGroup,
    in_base: u32,
    out_base: u32,
    shuffle: bool,
    base: bool,
    io: Io,
    net_scale: u32,
    out_scale: u32,
    in_size: (u32, u32),
    mean: [f32; 3],
    range: f32,
    colour: Option<Colour>,
    /// Feature bytes per tile, for the panel.
    pub arena_bytes: u64,
    /// Multiply-adds per submission ([`SUBMIT_MACS`]; tests lower it).
    pub submit_macs: u64,
}

impl Runner {
    /// Build for an input of `in_size` pixels, writing `out_scale` (the
    /// net's scale, or half of it) times that. `budget` caps the arena.
    pub fn new(
        k: &mut Kernels,
        queue: &wgpu::Queue,
        net: &Net,
        in_size: (u32, u32),
        out_scale: u32,
        budget: u64,
    ) -> Self {
        k.ensure(net);
        let device = k.device.clone();
        let elem = if k.half { 8u64 } else { 16 };
        let max_bind = device.limits().max_storage_buffer_binding_size.min(budget);
        let (iw, ih) = (in_size.0.max(1), in_size.1.max(1));
        // The largest square tile whose arena fits, floored where the halo
        // would leave nothing in the middle, capped at 768: measured on an
        // M4 Pro at 1080p, 512 pays ~25% in halo for the compact net and
        // 1024 runs slower again (a feature buffer outgrows the caches).
        //
        // A net with pooled levels is cut on its grid (`Net::align`): tile
        // sizes, origins and the halo are multiples of it, and the picture
        // is padded up to one (its edge repeated, `LoadU::lim`), so every
        // pool pairs the same pixels as it would over the whole picture.
        let align = net.align();
        let step = align.max(16);
        let halo = net.halo.next_multiple_of(align);
        let (pw, ph) = (iw.next_multiple_of(align), ih.next_multiple_of(align));
        // On the grid when there is one: the halo is, and so is 32 for any
        // align that divides it (pooled nets go by powers of two).
        let min_t = 2 * halo + 32.max(align);
        let mut t = 768u32;
        while t > min_t && net.values_per_tile(t) / 4 * elem > max_bind {
            t -= step;
        }
        let (tw, th) = (t.min(pw), t.min(ph));
        let plane = |b: usize| net.bufs[b].side(tw) * net.bufs[b].side(th);
        // Arena layout: buffers back to back, each [ch/4][h][w].
        let mut bases = Vec::with_capacity(net.bufs.len());
        let mut at = 0u32;
        for (i, b) in net.bufs.iter().enumerate() {
            bases.push(at);
            at += b.ch / 4 * plane(i);
        }
        let arena_elems = at.max(1) as u64;
        let arena_bytes = arena_elems * elem;
        let off = |s: &Slice| bases[s.buf] + s.ch / 4 * plane(s.buf);
        let dims = |s: &Slice| (net.bufs[s.buf].side(tw), net.bufs[s.buf].side(th));

        // Weights and per-op uniforms.
        let mut weights: Vec<[f32; 4]> = Vec::new();
        let mut uni: Vec<u8> = Vec::new();
        let mut ops = Vec::new();
        for op in &net.ops {
            let offset = uni.len() as u32;
            match op {
                Op::Conv(c) | Op::Deconv(c) => {
                    let de = matches!(op, Op::Deconv(_));
                    let key = conv_key(c, k.half);
                    let w = weights.len() as u32;
                    weights.extend(if de { repack_deconv(c) } else { repack(c) });
                    let oc4 = pad4(c.cout) / 4;
                    let b = weights.len() as u32;
                    weights.extend(vec4s(&c.bias, oc4 as usize));
                    let (mut alpha, mut r, mut beta) = (1.0, [0u32; 2], [0.0f32; 2]);
                    match &c.epi {
                        Epi::Plain { alpha: a, res, slope } => {
                            alpha = *a;
                            if let Some(sl) = slope {
                                weights.extend(vec4s(sl, oc4 as usize));
                            }
                            for (i, (s, be)) in res.iter().enumerate() {
                                r[i] = off(s);
                                beta[i] = *be;
                            }
                        }
                        Epi::Gate { res } => r[0] = off(res),
                        Epi::Silu => {}
                    }
                    let (dw, dh) = dims(&c.dst);
                    let (sw, sh) = dims(&c.src);
                    let u = ConvU {
                        dw,
                        dh,
                        sw,
                        sh,
                        src: off(&c.src),
                        dst: off(&c.dst),
                        ic4: pad4(c.cin) / 4,
                        w,
                        b,
                        oc4,
                        alpha,
                        r0: r[0],
                        beta0: beta[0],
                        r1: r[1],
                        beta1: beta[1],
                        _pad: 0,
                    };
                    uni.extend_from_slice(bytemuck::bytes_of(&u));
                    ops.push(if de {
                        Dispatch {
                            pipeline: Pipe::Deconv(key),
                            offset,
                            groups: (dw.div_ceil(8), dh.div_ceil(8), oc4),
                            macs: dw as u64 * dh as u64 * (c.cin * c.cout * 4) as u64,
                        }
                    } else {
                        Dispatch {
                            pipeline: Pipe::Conv(key),
                            offset,
                            groups: (dw.div_ceil(wgsl::TSX), dh.div_ceil(wgsl::TSY), oc4 / key.ocb),
                            macs: dw as u64 * dh as u64 * (c.cin * c.cout * c.k * c.k) as u64,
                        }
                    });
                }
                Op::Pool { src, dst } => {
                    let (sw, sh) = dims(src);
                    let (dw, dh) = dims(dst);
                    let u = MapU { dw, dh, sw, sh, src: off(src), dst: off(dst), ..Default::default() };
                    uni.extend_from_slice(bytemuck::bytes_of(&u));
                    ops.push(Dispatch { pipeline: Pipe::Pool, offset, groups: (dw.div_ceil(8), dh.div_ceil(8), src.n / 4), macs: 0 });
                }
                Op::LogSkip { src, dst, scale, offset: o } => {
                    let (w, h) = dims(src);
                    let n = src.n / 4 * w * h;
                    let u = MapU { dw: n, src: off(src), dst: off(dst), scale: *scale, offset: *o, ..Default::default() };
                    uni.extend_from_slice(bytemuck::bytes_of(&u));
                    let groups = n.div_ceil(64);
                    let gx = groups.min(65535);
                    ops.push(Dispatch { pipeline: Pipe::LogSkip, offset, groups: (gx, groups.div_ceil(gx), 1), macs: 0 });
                }
                Op::Copy { src, dst } => {
                    let (w, h) = dims(src);
                    let n = src.n / 4 * w * h;
                    let u = CopyU { n, src: off(src), dst: off(dst), _p: 0 };
                    uni.extend_from_slice(bytemuck::bytes_of(&u));
                    let groups = n.div_ceil(64);
                    // Spread over y past the 65535 x limit.
                    let gx = groups.min(65535);
                    ops.push(Dispatch { pipeline: Pipe::Copy, offset, groups: (gx, groups.div_ceil(gx), 1), macs: 0 });
                }
            }
            uni.resize(uni.len().next_multiple_of(SLOT as usize), 0);
        }
        use wgpu::util::DeviceExt;
        let weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nn-weights"),
            contents: bytemuck::cast_slice(if weights.is_empty() { &[[0.0f32; 4]] } else { &weights[..] }),
            usage: wgpu::BufferUsages::STORAGE,
        });
        uni.resize(uni.len().max(SLOT as usize), 0);
        let op_uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nn-op-uniforms"),
            contents: &uni,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nn-arena"),
            size: arena_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Tiles over the (padded) input: valid spans of `t - 2 halo`, each
        // tile sliding inwards at the far edge; what a tile writes stops at
        // the picture.
        let span = |n: u32, p: u32, t: u32| -> Vec<(u32, u32, u32)> {
            if p <= t {
                return vec![(0, 0, n)];
            }
            let v = t - 2 * halo;
            (0..n.div_ceil(v))
                .map(|i| {
                    let x0 = i * v;
                    let o = x0.saturating_sub(halo).min(p - t);
                    (o, x0, (x0 + v).min(n))
                })
                .collect()
        };
        let mut tiles = Vec::new();
        for (oy, y0, y1) in span(ih, ph, th) {
            for (ox, x0, x1) in span(iw, pw, tw) {
                let s = out_scale;
                tiles.push(Tile { ox, oy, v: [x0 * s, y0 * s, (x1 - x0) * s, (y1 - y0) * s] });
            }
        }
        let tile_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nn-tile-uniforms"),
            size: (tiles.len() as u64 * 2 * SLOT).max(SLOT),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let conv_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nn-conv"),
            layout: &k.conv_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &op_uniforms,
                        offset: 0,
                        size: wgpu::BufferSize::new(64),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: arena.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: weights_buf.as_entire_binding() },
            ],
        });
        let (out_base, shuffle) = match net.out {
            Out::Direct(s) => (off(&s), false),
            Out::Shuffle(s) => (off(&s), true),
        };
        let mut r = Self {
            tw,
            th,
            tiles,
            ops,
            _arena: arena,
            _weights: weights_buf,
            _op_uniforms: op_uniforms,
            tile_uniforms,
            conv_bg,
            in_base: off(&net.input),
            out_base,
            shuffle,
            base: net.base,
            io: net.io,
            net_scale: net.scale,
            out_scale,
            in_size: (iw, ih),
            mean: net.mean,
            range: net.range,
            colour: None,
            arena_bytes,
            submit_macs: SUBMIT_MACS,
        };
        r.set_colour(queue, Colour::RAW);
        r
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    pub fn tile_size(&self) -> (u32, u32) {
        (self.tw, self.th)
    }

    /// Write the tiles' load/emit uniforms for a colour handling; a no-op
    /// when it has not changed.
    pub fn set_colour(&mut self, queue: &wgpu::Queue, c: Colour) {
        if self.colour == Some(c) {
            return;
        }
        self.colour = Some(c);
        let row = |m: &[[f32; 3]; 3], i: usize| [m[i][0], m[i][1], m[i][2], 0.0];
        let mut bytes = vec![0u8; self.tiles.len() * 2 * SLOT as usize];
        for (i, t) in self.tiles.iter().enumerate() {
            let l = LoadU {
                ox: t.ox,
                oy: t.oy,
                tw: self.tw,
                th: self.th,
                dst: self.in_base,
                encode: match (c.encode, self.io) {
                    (false, _) => 0,
                    (true, Io::Upscale) => 1,
                    (true, Io::Hdr) => 2,
                },
                clip: c.clip.max(1e-6),
                range: self.range,
                mean: [self.mean[0], self.mean[1], self.mean[2], 0.0],
                m: [row(&c.to_709, 0), row(&c.to_709, 1), row(&c.to_709, 2)],
                lim: [self.in_size.0, self.in_size.1, 0, 0],
            };
            let mode = self.shuffle as u32
                | (self.base as u32) << 1
                | (c.encode as u32) << 2
                | ((self.out_scale != self.net_scale) as u32) << 3
                | ((self.io == Io::Hdr) as u32) << 4;
            let e = EmitU {
                ox: t.ox,
                oy: t.oy,
                tw: self.tw,
                th: self.th,
                vx: t.v[0],
                vy: t.v[1],
                vw: t.v[2],
                vh: t.v[3],
                obase: self.out_base,
                ibase: self.in_base,
                mode,
                clip: c.clip.max(1e-6),
                in_w: self.in_size.0,
                in_h: self.in_size.1,
                s: self.net_scale,
                _p: 0,
                mi: [row(&c.from_709, 0), row(&c.from_709, 1), row(&c.from_709, 2)],
                m: [row(&c.to_709, 0), row(&c.to_709, 1), row(&c.to_709, 2)],
            };
            let at = i * 2 * SLOT as usize;
            bytes[at..at + std::mem::size_of::<LoadU>()].copy_from_slice(bytemuck::bytes_of(&l));
            let at = at + SLOT as usize;
            bytes[at..at + std::mem::size_of::<EmitU>()].copy_from_slice(bytemuck::bytes_of(&e));
        }
        if !bytes.is_empty() {
            queue.write_buffer(&self.tile_uniforms, 0, &bytes);
        }
    }

    /// Multiply-adds per frame.
    pub fn macs(&self) -> u64 {
        self.ops.iter().map(|d| d.macs).sum::<u64>() * self.tiles.len() as u64
    }

    /// Record the whole frame: every tile, load to emit. `input` is the
    /// stage input (the picture at its pixel positions, any size that holds
    /// `in_size`), `output` an rgba16float storage view of `in_size *
    /// out_scale`.
    ///
    /// A frame heavier than [`SUBMIT_MACS`] is SUBMITTED in pieces through
    /// `queue`: what `encoder` holds so far goes first (it rendered the
    /// input), and `encoder` comes back fresh holding the last piece, so
    /// the caller submits it as it would have.
    pub fn encode(
        &self,
        k: &Kernels,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        output: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) {
        let device = &k.device;
        let tile_entry = |size: u64| {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &self.tile_uniforms,
                offset: 0,
                size: wgpu::BufferSize::new(size),
            })
        };
        let load_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nn-load"),
            layout: &k.load_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tile_entry(std::mem::size_of::<LoadU>() as u64) },
                wgpu::BindGroupEntry { binding: 1, resource: self._arena.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(input) },
            ],
        });
        let emit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nn-emit"),
            layout: &k.emit_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tile_entry(std::mem::size_of::<EmitU>() as u64) },
                wgpu::BindGroupEntry { binding: 1, resource: self._arena.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(input) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(output) },
            ],
        });
        let (load, emit, copy) = (k.load.as_ref().unwrap(), k.emit.as_ref().unwrap(), k.copy.as_ref().unwrap());
        let (pool, logskip) = (k.pool.as_ref(), k.logskip.as_ref());
        let split = self.macs() > self.submit_macs;
        let mut pass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("nn"), timestamp_writes: None })
            .forget_lifetime();
        let mut carried = 0u64;
        for (i, t) in self.tiles.iter().enumerate() {
            let at = (i as u64 * 2 * SLOT) as u32;
            pass.set_pipeline(load);
            pass.set_bind_group(0, &load_bg, &[at]);
            pass.dispatch_workgroups(self.tw.div_ceil(8), self.th.div_ceil(8), 1);
            for d in &self.ops {
                if split && carried > 0 && carried + d.macs > self.submit_macs {
                    drop(pass);
                    let done = std::mem::replace(
                        encoder,
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("nn") }),
                    );
                    queue.submit([done.finish()]);
                    pass = encoder
                        .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("nn"), timestamp_writes: None })
                        .forget_lifetime();
                    carried = 0;
                }
                carried += d.macs;
                pass.set_pipeline(match d.pipeline {
                    Pipe::Conv(key) => &k.convs[&key],
                    Pipe::Deconv(key) => &k.deconvs[&key],
                    Pipe::Pool => pool.expect("ensured with the net"),
                    Pipe::LogSkip => logskip.expect("ensured with the net"),
                    Pipe::Copy => copy,
                });
                pass.set_bind_group(0, &self.conv_bg, &[d.offset]);
                pass.dispatch_workgroups(d.groups.0, d.groups.1, d.groups.2);
            }
            pass.set_pipeline(emit);
            pass.set_bind_group(0, &emit_bg, &[at + SLOT as u32]);
            pass.dispatch_workgroups(t.v[2].div_ceil(8), t.v[3].div_ceil(8), 1);
        }
    }
}
