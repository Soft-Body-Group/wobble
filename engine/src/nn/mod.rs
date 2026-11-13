//! Small convolutional networks, run on the GPU in our own WGSL.
//!
//! A [`Net`] is DATA: feature buffers, a list of convolutions over them and
//! where the picture goes in and comes out. A node builds one from a
//! checkpoint (`crate::upscale` does, for the super-resolution models) and
//! hands it to the compositor through `StageDef::net`; [`gpu::Runner`] runs
//! it tile by tile. No Python, no ONNX runtime, no node touching wgpu.
//!
//! The vocabulary is exactly what the shipped architectures need and no
//! more: 3x3 / 1x1 convolutions that read a CHANNEL SLICE of one buffer and
//! write a slice of another (so a dense block's concatenation is free —
//! every layer appends to the same buffer), an optional nearest-neighbour
//! upsample folded into the read, a SiLU on the read, and an epilogue that
//! scales, adds residual slices and applies a leaky/parametric ReLU — or
//! SPAN's attention gate. For an encoder-decoder (HDRCNN, `crate::hdrcnn`)
//! a buffer may sit BELOW the tile's size (`Buf::div`), a 2x2 max pool
//! halves, a 4x4 stride-2 transposed conv doubles, and a log transform maps
//! a skip connection into the decoder's domain. Grow it deliberately,
//! alongside [`reference`], which is the CPU statement of what every op
//! means and what the GPU kernels are tested against.

pub mod gpu;
pub mod reference;
mod wgsl;

use std::sync::Arc;

/// A run of channels in one buffer: `[ch, ch + n)`. Channels are stored in
/// groups of four, so both ends are multiples of 4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slice {
    pub buf: usize,
    pub ch: u32,
    pub n: u32,
}

impl Slice {
    pub fn new(buf: usize, ch: u32, n: u32) -> Self {
        assert!(ch % 4 == 0 && n % 4 == 0, "a slice is whole channel groups");
        Self { buf, ch, n }
    }
}

/// One feature buffer: `ch` channels at `res / div` times the tile's size.
/// `div` > 1 is a buffer below the tile's size (a pooled level); tiles of a
/// net with one are cut on a grid of the largest `div` ([`Net::align`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buf {
    pub ch: u32,
    pub res: u32,
    pub div: u32,
}

impl Buf {
    /// Its side for a tile side `t`.
    pub fn side(&self, t: u32) -> u32 {
        t * self.res / self.div
    }
}

/// What a conv does to its output.
#[derive(Clone, Debug, PartialEq)]
pub enum Epi {
    /// `act(alpha * (conv + bias) + Σ beta_i * res_i)`; `slope` is the
    /// negative-side slope per output channel (a leaky ReLU repeats one
    /// value, a PReLU has its own), `None` for no activation. Residual
    /// slices have the output's channel count and resolution.
    Plain { alpha: f32, res: Vec<(Slice, f32)>, slope: Option<Vec<f32>> },
    /// SPAN's parameter-free attention: `(v + r) * (sigmoid(v) - 0.5)`,
    /// `v = conv + bias`.
    Gate { res: Slice },
    /// `silu(conv + bias)`.
    Silu,
}

impl Epi {
    pub fn none() -> Self {
        Epi::Plain { alpha: 1.0, res: Vec::new(), slope: None }
    }

    pub fn leaky(n: u32, slope: f32) -> Self {
        Epi::Plain { alpha: 1.0, res: Vec::new(), slope: Some(vec![slope; n as usize]) }
    }
}

/// A convolution, stride 1, zero padding to keep the size ("same").
#[derive(Clone, Debug, PartialEq)]
pub struct Conv {
    /// 1 or 3.
    pub k: u32,
    pub cin: u32,
    pub cout: u32,
    /// `[cout][cin][k][k]`, torch's layout.
    pub weight: Arc<[f32]>,
    pub bias: Arc<[f32]>,
    pub src: Slice,
    pub dst: Slice,
    /// Nearest-neighbour upsample of the source by this factor before the
    /// conv (1 = none); the destination is `up` times the source's size.
    pub up: u32,
    /// SiLU applied to the source as it is read.
    pub silu_in: bool,
    pub epi: Epi,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Conv(Conv),
    /// Copy channels from one slice to another of the same size.
    Copy { src: Slice, dst: Slice },
    /// 2x2 max pool, stride 2: the destination is half the source's size.
    Pool { src: Slice, dst: Slice },
    /// A transposed convolution, 4x4, stride 2, TensorFlow's `SAME`
    /// padding: the destination is twice the source's size, and output
    /// pixel `o` takes input `i` through tap `o - 2i + 1` (0..4) on each
    /// axis. `k` is 4 and `up` is 1; the weight is `[cout][cin][4][4]`.
    Deconv(Conv),
    /// `ln((v * scale + offset)² + 1/255)` per channel, `offset` per lane
    /// of each group of four: HDRCNN's map of an LDR encoder feature (or
    /// the input picture) into the log-HDR domain of its decoder.
    LogSkip { src: Slice, dst: Slice, scale: f32, offset: [f32; 4] },
}

/// What goes in and comes out of a tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Io {
    /// A super-resolution net: the input sRGB-encoded and clamped, the
    /// excess carried around it (`gpu::Colour`).
    #[default]
    Upscale,
    /// HDRCNN: the input divided by the clip level, clamped to 0..1 and
    /// gamma-2 encoded (the method linearises with `x²`); the output is
    /// the method's own blend — the net's log prediction (`exp(y) - 1/255`,
    /// times the clip) only where the input nears the clip, the input
    /// itself, untouched, everywhere else.
    Hdr,
}

/// Where the upscaled picture is read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Out {
    /// RGB in the first three channels of a slice at the net's scale.
    Direct(Slice),
    /// A pixel shuffle by the net's scale over `3 * scale²` channels at the
    /// tile's own size (torch's `PixelShuffle` channel order).
    Shuffle(Slice),
}

/// A whole network, for any tile size.
#[derive(Clone, Debug, PartialEq)]
pub struct Net {
    pub bufs: Vec<Buf>,
    /// Where the tile's input lands: rgb in channels 0..3, alpha in 3 (a
    /// first conv's weights for it are zero).
    pub input: Slice,
    /// Input normalisation: `(x - mean) * range`.
    pub mean: [f32; 3],
    pub range: f32,
    pub ops: Vec<Op>,
    pub out: Out,
    /// Add the input, nearest-upsampled, to the output (Real-ESRGAN's
    /// compact net learns the residual over it).
    pub base: bool,
    pub scale: u32,
    /// How far a pixel sees, in input pixels — the halo a tile carries.
    pub halo: u32,
    pub io: Io,
}

impl Net {
    /// Feature memory for one tile of side `t` (input pixels), in values.
    pub fn values_per_tile(&self, t: u32) -> u64 {
        self.bufs.iter().map(|b| b.ch as u64 * (b.side(t) as u64).pow(2)).sum()
    }

    /// The grid tiles are cut on: every buffer's size must be whole, and
    /// a pool must pair the same pixels in a tile as in the picture.
    pub fn align(&self) -> u32 {
        self.bufs.iter().map(|b| b.div).max().unwrap_or(1).max(1)
    }

    /// Multiply-adds per INPUT pixel: what a frame costs.
    pub fn macs_per_pixel(&self) -> u64 {
        self.ops
            .iter()
            .map(|op| match op {
                Op::Conv(c) => {
                    let b = &self.bufs[c.dst.buf];
                    c.cin as u64 * c.cout as u64 * (c.k * c.k) as u64 * (b.res * b.res) as u64
                        / (b.div * b.div) as u64
                }
                // Each output pixel sees a 2x2 of the input.
                Op::Deconv(c) => {
                    let b = &self.bufs[c.dst.buf];
                    c.cin as u64 * c.cout as u64 * 4 * (b.res * b.res) as u64 / (b.div * b.div) as u64
                }
                Op::Copy { .. } | Op::Pool { .. } | Op::LogSkip { .. } => 0,
            })
            .sum()
    }

    /// How far an output pixel sees, in input pixels: the radius of every
    /// conv on the way, at the resolution it runs at.
    pub fn receptive(&self) -> u32 {
        // One pixel of a buffer, in input pixels.
        let px = |buf: usize| self.bufs[buf].div as f32 / self.bufs[buf].res as f32;
        let r: f32 = self
            .ops
            .iter()
            .map(|op| match op {
                Op::Conv(c) => (c.k / 2) as f32 * px(c.dst.buf),
                Op::Pool { src, .. } => px(src.buf),
                Op::Deconv(c) => px(c.src.buf),
                Op::Copy { .. } | Op::LogSkip { .. } => 0.0,
            })
            .sum();
        r.ceil() as u32
    }

    /// Parameters, for the panel.
    pub fn params(&self) -> usize {
        self.ops
            .iter()
            .map(|op| match op {
                Op::Conv(c) | Op::Deconv(c) => c.weight.len() + c.bias.len(),
                Op::Copy { .. } | Op::Pool { .. } | Op::LogSkip { .. } => 0,
            })
            .sum()
    }
}

/// Round a channel count up to whole groups of four.
pub fn pad4(n: u32) -> u32 {
    n.div_ceil(4) * 4
}
