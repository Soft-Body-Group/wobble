//! RIFE v4.6 IFNet, implemented natively in WGSL compute: the network
//! behind the Flow node's motion vectors.
//!
//! A SLOW PATH, and a visible one (manifesto rule 8): this never runs
//! inside the live fused graph. The Flow node runs it on a click, over its
//! input's frame range, and writes the vectors to disk; everything
//! downstream reads those files like any plate.
//!
//! The network is hzwer's IFNet as shipped with RIFE v4.6 weights (MIT):
//! four coarse-to-fine IFBlocks (c = 192/128/96/64 at scales 8/4/2/1), each
//! a 2-conv downsampler, eight ResConv residual blocks and a
//! ConvTranspose+PixelShuffle head emitting a 4-channel intermediate flow
//! (F_t→0.xy, F_t→1.xy, in pixels) plus a fusion-mask logit. Weights load
//! straight from the official `flownet.pkl` (a torch zip, read by
//! [`crate::torchpt`] — no Python anywhere), fetched through the model
//! catalog as `rife-4.6`. The `.wifw` container survives for the golden
//! bundle (`tools/rife_convert.py --golden`, real torch outputs) and the
//! tests. Activations are per-channel-slope PReLU in the container;
//! LeakyReLU(0.2) — what v4.6 actually uses — is the constant-slope special
//! case.
//!
//! `reference` is a CPU mirror used by tests; `gpu` is the real
//! implementation. Both share this module's model structs and the exact
//! PyTorch semantics: `F.interpolate(align_corners=False)` resizes,
//! `grid_sample(padding_mode='border', align_corners=True)` warps.

pub mod gpu;
pub mod reference;

use anyhow::{bail, Context as _, Result};
use std::collections::HashMap;
use std::path::Path;

/// Coarse-to-fine pyramid scales, matching `scale_list=[8, 4, 2, 1]`.
pub const SCALES: [usize; 4] = [8, 4, 2, 1];
/// Feature width `c` per block.
pub const BLOCK_C: [usize; 4] = [192, 128, 96, 64];
/// Input planes per block: block0 = img0+img1+t (7); later blocks =
/// warped0+warped1+t+mask (8) with the downsampled 4-ch flow concatenated.
pub const BLOCK_IN: [usize; 4] = [7, 12, 12, 12];
/// Channels out of the ConvTranspose+PixelShuffle head: 4 flow + 1 mask + 1 unused.
pub const HEAD_C: usize = 6;

/// Dense row-major tensor, PyTorch dim order.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }
}

/// Conv3x3 + per-channel-slope activation (LeakyReLU 0.2 == constant slope).
pub struct ConvLayer {
    /// `[out_c, in_c, 3, 3]`
    pub w: Tensor,
    /// `[out_c]`
    pub b: Tensor,
    /// Negative-side slope per out channel.
    pub slope: Vec<f32>,
}

/// `relu(conv3x3(x) * beta + x)` — the v4.6 residual block.
pub struct ResConvLayer {
    /// `[c, c, 3, 3]`
    pub w: Tensor,
    /// `[c]`
    pub b: Tensor,
    /// Learned per-channel residual gain.
    pub beta: Vec<f32>,
    /// Activation slope per channel (0.2 for the stock LeakyReLU).
    pub slope: Vec<f32>,
}

/// ConvTranspose2d(c, 24, 4, stride 2, pad 1) + PixelShuffle(2). No activation.
pub struct DeconvLayer {
    /// `[in_c, 24, 4, 4]` — note ConvTranspose weight order is `[in, out, kh, kw]`.
    pub w: Tensor,
    /// `[24]`
    pub b: Tensor,
}

pub struct IfBlock {
    pub c: usize,
    pub in_planes: usize,
    pub conv0: [ConvLayer; 2],
    pub convblock: Vec<ResConvLayer>,
    pub lastconv: DeconvLayer,
}

pub struct RifeModel {
    pub blocks: Vec<IfBlock>,
}

/// Everything one inference produces, cropped back to the caller's size.
/// All planes are `[c][h][w]` f32.
pub struct RifeOutput {
    /// Net-native interpolated frame (inputs clamped to 0..1 as RIFE trains on).
    pub merged: Vec<f32>,
    /// Same flow + mask applied to the *unclamped* inputs — the HDR-honest blend.
    pub merged_hdr: Vec<f32>,
    /// 4 channels: F_t→0.xy then F_t→1.xy, in output pixels.
    pub flow: Vec<f32>,
    /// Fusion mask after sigmoid: weight of the img0 warp.
    pub mask: Vec<f32>,
}

// ---------------------------------------------------------------------------
// .wifw container
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 4] = b"WIFW";
const VERSION: u32 = 1;
/// The one architecture this build understands (RIFE v4.6-family IFNet).
const ARCH_IFNET_46: u32 = 1;

/// Parse a `.wifw` file into named tensors.
pub fn read_wifw(path: &Path) -> Result<HashMap<String, Tensor>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    parse_wifw(&bytes).with_context(|| format!("parse {}", path.display()))
}

pub fn parse_wifw(bytes: &[u8]) -> Result<HashMap<String, Tensor>> {
    struct Cur<'a> {
        b: &'a [u8],
        pos: usize,
    }
    impl<'a> Cur<'a> {
        fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
            if self.b.len() - self.pos < n {
                bail!("truncated wifw");
            }
            let out = &self.b[self.pos..self.pos + n];
            self.pos += n;
            Ok(out)
        }
        fn u32(&mut self) -> Result<u32> {
            let b = self.bytes(4)?;
            Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        }
    }
    let mut r = Cur { b: bytes, pos: 0 };
    if r.bytes(4)? != MAGIC {
        bail!("not a .wifw file (bad magic)");
    }
    let version = r.u32()?;
    if version != VERSION {
        bail!("wifw version {version}, this build reads {VERSION}");
    }
    let arch = r.u32()?;
    if arch != ARCH_IFNET_46 {
        bail!("wifw arch {arch}, this build implements IFNet v4.6 (arch {ARCH_IFNET_46})");
    }
    let _activation_hint = r.u32()?; // informational; slopes are explicit tensors
    let count = r.u32()? as usize;
    let mut out = HashMap::with_capacity(count);
    for _ in 0..count {
        let name_len = r.u32()? as usize;
        if name_len > 512 {
            bail!("implausible tensor name length {name_len}");
        }
        let name = String::from_utf8(r.bytes(name_len)?.to_vec()).context("tensor name not utf8")?;
        let ndim = r.u32()? as usize;
        if ndim > 8 {
            bail!("{name}: implausible ndim {ndim}");
        }
        let mut dims = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            dims.push(r.u32()? as usize);
        }
        let n: usize = dims.iter().product();
        if n > 64 << 20 {
            bail!("{name}: implausible element count {n}");
        }
        let raw = r.bytes(n * 4).with_context(|| format!("{name}: truncated data"))?;
        let data = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        out.insert(name, Tensor { dims, data });
    }
    Ok(out)
}

/// Serialize tensors into the `.wifw` container (converter parity; tests).
pub fn write_wifw(tensors: &[(String, Tensor)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    for v in [VERSION, ARCH_IFNET_46, 0, tensors.len() as u32] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for (name, t) in tensors {
        out.extend_from_slice(&(name.len() as u32).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            out.extend_from_slice(&(*d as u32).to_le_bytes());
        }
        for v in &t.data {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Model assembly
// ---------------------------------------------------------------------------

impl RifeModel {
    /// Load from a `.wifw` container (the golden bundle's format).
    pub fn load(path: &Path) -> Result<Self> {
        Self::from_tensors(read_wifw(path)?)
    }

    /// Load the official `flownet.pkl` — what the catalog downloads. A
    /// state dict saved from `DataParallel` names every key `module.…`;
    /// the teacher block (`block_tea`) is training-only and is ignored.
    pub fn load_pkl(path: &Path) -> Result<Self> {
        let sd = crate::torchpt::read(path)?;
        let t = sd
            .into_iter()
            .map(|(k, v)| {
                let k = k.strip_prefix("module.").map(str::to_string).unwrap_or(k);
                (k, Tensor { dims: v.dims, data: v.data })
            })
            .collect();
        Self::from_tensors(t).context("not a RIFE v4.6-family IFNet checkpoint")
    }

    /// Assemble and shape-check the four IFBlocks from named tensors.
    pub fn from_tensors(mut t: HashMap<String, Tensor>) -> Result<Self> {
        let mut blocks = Vec::with_capacity(4);
        for i in 0..4 {
            let c = BLOCK_C[i];
            let in_planes = BLOCK_IN[i];
            let conv0 = [
                conv_layer(&mut t, &format!("block{i}.conv0.0"), in_planes, c / 2)?,
                conv_layer(&mut t, &format!("block{i}.conv0.1"), c / 2, c)?,
            ];
            let mut convblock = Vec::with_capacity(8);
            for j in 0..8 {
                convblock.push(resconv_layer(&mut t, &format!("block{i}.convblock.{j}"), c)?);
            }
            let key = format!("block{i}.lastconv.0");
            let w = take(&mut t, &format!("{key}.weight"), &[c, 4 * HEAD_C, 4, 4])?;
            let b = take(&mut t, &format!("{key}.bias"), &[4 * HEAD_C])?;
            blocks.push(IfBlock {
                c,
                in_planes,
                conv0,
                convblock,
                lastconv: DeconvLayer { w, b },
            });
        }
        Ok(Self { blocks })
    }

    /// Deterministic small-magnitude random weights — test scaffolding, so the
    /// CPU/GPU parity suite runs without the real (converted) model on disk.
    pub fn random(seed: u64) -> Self {
        let mut rng = Lcg(seed | 1);
        fn conv(rng: &mut Lcg, in_c: usize, out_c: usize) -> ConvLayer {
            ConvLayer {
                w: rng.tensor(vec![out_c, in_c, 3, 3], 0.5 / (in_c as f32 * 9.0).sqrt()),
                b: rng.tensor(vec![out_c], 0.05),
                slope: (0..out_c).map(|_| 0.1 + rng.uniform() * 0.3).collect(),
            }
        }
        let mut blocks = Vec::with_capacity(4);
        for i in 0..4 {
            let c = BLOCK_C[i];
            let conv0 = [conv(&mut rng, BLOCK_IN[i], c / 2), conv(&mut rng, c / 2, c)];
            let mut convblock = Vec::with_capacity(8);
            for _ in 0..8 {
                convblock.push(ResConvLayer {
                    w: rng.tensor(vec![c, c, 3, 3], 0.5 / (c as f32 * 9.0).sqrt()),
                    b: rng.tensor(vec![c], 0.05),
                    beta: (0..c).map(|_| 0.5 + rng.uniform()).collect(),
                    slope: vec![0.2; c],
                });
            }
            let lastconv = DeconvLayer {
                w: rng.tensor(vec![c, 4 * HEAD_C, 4, 4], 0.5 / (c as f32 * 16.0).sqrt()),
                b: rng.tensor(vec![4 * HEAD_C], 0.02),
            };
            blocks.push(IfBlock { c, in_planes: BLOCK_IN[i], conv0, convblock, lastconv });
        }
        Self { blocks }
    }

    /// Flatten back to named tensors (inverse of `from_tensors`; tests).
    pub fn to_tensors(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        let vec1 = |v: &[f32]| Tensor { dims: vec![v.len()], data: v.to_vec() };
        for (i, blk) in self.blocks.iter().enumerate() {
            for (k, l) in blk.conv0.iter().enumerate() {
                out.push((format!("block{i}.conv0.{k}.0.weight"), l.w.clone()));
                out.push((format!("block{i}.conv0.{k}.0.bias"), l.b.clone()));
                out.push((format!("block{i}.conv0.{k}.1.weight"), vec1(&l.slope)));
            }
            for (j, l) in blk.convblock.iter().enumerate() {
                out.push((format!("block{i}.convblock.{j}.conv.weight"), l.w.clone()));
                out.push((format!("block{i}.convblock.{j}.conv.bias"), l.b.clone()));
                out.push((format!("block{i}.convblock.{j}.beta"), vec1(&l.beta)));
                out.push((format!("block{i}.convblock.{j}.relu.weight"), vec1(&l.slope)));
            }
            out.push((format!("block{i}.lastconv.0.weight"), blk.lastconv.w.clone()));
            out.push((format!("block{i}.lastconv.0.bias"), blk.lastconv.b.clone()));
        }
        out
    }
}

fn take(t: &mut HashMap<String, Tensor>, key: &str, dims: &[usize]) -> Result<Tensor> {
    let v = t.remove(key).with_context(|| format!("missing tensor {key}"))?;
    if v.dims != dims {
        bail!("{key}: shape {:?}, expected {:?}", v.dims, dims);
    }
    Ok(v)
}

/// Optional per-channel activation slope: PReLU weight if the pkl had one
/// (`.1.weight` under a Sequential conv, or `.relu.weight` in a ResConv),
/// broadcast if scalar; LeakyReLU 0.2 otherwise.
fn slope_of(t: &mut HashMap<String, Tensor>, key: &str, c: usize) -> Result<Vec<f32>> {
    match t.remove(key) {
        None => Ok(vec![0.2; c]),
        Some(s) => {
            let n = s.numel();
            if n == 1 {
                Ok(vec![s.data[0]; c])
            } else if n == c {
                Ok(s.data)
            } else {
                bail!("{key}: {n} slopes for {c} channels");
            }
        }
    }
}

fn conv_layer(t: &mut HashMap<String, Tensor>, key: &str, in_c: usize, out_c: usize) -> Result<ConvLayer> {
    Ok(ConvLayer {
        w: take(t, &format!("{key}.0.weight"), &[out_c, in_c, 3, 3])?,
        b: take(t, &format!("{key}.0.bias"), &[out_c])?,
        slope: slope_of(t, &format!("{key}.1.weight"), out_c)?,
    })
}

fn resconv_layer(t: &mut HashMap<String, Tensor>, key: &str, c: usize) -> Result<ResConvLayer> {
    let beta = t
        .remove(&format!("{key}.beta"))
        .with_context(|| format!("missing tensor {key}.beta"))?;
    if beta.numel() != c {
        bail!("{key}.beta: {} elements for {c} channels", beta.numel());
    }
    Ok(ResConvLayer {
        w: take(t, &format!("{key}.conv.weight"), &[c, c, 3, 3])?,
        b: take(t, &format!("{key}.conv.bias"), &[c])?,
        beta: beta.data,
        slope: slope_of(t, &format!("{key}.relu.weight"), c)?,
    })
}

// ---------------------------------------------------------------------------
// Shared plumbing
// ---------------------------------------------------------------------------

/// IFNet needs dims divisible by 32 (scale-8 block then /4 conv stack).
pub fn pad32(x: usize) -> usize {
    x.div_ceil(32) * 32
}

/// Edge-replicate a `[c][h][w]` plane stack out to `[c][ph][pw]` (right/bottom).
pub fn pad_planes(src: &[f32], c: usize, w: usize, h: usize, pw: usize, ph: usize) -> Vec<f32> {
    assert_eq!(src.len(), c * w * h);
    let mut out = vec![0f32; c * pw * ph];
    for ch in 0..c {
        for y in 0..ph {
            let sy = y.min(h - 1);
            let srow = &src[ch * w * h + sy * w..][..w];
            let drow = &mut out[ch * pw * ph + y * pw..][..pw];
            drow[..w].copy_from_slice(srow);
            for x in w..pw {
                drow[x] = srow[w - 1];
            }
        }
    }
    out
}

/// Crop `[c][ph][pw]` back to `[c][h][w]`.
pub fn crop_planes(src: &[f32], c: usize, pw: usize, ph: usize, w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; c * w * h];
    for ch in 0..c {
        for y in 0..h {
            out[ch * w * h + y * w..][..w]
                .copy_from_slice(&src[ch * pw * ph + y * pw..][..w]);
        }
    }
    out
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Splittable LCG for deterministic test data.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1 << 24) as f32
    }
    /// Uniform in [-s, s].
    pub fn signed(&mut self, s: f32) -> f32 {
        (self.uniform() * 2.0 - 1.0) * s
    }
    pub fn tensor(&mut self, dims: Vec<usize>, s: f32) -> Tensor {
        let n: usize = dims.iter().product();
        Tensor { dims, data: (0..n).map(|_| self.signed(s)).collect() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wifw_roundtrip_and_assembly() {
        let model = RifeModel::random(7);
        let bytes = write_wifw(&model.to_tensors());
        let parsed = parse_wifw(&bytes).unwrap();
        let rebuilt = RifeModel::from_tensors(parsed).unwrap();
        for (a, b) in model.blocks.iter().zip(rebuilt.blocks.iter()) {
            assert_eq!(a.conv0[0].w, b.conv0[0].w);
            assert_eq!(a.conv0[1].slope, b.conv0[1].slope);
            assert_eq!(a.convblock[3].beta, b.convblock[3].beta);
            assert_eq!(a.lastconv.w, b.lastconv.w);
        }
    }

    /// The official pickle, read in Rust, is the same model the python
    /// converter wrote — tensor for tensor. Self-skips without the files
    /// (`data/` is not in the repo).
    #[test]
    fn the_pkl_reads_as_the_converted_model() {
        let pkl = Path::new("../data/rife/flownet.pkl");
        let wifw = Path::new("../data/rife/rife46.wifw");
        if !pkl.exists() || !wifw.exists() {
            eprintln!("SKIP: no data/rife/flownet.pkl + rife46.wifw");
            return;
        }
        let a = RifeModel::load_pkl(pkl).unwrap();
        let b = RifeModel::load(wifw).unwrap();
        for (x, y) in a.blocks.iter().zip(&b.blocks) {
            assert_eq!(x.conv0[0].w, y.conv0[0].w);
            assert_eq!(x.conv0[1].slope, y.conv0[1].slope);
            for (p, q) in x.convblock.iter().zip(&y.convblock) {
                assert_eq!(p.w, q.w);
                assert_eq!(p.beta, q.beta);
            }
            assert_eq!(x.lastconv.w, y.lastconv.w);
            assert_eq!(x.lastconv.b, y.lastconv.b);
        }
    }

    #[test]
    fn wifw_missing_key_is_named() {
        let model = RifeModel::random(3);
        let mut tensors = model.to_tensors();
        tensors.retain(|(k, _)| k != "block2.convblock.5.beta");
        let err = match RifeModel::from_tensors(parse_wifw(&write_wifw(&tensors)).unwrap()) {
            Ok(_) => panic!("assembled despite missing tensor"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("block2.convblock.5.beta"), "{err}");
    }

    #[test]
    #[allow(clippy::erasing_op, clippy::identity_op)] // readable index math
    fn pad_crop_roundtrip() {
        let w = 5;
        let h = 3;
        let src: Vec<f32> = (0..2 * w * h).map(|i| i as f32).collect();
        let padded = pad_planes(&src, 2, w, h, 32, 32);
        // replicated edge
        assert_eq!(padded[0 * 32 * 32 + 0 * 32 + 31], src[w - 1]);
        assert_eq!(padded[0 * 32 * 32 + 31 * 32 + 0], src[(h - 1) * w]);
        assert_eq!(crop_planes(&padded, 2, 32, 32, w, h), src);
    }
}
