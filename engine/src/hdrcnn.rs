//! HDRCNN (Eilertsen et al., "HDR image reconstruction from a single
//! exposure using deep CNNs", SIGGRAPH Asia 2017; BSD-3,
//! github.com/gabrieleilertsen/hdrcnn) as an [`nn::Net`], built straight
//! from the published TensorLayer `.npz` — no TensorFlow, no conversion.
//!
//! The upstream graph (`network.py`), op for op:
//!
//! - **Encoder**: VGG16's thirteen 3x3 convs (ReLU) and five 2x2 max pools,
//!   fed `255 x - mean` in BGR. Here the input stays RGB and the first
//!   conv's input channels are reversed instead; the mean is the load's.
//! - **h6**: a 3x3 conv at 1/32 with BatchNorm and ReLU.
//! - **Decoder**: five 4x4 stride-2 transposed convs (BatchNorm, ReLU),
//!   each after a skip fusion: the decoder's channels concatenated with the
//!   encoder's matching pre-pool features mapped into its log domain
//!   (`ln((f / 255)² + 1/255)`), then a 1x1 conv. Then a 1x1 conv to rgb
//!   (BatchNorm, ReLU) and a last fusion with the input picture itself
//!   (`ln(x² + 1/255)`). Out comes `y`, log HDR.
//! - **The blend** (`get_final`): `(1 - a) x² + a (exp(y) - 1/255)`, `a`
//!   ramping from 0 at 0.95 to 1 at 1 in the input's brightest channel —
//!   so a pixel far from the clip is the input. `nn::gpu`'s HDR emit does
//!   it, and stores such a pixel exactly as it came.
//!
//! BatchNorm is folded into the conv before it (TensorLayer's inference
//! form, epsilon 1e-5; every variance is far above it). The npz holds the
//! 80 parameters in creation order — conv `W, b`, BatchNorm `beta, gamma,
//! moving_mean, moving_variance` — which is how they are read: by position,
//! each shape checked.
//!
//! Colour: the net was trained on display-referred pictures and linearises
//! its input with `x²`. It is given the plate divided by a clip level,
//! clamped to 0..1, and encoded with exactly that curve (`sqrt`), so its
//! linearisation inverts ours and the blend's `x²` is the plate. The
//! prediction is multiplied back by the clip level.
//!
//! The bottleneck's BatchNorm statistics say its input is ~1e9 before
//! normalising: far past f16, so this net always runs in f32 storage.

use crate::nn::{pad4, Buf, Conv, Epi, Io, Net, Op, Out, Slice};
use crate::npy::{Checkpoint, Tensor};
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Which trained parameters. The setup stores the variant name; the
/// catalog id is what is downloaded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Model {
    /// `hdrcnn_params_compr_regularized.npz`: trained on JPEG-compressed
    /// pictures with the CVPR 2019 temporal regularisation — the one for
    /// video (steadier frame to frame) and compressed sources.
    #[default]
    Regularized,
    /// `hdrcnn_params.npz`: the paper's parameters, a little more detail on
    /// clean stills.
    Original,
}

impl Model {
    pub const ALL: [Model; 2] = [Model::Regularized, Model::Original];

    /// The catalog id of its weights.
    pub fn model_id(self) -> &'static str {
        match self {
            Model::Regularized => "hdrcnn-compr-regularized",
            Model::Original => "hdrcnn",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Model::Regularized => "video (temporally steady)",
            Model::Original => "stills (original)",
        }
    }
}

/// VGG16's channel means, RGB, in 0..255.
pub const VGG_MEAN_RGB: [f32; 3] = [123.68, 116.779, 103.939];

/// TensorLayer's BatchNormLayer epsilon.
const BN_EPS: f32 = 1e-5;

/// How far a tile's halo reaches, input pixels. The net's receptive field
/// is hundreds of pixels (VGG16 to pool5); what an output pixel actually
/// depends on falls off far sooner, and past this the seams are below what
/// the blend shows (`engine/tests/hdrcnn.rs` measures it).
pub const HDR_HALO: u32 = 96;

/// The parameters by position, shapes checked as they are taken.
struct Params<'a> {
    ck: &'a Checkpoint,
    at: usize,
}

impl<'a> Params<'a> {
    fn take(&mut self, dims: &[usize]) -> Result<&'a Tensor> {
        let name = format!("/{}", self.at);
        self.at += 1;
        self.ck.get_shaped(&name, dims).with_context(|| format!("parameter {} (expected {dims:?})", self.at - 1))
    }

    /// A BatchNorm's `beta, gamma, mean, variance` as the scale and shift
    /// it applies.
    fn bn(&mut self, n: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let beta = self.take(&[n])?.data.clone();
        let gamma = self.take(&[n])?.data.clone();
        let mean = self.take(&[n])?.data.clone();
        let var = self.take(&[n])?.data.clone();
        let scale: Vec<f32> = (0..n).map(|i| gamma[i] / (var[i] + BN_EPS).sqrt()).collect();
        let shift: Vec<f32> = (0..n).map(|i| beta[i] - mean[i] * scale[i]).collect();
        Ok((scale, shift))
    }
}

/// A TF conv weight `[kh][kw][cin][cout]` as ours, `[cout][cin][kh][kw]`,
/// with input channel `i` of ours read from `map(i)` of theirs (None: a
/// padding lane, weight 0) and `cin` channels on our side.
fn tf_conv(w: &Tensor, cin: usize, map: &dyn Fn(usize) -> Option<usize>) -> Vec<f32> {
    let (kh, kw, tcin, cout) = (w.dims[0], w.dims[1], w.dims[2], w.dims[3]);
    let mut out = vec![0.0f32; cout * cin * kh * kw];
    for o in 0..cout {
        for i in 0..cin {
            let Some(ti) = map(i) else { continue };
            assert!(ti < tcin);
            for y in 0..kh {
                for x in 0..kw {
                    out[((o * cin + i) * kh + y) * kw + x] = w.data[((y * kw + x) * tcin + ti) * cout + o];
                }
            }
        }
    }
    out
}

/// Fold a per-output scale and shift into a conv's weight and bias.
fn fold(weight: &mut [f32], bias: &mut [f32], scale: &[f32], shift: &[f32]) {
    let per = weight.len() / bias.len();
    for (o, b) in bias.iter_mut().enumerate() {
        for w in &mut weight[o * per..(o + 1) * per] {
            *w *= scale[o];
        }
        *b = *b * scale[o] + shift[o];
    }
}

/// Build the net from the npz's parameter list.
pub fn build(ck: &Checkpoint) -> Result<Net> {
    if ck.tensors.len() != 80 {
        bail!("an HDRCNN checkpoint holds 80 parameters; this one {}", ck.tensors.len());
    }
    let mut p = Params { ck, at: 0 };
    let mut bufs: Vec<Buf> = Vec::new();
    let mut buf = |ch: u32, div: u32| {
        bufs.push(Buf { ch: pad4(ch), res: 1, div });
        bufs.len() - 1
    };
    let relu = |n: u32| Epi::leaky(n, 0.0);
    let mut ops: Vec<Op> = Vec::new();
    let all = |b: usize, n: u32| Slice::new(b, 0, pad4(n));

    // --- the encoder: VGG16 ------------------------------------------------
    let input = buf(4, 1);
    // (channels out, convs) per level; the last conv's output is the skip.
    let levels: [(u32, usize); 5] = [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)];
    let mut src = all(input, 4);
    let mut cin = 3u32;
    let mut skips = Vec::new();
    let mut div = 1u32;
    for (li, (ch, n)) in levels.into_iter().enumerate() {
        let a = buf(ch, div);
        let b = buf(ch, div);
        let skip = buf(ch, div);
        for k in 0..n {
            let w = p.take(&[3, 3, cin as usize, ch as usize])?;
            let bias = p.take(&[ch as usize])?.data.clone();
            let first = li == 0 && k == 0;
            let our_cin = if first { 4 } else { cin as usize };
            // VGG reads BGR: our RGB lane c is its channel 2 - c.
            let weight = if first {
                tf_conv(w, 4, &|i| (i < 3).then(|| 2 - i))
            } else {
                tf_conv(w, our_cin, &Some)
            };
            let dst = if k + 1 == n {
                all(skip, ch)
            } else if k % 2 == 0 {
                all(a, ch)
            } else {
                all(b, ch)
            };
            ops.push(Op::Conv(Conv {
                k: 3,
                cin: our_cin as u32,
                cout: ch,
                weight: weight.into(),
                bias: bias.into(),
                src,
                dst,
                up: 1,
                silu_in: false,
                epi: relu(ch),
            }));
            src = dst;
            cin = ch;
        }
        skips.push(all(skip, ch));
        div *= 2;
        let pooled = buf(ch, div);
        ops.push(Op::Pool { src, dst: all(pooled, ch) });
        src = all(pooled, ch);
    }

    // --- h6: 3x3 at 1/32, BatchNorm, ReLU -----------------------------------
    let h6 = buf(512, 32);
    {
        let w = p.take(&[3, 3, 512, 512])?;
        let mut bias = p.take(&[512])?.data.clone();
        let mut weight = tf_conv(w, 512, &Some);
        let (sc, sh) = p.bn(512)?;
        fold(&mut weight, &mut bias, &sc, &sh);
        ops.push(Op::Conv(Conv {
            k: 3,
            cin: 512,
            cout: 512,
            weight: weight.into(),
            bias: bias.into(),
            src,
            dst: all(h6, 512),
            up: 1,
            silu_in: false,
            epi: relu(512),
        }));
    }
    let mut cur = all(h6, 512);

    // --- the decoder -------------------------------------------------------
    // (deconv out channels, the resolution it lands at) for h1..h5.
    let ups: [(u32, u32); 5] = [(512, 16), (512, 8), (256, 4), (128, 2), (64, 1)];
    let mut cin = 512u32;
    for (i, (cout, at)) in ups.into_iter().enumerate() {
        if i > 0 {
            // Fuse the previous deconv's output (in `cur`, the first half
            // of its fusion buffer) with the skip at its size.
            cur = fuse(&mut p, &mut ops, &mut buf, cur, skips[5 - i], cin, at * 2)?;
        }
        let f = buf(2 * cout, at);
        deconv(&mut p, &mut ops, cur, cin, cout, Slice::new(f, 0, cout))?;
        cur = Slice::new(f, 0, cout);
        cin = cout;
    }
    // h6's fusion: beforepool1 at full size.
    cur = fuse(&mut p, &mut ops, &mut buf, cur, skips[0], 64, 1)?;

    // --- h7: 1x1 to rgb, BatchNorm, ReLU, fused with the picture -----------
    let f0 = buf(8, 1);
    {
        let w = p.take(&[1, 1, 64, 3])?;
        let mut bias = p.take(&[3])?.data.clone();
        let mut weight = tf_conv(w, 64, &Some);
        let (sc, sh) = p.bn(3)?;
        fold(&mut weight, &mut bias, &sc, &sh);
        ops.push(Op::Conv(Conv {
            k: 1,
            cin: 64,
            cout: 3,
            weight: weight.into(),
            bias: bias.into(),
            src: cur,
            dst: Slice::new(f0, 0, 4),
            up: 1,
            silu_in: false,
            epi: relu(3),
        }));
    }
    // The picture's own log map: the input slice holds `255 x - mean`.
    let m = VGG_MEAN_RGB.map(|v| v / 255.0);
    ops.push(Op::LogSkip {
        src: Slice::new(input, 0, 4),
        dst: Slice::new(f0, 4, 4),
        scale: 1.0 / 255.0,
        offset: [m[0], m[1], m[2], 0.0],
    });
    let out = buf(4, 1);
    {
        let w = p.take(&[1, 1, 6, 3])?;
        let bias = p.take(&[3])?.data.clone();
        // Lanes 0..3 the decoder, 4..7 the picture; 3 and 7 are padding.
        let weight = tf_conv(w, 8, &|i| match i {
            0..=2 => Some(i),
            4..=6 => Some(i - 1),
            _ => None,
        });
        ops.push(Op::Conv(Conv {
            k: 1,
            cin: 8,
            cout: 3,
            weight: weight.into(),
            bias: bias.into(),
            src: Slice::new(f0, 0, 8),
            dst: Slice::new(out, 0, 4),
            up: 1,
            silu_in: false,
            epi: Epi::none(),
        }));
    }
    if p.at != 80 {
        bail!("read {} of the checkpoint's 80 parameters", p.at);
    }
    let net = Net {
        bufs,
        input: Slice::new(input, 0, 4),
        mean: m,
        range: 255.0,
        ops,
        out: Out::Direct(Slice::new(out, 0, 4)),
        base: false,
        scale: 1,
        halo: 0,
        io: Io::Hdr,
    };
    let halo = net.receptive().min(HDR_HALO);
    Ok(Net { halo, ..net })
}

/// A 4x4 stride-2 transposed conv with its BatchNorm, ReLU, into `dst`.
fn deconv(p: &mut Params, ops: &mut Vec<Op>, src: Slice, cin: u32, cout: u32, dst: Slice) -> Result<()> {
    let w = p.take(&[4, 4, cout as usize, cin as usize])?;
    let mut bias = p.take(&[cout as usize])?.data.clone();
    // TF's [kh][kw][out][in] -> [out][in][kh][kw].
    let (ci, co) = (cin as usize, cout as usize);
    let mut weight = vec![0.0f32; co * ci * 16];
    for o in 0..co {
        for i in 0..ci {
            for t in 0..16 {
                weight[(o * ci + i) * 16 + t] = w.data[(t * co + o) * ci + i];
            }
        }
    }
    let (sc, sh) = p.bn(co)?;
    fold(&mut weight, &mut bias, &sc, &sh);
    ops.push(Op::Deconv(Conv {
        k: 4,
        cin,
        cout,
        weight: weight.into(),
        bias: bias.into(),
        src,
        dst,
        up: 1,
        silu_in: false,
        epi: Epi::leaky(cout, 0.0),
    }));
    Ok(())
}

/// One skip fusion: `skip`'s log map into the second half of `dec`'s
/// buffer, then a 1x1 conv over both into a fresh buffer of `ch` at the
/// same level (`div`).
fn fuse(
    p: &mut Params,
    ops: &mut Vec<Op>,
    buf: &mut dyn FnMut(u32, u32) -> usize,
    dec: Slice,
    skip: Slice,
    ch: u32,
    div: u32,
) -> Result<Slice> {
    ops.push(Op::LogSkip { src: skip, dst: Slice::new(dec.buf, ch, ch), scale: 1.0 / 255.0, offset: [0.0; 4] });
    let w = p.take(&[1, 1, 2 * ch as usize, ch as usize])?;
    let bias = p.take(&[ch as usize])?.data.clone();
    let weight = tf_conv(w, 2 * ch as usize, &Some);
    let out = buf(ch, div);
    ops.push(Op::Conv(Conv {
        k: 1,
        cin: 2 * ch,
        cout: ch,
        weight: weight.into(),
        bias: bias.into(),
        src: Slice::new(dec.buf, 0, 2 * ch),
        dst: Slice::new(out, 0, ch),
        up: 1,
        silu_in: false,
        epi: Epi::none(),
    }));
    Ok(Slice::new(out, 0, ch))
}

/// The model's net, parsed once per process and shared: `None` until its
/// weights are downloaded (or when they will not parse, which the panel
/// reports through [`load_error`]).
pub fn net(model: Model) -> Option<Arc<Net>> {
    static CACHE: OnceLock<Mutex<HashMap<Model, Result<Arc<Net>, String>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(r) = cache.lock().unwrap().get(&model) {
        return r.as_ref().ok().cloned();
    }
    let path = crate::models::ready_path(model.model_id()).ok()?;
    let built = crate::npy::read_npz(&path, "params.npy")
        .and_then(|ck| build(&ck))
        .map(Arc::new)
        .map_err(|e| format!("{e:#}"));
    if let Err(e) = &built {
        log::warn!("hdrcnn: {} will not load: {e}", model.label());
    }
    let out = built.as_ref().ok().cloned();
    cache.lock().unwrap().insert(model, built);
    out
}

/// Why a downloaded model would not build, if it would not.
pub fn load_error(model: Model) -> Option<String> {
    net(model).is_none().then(|| crate::models::ready_path(model.model_id()).err().map(|e| e.to_string()))?
}

// ------------------------------------------------------------ the run

/// What to reconstruct.
pub struct Job {
    pub model: Model,
    /// The rendered input frames (linear Rec.709, premultiplied), in time
    /// order.
    pub frames: Vec<PathBuf>,
    /// Where each frame's result goes: same file name, this directory.
    pub out: PathBuf,
    /// The level the plate clipped at, linear Rec.709.
    pub clip: f32,
}

/// Reconstruct every frame of `job` on the GPU, writing linear Rec.709
/// premultiplied EXRs the input's size and the done-marker last. Frames
/// already written are skipped (a cancelled run resumes); a partial run
/// leaves no marker. Runs on the caller's thread and blocks on the GPU per
/// frame: the app puts it on a worker. Cancellable between frames.
pub fn run(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    job: &Job,
    progress: &mut dyn FnMut(usize, usize),
    cancel: &AtomicBool,
) -> Result<()> {
    use crate::decode::ExrDecoder as _;
    let n_frames = job.frames.len();
    anyhow::ensure!(n_frames > 0, "nothing was rendered to reconstruct");
    std::fs::create_dir_all(&job.out).with_context(|| format!("creating {}", job.out.display()))?;
    let net = match net(job.model) {
        Some(n) => n,
        None => bail!("{}", load_error(job.model).unwrap_or_else(|| "the model will not load".into())),
    };
    let dec = crate::decode::AnyDecoder::default();
    let first = dec.read_header(&job.frames[0]).with_context(|| format!("reading {}", job.frames[0].display()))?;
    let (w, h) = (first.width, first.height);
    let tex = |usage, label| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage,
            view_formats: &[],
        })
    };
    let input = tex(wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, "hdrcnn-in");
    let output = tex(wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC, "hdrcnn-out");
    let (iv, ov) = (input.create_view(&Default::default()), output.create_view(&Default::default()));
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let row = (w as u64 * 8).div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hdrcnn-readback"),
        size: row * h as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // f32 storage: see the module header.
    let mut kernels = crate::nn::gpu::Kernels::new(device.clone(), false);
    let budget = (device.limits().max_storage_buffer_binding_size / 4).min(1 << 30);
    let mut runner = crate::nn::gpu::Runner::new(&mut kernels, &queue, &net, (w, h), 1, budget);
    let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    runner.set_colour(&queue, crate::nn::gpu::Colour { encode: true, to_709: id, from_709: id, clip: job.clip });

    let stride = first.row_stride_f16();
    let mut buf = vec![half::f16::ZERO; stride * h as usize];
    progress(0, n_frames);
    for (i, path) in job.frames.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            bail!("cancelled");
        }
        let name = path.file_name().context("a frame without a name")?;
        let dst = job.out.join(name);
        if dst.is_file() {
            progress(i + 1, n_frames);
            continue;
        }
        let info = dec.read_header(path).with_context(|| format!("reading {}", path.display()))?;
        anyhow::ensure!((info.width, info.height) == (w, h), "frame {i} is {}x{}, the first was {w}x{h}", info.width, info.height);
        dec.decode_into(path, info, stride, &mut buf)?;
        queue.write_texture(
            input.as_image_copy(),
            bytemuck::cast_slice(&buf),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(stride as u32 * 2), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let mut enc = device.create_command_encoder(&Default::default());
        runner.encode(&kernels, &queue, &mut enc, &iv, &ov, &sampler);
        enc.copy_texture_to_buffer(
            output.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row as u32), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit([enc.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().context("the GPU went away")?.context("reading the result back")?;
        let px: Vec<half::f16> = {
            let view = readback.slice(..).get_mapped_range().context("a mapped buffer")?;
            bytemuck::cast_slice(&view).to_vec()
        };
        readback.unmap();
        let tmp = job.out.join(format!(".{}.part.exr", name.to_string_lossy()));
        crate::bakes::write_rgba_f16_exr_tagged(
            &tmp,
            w as usize,
            h as usize,
            row as usize / 2,
            &px,
            Some(crate::color::Primaries::Rec709.chromaticities()),
        )?;
        std::fs::rename(&tmp, &dst).with_context(|| format!("writing {}", dst.display()))?;
        progress(i + 1, n_frames);
    }
    crate::bakes::mark_baked(&job.out).context("marking the result done")?;
    Ok(())
}
