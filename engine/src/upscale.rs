//! The super-resolution models behind the Upscale node, built as
//! [`nn::Net`]s straight from the upstream checkpoints.
//!
//! Three architectures, each read from its published `.pth` through the
//! restricted [`torchpt`] reader (no conversion step, no Python):
//!
//! - **Real-ESRGAN compact** (`SRVGGNetCompact`, realesr-general-x4v3):
//!   32 convs at the input's size, a pixel shuffle, and the input added
//!   back nearest-neighbour. Fast.
//! - **Real-ESRGAN x4plus** (`RRDBNet`): 23 residual-in-residual dense
//!   blocks, then two nearest 2x upsamples. The quality one, and HEAVY.
//! - **SPAN** (Swift Parameter-free Attention Network): six attention
//!   blocks of re-parameterised convs. Fast.
//!
//! [`run`] is the node's job: rendered frames in, upscaled frames out, on a
//! worker on a click (the Flow precedent). Each builder is the upstream
//! `forward` written as ops; [`nn::reference`]
//! runs the result on the CPU, and `engine/tests/upscale.rs` holds both
//! against numbers torch produced from the same files.

use crate::nn::{pad4, Buf, Conv, Epi, Io, Net, Op, Out, Slice};
use crate::npy::Tensor;
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Which model. The setup stores the variant name; the catalog id is what
/// is downloaded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Model {
    /// Real-ESRGAN realesr-general-x4v3.
    #[default]
    Compact,
    /// Real-ESRGAN RealESRGAN_x4plus.
    X4Plus,
    /// SPAN x4 (4xNomosUni_span_multijpg).
    Span,
}

impl Model {
    pub const ALL: [Model; 3] = [Model::Compact, Model::Span, Model::X4Plus];

    /// The catalog id of its weights.
    pub fn model_id(self) -> &'static str {
        match self {
            Model::Compact => "realesr-general-x4v3",
            Model::X4Plus => "realesrgan-x4plus",
            Model::Span => "span-nomosuni-x4",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Model::Compact => "Real-ESRGAN compact (fast)",
            Model::X4Plus => "Real-ESRGAN x4plus (quality)",
            Model::Span => "SPAN (fast)",
        }
    }

    /// The architecture's own scale.
    pub fn scale(self) -> u32 {
        4
    }

    /// Build the net from a state dict.
    pub fn build(self, sd: &HashMap<String, Tensor>) -> Result<Net> {
        match self {
            Model::Compact => srvgg(sd),
            Model::X4Plus => rrdb(sd),
            Model::Span => span(sd),
        }
    }
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
    // Not cached: only try once the file is there, so a download landing
    // later is picked up (a missing file is not remembered as a failure).
    let path = crate::models::ready_path(model.model_id()).ok()?;
    let built = crate::torchpt::read(&path)
        .and_then(|sd| model.build(&sd))
        .map(Arc::new)
        .map_err(|e| format!("{e:#}"));
    if let Err(e) = &built {
        log::warn!("upscale: {} will not load: {e}", model.label());
    }
    let out = built.as_ref().ok().cloned();
    cache.lock().unwrap().insert(model, built);
    out
}

/// The model's weights are downloaded (the right size; the hash was
/// checked as they landed). A stat: cheap enough for every frame.
pub fn weights_ready(model: Model) -> bool {
    crate::models::ready_path(model.model_id()).is_ok()
}

/// Why a downloaded model would not build, if it would not.
pub fn load_error(model: Model) -> Option<String> {
    net(model).is_none().then(|| crate::models::ready_path(model.model_id()).err().map(|e| e.to_string()))?
}

// ------------------------------------------------------------ the run

/// What to upscale.
pub struct Job {
    pub model: Model,
    /// The rendered input frames (linear Rec.709, premultiplied), in time
    /// order.
    pub frames: Vec<PathBuf>,
    /// Where each frame's result goes: same file name, this directory.
    pub out: PathBuf,
    /// 4, or 2 (the model's 4x, halved).
    pub scale: u32,
}

/// Upscale every frame of `job` on the GPU, writing linear Rec.709
/// premultiplied EXRs at `scale` times the input and the done-marker last.
/// Frames already written are skipped (a cancelled run resumes); a partial
/// run leaves no marker, so it is never mistaken for a finished one.
///
/// Runs on the caller's thread and blocks on the GPU per frame: the app
/// puts it on a worker. Cancellable between frames — and between the
/// submissions of one frame's tiles, which is also what keeps each one
/// under Windows' GPU watchdog.
pub fn run(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    job: &Job,
    progress: &mut dyn FnMut(usize, usize),
    cancel: &AtomicBool,
) -> Result<()> {
    use crate::decode::ExrDecoder as _;
    let n_frames = job.frames.len();
    anyhow::ensure!(n_frames > 0, "nothing was rendered to upscale");
    std::fs::create_dir_all(&job.out).with_context(|| format!("creating {}", job.out.display()))?;
    let net = match net(job.model) {
        Some(n) => n,
        None => bail!("{}", load_error(job.model).unwrap_or_else(|| "the model will not load".into())),
    };
    let dec = crate::decode::AnyDecoder::default();
    let first = dec.read_header(&job.frames[0]).with_context(|| format!("reading {}", job.frames[0].display()))?;
    let (w, h) = (first.width, first.height);
    let s = job.scale.clamp(1, net.scale);
    let (ow, oh) = (w * s, h * s);
    let limit = device.limits().max_texture_dimension_2d;
    anyhow::ensure!(ow <= limit && oh <= limit, "{ow}x{oh} is past this GPU's largest texture ({limit})");

    let tex = |w, h, usage, label| {
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
    let input = tex(w, h, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, "upscale-in");
    let output = tex(ow, oh, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC, "upscale-out");
    let (iv, ov) = (input.create_view(&Default::default()), output.create_view(&Default::default()));
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let row = (ow as u64 * 8).div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("upscale-readback"),
        size: row * oh as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut kernels = crate::nn::gpu::Kernels::new(device.clone(), true);
    // A quarter of what one binding may hold, at most 1 GiB: the arena is
    // per tile, so a smaller card only means more tiles.
    let budget = (device.limits().max_storage_buffer_binding_size / 4).min(1 << 30);
    let mut runner = crate::nn::gpu::Runner::new(&mut kernels, &queue, &net, (w, h), s, budget);
    // The frames are linear Rec.709 already: the runner only encodes.
    let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    runner.set_colour(&queue, crate::nn::gpu::Colour { encode: true, to_709: id, from_709: id, clip: 1.0 });

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
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row as u32), rows_per_image: Some(oh) },
            },
            wgpu::Extent3d { width: ow, height: oh, depth_or_array_layers: 1 },
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
            ow as usize,
            oh as usize,
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

// ------------------------------------------------------------- builders

/// The most a tile's halo carries, input pixels. A compact net sees 34 px
/// and gets all of it; an RRDB sees hundreds, and past this the error at a
/// seam is below what f16 storage resolves anyway (measured by the tiling
/// test).
pub const MAX_HALO: u32 = 40;

trait WithHalo {
    fn with_halo(self) -> Self;
}

impl WithHalo for Net {
    fn with_halo(mut self) -> Self {
        self.halo = self.receptive().min(MAX_HALO);
        self
    }
}

fn get<'a>(sd: &'a HashMap<String, Tensor>, k: &str) -> Result<&'a Tensor> {
    sd.get(k).with_context(|| format!("the checkpoint has no {k}"))
}

/// A torch Conv2d by its prefix (`<p>.weight`, `<p>.bias`).
fn conv(
    sd: &HashMap<String, Tensor>,
    p: &str,
    src: Slice,
    dst: Slice,
    epi: Epi,
) -> Result<Conv> {
    let w = get(sd, &format!("{p}.weight"))?;
    let b = get(sd, &format!("{p}.bias"))?;
    let [cout, cin, k, k2] = w.dims[..] else { bail!("{p}.weight is not 4-d") };
    if k != k2 || (k != 1 && k != 3) {
        bail!("{p} is a {k}x{k2} conv; only 1x1 and 3x3 are supported");
    }
    if src.n < pad4(cin as u32) || dst.n < pad4(cout as u32) {
        bail!("{p} is {cin}->{cout}, wider than its slices");
    }
    Ok(Conv {
        k: k as u32,
        cin: cin as u32,
        cout: cout as u32,
        weight: w.data.clone().into(),
        bias: b.data.clone().into(),
        src,
        dst,
        up: 1,
        silu_in: false,
        epi,
    })
}

/// Real-ESRGAN's compact net:
///
/// ```text
/// out = x
/// for conv, prelu in body: out = prelu(conv(out))
/// out = pixel_shuffle(conv_last(out)) + nearest(x)
/// ```
fn srvgg(sd: &HashMap<String, Tensor>) -> Result<Net> {
    // body.0 is the first conv, then (prelu, conv) pairs, the last conv
    // unactivated: find the depth from the keys.
    let mut convs = Vec::new();
    let mut i = 0;
    while sd.contains_key(&format!("body.{i}.weight")) {
        convs.push(i);
        i += 2;
    }
    if convs.len() < 2 {
        bail!("not a Real-ESRGAN compact checkpoint (no body.N convs)");
    }
    let feat = get(sd, "body.0.weight")?.dims[0] as u32;
    let last_out = get(sd, &format!("body.{}.weight", convs[convs.len() - 1]))?.dims[0] as u32;
    let scale = ((last_out / 3) as f32).sqrt().round() as u32;
    if scale * scale * 3 != last_out {
        bail!("compact net's last conv has {last_out} outputs, not 3 * scale²");
    }
    let (inp, a, b, o) = (0, 1, 2, 3);
    let bufs = vec![
        Buf { ch: 4, res: 1, div: 1 },
        Buf { ch: pad4(feat), res: 1, div: 1 },
        Buf { ch: pad4(feat), res: 1, div: 1 },
        Buf { ch: pad4(last_out), res: 1, div: 1 },
    ];
    let fs = |buf| Slice::new(buf, 0, pad4(feat));
    let mut ops = Vec::new();
    let mut src = Slice::new(inp, 0, 4);
    let mut cur = a;
    for (n, &idx) in convs.iter().enumerate() {
        let last = n + 1 == convs.len();
        if last {
            ops.push(Op::Conv(conv(sd, &format!("body.{idx}"), src, Slice::new(o, 0, pad4(last_out)), Epi::none())?));
        } else {
            let slope = get(sd, &format!("body.{}.weight", idx + 1))?.data.clone();
            let epi = Epi::Plain { alpha: 1.0, res: vec![], slope: Some(slope) };
            ops.push(Op::Conv(conv(sd, &format!("body.{idx}"), src, fs(cur), epi)?));
            src = fs(cur);
            cur = if cur == a { b } else { a };
        }
    }
    Ok(Net {
        bufs,
        input: Slice::new(inp, 0, 4),
        mean: [0.0; 3],
        range: 1.0,
        ops,
        out: Out::Shuffle(Slice::new(o, 0, pad4(last_out))),
        base: true,
        scale,
        halo: 0,
        io: Io::Upscale,
    }
    .with_halo())
}

/// ESRGAN's RRDBNet, x4:
///
/// ```text
/// feat = conv_first(x)
/// feat = feat + conv_body(body(feat))        # body: 23 RRDBs
/// feat = lrelu(conv_up1(nearest2x(feat)))
/// feat = lrelu(conv_up2(nearest2x(feat)))
/// out  = conv_last(lrelu(conv_hr(feat)))
/// RRDB(x) = rdb3(rdb2(rdb1(x))) * 0.2 + x
/// RDB(x):  x1..x4 = lrelu(convN(cat(x, x1..))), x5 = conv5(cat(...))
///          return x5 * 0.2 + x
/// ```
///
/// A dense block's concatenation is free: x lives in channels 0..64 of a
/// 192-channel buffer and each conv appends its 32. Three such buffers
/// rotate, so an RDB writes its answer into channels 0..64 of the next
/// without overwriting what it reads.
fn rrdb(sd: &HashMap<String, Tensor>) -> Result<Net> {
    let nf = get(sd, "conv_first.weight")?.dims[0] as u32;
    let gc = get(sd, "body.0.rdb1.conv1.weight")?.dims[0] as u32;
    let mut nb = 0;
    while sd.contains_key(&format!("body.{nb}.rdb1.conv1.weight")) {
        nb += 1;
    }
    if !sd.contains_key("conv_up2.weight") {
        bail!("only the x4 RRDBNet (conv_up1 + conv_up2) is supported");
    }
    let dense = nf + 4 * gc;
    let (inp, f, d0, gbuf, u1, u2, u3, o) = (0, 1, 2, 5, 6, 7, 8, 9);
    let bufs = vec![
        Buf { ch: 4, res: 1, div: 1 },
        Buf { ch: nf, res: 1, div: 1 },
        Buf { ch: dense, res: 1, div: 1 },
        Buf { ch: dense, res: 1, div: 1 },
        Buf { ch: dense, res: 1, div: 1 },
        Buf { ch: nf, res: 1, div: 1 },
        Buf { ch: nf, res: 2, div: 1 },
        Buf { ch: nf, res: 4, div: 1 },
        Buf { ch: nf, res: 4, div: 1 },
        Buf { ch: 4, res: 4, div: 1 },
    ];
    let mut ops = Vec::new();
    ops.push(Op::Conv(conv(sd, "conv_first", Slice::new(inp, 0, 4), Slice::new(f, 0, nf), Epi::none())?));
    ops.push(Op::Copy { src: Slice::new(f, 0, nf), dst: Slice::new(d0, 0, nf) });
    // The RRDB's input lives in channels 0..nf of dense buffer `cur`.
    let mut cur = 0usize;
    for b in 0..nb {
        let block_in = d0 + cur;
        let mut rin = cur;
        for r in 1..=3 {
            let buf = d0 + rin;
            let next = d0 + (rin + 1) % 3;
            for c in 1..=4u32 {
                let src = Slice::new(buf, 0, nf + (c - 1) * gc);
                let dst = Slice::new(buf, nf + (c - 1) * gc, gc);
                let e = Epi::leaky(gc, 0.2);
                ops.push(Op::Conv(conv(sd, &format!("body.{b}.rdb{r}.conv{c}"), src, dst, e)?));
            }
            // conv5: x5 * 0.2 + x, and for the last RDB the RRDB's own
            // residual folded in: (x5 * 0.2 + x) * 0.2 + block_in.
            let src = Slice::new(buf, 0, dense);
            let x = Slice::new(buf, 0, nf);
            let epi = if r < 3 {
                Epi::Plain { alpha: 0.2, res: vec![(x, 1.0)], slope: None }
            } else {
                Epi::Plain {
                    alpha: 0.04,
                    res: vec![(x, 0.2), (Slice::new(block_in, 0, nf), 1.0)],
                    slope: None,
                }
            };
            ops.push(Op::Conv(conv(sd, &format!("body.{b}.rdb{r}.conv5"), src, Slice::new(next, 0, nf), epi)?));
            rin = (rin + 1) % 3;
        }
        // Three RDBs moved the answer three buffers on: back where it
        // started. The output is in `next` of the last RDB.
        cur = rin;
    }
    let trunk = Slice::new(d0 + cur, 0, nf);
    ops.push(Op::Conv(conv(
        sd,
        "conv_body",
        trunk,
        Slice::new(gbuf, 0, nf),
        Epi::Plain { alpha: 1.0, res: vec![(Slice::new(f, 0, nf), 1.0)], slope: None },
    )?));
    let mut up1 = conv(sd, "conv_up1", Slice::new(gbuf, 0, nf), Slice::new(u1, 0, nf), Epi::leaky(nf, 0.2))?;
    up1.up = 2;
    ops.push(Op::Conv(up1));
    let mut up2 = conv(sd, "conv_up2", Slice::new(u1, 0, nf), Slice::new(u2, 0, nf), Epi::leaky(nf, 0.2))?;
    up2.up = 2;
    ops.push(Op::Conv(up2));
    ops.push(Op::Conv(conv(sd, "conv_hr", Slice::new(u2, 0, nf), Slice::new(u3, 0, nf), Epi::leaky(nf, 0.2))?));
    ops.push(Op::Conv(conv(sd, "conv_last", Slice::new(u3, 0, nf), Slice::new(o, 0, 4), Epi::none())?));
    Ok(Net {
        bufs,
        input: Slice::new(inp, 0, 4),
        mean: [0.0; 3],
        range: 1.0,
        ops,
        out: Out::Direct(Slice::new(o, 0, 4)),
        base: false,
        scale: 4,
        halo: 0,
        io: Io::Upscale,
    }
    .with_halo())
}

/// A SPAN `Conv3XC` folded into the one 3x3 conv it is at inference: the
/// 1x1 -> 3x3 -> 1x1 chain composed (biases carried through) plus the 1x1
/// skip on the centre tap — upstream's `update_params`, which runs on
/// every eval forward, so the checkpoint's own `eval_conv` may be stale.
fn conv3xc(sd: &HashMap<String, Tensor>, p: &str, src: Slice, dst: Slice, epi: Epi) -> Result<Conv> {
    let w1 = get(sd, &format!("{p}.conv.0.weight"))?;
    let b1 = get(sd, &format!("{p}.conv.0.bias"))?;
    let w2 = get(sd, &format!("{p}.conv.1.weight"))?;
    let b2 = get(sd, &format!("{p}.conv.1.bias"))?;
    let w3 = get(sd, &format!("{p}.conv.2.weight"))?;
    let b3 = get(sd, &format!("{p}.conv.2.bias"))?;
    let sw = get(sd, &format!("{p}.sk.weight"))?;
    let sb = get(sd, &format!("{p}.sk.bias"))?;
    let (n1, cin) = (w1.dims[0], w1.dims[1]); // [cin*g, cin, 1, 1]
    let m = w2.dims[0]; // [cout*g, cin*g, 3, 3]
    let cout = w3.dims[0]; // [cout, cout*g, 1, 1]
    if w2.dims[1] != n1 || w3.dims[1] != m || w2.dims[2] != 3 {
        bail!("{p} is not a Conv3XC");
    }
    // W12[m][i][t] = Σ_n w2[m][n][t] w1[n][i]; b12[m] = Σ_n Σ_t w2[m][n][t] b1[n] + b2[m]
    let mut w12 = vec![0.0f64; m * cin * 9];
    let mut b12 = vec![0.0f64; m];
    for mi in 0..m {
        b12[mi] = b2.data[mi] as f64;
        for n in 0..n1 {
            for t in 0..9 {
                let a = w2.data[(mi * n1 + n) * 9 + t] as f64;
                b12[mi] += a * b1.data[n] as f64;
                for i in 0..cin {
                    w12[(mi * cin + i) * 9 + t] += a * w1.data[n * cin + i] as f64;
                }
            }
        }
    }
    let mut w = vec![0.0f32; cout * cin * 9];
    let mut b = vec![0.0f32; cout];
    for o in 0..cout {
        let mut bo = b3.data[o] as f64 + sb.data[o] as f64;
        for mi in 0..m {
            let c = w3.data[o * m + mi] as f64;
            bo += c * b12[mi];
            for i in 0..cin {
                for t in 0..9 {
                    let at = (o * cin + i) * 9 + t;
                    w[at] = (w[at] as f64 + c * w12[(mi * cin + i) * 9 + t]) as f32;
                }
            }
        }
        for i in 0..cin {
            w[(o * cin + i) * 9 + 4] += sw.data[o * cin + i];
        }
        b[o] = bo as f32;
    }
    if src.n < pad4(cin as u32) || dst.n < pad4(cout as u32) {
        bail!("{p} is wider than its slices");
    }
    Ok(Conv {
        k: 3,
        cin: cin as u32,
        cout: cout as u32,
        weight: w.into(),
        bias: b.into(),
        src,
        dst,
        up: 1,
        silu_in: false,
        epi,
    })
}

/// SPAN:
///
/// ```text
/// x = (x - mean) * 255               # when the checkpoint has no `no_norm`
/// f  = conv_1(x)
/// b1 = block_1(f) … b6 = block_6(b5), and block_6's first conv's output
/// out = upsampler(conv_cat(cat(f, conv_2(b6), b1, b6.c1)))
/// SPAB(x): c1 = silu(conv(x)) (in place); c2 = conv(c1); c3 = conv(silu(c2))
///          return (c3 + x) * (sigmoid(c3) - 0.5)
/// ```
///
/// `cat` is free: the four tensors are written into the four quarters of
/// one buffer in the order conv_cat reads them.
fn span(sd: &HashMap<String, Tensor>) -> Result<Net> {
    let nf = get(sd, "conv_1.sk.weight")?.dims[0] as u32;
    let up_out = get(sd, "upsampler.0.weight")?.dims[0] as u32;
    let scale = ((up_out / 3) as f32).sqrt().round() as u32;
    if scale * scale * 3 != up_out {
        bail!("SPAN's upsampler has {up_out} outputs, not 3 * scale²");
    }
    let norm = !sd.contains_key("no_norm");
    let (inp, cat, x, c1, c2, o) = (0, 1, 2, 3, 4, 5);
    let bufs = vec![
        Buf { ch: 4, res: 1, div: 1 },
        Buf { ch: 4 * nf, res: 1, div: 1 },
        Buf { ch: 2 * nf, res: 1, div: 1 },
        Buf { ch: nf, res: 1, div: 1 },
        Buf { ch: nf, res: 1, div: 1 },
        Buf { ch: pad4(up_out), res: 1, div: 1 },
    ];
    // Quarters of `cat`: f, conv_2(b6), b1, block_6's c1.
    let q = |i: u32| Slice::new(cat, i * nf, nf);
    // Block outputs ping-pong between the halves of `x`, except b1, which
    // conv_cat reads too, so it is written straight into its quarter.
    let xs = |i: u32| Slice::new(x, i * nf, nf);
    let mut ops = vec![Op::Conv(conv3xc(sd, "conv_1", Slice::new(inp, 0, 4), q(0), Epi::none())?)];
    let mut input = q(0);
    for b in 1..=6u32 {
        let p = format!("block_{b}");
        // Upstream's SiLU is `inplace=True`: it overwrites c1, so what
        // block 6 hands conv_cat as its "pre-activation" output is the
        // ACTIVATED one. Storing silu(c1) is both that and c2's input.
        let c1s = if b == 6 { q(3) } else { Slice::new(c1, 0, nf) };
        ops.push(Op::Conv(conv3xc(sd, &format!("{p}.c1_r"), input, c1s, Epi::Silu)?));
        ops.push(Op::Conv(conv3xc(sd, &format!("{p}.c2_r"), c1s, Slice::new(c2, 0, nf), Epi::none())?));
        let out = if b == 1 { q(2) } else { xs(b % 2) };
        let mut k3 = conv3xc(sd, &format!("{p}.c3_r"), Slice::new(c2, 0, nf), out, Epi::Gate { res: input })?;
        k3.silu_in = true;
        ops.push(Op::Conv(k3));
        input = out;
    }
    ops.push(Op::Conv(conv3xc(sd, "conv_2", input, q(1), Epi::none())?));
    // conv_cat (1x1) into a half of `x` that block 6 is not holding.
    let cc = xs(1 - 6 % 2);
    ops.push(Op::Conv(conv(sd, "conv_cat", Slice::new(cat, 0, 4 * nf), cc, Epi::none())?));
    ops.push(Op::Conv(conv(sd, "upsampler.0", cc, Slice::new(o, 0, pad4(up_out)), Epi::none())?));
    Ok(Net {
        bufs,
        input: Slice::new(inp, 0, 4),
        mean: if norm { [0.4488, 0.4371, 0.4040] } else { [0.0; 3] },
        range: if norm { 255.0 } else { 1.0 },
        ops,
        out: Out::Shuffle(Slice::new(o, 0, pad4(up_out))),
        base: false,
        scale,
        halo: 0,
        io: Io::Upscale,
    }
    .with_halo())
}
