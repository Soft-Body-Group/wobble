//! Test-sequence generator: procedural animated RGBA half EXRs.
//!
//! Usage: gen_exr <out_dir> <width> <height> <frames> <compression> <variant>
//!   compression: none | rle | zips | zip | piz
//!   variant: fg (animated shapes over transparency) | bg (opaque gradient field)
//!            | passes (multi-layer) | crypto (rgba + a 6-rank CryptoObject with manifest)
//!
//! DWAA is not writable by the pure-Rust exr crate (decode-only); generate
//! DWAA variants with external OpenEXR tooling if present.

use exr::prelude::*;
use rayon::prelude::*;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        eprintln!("usage: gen_exr <out_dir> <width> <height> <frames> <none|rle|zips|zip|piz> <fg|bg|passes|crypto>");
        std::process::exit(1);
    }
    let out_dir = std::path::PathBuf::from(&args[1]);
    let w: usize = args[2].parse()?;
    let h: usize = args[3].parse()?;
    let frames: usize = args[4].parse()?;
    let compression = match args[5].as_str() {
        "none" => Compression::Uncompressed,
        "rle" => Compression::RLE,
        "zips" => Compression::ZIP1,
        "zip" => Compression::ZIP16,
        "piz" => Compression::PIZ,
        other => anyhow::bail!("unknown compression {other}"),
    };
    let passes = args[6] == "passes";
    let crypto = args[6] == "crypto";
    let fg = match args[6].as_str() {
        "fg" => true,
        "bg" | "passes" | "crypto" => false,
        other => anyhow::bail!("unknown variant {other}"),
    };
    std::fs::create_dir_all(&out_dir)?;
    if crypto {
        engine::cryptomatte::write_plate_with(&out_dir, w, h, frames, compression)?;
        println!("wrote {frames} frames {w}x{h} {compression:?} rgba+CryptoObject00..02 to {}", out_dir.display());
        return Ok(());
    }

    // A multi-LAYER plate: the base rgba plus prefixed render passes, which
    // is what a renderer actually writes and what the Read node's layer
    // picker is for. Separate path because the rgba fast path above uses
    // SpecificChannels, which cannot carry extra names.
    if passes {
        let t0 = std::time::Instant::now();
        (0..frames).into_par_iter().try_for_each(|f| -> anyhow::Result<()> {
            let path = out_dir.join(format!("frame.{f:04}.exr"));
            let t = f as f32 / frames.max(1) as f32;
            let mut base = [(); 4].map(|_| Vec::with_capacity(w * h));
            let mut diffuse = [(); 3].map(|_| Vec::with_capacity(w * h));
            let mut normal = [(); 3].map(|_| Vec::with_capacity(w * h));
            let mut depth = Vec::with_capacity(w * h);
            for y in 0..h {
                for x in 0..w {
                    let (r, g, b, a) = pixel(x, y, w, h, t, false);
                    base[0].push(f16::from_f32(r));
                    base[1].push(f16::from_f32(g));
                    base[2].push(f16::from_f32(b));
                    base[3].push(f16::from_f32(a));
                    // The passes are deliberately UNLIKE the beauty, so a
                    // mis-wired layer selection is obvious on screen.
                    let (u, v) = (x as f32 / w as f32, y as f32 / h as f32);
                    diffuse[0].push(f16::from_f32(u));
                    diffuse[1].push(f16::from_f32(1.0 - v));
                    diffuse[2].push(f16::from_f32((u * v).sqrt()));
                    normal[0].push(f16::from_f32(u * 2.0 - 1.0));
                    normal[1].push(f16::from_f32(v * 2.0 - 1.0));
                    normal[2].push(f16::from_f32(1.0));
                    // A depth ramp that moves with the frame.
                    depth.push(f16::from_f32(10.0 * (v + t)));
                }
            }
            let ch = |name: &str, data: Vec<f16>| {
                AnyChannel::new(name, FlatSamples::F16(data))
            };
            let [br, bg, bb, ba] = base;
            let [dr, dg, db] = diffuse;
            let [nx, ny, nz] = normal;
            let layer = Layer::new(
                (w, h),
                LayerAttributes::named("rgba"),
                Encoding {
                    compression,
                    blocks: Blocks::ScanLines,
                    line_order: LineOrder::Increasing,
                },
                AnyChannels::sort(
                    vec![
                        ch("R", br),
                        ch("G", bg),
                        ch("B", bb),
                        ch("A", ba),
                        ch("diffuse.R", dr),
                        ch("diffuse.G", dg),
                        ch("diffuse.B", db),
                        ch("N.X", nx),
                        ch("N.Y", ny),
                        ch("N.Z", nz),
                        ch("depth.Z", depth),
                    ]
                    .into_iter()
                    .collect(),
                ),
            );
            Image::from_layer(layer).write().non_parallel().to_file(&path)?;
            Ok(())
        })?;
        println!(
            "wrote {frames} frames {w}x{h} {:?} rgba+diffuse+N+depth to {} in {:.1}s",
            compression,
            out_dir.display(),
            t0.elapsed().as_secs_f32()
        );
        return Ok(());
    }

    let t0 = std::time::Instant::now();
    (0..frames).into_par_iter().try_for_each(|f| -> anyhow::Result<()> {
        let path = out_dir.join(format!("frame.{f:04}.exr"));
        let t = f as f32 / frames.max(1) as f32;
        let layer = Layer::new(
            (w, h),
            LayerAttributes::named("rgba"),
            Encoding {
                compression,
                blocks: Blocks::ScanLines,
                line_order: LineOrder::Increasing,
            },
            SpecificChannels::rgba(|pos: Vec2<usize>| {
                let (r, g, b, a) = pixel(pos.x(), pos.y(), w, h, t, fg);
                (
                    f16::from_f32(r),
                    f16::from_f32(g),
                    f16::from_f32(b),
                    f16::from_f32(a),
                )
            }),
        );
        Image::from_layer(layer).write().non_parallel().to_file(&path)?;
        Ok(())
    })?;
    println!(
        "wrote {frames} frames {w}x{h} {:?} {} to {} in {:.1}s",
        compression,
        if fg { "fg" } else { "bg" },
        out_dir.display(),
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}

/// Procedural content with real high-frequency detail so compression ratios
/// are honest-ish (smooth gradients alone over-flatter PIZ/ZIP).
fn pixel(x: usize, y: usize, w: usize, h: usize, t: f32, fg: bool) -> (f32, f32, f32, f32) {
    let u = x as f32 / w as f32;
    let v = y as f32 / h as f32;
    let tau = std::f32::consts::TAU;
    if fg {
        // Three orbiting soft discs + noise-ish speckle, premultiplied alpha.
        let mut acc = [0.0f32; 4];
        for k in 0..3 {
            let ph = t * tau + k as f32 * tau / 3.0;
            let cx = 0.5 + 0.3 * ph.cos();
            let cy = 0.5 + 0.3 * (ph * 1.3).sin();
            let d = ((u - cx).powi(2) + (v - cy).powi(2)).sqrt();
            let cover = (1.0 - (d / 0.16).powi(2)).clamp(0.0, 1.0);
            let e = cover * cover;
            let col = match k {
                0 => [4.0, 0.6, 0.2],
                1 => [0.3, 2.5, 0.8],
                _ => [0.4, 0.7, 3.5],
            };
            for c in 0..3 {
                acc[c] += col[c] * e * (1.0 - acc[3]);
            }
            acc[3] += e * (1.0 - acc[3]);
        }
        // Speckle detail (deterministic hash noise).
        let n = hash2(x as u32, y as u32 ^ ((t * 4096.0) as u32));
        for c in acc.iter_mut().take(3) {
            *c *= 0.85 + 0.3 * n;
        }
        (acc[0], acc[1], acc[2], acc[3])
    } else {
        // Opaque scrolling gradient + fine grain, HDR-ish sky values.
        let base = 0.18 + 1.4 * (v + 0.1 * (tau * (u * 3.0 + t)).sin()).clamp(0.0, 1.5);
        let n = hash2(x as u32, y as u32);
        let grain = 0.92 + 0.16 * n;
        (
            base * 0.9 * grain,
            base * 0.75 * grain,
            base * 1.2 * grain,
            1.0,
        )
    }
}

fn hash2(x: u32, y: u32) -> f32 {
    let mut n = x.wrapping_mul(374761393).wrapping_add(y.wrapping_mul(668265263));
    n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    ((n ^ (n >> 16)) & 0xffff) as f32 / 65535.0
}
