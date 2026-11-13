//! Planar tracker harness: make a synthetic shot, track a directory of
//! frames, draw the result. The same `engine::planar` the app runs, minus
//! the app — so a tracker change can be judged on pictures in seconds.
//!
//! ```text
//! planar synth <dir> [--frames N] [--size WxH] [--bulge A] [--occluder] [--noise S]
//! planar track <dir> x0 y0 x1 y1 x2 y2 x3 y3 [--model M] [--mesh CELLS]
//!              [--start K] [--reanchor R] [--overlay <outdir>] [--truth]
//! ```
//!
//! `synth` writes 16-bit PNGs plus `truth.txt` (the texture→frame view per
//! frame and a region quad on the plane); `track --truth` scores against
//! it. `--overlay` writes one PNG per frame: the plate in grey, the tracked
//! quad in cyan (red when lost), the mesh in yellow.

use engine::decode::{AnyDecoder, ExrDecoder};
use engine::planar::{self, synth, Model, Plane, TrackOpts, Tracker, H};
use engine::player::Sequence;
use half::f16;
use std::path::{Path, PathBuf};

fn usage() -> ! {
    eprintln!("usage:\n  planar synth <dir> [--frames N] [--size WxH] [--bulge A] [--occluder] [--noise S]\n  planar track <dir> (x0 y0 x1 y1 x2 y2 x3 y3 | --box x0 y0 x1 y1 (normalised) | --truth) [--model translation|similarity|affine|perspective] [--mesh CELLS] [--start K] [--reanchor R] [--overlay <outdir>] [--detail]");
    std::process::exit(2);
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("synth") => synth_cmd(&args[1..]),
        Some("track") => track_cmd(&args[1..]),
        _ => usage(),
    }
}

fn synth_cmd(args: &[String]) -> anyhow::Result<()> {
    let Some(dir) = args.first() else { usage() };
    let dir = PathBuf::from(dir);
    let frames: usize = flag(args, "--frames").map(|s| s.parse()).transpose()?.unwrap_or(48);
    let (w, h) = match flag(args, "--size") {
        Some(s) => {
            let (a, b) = s.split_once('x').unwrap_or(("960", "540"));
            (a.parse()?, b.parse()?)
        }
        None => (960usize, 540usize),
    };
    let bulge: f32 = flag(args, "--bulge").map(|s| s.parse()).transpose()?.unwrap_or(0.0);
    let noise: f32 = flag(args, "--noise").map(|s| s.parse()).transpose()?.unwrap_or(0.01);
    let occluder = args.iter().any(|a| a == "--occluder");
    let t0 = std::time::Instant::now();
    let quad = synth::write_sequence(&dir, w, h, frames, bulge, occluder, noise)?;
    println!(
        "wrote {frames} frames {w}x{h} to {} in {:.1}s\nregion: {} {} {} {} {} {} {} {}",
        dir.display(),
        t0.elapsed().as_secs_f32(),
        quad[0][0], quad[0][1], quad[1][0], quad[1][1], quad[2][0], quad[2][1], quad[3][0], quad[3][1]
    );
    Ok(())
}

/// truth.txt → (quad, (view, bulge amplitude) per frame).
fn read_truth(dir: &Path) -> anyhow::Result<(Vec<[f64; 2]>, Vec<(H, f32)>)> {
    let text = std::fs::read_to_string(dir.join("truth.txt"))?;
    let mut quad = Vec::new();
    let mut views = Vec::new();
    for line in text.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.first() == Some(&"quad") {
            let v: Vec<f64> = parts[1..].iter().map(|s| s.parse().unwrap()).collect();
            quad = v.chunks(2).map(|c| [c[0], c[1]]).collect();
        } else if parts.len() >= 10 {
            let mut h = [0.0; 9];
            for (i, v) in parts[1..10].iter().enumerate() {
                h[i] = v.parse()?;
            }
            let amp: f32 = parts.get(10).map(|s| s.parse()).transpose()?.unwrap_or(0.0);
            views.push((h, amp));
        }
    }
    Ok((quad, views))
}

struct Canvas {
    w: usize,
    h: usize,
    rgb: Vec<u8>,
}

impl Canvas {
    fn from_plane(p: &Plane) -> Self {
        let mut rgb = Vec::with_capacity(p.w * p.h * 3);
        for &v in &p.px {
            let g = (v.clamp(0.0, 1.0) * 255.0) as u8;
            rgb.extend_from_slice(&[g, g, g]);
        }
        Self { w: p.w, h: p.h, rgb }
    }
    fn dot(&mut self, x: f64, y: f64, c: [u8; 3]) {
        if x >= 0.0 && y >= 0.0 && (x as usize) < self.w && (y as usize) < self.h {
            let i = (y as usize * self.w + x as usize) * 3;
            self.rgb[i..i + 3].copy_from_slice(&c);
        }
    }
    fn line(&mut self, a: [f64; 2], b: [f64; 2], c: [u8; 3]) {
        // Thick enough to see at 2K: one pixel per 960 of width.
        let thick = (self.w / 960).max(1) as f64;
        let n = ((b[0] - a[0]).abs().max((b[1] - a[1]).abs()).ceil() as usize).max(1);
        for k in 0..=n {
            let t = k as f64 / n as f64;
            let (x, y) = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t);
            let mut d = 0.0;
            while d < thick {
                self.dot(x + d, y, c);
                self.dot(x, y + d, c);
                d += 1.0;
            }
        }
    }
    fn cross(&mut self, p: [f64; 2], r: f64, c: [u8; 3]) {
        self.line([p[0] - r, p[1]], [p[0] + r, p[1]], c);
        self.line([p[0], p[1] - r], [p[0], p[1] + r], c);
    }
    fn write(&self, path: &Path) -> anyhow::Result<()> {
        let file = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut enc = png::Encoder::new(file, self.w as u32, self.h as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header()?.write_image_data(&self.rgb)?;
        Ok(())
    }
}

fn track_cmd(args: &[String]) -> anyhow::Result<()> {
    let Some(dir) = args.first() else { usage() };
    let dir = PathBuf::from(dir);
    let nums: Vec<f64> = args[1..].iter().take_while(|a| a.parse::<f64>().is_ok()).map(|a| a.parse().unwrap()).collect();
    let truth = if args.iter().any(|a| a == "--truth") { Some(read_truth(&dir)?) } else { None };
    // `--box x0 y0 x1 y1` is a region in NORMALISED coordinates (0..1), so
    // "the middle" needs no knowledge of the frame size.
    let boxed: Option<Vec<f64>> = args
        .iter()
        .position(|a| a == "--box")
        .map(|i| args[i + 1..].iter().take(4).filter_map(|s| s.parse().ok()).collect())
        .filter(|v: &Vec<f64>| v.len() == 4);
    let quad: Vec<[f64; 2]> = if nums.len() >= 8 {
        nums.chunks(2).take(4).map(|c| [c[0], c[1]]).collect()
    } else if let Some((q, _)) = &truth {
        q.clone()
    } else if boxed.is_none() {
        usage()
    } else {
        Vec::new()
    };
    let model = match flag(args, "--model").unwrap_or("perspective") {
        "translation" => Model::Translation,
        "similarity" => Model::Similarity,
        "affine" => Model::Affine,
        _ => Model::Perspective,
    };
    let mesh_cells: Option<usize> = flag(args, "--mesh").map(|s| s.parse()).transpose()?;
    let start: usize = flag(args, "--start").map(|s| s.parse()).transpose()?.unwrap_or(0);
    let reanchor: f32 = flag(args, "--reanchor").map(|s| s.parse()).transpose()?.unwrap_or(0.8);
    let overlay = flag(args, "--overlay").map(PathBuf::from);
    let detail = args.iter().any(|a| a == "--detail");
    if let Some(o) = &overlay {
        std::fs::create_dir_all(o)?;
    }

    let decoder = AnyDecoder::new(false);
    let seq = Sequence::from_dir(&dir, &decoder)?;
    let info = seq.info;
    let quad: Vec<[f64; 2]> = match boxed {
        Some(b) if quad.is_empty() => {
            let (w, h) = (info.width as f64, info.height as f64);
            vec![[b[0] * w, b[1] * h], [b[2] * w, b[1] * h], [b[2] * w, b[3] * h], [b[0] * w, b[3] * h]]
        }
        _ => quad,
    };
    let stride = info.row_stride_f16();
    let mut buf = vec![f16::ZERO; stride * info.height as usize];
    let load = |i: usize, buf: &mut Vec<f16>| -> anyhow::Result<Plane> {
        decoder.decode_into(&seq.frames[i], info, stride, buf)?;
        Ok(Plane::from_rgba_f16(buf, info.width as usize, info.height as usize, stride))
    };
    let n = seq.frames.len();
    if start >= n {
        anyhow::bail!("start {start} past the {n} frames");
    }
    let opts = TrackOpts {
        model,
        reanchor_below: reanchor,
        mesh: mesh_cells.map(|cells| planar::MeshOpts { cells, ..Default::default() }),
        detail: detail.then_some(planar::DETAIL_RADIUS),
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let reference = load(start, &mut buf)?;
    let mut tracker = Tracker::new(reference, quad.clone(), opts);
    println!("{} frames {}x{} model {} mesh {:?} from {start}", n, info.width, info.height, model.label(), mesh_cells);
    let mut worst = 0.0f64;
    let mut sum = 0.0f64;
    let mut lost = 0;
    let mut reanchors = 0;
    let mut mesh_worst = 0.0f64;
    for t in start..n {
        let lt = std::time::Instant::now();
        let frame = load(t, &mut buf)?;
        let load_ms = lt.elapsed().as_secs_f32() * 1000.0;
        let ft = std::time::Instant::now();
        let r = tracker.step(frame.clone());
        let ms = ft.elapsed().as_secs_f32() * 1000.0;
        let corners = tracker.corners(&r.h);
        let mut line = format!(
            "frame {t:4} score {:.3} cov {:.2} rej {:.2} jump {:.2} {:5.1}ms (load {:.0}ms)",
            r.score, r.coverage, r.rejected, r.jump, ms, load_ms
        );
        if let Some(why) = r.why {
            line.push_str(&format!(" LOST({})", why.label()));
            lost += 1;
        }
        if r.reanchored {
            line.push_str(" reanchor");
            reanchors += 1;
        }
        if let Some((_, views)) = &truth {
            if let (Some((v0, _)), Some((vt, amp))) = (views.get(start), views.get(t)) {
                let rel = planar::h_mul(vt, &planar::h_inv(v0).unwrap());
                let err = quad
                    .iter()
                    .map(|p| {
                        let (a, b) = planar::h_apply(&r.h, p[0], p[1]);
                        let (c, d) = planar::h_apply(&rel, p[0], p[1]);
                        ((a - c).powi(2) + (b - d).powi(2)).sqrt()
                    })
                    .fold(0.0, f64::max);
                worst = worst.max(err);
                sum += err;
                line.push_str(&format!(" err {err:.3}px"));
                // Mesh vertices against where the surface truly put them:
                // the plane's image, displaced by the frame's bulge (the
                // field is smooth, so a few fixed-point steps invert it).
                if let Some(m) = tracker.mesh() {
                    let mut me = 0.0f64;
                    let mut me_sum = 0.0f64;
                    let mut plane_dev = 0.0f64;
                    let voted = r.mesh_conf.iter().filter(|c| **c > 0.0).count();
                    for (i, v) in m.verts.iter().enumerate() {
                        let (c, d) = planar::h_apply(&rel, v[0] as f64, v[1] as f64);
                        let (mut qx, mut qy) = (c, d);
                        if *amp != 0.0 {
                            for _ in 0..20 {
                                let (dx, dy) = synth::bulge_offset(*amp, qx, qy, info.width as usize, info.height as usize);
                                qx = c + dx;
                                qy = d + dy;
                            }
                        }
                        let p = r.mesh[i];
                        let e = ((p[0] as f64 - qx).powi(2) + (p[1] as f64 - qy).powi(2)).sqrt();
                        me = me.max(e);
                        me_sum += e;
                        plane_dev = plane_dev.max(((c - qx).powi(2) + (d - qy).powi(2)).sqrt());
                    }
                    mesh_worst = mesh_worst.max(me);
                    line.push_str(&format!(
                        " mesh max {me:.2} mean {:.2}px, {voted}/{} voted (surface off plane by {plane_dev:.1})",
                        me_sum / m.verts.len().max(1) as f64,
                        m.verts.len()
                    ));
                }
            }
        }
        println!("{line}");
        if let Some(o) = &overlay {
            let mut cv = Canvas::from_plane(&frame);
            if let Some(m) = tracker.mesh() {
                for tri in &m.tris {
                    for k in 0..3 {
                        let a = r.mesh[tri[k] as usize];
                        let b = r.mesh[tri[(k + 1) % 3] as usize];
                        cv.line([a[0] as f64, a[1] as f64], [b[0] as f64, b[1] as f64], [220, 200, 40]);
                    }
                }
            }
            let col = if r.lost { [255, 40, 40] } else { [40, 230, 255] };
            for k in 0..corners.len() {
                cv.line(corners[k], corners[(k + 1) % corners.len()], col);
            }
            let c = planar::poly_centroid(&corners);
            cv.cross(c, 6.0, col);
            cv.write(&o.join(format!("overlay.{t:04}.png")))?;
        }
    }
    let tracked = n - start;
    println!(
        "done: {tracked} frames in {:.2}s ({:.1} ms/frame), {lost} lost, {reanchors} re-anchors",
        t0.elapsed().as_secs_f32(),
        t0.elapsed().as_secs_f32() * 1000.0 / tracked.max(1) as f32
    );
    if truth.is_some() {
        println!("corner error: worst {worst:.3}px mean {:.3}px; mesh worst {mesh_worst:.2}px", sum / tracked.max(1) as f64);
    }
    Ok(())
}
