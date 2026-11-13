//! Point tracker harness: track one point through a directory of frames
//! with the same `engine::pointtrack` the Tracker node runs, and print
//! every frame's answer.
//!
//! ```text
//! cargo run --release -p engine --example point -- <dir> cx cy hx hy \
//!     [--search S] [--model translation|similarity] [--blob T]
//! ```
//!
//! Everything NORMALISED like the node: centre, pattern half-size and
//! search (fraction of the frame's longer side).

use engine::decode::{AnyDecoder, ExrDecoder};
use engine::planar::{Model, Plane};
use engine::player::Sequence;
use engine::pointtrack::{PointOpts, PointTracker};
use half::f16;

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 5 {
        eprintln!("usage: point <dir> cx cy hx hy [--search S] [--model translation|similarity]");
        std::process::exit(2);
    }
    let n: Vec<f64> = args[1..5].iter().map(|s| s.parse()).collect::<Result<_, _>>()?;
    let search: f64 = flag(&args, "--search").map(|s| s.parse()).transpose()?.unwrap_or(0.08);
    let model = match flag(&args, "--model") {
        Some("translation") => Model::Translation,
        _ => Model::Similarity,
    };
    let decoder = AnyDecoder::new(false);
    let seq = Sequence::from_dir(std::path::Path::new(&args[0]), &decoder)?;
    let info = seq.info;
    let (w, h) = (info.width as usize, info.height as usize);
    let stride = info.row_stride_f16();
    let mut buf = vec![f16::ZERO; stride * h];
    let mut tr: Option<PointTracker> = None;
    let t0 = std::time::Instant::now();
    for (i, path) in seq.frames.iter().enumerate() {
        decoder.decode_into(path, info, stride, &mut buf)?;
        let plane = Plane::from_rgba_f16(&buf, w, h, stride);
        let t = std::time::Instant::now();
        match &mut tr {
            None => {
                tr = Some(PointTracker::new(
                    plane,
                    [n[0] * w as f64, n[1] * h as f64],
                    [n[2] * w as f64, n[3] * h as f64],
                    PointOpts { model, search: search * w.max(h) as f64, ..Default::default() },
                ));
                println!("{i:4} seed ({:.4}, {:.4})", n[0], n[1]);
            }
            Some(tr) => {
                let r = tr.step(plane);
                println!(
                    "{i:4} ({:.4}, {:.4}) score {:.3} {}{} {:.0} ms",
                    r.pos[0] / w as f64,
                    r.pos[1] / h as f64,
                    r.score,
                    if r.lost { "LOST " } else { "" },
                    r.why.map(|l| l.label()).unwrap_or(""),
                    t.elapsed().as_secs_f32() * 1000.0
                );
            }
        }
    }
    println!("{} frames in {:.2}s (decode included)", seq.frames.len(), t0.elapsed().as_secs_f32());
    Ok(())
}
