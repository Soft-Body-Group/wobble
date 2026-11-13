//! How long a frame takes to decode: `cargo run --release -p engine --example decode_time -- <dir> [frames]`.
use engine::decode::{AnyDecoder, ExrDecoder};
use engine::player::Sequence;
use half::f16;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).expect("dir");
    let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    for single in [false, true] {
        let dec = AnyDecoder::new(single);
        let seq = Sequence::from_dir(Path::new(&dir), &dec)?;
        let info = seq.info;
        let stride = info.row_stride_f16();
        let mut buf = vec![f16::ZERO; stride * info.height as usize];
        let t0 = std::time::Instant::now();
        for f in seq.frames.iter().take(n) {
            dec.decode_into(f, info, stride, &mut buf)?;
        }
        println!(
            "single_threaded={single}: {} frames {}x{} at {:.0} ms/frame",
            n.min(seq.frames.len()),
            info.width,
            info.height,
            t0.elapsed().as_secs_f32() * 1000.0 / n.min(seq.frames.len()).max(1) as f32
        );
    }
    Ok(())
}
