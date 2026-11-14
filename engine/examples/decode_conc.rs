//! Decode throughput with K frames in flight at once:
//! `cargo run --release -p engine --example decode_conc -- <dir>`.
//! Prints frames/s for the multi-threaded and the single-threaded decoder
//! at each concurrency, with a fresh buffer per decode (as the player has).
use engine::decode::{AnyDecoder, ExrDecoder};
use engine::player::Sequence;
use half::f16;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).expect("dir");
    for single in [false, true] {
        for k in [1usize, 2, 4, 8, 16] {
            let dec = Arc::new(AnyDecoder::new(single));
            let seq = Arc::new(Sequence::from_dir(Path::new(&dir), &*dec)?);
            let n = seq.frames.len().min(24);
            let next = Arc::new(AtomicUsize::new(0));
            let t0 = std::time::Instant::now();
            let hs: Vec<_> = (0..k)
                .map(|_| {
                    let (dec, seq, next) = (dec.clone(), seq.clone(), next.clone());
                    std::thread::spawn(move || {
                        let info = seq.info;
                        let stride = info.row_stride_f16();
                        let reuse = std::env::var_os("REUSE").is_some();
                        let mut keep = vec![f16::ZERO; if reuse { stride * info.height as usize } else { 0 }];
                        loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        let mut fresh: Vec<f16>;
                        let buf: &mut Vec<f16> = if reuse { &mut keep } else { fresh = vec![f16::ZERO; stride * info.height as usize]; &mut fresh };
                        dec.decode_into(&seq.frames[i], info, stride, buf).unwrap();
                        }
                    })
                })
                .collect();
            for h in hs {
                h.join().unwrap();
            }
            let el = t0.elapsed().as_secs_f32();
            println!("single_threaded={single} in flight {k:2}: {:.1} frames/s ({:.0} ms a frame)", n as f32 / el, el * 1000.0 / n as f32 * k as f32);
        }
    }
    Ok(())
}
