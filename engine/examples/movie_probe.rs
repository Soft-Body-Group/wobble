//! Decode a few frames of a movie through the app's own decoder and print
//! their means; with `DUMP=<dir>` each frame is also written as a PPM.
//!
//! `cargo run --release -p engine --features movie --example movie_probe -- <file>...`
//!
//! To see what a RELEASE build sees, link the FFmpeg it bundles:
//! `FFMPEG_DIR=target/ffmpeg-lgpl PKG_CONFIG_PATH=target/ffmpeg-lgpl/lib/pkgconfig`
//! for the build and `DYLD_LIBRARY_PATH=target/ffmpeg-lgpl/lib` for the run.

#[cfg(feature = "movie")]
fn main() {
    use engine::decode::ExrDecoder;
    use engine::movie::MovieDecoder;
    use half::f16;

    for arg in std::env::args().skip(1) {
        let p = std::path::Path::new(&arg);
        let d = MovieDecoder::default();
        let info = match d.read_header(p) {
            Ok(i) => i,
            Err(e) => {
                println!("{arg}: {e:#}");
                continue;
            }
        };
        println!("{arg}: {}x{}, {:?} frames", info.width, info.height, d.frame_count(p));
        let stride = info.width as usize * 4;
        let mut buf = vec![f16::ZERO; stride * info.height as usize];
        for f in [0u64, 1, 2, 30, 10, 100] {
            let t = std::time::Instant::now();
            if let Err(e) = d.decode_frame_into(p, info, stride, None, f, &mut buf) {
                println!("  frame {f}: {e:#}");
                continue;
            }
            let mean = buf.iter().map(|v| v.to_f32()).sum::<f32>() / buf.len() as f32;
            println!("  frame {f}: mean {mean:.4} in {:?}", t.elapsed());
            if let Ok(dir) = std::env::var("DUMP") {
                let stem = p.file_stem().unwrap_or_default().to_string_lossy();
                let mut out = format!("P6 {} {} 255\n", info.width, info.height).into_bytes();
                for px in buf.chunks(4) {
                    out.extend(px[..3].iter().map(|c| (c.to_f32().clamp(0.0, 1.0) * 255.0) as u8));
                }
                std::fs::write(format!("{dir}/{stem}_{f}.ppm"), out).unwrap();
            }
        }
    }
}

#[cfg(not(feature = "movie"))]
fn main() {
    eprintln!("movie_probe needs --features movie");
}
