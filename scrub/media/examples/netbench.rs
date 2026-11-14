//! Image sequences over (simulated) network storage: achievable frames per
//! second, BEFORE (each decode worker reads its own file, then decodes)
//! and AFTER (the sequence's read-ahead: many readers, raw bytes ahead of
//! the decoders), at a 1 GbE and a 10 GbE link and from the local disk.
//!
//!   cargo run --release -p media --example netbench [frames]
//!
//! Test frames are written once into testdata/net (2K and 4K; EXR half RGB
//! uncompressed / ZIP / PIZ, 10-bit DPX; DWAA when testdata/net/*_dwaa.exr
//! exist — Blender writes them). Frames are hard links of one file each,
//! read uncached, so the disk is not what is measured; the local cache is
//! off.

use media::netio::{self, NetSim};
use media::Source;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

fn write_exr(path: &Path, w: usize, h: usize, comp: exr::compression::Compression) {
    use exr::prelude::*;
    let mut seed = 1u64;
    let mut noise = vec![0f32; w * h];
    for v in &mut noise {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.04;
    }
    let ch = SpecificChannels::rgb(|p: Vec2<usize>| {
        let (x, y) = (p.x() as f32, p.y() as f32);
        let n = noise[p.y() * w + p.x()];
        (f16::from_f32(x / w as f32 + n), f16::from_f32(y / h as f32 + n), f16::from_f32(0.5 + 0.4 * (x / 37.0).sin() * (y / 23.0).cos()))
    });
    let mut layer = Layer::new((w, h), LayerAttributes::default(), Encoding::FAST_LOSSLESS, ch);
    layer.encoding.compression = comp;
    Image::from_layer(layer).write().to_file(path).expect("write exr");
}

fn write_dpx(path: &Path, w: usize, h: usize) {
    let b = media::dpx::encode_10bit(w, h, |x, y| {
        let seed = ((x as u64) << 32 | y as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let n = ((seed >> 58) as u16).min(15);
        [(x * 1023 / w) as u16 ^ n, (y * 1023 / h) as u16 ^ n, 512]
    });
    std::fs::write(path, b).expect("write dpx");
}

/// A directory of `n` hard links to `src`, named as a sequence.
fn links(src: &Path, dir: &Path, n: usize) -> PathBuf {
    let ext = src.extension().unwrap().to_string_lossy().into_owned();
    std::fs::create_dir_all(dir).unwrap();
    for i in 0..n {
        let p = dir.join(format!("f.{:04}.{ext}", 1001 + i));
        if !p.exists() {
            std::fs::hard_link(src, &p).unwrap();
        }
    }
    dir.join(format!("f.1001.{ext}"))
}

fn workers() -> usize {
    (media::sys::cores() / 2).clamp(2, 8)
}

/// Before: each worker reads its frame's file then decodes it.
fn before(sample: &Path, n: u64) -> f64 {
    let seq = Arc::new(media::sequence::Sequence::open(sample).expect("open"));
    let pat = media::sequence::Pattern::of(sample).unwrap().0;
    let next = Arc::new(AtomicU64::new(0));
    let t = Instant::now();
    let hs: Vec<_> = (0..workers())
        .map(|_| {
            let (next, pat, seq) = (next.clone(), pat.clone(), seq.clone());
            std::thread::spawn(move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                let p = pat.path(1001 + i as i64);
                let bytes = netio::read_file(&p).unwrap();
                let _ = seq.decode_bytes(&bytes, &p).expect("decode");
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    n as f64 / t.elapsed().as_secs_f64()
}

/// After: the sequence's read-ahead, decoders taking frames in order.
fn after(sample: &Path, n: u64) -> f64 {
    let seq = Arc::new(media::sequence::Sequence::open(sample).expect("open"));
    let next = Arc::new(AtomicU64::new(0));
    let t = Instant::now();
    let all: Vec<u64> = (0..n).collect();
    seq.prefetch(&all[..8.min(all.len())], true);
    let hs: Vec<_> = (0..workers())
        .map(|_| {
            let (next, seq) = (next.clone(), seq.clone());
            std::thread::spawn(move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                // What the frame server says each redraw: from here on.
                let ahead: Vec<u64> = (i..(i + 8).min(n)).collect();
                seq.prefetch(&ahead, true);
                let _ = seq.decode(i).expect("decode");
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    n as f64 / t.elapsed().as_secs_f64()
}

fn main() {
    media::movie::init();
    std::env::set_var("SCRUB_NETCACHE_GB", "0");
    let n: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(48);
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/net");
    std::fs::create_dir_all(&root).unwrap();
    use exr::compression::Compression as C;
    let mut cases: Vec<(String, PathBuf)> = Vec::new();
    for (res, w, h) in [("2K", 2048, 1080), ("4K", 4096, 2160)] {
        for (name, c) in [("EXR none", C::Uncompressed), ("EXR ZIP", C::ZIP16), ("EXR PIZ", C::PIZ)] {
            let f = root.join(format!("{}_{}.exr", res.to_lowercase(), name.split(' ').nth(1).unwrap().to_lowercase()));
            if !f.exists() {
                write_exr(&f, w, h, c);
            }
            cases.push((format!("{res} {name}"), f));
        }
        let d = root.join(format!("{}_dwaa.exr", res.to_lowercase()));
        if d.exists() {
            cases.push((format!("{res} EXR DWAA"), d));
        }
        let f = root.join(format!("{}_10bit.dpx", res.to_lowercase()));
        if !f.exists() {
            write_dpx(&f, w, h);
        }
        cases.push((format!("{res} DPX 10-bit"), f));
    }
    println!("{n} frames a run, {} decode workers, {} cores\n", workers(), media::sys::cores());
    println!("| clip | MB/frame | link | before fps | after fps |");
    println!("|---|---|---|---|---|");
    for (name, f) in &cases {
        let mb = std::fs::metadata(f).unwrap().len() as f64 / 1e6;
        let dir = root.join(format!("seq_{}", f.file_stem().unwrap().to_string_lossy()));
        let sample = links(f, &dir, n as usize);
        for (link, sim) in [("1 GbE", Some(NetSim::GBE_1)), ("10 GbE", Some(NetSim::GBE_10)), ("local", None)] {
            netio::set_sim(sim);
            let b = before(&sample, n);
            netio::set_sim(sim);
            let a = after(&sample, n);
            println!("| {name} | {mb:.1} | {link} | {b:.1} | {a:.1} |");
        }
    }
    netio::set_sim(None);
}
