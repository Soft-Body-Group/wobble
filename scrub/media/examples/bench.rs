//! Decode benchmarks for one clip.
//!
//! ```text
//! cargo run --release -p media --example bench -- <file> [frames]
//! ```
//!
//! Prints, for the clip:
//! - **raw**: one thread, one warm decoder, frames in order (`Source::decode`);
//! - **play fwd / rev**: playback through the frame server as fast as it
//!   delivers (the playhead advances the moment its frame is cached);
//! - **seek**: random jumps on a cold cache, time until the frame is cached;
//! - **scrub**: a 60 Hz drag sweeping the clip, what fraction of redraws had
//!   the exact frame under the hand, and how far behind the picture was.
//!
//! `EDIFY_WORKERS`, `EDIFY_FF_THREADS` (`frame:0`, `slice:8`) and
//! `EDIFY_CACHE_GB` change the server's setup.

use media::{Cache, FrameServer, PlayState};
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().expect("usage: bench <file> [frames]"));
    let limit: Option<u64> = args.next().and_then(|s| s.parse().ok());
    let source = media::open(&path)?;
    let info = source.info().clone();
    let n = limit.unwrap_or(info.frames).min(info.frames);
    println!(
        "{}  {}x{}  {} frames @ {:.3} fps  {}  intra={}",
        path.display(),
        info.width,
        info.height,
        info.frames,
        info.fps(),
        info.label,
        info.intra
    );

    // Raw: single decoder, in order.
    let t0 = Instant::now();
    let mut bytes = 0usize;
    for f in 0..n {
        bytes += source.decode(f)?.bytes();
    }
    let raw = n as f64 / t0.elapsed().as_secs_f64();
    println!(
        "raw        {raw:7.1} fps   ({:.1} MB/frame, one thread)",
        bytes as f64 / n as f64 / 1e6
    );

    // Raw seek: random frames on one thread through the session pool.
    let mut rng = 0x2545f4914f6cdd1du64;
    let mut lat = Vec::new();
    for _ in 0..20 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let t0 = Instant::now();
        source.decode(rng % n)?;
        lat.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    lat.sort_by(|a, b| a.total_cmp(b));
    println!("raw seek   median {:6.1} ms   max {:6.1} ms", lat[lat.len() / 2], lat[lat.len() - 1]);

    let budget = Cache::default_budget();
    let server = || FrameServer::new(source.clone(), Cache::new(budget));

    for (name, dir) in [("play fwd", 1i8), ("play rev", -1)] {
        let s = server();
        let start = if dir > 0 { 0 } else { n - 1 };
        let t0 = Instant::now();
        let mut shown = 0u64;
        let mut p = start;
        loop {
            s.set(PlayState { frame: p, dir, step: 1, playing: true, scrubbing: false, looping: false });
            if s.wait_for(p, Duration::from_secs(10)).is_none() {
                println!("{name}   timed out at frame {p}");
                break;
            }
            shown += 1;
            if (dir > 0 && p + 1 >= n) || (dir < 0 && p == 0) {
                break;
            }
            p = (p as i64 + dir as i64) as u64;
        }
        let fps = shown as f64 / t0.elapsed().as_secs_f64();
        println!("{name}   {fps:7.1} fps   ({} workers)", s.workers());
    }

    // Seek: clicks at random frames, 150 ms apart; only the decoder
    // sessions are warm (the real situation mid-edit).
    let mut rng = 0x9e3779b97f4a7c15u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let s = server();
    let mut lat = Vec::new();
    for _ in 0..40 {
        let target = next() % n;
        // A cached neighbour from the last seek does not count.
        if s.is_cached(target) {
            continue;
        }
        let t0 = Instant::now();
        s.set(PlayState::paused(target));
        if s.wait_for(target, Duration::from_secs(10)).is_none() {
            println!("seek timed out at {target}");
            continue;
        }
        lat.push(t0.elapsed().as_secs_f64() * 1000.0);
        // A person looks at the frame before the next click; the server
        // fills in the neighbours meanwhile.
        std::thread::sleep(Duration::from_millis(150));
    }
    lat.sort_by(|a, b| a.total_cmp(b));
    let pct = |p: f64| lat[((lat.len() - 1) as f64 * p) as usize];
    println!(
        "seek       median {:6.1} ms   p90 {:6.1} ms   max {:6.1} ms   ({} jumps)",
        pct(0.5),
        pct(0.9),
        lat[lat.len() - 1],
        lat.len()
    );

    // Scrub: a hand dragging across the clip at 60 Hz, 2 frames a redraw,
    // there and back. Exact = the frame under the hand is on screen at
    // that redraw; lag = frames between the hand and what the viewer shows
    // (`FrameServer::pick`, the viewer's own rule).
    let s = server();
    let mut path: Vec<u64> = (0..n).step_by(2).collect();
    path.extend((0..n).rev().step_by(2));
    let (mut exact, mut lag_sum, mut shown_at) = (0usize, 0u64, 0u64);
    let t0 = Instant::now();
    for (i, &target) in path.iter().enumerate() {
        let dir = if i < path.len() / 2 { 1 } else { -1 };
        s.set(PlayState { frame: target, dir, step: 2, playing: false, scrubbing: true, looping: false });
        let tick_end = t0 + Duration::from_secs_f64((i + 1) as f64 / 60.0);
        std::thread::sleep(tick_end.saturating_duration_since(Instant::now()));
        match s.pick(target, Some(shown_at)) {
            Some((f, _)) if f == target => {
                exact += 1;
                shown_at = f;
            }
            Some((f, _)) => {
                shown_at = f;
                lag_sum += f.abs_diff(target);
            }
            None => lag_sum += shown_at.abs_diff(target),
        }
    }
    println!(
        "scrub      {:5.1}% exact   mean lag {:4.1} frames   ({} redraws at 60 Hz, 2 frames each)",
        exact as f64 * 100.0 / path.len() as f64,
        lag_sum as f64 / path.len() as f64,
        path.len()
    );
    // Jumps: the hand lands on an uncached frame mid-GOP. Time until a
    // picture near it is on screen (the keyframe a long-GOP decode passes)
    // and until the exact frame is.
    let (mut near_ms, mut exact_ms, mut jumps) = (0.0, 0.0, 0);
    for k in 0..8u64 {
        let s = server();
        let from = (k * n / 8).min(n - 1);
        s.set(PlayState::paused(from));
        s.wait_for(from, Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(150));
        let target = (from + n / 2 + 17) % n;
        let t0 = Instant::now();
        s.set(PlayState { frame: target, dir: 1, step: 1, playing: false, scrubbing: true, looping: false });
        loop {
            if s.pick(target, Some(from)).is_some() || t0.elapsed() > Duration::from_secs(10) {
                break;
            }
            std::thread::sleep(Duration::from_micros(500));
        }
        near_ms += t0.elapsed().as_secs_f64() * 1000.0;
        s.wait_for(target, Duration::from_secs(10));
        exact_ms += t0.elapsed().as_secs_f64() * 1000.0;
        jumps += 1;
    }
    println!(
        "jump       first picture {:6.1} ms   exact {:6.1} ms   (mean of {jumps} scrubs landing mid-GOP)",
        near_ms / jumps as f64,
        exact_ms / jumps as f64
    );
    let st = s.stats();
    if st.errors > 0 {
        println!("errors     {} ({})", st.errors, st.last_error.unwrap_or_default());
    }
    Ok(())
}
