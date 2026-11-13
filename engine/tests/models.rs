//! The model downloader, end to end, against a throwaway HTTP server on
//! localhost. No network and no real weights: what is being checked is the
//! part that is easy to get quietly wrong — resume offsets, the hash
//! covering the resumed prefix as well as the new bytes, and a bad file
//! never being left where `state` would call it Ready.

use engine::models::{self, Format, ModelSpec, State};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Deterministic bytes, big enough to cross the downloader's 1 MB chunk and
/// progress-report boundaries.
fn payload() -> Vec<u8> {
    (0..3_000_000u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect()
}

/// A one-shot server that speaks just enough HTTP: 200 with Content-Length,
/// or 206 for `Range: bytes=N-`. Returns its port and a flag to stop it.
fn serve(body: Vec<u8>) -> (u16, Arc<AtomicBool>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    let stop = Arc::new(AtomicBool::new(false));
    let stop2 = stop.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            if stop2.load(Ordering::Relaxed) {
                return;
            }
            let Ok(conn) = conn else { continue };
            let body = body.clone();
            std::thread::spawn(move || handle(conn, body));
        }
    });
    (port, stop)
}

fn handle(mut conn: TcpStream, body: Vec<u8>) {
    let mut reader = BufReader::new(conn.try_clone().unwrap());
    let mut start = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
            break;
        }
        let lower = line.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("range:") {
            if let Some(n) = v.trim().strip_prefix("bytes=").and_then(|r| {
                r.split('-').next().and_then(|s| s.trim().parse::<usize>().ok())
            }) {
                start = n.min(body.len());
            }
        }
    }
    let slice = &body[start..];
    let head = if start > 0 {
        format!(
            "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {}-{}/{}\r\nConnection: close\r\n\r\n",
            slice.len(),
            start,
            body.len() - 1,
            body.len()
        )
    } else {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            slice.len()
        )
    };
    let _ = conn.write_all(head.as_bytes());
    let _ = conn.write_all(slice);
    let _ = conn.flush();
}

fn sha_of(bytes: &[u8]) -> String {
    // A counter, not `{:p}`: for a slice that prints
    // `Pointer { addr: 0x…, metadata: N }`, and the `:` is not a legal
    // filename character on Windows (every caller failed there).
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir();
    let p = dir.join(format!("wobble-models-sha-{}-{n}.bin", std::process::id()));
    std::fs::write(&p, bytes).unwrap();
    let h = models::hash_file(&p).unwrap();
    let _ = std::fs::remove_file(&p);
    h
}

/// A spec pointing at the throwaway server. Leaked because `fetch` wants a
/// `&'static` — a handful of specs across a test run is nothing.
fn spec_for(port: u16, len: u64, sha: &str, file: &'static str) -> &'static ModelSpec {
    Box::leak(Box::new(ModelSpec {
        id: "test-model",
        label: "test",
        family: "tracking",
        format: Format::Native,
        url: Box::leak(format!("http://127.0.0.1:{port}/w.bin").into_boxed_str()),
        file,
        bytes: len,
        sha256: Box::leak(sha.to_string().into_boxed_str()),
        license: "-",
        commercial: true,
        source: "-",
        about: "-",
    }))
}

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("wobble-dl-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Run a download to completion (or failure) and hand back the job.
fn drive(dir: &PathBuf, spec: &'static ModelSpec) -> models::Download {
    let mut dl = models::fetch_into(dir.clone(), spec);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while dl.poll() {
        assert!(std::time::Instant::now() < deadline, "download hung");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    dl
}

#[test]
fn downloads_verifies_and_lands_in_the_cache() {
    let body = payload();
    let (port, stop) = serve(body.clone());
    let spec = spec_for(port, body.len() as u64, &sha_of(&body), "clean.bin");
    let dir = tmpdir("clean");

    let dl = drive(&dir, spec);
    assert_eq!(dl.error, None, "clean download must not fail");
    assert_eq!(models::state_in(&dir, spec), State::Ready);
    assert_eq!(std::fs::read(spec.path_in(&dir)).unwrap(), body);
    // The part file is gone once the real one is in place.
    assert!(!spec.part_path_in(&dir).exists());
    stop.store(true, Ordering::Relaxed);
}

#[test]
fn resumes_a_partial_and_hashes_the_whole_file() {
    let body = payload();
    let (port, stop) = serve(body.clone());
    let spec = spec_for(port, body.len() as u64, &sha_of(&body), "resume.bin");
    let dir = tmpdir("resume");
    // An interrupted download: the first 1.2 MB are already on disk.
    let have = 1_200_000usize;
    std::fs::write(spec.part_path_in(&dir), &body[..have]).unwrap();
    assert_eq!(
        models::state_in(&dir, spec),
        State::Partial { have: have as u64 }
    );

    let dl = drive(&dir, spec);
    assert_eq!(dl.error, None, "resume must not fail");
    // The hash covers the prefix it resumed on top of, not just the tail —
    // if it did not, this file would be rejected as a mismatch.
    assert_eq!(models::state_in(&dir, spec), State::Ready);
    assert_eq!(std::fs::read(spec.path_in(&dir)).unwrap(), body);
    stop.store(true, Ordering::Relaxed);
}

#[test]
fn a_corrupt_partial_is_caught_by_the_hash_and_not_left_behind() {
    let body = payload();
    let (port, stop) = serve(body.clone());
    let spec = spec_for(port, body.len() as u64, &sha_of(&body), "corrupt.bin");
    let dir = tmpdir("corrupt");
    // Same LENGTH as a real prefix, different bytes: only hashing the
    // resumed prefix can catch this. Resuming blind would produce a file of
    // exactly the right size that `state` would happily call Ready.
    let have = 1_200_000usize;
    std::fs::write(spec.part_path_in(&dir), vec![0xABu8; have]).unwrap();

    let dl = drive(&dir, spec);
    let err = dl.error.clone().expect("a corrupt resume must fail, not land");
    assert!(err.contains("SHA-256"), "unexpected error: {err}");
    assert!(!spec.path_in(&dir).exists(), "bad bytes must not reach the cache");
    // ...and the bad part file is dropped, so retrying is a clean download
    // rather than an infinite resume of the same wrong prefix.
    assert!(!spec.part_path_in(&dir).exists());
    assert_eq!(models::state_in(&dir, spec), State::Missing);
    stop.store(true, Ordering::Relaxed);
}

#[test]
fn a_size_that_disagrees_with_the_registry_is_refused() {
    let body = payload();
    let (port, stop) = serve(body.clone());
    // The registry claims a different size than the server serves.
    let spec = spec_for(port, body.len() as u64 + 999, &sha_of(&body), "wrongsize.bin");
    let dir = tmpdir("wrongsize");

    let dl = drive(&dir, spec);
    let err = dl.error.clone().expect("a size mismatch must fail");
    assert!(err.contains("registry is out of date"), "unexpected error: {err}");
    assert!(!spec.path_in(&dir).exists());
    stop.store(true, Ordering::Relaxed);
}

#[test]
fn cancelling_keeps_the_part_file_so_a_later_fetch_resumes() {
    let body = payload();
    let (port, stop) = serve(body.clone());
    let spec = spec_for(port, body.len() as u64, &sha_of(&body), "cancel.bin");
    let dir = tmpdir("cancel");

    let mut dl = models::fetch_into(dir.clone(), spec);
    // Cancel as soon as anything has landed — mid-stream is the case that
    // matters.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        dl.poll();
        if dl.have > 0 || dl.finished {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "download never started");
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    dl.cancel();
    while dl.poll() {
        assert!(std::time::Instant::now() < deadline, "cancel hung");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_eq!(dl.error, None, "a cancel is not a failure to report");
    assert!(!spec.path_in(&dir).exists(), "a cancelled fetch must not look finished");

    // Whatever it stopped at, finishing from there must produce the file.
    let dl = drive(&dir, spec);
    assert_eq!(dl.error, None);
    assert_eq!(std::fs::read(spec.path_in(&dir)).unwrap(), body);
    stop.store(true, Ordering::Relaxed);
}

#[test]
fn nothing_in_the_catalog_downloads_without_being_asked() {
    // Constructing state for every real model must never touch the network
    // or create anything: the panel calls this every frame.
    let dir = tmpdir("catalog");
    for m in models::CATALOG {
        assert_eq!(models::state_in(&dir, m), State::Missing);
    }
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
}

/// Guard for the read loop that hashes a resumed prefix: it must handle a
/// part file that is shorter than it was when `state` measured it.
#[test]
fn a_part_file_that_shrank_starts_over_cleanly() {
    let body = payload();
    let (port, stop) = serve(body.clone());
    let spec = spec_for(port, body.len() as u64, &sha_of(&body), "shrank.bin");
    let dir = tmpdir("shrank");
    std::fs::write(spec.part_path_in(&dir), &body[..10]).unwrap();

    let dl = drive(&dir, spec);
    assert_eq!(dl.error, None);
    assert_eq!(std::fs::read(spec.path_in(&dir)).unwrap(), body);
    stop.store(true, Ordering::Relaxed);
}

/// The `Read` import is used by the server helper; keep clippy quiet about
/// it if the helper ever stops reading bodies.
#[allow(dead_code)]
fn _uses_read(mut r: impl Read) {
    let mut v = Vec::new();
    let _ = r.read_to_end(&mut v);
}
