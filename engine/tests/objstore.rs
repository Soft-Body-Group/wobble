//! `s3://` reads end to end against a throwaway S3 on localhost: a real
//! multi-part EXR sequence in a fake bucket, read through the same
//! `Sequence` / `ExrDecoder` path a Read uses. What is being checked is
//! the part that is easy to get quietly wrong: that a one-layer decode
//! fetches that layer's bytes and not the file, that a second decode
//! fetches nothing, that the disk cache outlives the process's memory, and
//! that a frame appearing on the server is seen.

use engine::decode::{ExrDecoder, ExrsDecoder};
use engine::objstore::{self, Config, Endpoint};
use engine::player::Sequence;
use engine::vfs;
use half::f16;
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Cursor, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

type Bucket = Arc<Mutex<BTreeMap<String, Vec<u8>>>>;

struct Server {
    port: u16,
    #[allow(dead_code)]
    bucket: Bucket,
    requests: Arc<AtomicU64>,
    /// Bytes of object data sent (not headers, not listings).
    sent: Arc<AtomicU64>,
    /// Credential id requests must be signed with, or None for anonymous.
    require: Option<&'static str>,
}

fn serve(bucket: Bucket, require: Option<&'static str>) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    let requests = Arc::new(AtomicU64::new(0));
    let sent = Arc::new(AtomicU64::new(0));
    let (b, r, s) = (bucket.clone(), requests.clone(), sent.clone());
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(conn) = conn else { continue };
            let (b, r, s) = (b.clone(), r.clone(), s.clone());
            std::thread::spawn(move || handle(conn, b, r, s, require));
        }
    });
    Server { port, bucket, requests, sent, require }
}

fn reply(conn: &mut TcpStream, status: &str, extra: &str, body: &[u8]) {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{extra}Connection: close\r\n\r\n",
        body.len()
    );
    let _ = conn.write_all(head.as_bytes());
    let _ = conn.write_all(body);
    let _ = conn.flush();
}

fn handle(mut conn: TcpStream, bucket: Bucket, requests: Arc<AtomicU64>, sent: Arc<AtomicU64>, require: Option<&str>) {
    let mut reader = BufReader::new(conn.try_clone().unwrap());
    let mut first = String::new();
    if reader.read_line(&mut first).unwrap_or(0) == 0 {
        return;
    }
    let mut range: Option<(u64, u64)> = None;
    let mut auth: Option<String> = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
            break;
        }
        let lower = line.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("range:") {
            let v = v.trim().trim_start_matches("bytes=");
            let mut it = v.split('-');
            let a = it.next().and_then(|s| s.parse().ok());
            let b = it.next().and_then(|s| s.parse().ok());
            if let (Some(a), Some(b)) = (a, b) {
                range = Some((a, b));
            }
        }
        if lower.starts_with("authorization:") {
            auth = Some(line[14..].trim().to_string());
        }
    }
    requests.fetch_add(1, Ordering::Relaxed);
    let mut parts = first.split_whitespace();
    let method = parts.next().unwrap_or("");
    let target = parts.next().unwrap_or("/");
    if let Some(id) = require {
        let ok = auth.as_deref().is_some_and(|a| {
            a.starts_with("AWS4-HMAC-SHA256 Credential=")
                && a.contains(&format!("Credential={id}/"))
                && a.contains("SignedHeaders=host;x-amz-content-sha256;x-amz-date")
                && a.contains("Signature=")
        });
        if !ok {
            reply(&mut conn, "403 Forbidden", "", b"<Error><Code>AccessDenied</Code><Message>bad signature</Message></Error>");
            return;
        }
    }
    let (path, query) = match target.find('?') {
        Some(i) => (&target[..i], &target[i + 1..]),
        None => (target, ""),
    };
    // Path style: /bucket/key
    let key = path.trim_start_matches('/').splitn(2, '/').nth(1).unwrap_or("").to_string();
    let key = percent_decode(&key);
    if query.contains("list-type=2") {
        let prefix = query
            .split('&')
            .find_map(|kv| kv.strip_prefix("prefix="))
            .map(percent_decode)
            .unwrap_or_default();
        let b = bucket.lock().unwrap();
        let mut xml = String::from("<?xml version=\"1.0\"?><ListBucketResult><IsTruncated>false</IsTruncated>");
        let mut prefixes = Vec::new();
        for (k, v) in b.iter().filter(|(k, _)| k.starts_with(&prefix)) {
            let rest = &k[prefix.len()..];
            match rest.find('/') {
                Some(i) => {
                    let p = format!("{prefix}{}/", &rest[..i]);
                    if !prefixes.contains(&p) {
                        prefixes.push(p);
                    }
                }
                None => xml.push_str(&format!("<Contents><Key>{k}</Key><Size>{}</Size></Contents>", v.len())),
            }
        }
        for p in prefixes {
            xml.push_str(&format!("<CommonPrefixes><Prefix>{p}</Prefix></CommonPrefixes>"));
        }
        xml.push_str("</ListBucketResult>");
        reply(&mut conn, "200 OK", "Content-Type: application/xml\r\n", xml.as_bytes());
        return;
    }
    let body = bucket.lock().unwrap().get(&key).cloned();
    let Some(body) = body else {
        reply(&mut conn, "404 Not Found", "", b"<Error><Code>NoSuchKey</Code></Error>");
        return;
    };
    let etag = format!("ETag: \"{:x}\"\r\n", body.len() ^ 0xabcdef);
    if method == "HEAD" {
        let head = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n{etag}Connection: close\r\n\r\n",
            body.len()
        );
        let _ = conn.write_all(head.as_bytes());
        return;
    }
    match range {
        Some((a, b)) if (a as usize) < body.len() => {
            let b = (b as usize).min(body.len() - 1);
            let slice = &body[a as usize..=b];
            sent.fetch_add(slice.len() as u64, Ordering::Relaxed);
            reply(
                &mut conn,
                "206 Partial Content",
                &format!("{etag}Content-Range: bytes {a}-{b}/{}\r\n", body.len()),
                slice,
            );
        }
        _ => {
            sent.fetch_add(body.len() as u64, Ordering::Relaxed);
            reply(&mut conn, "200 OK", &etag, &body);
        }
    }
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() + 0 && i + 2 <= b.len() - 1 {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// A two-part EXR, each part a flat colour, float32 uncompressed so each
/// part is a known 4 MiB — big enough that fetching one part and not the
/// other shows in the byte count.
fn two_part_exr(w: usize, h: usize, a: f32, b: f32) -> Vec<u8> {
    use exr::prelude::*;
    let chan = |name: &str, v: f32| AnyChannel::new(name, FlatSamples::F32(vec![v; w * h]));
    let part = |name: &str, v: f32| {
        Layer::new(
            (w, h),
            LayerAttributes::named(name),
            Encoding::UNCOMPRESSED,
            AnyChannels::sort(
                vec![chan("R", v), chan("G", v * 0.5), chan("B", v * 0.25), chan("A", 1.0)]
                    .into_iter()
                    .collect(),
            ),
        )
    };
    let layers: Layers<AnyChannels<FlatSamples>> = vec![part("beauty", a), part("spec", b)].into_iter().collect();
    let image = Image::from_layers(ImageAttributes::new(IntegerBounds::from_dimensions((w, h))), layers);
    let mut out = Cursor::new(Vec::new());
    image.write().to_buffered(&mut out).unwrap();
    out.into_inner()
}

fn config_for(port: u16, cache: &Path, key: &str) -> Config {
    Config {
        default: Endpoint {
            url: Some(format!("http://127.0.0.1:{port}")),
            region: Some("auto".into()),
            access_key: Some(key.into()),
            secret_key: Some("SECRET".into()),
            path_style: Some(true),
        },
        buckets: Default::default(),
        cache_dir: cache.to_path_buf(),
        cache_bytes: 1 << 30,
    }
}

fn px(buf: &[f16], stride: usize, x: usize, y: usize) -> [f32; 4] {
    let i = y * stride + x * 4;
    [buf[i].to_f32(), buf[i + 1].to_f32(), buf[i + 2].to_f32(), buf[i + 3].to_f32()]
}

/// One test: the configuration and the caches are process-wide.
#[test]
fn a_plate_in_a_bucket_reads_like_a_directory() {
    let (w, h) = (512usize, 512usize);
    let bucket: Bucket = Default::default();
    for (i, v) in [(1001, 0.1f32), (1002, 0.2), (1003, 0.3)] {
        bucket.lock().unwrap().insert(format!("shots/sh010/plate.{i}.exr"), two_part_exr(w, h, v, 0.9));
    }
    bucket.lock().unwrap().insert("shots/sh010/notes.txt".into(), b"hi".to_vec());
    bucket.lock().unwrap().insert("shots/sh020/other.1001.exr".into(), two_part_exr(8, 8, 0.5, 0.5));
    let srv = serve(bucket.clone(), Some("AKTEST"));
    let cache = std::env::temp_dir().join(format!("wobble-objstore-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&cache);
    objstore::set_config(config_for(srv.port, &cache, "AKTEST"));
    objstore::reset_stats();

    let dec = ExrsDecoder::default();
    let dir = Path::new("s3://plates/shots/sh010");

    // The same file locally, so a remote mismatch is the transport's fault.
    let local = cache.join("local.exr");
    std::fs::create_dir_all(&cache).unwrap();
    std::fs::write(&local, &bucket.lock().unwrap()["shots/sh010/plate.1002.exr"]).unwrap();
    let linfo = dec.read_header(&local).unwrap();
    let lstride = linfo.row_stride_f16();
    let mut lbuf = vec![f16::ZERO; lstride * h];
    dec.decode_layer_into(&local, linfo, lstride, Some("beauty"), &mut lbuf).unwrap();
    let lp = px(&lbuf, lstride, 7, 7);
    assert!((lp[0] - 0.2).abs() < 1e-3, "LOCAL beauty {lp:?}");

    // --- the directory: only the images, sorted, header from the first.
    assert!(vfs::is_dir(dir));
    assert!(!vfs::is_dir(Path::new("s3://plates/shots/nothing")));
    let seq = Sequence::from_dir(dir, &dec).expect("sequence from a bucket prefix");
    assert_eq!(seq.frames.len(), 3);
    assert_eq!(seq.frames[0], PathBuf::from("s3://plates/shots/sh010/plate.1001.exr"));
    assert_eq!((seq.info.width, seq.info.height), (512, 512));
    let listed = vfs::list(Path::new("s3://plates/shots")).unwrap();
    assert!(listed.contains(&PathBuf::from("s3://plates/shots/sh010")), "{listed:?}");
    assert!(listed.contains(&PathBuf::from("s3://plates/shots/sh020")), "{listed:?}");

    // --- one layer of a two-part file costs that part, not the file.
    let frame = &seq.frames[1];
    let size = bucket.lock().unwrap()["shots/sh010/plate.1002.exr"].len() as u64;
    let names: Vec<String> = dec.layers(frame).unwrap().iter().map(|l| l.label().to_string()).collect();
    assert_eq!(names, ["beauty", "spec"]);
    let stride = seq.info.row_stride_f16();
    let mut buf = vec![f16::ZERO; stride * h];
    srv.sent.store(0, Ordering::Relaxed);
    dec.decode_layer_into(frame, seq.info, stride, Some("spec"), &mut buf).unwrap();
    let p = px(&buf, stride, 100, 200);
    assert!((p[0] - 0.9).abs() < 1e-3 && (p[1] - 0.45).abs() < 1e-3, "{p:?}");
    let fetched = srv.sent.load(Ordering::Relaxed);
    assert!(
        fetched < size * 3 / 4,
        "one part of two fetched {fetched} of {size} bytes — the range reads are not selective"
    );
    assert!(fetched >= 4 << 20, "a 4 MiB part cannot arrive in {fetched} bytes");

    // --- the same bytes again: nothing leaves the machine.
    let before = srv.requests.load(Ordering::Relaxed);
    dec.decode_layer_into(frame, seq.info, stride, Some("spec"), &mut buf).unwrap();
    assert_eq!(srv.requests.load(Ordering::Relaxed), before, "a cached decode made a request");
    // The other part is fetched on demand and the picture is right.
    dec.decode_layer_into(frame, seq.info, stride, Some("beauty"), &mut buf).unwrap();
    let p = px(&buf, stride, 7, 7);
    assert!((p[0] - 0.2).abs() < 1e-3, "{p:?}");

    // --- the disk cache outlives the in-process one.
    objstore::set_config(config_for(srv.port, &cache, "AKTEST"));
    let before_sent = srv.sent.load(Ordering::Relaxed);
    dec.decode_layer_into(frame, seq.info, stride, Some("spec"), &mut buf).unwrap();
    assert_eq!(srv.sent.load(Ordering::Relaxed), before_sent, "a second process re-fetched cached bytes");
    assert!((px(&buf, stride, 1, 1)[0] - 0.9).abs() < 1e-3);

    // --- a frame that is not there yet, then is (a render arriving).
    let later = Path::new("s3://plates/shots/sh010/plate.1004.exr");
    assert!(!vfs::exists(later));
    bucket.lock().unwrap().insert("shots/sh010/plate.1004.exr".into(), two_part_exr(8, 8, 0.4, 0.4));
    std::thread::sleep(std::time::Duration::from_millis(2100));
    assert!(vfs::exists(later), "a frame written after the first look is never seen");
    let mut small = vec![f16::ZERO; 8 * 4 * 8];
    let info = dec.read_header(later).unwrap();
    dec.decode_into(later, info, 8 * 4, &mut small).unwrap();
    assert!((small[0].to_f32() - 0.4).abs() < 1e-3);

    // --- whole-file read, for the decoders that take bytes.
    let bytes = vfs::read(Path::new("s3://plates/shots/sh010/notes.txt")).unwrap();
    assert_eq!(bytes, b"hi");
    assert!(vfs::open(Path::new("s3://plates/shots/sh010/missing.exr")).is_err());

    // --- wrong credentials fail with a reason a person can act on.
    objstore::set_config(config_for(srv.port, &cache, "SOMEONE_ELSE"));
    let err = match vfs::open(Path::new("s3://plates/shots/sh010/plate.1001.exr")) { Err(e) => format!("{e:#}"), Ok(_) => panic!("opened with the wrong key") };
    let err = format!("{err:#}");
    assert!(err.contains("403") && err.contains("credentials"), "{err}");

    // --- an anonymous bucket needs no key at all.
    let public = serve(bucket.clone(), None);
    let mut c = config_for(public.port, &cache, "");
    c.default.access_key = None;
    c.default.secret_key = None;
    objstore::set_config(c);
    let mut r = vfs::open(Path::new("s3://plates/shots/sh010/notes.txt")).unwrap();
    let mut s = String::new();
    r.read_to_string(&mut s).unwrap();
    assert_eq!(s, "hi");
    assert!(objstore::describe("plates").contains("anonymous"));

    let _ = std::fs::remove_dir_all(&cache);
    let _ = srv.require;
}
