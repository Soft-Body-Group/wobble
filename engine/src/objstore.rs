//! `s3://bucket/key` behind [`crate::vfs`]: plates that live in object
//! storage (Cloudflare R2, AWS S3, MinIO, Garage — anything speaking the S3
//! API) read like a directory of files.
//!
//! Object storage is not a filesystem: there is no `open`, no `seek`, only
//! `GET` over HTTP — but a `GET` can ask for a byte RANGE, and that is what
//! makes it work here. [`RemoteReader`] is a `Read + Seek` over one object
//! that fetches the bytes a read actually touches, in whole [`UNIT`]s,
//! growing the fetch while the reads stay sequential. The EXR decoder
//! reads the header and the offset table, then seeks to the chunks of the
//! ONE part it wants (`decode_part_into` is a filtered chunk read), so a
//! Read of one layer of a 20-AOV multi-part pulls that layer's bytes and
//! not the file. Every unit fetched lands in an on-disk cache
//! (`~/.wobble/objcache`, one sparse file + a bitmap per object, keyed by
//! the object's ETag so an overwritten frame is a different entry), so the
//! second decode of a frame reads from NVMe with no request at all.
//!
//! Requests are signed with AWS Signature V4 when credentials are
//! configured and sent plain when they are not (a public bucket). The
//! signer is here in full — ~80 lines against a dependency tree — and is
//! pinned to AWS's own published test vector.
//!
//! Configuration (`~/.wobble/storage.json`, see [`Config`]) names an
//! endpoint, a region and credentials, per bucket or as a default; the
//! usual `AWS_*` environment variables override it. A setup file carries
//! only the `s3://` path — never a key, never an endpoint — so a comp
//! moves between machines and the credentials stay on each one.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{anyhow, Context, Result};
use sha2::{Digest, Sha256};

/// The path prefix that routes a path here.
pub const SCHEME: &str = "s3://";
/// Cache granularity: a fetch covers whole units, and the bitmap has one
/// bit per unit. 1 MiB is a few scanline chunks of a 2K half-float EXR —
/// big enough that a request is not all latency, small enough that a
/// one-layer read of a multi-part file leaves most of it unfetched.
pub const UNIT: u64 = 1 << 20;
/// Readahead cap: consecutive sequential reads double their fetch up to
/// this, so a whole-file decode is ~size/32MiB requests, not size/1MiB.
const MAX_SPAN: u64 = 32 << 20;
/// How long a HEAD that found the object is trusted. A frame that exists
/// keeps existing; the cost of trusting it is a rewritten frame read
/// from the old entry for this long.
const POSITIVE_TTL: Duration = Duration::from_secs(30);
/// How long a HEAD that found nothing is trusted. Short, because a render
/// arriving is exactly the case a Read's range past its source exists
/// for (see `Sequence::trimmed`).
const NEGATIVE_TTL: Duration = Duration::from_secs(2);
/// How long a listing is trusted (the browser lists on every repaint).
const LIST_TTL: Duration = Duration::from_secs(5);
/// Default on-disk cache budget.
pub const DEFAULT_CACHE_BYTES: u64 = 20 << 30;

const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// Is this a path this module answers for?
pub fn is_remote(path: &Path) -> bool {
    path.to_str().is_some_and(|s| s.starts_with(SCHEME))
}

/// One object (or prefix) address: `s3://bucket/key`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Loc {
    pub bucket: String,
    /// May be empty (the bucket root) and may end in `/` (a prefix).
    pub key: String,
}

impl Loc {
    pub fn parse(path: &Path) -> Result<Loc> {
        let s = path.to_str().ok_or_else(|| anyhow!("{} is not UTF-8", path.display()))?;
        let rest = s
            .strip_prefix(SCHEME)
            .ok_or_else(|| anyhow!("{s} is not an s3:// path"))?;
        let (bucket, key) = match rest.find('/') {
            Some(i) => (&rest[..i], rest[i + 1..].to_string()),
            None => (rest, String::new()),
        };
        anyhow::ensure!(
            !bucket.is_empty()
                && bucket.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.'),
            "{s}: not a bucket name"
        );
        Ok(Loc { bucket: bucket.to_string(), key })
    }

    /// The `s3://` spelling, as a path the rest of the app can hold.
    pub fn to_path(&self) -> PathBuf {
        PathBuf::from(format!("{SCHEME}{}/{}", self.bucket, self.key))
    }

    fn url(&self) -> String {
        format!("{SCHEME}{}/{}", self.bucket, self.key)
    }
}

// ---------------------------------------------------------------- config

/// Where a bucket lives and how to talk to it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Endpoint {
    /// `https://<account>.r2.cloudflarestorage.com`, `http://127.0.0.1:9000`
    /// … `None` means AWS itself (`https://<bucket>.s3.<region>.amazonaws.com`).
    pub url: Option<String>,
    /// SigV4 region. R2 takes `auto`, MinIO ignores it, AWS needs the real one.
    pub region: Option<String>,
    pub access_key: Option<String>,
    pub secret_key: Option<String>,
    /// `{endpoint}/{bucket}/{key}` rather than `{bucket}.{endpoint}/{key}`.
    /// Defaults to true when an endpoint is given (R2, MinIO, Garage), false
    /// for AWS.
    pub path_style: Option<bool>,
}

impl Endpoint {
    /// `other`'s set fields over `self`'s.
    fn over(&self, other: &Endpoint) -> Endpoint {
        Endpoint {
            url: other.url.clone().or_else(|| self.url.clone()),
            region: other.region.clone().or_else(|| self.region.clone()),
            access_key: other.access_key.clone().or_else(|| self.access_key.clone()),
            secret_key: other.secret_key.clone().or_else(|| self.secret_key.clone()),
            path_style: other.path_style.or(self.path_style),
        }
    }
}

/// The whole configuration: a default endpoint, per-bucket overrides and
/// the cache. Built from `~/.wobble/storage.json` with the environment
/// over it ([`Config::load`]), or handed in whole ([`set_config`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    pub default: Endpoint,
    pub buckets: BTreeMap<String, Endpoint>,
    pub cache_dir: PathBuf,
    pub cache_bytes: u64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            default: Endpoint::default(),
            buckets: BTreeMap::new(),
            cache_dir: default_cache_dir(),
            cache_bytes: DEFAULT_CACHE_BYTES,
        }
    }
}

/// `$WOBBLE_STORAGE_CONFIG`, else `~/.wobble/storage.json`.
pub fn config_path() -> PathBuf {
    if let Some(p) = std::env::var_os("WOBBLE_STORAGE_CONFIG") {
        return PathBuf::from(p);
    }
    crate::paths::home().unwrap_or_else(|| PathBuf::from(".")).join(".wobble/storage.json")
}

/// `$WOBBLE_OBJCACHE_DIR`, else `~/.wobble/objcache`.
pub fn default_cache_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("WOBBLE_OBJCACHE_DIR") {
        return PathBuf::from(p);
    }
    crate::paths::home().unwrap_or_else(|| PathBuf::from(".")).join(".wobble/objcache")
}

impl Config {
    /// The file, then the environment over its default endpoint:
    /// `WOBBLE_S3_ENDPOINT` / `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`,
    /// `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` / `AWS_DEFAULT_REGION`,
    /// `WOBBLE_OBJCACHE_DIR`, `WOBBLE_OBJCACHE_BYTES`.
    pub fn load() -> Config {
        let mut c = Config::from_file(&config_path()).unwrap_or_default();
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        let env = Endpoint {
            url: var("WOBBLE_S3_ENDPOINT").or_else(|| var("AWS_ENDPOINT_URL")),
            region: var("AWS_REGION").or_else(|| var("AWS_DEFAULT_REGION")),
            access_key: var("AWS_ACCESS_KEY_ID"),
            secret_key: var("AWS_SECRET_ACCESS_KEY"),
            path_style: None,
        };
        c.default = c.default.over(&env);
        c.cache_dir = default_cache_dir_over(c.cache_dir);
        if let Some(b) = var("WOBBLE_OBJCACHE_BYTES").and_then(|v| v.parse().ok()) {
            c.cache_bytes = b;
        }
        c
    }

    /// Parse `storage.json`. Needs the `serde` feature for the JSON
    /// reader; without it the environment is the whole configuration.
    pub fn from_file(path: &Path) -> Option<Config> {
        #[cfg(feature = "serde")]
        {
            let text = fs::read_to_string(path).ok()?;
            let v: serde_json::Value = match serde_json::from_str(&text) {
                Ok(v) => v,
                Err(e) => {
                    log::warn!("{}: {e}", path.display());
                    return None;
                }
            };
            let ep = |v: &serde_json::Value| Endpoint {
                url: v.get("endpoint").and_then(|x| x.as_str()).map(str::to_string),
                region: v.get("region").and_then(|x| x.as_str()).map(str::to_string),
                access_key: v.get("access_key").and_then(|x| x.as_str()).map(str::to_string),
                secret_key: v.get("secret_key").and_then(|x| x.as_str()).map(str::to_string),
                path_style: v.get("path_style").and_then(|x| x.as_bool()),
            };
            let mut c = Config { default: ep(&v), ..Config::default() };
            if let Some(b) = v.get("buckets").and_then(|b| b.as_object()) {
                for (name, e) in b {
                    c.buckets.insert(name.clone(), ep(e));
                }
            }
            if let Some(d) = v.get("cache_dir").and_then(|x| x.as_str()) {
                c.cache_dir = crate::paths::expand_home(d);
            }
            if let Some(b) = v.get("cache_bytes").and_then(|x| x.as_u64()) {
                c.cache_bytes = b;
            }
            Some(c)
        }
        #[cfg(not(feature = "serde"))]
        {
            let _ = path;
            None
        }
    }

    /// Everything a request to `bucket` needs.
    fn resolve(&self, bucket: &str) -> Resolved {
        let e = match self.buckets.get(bucket) {
            Some(b) => self.default.over(b),
            None => self.default.clone(),
        };
        let region = e.region.clone().unwrap_or_else(|| "us-east-1".into());
        let path_style = e.path_style.unwrap_or(e.url.is_some());
        let base = match &e.url {
            Some(u) => {
                let u = u.trim_end_matches('/');
                if path_style {
                    format!("{u}/{bucket}")
                } else {
                    // `https://host` -> `https://bucket.host`.
                    match u.find("://") {
                        Some(i) => format!("{}://{bucket}.{}", &u[..i], &u[i + 3..]),
                        None => format!("https://{bucket}.{u}"),
                    }
                }
            }
            None => format!("https://{bucket}.s3.{region}.amazonaws.com"),
        };
        let host = base
            .split("://")
            .nth(1)
            .unwrap_or(&base)
            .split('/')
            .next()
            .unwrap_or_default()
            .to_string();
        let creds = match (e.access_key, e.secret_key) {
            (Some(a), Some(s)) => Some(Credentials { access_key: a, secret_key: s }),
            _ => None,
        };
        Resolved { base, host, region, creds, path_style }
    }
}

fn default_cache_dir_over(from_file: PathBuf) -> PathBuf {
    match std::env::var_os("WOBBLE_OBJCACHE_DIR") {
        Some(p) => PathBuf::from(p),
        None => from_file,
    }
}

#[derive(Clone, Debug)]
struct Credentials {
    access_key: String,
    secret_key: String,
}

#[derive(Clone, Debug)]
struct Resolved {
    /// `https://host/bucket` (path style) or `https://bucket.host`.
    base: String,
    host: String,
    region: String,
    creds: Option<Credentials>,
    path_style: bool,
}

impl Resolved {
    /// The request path as SigV4 canonicalises it (segments encoded, `/` kept).
    fn canonical_path(&self, bucket: &str, key: &str) -> String {
        let key_part = key.split('/').map(|s| uri_encode(s, true)).collect::<Vec<_>>().join("/");
        let prefix = match self.base.find("://").map(|i| &self.base[i + 3..]) {
            // Path-style base may carry a path of its own (`host/base/bucket`).
            Some(rest) => match rest.find('/') {
                Some(i) => rest[i..].to_string(),
                None => String::new(),
            },
            None => String::new(),
        };
        let _ = bucket;
        format!("{prefix}/{key_part}")
    }

    fn url(&self, key: &str, query: &[(&str, String)]) -> String {
        let key_part = key.split('/').map(|s| uri_encode(s, true)).collect::<Vec<_>>().join("/");
        let mut u = format!("{}/{key_part}", self.base);
        if !query.is_empty() {
            u.push('?');
            u.push_str(&canonical_query(query));
        }
        u
    }
}

static CONFIG: RwLock<Option<Arc<Config>>> = RwLock::new(None);

/// The live configuration (loaded on first use).
pub fn config() -> Arc<Config> {
    if let Some(c) = CONFIG.read().ok().and_then(|g| g.clone()) {
        return c;
    }
    let c = Arc::new(Config::load());
    if let Ok(mut g) = CONFIG.write() {
        if g.is_none() {
            *g = Some(c.clone());
        }
    }
    c
}

/// Replace the configuration (tests; a settings panel). Drops every
/// in-process cache, so the next request goes where the new config says.
pub fn set_config(c: Config) {
    if let Ok(mut g) = CONFIG.write() {
        *g = Some(Arc::new(c));
    }
    forget_all();
}

/// Re-read the file and environment on the next request.
pub fn reload() {
    if let Ok(mut g) = CONFIG.write() {
        *g = None;
    }
    forget_all();
}

fn forget_all() {
    if let Ok(mut h) = HEADS.lock() {
        h.take();
    }
    if let Ok(mut l) = LISTS.lock() {
        l.take();
    }
    if let Ok(mut o) = OBJECTS.lock() {
        o.take();
    }
}

// ---------------------------------------------------------------- SigV4

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn sha256_hex(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}

fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    const BLOCK: usize = 64;
    let mut k = [0u8; BLOCK];
    if key.len() > BLOCK {
        k[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; BLOCK];
    let mut opad = [0x5cu8; BLOCK];
    for i in 0..BLOCK {
        ipad[i] ^= k[i];
        opad[i] ^= k[i];
    }
    let inner = Sha256::new().chain_update(ipad).chain_update(msg).finalize();
    Sha256::new().chain_update(opad).chain_update(inner).finalize().into()
}

/// AWS's URI encoding: unreserved characters kept, everything else `%XX`
/// (upper-case hex), `/` kept unless `encode_slash`.
fn uri_encode(s: &str, encode_slash: bool) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        let keep = b.is_ascii_alphanumeric()
            || b == b'-'
            || b == b'_'
            || b == b'.'
            || b == b'~'
            || (b == b'/' && !encode_slash);
        if keep {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn canonical_query(query: &[(&str, String)]) -> String {
    let mut q: Vec<(String, String)> =
        query.iter().map(|(k, v)| (uri_encode(k, true), uri_encode(v, true))).collect();
    q.sort();
    q.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("&")
}

/// `YYYYMMDDTHHMMSSZ` for a UNIX time.
fn amz_date(secs: u64) -> String {
    let days = secs / 86400;
    let rem = secs % 86400;
    // Howard Hinnant's days -> civil.
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let mut y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    if m <= 2 {
        y += 1;
    }
    format!(
        "{y:04}{m:02}{d:02}T{:02}{:02}{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// What a request is, for signing.
struct SignInput<'a> {
    method: &'a str,
    host: &'a str,
    /// Raw path (unencoded segments), leading `/`.
    path: &'a str,
    query: &'a [(&'a str, String)],
    /// Extra headers to sign, lower-case names. `host` is added.
    headers: &'a [(&'a str, &'a str)],
    payload_hash: &'a str,
    amz_date: &'a str,
    region: &'a str,
    service: &'a str,
}

/// The `Authorization` header value.
fn authorization(i: &SignInput, creds: &Credentials) -> String {
    let mut headers: Vec<(String, String)> = i
        .headers
        .iter()
        .map(|(k, v)| (k.to_ascii_lowercase(), v.split_whitespace().collect::<Vec<_>>().join(" ")))
        .collect();
    headers.push(("host".into(), i.host.to_string()));
    headers.sort();
    let canonical_headers: String =
        headers.iter().map(|(k, v)| format!("{k}:{v}\n")).collect();
    let signed: String = headers.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>().join(";");
    let canonical_uri = if i.path.is_empty() {
        "/".to_string()
    } else {
        i.path.split('/').map(|s| uri_encode(s, true)).collect::<Vec<_>>().join("/")
    };
    let canonical = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        i.method,
        canonical_uri,
        canonical_query(i.query),
        canonical_headers,
        signed,
        i.payload_hash
    );
    let date = &i.amz_date[..8];
    let scope = format!("{date}/{}/{}/aws4_request", i.region, i.service);
    let to_sign = format!(
        "AWS4-HMAC-SHA256\n{}\n{scope}\n{}",
        i.amz_date,
        sha256_hex(canonical.as_bytes())
    );
    let k_date = hmac_sha256(format!("AWS4{}", creds.secret_key).as_bytes(), date.as_bytes());
    let k_region = hmac_sha256(&k_date, i.region.as_bytes());
    let k_service = hmac_sha256(&k_region, i.service.as_bytes());
    let k_signing = hmac_sha256(&k_service, b"aws4_request");
    let signature = hex(&hmac_sha256(&k_signing, to_sign.as_bytes()));
    format!(
        "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed}, Signature={signature}",
        creds.access_key
    )
}

// ---------------------------------------------------------------- HTTP

/// Requests made and bytes fetched over the process's life — a UI readout
/// and the tests' proof that a cache hit costs nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub requests: u64,
    pub bytes: u64,
}

static REQUESTS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

pub fn stats() -> Stats {
    Stats { requests: REQUESTS.load(Ordering::Relaxed), bytes: BYTES.load(Ordering::Relaxed) }
}

pub fn reset_stats() {
    REQUESTS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
}

fn agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_connect(Some(Duration::from_secs(20)))
            .timeout_global(Some(Duration::from_secs(120)))
            .user_agent(concat!("wobble/", env!("CARGO_PKG_VERSION")))
            .build()
            .into()
    })
}

struct Resp {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Resp {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

/// One signed (or anonymous) request.
fn call(
    method: &str,
    r: &Resolved,
    bucket: &str,
    key: &str,
    query: &[(&str, String)],
    range: Option<(u64, u64)>,
) -> Result<Resp> {
    let url = r.url(key, query);
    let mut req = match method {
        "HEAD" => agent().head(&url),
        _ => agent().get(&url),
    };
    let range_v = range.map(|(a, b)| format!("bytes={a}-{b}"));
    if let Some(v) = &range_v {
        req = req.header("range", v);
    }
    req = req.header("accept-encoding", "identity");
    if let Some(creds) = &r.creds {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let date = amz_date(now);
        let path = r.canonical_path(bucket, key);
        let headers = [("x-amz-content-sha256", EMPTY_SHA256), ("x-amz-date", date.as_str())];
        let auth = authorization(
            &SignInput {
                method,
                host: &r.host,
                path: &path,
                query,
                headers: &headers,
                payload_hash: EMPTY_SHA256,
                amz_date: &date,
                region: &r.region,
                service: "s3",
            },
            creds,
        );
        req = req
            .header("x-amz-date", &date)
            .header("x-amz-content-sha256", EMPTY_SHA256)
            .header("authorization", &auth);
    }
    REQUESTS.fetch_add(1, Ordering::Relaxed);
    let mut resp = req.call().map_err(|e| anyhow!("{method} {url}: {e}"))?;
    let status = resp.status().as_u16();
    let headers = resp
        .headers()
        .iter()
        .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
        .collect();
    let body = if method == "HEAD" {
        Vec::new()
    } else {
        resp.body_mut()
            .with_config()
            .limit(u64::MAX)
            .read_to_vec()
            .with_context(|| format!("{method} {url}: reading the body"))?
    };
    BYTES.fetch_add(body.len() as u64, Ordering::Relaxed);
    Ok(Resp { status, headers, body })
}

fn err_of(what: &str, loc: &Loc, resp: &Resp) -> anyhow::Error {
    let text = String::from_utf8_lossy(&resp.body);
    let code = tag_text(&text, "Code").unwrap_or_default();
    let msg = tag_text(&text, "Message").unwrap_or_default();
    let detail = match (code.is_empty(), msg.is_empty()) {
        (true, true) => String::new(),
        (false, true) => format!(": {code}"),
        _ => format!(": {code} — {msg}"),
    };
    let hint = match resp.status {
        403 => " (check the credentials in ~/.wobble/storage.json or AWS_* env)",
        _ => "",
    };
    anyhow!("{what} {} returned HTTP {}{detail}{hint}", loc.url(), resp.status)
}

// ---------------------------------------------------------------- HEAD

/// What a HEAD says about an object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjMeta {
    pub size: u64,
    pub etag: String,
}

static HEADS: Mutex<Option<HashMap<String, (Instant, Option<ObjMeta>)>>> = Mutex::new(None);

/// Size + ETag, or `None` when there is no such object. Cached
/// ([`POSITIVE_TTL`] / [`NEGATIVE_TTL`]).
pub fn head(loc: &Loc) -> Result<Option<ObjMeta>> {
    let url = loc.url();
    if let Ok(mut g) = HEADS.lock() {
        if let Some((at, m)) = g.get_or_insert_with(HashMap::new).get(&url) {
            let ttl = if m.is_some() { POSITIVE_TTL } else { NEGATIVE_TTL };
            if at.elapsed() < ttl {
                return Ok(m.clone());
            }
        }
    }
    let r = config().resolve(&loc.bucket);
    let resp = call("HEAD", &r, &loc.bucket, &loc.key, &[], None)?;
    let meta = match resp.status {
        200 => {
            let size = resp
                .header("content-length")
                .and_then(|v| v.trim().parse::<u64>().ok())
                .ok_or_else(|| anyhow!("HEAD {}: no Content-Length", url))?;
            let etag = resp.header("etag").unwrap_or("").trim_matches('"').to_string();
            Some(ObjMeta { size, etag })
        }
        404 => None,
        _ => return Err(err_of("HEAD", loc, &resp)),
    };
    if let Ok(mut g) = HEADS.lock() {
        g.get_or_insert_with(HashMap::new).insert(url, (Instant::now(), meta.clone()));
    }
    Ok(meta)
}

/// Does the object exist?
pub fn exists(loc: &Loc) -> bool {
    !loc.key.is_empty() && !loc.key.ends_with('/') && matches!(head(loc), Ok(Some(_)))
}

// ---------------------------------------------------------------- LIST

static LISTS: Mutex<Option<HashMap<String, (Instant, Vec<PathBuf>)>>> = Mutex::new(None);

/// The objects and "sub-directories" (common prefixes) directly under a
/// prefix, as `s3://` paths — what `read_dir` gives for a directory.
pub fn list(loc: &Loc) -> Result<Vec<PathBuf>> {
    let prefix = if loc.key.is_empty() || loc.key.ends_with('/') {
        loc.key.clone()
    } else {
        format!("{}/", loc.key)
    };
    let cache_key = format!("{SCHEME}{}/{prefix}", loc.bucket);
    if let Ok(mut g) = LISTS.lock() {
        if let Some((at, v)) = g.get_or_insert_with(HashMap::new).get(&cache_key) {
            if at.elapsed() < LIST_TTL {
                return Ok(v.clone());
            }
        }
    }
    let r = config().resolve(&loc.bucket);
    let mut out = Vec::new();
    let mut token: Option<String> = None;
    loop {
        let mut query: Vec<(&str, String)> = vec![
            ("list-type", "2".into()),
            ("delimiter", "/".into()),
            ("max-keys", "1000".into()),
        ];
        if !prefix.is_empty() {
            query.push(("prefix", prefix.clone()));
        }
        if let Some(t) = &token {
            query.push(("continuation-token", t.clone()));
        }
        let resp = call("GET", &r, &loc.bucket, "", &query, None)?;
        if resp.status != 200 {
            return Err(err_of("LIST", loc, &resp));
        }
        let xml = String::from_utf8_lossy(&resp.body);
        for block in tag_blocks(&xml, "Contents") {
            if let Some(k) = tag_text(block, "Key") {
                if k != prefix {
                    out.push(PathBuf::from(format!("{SCHEME}{}/{k}", loc.bucket)));
                }
            }
        }
        for block in tag_blocks(&xml, "CommonPrefixes") {
            if let Some(p) = tag_text(block, "Prefix") {
                let p = p.trim_end_matches('/');
                out.push(PathBuf::from(format!("{SCHEME}{}/{p}", loc.bucket)));
            }
        }
        let truncated = tag_text(&xml, "IsTruncated").is_some_and(|t| t == "true");
        token = tag_text(&xml, "NextContinuationToken");
        if !truncated || token.is_none() {
            break;
        }
    }
    if let Ok(mut g) = LISTS.lock() {
        g.get_or_insert_with(HashMap::new).insert(cache_key, (Instant::now(), out.clone()));
    }
    Ok(out)
}

/// Is there anything under this prefix? (`Path::is_dir` for an `s3://`.)
pub fn is_dir(loc: &Loc) -> bool {
    list(loc).map(|v| !v.is_empty()).unwrap_or(false)
}

/// Every `<tag>…</tag>` block's inner text, in order (no nesting of the
/// same tag — S3's listings have none).
fn tag_blocks<'a>(xml: &'a str, tag: &str) -> Vec<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(i) = rest.find(&open) {
        let after = &rest[i + open.len()..];
        match after.find(&close) {
            Some(j) => {
                out.push(&after[..j]);
                rest = &after[j + close.len()..];
            }
            None => break,
        }
    }
    out
}

/// The first `<tag>…</tag>`'s text, unescaped.
fn tag_text(xml: &str, tag: &str) -> Option<String> {
    tag_blocks(xml, tag).first().map(|s| xml_unescape(s))
}

fn xml_unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        let after = &rest[i..];
        let Some(j) = after.find(';') else {
            out.push_str(after);
            return out;
        };
        let ent = &after[1..j];
        match ent {
            "lt" => out.push('<'),
            "gt" => out.push('>'),
            "amp" => out.push('&'),
            "quot" => out.push('"'),
            "apos" => out.push('\''),
            e if e.starts_with("#x") => {
                if let Some(c) = u32::from_str_radix(&e[2..], 16).ok().and_then(char::from_u32) {
                    out.push(c);
                }
            }
            e if e.starts_with('#') => {
                if let Some(c) = e[1..].parse::<u32>().ok().and_then(char::from_u32) {
                    out.push(c);
                }
            }
            _ => out.push_str(&after[..=j]),
        }
        rest = &after[j + 1..];
    }
    out.push_str(rest);
    out
}

// ---------------------------------------------------------------- cache

/// One object's on-disk cache: a sparse `data` file the fetched units are
/// written into at their offsets, and a `have` bitmap saying which.
struct Cached {
    loc: Loc,
    resolved: Resolved,
    meta: ObjMeta,
    file: File,
    have: Mutex<Vec<u8>>,
    have_path: PathBuf,
}

static OBJECTS: Mutex<Option<HashMap<String, Arc<Cached>>>> = Mutex::new(None);

fn read_at(file: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_at(buf, off)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        file.seek_read(buf, off)
    }
}

fn write_all_at(file: &File, mut buf: &[u8], mut off: u64) -> io::Result<()> {
    while !buf.is_empty() {
        #[cfg(unix)]
        let n = {
            use std::os::unix::fs::FileExt;
            file.write_at(buf, off)?
        };
        #[cfg(windows)]
        let n = {
            use std::os::windows::fs::FileExt;
            file.seek_write(buf, off)?
        };
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::WriteZero, "write_at wrote nothing"));
        }
        buf = &buf[n..];
        off += n as u64;
    }
    Ok(())
}

impl Cached {

    fn has(have: &[u8], unit: u64) -> bool {
        have.get((unit / 8) as usize).is_some_and(|b| b & (1 << (unit % 8)) != 0)
    }

    /// Make `[start, start + len)` resident: fetch the units it touches
    /// that are not yet here, each run of missing units in one request.
    fn ensure(&self, start: u64, len: u64) -> Result<()> {
        if len == 0 || start >= self.meta.size {
            return Ok(());
        }
        let end = (start + len).min(self.meta.size);
        let u0 = start / UNIT;
        let u1 = (end - 1) / UNIT;
        let missing: Vec<u64> = {
            let have = self.have.lock().map_err(|_| anyhow!("cache bitmap poisoned"))?;
            (u0..=u1).filter(|u| !Self::has(&have, *u)).collect()
        };
        if missing.is_empty() {
            return Ok(());
        }
        // Runs of consecutive missing units.
        let mut i = 0;
        while i < missing.len() {
            let mut j = i;
            while j + 1 < missing.len() && missing[j + 1] == missing[j] + 1 {
                j += 1;
            }
            let a = missing[i] * UNIT;
            let b = ((missing[j] + 1) * UNIT).min(self.meta.size) - 1;
            self.fetch(a, b)?;
            i = j + 1;
        }
        Ok(())
    }

    /// GET bytes `a..=b` and file them.
    fn fetch(&self, a: u64, b: u64) -> Result<()> {
        let resp = call("GET", &self.resolved, &self.loc.bucket, &self.loc.key, &[], Some((a, b)))?;
        match resp.status {
            206 => {}
            // The server ignored the range and sent everything: still usable
            // if it is the whole object.
            200 if resp.body.len() as u64 == self.meta.size => {
                write_all_at(&self.file, &resp.body, 0)?;
                let mut have = self.have.lock().map_err(|_| anyhow!("cache bitmap poisoned"))?;
                have.fill(0xff);
                self.persist(&have);
                return Ok(());
            }
            _ => return Err(err_of("GET", &self.loc, &resp)),
        }
        let want = (b - a + 1) as usize;
        anyhow::ensure!(
            resp.body.len() == want,
            "GET {} bytes {a}-{b}: got {} bytes, wanted {want} (did the object change under us?)",
            self.loc.url(),
            resp.body.len()
        );
        write_all_at(&self.file, &resp.body, a)
            .with_context(|| format!("writing the cache for {}", self.loc.url()))?;
        let mut have = self.have.lock().map_err(|_| anyhow!("cache bitmap poisoned"))?;
        for u in a / UNIT..=b / UNIT {
            have[(u / 8) as usize] |= 1 << (u % 8);
        }
        self.persist(&have);
        Ok(())
    }

    fn persist(&self, have: &[u8]) {
        if let Err(e) = fs::write(&self.have_path, have) {
            log::debug!("{}: {e}", self.have_path.display());
        }
    }
}

/// Cache directory for one object version.
fn cache_dir_for(cfg: &Config, loc: &Loc, meta: &ObjMeta) -> PathBuf {
    let id = sha256_hex(format!("{}|{}|{}", loc.url(), meta.etag, meta.size).as_bytes());
    cfg.cache_dir.join(&id[..32])
}

fn open_cached(loc: &Loc) -> Result<Arc<Cached>> {
    let meta = head(loc)?.ok_or_else(|| {
        anyhow!("{}: no such object", loc.url())
    })?;
    let id = format!("{}|{}|{}", loc.url(), meta.etag, meta.size);
    if let Ok(mut g) = OBJECTS.lock() {
        if let Some(c) = g.get_or_insert_with(HashMap::new).get(&id) {
            return Ok(c.clone());
        }
    }
    let cfg = config();
    let dir = cache_dir_for(&cfg, loc, &meta);
    let fresh = !dir.is_dir();
    fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let data = dir.join("data");
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&data)
        .with_context(|| format!("opening {}", data.display()))?;
    if file.metadata().map(|m| m.len()).unwrap_or(0) != meta.size {
        file.set_len(meta.size)?;
    }
    let have_path = dir.join("have");
    let nbytes = (meta.size.div_ceil(UNIT)).div_ceil(8) as usize;
    let have = match fs::read(&have_path) {
        Ok(v) if v.len() == nbytes => v,
        _ => vec![0u8; nbytes],
    };
    let c = Arc::new(Cached {
        loc: loc.clone(),
        resolved: cfg.resolve(&loc.bucket),
        meta,
        file,
        have: Mutex::new(have),
        have_path,
    });
    if let Ok(mut g) = OBJECTS.lock() {
        g.get_or_insert_with(HashMap::new).insert(id, c.clone());
    }
    if fresh {
        evict_to_budget(&cfg);
    }
    Ok(c)
}

/// Bytes an entry really occupies (a sparse file's logical size lies).
fn occupied(path: &Path) -> u64 {
    let Ok(m) = fs::metadata(path) else { return 0 };
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        m.blocks() * 512
    }
    #[cfg(not(unix))]
    {
        m.len()
    }
}

/// Drop the least recently touched entries until the cache fits its
/// budget. Entries open in this process are kept.
fn evict_to_budget(cfg: &Config) {
    let Ok(rd) = fs::read_dir(&cfg.cache_dir) else { return };
    let mut entries: Vec<(SystemTime, u64, PathBuf)> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .map(|p| {
            let data = p.join("data");
            let mtime = fs::metadata(&data)
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            (mtime, occupied(&data), p)
        })
        .collect();
    let mut total: u64 = entries.iter().map(|e| e.1).sum();
    if total <= cfg.cache_bytes {
        return;
    }
    let open: Vec<PathBuf> = OBJECTS
        .lock()
        .ok()
        .and_then(|g| {
            g.as_ref().map(|m| m.values().map(|c| cache_dir_for(cfg, &c.loc, &c.meta)).collect())
        })
        .unwrap_or_default();
    entries.sort();
    for (_, bytes, p) in entries {
        if total <= cfg.cache_bytes {
            break;
        }
        if open.contains(&p) {
            continue;
        }
        if fs::remove_dir_all(&p).is_ok() {
            total = total.saturating_sub(bytes);
        }
    }
}

/// Every cached byte, gone. (A settings button; tests.)
pub fn clear_cache() -> Result<()> {
    let cfg = config();
    if let Ok(mut o) = OBJECTS.lock() {
        o.take();
    }
    if cfg.cache_dir.is_dir() {
        fs::remove_dir_all(&cfg.cache_dir)?;
    }
    Ok(())
}

// ---------------------------------------------------------------- reader

/// `Read + Seek` over one object. Reads fetch what they touch (in
/// [`UNIT`]s, through the disk cache) and nothing else; a seek is free.
pub struct RemoteReader {
    obj: Arc<Cached>,
    pos: u64,
    /// The unit around `pos`, or empty.
    buf: Vec<u8>,
    buf_start: u64,
    /// Bytes to make resident on the next miss; doubles while the reads
    /// stay sequential, resets when they jump.
    readahead: u64,
    last_end: u64,
}

impl RemoteReader {
    pub fn open(loc: &Loc) -> Result<RemoteReader> {
        let obj = open_cached(loc)?;
        Ok(RemoteReader {
            obj,
            pos: 0,
            buf: Vec::new(),
            buf_start: 0,
            readahead: UNIT,
            last_end: u64::MAX,
        })
    }

    pub fn size(&self) -> u64 {
        self.obj.meta.size
    }

    /// Bring the unit under `pos` into `buf`.
    fn fill(&mut self) -> io::Result<()> {
        let size = self.obj.meta.size;
        if self.pos >= size {
            self.buf.clear();
            return Ok(());
        }
        if self.pos == self.last_end {
            self.readahead = (self.readahead * 2).min(MAX_SPAN);
        } else {
            self.readahead = UNIT;
        }
        let start = self.pos - self.pos % UNIT;
        self.obj
            .ensure(start, self.readahead)
            .map_err(|e| io::Error::other(format!("{e:#}")))?;
        let len = UNIT.min(size - start) as usize;
        self.buf.resize(len, 0);
        let mut got = 0;
        while got < len {
            let n = read_at(&self.obj.file, &mut self.buf[got..], start + got as u64)?;
            if n == 0 {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "cache file short"));
            }
            got += n;
        }
        self.buf_start = start;
        Ok(())
    }

    fn in_buf(&self) -> bool {
        !self.buf.is_empty()
            && self.pos >= self.buf_start
            && self.pos < self.buf_start + self.buf.len() as u64
    }
}

impl Read for RemoteReader {
    fn read(&mut self, dst: &mut [u8]) -> io::Result<usize> {
        if self.pos >= self.obj.meta.size || dst.is_empty() {
            return Ok(0);
        }
        if !self.in_buf() {
            self.fill()?;
        }
        let off = (self.pos - self.buf_start) as usize;
        let n = dst.len().min(self.buf.len() - off);
        dst[..n].copy_from_slice(&self.buf[off..off + n]);
        self.pos += n as u64;
        self.last_end = self.pos;
        Ok(n)
    }
}

impl BufRead for RemoteReader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if self.pos >= self.obj.meta.size {
            return Ok(&[]);
        }
        if !self.in_buf() {
            self.fill()?;
        }
        let off = (self.pos - self.buf_start) as usize;
        Ok(&self.buf[off..])
    }

    fn consume(&mut self, amt: usize) {
        self.pos += amt as u64;
        self.last_end = self.pos;
    }
}

impl Seek for RemoteReader {
    fn seek(&mut self, to: SeekFrom) -> io::Result<u64> {
        let size = self.obj.meta.size as i64;
        let next = match to {
            SeekFrom::Start(n) => n as i64,
            SeekFrom::End(d) => size + d,
            SeekFrom::Current(d) => self.pos as i64 + d,
        };
        if next < 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "seek before start"));
        }
        self.pos = next as u64;
        Ok(self.pos)
    }
}

/// The whole object.
pub fn read_all(loc: &Loc) -> Result<Vec<u8>> {
    let mut r = RemoteReader::open(loc)?;
    r.readahead = MAX_SPAN;
    let mut out = Vec::with_capacity(r.size() as usize);
    r.read_to_end(&mut out)?;
    Ok(out)
}

/// A one-line account of the configuration, for a panel: where the
/// bucket resolves and whether requests will be signed. Never the secret.
pub fn describe(bucket: &str) -> String {
    let r = config().resolve(bucket);
    format!(
        "{} ({}, {})",
        r.base,
        if r.creds.is_some() { "signed" } else { "anonymous" },
        if r.path_style { "path-style" } else { "virtual-hosted" }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_parse_and_print() {
        let l = Loc::parse(Path::new("s3://plates/shots/sh010/a.1001.exr")).unwrap();
        assert_eq!(l.bucket, "plates");
        assert_eq!(l.key, "shots/sh010/a.1001.exr");
        assert_eq!(l.to_path(), PathBuf::from("s3://plates/shots/sh010/a.1001.exr"));
        let root = Loc::parse(Path::new("s3://plates")).unwrap();
        assert_eq!(root.key, "");
        assert!(Loc::parse(Path::new("s3:///nope")).is_err());
        assert!(Loc::parse(Path::new("/local/path")).is_err());
        assert!(is_remote(Path::new("s3://b/k")));
        assert!(!is_remote(Path::new("/Volumes/plates")));
        // Path arithmetic the rest of the app does keeps the scheme intact.
        let p = Path::new("s3://plates/shots/a.1001.exr");
        assert_eq!(p.parent().unwrap(), Path::new("s3://plates/shots"));
        assert_eq!(p.extension().unwrap(), "exr");
        assert_eq!(Path::new("s3://plates").join("x"), PathBuf::from("s3://plates/x"));
    }

    /// AWS's own `get-vanilla` vector from the SigV4 test suite.
    #[test]
    fn sigv4_get_vanilla_with_date_header() {
        let creds = Credentials {
            access_key: "AKIDEXAMPLE".into(),
            secret_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".into(),
        };
        let auth = authorization(
            &SignInput {
                method: "GET",
                host: "example.amazonaws.com",
                path: "/",
                query: &[],
                headers: &[("x-amz-date", "20150830T123600Z")],
                payload_hash: EMPTY_SHA256,
                amz_date: "20150830T123600Z",
                region: "us-east-1",
                service: "service",
            },
            &creds,
        );
        assert_eq!(
            auth,
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, \
             SignedHeaders=host;x-amz-date, \
             Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"
        );
    }

    #[test]
    fn hmac_and_encoding_are_the_rfc_ones() {
        // RFC 4231 test case 2.
        let mac = hmac_sha256(b"Jefe", b"what do ya want for nothing?");
        assert_eq!(
            hex(&mac),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        assert_eq!(uri_encode("a b/c~d", false), "a%20b/c~d");
        assert_eq!(uri_encode("a b/c", true), "a%20b%2Fc");
        assert_eq!(amz_date(1_440_938_160), "20150830T123600Z");
        assert_eq!(amz_date(0), "19700101T000000Z");
        assert_eq!(amz_date(951_782_400), "20000229T000000Z");
    }

    #[test]
    fn resolve_builds_the_three_url_shapes() {
        let mut c = Config::default();
        // AWS: virtual-hosted, region in the host.
        c.default.region = Some("eu-west-2".into());
        let r = c.resolve("plates");
        assert_eq!(r.base, "https://plates.s3.eu-west-2.amazonaws.com");
        assert_eq!(r.host, "plates.s3.eu-west-2.amazonaws.com");
        assert!(!r.path_style);
        assert_eq!(r.url("a b.exr", &[]), "https://plates.s3.eu-west-2.amazonaws.com/a%20b.exr");
        // R2 / MinIO: path-style by default.
        c.default.url = Some("http://127.0.0.1:9000/".into());
        let r = c.resolve("plates");
        assert_eq!(r.base, "http://127.0.0.1:9000/plates");
        assert_eq!(r.host, "127.0.0.1:9000");
        assert_eq!(r.canonical_path("plates", "x/y z.exr"), "/plates/x/y%20z.exr");
        // Virtual-hosted on a custom endpoint, and a per-bucket override.
        c.buckets.insert(
            "other".into(),
            Endpoint { path_style: Some(false), url: Some("https://acct.r2.cloudflarestorage.com".into()), ..Default::default() },
        );
        let r = c.resolve("other");
        assert_eq!(r.base, "https://other.acct.r2.cloudflarestorage.com");
        assert_eq!(r.canonical_path("other", "k"), "/k");
        assert_eq!(r.region, "eu-west-2", "the default's region carries into the override");
        assert!(r.creds.is_none());
        assert!(describe("other").contains("anonymous"));
    }

    #[test]
    fn listing_xml_is_read() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult><Name>plates</Name><Prefix>shots/</Prefix><IsTruncated>false</IsTruncated>
<Contents><Key>shots/</Key><Size>0</Size></Contents>
<Contents><Key>shots/a &amp; b.1001.exr</Key><Size>10</Size></Contents>
<CommonPrefixes><Prefix>shots/sub/</Prefix></CommonPrefixes>
</ListBucketResult>"#;
        let keys: Vec<String> =
            tag_blocks(xml, "Contents").iter().filter_map(|b| tag_text(b, "Key")).collect();
        assert_eq!(keys, vec!["shots/", "shots/a & b.1001.exr"]);
        assert_eq!(tag_text(xml, "IsTruncated").as_deref(), Some("false"));
        assert_eq!(tag_text(xml, "NextContinuationToken"), None);
        assert_eq!(xml_unescape("&lt;x&gt; &#65;&#x42; &bogus;"), "<x> AB &bogus;");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn config_file_is_read() {
        let dir = std::env::temp_dir().join(format!("wobble-objstore-cfg-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("storage.json");
        fs::write(
            &p,
            r#"{ "endpoint": "https://acct.r2.cloudflarestorage.com", "region": "auto",
                 "access_key": "AK", "secret_key": "SK", "cache_bytes": 1024,
                 "buckets": { "pub": { "access_key": null, "path_style": false } } }"#,
        )
        .unwrap();
        let c = Config::from_file(&p).unwrap();
        assert_eq!(c.default.region.as_deref(), Some("auto"));
        assert_eq!(c.cache_bytes, 1024);
        let r = c.resolve("plates");
        assert!(r.path_style);
        assert!(r.creds.is_some());
        assert_eq!(r.base, "https://acct.r2.cloudflarestorage.com/plates");
        let r = c.resolve("pub");
        assert!(!r.path_style);
        assert_eq!(r.base, "https://pub.acct.r2.cloudflarestorage.com");
        assert!(Config::from_file(&dir.join("missing.json")).is_none());
        let _ = fs::remove_dir_all(&dir);
    }
}
