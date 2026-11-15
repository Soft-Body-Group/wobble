//! 3D Gaussian splat captures: reading `.ply` (the 3DGS training layout)
//! and `.spz` (Niantic's packed format, gzip versions 1–3) into one
//! GPU-ready cloud, loaded off the UI thread and cached per path.
//!
//! The cloud keeps the file's OWN coordinate frame. A 3DGS `.ply` is
//! right-down-forward (COLMAP's camera frame), an `.spz` is right-up-back,
//! which is wobble's (Y up, the camera looking down −Z); [`Native`] says
//! which, and the renderer turns the first into the second with its model
//! matrix rather than rewriting millions of splats and their SH.
//!
//! Activations are applied here, once: opacity is a sigmoid, scales are
//! exponentials (kept as logs, which half floats hold well), the base
//! colour is `0.5 + C0 * dc`. What the GPU reads per splat is
//! [`SPLAT_WORDS`] u32s; see [`Cloud::splats`].
//!
//! Nothing here refuses a big capture: when the device cannot hold one
//! (`set_gpu_budget`), SH bands go first, then every k-th splat is kept,
//! and [`Cloud::notes`] says what was given up.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// u32 words per splat: position (3 f32), rotation (4 f16: w x, y z),
/// log scale (x y, z + opacity), colour (r g, b + 0).
pub const SPLAT_WORDS: usize = 9;
/// SH band 0's constant.
pub const SH_C0: f32 = 0.282_094_8;
/// Most splats a cloud keeps: the sort's dispatch is one row of blocks.
pub const MAX_SPLATS: usize = 65_535 * 512;

/// The frame a file's numbers are in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Native {
    /// Right, down, forward: a 3DGS `.ply`.
    Rdf,
    /// Right, up, back: an `.spz`, and wobble's own.
    Rub,
}

/// One capture, ready to upload.
#[derive(Debug)]
pub struct Cloud {
    pub count: usize,
    /// `count * SPLAT_WORDS` words; see [`SPLAT_WORDS`].
    pub splats: Vec<u32>,
    /// SH bands kept (0..=3).
    pub sh_degree: u32,
    /// Words per splat in `sh`: the band-1.. coefficients as halves,
    /// coefficient-major (r g b of coefficient 1, then 2, …), padded to a
    /// word. 0 when `sh_degree` is 0.
    pub sh_words: usize,
    pub sh: Vec<u32>,
    pub native: Native,
    /// A robust centre and radius (the middle 90% of splats), native frame.
    pub center: [f32; 3],
    pub radius: f32,
    /// Trained with the mip-splatting 2D filter (an `.spz` flag): the
    /// renderer compensates opacity for its own dilation.
    pub antialiased: bool,
    /// What was given up to fit, for the panel.
    pub notes: Vec<String>,
    /// Splats in the file before any thinning.
    pub file_count: usize,
}

/// Coefficients past the DC term for an SH degree.
pub fn sh_coeffs(degree: u32) -> usize {
    match degree {
        0 => 0,
        1 => 3,
        2 => 8,
        _ => 15,
    }
}

/// Words per splat for an SH degree's halves.
pub fn sh_words(degree: u32) -> usize {
    (sh_coeffs(degree) * 3).div_ceil(2)
}

/// Bytes per splat the GPU holds beyond its SH: the splat itself, the
/// projected record (8 words), two key/value pairs for the sort.
pub const GPU_BYTES_PER_SPLAT: u64 = (SPLAT_WORDS as u64 + 8 + 4) * 4;

static BUDGET: AtomicU64 = AtomicU64::new(128 << 20);

/// The largest single storage binding the device takes. Set once by the
/// compositor; the loader fits clouds under it.
pub fn set_gpu_budget(max_binding: u64) {
    BUDGET.store(max_binding.max(1 << 20), Ordering::Relaxed);
}

fn budget() -> u64 {
    BUDGET.load(Ordering::Relaxed)
}

fn h2(a: f32, b: f32) -> u32 {
    half::f16::from_f32(a).to_bits() as u32 | ((half::f16::from_f32(b).to_bits() as u32) << 16)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// One splat as read, before packing. Rotation is (w, x, y, z), scale is
/// LOG, opacity is linear 0..1, `dc` is the band-0 coefficient.
#[derive(Clone, Copy, Debug, Default)]
pub struct Raw {
    pub pos: [f32; 3],
    pub rot: [f32; 4],
    pub log_scale: [f32; 3],
    pub opacity: f32,
    pub dc: [f32; 3],
}

/// Pack raw splats (plus SH rows, `coeffs * 3` floats each, coefficient-
/// major) into a cloud, thinning to fit.
pub fn pack(
    raws: &[Raw],
    sh: &[f32],
    file_degree: u32,
    native: Native,
    antialiased: bool,
) -> Cloud {
    let file_count = raws.len();
    let file_coeffs = sh_coeffs(file_degree);
    let mut notes = Vec::new();
    let budget = budget();
    // SH bands go first: a capture without view-dependent colour still
    // reads; one that does not fit does not draw at all.
    let mut degree = file_degree.min(3);
    while degree > 0 && (file_count as u64) * sh_words(degree) as u64 * 4 > budget {
        degree -= 1;
    }
    if degree < file_degree {
        notes.push(format!("SH kept to band {degree} of {file_degree} to fit the GPU"));
    }
    // Then splats: every k-th, so what is left still covers the scene.
    let mut stride = 1usize;
    let fits = |n: usize| {
        let per = (SPLAT_WORDS as u64 * 4).max(32);
        (n as u64) * per <= budget && n <= MAX_SPLATS
    };
    while !fits(file_count.div_ceil(stride)) {
        stride += 1;
    }
    if stride > 1 {
        notes.push(format!(
            "kept every {stride}th splat ({} of {file_count}) to fit the GPU",
            file_count.div_ceil(stride)
        ));
    }
    let words = sh_words(degree);
    let coeffs = sh_coeffs(degree);
    let n = file_count.div_ceil(stride);
    let mut splats = Vec::with_capacity(n * SPLAT_WORDS);
    let mut shv = Vec::with_capacity(n * words);
    let mut halves: Vec<f32> = Vec::with_capacity(words * 2);
    for (i, r) in raws.iter().enumerate().step_by(stride) {
        let q = r.rot;
        let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        let q = if l > 1e-12 && l.is_finite() { q.map(|v| v / l) } else { [1.0, 0.0, 0.0, 0.0] };
        let c = r.dc.map(|d| 0.5 + SH_C0 * d);
        let ls = r.log_scale.map(|s| if s.is_finite() { s.clamp(-20.0, 10.0) } else { -20.0 });
        splats.extend_from_slice(&[
            r.pos[0].to_bits(),
            r.pos[1].to_bits(),
            r.pos[2].to_bits(),
            h2(q[0], q[1]),
            h2(q[2], q[3]),
            h2(ls[0], ls[1]),
            h2(ls[2], r.opacity.clamp(0.0, 1.0)),
            h2(c[0], c[1]),
            h2(c[2], 0.0),
        ]);
        if words > 0 {
            halves.clear();
            let row = &sh[i * file_coeffs * 3..(i + 1) * file_coeffs * 3];
            halves.extend_from_slice(&row[..coeffs * 3]);
            if halves.len() % 2 == 1 {
                halves.push(0.0);
            }
            shv.extend(halves.chunks(2).map(|p| h2(p[0], p[1])));
        }
    }
    let (center, radius) = robust_bounds(raws, stride);
    Cloud {
        count: n,
        splats,
        sh_degree: degree,
        sh_words: words,
        sh: shv,
        native,
        center,
        radius,
        antialiased,
        notes,
        file_count,
    }
}

/// The middle 90% of splat positions, per axis: a centre and a radius
/// that ignore the stray floaters every capture has.
fn robust_bounds(raws: &[Raw], stride: usize) -> ([f32; 3], f32) {
    if raws.is_empty() {
        return ([0.0; 3], 1.0);
    }
    let step = (raws.len() / 200_000).max(stride).max(1);
    let mut c = [0.0f32; 3];
    let mut half = 0.0f32;
    for (a, ca) in c.iter_mut().enumerate() {
        let mut v: Vec<f32> =
            raws.iter().step_by(step).map(|r| r.pos[a]).filter(|x| x.is_finite()).collect();
        if v.is_empty() {
            continue;
        }
        v.sort_by(|x, y| x.total_cmp(y));
        let lo = v[v.len() * 5 / 100];
        let hi = v[(v.len() * 95 / 100).min(v.len() - 1)];
        *ca = 0.5 * (lo + hi);
        half = half.max(0.5 * (hi - lo));
    }
    (c, half.max(1e-3))
}

// --- .ply ------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Ty {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Ty {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "char" | "int8" => Ty::I8,
            "uchar" | "uint8" => Ty::U8,
            "short" | "int16" => Ty::I16,
            "ushort" | "uint16" => Ty::U16,
            "int" | "int32" => Ty::I32,
            "uint" | "uint32" => Ty::U32,
            "float" | "float32" => Ty::F32,
            "double" | "float64" => Ty::F64,
            _ => return None,
        })
    }
    fn size(self) -> usize {
        match self {
            Ty::I8 | Ty::U8 => 1,
            Ty::I16 | Ty::U16 => 2,
            Ty::I32 | Ty::U32 | Ty::F32 => 4,
            Ty::F64 => 8,
        }
    }
    fn read(self, b: &[u8]) -> f32 {
        match self {
            Ty::I8 => b[0] as i8 as f32,
            Ty::U8 => b[0] as f32,
            Ty::I16 => i16::from_le_bytes([b[0], b[1]]) as f32,
            Ty::U16 => u16::from_le_bytes([b[0], b[1]]) as f32,
            Ty::I32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
            Ty::U32 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
            Ty::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            Ty::F64 => f64::from_le_bytes(b[..8].try_into().unwrap()) as f32,
        }
    }
}

/// A binary little-endian 3DGS `.ply`.
pub fn parse_ply(bytes: &[u8]) -> Result<Cloud, String> {
    let end = bytes
        .windows(11)
        .take(64 * 1024)
        .position(|w| w == b"end_header\n")
        .ok_or("not a PLY file (no end_header)")?;
    let header = std::str::from_utf8(&bytes[..end]).map_err(|_| "PLY header is not text")?;
    let body = &bytes[end + 11..];
    let mut lines = header.lines();
    if lines.next().map(str::trim) != Some("ply") {
        return Err("not a PLY file".into());
    }
    let mut format_ok = false;
    // (element name, count, properties) in order.
    let mut elements: Vec<(String, usize, Vec<(String, Ty)>)> = Vec::new();
    for l in lines {
        let w: Vec<&str> = l.split_whitespace().collect();
        match w.as_slice() {
            ["format", f, ..] => {
                if *f != "binary_little_endian" {
                    return Err(format!("PLY format {f} is not supported (binary_little_endian only)"));
                }
                format_ok = true;
            }
            ["element", name, n] => {
                elements.push((name.to_string(), n.parse().map_err(|_| "bad PLY element count")?, Vec::new()))
            }
            ["property", "list", ..] => {
                let e = elements.last_mut().ok_or("PLY property before any element")?;
                if e.0 == "vertex" {
                    return Err("PLY vertex lists are not supported".into());
                }
                // A list in another element: we stop before it anyway.
                e.2.push(("<list>".into(), Ty::U8));
            }
            ["property", ty, name] => {
                let t = Ty::parse(ty).ok_or_else(|| format!("unknown PLY type {ty}"))?;
                elements.last_mut().ok_or("PLY property before any element")?.2.push((name.to_string(), t));
            }
            _ => {}
        }
    }
    if !format_ok {
        return Err("PLY has no format line".into());
    }
    // Elements before the vertices would have to be skipped; captures put
    // the vertices first.
    let (vi, (_, n, props)) = elements
        .iter()
        .enumerate()
        .find(|(_, e)| e.0 == "vertex")
        .ok_or("PLY has no vertex element")?;
    if vi != 0 {
        return Err("PLY vertices must be the first element".into());
    }
    let n = *n;
    let mut offs: HashMap<&str, (usize, Ty)> = HashMap::new();
    let mut stride = 0;
    for (name, t) in props {
        offs.insert(name.as_str(), (stride, *t));
        stride += t.size();
    }
    if body.len() < n * stride {
        return Err(format!("PLY is cut short: {n} splats need {} bytes, it has {}", n * stride, body.len()));
    }
    let need = |k: &str| offs.get(k).copied().ok_or_else(|| format!("not a Gaussian splat PLY (no {k})"));
    let px = [need("x")?, need("y")?, need("z")?];
    let dc = [need("f_dc_0")?, need("f_dc_1")?, need("f_dc_2")?];
    let op = need("opacity")?;
    let sc = [need("scale_0")?, need("scale_1")?, need("scale_2")?];
    let rt = [need("rot_0")?, need("rot_1")?, need("rot_2")?, need("rot_3")?];
    let mut rest = Vec::new();
    while let Some(o) = offs.get(format!("f_rest_{}", rest.len()).as_str()) {
        rest.push(*o);
    }
    let degree = match rest.len() / 3 {
        c if c >= 15 => 3,
        c if c >= 8 => 2,
        c if c >= 3 => 1,
        _ => 0,
    };
    let coeffs = sh_coeffs(degree);
    // f_rest is channel-major: all of red's coefficients, then green's,
    // then blue's.
    let per_channel = rest.len() / 3;
    let mut raws = Vec::with_capacity(n);
    let mut sh = Vec::with_capacity(n * coeffs * 3);
    let rd = |row: &[u8], (o, t): (usize, Ty)| t.read(&row[o..]);
    for row in body[..n * stride].chunks_exact(stride) {
        raws.push(Raw {
            pos: px.map(|p| rd(row, p)),
            rot: rt.map(|p| rd(row, p)),
            log_scale: sc.map(|p| rd(row, p)),
            opacity: sigmoid(rd(row, op)),
            dc: dc.map(|p| rd(row, p)),
        });
        for k in 0..coeffs {
            for ch in 0..3 {
                sh.push(rd(row, rest[ch * per_channel + k]));
            }
        }
    }
    Ok(pack(&raws, &sh, degree, Native::Rdf, false))
}

// --- .spz ------------------------------------------------------------------

const NGSP: u32 = 0x5053_474e;

/// An `.spz`: gzip around a 16-byte header and the attribute streams
/// (positions, alphas, colours, scales, rotations, SH).
pub fn parse_spz(bytes: &[u8]) -> Result<Cloud, String> {
    if bytes.len() >= 8 && u32::from_le_bytes(bytes[0..4].try_into().unwrap()) == NGSP {
        let v = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        return Err(format!("SPZ version {v} (zstd streams) is not supported yet; re-save as version 3"));
    }
    let mut raw = Vec::new();
    {
        use std::io::Read;
        flate2::read::GzDecoder::new(bytes)
            .read_to_end(&mut raw)
            .map_err(|e| format!("SPZ is not gzip: {e}"))?;
    }
    decode_spz(&raw)
}

/// The decompressed body of an `.spz`.
pub fn decode_spz(raw: &[u8]) -> Result<Cloud, String> {
    if raw.len() < 16 {
        return Err("SPZ is too short".into());
    }
    let u = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
    if u(0) != NGSP {
        return Err("not an SPZ file".into());
    }
    let version = u(4);
    if !(1..=3).contains(&version) {
        return Err(format!("SPZ version {version} is not supported"));
    }
    let n = u(8) as usize;
    let degree = raw[12] as u32;
    let frac = raw[13] as i32;
    let flags = raw[14];
    if degree > 3 {
        return Err(format!("SPZ SH degree {degree} is not supported"));
    }
    let coeffs = sh_coeffs(degree);
    let pos_b = if version == 1 { 6 } else { 9 };
    let rot_b = if version >= 3 { 4 } else { 3 };
    let sizes = [pos_b, 1, 3, 3, rot_b, coeffs * 3];
    let total: usize = sizes.iter().map(|s| s * n).sum();
    if raw.len() < 16 + total {
        return Err(format!("SPZ is cut short: {n} splats need {total} bytes"));
    }
    let mut at = 16;
    let mut stream = |s: usize| {
        let r = &raw[at..at + s * n];
        at += s * n;
        r
    };
    let (pos, alpha, col, scl, rot, shb) =
        (stream(sizes[0]), stream(sizes[1]), stream(sizes[2]), stream(sizes[3]), stream(sizes[4]), stream(sizes[5]));
    let fscale = 1.0 / (1i64 << frac.clamp(0, 30)) as f32;
    const COLOR_SCALE: f32 = 0.15;
    let mut raws = Vec::with_capacity(n);
    for i in 0..n {
        let p = if version == 1 {
            let h = |k: usize| half::f16::from_le_bytes([pos[i * 6 + k * 2], pos[i * 6 + k * 2 + 1]]).to_f32();
            [h(0), h(1), h(2)]
        } else {
            [0, 1, 2].map(|k| {
                let b = &pos[i * 9 + k * 3..];
                let mut v = b[0] as i32 | (b[1] as i32) << 8 | (b[2] as i32) << 16;
                if v & 0x80_0000 != 0 {
                    v |= !0xff_ffff;
                }
                v as f32 * fscale
            })
        };
        // Stored x y z w; ours is w x y z.
        let q = if version >= 3 {
            let r = &rot[i * 4..i * 4 + 4];
            let mut comp = u32::from_le_bytes([r[0], r[1], r[2], r[3]]);
            let largest = (comp >> 30) as usize;
            let mut q = [0.0f32; 4];
            let mut ss = 0.0;
            const MASK: u32 = (1 << 9) - 1;
            for k in (0..4).rev() {
                if k != largest {
                    let mag = comp & MASK;
                    let neg = (comp >> 9) & 1;
                    comp >>= 10;
                    let v = std::f32::consts::FRAC_1_SQRT_2 * mag as f32 / MASK as f32;
                    q[k] = if neg == 1 { -v } else { v };
                    ss += q[k] * q[k];
                }
            }
            q[largest] = (1.0 - ss).max(0.0).sqrt();
            q
        } else {
            let r = &rot[i * 3..i * 3 + 3];
            let xyz = [0, 1, 2].map(|k| r[k] as f32 / 127.5 - 1.0);
            let w = (1.0 - xyz.iter().map(|v| v * v).sum::<f32>()).max(0.0).sqrt();
            [xyz[0], xyz[1], xyz[2], w]
        };
        raws.push(Raw {
            pos: p,
            rot: [q[3], q[0], q[1], q[2]],
            log_scale: [0, 1, 2].map(|k| scl[i * 3 + k] as f32 / 16.0 - 10.0),
            opacity: alpha[i] as f32 / 255.0,
            dc: [0, 1, 2].map(|k| (col[i * 3 + k] as f32 / 255.0 - 0.5) / COLOR_SCALE),
        });
    }
    // SH is coefficient-major already (r g b per coefficient).
    let sh: Vec<f32> = shb.iter().map(|&b| (b as f32 - 128.0) / 128.0).collect();
    Ok(pack(&raws, &sh, degree, Native::Rub, flags & 1 != 0))
}

/// Read a capture by its extension.
pub fn load_file(path: &str) -> Result<Cloud, String> {
    let p = crate::paths::expand_home(path);
    let bytes = std::fs::read(&p).map_err(|e| format!("cannot read {path}: {e}"))?;
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".spz") {
        parse_spz(&bytes)
    } else if lower.ends_with(".ply") {
        parse_ply(&bytes)
    } else if bytes.starts_with(b"ply") {
        parse_ply(&bytes)
    } else {
        parse_spz(&bytes)
    }
}

// --- the cache ------------------------------------------------------------

/// Where a path's load stands.
#[derive(Clone, Debug)]
pub enum State {
    Loading,
    Ready(Arc<Cloud>),
    Failed(String),
}

struct Entry {
    stamp: Option<std::time::SystemTime>,
    state: State,
    /// Bumped when the state changes: the renderer re-uploads, and the
    /// graph differs so the viewer redraws.
    gen: u64,
}

static GEN: AtomicU64 = AtomicU64::new(1);

fn cache() -> &'static Mutex<HashMap<String, Entry>> {
    static C: OnceLock<Mutex<HashMap<String, Entry>>> = OnceLock::new();
    C.get_or_init(Default::default)
}

fn stamp_of(path: &str) -> Option<std::time::SystemTime> {
    std::fs::metadata(crate::paths::expand_home(path)).and_then(|m| m.modified()).ok()
}

/// Ask for `path`; starts a load in the background the first time (and
/// again when the file changes). Returns the state and its generation.
pub fn request(path: &str) -> (State, u64) {
    if path.is_empty() {
        return (State::Failed("no file picked".into()), 0);
    }
    let stamp = stamp_of(path);
    let mut c = cache().lock().unwrap();
    if let Some(e) = c.get(path) {
        let stale = !matches!(e.state, State::Loading) && e.stamp != stamp;
        if !stale {
            return (e.state.clone(), e.gen);
        }
    }
    let gen = GEN.fetch_add(1, Ordering::Relaxed);
    c.insert(path.to_string(), Entry { stamp, state: State::Loading, gen });
    drop(c);
    let owned = path.to_string();
    let run = move || {
        let state = match load_file(&owned) {
            Ok(cl) => State::Ready(Arc::new(cl)),
            Err(e) => State::Failed(e),
        };
        let gen = GEN.fetch_add(1, Ordering::Relaxed);
        cache().lock().unwrap().insert(owned, Entry { stamp, state, gen });
    };
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::Builder::new().name("splat-load".into()).spawn(run).ok();
    #[cfg(target_arch = "wasm32")]
    run();
    (State::Loading, gen)
}

/// The state without starting anything.
pub fn peek(path: &str) -> Option<(State, u64)> {
    cache().lock().unwrap().get(path).map(|e| (e.state.clone(), e.gen))
}

/// Is any capture still loading? (The app keeps redrawing until not.)
pub fn any_loading() -> bool {
    cache().lock().unwrap().values().any(|e| matches!(e.state, State::Loading))
}

/// Hand a cloud to the cache under a name, for tests and generated clouds.
pub fn insert(path: &str, cloud: Cloud) -> u64 {
    let gen = GEN.fetch_add(1, Ordering::Relaxed);
    cache()
        .lock()
        .unwrap()
        .insert(path.to_string(), Entry { stamp: stamp_of(path), state: State::Ready(Arc::new(cloud)), gen });
    gen
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ply(n: usize, rest: usize) -> Vec<u8> {
        let mut props = vec!["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        props.extend((0..rest).map(|i| format!("f_rest_{i}")));
        props.extend(
            ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"].map(String::from),
        );
        let mut h = format!("ply\nformat binary_little_endian 1.0\nelement vertex {n}\n");
        for p in &props {
            h += &format!("property float {p}\n");
        }
        h += "end_header\n";
        let mut b = h.into_bytes();
        for i in 0..n {
            for p in &props {
                let v: f32 = match p.as_str() {
                    "x" => i as f32,
                    "y" => 2.0,
                    "z" => -3.0,
                    "f_dc_0" => 1.0,
                    "opacity" => 0.0,
                    "scale_0" | "scale_1" | "scale_2" => -2.0,
                    "rot_0" => 2.0,
                    s if s.starts_with("f_rest_") => s[7..].parse::<f32>().unwrap() / 100.0,
                    _ => 0.0,
                };
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        b
    }

    fn word_f16(w: u32, hi: bool) -> f32 {
        half::f16::from_bits(if hi { (w >> 16) as u16 } else { w as u16 }).to_f32()
    }

    #[test]
    fn a_ply_reads_with_its_activations() {
        let c = parse_ply(&ply(3, 45)).unwrap();
        assert_eq!(c.count, 3);
        assert_eq!(c.sh_degree, 3);
        assert_eq!(c.native, Native::Rdf);
        let s = &c.splats[SPLAT_WORDS..2 * SPLAT_WORDS];
        assert_eq!(f32::from_bits(s[0]), 1.0);
        assert_eq!(f32::from_bits(s[2]), -3.0);
        // rot normalised: (2,0,0,0) -> (1,0,0,0)
        assert!((word_f16(s[3], false) - 1.0).abs() < 1e-3);
        assert!((word_f16(s[5], false) + 2.0).abs() < 1e-3);
        assert!((word_f16(s[6], true) - 0.5).abs() < 1e-3, "sigmoid(0)");
        assert!((word_f16(s[7], false) - (0.5 + SH_C0)).abs() < 1e-3);
        // Channel-major in the file, coefficient-major in the cloud:
        // coefficient 1's green is f_rest_15.
        assert_eq!(c.sh_words, 23);
        let g1 = word_f16(c.sh[0], true);
        assert!((g1 - 0.15).abs() < 1e-3, "{g1}");
    }

    #[test]
    fn an_spz_v3_reads() {
        let n = 2usize;
        let mut b = Vec::new();
        b.extend_from_slice(&NGSP.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&(n as u32).to_le_bytes());
        b.extend_from_slice(&[0, 12, 1, 0]);
        for i in 0..n {
            for k in 0..3 {
                let v: i32 = if k == 0 { (i as i32 + 1) << 12 } else { -(1 << 11) };
                b.extend_from_slice(&v.to_le_bytes()[..3]);
            }
        }
        b.extend_from_slice(&[255, 0]);
        b.extend_from_slice(&[128, 128, 128, 128, 128, 128]);
        b.extend_from_slice(&[144, 144, 144, 0, 0, 0]);
        // Smallest three, largest = w (index 3), the rest 0.
        for _ in 0..n {
            b.extend_from_slice(&(3u32 << 30).to_le_bytes());
        }
        let c = decode_spz(&b).unwrap();
        assert_eq!(c.count, 2);
        assert_eq!(c.native, Native::Rub);
        assert!(c.antialiased);
        let s = &c.splats[..SPLAT_WORDS];
        assert_eq!(f32::from_bits(s[0]), 1.0);
        assert_eq!(f32::from_bits(s[1]), -0.5);
        assert!((word_f16(s[3], false) - 1.0).abs() < 1e-3, "w");
        assert!((word_f16(s[5], false) + 1.0).abs() < 1e-2, "144/16-10");
        assert!((word_f16(s[6], true) - 1.0).abs() < 1e-3);
        let s1 = &c.splats[SPLAT_WORDS..];
        assert!((word_f16(s1[5], false) + 10.0).abs() < 1e-2);
    }

    #[test]
    fn a_tight_budget_drops_sh_then_splats() {
        set_gpu_budget(1 << 20);
        let raws = vec![Raw { rot: [1.0, 0.0, 0.0, 0.0], ..Default::default() }; 40_000];
        let sh = vec![0.0; 40_000 * 45];
        let c = pack(&raws, &sh, 3, Native::Rub, false);
        set_gpu_budget(128 << 20);
        assert!(c.sh_degree < 3);
        assert!(c.count < 40_000 && c.count * SPLAT_WORDS * 4 <= 1 << 20);
        assert_eq!(c.notes.len(), 2, "{:?}", c.notes);
    }
}
