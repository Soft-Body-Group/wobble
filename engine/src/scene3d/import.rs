//! Cameras and transforms from files, for the 3D cards nodes: a Camera or
//! Card whose source is a file reads its pose here, per frame.
//!
//! Three formats, all text, all read in Rust with no library:
//!
//! - **JSON**, our own small schema ([`parse_json`] has it), the easy
//!   target for an exporter script;
//! - **.chan**: one line a frame, `frame tx ty tz rx ry rz [vfov]`,
//!   rotations in degrees applied X then Y then Z, the optional last
//!   column a vertical field of view in degrees;
//! - **.usda**: USD's text form. Camera and Xform prims, their
//!   `xformOp`s (translate, rotate*, scale, orient, transform) in
//!   `xformOpOrder`, time samples, parents composed, `upAxis` honoured;
//!   a camera's focal length, apertures and clipping range. Not a USD
//!   implementation: no references, layers, variants or binary crates.
//!
//! Everything comes out in the cards' convention (`crate::scene3d`): Y up,
//! the camera down −Z. A Z-up file is turned −90° about X on the way in.

use super::{identity, mat_mul, rotation, Mat4, Xform};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// A camera's optics at a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lens {
    pub focal: f32,
    pub aperture: [f32; 2],
    pub near: f32,
    pub far: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackKind {
    Camera,
    Xform,
}

/// One object's animation: a pose (and for a camera, optics) per key.
#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    pub name: String,
    pub kind: TrackKind,
    /// (frame, world pose, optics), sorted by frame; Euler angles unwrapped
    /// key to key so sampling between them never spins the long way.
    pub keys: Vec<(f32, Xform, Option<Lens>)>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SceneFile {
    pub fps: Option<f32>,
    pub tracks: Vec<Track>,
}

impl SceneFile {
    /// The named track (of `kind` first: a USD camera is often a Camera
    /// prim named like the Xform above it), or the first of its kind when
    /// `name` is empty.
    pub fn track(&self, name: &str, kind: TrackKind) -> Option<&Track> {
        if name.is_empty() {
            self.tracks.iter().find(|t| t.kind == kind)
        } else {
            self.tracks
                .iter()
                .find(|t| t.name == name && t.kind == kind)
                .or_else(|| self.tracks.iter().find(|t| t.name == name))
        }
    }
}

impl Track {
    /// The pose at `frame`: linear between keys, held past either end.
    pub fn sample(&self, frame: f32) -> Option<(Xform, Option<Lens>)> {
        let first = self.keys.first()?;
        let last = self.keys.last()?;
        if frame <= first.0 {
            return Some((first.1, first.2));
        }
        if frame >= last.0 {
            return Some((last.1, last.2));
        }
        let i = self.keys.partition_point(|k| k.0 <= frame);
        let (a, b) = (&self.keys[i - 1], &self.keys[i]);
        let s = (frame - a.0) / (b.0 - a.0).max(1e-9);
        let lerp3 = |x: [f32; 3], y: [f32; 3]| [0, 1, 2].map(|k| x[k] + (y[k] - x[k]) * s);
        let xf = Xform {
            translate: lerp3(a.1.translate, b.1.translate),
            rotate: lerp3(a.1.rotate, b.1.rotate),
            scale: lerp3(a.1.scale, b.1.scale),
        };
        let lens = match (a.2, b.2) {
            (Some(p), Some(q)) => Some(Lens {
                focal: p.focal + (q.focal - p.focal) * s,
                aperture: [0, 1].map(|k| p.aperture[k] + (q.aperture[k] - p.aperture[k]) * s),
                near: p.near + (q.near - p.near) * s,
                far: p.far + (q.far - p.far) * s,
            }),
            (l, _) => l,
        };
        Some((xf, lens))
    }

    /// Sorted, deduplicated, angles unwrapped.
    fn finish(mut self) -> Self {
        self.keys.sort_by(|a, b| a.0.total_cmp(&b.0));
        self.keys.dedup_by(|a, b| a.0 == b.0);
        for i in 1..self.keys.len() {
            let prev = self.keys[i - 1].1.rotate;
            let r = &mut self.keys[i].1.rotate;
            for k in 0..3 {
                while r[k] - prev[k] > 180.0 {
                    r[k] -= 360.0;
                }
                while r[k] - prev[k] < -180.0 {
                    r[k] += 360.0;
                }
            }
        }
        self
    }
}

/// Z up → Y up: −90° about X.
fn z_up_to_y_up() -> Mat4 {
    let r = rotation([-90.0, 0.0, 0.0]);
    let mut m = identity();
    for i in 0..3 {
        m[i][..3].copy_from_slice(&r[i]);
    }
    m
}

fn pose(m: &Mat4) -> Xform {
    Xform::from_matrix(m)
}

/// A file, by extension.
pub fn parse(path: &Path, text: &str) -> Result<SceneFile, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "json" => parse_json(text),
        "chan" => parse_chan(text, path.file_stem().and_then(|s| s.to_str()).unwrap_or("camera")),
        "usda" => parse_usda(text),
        "usd" | "usdc" | "usdz" => Err("binary USD is not read: save it as .usda".into()),
        "abc" => Err("Alembic is not read yet: send it through Blender, or export .usda or .json".into()),
        _ => Err(format!("unknown scene file type .{ext} (json, chan, usda)")),
    }
}

/// Load through a cache keyed on path and modification time, so a node
/// can ask every frame and a re-export is picked up.
pub fn load(path: &str) -> Result<Arc<SceneFile>, String> {
    type Cache = Mutex<HashMap<PathBuf, (std::time::SystemTime, Arc<SceneFile>)>>;
    static CACHE: std::sync::OnceLock<Cache> = std::sync::OnceLock::new();
    let p = crate::paths::expand_home(path);
    let mtime = std::fs::metadata(&p)
        .and_then(|m| m.modified())
        .map_err(|e| format!("{}: {e}", p.display()))?;
    let cache = CACHE.get_or_init(Default::default);
    if let Some((t, f)) = cache.lock().unwrap().get(&p) {
        if *t == mtime {
            return Ok(f.clone());
        }
    }
    let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    let f = Arc::new(parse(&p, &text)?);
    cache.lock().unwrap().insert(p, (mtime, f.clone()));
    Ok(f)
}

// ---------------------------------------------------------------- JSON

/// The JSON schema:
///
/// ```json
/// { "fps": 24, "up": "y",
///   "objects": [
///     { "name": "shotcam", "type": "camera",
///       "frames": [ { "frame": 1, "translate": [0, 1.6, 5], "rotate": [-5, 0, 0],
///                     "focal": 35, "aperture": [36, 24], "near": 0.1, "far": 1000 } ] },
///     { "name": "wall", "type": "xform",
///       "frames": [ { "frame": 1, "matrix": [16 numbers, row-major, translation in the last COLUMN] } ] } ] }
/// ```
///
/// A frame gives `matrix` or any of `translate` / `rotate` (degrees, X
/// then Y then Z) / `scale`. Optics ride on camera frames and hold until
/// changed. `"up": "z"` takes a Z-up scene.
#[cfg(not(feature = "serde"))]
pub fn parse_json(_text: &str) -> Result<SceneFile, String> {
    Err("json scene files need the engine's serde feature".into())
}

#[cfg(feature = "serde")]
pub fn parse_json(text: &str) -> Result<SceneFile, String> {
    let v: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("json: {e}"))?;
    let up_z = v.get("up").and_then(|u| u.as_str()).is_some_and(|u| u.eq_ignore_ascii_case("z"));
    let fix = if up_z { z_up_to_y_up() } else { identity() };
    let arr3 = |x: Option<&serde_json::Value>| -> Option<[f32; 3]> {
        let a = x?.as_array()?;
        (a.len() == 3).then(|| [0, 1, 2].map(|i| a[i].as_f64().unwrap_or(0.0) as f32))
    };
    let mut out = SceneFile { fps: v.get("fps").and_then(|f| f.as_f64()).map(|f| f as f32), tracks: Vec::new() };
    for (i, o) in v.get("objects").and_then(|o| o.as_array()).into_iter().flatten().enumerate() {
        let name = o.get("name").and_then(|n| n.as_str()).map(str::to_string).unwrap_or(format!("object{i}"));
        let kind = match o.get("type").and_then(|t| t.as_str()).unwrap_or("xform") {
            "camera" => TrackKind::Camera,
            _ => TrackKind::Xform,
        };
        let mut lens = Lens { focal: 50.0, aperture: [36.0, 24.0], near: 0.1, far: 10000.0 };
        let mut track = Track { name, kind, keys: Vec::new() };
        for f in o.get("frames").and_then(|f| f.as_array()).into_iter().flatten() {
            let frame = f.get("frame").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
            let m = if let Some(a) = f.get("matrix").and_then(|m| m.as_array()).filter(|a| a.len() == 16) {
                let mut m = identity();
                for (k, x) in a.iter().enumerate() {
                    m[k / 4][k % 4] = x.as_f64().unwrap_or(0.0) as f32;
                }
                m
            } else {
                Xform {
                    translate: arr3(f.get("translate")).unwrap_or([0.0; 3]),
                    rotate: arr3(f.get("rotate")).unwrap_or([0.0; 3]),
                    scale: arr3(f.get("scale")).unwrap_or([1.0; 3]),
                }
                .matrix()
            };
            let num = |k: &str| f.get(k).and_then(|x| x.as_f64()).map(|x| x as f32);
            if let Some(x) = num("focal") {
                lens.focal = x;
            }
            if let Some(a) = f.get("aperture").and_then(|a| a.as_array()).filter(|a| a.len() == 2) {
                lens.aperture = [0, 1].map(|i| a[i].as_f64().unwrap_or(36.0) as f32);
            }
            if let Some(x) = num("near") {
                lens.near = x;
            }
            if let Some(x) = num("far") {
                lens.far = x;
            }
            let l = (kind == TrackKind::Camera).then_some(lens);
            track.keys.push((frame, pose(&mat_mul(&fix, &m)), l));
        }
        out.tracks.push(track.finish());
    }
    if out.tracks.is_empty() {
        return Err("json: no objects".into());
    }
    Ok(out)
}

// ---------------------------------------------------------------- .chan

/// `frame tx ty tz rx ry rz [vfov]`, whitespace separated, `#` comments.
/// A vertical field of view becomes a focal length on a 24mm-high back.
pub fn parse_chan(text: &str, name: &str) -> Result<SceneFile, String> {
    let mut track = Track { name: name.to_string(), kind: TrackKind::Camera, keys: Vec::new() };
    for (n, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let v: Vec<f32> = line
            .split_whitespace()
            .map(|x| x.parse::<f32>().map_err(|_| format!("chan line {}: not a number: {x}", n + 1)))
            .collect::<Result<_, _>>()?;
        if v.len() < 7 {
            return Err(format!("chan line {}: want frame tx ty tz rx ry rz [vfov]", n + 1));
        }
        let ap = [36.0f32, 24.0];
        let lens = v.get(7).map(|vfov| Lens {
            focal: ap[1] * 0.5 / (vfov.to_radians() * 0.5).tan().max(1e-6),
            aperture: ap,
            near: 0.1,
            far: 10000.0,
        });
        let xf = Xform { translate: [v[1], v[2], v[3]], rotate: [v[4], v[5], v[6]], scale: [1.0; 3] };
        track.keys.push((v[0], xf, lens));
    }
    if track.keys.is_empty() {
        return Err("chan: no frames".into());
    }
    Ok(SceneFile { fps: None, tracks: vec![track.finish()] })
}

// ---------------------------------------------------------------- .usda

#[derive(Clone, Debug, PartialEq)]
enum Val {
    Num(f64),
    Str(String),
    Ident(String),
    List(Vec<Val>),
    /// A time-sample dictionary.
    Samples(Vec<(f64, Val)>),
}

impl Val {
    fn nums(&self) -> Vec<f64> {
        match self {
            Val::Num(n) => vec![*n],
            Val::List(l) => l.iter().flat_map(|v| v.nums()).collect(),
            _ => Vec::new(),
        }
    }
}

struct Lexer<'a> {
    s: &'a [u8],
    i: usize,
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Word(String),
    Str(String),
    Num(f64),
    Punct(u8),
}

impl Lexer<'_> {
    fn skip_ws(&mut self) {
        while self.i < self.s.len() {
            let c = self.s[self.i];
            if c == b'#' {
                while self.i < self.s.len() && self.s[self.i] != b'\n' {
                    self.i += 1;
                }
            } else if c.is_ascii_whitespace() {
                self.i += 1;
            } else {
                break;
            }
        }
    }

    fn next(&mut self) -> Option<Tok> {
        self.skip_ws();
        let c = *self.s.get(self.i)?;
        if c == b'"' || c == b'\'' {
            // Triple-quoted too: a doc string.
            let triple = self.s[self.i..].starts_with(if c == b'"' { b"\"\"\"" } else { b"'''" });
            let q = if triple { 3 } else { 1 };
            self.i += q;
            let start = self.i;
            while self.i < self.s.len() {
                if triple {
                    if self.s[self.i..].starts_with(&[c, c, c]) {
                        break;
                    }
                } else if self.s[self.i] == c {
                    break;
                } else if self.s[self.i] == b'\\' {
                    self.i += 1;
                }
                self.i += 1;
            }
            let t = String::from_utf8_lossy(&self.s[start..self.i.min(self.s.len())]).into_owned();
            self.i = (self.i + q).min(self.s.len());
            return Some(Tok::Str(t));
        }
        if c == b'@' {
            // An asset path: @...@, read as a string.
            self.i += 1;
            let start = self.i;
            while self.i < self.s.len() && self.s[self.i] != b'@' {
                self.i += 1;
            }
            let t = String::from_utf8_lossy(&self.s[start..self.i]).into_owned();
            self.i += 1;
            return Some(Tok::Str(t));
        }
        if c.is_ascii_digit() || ((c == b'-' || c == b'+' || c == b'.') && self.s.get(self.i + 1).is_some_and(|d| d.is_ascii_digit() || *d == b'.')) {
            let start = self.i;
            self.i += 1;
            while self.i < self.s.len() {
                let d = self.s[self.i];
                let exp_sign = (d == b'-' || d == b'+') && matches!(self.s[self.i - 1], b'e' | b'E');
                if d.is_ascii_digit() || d == b'.' || d == b'e' || d == b'E' || exp_sign {
                    self.i += 1;
                } else {
                    break;
                }
            }
            let t = std::str::from_utf8(&self.s[start..self.i]).unwrap_or("0");
            return Some(Tok::Num(t.parse().unwrap_or(0.0)));
        }
        if c.is_ascii_alphabetic() || c == b'_' || c == b'<' {
            let start = self.i;
            if c == b'<' {
                // A path reference <...>.
                while self.i < self.s.len() && self.s[self.i] != b'>' {
                    self.i += 1;
                }
                self.i += 1;
                return Some(Tok::Str(String::from_utf8_lossy(&self.s[start..self.i]).into_owned()));
            }
            while self.i < self.s.len() {
                let d = self.s[self.i];
                if d.is_ascii_alphanumeric() || d == b'_' || d == b':' || d == b'.' || d == b'[' && self.s.get(self.i + 1) == Some(&b']') || d == b']' && self.s[self.i - 1] == b'[' {
                    self.i += 1;
                } else {
                    break;
                }
            }
            return Some(Tok::Word(String::from_utf8_lossy(&self.s[start..self.i]).into_owned()));
        }
        if c == b'-' && self.s.get(self.i + 1).is_some_and(|d| d.is_ascii_alphabetic()) {
            // -inf
            self.i += 1;
            return match self.next() {
                Some(Tok::Word(w)) if w == "inf" => Some(Tok::Num(f64::NEG_INFINITY)),
                other => other,
            };
        }
        self.i += 1;
        Some(Tok::Punct(c))
    }
}

struct Parser<'a> {
    lx: Lexer<'a>,
    peeked: Option<Tok>,
}

#[derive(Debug, Default)]
struct Prim {
    kind: String,
    name: String,
    attrs: HashMap<String, Val>,
    kids: Vec<Prim>,
}

impl<'a> Parser<'a> {
    fn new(text: &'a str) -> Self {
        Parser { lx: Lexer { s: text.as_bytes(), i: 0 }, peeked: None }
    }
    fn peek(&mut self) -> Option<&Tok> {
        if self.peeked.is_none() {
            self.peeked = self.lx.next();
        }
        self.peeked.as_ref()
    }
    fn bump(&mut self) -> Option<Tok> {
        self.peek();
        self.peeked.take()
    }
    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == Some(&Tok::Punct(c)) {
            self.bump();
            true
        } else {
            false
        }
    }

    /// Skip a balanced ( ) / [ ] / { } group whose opener was just eaten.
    fn skip_group(&mut self, open: u8) {
        let close = match open {
            b'(' => b')',
            b'[' => b']',
            _ => b'}',
        };
        let mut depth = 1;
        while let Some(t) = self.bump() {
            match t {
                Tok::Punct(c) if c == open => depth += 1,
                Tok::Punct(c) if c == close => {
                    depth -= 1;
                    if depth == 0 {
                        return;
                    }
                }
                _ => {}
            }
        }
    }

    fn value(&mut self) -> Val {
        match self.bump() {
            Some(Tok::Num(n)) => Val::Num(n),
            Some(Tok::Str(s)) => Val::Str(s),
            Some(Tok::Word(w)) => match w.as_str() {
                "inf" => Val::Num(f64::INFINITY),
                "nan" => Val::Num(f64::NAN),
                "true" => Val::Num(1.0),
                "false" => Val::Num(0.0),
                _ => Val::Ident(w),
            },
            Some(Tok::Punct(c @ (b'(' | b'['))) => {
                let close = if c == b'(' { b')' } else { b']' };
                let mut l = Vec::new();
                loop {
                    if self.eat(close) || self.peek().is_none() {
                        break;
                    }
                    if self.eat(b',') {
                        continue;
                    }
                    l.push(self.value());
                }
                Val::List(l)
            }
            Some(Tok::Punct(b'{')) => {
                // Time samples: { frame: value, ... }
                let mut l = Vec::new();
                loop {
                    if self.eat(b'}') || self.peek().is_none() {
                        break;
                    }
                    if self.eat(b',') {
                        continue;
                    }
                    let Some(Tok::Num(t)) = self.bump() else { continue };
                    self.eat(b':');
                    l.push((t, self.value()));
                }
                Val::Samples(l)
            }
            _ => Val::Ident(String::new()),
        }
    }

    /// The body of a prim, after its `{`.
    fn body(&mut self, prim: &mut Prim) {
        loop {
            match self.peek().cloned() {
                None => return,
                Some(Tok::Punct(b'}')) => {
                    self.bump();
                    return;
                }
                Some(Tok::Word(w)) if w == "def" || w == "over" || w == "class" => {
                    self.bump();
                    let mut kid = Prim::default();
                    if let Some(Tok::Word(t)) = self.peek().cloned() {
                        self.bump();
                        kid.kind = t;
                    }
                    if let Some(Tok::Str(n)) = self.bump() {
                        kid.name = n;
                    }
                    if self.eat(b'(') {
                        self.skip_group(b'(');
                    }
                    if self.eat(b'{') {
                        self.body(&mut kid);
                    }
                    prim.kids.push(kid);
                }
                Some(Tok::Word(w)) if w == "variantSet" => {
                    self.bump();
                    // variantSet "x" = { ... }: not read.
                    while let Some(t) = self.bump() {
                        if t == Tok::Punct(b'{') {
                            self.skip_group(b'{');
                            break;
                        }
                    }
                }
                Some(Tok::Punct(c)) => {
                    self.bump();
                    if matches!(c, b'(' | b'[' | b'{') {
                        self.skip_group(c);
                    }
                }
                Some(_) => {
                    // An attribute or relationship: words up to `=` (or the
                    // end of the line's declaration), the last word its name.
                    let mut words = Vec::new();
                    while let Some(Tok::Word(w)) | Some(Tok::Str(w)) = self.peek().cloned() {
                        self.bump();
                        words.push(w);
                    }
                    let Some(name) = words.last().cloned() else {
                        self.bump();
                        continue;
                    };
                    if self.eat(b'=') {
                        let v = self.value();
                        prim.attrs.insert(name, v);
                    }
                    if self.eat(b'(') {
                        self.skip_group(b'(');
                    }
                }
            }
        }
    }
}

/// An attribute's value at a frame: a time-sampled one held/lerped, a
/// plain one as it is.
fn at_time(attrs: &HashMap<String, Val>, name: &str, frame: f64) -> Option<Vec<f64>> {
    if let Some(Val::Samples(s)) = attrs.get(&format!("{name}.timeSamples")) {
        if s.is_empty() {
            return None;
        }
        let i = s.partition_point(|(t, _)| *t <= frame);
        if i == 0 {
            return Some(s[0].1.nums());
        }
        if i >= s.len() {
            return Some(s[s.len() - 1].1.nums());
        }
        let (a, b) = (&s[i - 1], &s[i]);
        let (va, vb) = (a.1.nums(), b.1.nums());
        if va.len() != vb.len() {
            return Some(va);
        }
        let u = (frame - a.0) / (b.0 - a.0).max(1e-12);
        return Some(va.iter().zip(&vb).map(|(x, y)| x + (y - x) * u).collect());
    }
    attrs.get(name).map(|v| v.nums()).filter(|v| !v.is_empty())
}

/// Every time a prim's attributes are sampled at.
fn sample_times(p: &Prim, out: &mut Vec<f64>) {
    for (k, v) in &p.attrs {
        if let (true, Val::Samples(s)) = (k.ends_with(".timeSamples"), v) {
            out.extend(s.iter().map(|(t, _)| *t));
        }
    }
}

fn op_matrix(attrs: &HashMap<String, Val>, op: &str, frame: f64) -> Option<Mat4> {
    let base = op.trim_start_matches("!invert!");
    let v = at_time(attrs, base, frame)?;
    let ty = base.strip_prefix("xformOp:")?.split(':').next()?;
    let f = |i: usize| v.get(i).copied().unwrap_or(0.0) as f32;
    let m = match ty {
        "translate" => Xform { translate: [f(0), f(1), f(2)], ..Default::default() }.matrix(),
        "scale" => {
            let s = if v.len() == 1 { [f(0); 3] } else { [f(0), f(1), f(2)] };
            Xform { scale: s, ..Default::default() }.matrix()
        }
        "rotateX" => Xform { rotate: [f(0), 0.0, 0.0], ..Default::default() }.matrix(),
        "rotateY" => Xform { rotate: [0.0, f(0), 0.0], ..Default::default() }.matrix(),
        "rotateZ" => Xform { rotate: [0.0, 0.0, f(0)], ..Default::default() }.matrix(),
        "rotateXYZ" => Xform { rotate: [f(0), f(1), f(2)], ..Default::default() }.matrix(),
        t if t.starts_with("rotate") && t.len() == 9 => {
            // rotateABC: A first, then B, then C.
            let mut m = identity();
            for (k, axis) in t[6..].chars().enumerate() {
                let mut r = [0.0f32; 3];
                let i = match axis {
                    'X' => 0,
                    'Y' => 1,
                    _ => 2,
                };
                r[i] = f(k);
                m = mat_mul(&Xform { rotate: r, ..Default::default() }.matrix(), &m);
            }
            m
        }
        "orient" => {
            // Quaternion (w, x, y, z).
            let (w, x, y, z) = (f(0), f(1), f(2), f(3));
            let mut m = identity();
            let r = [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
                [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
            ];
            for i in 0..3 {
                m[i][..3].copy_from_slice(&r[i]);
            }
            m
        }
        "transform" => {
            // USD is row-vector: translation in the last ROW. Transpose.
            let mut m = identity();
            for r in 0..4 {
                for c in 0..4 {
                    m[c][r] = f(r * 4 + c);
                }
            }
            m
        }
        _ => return None,
    };
    if op.starts_with("!invert!") {
        let x = Xform::from_matrix(&m);
        return Some(x.inverse_scaled([1.0; 3]));
    }
    Some(m)
}

/// A prim's local matrix at a frame: its ops in `xformOpOrder`, the first
/// applied LAST (USD's order: the list reads outermost first).
fn local_matrix(p: &Prim, frame: f64) -> Mat4 {
    let order: Vec<String> = match p.attrs.get("xformOpOrder") {
        Some(Val::List(l)) => l
            .iter()
            .filter_map(|v| match v {
                Val::Str(s) | Val::Ident(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    let mut m = identity();
    for op in &order {
        if let Some(o) = op_matrix(&p.attrs, op, frame) {
            m = mat_mul(&m, &o);
        }
    }
    m
}

pub fn parse_usda(text: &str) -> Result<SceneFile, String> {
    if !text.trim_start().starts_with("#usda") {
        return Err("not a .usda text layer (it should start with #usda)".into());
    }
    // Layer metadata: the first ( ... ), before any prim.
    let mut up_z = false;
    let mut fps = None;
    // USD's camera lengths are tenths of a scene unit; a unit is this many
    // metres (0.01 when the layer does not say).
    let mut metres = 0.01f32;
    let meta_start = text.find('(');
    if let Some(i) = meta_start {
        let first_def = text.find("def ").unwrap_or(text.len());
        if i < first_def {
            let meta = &text[i..first_def];
            up_z = meta.contains("upAxis = \"Z\"") || meta.contains("upAxis = 'Z'");
            if let Some(j) = meta.find("metersPerUnit") {
                let rest = meta[j + 13..].trim_start().trim_start_matches('=').trim_start();
                let n: String = rest.chars().take_while(|c| c.is_ascii_digit() || *c == '.' || *c == 'e' || *c == '-').collect();
                if let Ok(v) = n.parse::<f32>() {
                    metres = v;
                }
            }
            for key in ["framesPerSecond", "timeCodesPerSecond"] {
                if let Some(j) = meta.find(key) {
                    let rest = meta[j + key.len()..].trim_start().trim_start_matches('=').trim_start();
                    let n: String = rest.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
                    if let Ok(v) = n.parse::<f32>() {
                        fps = fps.or(Some(v));
                    }
                }
            }
        }
    }
    let mut ps = Parser::new(text);
    let mut root = Prim::default();
    ps.body(&mut root);
    let fix = if up_z { z_up_to_y_up() } else { identity() };
    let mut out = SceneFile { fps, tracks: Vec::new() };
    // Walk with the chain of ancestors, so each Camera/Xform's world
    // matrix composes its parents at every one of its frames.
    fn walk<'p>(p: &'p Prim, chain: &mut Vec<&'p Prim>, fix: &Mat4, mm: f32, out: &mut SceneFile) {
        chain.push(p);
        // A transform is a prim with ops of its own — a mesh under a moved
        // Xform is that Xform, not a second object — and an Xform holding a
        // camera is the camera's.
        let holds_camera = p.kids.iter().any(|k| k.kind == "Camera");
        let kind = match p.kind.as_str() {
            "Camera" => Some(TrackKind::Camera),
            _ if p.attrs.contains_key("xformOpOrder") && !holds_camera => Some(TrackKind::Xform),
            _ => None,
        };
        if let Some(kind) = kind.filter(|_| !p.name.is_empty()) {
            let mut times = Vec::new();
            for a in chain.iter() {
                sample_times(a, &mut times);
            }
            times.sort_by(f64::total_cmp);
            times.dedup();
            if times.is_empty() {
                times.push(0.0);
            }
            let mut track = Track { name: p.name.clone(), kind, keys: Vec::new() };
            for t in times {
                let mut m = *fix;
                for a in chain.iter() {
                    m = mat_mul(&m, &local_matrix(a, t));
                }
                let lens = (kind == TrackKind::Camera).then(|| {
                    let g = |n: &str, d: f64| at_time(&p.attrs, n, t).and_then(|v| v.first().copied()).unwrap_or(d) as f32;
                    let clip = at_time(&p.attrs, "clippingRange", t).unwrap_or(vec![1.0, 1_000_000.0]);
                    Lens {
                        focal: g("focalLength", 50.0 / mm as f64) * mm,
                        aperture: [g("horizontalAperture", 20.955 / mm as f64) * mm, g("verticalAperture", 15.2908 / mm as f64) * mm],
                        near: clip.first().copied().unwrap_or(1.0) as f32,
                        far: clip.get(1).copied().unwrap_or(1e6) as f32,
                    }
                });
                track.keys.push((t as f32, Xform::from_matrix(&m), lens));
            }
            out.tracks.push(track.finish());
        }
        for k in &p.kids {
            walk(k, chain, fix, mm, out);
        }
        chain.pop();
    }
    // Tenths of a unit, in mm.
    let mm = 100.0 * metres;
    for k in &root.kids {
        walk(k, &mut Vec::new(), &fix, mm, &mut out);
    }
    if out.tracks.is_empty() {
        return Err("usda: no Camera or Xform prims".into());
    }
    Ok(out)
}

/// The rotation of a pose as a 3x3, for tests.
#[cfg(test)]
fn rot_of(x: &Xform) -> [[f32; 3]; 3] {
    rotation(x.rotate)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    #[cfg(feature = "serde")]
    #[test]
    fn json_reads_a_camera_and_a_card() {
        let f = parse_json(
            r#"{ "fps": 25, "objects": [
                { "name": "cam", "type": "camera", "frames": [
                    { "frame": 1, "translate": [0, 1, 5], "rotate": [0, 10, 0], "focal": 35 },
                    { "frame": 11, "translate": [2, 1, 5], "rotate": [0, 30, 0] } ] },
                { "name": "wall", "frames": [ { "frame": 1, "matrix": [2,0,0,1, 0,2,0,2, 0,0,2,3, 0,0,0,1] } ] } ] }"#,
        )
        .unwrap();
        assert_eq!(f.fps, Some(25.0));
        let cam = f.track("", TrackKind::Camera).unwrap();
        let (x, lens) = cam.sample(6.0).unwrap();
        assert!(near(x.translate[0], 1.0) && near(x.rotate[1], 20.0), "{x:?}");
        assert!(near(lens.unwrap().focal, 35.0), "optics hold");
        let wall = f.track("wall", TrackKind::Xform).unwrap();
        let (w, _) = wall.sample(100.0).unwrap();
        assert!(near(w.translate[2], 3.0) && near(w.scale[0], 2.0), "{w:?}");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn z_up_turns_to_y_up() {
        let f = parse_json(r#"{ "up": "z", "objects": [ { "name": "a", "frames": [ { "frame": 0, "translate": [1, 2, 3] } ] } ] }"#).unwrap();
        let (x, _) = f.tracks[0].sample(0.0).unwrap();
        // Blender's (x, y, z) is our (x, z, -y).
        assert!(near(x.translate[0], 1.0) && near(x.translate[1], 3.0) && near(x.translate[2], -2.0), "{x:?}");
    }

    #[test]
    fn chan_reads_lines_and_a_vertical_fov() {
        let f = parse_chan("# comment\n1 0 0 5 0 0 0 40\n2 1 0 5 0 350 0 40\n", "shot").unwrap();
        let t = &f.tracks[0];
        assert_eq!(t.name, "shot");
        let (x, lens) = t.sample(1.5).unwrap();
        assert!(near(x.translate[0], 0.5));
        assert!(near(x.rotate[1], -5.0), "350 unwraps to -10, halfway is -5: {x:?}");
        let f40 = 12.0 / (20.0f32.to_radians()).tan();
        assert!(near(lens.unwrap().focal, f40));
    }

    #[test]
    fn usda_reads_cameras_xforms_parents_and_samples() {
        let text = r#"#usda 1.0
(
    upAxis = "Y"
    timeCodesPerSecond = 24
)

def Xform "rig" (
    kind = "group"
)
{
    double3 xformOp:translate = (0, 0, 10)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Camera "shotcam"
    {
        float focalLength.timeSamples = {
            1: 35,
            11: 50,
        }
        float horizontalAperture = 36
        float verticalAperture = 24
        float2 clippingRange = (0.1, 1000)
        double3 xformOp:translate.timeSamples = {
            1: (0, 1, 0),
            11: (4, 1, 0),
        }
        float3 xformOp:rotateXYZ = (0, 90, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
    }

    def Mesh "card1"
    {
        matrix4d xformOp:transform = ( (2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0), (1, 2, 3, 1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
        point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
    }
}
"#;
        let f = parse_usda(text).unwrap();
        assert_eq!(f.fps, Some(24.0));
        let names: Vec<&str> = f.tracks.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["shotcam", "card1"], "an Xform holding a camera is the camera's");
        let cam = f.track("shotcam", TrackKind::Camera).unwrap();
        let (x, lens) = cam.sample(6.0).unwrap();
        assert!(near(x.translate[0], 2.0) && near(x.translate[2], 10.0), "parent + own: {x:?}");
        let r = rot_of(&x);
        assert!(near(r[0][2], 1.0), "turned 90 about Y: {r:?}");
        let l = lens.unwrap();
        assert!(near(l.focal, 42.5) && near(l.aperture[0], 36.0) && near(l.near, 0.1));
        let card = f.track("card1", TrackKind::Xform).unwrap();
        let (c, _) = card.sample(1.0).unwrap();
        assert!(near(c.translate[0], 1.0) && near(c.translate[1], 2.0) && near(c.translate[2], 13.0), "{c:?}");
        assert!(near(c.scale[1], 2.0));
    }

    /// What Blender 5.2's USD exporter writes: Z up, metres, the camera
    /// an Xform holding a Camera prim, lengths in tenths of a unit, meshes
    /// under their objects' Xforms.
    #[test]
    fn a_blender_usd_export_reads_in_our_convention() {
        let text = r#"#usda 1.0
(
    defaultPrim = "root"
    metersPerUnit = 1
    timeCodesPerSecond = 24
    upAxis = "Z"
)

def Xform "root"
{
    def Xform "wall"
    {
        float3 xformOp:rotateXYZ = (90, -0, 0)
        double3 xformOp:translate = (0, 0, 1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
        def Mesh "Plane"
        {
            point3f[] points = [(-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 1, 0)]
        }
    }
    def Xform "shotcam"
    {
        float3 xformOp:rotateXYZ.timeSamples = {
            1: (90, -0, 0),
        }
        double3 xformOp:translate.timeSamples = {
            1: (0, -10, 1),
            24: (2, -10, 1),
        }
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
        def Camera "shotcam"
        {
            float2 clippingRange = (0.1, 1000)
            float focalLength = 0.35
            float horizontalAperture = 0.36
            float verticalAperture = 0.2025
        }
    }
}
"#;
        let f = parse_usda(text).unwrap();
        let names: Vec<(&str, TrackKind)> = f.tracks.iter().map(|t| (t.name.as_str(), t.kind)).collect();
        assert_eq!(names, [("wall", TrackKind::Xform), ("shotcam", TrackKind::Camera)]);
        let (cam, lens) = f.track("shotcam", TrackKind::Camera).unwrap().sample(1.0).unwrap();
        let l = lens.unwrap();
        assert!(near(l.focal, 35.0) && near(l.aperture[0], 36.0), "{l:?}");
        // Blender (0, -10, 1) is ours (0, 1, 10): in front of the origin,
        // looking down -Z at it, level.
        assert!(near(cam.translate[0], 0.0) && near(cam.translate[1], 1.0) && near(cam.translate[2], 10.0), "{cam:?}");
        let fwd = { let r = rot_of(&cam); [-r[0][2], -r[1][2], -r[2][2]] };
        assert!(near(fwd[2], -1.0), "looks down -Z: {fwd:?}");
        // The wall stands up facing the camera: its normal (+Z local) is +Z.
        let (wall, _) = f.track("wall", TrackKind::Xform).unwrap().sample(1.0).unwrap();
        let r = rot_of(&wall);
        assert!(near(r[2][2], 1.0) && near(wall.translate[1], 1.0), "{wall:?}");
    }

    #[test]
    fn the_wrong_kinds_of_file_say_what_to_do() {
        assert!(parse(Path::new("a.abc"), "").unwrap_err().contains("Blender"));
        assert!(parse(Path::new("a.usdc"), "").unwrap_err().contains(".usda"));
    }
}
