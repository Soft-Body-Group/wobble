//! LUT files: parsed once, shared by the LUT node and the viewer's AgX.
//!
//! Formats read here, all plain text:
//!
//! - **`.cube`** (Adobe / Resolve): `LUT_3D_SIZE` and/or `LUT_1D_SIZE`,
//!   `DOMAIN_MIN` / `DOMAIN_MAX` (per channel), `LUT_1D_INPUT_RANGE` /
//!   `LUT_3D_INPUT_RANGE` (Resolve's form). A file with BOTH sizes is a 1D
//!   shaper followed by a cube, applied in that order, which is how Resolve
//!   writes one.
//! - **`.3dl`** (Autodesk / Lustre / Nuke): integer triplets, BLUE fastest,
//!   an optional input mesh line, the output depth from a `Mesh in out`
//!   header or else inferred from the largest value (10/12/14/16 bit).
//! - **`.spi1d`** / **`.spi3d`** (Sony Imageworks, OCIO's own formats).
//! - **`.csp`** (Rising Sun Cinespace, which Resolve also exports): a
//!   per-channel prelut of explicit in/out points joined by a cubic, then
//!   a 1D or 3D table. The prelut is resampled onto an even
//!   [`CSP_RESAMPLE`]-point table, which is the one approximation here.
//!
//! Not read: CLF / CTF and OCIO configs. They are XML or carry
//! their own shapers and matrices, and half a reader for them would render
//! something plausible and wrong.
//!
//! Interpolation is linear for a 1D table and, for a cube, TETRAHEDRAL by
//! default (OCIO's, and Resolve's when its project setting says so) or
//! TRILINEAR ([`Interp`]; Resolve's out-of-the-box project setting).
//! They differ most on the cube's diagonal, which is where every grey
//! lives, so matching Resolve means matching its setting. Input outside the domain clamps. [`Lut::apply`] is the CPU
//! mirror the GPU is tested against, and [`Lut::texture`] is the one place
//! that lays a table out for the GPU — the WGSL that reads it is
//! `nodes::lut`'s, and both sides take the layout from [`Layout`].

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// The largest cube we take: a 256^3 table is already 200 MB of floats.
pub const MAX_3D: usize = 256;
/// The largest 1D table (`.cube` allows 65536; spi1d files go further).
pub const MAX_1D: usize = 1 << 20;

/// How a cube is interpolated between its lattice points.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Interp {
    /// Six tetrahedra a cell: OCIO's choice, Resolve's "Tetrahedral".
    #[default]
    Tetrahedral,
    /// Eight corners a cell: Resolve's default project setting.
    Trilinear,
}

impl Interp {
    pub const ALL: [Self; 2] = [Self::Tetrahedral, Self::Trilinear];
    pub fn label(self) -> &'static str {
        match self {
            Self::Tetrahedral => "tetrahedral",
            Self::Trilinear => "trilinear",
        }
    }
}

/// How many even steps a `.csp` prelut's cubic is resampled to.
pub const CSP_RESAMPLE: usize = 16384;

/// A 1D table: `data[i]` is the output at input `min + i/(n-1) * (max-min)`,
/// per channel.
#[derive(Clone, Debug, PartialEq)]
pub struct Lut1d {
    pub data: Vec<[f32; 3]>,
    pub min: [f32; 3],
    pub max: [f32; 3],
}

/// A 3D LUT: `size`^3 RGB entries, red fastest (the .cube order), over the
/// input domain `min..max` per channel.
#[derive(Clone, Debug, PartialEq)]
pub struct Lut3d {
    pub size: usize,
    pub data: Vec<[f32; 3]>,
    pub min: [f32; 3],
    pub max: [f32; 3],
}

/// What a LUT file holds: an optional 1D shaper, then an optional cube.
/// At least one is present.
#[derive(Clone, Debug)]
pub struct Lut {
    pub shaper: Option<Lut1d>,
    pub cube: Option<Lut3d>,
    /// Hash of the file's bytes: what the render cache keys on, so an
    /// edited file under the same name never replays a stale render.
    pub hash: u64,
    /// The format it was read as (`"cube"`, `"3dl"`, `"spi1d"`, `"spi3d"`, `"csp"`).
    pub format: &'static str,
}

fn norm(v: f32, mn: f32, mx: f32) -> f32 {
    let d = mx - mn;
    let t = if d.abs() > 1e-20 { (v - mn) / d } else { 0.0 };
    t.clamp(0.0, 1.0)
}

impl Lut1d {
    /// Linear interpolation per channel, input clamped to the domain.
    pub fn sample(&self, c: [f32; 3]) -> [f32; 3] {
        let n = self.data.len();
        let last = (n - 1) as f32;
        [0, 1, 2].map(|k| {
            let t = norm(c[k], self.min[k], self.max[k]) * last;
            let i = (t.floor() as usize).min(n - 2);
            let f = t - i as f32;
            let (a, b) = (self.data[i][k], self.data[i + 1][k]);
            a + (b - a) * f
        })
    }
}

impl Lut3d {
    /// A cube over the 0..1 domain.
    pub fn new(size: usize, data: Vec<[f32; 3]>) -> Self {
        Self { size, data, min: [0.0; 3], max: [1.0; 3] }
    }

    /// Parse a `.cube` that must hold a 3D table (the viewer's AgX cubes).
    pub fn parse_cube(text: &str) -> Result<Self> {
        parse_cube(text)?.cube.context("a 1D .cube, not a 3D one")
    }

    pub fn at(&self, r: usize, g: usize, b: usize) -> [f32; 3] {
        self.data[r + self.size * (g + self.size * b)]
    }

    /// Interpolate `c` as `how` says.
    pub fn sample_with(&self, c: [f32; 3], how: Interp) -> [f32; 3] {
        match how {
            Interp::Tetrahedral => self.sample(c),
            Interp::Trilinear => self.sample_trilinear(c),
        }
    }

    /// Trilinear interpolation: the eight corners of the cell, weighted.
    /// Input normalised by the domain and clamped to it.
    pub fn sample_trilinear(&self, c: [f32; 3]) -> [f32; 3] {
        let n = (self.size - 1) as f32;
        let p = [0, 1, 2].map(|k| norm(c[k], self.min[k], self.max[k]) * n);
        let i = p.map(|v| (v.floor() as usize).min(self.size - 2));
        let [fr, fg, fb] = [p[0] - i[0] as f32, p[1] - i[1] as f32, p[2] - i[2] as f32];
        let v = |dr, dg, db| self.at(i[0] + dr, i[1] + dg, i[2] + db);
        let lerp = |a: [f32; 3], b: [f32; 3], t: f32| [0, 1, 2].map(|k| a[k] + (b[k] - a[k]) * t);
        let x00 = lerp(v(0, 0, 0), v(1, 0, 0), fr);
        let x10 = lerp(v(0, 1, 0), v(1, 1, 0), fr);
        let x01 = lerp(v(0, 0, 1), v(1, 0, 1), fr);
        let x11 = lerp(v(0, 1, 1), v(1, 1, 1), fr);
        lerp(lerp(x00, x10, fg), lerp(x01, x11, fg), fb)
    }

    /// Tetrahedral interpolation — what OCIO uses for these cubes. Input
    /// normalised by the domain and clamped to it, as OCIO does.
    pub fn sample(&self, c: [f32; 3]) -> [f32; 3] {
        let n = (self.size - 1) as f32;
        let p = [0, 1, 2].map(|k| norm(c[k], self.min[k], self.max[k]) * n);
        let i = p.map(|v| (v.floor() as usize).min(self.size - 2));
        let f = [p[0] - i[0] as f32, p[1] - i[1] as f32, p[2] - i[2] as f32];
        let v = |dr, dg, db| self.at(i[0] + dr, i[1] + dg, i[2] + db);
        let (c000, c111) = (v(0, 0, 0), v(1, 1, 1));
        let lerp4 = |a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3], w: [f32; 4]| {
            [0, 1, 2].map(|k| a[k] * w[0] + b[k] * w[1] + c[k] * w[2] + d[k] * w[3])
        };
        let [fr, fg, fb] = f;
        if fr > fg {
            if fg > fb {
                lerp4(c000, v(1, 0, 0), v(1, 1, 0), c111, [1.0 - fr, fr - fg, fg - fb, fb])
            } else if fr > fb {
                lerp4(c000, v(1, 0, 0), v(1, 0, 1), c111, [1.0 - fr, fr - fb, fb - fg, fg])
            } else {
                lerp4(c000, v(0, 0, 1), v(1, 0, 1), c111, [1.0 - fb, fb - fr, fr - fg, fg])
            }
        } else if fb > fg {
            lerp4(c000, v(0, 0, 1), v(0, 1, 1), c111, [1.0 - fb, fb - fg, fg - fr, fr])
        } else if fb > fr {
            lerp4(c000, v(0, 1, 0), v(0, 1, 1), c111, [1.0 - fg, fg - fb, fb - fr, fr])
        } else {
            lerp4(c000, v(0, 1, 0), v(1, 1, 0), c111, [1.0 - fg, fg - fr, fr - fb, fb])
        }
    }
}

/// How a [`Lut`] sits in its GPU texture (rgba, one 2D texture):
///
/// - the cube's blue slices are tiles of `size` x `size`, `tiles` to a
///   row: entry (r, g, b) is texel `(r + size·(b % tiles), g + size·(b / tiles))`;
/// - the 1D table starts at row `row1d`, `width` entries to a row.
///
/// 2D, not 3D, so the fused pass's bindings stay one kind of texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub width: u32,
    pub height: u32,
    pub tiles: u32,
    pub row1d: u32,
}

impl Lut {
    /// The shaper then the cube, the cube tetrahedral.
    pub fn apply(&self, c: [f32; 3]) -> [f32; 3] {
        self.apply_with(c, Interp::Tetrahedral)
    }

    /// The shaper then the cube, the cube interpolated as `how` says.
    pub fn apply_with(&self, c: [f32; 3], how: Interp) -> [f32; 3] {
        let c = self.shaper.as_ref().map_or(c, |s| s.sample(c));
        self.cube.as_ref().map_or(c, |l| l.sample_with(c, how))
    }

    pub fn layout(&self) -> Layout {
        let (n, tiles, rows3) = match &self.cube {
            Some(l) => {
                let n = l.size as u32;
                let tiles = (n as f32).sqrt().ceil() as u32;
                (n, tiles, n * n.div_ceil(tiles))
            }
            None => (0, 1, 0),
        };
        let width = (n * tiles).max(1024);
        let rows1 = self.shaper.as_ref().map_or(0, |s| (s.data.len() as u32).div_ceil(width));
        Layout { width, height: (rows3 + rows1).max(1), tiles, row1d: rows3 }
    }

    /// The table as rgba texels in [`Layout`] order (alpha 1, unused texels 0).
    pub fn texture(&self) -> (Layout, Vec<[f32; 4]>) {
        let lay = self.layout();
        let w = lay.width as usize;
        let mut px = vec![[0.0f32; 4]; w * lay.height as usize];
        if let Some(l) = &self.cube {
            let (n, t) = (l.size, lay.tiles as usize);
            for b in 0..n {
                for g in 0..n {
                    for r in 0..n {
                        let [x, y, z] = l.at(r, g, b);
                        px[(r + n * (b % t)) + w * (g + n * (b / t))] = [x, y, z, 1.0];
                    }
                }
            }
        }
        if let Some(s) = &self.shaper {
            let base = w * lay.row1d as usize;
            for (i, [x, y, z]) in s.data.iter().enumerate() {
                px[base + i] = [*x, *y, *z, 1.0];
            }
        }
        (lay, px)
    }
}

fn finite(v: [f32; 3], line: &str) -> Result<[f32; 3]> {
    if v.iter().all(|x| x.is_finite()) {
        Ok(v)
    } else {
        bail!("a value that is not a number: {line}")
    }
}

fn floats(l: &str) -> Option<Vec<f32>> {
    l.split_whitespace().map(|t| t.parse::<f32>().ok()).collect()
}

fn check_1d(n: usize) -> Result<()> {
    if !(2..=MAX_1D).contains(&n) {
        bail!("1D LUT size {n} (2..{MAX_1D})");
    }
    Ok(())
}

fn check_3d(n: usize) -> Result<()> {
    if !(2..=MAX_3D).contains(&n) {
        bail!("3D LUT size {n} (2..{MAX_3D})");
    }
    Ok(())
}

/// Parse an Adobe / Resolve `.cube`: 1D, 3D, or a 1D shaper and a cube.
pub fn parse_cube(text: &str) -> Result<Lut> {
    let text = text.trim_start_matches('\u{feff}');
    let (mut n1, mut n3) = (0usize, 0usize);
    let (mut dmin, mut dmax) = ([0.0f32; 3], [1.0f32; 3]);
    let (mut r1, mut r3): (Option<[f32; 2]>, Option<[f32; 2]>) = (None, None);
    let mut data: Vec<[f32; 3]> = Vec::new();
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        if let Some(v) = floats(l) {
            if v.len() != 3 {
                bail!("a .cube entry needs three values: {l}");
            }
            data.push(finite([v[0], v[1], v[2]], l)?);
            continue;
        }
        let key = l.split_whitespace().next().unwrap_or("");
        let rest = l[key.len()..].trim();
        let nums = || floats(rest).with_context(|| format!("unreadable {key}: {l}"));
        let int = || rest.parse::<usize>().with_context(|| format!("unreadable {key}: {l}"));
        let three = |v: Vec<f32>| -> Result<[f32; 3]> {
            match v[..] {
                [a, b, c] => Ok([a, b, c]),
                _ => bail!("{key} needs three values: {l}"),
            }
        };
        let two = |v: Vec<f32>| -> Result<[f32; 2]> {
            match v[..] {
                [a, b] => Ok([a, b]),
                _ => bail!("{key} needs two values: {l}"),
            }
        };
        match key {
            "LUT_1D_SIZE" => n1 = int()?,
            "LUT_3D_SIZE" => n3 = int()?,
            "DOMAIN_MIN" => dmin = three(nums()?)?,
            "DOMAIN_MAX" => dmax = three(nums()?)?,
            "LUT_1D_INPUT_RANGE" => r1 = Some(two(nums()?)?),
            "LUT_3D_INPUT_RANGE" => r3 = Some(two(nums()?)?),
            // TITLE and the rest (LUT_IN_VIDEO_RANGE, …) change no number
            // we could honour.
            _ if key.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) => {}
            _ => bail!("unreadable .cube line: {l}"),
        }
    }
    if n1 == 0 && n3 == 0 {
        bail!("no LUT_1D_SIZE or LUT_3D_SIZE: not a .cube LUT");
    }
    let want = n1 + n3 * n3 * n3;
    if data.len() != want {
        bail!("the header says {want} entries, the file has {}", data.len());
    }
    let range = |r: Option<[f32; 2]>| match r {
        Some([a, b]) => ([a; 3], [b; 3]),
        None => (dmin, dmax),
    };
    let shaper = if n1 > 0 {
        check_1d(n1)?;
        let (min, max) = range(r1);
        Some(Lut1d { data: data[..n1].to_vec(), min, max })
    } else {
        None
    };
    let cube = if n3 > 0 {
        check_3d(n3)?;
        let (min, max) = range(r3);
        Some(Lut3d { size: n3, data: data[n1..].to_vec(), min, max })
    } else {
        None
    };
    Ok(Lut { shaper, cube, hash: 0, format: "cube" })
}

fn cube_root(m: usize) -> Option<usize> {
    let n = (m as f64).cbrt().round() as usize;
    (n * n * n == m).then_some(n)
}

/// Parse an Autodesk / Lustre `.3dl`: integer triplets, blue fastest.
pub fn parse_3dl(text: &str) -> Result<Lut> {
    let mut out_bits: Option<u32> = None;
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        let key = l.split_whitespace().next().unwrap_or("");
        if key.eq_ignore_ascii_case("3DMESH") {
            continue;
        }
        if key.eq_ignore_ascii_case("Mesh") {
            let v: Vec<u32> = l[key.len()..]
                .split_whitespace()
                .map(|t| t.parse().ok())
                .collect::<Option<_>>()
                .with_context(|| format!("unreadable Mesh line: {l}"))?;
            out_bits = v.get(1).copied();
            continue;
        }
        if key.eq_ignore_ascii_case("LUT8") || key.eq_ignore_ascii_case("gamma") {
            continue;
        }
        rows.push(floats(l).with_context(|| format!("unreadable .3dl line: {l}"))?);
    }
    // The mesh line, when there is one, is the first row: its length is the
    // cube's size (a 3-point mesh is the one ambiguous case, settled by the
    // count of rows that follow).
    let triples = rows.iter().filter(|r| r.len() == 3).count();
    let (mesh, body): (Option<&Vec<f32>>, &[Vec<f32>]) = match rows.first() {
        Some(f) if f.len() != 3 => (Some(f), &rows[1..]),
        Some(f) if cube_root(triples).is_none() && cube_root(triples - 1) == Some(3) => {
            (Some(f), &rows[1..])
        }
        _ => (None, &rows[..]),
    };
    if let Some(bad) = body.iter().find(|r| r.len() != 3) {
        bail!(".3dl entries need three values, found {}", bad.len());
    }
    let n = cube_root(body.len()).with_context(|| format!("{} entries is not a cube", body.len()))?;
    if let Some(m) = mesh {
        if m.len() != n {
            bail!("the mesh line has {} points but the cube is {n}", m.len());
        }
    }
    check_3d(n)?;
    let top = body.iter().flatten().fold(0.0f32, |a, &b| a.max(b));
    let scale = match out_bits {
        Some(b) if (1..=32).contains(&b) => ((1u64 << b) - 1) as f32,
        Some(b) => bail!("an output depth of {b} bits"),
        // Floats in 0..1 are already normalised; integers say their depth
        // by their largest value.
        None if top <= 1.0 => 1.0,
        None => [10u32, 12, 14, 16]
            .into_iter()
            .map(|b| ((1u64 << b) - 1) as f32)
            .find(|&s| top <= s)
            .with_context(|| format!("values up to {top}: deeper than 16 bit"))?,
    };
    let mut data = vec![[0.0f32; 3]; n * n * n];
    for (k, r) in body.iter().enumerate() {
        let (b, g, rr) = (k % n, (k / n) % n, k / (n * n));
        data[rr + n * (g + n * b)] = finite([r[0] / scale, r[1] / scale, r[2] / scale], "3dl")?;
    }
    Ok(Lut { shaper: None, cube: Some(Lut3d::new(n, data)), hash: 0, format: "3dl" })
}

/// Parse a Sony Imageworks `.spi1d`.
pub fn parse_spi1d(text: &str) -> Result<Lut> {
    let (mut from, mut len, mut comps) = ([0.0f32, 1.0], 0usize, 1usize);
    let mut data: Vec<[f32; 3]> = Vec::new();
    let mut inside = false;
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        if inside {
            if l.starts_with('}') {
                inside = false;
                continue;
            }
            let v = floats(l).with_context(|| format!("unreadable .spi1d value: {l}"))?;
            let c = match (comps, &v[..]) {
                (1, [a]) => [*a; 3],
                (3, [a, b, c]) => [*a, *b, *c],
                _ => bail!("expected {comps} value(s) a line: {l}"),
            };
            data.push(finite(c, l)?);
            continue;
        }
        let key = l.split_whitespace().next().unwrap_or("");
        let rest = l[key.len()..].trim();
        match key {
            "Version" => {}
            "From" => {
                from = match floats(rest).as_deref() {
                    Some([a, b]) => [*a, *b],
                    _ => bail!("unreadable From: {l}"),
                }
            }
            "Length" => len = rest.parse().with_context(|| format!("unreadable Length: {l}"))?,
            "Components" => {
                comps = rest.parse().with_context(|| format!("unreadable Components: {l}"))?;
                if comps != 1 && comps != 3 {
                    bail!("{comps} components (1 or 3 are read)");
                }
            }
            "{" => inside = true,
            _ => bail!("unreadable .spi1d line: {l}"),
        }
    }
    if data.len() != len {
        bail!("Length {len} but {} values", data.len());
    }
    check_1d(len)?;
    let shaper = Lut1d { data, min: [from[0]; 3], max: [from[1]; 3] };
    Ok(Lut { shaper: Some(shaper), cube: None, hash: 0, format: "spi1d" })
}

/// Parse a Sony Imageworks `.spi3d`: indexed entries, any order.
pub fn parse_spi3d(text: &str) -> Result<Lut> {
    let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#'));
    let head = lines.next().unwrap_or("");
    if !head.starts_with("SPILUT") {
        bail!("not an .spi3d (no SPILUT header)");
    }
    let _ = lines.next().context("no dimensions line")?;
    let size: Vec<usize> = lines
        .next()
        .context("no size line")?
        .split_whitespace()
        .map(|t| t.parse().ok())
        .collect::<Option<_>>()
        .context("unreadable size line")?;
    let n = match size[..] {
        [a, b, c] if a == b && b == c => a,
        _ => bail!("sizes {size:?}: only a cube (equal sides) is read"),
    };
    check_3d(n)?;
    let mut data = vec![[0.0f32; 3]; n * n * n];
    let mut seen = vec![false; n * n * n];
    for l in lines {
        let v = floats(l).with_context(|| format!("unreadable .spi3d line: {l}"))?;
        let [i, j, k, r, g, b] = v[..] else { bail!(".spi3d entries are `i j k r g b`: {l}") };
        let idx = [i, j, k].map(|x| x as usize);
        if [i, j, k].iter().any(|&x| x < 0.0 || x.fract() != 0.0) || idx.iter().any(|&x| x >= n) {
            bail!("index out of the cube: {l}");
        }
        let at = idx[0] + n * (idx[1] + n * idx[2]);
        data[at] = finite([r, g, b], l)?;
        seen[at] = true;
    }
    if let Some(miss) = seen.iter().position(|s| !s) {
        bail!("entry {} {} {} is missing", miss % n, (miss / n) % n, miss / (n * n));
    }
    Ok(Lut { shaper: None, cube: Some(Lut3d::new(n, data)), hash: 0, format: "spi3d" })
}

/// One channel of a `.csp` prelut: increasing `ins`, matching `outs`.
struct Prelut {
    ins: Vec<f32>,
    outs: Vec<f32>,
}

impl Prelut {
    /// The prelut between its points is a CUBIC, as Rising Sun's reference
    /// code (and OCIO, which carries it) draws it: Hermite segments whose
    /// slope at an inner point is the chord across its two neighbours, and
    /// at an end twice the end segment's slope less the next point's. Out
    /// of range clamps to the end values.
    fn eval(&self, x: f32) -> f32 {
        let (i, o) = (&self.ins, &self.outs);
        let n = i.len();
        if x <= i[0] {
            return o[0];
        }
        if x >= i[n - 1] {
            return o[n - 1];
        }
        let k = i.partition_point(|&v| v <= x).clamp(1, n - 1);
        let (x0, x1, y0, y1) = (i[k - 1], i[k], o[k - 1], o[k]);
        let h = (x1 - x0).max(1e-20);
        let t = (x - x0) / h;
        if n == 2 {
            return y0 + (y1 - y0) * t;
        }
        let inner = |j: usize| (o[j + 1] - o[j - 1]) / (i[j + 1] - i[j - 1]);
        let slope = |j: usize| {
            if j == 0 {
                2.0 * (o[1] - o[0]) / (i[1] - i[0]) - inner(1)
            } else if j == n - 1 {
                2.0 * (o[n - 1] - o[n - 2]) / (i[n - 1] - i[n - 2]) - inner(n - 2)
            } else {
                inner(j)
            }
        };
        let (m0, m1) = (slope(k - 1), slope(k));
        let (t2, t3) = (t * t, t * t * t);
        (2.0 * t3 - 3.0 * t2 + 1.0) * y0
            + (t3 - 2.0 * t2 + t) * h * m0
            + (-2.0 * t3 + 3.0 * t2) * y1
            + (t3 - t2) * h * m1
    }

    fn is_identity(&self) -> bool {
        self.ins.iter().zip(&self.outs).all(|(a, b)| (a - b).abs() < 1e-6)
            && self.ins[0].abs() < 1e-6
            && (self.ins[self.ins.len() - 1] - 1.0).abs() < 1e-6
    }
}

/// Three per-channel functions on even steps over each channel's own
/// domain, evaluated by `f(channel, x)`.
fn resample(pre: &[Prelut; 3], n: usize, f: impl Fn(usize, f32) -> f32) -> Lut1d {
    let min = [0, 1, 2].map(|k| pre[k].ins[0]);
    let max = [0, 1, 2].map(|k| pre[k].ins[pre[k].ins.len() - 1]);
    let data = (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1) as f32;
            [0, 1, 2].map(|k| f(k, min[k] + (max[k] - min[k]) * t))
        })
        .collect();
    Lut1d { data, min, max }
}

/// Parse a Cinespace `.csp`: a prelut a channel, then a 1D or 3D table.
pub fn parse_csp(text: &str) -> Result<Lut> {
    let text = text.trim_start_matches('\u{feff}');
    let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#'));
    if !lines.next().is_some_and(|l| l.starts_with("CSPLUTV100")) {
        bail!("not a .csp (no CSPLUTV100 header)");
    }
    let kind = lines.next().context("no 1D / 3D line")?;
    let three = match kind {
        "3D" => true,
        "1D" => false,
        _ => bail!("expected 1D or 3D, found {kind}"),
    };
    let mut toks: Vec<&str> = Vec::new();
    let mut meta = false;
    for l in lines {
        if l.starts_with("BEGIN METADATA") {
            meta = true;
        } else if l.starts_with("END METADATA") {
            meta = false;
        } else if !meta {
            toks.extend(l.split_whitespace());
        }
    }
    let mut it = toks.into_iter();
    let mut num = |what: &str| -> Result<f32> {
        let t = it.next().with_context(|| format!("the file ends before the {what}"))?;
        let v: f32 = t.parse().with_context(|| format!("unreadable {what}: {t}"))?;
        if !v.is_finite() {
            bail!("a {what} that is not a number: {t}");
        }
        Ok(v)
    };
    let count = |v: f32, what: &str| -> Result<usize> {
        if v < 0.0 || v.fract() != 0.0 {
            bail!("{what} {v} is not a count");
        }
        Ok(v as usize)
    };
    let mut pre: Vec<Prelut> = Vec::new();
    for ch in ["red", "green", "blue"] {
        let n = count(num("prelut size")?, "prelut size")?;
        if !(2..=MAX_1D).contains(&n) {
            bail!("{ch} prelut has {n} points (2..{MAX_1D})");
        }
        let ins = (0..n).map(|_| num("prelut input")).collect::<Result<Vec<_>>>()?;
        let outs = (0..n).map(|_| num("prelut output")).collect::<Result<Vec<_>>>()?;
        if ins.windows(2).any(|w| w[1] <= w[0]) {
            bail!("the {ch} prelut's inputs do not increase");
        }
        pre.push(Prelut { ins, outs });
    }
    let pre: [Prelut; 3] = pre.try_into().ok().context("three preluts")?;
    let identity = pre.iter().all(Prelut::is_identity);
    let lut = if three {
        let s: Vec<usize> =
            (0..3).map(|_| num("cube size").and_then(|v| count(v, "cube size"))).collect::<Result<_>>()?;
        if s[0] != s[1] || s[1] != s[2] {
            bail!("sizes {s:?}: only a cube (equal sides) is read");
        }
        let n = s[0];
        check_3d(n)?;
        let data = (0..n * n * n)
            .map(|_| Ok([num("cube entry")?, num("cube entry")?, num("cube entry")?]))
            .collect::<Result<Vec<_>>>()?;
        // Two points a channel is a straight line, which a table holds
        // exactly; anything more is a cubic, resampled.
        let straight = pre.iter().all(|p| p.ins.len() == 2);
        let shaper = if identity {
            None
        } else if straight {
            let data = (0..2).map(|i| [0, 1, 2].map(|k| pre[k].outs[i])).collect();
            let min = [0, 1, 2].map(|k| pre[k].ins[0]);
            let max = [0, 1, 2].map(|k| pre[k].ins[1]);
            Some(Lut1d { data, min, max })
        } else {
            Some(resample(&pre, CSP_RESAMPLE, |k, x| pre[k].eval(x)))
        };
        Lut { shaper, cube: Some(Lut3d::new(n, data)), hash: 0, format: "csp" }
    } else {
        let n = count(num("1D size")?, "1D size")?;
        check_1d(n)?;
        let data = (0..n)
            .map(|_| Ok([num("1D entry")?, num("1D entry")?, num("1D entry")?]))
            .collect::<Result<Vec<_>>>()?;
        let table = Lut1d { data, min: [0.0; 3], max: [1.0; 3] };
        // The prelut feeds the table: compose the two into one 1D.
        let shaper = if identity {
            table
        } else {
            let m = CSP_RESAMPLE.max(n);
            resample(&pre, m, |k, x| {
                let mut c = [0.0; 3];
                c[k] = pre[k].eval(x);
                table.sample(c)[k]
            })
        };
        Lut { shaper: Some(shaper), cube: None, hash: 0, format: "csp" }
    };
    if it.next().is_some() {
        bail!("more values than the header says");
    }
    Ok(lut)
}

/// The extensions [`parse`] reads, for the file dialog.
pub const EXTS: [&str; 5] = ["cube", "3dl", "spi1d", "spi3d", "csp"];

/// Parse `bytes` by the file's extension.
pub fn parse(ext: &str, bytes: &[u8]) -> Result<Lut> {
    let text = std::str::from_utf8(bytes).context("not a text file")?;
    let mut lut = match ext.to_ascii_lowercase().as_str() {
        "cube" => parse_cube(text)?,
        "3dl" => parse_3dl(text)?,
        "spi1d" => parse_spi1d(text)?,
        "spi3d" => parse_spi3d(text)?,
        "csp" => parse_csp(text)?,
        "" => bail!("no extension: cannot tell the format"),
        e => bail!(".{e} is not a LUT format wobble reads (.cube .3dl .spi1d .spi3d .csp)"),
    };
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (ext.to_ascii_lowercase(), bytes).hash(&mut h);
    lut.hash = h.finish();
    Ok(lut)
}

/// How long a load is trusted before the file is looked at again — a
/// local stat, or a re-read where there is nothing to stat.
const RECHECK_LOCAL: std::time::Duration = std::time::Duration::from_millis(500);
const RECHECK_REMOTE: std::time::Duration = std::time::Duration::from_secs(5);

type Sig = Option<(std::time::SystemTime, u64)>;

struct Entry {
    checked: web_time::Instant,
    sig: Sig,
    got: Result<Arc<Lut>, String>,
}

fn cache() -> &'static Mutex<HashMap<PathBuf, Entry>> {
    static C: std::sync::OnceLock<Mutex<HashMap<PathBuf, Entry>>> = std::sync::OnceLock::new();
    C.get_or_init(Default::default)
}

fn signature(path: &Path) -> Sig {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if crate::objstore::is_remote(path) {
            return None;
        }
        let m = std::fs::metadata(path).ok()?;
        Some((m.modified().ok()?, m.len()))
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = path;
        None
    }
}

fn read_parse(path: &Path) -> Result<Arc<Lut>, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let bytes = crate::vfs::read(path).map_err(|e| format!("{e:#}"))?;
    parse(ext, &bytes).map(Arc::new).map_err(|e| format!("{e:#}"))
}

/// The LUT at a TYPED path (a leading `~` expanded), read through `vfs`.
/// Cached: the file is stat'ed at most every half second and re-read only
/// when its time or size moved, so asking every frame costs nothing. The
/// error is written for a person; an empty path is `Err("")`.
pub fn load(typed: &str) -> Result<Arc<Lut>, String> {
    if typed.trim().is_empty() {
        return Err(String::new());
    }
    let path = crate::paths::expand_home(typed);
    let now = web_time::Instant::now();
    let mut c = cache().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(e) = c.get_mut(&path) {
        let wait = if e.sig.is_some() { RECHECK_LOCAL } else { RECHECK_REMOTE };
        if now.duration_since(e.checked) < wait {
            return e.got.clone();
        }
        let sig = signature(&path);
        if sig.is_some() && sig == e.sig {
            e.checked = now;
            return e.got.clone();
        }
    }
    let sig = signature(&path);
    let got = read_parse(&path);
    c.insert(path, Entry { checked: now, sig, got: got.clone() });
    got
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    /// A 2^3 cube that swaps red and blue, in .cube order.
    const SWAP: &str = "TITLE \"swap\"\n# a comment\nLUT_3D_SIZE 2\n\
        0 0 0\n0 0 1\n0 1 0\n0 1 1\n1 0 0\n1 0 1\n1 1 0\n1 1 1\n";

    #[test]
    fn a_cube_reads_red_fastest() {
        let l = parse_cube(SWAP).unwrap();
        let c = l.cube.as_ref().unwrap();
        assert_eq!(c.size, 2);
        assert_eq!(c.at(1, 0, 0), [0.0, 0.0, 1.0]);
        assert!(close(l.apply([0.2, 0.5, 0.9]), [0.9, 0.5, 0.2]));
        assert!(l.shaper.is_none());
    }

    #[test]
    fn a_1d_cube_honours_its_domain() {
        // Output = input / 2 over a domain of 0..2: an input of 1 is the
        // middle entry.
        let t = "LUT_1D_SIZE 3\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 2 2 2\n0 0 0\n0.5 0.5 0.5\n1 1 1\n";
        let l = parse_cube(t).unwrap();
        assert!(close(l.apply([1.0, 0.5, 2.0]), [0.5, 0.25, 1.0]));
        // Outside the domain clamps.
        assert!(close(l.apply([-1.0, 3.0, 0.0]), [0.0, 1.0, 0.0]));
        let r = "LUT_1D_SIZE 2\nLUT_1D_INPUT_RANGE -1 1\n0 0 0\n1 1 1\n";
        assert!(close(parse_cube(r).unwrap().apply([0.0, 0.5, -1.0]), [0.5, 0.75, 0.0]));
    }

    #[test]
    fn a_shaper_and_a_cube_in_one_file_apply_in_order() {
        // 1D squares (approximately — two points: identity on 0..4 domain
        // mapped to 0..1), then the swap cube.
        let t = format!(
            "LUT_1D_SIZE 2\nLUT_3D_SIZE 2\nLUT_1D_INPUT_RANGE 0 4\nLUT_3D_INPUT_RANGE 0 1\n0 0 0\n1 1 1\n{}",
            SWAP.lines().skip(3).collect::<Vec<_>>().join("\n")
        );
        let l = parse_cube(&t).unwrap();
        assert!(l.shaper.is_some() && l.cube.is_some());
        assert!(close(l.apply([0.4, 2.0, 4.0]), [1.0, 0.5, 0.1]));
        let lay = l.layout();
        assert_eq!(lay.row1d, 2);
        let (_, px) = l.texture();
        assert_eq!(px[(lay.width * lay.row1d) as usize + 1], [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn a_domain_moves_the_cube() {
        let t = "LUT_3D_SIZE 2\nDOMAIN_MIN -1 -1 -1\nDOMAIN_MAX 1 1 1\n\
            0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n";
        let l = parse_cube(t).unwrap();
        assert!(close(l.apply([0.0, -1.0, 1.0]), [0.5, 0.0, 1.0]));
    }

    #[test]
    fn bad_cubes_say_why() {
        for (t, want) in [
            ("LUT_3D_SIZE 2\n0 0 0\n", "entries"),
            ("0 0 0\n", "LUT_1D_SIZE"),
            ("LUT_3D_SIZE x\n", "LUT_3D_SIZE"),
            ("LUT_1D_SIZE 2\n0 0\n1 1\n", "three"),
            ("LUT_1D_SIZE 2\n0 0 nan\n1 1 1\n", "not a number"),
            ("LUT_1D_SIZE 1\n0 0 0\n", "size 1"),
            ("LUT_1D_SIZE 2\nDOMAIN_MIN 0 0\n0 0 0\n1 1 1\n", "three values"),
            ("LUT_3D_SIZE 2\n0 0 0 junk\n", "unreadable"),
        ] {
            let e = format!("{:#}", parse_cube(t).unwrap_err());
            assert!(e.contains(want), "{t:?}: {e}");
        }
        assert!(Lut3d::parse_cube("LUT_1D_SIZE 2\n0 0 0\n1 1 1\n").is_err());
    }

    #[test]
    fn a_3dl_reads_blue_fastest_and_infers_depth() {
        // The swap cube again, blue fastest, 10-bit, with a mesh line.
        let mut t = String::from("# Lustre\n0 1023\n");
        for r in 0..2 {
            for g in 0..2 {
                for b in 0..2 {
                    t += &format!("{} {} {}\n", b * 1023, g * 1023, r * 1023);
                }
            }
        }
        let l = parse_3dl(&t).unwrap();
        assert!(close(l.apply([0.2, 0.5, 0.9]), [0.9, 0.5, 0.2]));
        // A Mesh header names the depth: 12 bit makes 4095 white.
        let mut m = String::from("3DMESH\nMesh 1 12\n0 4095\n");
        for k in 0..8 {
            let v = if k == 7 { 4095 } else { 0 };
            m += &format!("{v} {v} {v}\n");
        }
        let l = parse_3dl(&m).unwrap();
        assert!(close(l.apply([1.0, 1.0, 1.0]), [1.0; 3]));
        // Without a mesh line.
        let bare: String = t.lines().skip(2).map(|l| format!("{l}\n")).collect();
        assert!(close(parse_3dl(&bare).unwrap().apply([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0]));
    }

    #[test]
    fn bad_3dls_say_why() {
        for (t, want) in [
            ("0 0 0\n0 0 0\n", "not a cube"),
            ("0 341 682 1023\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n", "mesh line"),
            ("0 0 0 0\n0 0\n", "three values"),
            ("x y z\n", "unreadable"),
            ("0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 0\n0 0 99999999\n", "16 bit"),
        ] {
            let e = format!("{:#}", parse_3dl(t).unwrap_err());
            assert!(e.contains(want), "{t:?}: {e}");
        }
    }

    #[test]
    fn spi1d_reads_one_and_three_components() {
        let t = "Version 1\nFrom 0 2\nLength 3\nComponents 1\n{\n0\n0.5\n1\n}\n";
        let l = parse_spi1d(t).unwrap();
        assert!(close(l.apply([1.0, 2.0, 0.5]), [0.5, 1.0, 0.25]));
        let t3 = "Version 1\nFrom 0.0 1.0\nLength 2\nComponents 3\n{\n 1 0 0\n 0 1 1\n}\n";
        assert!(close(parse_spi1d(t3).unwrap().apply([0.5; 3]), [0.5; 3]));
        for (t, want) in [
            ("Version 1\nLength 3\nComponents 1\n{\n0\n1\n}\n", "Length 3"),
            ("Version 1\nLength 2\nComponents 2\n", "components"),
            ("Version 1\nLength 2\nComponents 3\n{\n0\n1\n}\n", "3 value"),
            ("Bogus\n", "unreadable"),
        ] {
            let e = format!("{:#}", parse_spi1d(t).unwrap_err());
            assert!(e.contains(want), "{t:?}: {e}");
        }
    }

    #[test]
    fn spi3d_reads_indexed_entries_in_any_order() {
        let mut t = String::from("SPILUT 1.0\n3 3\n2 2 2\n");
        // Backwards, to prove the indices place them.
        for k in (0..8).rev() {
            let (i, j, kk) = (k % 2, (k / 2) % 2, k / 4);
            t += &format!("{i} {j} {kk} {kk} {j} {i}\n");
        }
        let l = parse_spi3d(&t).unwrap();
        assert!(close(l.apply([0.2, 0.5, 0.9]), [0.9, 0.5, 0.2]));
        for (t, want) in [
            ("nope\n", "SPILUT"),
            ("SPILUT 1.0\n3 3\n2 3 2\n", "equal sides"),
            ("SPILUT 1.0\n3 3\n2 2 2\n0 0 0 0 0 0\n", "missing"),
            ("SPILUT 1.0\n3 3\n2 2 2\n0 0 5 0 0 0\n", "out of the cube"),
            ("SPILUT 1.0\n3 3\n2 2 2\n0 0 0 0 0\n", "i j k"),
        ] {
            let e = format!("{:#}", parse_spi3d(t).unwrap_err());
            assert!(e.contains(want), "{t:?}: {e}");
        }
    }

    #[test]
    fn parse_picks_by_extension_and_hashes_the_bytes() {
        let a = parse("CUBE", SWAP.as_bytes()).unwrap();
        let b = parse("cube", format!("{SWAP}\n").as_bytes()).unwrap();
        assert_eq!(a.format, "cube");
        assert_ne!(a.hash, b.hash, "different bytes, different key");
        assert!(format!("{:#}", parse("csp", b"x").unwrap_err()).contains(".csp"));
        assert!(parse("cube", &[0xff, 0xfe]).is_err());
    }

    /// Resolve's shaper + cube layout and a Cinespace file with a
    /// non-uniform prelut, each through both interpolations, against
    /// OpenColorIO 2.6's CPU processor on the same files (FileTransform,
    /// INTERP_TETRAHEDRAL / INTERP_LINEAR). Inputs cover grey, both clamps
    /// and off-diagonal colours.
    #[test]
    fn resolve_cube_and_csp_match_opencolorio() {
        const RESOLVE: &str = "# Created by: DaVinci Resolve\nLUT_1D_SIZE 6\nLUT_1D_INPUT_RANGE -0.1 2.0\nLUT_3D_SIZE 3\nLUT_3D_INPUT_RANGE 0.0 1.2\n\n0.000000 0.000000 0.000000\n0.536656 0.491935 0.447214\n0.758947 0.695701 0.632456\n0.929516 0.852056 0.774597\n1.073313 0.983870 0.894427\n1.200000 1.100000 1.000000\n0.000000 0.050000 0.000000\n0.000000 0.050000 0.000000\n0.000000 0.050000 0.000000\n0.000000 0.050000 0.150000\n0.250000 0.050000 0.150000\n0.500000 0.050000 0.150000\n0.000000 0.050000 0.300000\n0.500000 0.050000 0.300000\n1.000000 0.050000 0.300000\n0.050000 0.050000 0.000000\n0.050000 0.050000 0.150000\n0.050000 0.050000 0.300000\n0.050000 0.175000 0.150000\n0.300000 0.175000 0.300000\n0.550000 0.175000 0.450000\n0.050000 0.300000 0.300000\n0.550000 0.300000 0.450000\n1.050000 0.300000 0.600000\n0.100000 0.050000 0.000000\n0.100000 0.050000 0.300000\n0.100000 0.050000 0.600000\n0.100000 0.550000 0.150000\n0.350000 0.550000 0.450000\n0.600000 0.550000 0.750000\n0.100000 1.050000 0.300000\n0.600000 1.050000 0.600000\n1.100000 1.050000 0.900000\n";
        const CSP: &str = "CSPLUTV100\n3D\n\nBEGIN METADATA\nref\nEND METADATA\n\n4\n0.0 0.1 0.4 1.0\n0.0 0.3 0.7 1.0\n4\n0.0 0.1 0.4 1.0\n0.0 0.3 0.7 1.0\n4\n0.0 0.1 0.4 1.0\n0.0 0.3 0.7 1.0\n\n3 3 3\n0.000000 0.050000 0.000000\n0.000000 0.050000 0.000000\n0.000000 0.050000 0.000000\n0.000000 0.050000 0.150000\n0.250000 0.050000 0.150000\n0.500000 0.050000 0.150000\n0.000000 0.050000 0.300000\n0.500000 0.050000 0.300000\n1.000000 0.050000 0.300000\n0.050000 0.050000 0.000000\n0.050000 0.050000 0.150000\n0.050000 0.050000 0.300000\n0.050000 0.175000 0.150000\n0.300000 0.175000 0.300000\n0.550000 0.175000 0.450000\n0.050000 0.300000 0.300000\n0.550000 0.300000 0.450000\n1.050000 0.300000 0.600000\n0.100000 0.050000 0.000000\n0.100000 0.050000 0.300000\n0.100000 0.050000 0.600000\n0.100000 0.550000 0.150000\n0.350000 0.550000 0.450000\n0.600000 0.550000 0.750000\n0.100000 1.050000 0.300000\n0.600000 1.050000 0.600000\n1.100000 1.050000 0.900000\n";
        const INS: [[f32; 3]; 8] = [[0.0, 0.0, 0.0], [0.18, 0.18, 0.18], [0.5, 0.25, 0.75], [1.0, 1.0, 1.0], [0.9, 0.05, 0.4], [-0.3, 1.5, 2.5], [0.33, 0.71, 0.12], [0.04, 0.6, 0.95]];
        let cases: [(&str, &str, Interp, [[f32; 3]; 8]); 4] = [
            ("cube", RESOLVE, Interp::Tetrahedral, [[0.0576764, 0.0721832, 0.0559017], [0.161494, 0.112113, 0.156525], [0.237099, 0.157806, 0.277408], [0.580671, 0.365663, 0.474481], [0.186617, 0.0866023, 0.220529], [0.0833333, 0.723969, 0.239691], [0.279144, 0.115709, 0.22885], [0.144737, 0.262853, 0.227548]]),
            ("cube", RESOLVE, Interp::Trilinear, [[0.0192663, 0.0543304, 0.0349508], [0.106327, 0.0839506, 0.126434], [0.232886, 0.15071, 0.269904], [0.535828, 0.314534, 0.457646], [0.140733, 0.0794342, 0.209566], [0.0833333, 0.649227, 0.239691], [0.275873, 0.105403, 0.223184], [0.136614, 0.248487, 0.209382]]),
            ("csp", CSP, Interp::Tetrahedral, [[0.0, 0.05, 0.0], [0.25997, 0.158321, 0.25997], [0.514583, 0.52309, 0.599479], [1.1, 1.05, 0.9], [0.25125, 0.23125, 0.466319], [0.1, 1.05, 0.3], [0.616514, 0.217309, 0.409794], [0.248773, 0.877836, 0.341111]]),
            ("csp", CSP, Interp::Trilinear, [[0.0, 0.05, 0.0], [0.231063, 0.143868, 0.242626], [0.506313, 0.51847, 0.585632], [1.1, 1.05, 0.9], [0.246383, 0.149687, 0.463097], [0.1, 1.05, 0.3], [0.602585, 0.200012, 0.396424], [0.224329, 0.871633, 0.340007]]),
        ];
        for (ext, text, how, want) in cases {
            let lut = parse(ext, text.as_bytes()).unwrap();
            for (c, w) in INS.iter().zip(want) {
                let got = lut.apply_with(*c, how);
                assert!(
                    got.iter().zip(w).all(|(a, b)| (a - b).abs() < 1e-4),
                    "{ext} {how:?}: {c:?} -> {got:?}, OCIO says {w:?}"
                );
            }
        }
    }

    #[test]
    fn a_byte_order_mark_is_not_a_bad_line() {
        let l = parse("cube", "\u{feff}TITLE \"x\"\nLUT_1D_SIZE 2\n0 0 0\n1 1 1\n".as_bytes()).unwrap();
        assert!(l.shaper.is_some());
    }

    #[test]
    fn a_csp_1d_composes_its_prelut_and_an_identity_prelut_is_dropped() {
        // The prelut's three points lie on x², which its cubic draws
        // exactly; the table halves. OCIO 2.6 gives the same numbers.
        let t = "CSPLUTV100\n1D\n3\n0 0.5 1\n0 0.25 1\n3\n0 0.5 1\n0 0.25 1\n3\n0 0.5 1\n0 0.25 1\n\n2\n0 0 0\n0.5 0.5 0.5\n";
        let l = parse("csp", t.as_bytes()).unwrap();
        assert!(l.cube.is_none());
        let o = l.apply([0.5, 0.75, 1.0]);
        assert!(close(o, [0.125, 0.28125, 0.5]), "{o:?}");
        let o = l.apply([0.3, 0.1, 0.9]);
        assert!(close(o, [0.045, 0.005, 0.405]), "{o:?}");
        let id = "CSPLUTV100\n3D\n2\n0 1\n0 1\n2\n0 1\n0 1\n2\n0 1\n0 1\n2 2 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n";
        let l = parse("csp", id.as_bytes()).unwrap();
        assert!(l.shaper.is_none() && l.cube.is_some());
        assert!(close(l.apply([0.2, 0.4, 0.6]), [0.2, 0.4, 0.6]));
    }

    #[test]
    fn bad_csps_say_why() {
        for (t, why) in [
            ("CSPLUTV100\n2D\n", "1D or 3D"),
            ("LUT_3D_SIZE 2\n", "CSPLUTV100"),
            ("CSPLUTV100\n3D\n2\n1 0\n0 1\n", "do not increase"),
            ("CSPLUTV100\n1D\n2\n0 1\n0 1\n2\n0 1\n0 1\n2\n0 1\n0 1\n2\n0 0 0\n", "ends before"),
            ("CSPLUTV100\n3D\n2\n0 1\n0 1\n2\n0 1\n0 1\n2\n0 1\n0 1\n2 3 2\n", "equal sides"),
        ] {
            let e = format!("{:#}", parse("csp", t.as_bytes()).unwrap_err());
            assert!(e.contains(why), "{t:?}: {e}");
        }
    }

    /// Published RGB->XYZ for DaVinci Wide Gamut (Blackmagic's note, as
    /// colour-science carries it): the chromaticities above must solve to it.
    #[test]
    fn davinci_wide_gamut_solves_to_its_published_matrix() {
        let m = crate::color::Primaries::DaVinciWideGamut.to_xyz();
        let want = [
            [0.70062239, 0.14877482, 0.10105872],
            [0.27411851, 0.87363190, -0.14775041],
            [-0.09896291, -0.13789533, 1.32591599],
        ];
        for r in 0..3 {
            for c in 0..3 {
                assert!((m[r][c] - want[r][c]).abs() < 1e-4, "{m:?}");
            }
        }
    }

    #[test]
    fn load_reads_rereads_and_reports() {
        let dir = std::env::temp_dir().join(format!("wobble-lut-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("swap.cube");
        std::fs::write(&p, SWAP).unwrap();
        let s = p.to_str().unwrap();
        let a = load(s).unwrap();
        assert!(Arc::ptr_eq(&a, &load(s).unwrap()), "a second ask is the cache");
        assert_eq!(load("").unwrap_err(), "");
        assert!(!load(dir.join("missing.cube").to_str().unwrap()).unwrap_err().is_empty());
        // An edit (a new size) is picked up once the recheck window passes.
        std::fs::write(&p, "LUT_1D_SIZE 2\n0 0 0\n1 1 1\n").unwrap();
        std::thread::sleep(RECHECK_LOCAL + std::time::Duration::from_millis(50));
        let b = load(s).unwrap();
        assert!(b.cube.is_none() && b.shaper.is_some());
        std::fs::remove_dir_all(&dir).ok();
    }
}
