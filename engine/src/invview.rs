//! Inverse view transform: the viewer's display rendering run BACKWARDS,
//! as a 3D table, so a display-referred picture (an AI-generated clip, a
//! graded Rec.709 deliverable) comes into the comp as scene-linear values a
//! blur, a glow or a grade can treat as light.
//!
//! The forward side is [`ViewSettings::display_rgb`], the CPU mirror of
//! what the viewer draws, so the promise is exact: an inverted plate shown
//! through the SAME view transform reads back the codes it came from (to
//! the table's interpolation). What the inversion cannot do is invent the
//! highlights the display clipped: a code of 1.0 lands at the transform's
//! own white point (~16 for ACES 1.3 SDR, ~1000 for AgX), and the shape
//! above it is the Highlight Rebuild / HDRCNN / LTX HDR nodes' job.
//!
//! The table is solved per lattice point (Levenberg–Marquardt on the
//! forward transform, warm-started from the neutral axis's 1D inverse) and
//! stored as ACEScct: the inverse is steep near white, and a log encoding
//! interpolates there where a linear one would ring. The LUT node decodes
//! it (`LutOut::Space(ACESCCT)`) into the working space, so the table does
//! not depend on the comp. Built once per pick and cached for the process.

use crate::color::Curve;
use crate::lut::{Lut, Lut3d};
use crate::viewtf::{Eotf, Gamut, Look, ViewSettings, ViewTransform};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Lattice points per side. 33 lands the round trip inside an 8-bit code
/// value everywhere the forward transform can reach (the tests measure it).
pub const SIZE: usize = 33;

/// Bumped when the solve changes, so a render cached from an older table
/// is a different key.
const GEN: u64 = 1;

/// Which display rendering to undo: the viewer's own pick, by value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct InverseView {
    pub transform: ViewTransform,
    pub display: Gamut,
    pub eotf: Eotf,
    pub look: Look,
}

impl Default for InverseView {
    /// The viewer's default: ACES 1.3, Rec.709 primaries, sRGB curve.
    fn default() -> Self {
        Self { transform: ViewTransform::Aces13, display: Gamut::Rec709, eotf: Eotf::Srgb, look: Look::None }
    }
}

impl InverseView {
    /// The transforms that can be inverted here (OpenDRT has no CPU mirror).
    pub const TRANSFORMS: [ViewTransform; 5] = [
        ViewTransform::Aces13,
        ViewTransform::Aces12,
        ViewTransform::AgX,
        ViewTransform::Filmic,
        ViewTransform::None,
    ];

    fn key(&self) -> u64 {
        (((GEN * 16 + self.transform as u64) * 16 + self.display as u64) * 16 + self.eotf as u64) * 32
            + self.look as u64
    }

    /// The forward transform: ACEScg scene-linear in, display codes out.
    fn settings(&self) -> ViewSettings {
        ViewSettings {
            transform: self.transform,
            working: Gamut::AcesCg,
            display: self.display,
            eotf: self.eotf,
            look: self.look,
            ..Default::default()
        }
    }

    /// Why this pick cannot be inverted, if it cannot.
    pub fn check(&self) -> Result<(), String> {
        match self.transform {
            ViewTransform::OpenDrt => Err("OpenDRT cannot be inverted (no CPU mirror)".into()),
            ViewTransform::AgX if !crate::viewtf::agx::available() => {
                Err("AgX needs Blender installed (its cubes are read from there)".into())
            }
            _ => Ok(()),
        }
    }

    /// The table, built on first ask and cached.
    pub fn table(&self) -> Result<Arc<Lut>, String> {
        static CACHE: OnceLock<Mutex<HashMap<u64, Result<Arc<Lut>, String>>>> = OnceLock::new();
        let cache = CACHE.get_or_init(Default::default);
        let key = self.key();
        if let Some(got) = cache.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return got.clone();
        }
        let got = self.check().and_then(|_| build(self, SIZE)).map(Arc::new);
        cache.lock().unwrap_or_else(|e| e.into_inner()).insert(key, got.clone());
        got
    }
}

const CCT: Curve = Curve::AcesCct;

/// The forward transform over ACEScct values.
struct Fwd {
    s: ViewSettings,
}

impl Fwd {
    fn at(&self, u: [f64; 3]) -> Option<[f64; 3]> {
        let x = u.map(|v| CCT.to_linear(v as f32));
        self.s.display_rgb(x).map(|d| d.map(|v| v as f64))
    }
}

/// Neutral-axis inverse: ACEScct value whose grey shows as code `d`.
struct Neutral {
    u: Vec<f64>,
    d: Vec<f64>,
}

impl Neutral {
    fn new(f: &Fwd, lo: f64, hi: f64) -> Option<Self> {
        let n = 2048;
        let mut u = Vec::with_capacity(n);
        let mut d = Vec::with_capacity(n);
        let mut best = f64::MIN;
        for i in 0..n {
            let v = lo + (hi - lo) * i as f64 / (n - 1) as f64;
            let o = f.at([v; 3])?;
            // Monotone by construction: a flat or dipping tail is cut where
            // the display stops rising.
            let m = (o[0] + o[1] + o[2]) / 3.0;
            if m > best + 1e-9 {
                best = m;
                u.push(v);
                d.push(m);
            }
        }
        Some(Self { u, d })
    }

    fn inv(&self, d: f64) -> f64 {
        let i = self.d.partition_point(|&x| x < d);
        if i == 0 {
            return self.u[0];
        }
        if i >= self.d.len() {
            return *self.u.last().unwrap();
        }
        let (d0, d1) = (self.d[i - 1], self.d[i]);
        let t = (d - d0) / (d1 - d0).max(1e-12);
        self.u[i - 1] + t * (self.u[i] - self.u[i - 1])
    }

    /// The ACEScct value where the grey reaches its brightest code.
    fn top(&self) -> f64 {
        *self.u.last().unwrap()
    }
}

fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-30 {
        return None;
    }
    let col = |k: usize| {
        let mut m = a;
        for r in 0..3 {
            m[r][k] = b[r];
        }
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    Some([col(0) / det, col(1) / det, col(2) / det])
}

fn err2(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).powi(2)).sum()
}

/// Bounded Levenberg–Marquardt: the ACEScct triple showing as `d`, or the
/// nearest the transform can show.
fn invert(f: &Fwd, d: [f64; 3], start: [f64; 3], lo: f64, hi: f64) -> [f64; 3] {
    let clamp = |u: [f64; 3]| u.map(|v| v.clamp(lo, hi));
    let mut u = clamp(start);
    let Some(mut o) = f.at(u) else { return u };
    let mut e = err2(o, d);
    let mut lambda = 1e-3;
    const H: f64 = 1e-4;
    for _ in 0..40 {
        if e < 1e-14 {
            break;
        }
        // Forward differences, stepping inward at the upper bound.
        let mut j = [[0.0f64; 3]; 3];
        for k in 0..3 {
            let mut v = u;
            let h = if v[k] + H > hi { -H } else { H };
            v[k] += h;
            let Some(ov) = f.at(v) else { return u };
            for r in 0..3 {
                j[r][k] = (ov[r] - o[r]) / h;
            }
        }
        let r = [o[0] - d[0], o[1] - d[1], o[2] - d[2]];
        let mut jtj = [[0.0f64; 3]; 3];
        let mut jtr = [0.0f64; 3];
        for a in 0..3 {
            for b in 0..3 {
                jtj[a][b] = (0..3).map(|i| j[i][a] * j[i][b]).sum();
            }
            jtr[a] = -(0..3).map(|i| j[i][a] * r[i]).sum::<f64>();
        }
        let mut moved = false;
        for _ in 0..8 {
            let mut m = jtj;
            for a in 0..3 {
                m[a][a] += lambda * (jtj[a][a] + 1e-9);
            }
            let Some(step) = solve3(m, jtr) else { break };
            let nu = clamp([u[0] + step[0], u[1] + step[1], u[2] + step[2]]);
            let Some(no) = f.at(nu) else { break };
            let ne = err2(no, d);
            if ne < e {
                let small = (0..3).all(|k| (nu[k] - u[k]).abs() < 1e-9);
                u = nu;
                o = no;
                e = ne;
                lambda = (lambda * 0.3).max(1e-9);
                moved = !small;
                break;
            }
            lambda *= 4.0;
        }
        if !moved {
            break;
        }
    }
    u
}

/// Solve the table at `n` points a side.
pub fn build(v: &InverseView, n: usize) -> Result<Lut, String> {
    let f = Fwd { s: v.settings() };
    let lo = CCT.from_linear(0.0) as f64;
    // Far past any display's white: 2^16 is half-float's ceiling.
    let ceiling = CCT.from_linear(65504.0) as f64;
    let neutral = Neutral::new(&f, lo, ceiling).ok_or("the view transform cannot be evaluated here")?;
    // A coloured highlight needs channels past the grey's white (the
    // rendering desaturates there); two stops of room, no more, or a code
    // the display cannot show runs off to the ceiling.
    let hi = (neutral.top() + 2.0 / 17.52).min(ceiling);
    let last = (n - 1) as f64;
    // The warm start: the grey's inverse per channel IN THE DISPLAY'S
    // PRIMARIES, carried to ACEScg. That is exact for a per-channel curve
    // after a matrix (Filmic, None) and close for ACES. Starting per channel
    // in ACEScct instead lands saturated codes where a display channel is
    // below zero, which every transform clamps: the gradient there is zero
    // and the solve cannot leave (half the cube failed that way).
    let to_work = crate::color::primaries_matrix(v.display.primaries(), crate::color::Primaries::Ap1);
    let data: Vec<[f32; 3]> = (0..n * n * n)
        .into_par_iter()
        .map(|i| {
            let d = [(i % n) as f64 / last, ((i / n) % n) as f64 / last, (i / (n * n)) as f64 / last];
            let lin = d.map(|c| CCT.to_linear(neutral.inv(c) as f32));
            let start = crate::color::mat3_apply(to_work, lin).map(|x| CCT.from_linear(x) as f64);
            let direct = invert(&f, d, start, lo, hi);
            let miss = |u: [f64; 3]| f.at(u).map_or(f64::MAX, |o| err2(o, d));
            if miss(direct) < 1e-10 {
                return direct.map(|u| u as f32);
            }
            // Missed: the target is out of reach, or the start sat in a
            // clamped (flat) region and the solve fell into a local minimum
            // — pure green through ACES landed on BLACK. Walk the target out
            // from its own grey instead, warm-starting each step, so the
            // answer stays on the branch that is continuous with neutral
            // and an unreachable corner gets the nearest colour it can show.
            let g = (d[0] + d[1] + d[2]) / 3.0;
            let mut u = [neutral.inv(g); 3];
            const STEPS: usize = 8;
            for k in 1..=STEPS {
                let t = k as f64 / STEPS as f64;
                let dt = d.map(|c| g + t * (c - g));
                u = invert(&f, dt, u, lo, hi);
            }
            if miss(u) < miss(direct) { u } else { direct }.map(|u| u as f32)
        })
        .collect();
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (v.key(), n).hash(&mut h);
    Ok(Lut { shaper: None, cube: Some(Lut3d::new(n, data)), hash: h.finish(), format: "inverse view" })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Display codes in, scene-linear ACEScg out (the table, decoded), then
    /// the forward transform: the codes again.
    fn round_trip(v: InverseView) -> (f32, f32) {
        let lut = v.table().unwrap();
        let s = v.settings();
        let (mut worst, mut sum, mut count) = (0.0f32, 0.0f32, 0);
        // Off the lattice: a prime stride through the cube, inside what
        // the display can show (an in-gamut grey ramp plus colours that are
        // not at the gamut's edge).
        for i in 0..4000u32 {
            let t = |k: u32| ((i * k) % 997) as f32 / 996.0;
            let grey = t(1);
            let c = [0.1 + 0.8 * t(7), 0.1 + 0.8 * t(13), 0.1 + 0.8 * t(29)];
            let c = [c[0] * 0.3 + grey * 0.7, c[1] * 0.3 + grey * 0.7, c[2] * 0.3 + grey * 0.7];
            let x = lut.apply(c).map(|u| CCT.to_linear(u));
            let back = s.display_rgb(x).unwrap();
            let e = (0..3).map(|k| (back[k] - c[k]).abs()).fold(0.0, f32::max);
            worst = worst.max(e);
            sum += e;
            count += 1;
        }
        (worst, sum / count as f32)
    }

    #[test]
    fn aces_round_trips_inside_a_code_value() {
        let t = std::time::Instant::now();
        let (worst, mean) = round_trip(InverseView::default());
        eprintln!("ACES 1.3 sRGB: worst {worst:.5}, mean {mean:.6}, built in {:?}", t.elapsed());
        assert!(worst < 1.0 / 255.0, "worst {worst}");
        assert!(mean < 0.1 / 255.0, "mean {mean}");
    }

    #[test]
    fn filmic_and_rec1886_round_trip_too() {
        let v = InverseView { transform: ViewTransform::Filmic, eotf: Eotf::Rec1886, ..Default::default() };
        let (worst, mean) = round_trip(v);
        eprintln!("Narkowicz 2.4: worst {worst:.5}, mean {mean:.6}");
        assert!(worst < 1.0 / 255.0, "worst {worst}");
    }

    /// White comes back as light: well above 1, and greys stay grey.
    #[test]
    fn white_is_brighter_than_one_and_grey_stays_grey() {
        let lut = InverseView::default().table().unwrap();
        let w = lut.apply([1.0; 3]).map(|u| CCT.to_linear(u));
        assert!(w[1] > 8.0, "white {w:?}");
        let g = lut.apply([0.46; 3]).map(|u| CCT.to_linear(u));
        assert!((g[0] - g[2]).abs() / g[1] < 0.01, "grey {g:?}");
        // sRGB 0.46 through ACES 1.3 is about mid grey.
        assert!(g[1] > 0.1 && g[1] < 0.3, "mid {g:?}");
        let k = lut.apply([0.0; 3]).map(|u| CCT.to_linear(u));
        assert!(k.iter().all(|v| v.abs() < 1e-3), "black {k:?}");
    }

    #[test]
    fn opendrt_says_why() {
        let v = InverseView { transform: ViewTransform::OpenDrt, ..Default::default() };
        assert!(v.table().is_err());
    }
}

#[cfg(test)]
mod probe {
    use super::*;
    #[test]
    #[ignore]
    fn lattice_residuals() {
        for v in [InverseView::default(), InverseView { transform: ViewTransform::Filmic, eotf: Eotf::Rec1886, ..Default::default() }] {
            let lut = v.table().unwrap();
            let s = v.settings();
            let n = SIZE;
            let cube = lut.cube.as_ref().unwrap();
            let mut bad = Vec::new();
            for i in 0..n * n * n {
                let d = [(i % n) as f32 / (n - 1) as f32, ((i / n) % n) as f32 / (n - 1) as f32, (i / (n * n)) as f32 / (n - 1) as f32];
                let u = cube.data[i];
                let back = s.display_rgb(u.map(|x| CCT.to_linear(x))).unwrap();
                let e = (0..3).map(|k| (back[k] - d[k]).abs()).fold(0.0, f32::max);
                if e > 1.0 / 255.0 { bad.push((e, d, u, back)); }
            }
            bad.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            eprintln!("{:?}: {} bad of {}", v.transform, bad.len(), n * n * n);
            for b in bad.iter().take(8) { eprintln!("  {:?}", b); }
        }
    }
}
