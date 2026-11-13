//! spektra-lut: bake spektrafilm's film -> print -> scan chain into a 3D LUT
//! and write it out as a wobble Shader node.
//!
//! spektrafilm's colour chain is per pixel once its spatial and stochastic
//! effects (halation, diffusion, DIR-coupler diffusion, grain, glare) are off
//! — upstream's own `lut_mode` is exactly that — so a LUT reproduces it. The
//! LUT's input is the comp's working space, log2-shaped per channel over
//! [LO, HI] stops; its output is the scan, linear, in the same working space.
//! Camera exposure is NOT baked: it is a uniform shift of the shaper, so the
//! node's exposure slider never re-bakes.
//!
//!   spektra-lut list  --data <spektrafilm data dir>
//!   spektra-lut bake  --data <dir> --recipe <recipe.json> --out <shader.wgsl>
//!   spektra-lut check --data <dir> --recipe <recipe.json> [--samples N]
//!   spektra-lut eval  --data <dir> --recipe <recipe.json> --rgb r,g,b[;r,g,b]
//!
//! A recipe: {"film", "print", "space", "size", "scan_film", "params"} where
//! `params` is a partial spektrafilm RuntimeParams JSON layered on the
//! film's stock defaults (spektrorganize's preset shape).

use serde_json::{Value, json};
use spektrafilm_core::params::RuntimeParams;
use spektrafilm_core::pipeline::Pipeline;
use spektrafilm_core::profile;
use spektrafilm_gpu::ComputeBackend;
use spektrafilm_math::image::ImageBuf;
use spektrafilm_math::precision::{from_f64, to_f64};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The shaper's range, in stops of the working-space value (0 = 1.0, mid grey
/// is -2.47). Below LO is film-base black; above HI every stock has long
/// since shouldered on the print.
const LO: f64 = -12.0;
const HI: f64 = 6.0;
/// Version of the shader this writes. Part of the recipe line, so a node
/// baked by an older baker can be told apart.
const BAKER: u32 = 1;

struct Args {
    cmd: String,
    data: Option<PathBuf>,
    recipe: Option<PathBuf>,
    out: Option<PathBuf>,
    samples: usize,
    rgb: Option<String>,
}

fn args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let cmd = it.next().ok_or("usage: spektra-lut list|bake|check --data DIR [--recipe R] [--out F]")?;
    let mut a = Args { cmd, data: None, recipe: None, out: None, samples: 4000, rgb: None };
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or(format!("{k} needs a value"));
        match k.as_str() {
            "--data" => a.data = Some(v()?.into()),
            "--recipe" => a.recipe = Some(v()?.into()),
            "--out" => a.out = Some(v()?.into()),
            "--rgb" => a.rgb = Some(v()?),
            "--samples" => a.samples = v()?.parse().map_err(|e| format!("--samples: {e}"))?,
            _ => return Err(format!("unknown argument {k}")),
        }
    }
    Ok(a)
}

fn main() {
    let r = args().and_then(|a| run(&a));
    match r {
        Ok(v) => println!("{v}"),
        Err(e) => {
            println!("{}", json!({ "ok": false, "error": e }));
            std::process::exit(1);
        }
    }
}

fn data_dir(a: &Args) -> Result<PathBuf, String> {
    let d = a
        .data
        .clone()
        .or_else(|| std::env::var_os("SPEKTRAFILM_DATA").map(PathBuf::from))
        .ok_or("no spektrafilm data dir (--data or $SPEKTRAFILM_DATA)")?;
    if !d.join("profiles").is_dir() {
        return Err(format!("{} has no profiles/ — not a spektrafilm data dir", d.display()));
    }
    Ok(d)
}

fn run(a: &Args) -> Result<Value, String> {
    let data = data_dir(a)?;
    match a.cmd.as_str() {
        "list" => list(&data),
        "bake" => {
            let recipe = read_recipe(a)?;
            let out = a.out.as_ref().ok_or("bake needs --out")?;
            let t = Instant::now();
            let lut = Lut::bake(&recipe, &data)?;
            let src = lut.shader(&recipe);
            std::fs::write(out, &src).map_err(|e| format!("writing {}: {e}", out.display()))?;
            Ok(json!({ "ok": true, "ms": t.elapsed().as_millis() as u64, "bytes": src.len(), "size": lut.n }))
        }
        "eval" => {
            // `--rgb r,g,b[;r,g,b...]`: the chain and the LUT at those colours.
            let recipe = read_recipe(a)?;
            let rgb: Vec<[f64; 3]> = a
                .rgb
                .as_deref()
                .ok_or("eval needs --rgb")?
                .split(';')
                .map(|t| {
                    let v: Vec<f64> = t.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                    if v.len() == 3 { Ok([v[0], v[1], v[2]]) } else { Err(format!("bad colour {t:?}")) }
                })
                .collect::<Result<_, _>>()?;
            let lut = Lut::bake(&recipe, &data)?.halved();
            let pipe = pipeline(&recipe, &data)?;
            let truth = process(&pipe, &rgb);
            let via: Vec<[f64; 3]> = rgb.iter().map(|c| lut.eval(*c)).collect();
            Ok(json!({ "ok": true, "chain": truth, "lut": via }))
        }
        "check" => {
            let recipe = read_recipe(a)?;
            check(&recipe, &data, a.samples)
        }
        c => Err(format!("unknown command {c}")),
    }
}

fn list(data: &Path) -> Result<Value, String> {
    let mut films = Vec::new();
    let mut papers = Vec::new();
    let rd = std::fs::read_dir(data.join("profiles")).map_err(|e| e.to_string())?;
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let name = p.file_stem().unwrap().to_string_lossy().to_string();
        match profile::load_profile(&p) {
            Ok(pr) if pr.is_paper() => papers.push(json!({ "id": name, "label": pr.info.name.clone() })),
            Ok(pr) => films.push(json!({
                "id": name,
                "label": pr.info.name.clone(),
                "print": pr.info.target_print.clone(),
            })),
            Err(_) => {}
        }
    }
    let by_id = |v: &mut Vec<Value>| v.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    by_id(&mut films);
    by_id(&mut papers);
    Ok(json!({ "ok": true, "films": films, "papers": papers }))
}

fn read_recipe(a: &Args) -> Result<Value, String> {
    let p = a.recipe.as_ref().ok_or("needs --recipe")?;
    let t = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
    serde_json::from_str(&t).map_err(|e| format!("recipe: {e}"))
}

/// A spektrafilm pipeline set up as a pure per-pixel transform from the
/// recipe's working space to itself.
fn pipeline(recipe: &Value, data: &Path) -> Result<Pipeline, String> {
    let film_id = recipe["film"].as_str().ok_or("recipe has no film")?;
    let load = |n: &str| profile::load_profile_by_name(data, n).map_err(|e| format!("profile '{n}': {e}"));
    let film = load(film_id)?;
    let empty = json!({});
    let overrides = recipe.get("params").unwrap_or(&empty);
    let mut p: RuntimeParams =
        spektrafilm_core::stock_presets::stock_defaults_with_overrides(&film, overrides, data)?;
    // A stock with no paired print (a slide, a print film) is scanned as it
    // is unless the recipe names a paper.
    let paper = recipe["print"].as_str().filter(|s| !s.is_empty());
    let scan_film = recipe["scan_film"]
        .as_bool()
        .unwrap_or(paper.is_none() && film.info.target_print.is_none());
    p.set_scan_film(scan_film);
    let print_id = match paper {
        Some(s) => s.to_string(),
        None => film.info.target_print.clone().unwrap_or_else(|| film_id.to_string()),
    };
    let print = if scan_film { film.clone() } else { load(&print_id)? };
    let space = recipe["space"].as_str().unwrap_or("ACEScg");
    p.io.input_color_space = space.into();
    p.io.input_cctf_decoding = false;
    p.io.output_color_space = space.into();
    p.io.output_cctf_encoding = false;
    p.io.output_gamut_compress.algorithm = "off".into();
    p.io.upscale_factor = 1.0;
    p.io.crop = false;
    p.scanner.white_correction = false;
    p.scanner.black_correction = false;
    // Exposure is the node's uniform, never baked; autoexposure meters a
    // frame, which a LUT has none of.
    p.camera.auto_exposure = false;
    p.camera.exposure_compensation_ev = 0.0;
    p.debug.deactivate_spatial_effects = true;
    p.debug.deactivate_stochastic_effects = true;
    Pipeline::new_with_spectral(film, print, p, data).map_err(|e| format!("building pipeline: {e}"))
}

/// Run linear RGB triples through the chain.
fn process(pipe: &Pipeline, rgb: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let backend = spektrafilm_gpu::cpu_backend::CpuBackend;
    let backend: &dyn ComputeBackend = &backend;
    // Any shape will do: nothing in the chain is spatial any more. A near-
    // square keeps every stage's row loops reasonable.
    let n = rgb.len();
    let w = (n as f64).sqrt().ceil() as usize;
    let h = n.div_ceil(w);
    let mut data = Vec::with_capacity(w * h * 3);
    for i in 0..w * h {
        let c = rgb.get(i).copied().unwrap_or([0.18; 3]);
        data.extend(c.iter().map(|v| from_f64(*v)));
    }
    let out = pipe.process(ImageBuf::from_data(w as u32, h as u32, data), backend);
    (0..n).map(|i| [0, 1, 2].map(|k| to_f64(out.data[i * 3 + k]))).collect()
}

fn lin_of(stops: f64) -> f64 {
    stops.exp2()
}

/// The table holds the scan ENCODED, `v^(1/OUT_G)`, and the shader decodes
/// after interpolating: interpolated linear, a shadow's error is many code
/// values; encoded, the print's own near-perceptual response is close to
/// straight between nodes.
const OUT_G: f64 = 2.4;

fn enc(v: f64) -> f64 {
    v.max(0.0).powf(1.0 / OUT_G)
}

fn dec(v: f64) -> f64 {
    v.max(0.0).powf(OUT_G)
}

fn coord(v: f64) -> f64 {
    ((v.max(LO.exp2()).log2() - LO) / (HI - LO)).clamp(0.0, 1.0)
}

struct Lut {
    n: usize,
    /// r fastest, then g, then b.
    data: Vec<[f64; 3]>,
}

impl Lut {
    fn bake(recipe: &Value, data: &Path) -> Result<Lut, String> {
        let n = recipe["size"].as_u64().unwrap_or(33).clamp(9, 65) as usize;
        let pipe = pipeline(recipe, data)?;
        let s = |i: usize| lin_of(LO + (HI - LO) * i as f64 / (n - 1) as f64);
        let mut grid = Vec::with_capacity(n * n * n);
        for b in 0..n {
            for g in 0..n {
                for r in 0..n {
                    grid.push([s(r), s(g), s(b)]);
                }
            }
        }
        let out = process(&pipe, &grid);
        if let Some(bad) = out.iter().find(|c| c.iter().any(|v| !v.is_finite())) {
            return Err(format!("the chain produced a non-finite value {bad:?}"));
        }
        Ok(Lut { n, data: out.iter().map(|c| c.map(enc)).collect() })
    }

    /// The table as the shader holds it: every entry through a half float.
    fn halved(&self) -> Lut {
        Lut { n: self.n, data: self.data.iter().map(|c| c.map(|v| half_to_f64(f16_bits(v as f32)))).collect() }
    }

    fn at(&self, r: usize, g: usize, b: usize) -> [f64; 3] {
        self.data[r + self.n * (g + self.n * b)]
    }

    /// Tetrahedral in the shaped domain — exactly what the shader does
    /// (4 fetches, and measurably closer to the chain than trilinear).
    fn eval(&self, c: [f64; 3]) -> [f64; 3] {
        let m = (self.n - 1) as f64;
        let p = c.map(|v| coord(v) * m);
        let i = p.map(|v| (v.floor() as usize).min(self.n - 2));
        let f = [p[0] - i[0] as f64, p[1] - i[1] as f64, p[2] - i[2] as f64];
        // Walk 000 -> 111 along the axes in order of decreasing fraction.
        let mut ax = [0usize, 1, 2];
        ax.sort_by(|a, b| f[*b].partial_cmp(&f[*a]).unwrap());
        let mut d = [0usize; 3];
        let mut prev = self.at(i[0], i[1], i[2]);
        let mut o = prev;
        for a in ax {
            d[a] = 1;
            let v = self.at(i[0] + d[0], i[1] + d[1], i[2] + d[2]);
            for k in 0..3 {
                o[k] += f[a] * (v[k] - prev[k]);
            }
            prev = v;
        }
        o.map(dec)
    }

    fn shader(&self, recipe: &Value) -> String {
        let n = self.n;
        let mut words = Vec::with_capacity(n * n * n * 2);
        for c in &self.data {
            words.push(pack2(c[0], c[1]));
            words.push(pack2(c[2], 0.0));
        }
        let mut body = String::with_capacity(words.len() * 12);
        for (i, w) in words.iter().enumerate() {
            if i % 12 == 0 {
                body.push_str("\n    ");
            }
            body.push_str(&format!("{w}u,"));
        }
        let mut line = recipe.clone();
        line["baker"] = json!(BAKER);
        let label = recipe["label"].as_str().unwrap_or("Spektrafilm");
        format!(
            r#"//! name: {label}
//! input: front
//! group: Film
//! param: float exposure = 0.0 [-6, 6] "exposure"
//! param: float mix = 1.0 [0, 1] "mix"
// The film -> print -> scan chain of spektrafilm (GPL-3.0,
// github.com/andreavolpato/spektrafilm, Rust port spektrafilm-rs), baked by
// the spektrafilm wobbin into a {n}^3 LUT (log2 in, tetrahedral). Grain, halation and diffusion are
// not in it. Re-bake from the wobbin's panel; do not edit the table.
// spektrafilm: {line}

const SF_N: i32 = {n};
const SF_LO: f32 = {LO:.1};
const SF_HI: f32 = {HI:.1};
const SF_G: f32 = {OUT_G:.1};
const SF_LUT = array<u32, {len}>({body}
);

// A half float by hand: `unpack2x16float` needs a capability wobble does not
// ask for. Every entry is >= 0, so the sign bit is never set.
fn sf_h(h: u32) -> f32 {{
    let e = (h >> 10u) & 31u;
    let m = h & 1023u;
    if (e == 0u) {{
        return f32(m) * 5.9604645e-8;
    }}
    return bitcast<f32>(((e + 112u) << 23u) | (m << 13u));
}}

fn sf_at(i: vec3i) -> vec3f {{
    let k = 2 * (i.x + SF_N * (i.y + SF_N * i.z));
    let rg = SF_LUT[k];
    return vec3f(sf_h(rg & 0xffffu), sf_h(rg >> 16u), sf_h(SF_LUT[k + 1] & 0xffffu));
}}

fn sf_lut(c: vec3f) -> vec3f {{
    let m = f32(SF_N - 1);
    let p = clamp((log2(max(c, vec3f(exp2(SF_LO)))) - SF_LO) / (SF_HI - SF_LO), vec3f(0.0), vec3f(1.0)) * m;
    let i = min(vec3i(floor(p)), vec3i(SF_N - 2));
    let f = p - vec3f(i);
    // Tetrahedral: step 000 -> 111 along the axes, largest fraction first.
    var a = vec3i(1, 0, 0);
    var b = vec3i(1, 1, 0);
    var w = vec3f(f.x, f.y, f.z);
    if (f.x >= f.y) {{
        if (f.y >= f.z) {{
            a = vec3i(1, 0, 0); b = vec3i(1, 1, 0); w = vec3f(f.x, f.y, f.z);
        }} else if (f.x >= f.z) {{
            a = vec3i(1, 0, 0); b = vec3i(1, 0, 1); w = vec3f(f.x, f.z, f.y);
        }} else {{
            a = vec3i(0, 0, 1); b = vec3i(1, 0, 1); w = vec3f(f.z, f.x, f.y);
        }}
    }} else {{
        if (f.x >= f.z) {{
            a = vec3i(0, 1, 0); b = vec3i(1, 1, 0); w = vec3f(f.y, f.x, f.z);
        }} else if (f.y >= f.z) {{
            a = vec3i(0, 1, 0); b = vec3i(0, 1, 1); w = vec3f(f.y, f.z, f.x);
        }} else {{
            a = vec3i(0, 0, 1); b = vec3i(0, 1, 1); w = vec3f(f.z, f.y, f.x);
        }}
    }}
    let v0 = sf_at(i);
    let va = sf_at(i + a);
    let vb = sf_at(i + b);
    let v1 = sf_at(i + vec3i(1));
    let e = v0 + w.x * (va - v0) + w.y * (vb - va) + w.z * (v1 - vb);
    return pow(max(e, vec3f(0.0)), vec3f(SF_G));
}}

fn pass1(uv: vec2f, frag: vec2f) -> vec4f {{
    let s = tex(front, uv);
    if (s.a <= 0.0) {{
        return s;
    }}
    let lin = s.rgb / s.a;
    let film = sf_lut(lin * exp2(P.exposure));
    return vec4f(mix(lin, film, P.mix) * s.a, s.a);
}}
"#,
            len = words.len(),
        )
    }
}

/// Two halves in one u32, low half first — WGSL's `unpack2x16float` order.
fn pack2(a: f64, b: f64) -> u32 {
    f16_bits(a as f32) as u32 | (f16_bits(b as f32) as u32) << 16
}

/// f32 -> IEEE half, round to nearest even, saturating to the largest half.
fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let x = x.abs();
    if x.is_nan() {
        return sign | 0x7e00;
    }
    if x >= 65504.0 {
        return sign | 0x7bff;
    }
    if x < 6.103_515_6e-5 {
        // Subnormal half: a multiple of 2^-24.
        let m = (x * 16_777_216.0).round_ties_even() as u16;
        return sign | m;
    }
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let man = bits & 0x7f_ffff;
    let mut h = ((exp as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h.min(0x7bff) as u16
}

fn half_to_f64(h: u16) -> f64 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f64;
    s * if e == 0 { m * (-24f64).exp2() } else { (1.0 + m / 1024.0) * ((e - 15) as f64).exp2() }
}

/// sRGB code value (0..255) of a linear value, for an error a person can read.
fn srgb8(v: f64) -> f64 {
    let v = v.clamp(0.0, 1.0);
    255.0 * if v <= 0.003_130_8 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

/// How far the LUT (through the shader's own half-float table) lands from
/// the chain itself, on random colours across the range a plate occupies.
fn check(recipe: &Value, data: &Path, samples: usize) -> Result<Value, String> {
    let t = Instant::now();
    let lut = Lut::bake(recipe, data)?;
    let bake_ms = t.elapsed().as_millis() as u64;
    let halfed = lut.halved();
    // xorshift: deterministic and dependency-free.
    let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    // Plates: -10..+5 stops around 1.0, channels within `spread` stops of
    // each other (6 is already a saturated primary), plus greys.
    let spread = 6.0;
    let mut pts = Vec::with_capacity(samples);
    for i in 0..samples {
        if i % 8 == 0 {
            let v = lin_of(-10.0 + 15.0 * rnd());
            pts.push([v; 3]);
        } else {
            let base = -10.0 + 15.0 * rnd();
            pts.push([0; 3].map(|_| lin_of((base + spread * (rnd() - 0.5)).clamp(-10.0, 5.0))));
        }
    }
    let pipe = pipeline(recipe, data)?;
    let truth = process(&pipe, &pts);
    let (mut max, mut sum, mut worst) = (0.0f64, 0.0f64, [0.0; 3]);
    for (p, want) in pts.iter().zip(&truth) {
        let got = halfed.eval(*p);
        let e = (0..3).map(|k| (srgb8(got[k]) - srgb8(want[k])).abs()).fold(0.0, f64::max);
        sum += e;
        if e > max {
            max = e;
            worst = *p;
        }
    }
    // Is the shaper's ceiling past the print's shoulder? A grey at HI and
    // two stops beyond it should scan the same.
    let top = process(&pipe, &[[lin_of(HI); 3], [lin_of(HI + 2.0); 3]]);
    Ok(json!({
        "ok": true,
        "grey_at_ceiling": top[0],
        "grey_past_ceiling": top[1],
        "size": lut.n,
        "bake_ms": bake_ms,
        "samples": samples,
        "max_srgb8": max,
        "mean_srgb8": sum / samples as f64,
        "worst_at": worst,
    }))
}
