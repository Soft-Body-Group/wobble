//! GLSL dialect of the Shader node: Flame Matchbox shaders, pasted as they are.
//!
//! The node's source stays ONE string, so everything downstream of
//! `usershader::compile` (topology hash, param values, undo, setups) is the
//! same as for WGSL. That string may hold:
//!
//! ```text
//! <the .glsl, unchanged>
//! //! pass
//! <the next .glsl of a multi-pass shader>
//! <ShaderNodePreset ...> the optional .xml, anywhere </ShaderNodePreset>
//! ```
//!
//! # How a fragment shader becomes a Wobble pass
//!
//! naga's GLSL front end only reads `#version 440+` and has no combined
//! samplers, while Matchbox shaders are mostly GLSL 1.20. So each pass is
//! rewritten at the TOKEN level (line for line, so errors point at the user's
//! text) and compiled as a COMPUTE shader, then written out as WGSL:
//!
//! * `uniform`s leave the file. Images become texture bindings, everything
//!   else a private global filled from the node's uniform buffer at the top
//!   of `main`. That also legalises what GLSL 1.20 allows and naga does not,
//!   such as a global initialised from a uniform.
//! * `main` is renamed and called from a generated entry point, which supplies
//!   `gl_FragCoord` and stores `gl_FragColor`.
//! * Matchbox's origin is bottom-left and Wobble's is top-left. The flip lives
//!   in `gl_FragCoord` and in every texture read, so the shader never sees it.
//!
//! Each pass is its own translation unit, as its `.glsl` was, so a GLSL node
//! has one WGSL module per pass (`ShaderDef::pass_wgsl`) over one bind layout.

use crate::usershader::{
    self, ParamTy, PassStep, ShaderDef, ShaderError, ShaderParam, BUILTIN_CELLS,
};
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

/// Marks a pass boundary in the node's source.
pub const PASS_MARK: &str = "//! pass";

/// Image inputs a GLSL node may declare. Matchbox shaders go past the WGSL
/// dialect's 4, and each one is still a full-canvas resolve.
pub const MAX_INPUTS: usize = 8;

/// Dispatches one node may run, counting every repeat of a repeated pass.
/// Results are recycled once nothing reads them, so this bounds time more than
/// memory.
pub const MAX_STEPS: usize = 64;

/// A slider range this wide is a preset's way of saying "unbounded".
const UNBOUNDED: f32 = 100_000.0;

/// Is this source the GLSL dialect? Decided from the text alone, so pasting a
/// `.glsl` into any Shader node just works.
pub fn is_glsl(src: &str) -> bool {
    if src.contains("<ShaderNodePreset") {
        return true;
    }
    if src.lines().find(|l| !l.trim().is_empty()).is_some_and(|l| l.trim() == "//! glsl") {
        return true;
    }
    has_void_main(src) && !src.contains("fn pass1") && !src.contains("fn main_image")
}

fn has_void_main(src: &str) -> bool {
    src.match_indices("void").any(|(i, _)| {
        src[i + 4..].trim_start().strip_prefix("main").is_some_and(|r| r.trim_start().starts_with('('))
    })
}

/// Join a Matchbox shader's files into one node source.
pub fn bundle(passes: &[String], xml: Option<&str>) -> String {
    let mut out = String::new();
    for (i, p) in passes.iter().enumerate() {
        if i > 0 {
            let _ = writeln!(out, "{PASS_MARK}");
        }
        out.push_str(p);
        if !p.ends_with('\n') {
            out.push('\n');
        }
    }
    if let Some(x) = xml {
        out.push('\n');
        out.push_str(x.trim());
        out.push('\n');
    }
    out
}

/// Read a Matchbox shader from any one of its files: every pass beside it
/// (`name.glsl`, or `name.1.glsl`, `name.2.glsl`...) and `name.xml`.
pub fn read_files(path: &std::path::Path) -> std::io::Result<String> {
    let dir = path.parent().unwrap_or(std::path::Path::new("."));
    let stem = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
    let base = match stem.rsplit_once('.') {
        Some((b, n)) if n.parse::<u32>().is_ok() => b.to_string(),
        _ => stem,
    };
    let mut passes: Vec<(u32, std::path::PathBuf)> = Vec::new();
    for e in std::fs::read_dir(dir)?.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let Some(s) = name.strip_suffix(".glsl") else { continue };
        if s == base {
            passes.push((0, e.path()));
        } else if let Some(n) = s.strip_prefix(&base).and_then(|r| r.strip_prefix('.')) {
            if let Ok(n) = n.parse() {
                passes.push((n, e.path()));
            }
        }
    }
    passes.sort();
    let texts = passes
        .iter()
        .map(|(_, p)| Ok(String::from_utf8_lossy(&std::fs::read(p)?).into_owned()))
        .collect::<std::io::Result<Vec<_>>>()?;
    let xml = std::fs::read(dir.join(format!("{base}.xml")))
        .ok()
        .map(|b| String::from_utf8_lossy(&b).into_owned());
    Ok(bundle(&texts, xml.as_deref()))
}

/// What a fresh GLSL node starts as.
pub const STARTER: &str = r#"//! glsl
// Paste a Matchbox .glsl over this. Paste its .xml below it for the real
// labels, ranges and defaults.
uniform sampler2D front;
uniform float adsk_result_w, adsk_result_h;
uniform float amount;

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(adsk_result_w, adsk_result_h);
    vec3 c = texture2D(front, uv).rgb;
    gl_FragColor = vec4(c * (1.0 + amount), 1.0);
}
"#;

// ---------------------------------------------------------------- the XML

#[derive(Clone, Debug, Default)]
struct XmlUniform {
    name: String,
    label: String,
    default: Option<String>,
    min: Option<f32>,
    max: Option<f32>,
    /// (default, min, max) per lane.
    subs: Vec<(Option<String>, Option<f32>, Option<f32>)>,
    choices: Vec<(String, i32)>,
    page: i32,
    col: i32,
    row: i32,
    /// Port order, for images.
    index: Option<i32>,
    input_type: String,
    /// `NoInput="White"`: an unconnected input reads as white.
    white: bool,
    wrap: Wrap,
    nearest: bool,
}

#[derive(Clone, Debug, Default)]
struct Preset {
    name: String,
    matte_provider: bool,
    non_commercial: bool,
    uniforms: Vec<XmlUniform>,
    pages: HashMap<i32, String>,
    cols: HashMap<(i32, i32), String>,
    /// Passes the preset asks to repeat, as (pass index, executions).
    repeats: Vec<(usize, u32)>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
enum Wrap {
    #[default]
    Clamp,
    Repeat,
    Mirror,
    Black,
}

fn unescape(s: &str) -> String {
    s.replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

/// Attributes of one tag body (`Name a="1" b='2' /`).
fn attrs(body: &str) -> (String, HashMap<String, String>) {
    let b = body.trim().trim_end_matches('/').trim_end();
    let end = b.find(char::is_whitespace).unwrap_or(b.len());
    let name = b[..end].to_string();
    let mut map = HashMap::new();
    let mut rest = &b[end..];
    while let Some(eq) = rest.find('=') {
        let key = rest[..eq].trim().to_string();
        let after = rest[eq + 1..].trim_start();
        let Some(q) = after.chars().next().filter(|c| *c == '"' || *c == '\'') else { break };
        let Some(close) = after[1..].find(q) else { break };
        map.insert(key, unescape(&after[1..1 + close]));
        rest = &after[close + 2..];
    }
    (name, map)
}

fn parse_preset(xml: &str) -> Preset {
    let mut p = Preset::default();
    let mut cur: Option<XmlUniform> = None;
    let mut rest = xml;
    let num = |m: &HashMap<String, String>, k: &str| m.get(k).and_then(|v| v.trim().parse::<f32>().ok());
    let int = |m: &HashMap<String, String>, k: &str| m.get(k).and_then(|v| v.trim().parse::<i32>().ok());
    while let Some(open) = rest.find('<') {
        rest = &rest[open + 1..];
        if let Some(r) = rest.strip_prefix("!--") {
            rest = r.split_once("-->").map_or("", |x| x.1);
            continue;
        }
        // A `>` inside a quoted attribute must not end the tag.
        let mut quote = None;
        let mut end = None;
        for (i, c) in rest.char_indices() {
            match (quote, c) {
                (None, '"' | '\'') => quote = Some(c),
                (Some(q), c) if c == q => quote = None,
                (None, '>') => {
                    end = Some(i);
                    break;
                }
                _ => {}
            }
        }
        let Some(end) = end else { break };
        let body = &rest[..end];
        rest = &rest[end + 1..];
        if body.starts_with('?') || body.starts_with('!') {
            continue;
        }
        if body.starts_with("/Uniform") {
            p.uniforms.extend(cur.take());
            continue;
        }
        if body.starts_with('/') {
            continue;
        }
        let self_closed = body.trim_end().ends_with('/');
        let (tag, a) = attrs(body);
        let get = |k: &str| a.get(k).cloned().unwrap_or_default();
        match tag.as_str() {
            "ShaderNodePreset" => {
                p.name = get("Name");
                p.matte_provider = get("MatteProvider").eq_ignore_ascii_case("true");
                p.non_commercial = get("CommercialUsePermitted").eq_ignore_ascii_case("false");
            }
            "Shader" => {
                let n = int(&a, "NbExecutions").unwrap_or(1);
                let i = int(&a, "Index").unwrap_or(1);
                if n > 1 && i >= 1 {
                    p.repeats.push((i as usize - 1, n as u32));
                }
            }
            "Page" => {
                if let Some(i) = int(&a, "Page") {
                    p.pages.insert(i, get("Name"));
                }
            }
            "Col" => {
                if let (Some(pg), Some(c)) = (int(&a, "Page"), int(&a, "Col")) {
                    p.cols.insert((pg, c), get("Name"));
                }
            }
            "Uniform" => {
                let wrap_of = |v: String| match v.as_str() {
                    "GL_REPEAT" => Wrap::Repeat,
                    "GL_MIRRORED_REPEAT" => Wrap::Mirror,
                    "GL_CLAMP_TO_BORDER" | "GL_CLAMP" => Wrap::Black,
                    _ => Wrap::Clamp,
                };
                let u = XmlUniform {
                    name: get("Name"),
                    label: get("DisplayName"),
                    default: a.get("Default").cloned(),
                    min: num(&a, "Min"),
                    max: num(&a, "Max"),
                    page: int(&a, "Page").unwrap_or(0),
                    col: int(&a, "Col").unwrap_or(0),
                    row: int(&a, "Row").unwrap_or(0),
                    index: int(&a, "Index"),
                    input_type: get("InputType"),
                    white: get("NoInput").eq_ignore_ascii_case("white"),
                    wrap: wrap_of(get("GL_TEXTURE_WRAP_S")),
                    nearest: get("GL_TEXTURE_MAG_FILTER") == "GL_NEAREST",
                    ..Default::default()
                };
                if self_closed {
                    p.uniforms.push(u);
                } else {
                    p.uniforms.extend(cur.replace(u));
                }
            }
            "SubUniform" => {
                if let Some(u) = &mut cur {
                    u.subs.push((a.get("Default").cloned(), num(&a, "Min"), num(&a, "Max")));
                }
            }
            "PopupEntry" => {
                if let (Some(u), Some(v)) = (&mut cur, int(&a, "Value")) {
                    u.choices.push((get("Title"), v));
                }
            }
            _ => {}
        }
    }
    p.uniforms.extend(cur);
    p
}

// ------------------------------------------------------------- the tokens

#[derive(Clone, Copy, PartialEq, Debug)]
enum Tk {
    Ident,
    Num,
    Punct,
    Space,
}

#[derive(Clone, Debug)]
struct Tok {
    kind: Tk,
    text: String,
    /// Part of a `#` line: rewritten, but never read as a declaration.
    directive: bool,
}

/// Replace comments with spaces, keeping every newline.
fn strip_comments(src: &str) -> String {
    let b: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == '/' && b.get(i + 1) == Some(&'/') {
            while i < b.len() && b[i] != '\n' {
                i += 1;
            }
        } else if b[i] == '/' && b.get(i + 1) == Some(&'*') {
            i += 2;
            while i < b.len() && !(b[i] == '*' && b.get(i + 1) == Some(&'/')) {
                if b[i] == '\n' {
                    out.push('\n');
                }
                i += 1;
            }
            i += 2;
            out.push(' ');
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    out
}

fn tokenize(src: &str) -> Vec<Tok> {
    let b: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    let mut directive = false;
    let mut line_start = true;
    while i < b.len() {
        let c = b[i];
        let s = i;
        let kind = if c.is_whitespace() {
            while i < b.len() && b[i].is_whitespace() {
                if b[i] == '\n' {
                    // A trailing backslash continues the directive.
                    let cont = i > 0 && b[i - 1] == '\\';
                    if !cont {
                        directive = false;
                    }
                    line_start = true;
                }
                i += 1;
            }
            Tk::Space
        } else if c.is_ascii_alphabetic() || c == '_' {
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == '_') {
                i += 1;
            }
            line_start = false;
            Tk::Ident
        } else if c.is_ascii_digit() || (c == '.' && b.get(i + 1).is_some_and(char::is_ascii_digit)) {
            while i < b.len()
                && (b[i].is_ascii_alphanumeric()
                    || b[i] == '.'
                    || ((b[i] == '+' || b[i] == '-') && matches!(b[i - 1], 'e' | 'E')))
            {
                i += 1;
            }
            line_start = false;
            Tk::Num
        } else {
            if c == '#' && line_start {
                directive = true;
            }
            line_start = false;
            i += 1;
            Tk::Punct
        };
        out.push(Tok { kind, text: b[s..i].iter().collect(), directive });
    }
    out
}

// ------------------------------------------------------------- one pass

#[derive(Clone, Debug)]
struct Uniform {
    ty: String,
    name: String,
    array: bool,
    /// `uniform float x = 0.5;`
    init: Option<String>,
}

#[derive(Default)]
struct Scan {
    toks: Vec<Tok>,
    uniforms: Vec<Uniform>,
    /// Private globals standing in for `varying`s: (type, name).
    varyings: Vec<(String, String)>,
    /// Names that mean the output colour (`out vec4 fragColor`).
    outs: HashSet<String>,
    /// Token ranges of `name = init` statements to run at the top of main.
    hoisted: Vec<(String, std::ops::Range<usize>)>,
    /// Functions to re-emit later: (their tokens, the token to follow). naga
    /// needs a callee defined before its caller, which a prototype sidesteps
    /// in GLSL proper.
    moves: Vec<(std::ops::Range<usize>, usize)>,
    /// Tokens removed from the file AFTER rewriting, because a hoisted
    /// initialiser still needs their text.
    hidden: Vec<usize>,
    /// Functions declared but never defined: Flame's, if anyone's.
    undefined: Vec<String>,
}

const QUALIFIERS: &[&str] = &[
    "const", "uniform", "varying", "attribute", "in", "out", "inout", "highp", "mediump", "lowp",
    "flat", "smooth", "noperspective", "centroid", "invariant",
];

fn blank(t: &mut Tok) {
    if t.kind != Tk::Space {
        t.kind = Tk::Space;
        t.text = " ".into();
    }
}

/// Adsk helpers we can define. A shader's own prototypes of these go away.
fn adsk_fn(name: &str) -> Option<&'static str> {
    Some(match name {
        "adsk_getLuminance" => {
            "float adsk_getLuminance(vec3 c) { return dot(c, wob_P.c5.xyz); }\n"
        }
        "adsk_rgb2hsv" => {
            "vec3 adsk_rgb2hsv(vec3 c) { vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0); \
             vec4 p = c.g < c.b ? vec4(c.bg, K.wz) : vec4(c.gb, K.xy); \
             vec4 q = c.r < p.x ? vec4(p.xyw, c.r) : vec4(c.r, p.yzx); \
             float d = q.x - min(q.w, q.y); float e = 1.0e-10; \
             return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x); }\n"
        }
        "adsk_hsv2rgb" => {
            "vec3 adsk_hsv2rgb(vec3 c) { vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0); \
             vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www); \
             return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y); }\n"
        }
        "adsk_scene2log" => {
            "vec3 adsk_scene2log(vec3 c) { vec3 v = max(c * 0.9892 + vec3(0.0108), vec3(1.0e-6)); \
             return (vec3(685.0) + 300.0 * log(v) / log(10.0)) / 1023.0; }\n"
        }
        "adsk_log2scene" => {
            "vec3 adsk_log2scene(vec3 c) { vec3 e = (c * 1023.0 - vec3(685.0)) / 300.0; \
             return (exp(e * log(10.0)) - vec3(0.0108)) / 0.9892; }\n"
        }
        "adsk_shadows" => {
            "float adsk_shadows(float l, float m) { return 1.0 - smoothstep(0.0, max(m, 1.0e-4), l); }\n"
        }
        "adsk_highlights" => {
            "float adsk_highlights(float l, float m) { return smoothstep(min(m, 0.9999), 1.0, l); }\n"
        }
        _ => return None,
    })
}

/// Helpers written from their names alone, so a result may differ from Flame's.
const ADSK_APPROX: &[&str] = &["adsk_scene2log", "adsk_log2scene", "adsk_shadows", "adsk_highlights"];
const ADSK_FNS: &[&str] = &[
    "adsk_getLuminance", "adsk_rgb2hsv", "adsk_hsv2rgb", "adsk_scene2log", "adsk_log2scene",
    "adsk_shadows", "adsk_highlights",
];

fn scan(src: &str) -> Scan {
    let clean = strip_comments(src);
    let mut s = Scan { toks: tokenize(&clean), ..Default::default() };
    // Directives naga would refuse; the prelude sets its own version.
    let mut i = 0;
    while i < s.toks.len() {
        if s.toks[i].directive && s.toks[i].text == "#" {
            let word = s.toks[i + 1..]
                .iter()
                .find(|t| t.kind != Tk::Space)
                .map(|t| t.text.clone())
                .unwrap_or_default();
            if matches!(word.as_str(), "version" | "extension" | "pragma") {
                let mut j = i;
                while j < s.toks.len() && s.toks[j].directive {
                    blank(&mut s.toks[j]);
                    s.toks[j].directive = false;
                    j += 1;
                }
            }
        }
        i += 1;
    }

    let sig: Vec<usize> = (0..s.toks.len())
        .filter(|&i| s.toks[i].kind != Tk::Space && !s.toks[i].directive)
        .collect();
    let mut non_const: HashSet<String> = HashSet::new();
    // (name, first token, last token, has a body), in source order.
    let mut funcs: Vec<(String, usize, usize, bool)> = Vec::new();
    let mut p = 0;
    while p < sig.len() {
        // One top-level statement: sig[p..=q].
        let (mut depth, mut q, mut saw_eq, mut body) = (0i32, p, false, None);
        let is_struct = s.toks[sig[p]].text == "struct";
        while q < sig.len() {
            let t = s.toks[sig[q]].text.as_str();
            match t {
                "(" | "[" => depth += 1,
                ")" | "]" => depth -= 1,
                "=" if depth == 0 => saw_eq = true,
                "{" => {
                    if depth == 0 && !saw_eq && !is_struct && body.is_none() {
                        body = Some(q);
                    }
                    depth += 1;
                }
                "}" => {
                    depth -= 1;
                    if depth == 0 && body.is_some() {
                        break;
                    }
                }
                ";" if depth == 0 => break,
                _ => {}
            }
            q += 1;
        }
        let q = q.min(sig.len() - 1);
        let stmt: Vec<usize> = sig[p..=q].to_vec();
        p = q + 1;
        let text = |s: &Scan, k: usize| s.toks[stmt[k]].text.clone();

        // `precision highp float;` means nothing here.
        if text(&s, 0) == "precision" {
            stmt.iter().for_each(|&k| blank(&mut s.toks[k]));
            continue;
        }
        if is_struct {
            continue;
        }
        // Qualifiers, including `layout(...)`.
        let mut k = 0;
        let mut quals: Vec<String> = Vec::new();
        while k < stmt.len() {
            let w = text(&s, k);
            if w == "layout" {
                let mut d = 0;
                k += 1;
                while k < stmt.len() {
                    match s.toks[stmt[k]].text.as_str() {
                        "(" => d += 1,
                        ")" => {
                            d -= 1;
                            if d == 0 {
                                k += 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                    k += 1;
                }
                continue;
            }
            if !QUALIFIERS.contains(&w.as_str()) {
                break;
            }
            quals.push(w);
            k += 1;
        }
        if k + 1 >= stmt.len() {
            continue;
        }
        let ty = text(&s, k);
        let head = k;
        k += 1;
        // Skip an array suffix on the type (`float[3] x`).
        if text(&s, k) == "[" {
            while k < stmt.len() && text(&s, k) != "]" {
                k += 1;
            }
            k += 1;
        }
        if k >= stmt.len() {
            continue;
        }
        let name_tok = k;
        let name = text(&s, k);

        // A function, or its prototype.
        if stmt.get(k + 1).is_some_and(|&t| s.toks[t].text == "(") {
            if name == "main" && body.is_some() {
                s.toks[stmt[name_tok]].text = "wob_main".into();
            } else if body.is_none() && adsk_fn(&name).is_some() {
                stmt.iter().for_each(|&t| blank(&mut s.toks[t]));
                continue;
            }
            funcs.push((name, stmt[0], *stmt.last().unwrap(), body.is_some()));
            continue;
        }
        let _ = head;

        // Declarators, split on top-level commas.
        let mut decls: Vec<(usize, usize)> = Vec::new();
        let (mut a, mut d) = (k, 0);
        for j in k..stmt.len() {
            match text(&s, j).as_str() {
                "(" | "[" | "{" => d += 1,
                ")" | "]" | "}" => d -= 1,
                "," if d == 0 => {
                    decls.push((a, j));
                    a = j + 1;
                }
                ";" if d == 0 => {
                    decls.push((a, j));
                    a = j + 1;
                }
                _ => {}
            }
        }
        if a < stmt.len() {
            decls.push((a, stmt.len()));
        }
        let parts: Vec<(String, bool, Option<usize>, usize)> = decls
            .iter()
            .filter(|(a, b)| a < b)
            .map(|&(a, b)| {
                let eq = (a..b).find(|&j| text(&s, j) == "=");
                let array = (a..eq.unwrap_or(b)).any(|j| text(&s, j) == "[");
                (text(&s, a), array, eq, b)
            })
            .collect();

        if quals.iter().any(|q| q == "uniform") {
            for (n, array, eq, b) in &parts {
                let init = eq.map(|e| {
                    (e + 1..*b).map(|j| text(&s, j)).collect::<Vec<_>>().join(" ")
                });
                s.uniforms.push(Uniform { ty: ty.clone(), name: n.clone(), array: *array, init });
            }
            stmt.iter().for_each(|&t| blank(&mut s.toks[t]));
            continue;
        }
        if quals.iter().any(|q| q == "out") {
            for (n, ..) in &parts {
                s.outs.insert(n.clone());
            }
            stmt.iter().for_each(|&t| blank(&mut s.toks[t]));
            continue;
        }
        if quals.iter().any(|q| q == "varying" || q == "in" || q == "attribute") {
            for (n, ..) in &parts {
                s.varyings.push((ty.clone(), n.clone()));
            }
            stmt.iter().for_each(|&t| blank(&mut s.toks[t]));
            continue;
        }

        // A plain global. `const` stays const unless it reads something that
        // is only known at run time, which GLSL 1.20 compilers let through.
        let is_const = quals.iter().any(|q| q == "const");
        let uniform_names: HashSet<&str> = s.uniforms.iter().map(|u| u.name.as_str()).collect();
        let dynamic = parts.iter().any(|(_, _, eq, b)| {
            eq.is_some_and(|e| {
                (e + 1..*b).any(|j| {
                    let w = text(&s, j);
                    uniform_names.contains(w.as_str()) || non_const.contains(&w)
                })
            })
        });
        if is_const && !dynamic {
            continue;
        }
        for (n, _, eq, b) in &parts {
            non_const.insert(n.clone());
            if let Some(e) = eq {
                s.hoisted.push((n.clone(), stmt[*e + 1]..stmt[*b - 1] + 1));
                // `;` at the end of the last declarator is not part of it.
                if text(&s, *b - 1) == ";" || text(&s, *b - 1) == "," {
                    let r = &mut s.hoisted.last_mut().unwrap().1;
                    r.end = stmt[*b - 1];
                }
                s.hidden.extend((*e..*b).map(|j| stmt[j]));
            }
        }
        if is_const {
            for j in 0..head {
                if text(&s, j) == "const" {
                    blank(&mut s.toks[stmt[j]]);
                }
            }
        }
    }
    order_functions(&mut s, &funcs);
    size_arrays(&mut s, &sig);
    s
}

/// `float t[] = float[](1.0, 2.0)` gets its length written in: naga needs one.
fn size_arrays(s: &mut Scan, sig: &[usize]) {
    let text = |s: &Scan, i: usize| sig.get(i).map_or("", |&t| s.toks[t].text.as_str()).to_string();
    for i in 0..sig.len() {
        let shape = ["[", "]", "="].iter().enumerate().all(|(k, w)| text(s, i + k) == *w)
            && text(s, i + 4) == "["
            && text(s, i + 5) == "]"
            && text(s, i + 6) == "(";
        if !shape {
            continue;
        }
        let (mut depth, mut n, mut j) = (0, 1, i + 6);
        while j < sig.len() {
            match text(s, j).as_str() {
                "(" | "[" | "{" => depth += 1,
                ")" | "]" | "}" => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                "," if depth == 1 => n += 1,
                _ => {}
            }
            j += 1;
        }
        s.toks[sig[i + 1]].text = format!("{n}]");
        s.toks[sig[i + 5]].text = format!("{n}]");
    }
}

/// Delay any function that calls one defined after it until that callee has
/// been emitted. Functions only ever move DOWN, so every global they use is
/// still declared above them.
fn order_functions(s: &mut Scan, funcs: &[(String, usize, usize, bool)]) {
    let defs: Vec<_> = funcs.iter().filter(|f| f.3).collect();
    let names: HashSet<&str> = defs.iter().map(|f| f.0.as_str()).collect();
    let protos: Vec<_> = funcs.iter().filter(|f| !f.3).collect();
    s.undefined = protos
        .iter()
        .filter(|f| !names.contains(f.0.as_str()))
        .map(|f| f.0.clone())
        .collect();
    if !protos.iter().any(|f| names.contains(f.0.as_str())) {
        return;
    }
    // naga numbers a function where it is first DECLARED, so a prototype
    // would pin it ahead of its own callees.
    for f in protos.iter().filter(|f| names.contains(f.0.as_str())) {
        (f.1..=f.2).for_each(|t| blank(&mut s.toks[t]));
    }
    let calls: Vec<HashSet<&str>> = defs
        .iter()
        .map(|f| {
            s.toks[f.1..=f.2]
                .iter()
                .filter(|t| t.kind == Tk::Ident && t.text != f.0)
                .filter_map(|t| names.get(t.text.as_str()).copied())
                .collect()
        })
        .collect();
    let mut done: HashSet<&str> = HashSet::new();
    let mut waiting: Vec<usize> = Vec::new();
    let mut moves = Vec::new();
    for (i, f) in defs.iter().enumerate() {
        waiting.push(i);
        while let Some(k) = waiting.iter().position(|&w| calls[w].is_subset(&done)) {
            let w = waiting.remove(k);
            done.insert(defs[w].0.as_str());
            if w != i {
                moves.push((defs[w].1..defs[w].2 + 1, f.2));
            }
        }
    }
    s.moves = moves;
}

// ---------------------------------------------------------- compiling

#[derive(Clone, Debug)]
struct Input {
    name: String,
    matte: bool,
    white: bool,
    wrap: Wrap,
    nearest: bool,
}

fn glsl_scalar_ty(ty: &str) -> Option<ParamTy> {
    Some(match ty {
        "float" => ParamTy::Float,
        "int" => ParamTy::Int,
        "bool" => ParamTy::Bool,
        "vec2" => ParamTy::Vec2,
        "vec3" => ParamTy::Color,
        "vec4" => ParamTy::Vec4,
        _ => return None,
    })
}

fn parse_nums(s: &str) -> Vec<f32> {
    s.split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E')))
        .filter_map(|p| p.parse().ok())
        .collect()
}

fn bool_text(s: &str) -> f32 {
    let t = s.trim();
    (t.eq_ignore_ascii_case("true") || t == "1") as u32 as f32
}

fn make_param(u: &Uniform, ty: ParamTy, xml: Option<&XmlUniform>, preset: &Preset) -> ShaderParam {
    let lanes = ty.lanes();
    let mut default = [0f32; 4];
    if let Some(init) = &u.init {
        let v = parse_nums(init);
        match v.len() {
            0 => default[0] = bool_text(init),
            1 => default[..lanes].fill(v[0]),
            _ => v.iter().take(lanes).enumerate().for_each(|(i, x)| default[i] = *x),
        }
    }
    let (mut min, mut max) = match ty {
        ParamTy::Int => (0.0, 10.0),
        ParamTy::Bool => (0.0, 1.0),
        ParamTy::Vec2 | ParamTy::Vec4 => (-1.0, 2.0),
        _ => (0.0, 1.0),
    };
    let mut label = u.name.clone();
    let mut group = None;
    let mut choices = Vec::new();
    if let Some(x) = xml {
        if !x.label.is_empty() {
            label = x.label.clone();
        }
        choices = x.choices.clone();
        if !x.subs.is_empty() {
            for (i, (d, ..)) in x.subs.iter().take(lanes).enumerate() {
                if let Some(v) = d.as_deref().and_then(|d| d.trim().parse().ok()) {
                    default[i] = v;
                }
            }
        } else if let Some(d) = &x.default {
            if ty == ParamTy::Bool {
                default[0] = bool_text(d);
            } else {
                let v = parse_nums(d);
                if v.len() == 1 {
                    default[..lanes].fill(v[0]);
                } else {
                    v.iter().take(lanes).enumerate().for_each(|(i, n)| default[i] = *n);
                }
            }
        }
        let lo = x.min.or_else(|| x.subs.iter().filter_map(|s| s.1).reduce(f32::min));
        let hi = x.max.or_else(|| x.subs.iter().filter_map(|s| s.2).reduce(f32::max));
        if let (Some(lo), Some(hi)) = (lo, hi) {
            if lo < hi {
                min = lo;
                max = hi;
            }
        }
        // Presets write +-1e6 for "no limit". A slider over that is useless,
        // so an equal pair asks the panel for a plain drag.
        if max - min >= UNBOUNDED {
            min = 0.0;
            max = 0.0;
        }
        let page = preset.pages.get(&x.page).filter(|s| !s.is_empty());
        let col = preset.cols.get(&(x.page, x.col)).filter(|s| !s.is_empty());
        group = match (page, col) {
            (Some(p), Some(c)) if preset.pages.len() > 1 => Some(format!("{p} / {c}")),
            (_, Some(c)) => Some(c.clone()),
            (Some(p), None) => Some(p.clone()),
            _ => None,
        };
    }
    ShaderParam { name: u.name.clone(), label, ty, default, min, max, group, choices }
}

/// Split the container: (preset text, [(pass text, first line, 1-based)]).
fn split(src: &str) -> (Option<String>, Vec<(String, usize)>) {
    let mut text = src.to_string();
    let mut xml = None;
    if let Some(start) = text.find("<ShaderNodePreset") {
        let start = text[..start].rfind("<?xml").unwrap_or(start);
        let close = "</ShaderNodePreset>";
        let end = text[start..].find(close).map_or(text.len(), |e| start + e + close.len());
        xml = Some(text[start..end].to_string());
        let lines = text[start..end].matches('\n').count();
        text.replace_range(start..end, &"\n".repeat(lines));
    }
    let mut passes = vec![(String::new(), 1)];
    for (i, line) in text.lines().enumerate() {
        if line.trim() == PASS_MARK {
            passes.push((String::new(), i + 2));
        } else {
            let p = &mut passes.last_mut().unwrap().0;
            p.push_str(line);
            p.push('\n');
        }
    }
    (xml, passes)
}

/// The adsk uniforms we can answer, as a GLSL expression.
fn builtin_expr(name: &str, inputs: &[Input]) -> Option<(&'static str, String)> {
    Some(match name {
        "adsk_result_w" => ("float", "wob_P.c0.x".into()),
        "adsk_result_h" => ("float", "wob_P.c0.y".into()),
        "adsk_result_frameratio" => ("float", "wob_P.c4.x".into()),
        "adsk_result_pixelratio" => ("float", "1.0".into()),
        "adsk_time" => ("float", "wob_P.c2.x".into()),
        "adsk_result_execution" => ("int", "wob_P.c3.y".into()),
        "adsk_degrade" => ("bool", "false".into()),
        // There is no frame-to-frame feedback, so every frame is a first one.
        "adsk_accum_no_prev_frame" => ("bool", "true".into()),
        _ => {
            let rest = name.strip_prefix("adsk_")?;
            let (input, what) = rest.rsplit_once('_')?;
            inputs.iter().find(|i| i.name == input || i.name == format!("adsk_{input}"))?;
            let size = format!("vec2(textureSize(sampler2D({input}, wob_samp), 0))");
            match what {
                "w" => ("float", format!("{size}.x")),
                "h" => ("float", format!("{size}.y")),
                "frameratio" => ("float", format!("({size}.x / max({size}.y, 1.0))")),
                "pixelratio" => ("float", "1.0".into()),
                _ => return None,
            }
        }
    })
}

fn result_index(name: &str) -> Option<usize> {
    name.strip_prefix("adsk_results_pass")?.parse().ok()
}

fn sample_fn(i: &Input, port: usize) -> String {
    let n = &i.name;
    // Every local is prefixed: an input may be called `uv` or `p`.
    let wrap = match i.wrap {
        Wrap::Clamp | Wrap::Black => "wob_uv",
        Wrap::Repeat => "fract(wob_uv)",
        Wrap::Mirror => "(1.0 - abs(mod(wob_uv, 2.0) - 1.0))",
    };
    let read = if i.nearest {
        format!(
            "ivec2 wob_s = textureSize(sampler2D({n}, wob_samp), 0); \
             ivec2 wob_q = clamp(ivec2(floor(vec2(wob_p.x, 1.0 - wob_p.y) * vec2(wob_s))), ivec2(0), wob_s - ivec2(1)); \
             vec4 wob_c = texelFetch(sampler2D({n}, wob_samp), wob_q, 0);"
        )
    } else {
        format!("vec4 wob_c = textureLod(sampler2D({n}, wob_samp), vec2(wob_p.x, 1.0 - wob_p.y), 0.0);")
    };
    let black = if i.wrap == Wrap::Black {
        "if (wob_uv.x < 0.0 || wob_uv.y < 0.0 || wob_uv.x > 1.0 || wob_uv.y > 1.0) return vec4(0.0);"
    } else {
        ""
    };
    let matte = if i.matte { "wob_c = wob_c.aaaa;" } else { "" };
    let white = if i.white {
        format!("if ((wob_P.c3.z & {}) == 0) return vec4(1.0);", 1 << port)
    } else {
        String::new()
    };
    format!(
        "vec4 wob_tex_{n}(vec2 wob_uv) {{ {white} {black} vec2 wob_p = {wrap}; {read} {matte} return wob_c; }}\n\
         vec4 wob_tex_{n}(vec2 wob_uv, float wob_b) {{ return wob_tex_{n}(wob_uv); }}\n\
         vec4 wob_lod_{n}(vec2 wob_uv, float wob_l) {{ if (wob_l <= 0.0) return wob_tex_{n}(wob_uv); \
         vec2 wob_d = pow(2.0, wob_l) / vec2(textureSize(sampler2D({n}, wob_samp), 0)); vec4 wob_a = vec4(0.0); \
         for (int wob_y = 0; wob_y < 4; wob_y++) for (int wob_x = 0; wob_x < 4; wob_x++) \
         wob_a += wob_tex_{n}(wob_uv + (vec2(float(wob_x), float(wob_y)) - 1.5) * 0.25 * wob_d); \
         return wob_a / 16.0; }}\n"
    )
}

/// Rewrite identifiers, returning each token's final text.
fn rewrite(s: &Scan, inputs: &[Input]) -> (Vec<String>, usize) {
    let mut sites = 0;
    let t = &s.toks;
    let next_sig = |mut i: usize| {
        i += 1;
        while i < t.len() && t[i].kind == Tk::Space {
            i += 1;
        }
        i
    };
    let mut out: Vec<String> = t.iter().map(|x| x.text.clone()).collect();
    let mut i = 0;
    while i < t.len() {
        if t[i].kind != Tk::Ident {
            i += 1;
            continue;
        }
        let w = t[i].text.as_str();
        let n = next_sig(i);
        let call = t.get(n).is_some_and(|x| x.text == "(");
        match w {
            "texture2D" | "texture" | "texture2DLod" | "textureLod" if call => {
                // A read straight off a named input gets that input's own
                // wrap, filter and matte handling.
                let a = next_sig(n);
                let c = next_sig(a);
                let named = t.get(a).filter(|x| x.kind == Tk::Ident).map(|x| x.text.as_str());
                let direct = named.filter(|nm| inputs.iter().any(|i| i.name == *nm));
                let stem = if w.ends_with("Lod") { "wob_lod" } else { "wob_tex" };
                if let (Some(nm), true) = (direct, t.get(c).is_some_and(|x| x.text == ",")) {
                    out[i] = format!("{stem}_{nm}");
                    out[a] = String::new();
                    out[c] = String::new();
                } else {
                    out[i] = stem.into();
                }
            }
            "dFdx" | "dFdxFine" | "dFdxCoarse" | "dFdy" | "dFdyFine" | "dFdyCoarse" | "fwidth"
                if call =>
            {
                let f = match w.as_bytes()[0..4] {
                    [b'd', b'F', b'd', b'x'] => "wob_ddx",
                    [b'd', b'F', b'd', b'y'] => "wob_ddy",
                    _ => "wob_fw",
                };
                out[i] = f.into();
                out[n] = format!("({sites}, ");
                sites += 1;
            }
            "texelFetch" if call => out[i] = "wob_fetch".into(),
            "textureSize" if call => out[i] = "wob_size".into(),
            "sampler2D" => out[i] = "texture2D".into(),
            "texture" | "sample" | "filter" | "input" | "output" => out[i] = format!("wob_v_{w}"),
            "gl_FragColor" => out[i] = "wob_FragColor".into(),
            "gl_FragCoord" => out[i] = "wob_FragCoord".into(),
            "gl_FragData" | "gl_TexCoord" => {
                // `gl_FragData[0]`: swallow the subscript.
                let close = next_sig(next_sig(n));
                if t.get(n).is_some_and(|x| x.text == "[")
                    && t.get(close).is_some_and(|x| x.text == "]")
                {
                    for o in out.iter_mut().take(close + 1).skip(i + 1) {
                        o.clear();
                    }
                    i = close;
                }
                out[next_back(t, w, i)] = if w == "gl_FragData" {
                    "wob_FragColor".into()
                } else {
                    "wob_TexCoord".into()
                };
            }
            "discard" => out[i] = "{ wob_FragColor = vec4(0.0); return; }".into(),
            _ if s.outs.contains(w) => out[i] = "wob_FragColor".into(),
            _ => {}
        }
        i += 1;
    }
    for (r, after) in &s.moves {
        let text = out[r.clone()].concat();
        out[r.clone()].iter_mut().for_each(String::clear);
        out[*after].push('\n');
        out[*after].push_str(&text);
    }
    (out, sites)
}

/// Index of the identifier `w` at or before `i`.
fn next_back(t: &[Tok], w: &str, mut i: usize) -> usize {
    while i > 0 && t[i].text != w {
        i -= 1;
    }
    i
}

struct PassOut {
    wgsl: String,
}

#[allow(clippy::too_many_arguments)]
fn build_pass(
    pass: usize,
    n_passes: usize,
    scan: &Scan,
    line0: usize,
    inputs: &[Input],
    slots: &[String],
    n_slots: usize,
    params: &[ShaderParam],
    keep_alpha: bool,
    errors: &mut Vec<ShaderError>,
    notes: &mut Vec<String>,
) -> Option<PassOut> {
    let mut w = String::from("#version 450 core\n");
    w.push_str("layout(local_size_x = 8, local_size_y = 8) in;\n");
    w.push_str("layout(set = 0, binding = 0, std140) uniform WobP {\n");
    w.push_str("  vec4 c0; vec4 c1; vec4 c2; ivec4 c3; vec4 c4; vec4 c5;\n");
    for (i, p) in params.iter().enumerate() {
        let t = if matches!(p.ty, ParamTy::Int | ParamTy::Bool) { "ivec4" } else { "vec4" };
        let _ = write!(w, "  {t} c{};", BUILTIN_CELLS + i);
    }
    w.push_str("\n} wob_P;\n");
    w.push_str("layout(set = 0, binding = 1) uniform sampler wob_samp;\n");
    let mut b = 2;
    for i in inputs {
        let _ = writeln!(w, "layout(set = 0, binding = {b}) uniform texture2D {};", i.name);
        b += 1;
    }
    for (i, name) in slots.iter().enumerate() {
        let _ = writeln!(w, "layout(set = 0, binding = {}) uniform texture2D {name};", b + i);
    }
    b += n_slots;
    let _ = writeln!(w, "layout(set = 0, binding = {b}, rgba16f) uniform writeonly image2D wob_out;");
    w.push_str(
        "vec4 wob_FragCoord; vec4 wob_FragColor; vec4 wob_TexCoord;\n\
         vec4 wob_tex(texture2D t, vec2 uv) { return textureLod(sampler2D(t, wob_samp), vec2(uv.x, 1.0 - uv.y), 0.0); }\n\
         vec4 wob_tex(texture2D t, vec2 uv, float b) { return wob_tex(t, uv); }\n\
         vec4 wob_lod(texture2D t, vec2 uv, float l) { if (l <= 0.0) return wob_tex(t, uv); \
         vec2 d = pow(2.0, l) / vec2(textureSize(sampler2D(t, wob_samp), 0)); vec4 a = vec4(0.0); \
         for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++) \
         a += wob_tex(t, uv + (vec2(float(x), float(y)) - 1.5) * 0.25 * d); return a / 16.0; }\n\
         ivec2 wob_size(texture2D t, int l) { return textureSize(sampler2D(t, wob_samp), l); }\n\
         vec4 wob_fetch(texture2D t, ivec2 p, int l) { ivec2 s = textureSize(sampler2D(t, wob_samp), 0); \
         return texelFetch(sampler2D(t, wob_samp), ivec2(p.x, s.y - 1 - p.y), 0); }\n",
    );
    for (port, i) in inputs.iter().enumerate() {
        w.push_str(&sample_fn(i, port));
    }

    // Uniforms this pass declared.
    let mut init = String::new();
    for u in &scan.uniforms {
        if u.ty == "sampler2D" {
            continue;
        }
        if let Some((ty, expr)) = builtin_expr(&u.name, inputs) {
            let _ = writeln!(w, "{ty} {};", u.name);
            let _ = writeln!(init, "  {} = {expr};", u.name);
        } else if let Some(i) = params.iter().position(|p| p.name == u.name) {
            let c = BUILTIN_CELLS + i;
            let p = &params[i];
            let (ty, e) = match p.ty {
                ParamTy::Float => ("float", format!("wob_P.c{c}.x")),
                ParamTy::Int => ("int", format!("wob_P.c{c}.x")),
                ParamTy::Bool => ("bool", format!("wob_P.c{c}.x != 0")),
                ParamTy::Vec2 => ("vec2", format!("wob_P.c{c}.xy")),
                ParamTy::Color => ("vec3", format!("wob_P.c{c}.xyz")),
                ParamTy::Vec4 => ("vec4", format!("wob_P.c{c}")),
            };
            let _ = writeln!(w, "{ty} {};", u.name);
            let _ = writeln!(init, "  {} = {e};", u.name);
        } else {
            // Arrays, matrices, unknown adsk_ values: declared, left zero.
            let arr = if u.array { "[64]" } else { "" };
            let _ = writeln!(w, "{} {}{arr};", u.ty, u.name);
            let note = format!("`{}` has no source here and stays 0", u.name);
            if !notes.contains(&note) {
                notes.push(note);
            }
        }
    }
    for (ty, name) in &scan.varyings {
        let _ = writeln!(w, "{ty} {name};");
        if ty == "vec2" {
            let _ = writeln!(init, "  {name} = wob_TexCoord.xy;");
        }
    }
    let (mut rendered, sites) = rewrite(scan, inputs);
    let hoisted: Vec<(String, String)> = scan
        .hoisted
        .iter()
        .map(|(n, r)| (n.clone(), rendered[r.clone()].concat()))
        .collect();
    for &t in &scan.hidden {
        rendered[t] = " ".into();
    }
    let body: String = rendered.concat();
    if scan.toks.iter().any(|t| matches!(t.text.as_str(), "texture2DLod" | "textureLod")) {
        let note = "mip lookups are approximated with a 16-tap box".to_string();
        if !notes.contains(&note) {
            notes.push(note);
        }
    }
    for name in ADSK_FNS {
        if body.contains(name) {
            w.push_str(adsk_fn(name).unwrap());
            if ADSK_APPROX.contains(name) {
                let note = format!("`{name}` is an approximation of Flame's");
                if !notes.contains(&note) {
                    notes.push(note);
                }
            }
        }
    }
    if let Some(name) = scan.undefined.first() {
        errors.push(ShaderError {
            line: None,
            msg: format!("`{name}` is a Flame function Wobble does not have"),
        });
        return None;
    }
    // A compute shader has no derivatives. The shader is run again one pixel
    // right and one pixel up, each `dFdx` site recording its value, and the
    // real run subtracts: true forward differences, at three times the cost.
    if sites > 0 {
        let _ = writeln!(w, "int wob_dmode; vec4 wob_dxv[{sites}]; vec4 wob_dyv[{sites}];");
        for (ty, wide, back) in [
            ("float", "vec4(v, 0.0, 0.0, 0.0)", ".x"),
            ("vec2", "vec4(v, 0.0, 0.0)", ".xy"),
            ("vec3", "vec4(v, 0.0)", ".xyz"),
            ("vec4", "v", ""),
        ] {
            let _ = writeln!(
                w,
                "{ty} wob_ddx(int k, {ty} v) {{ if (wob_dmode == 1) wob_dxv[k] = {wide}; return wob_dxv[k]{back} - v; }}\n\
                 {ty} wob_ddy(int k, {ty} v) {{ if (wob_dmode == 2) wob_dyv[k] = {wide}; return wob_dyv[k]{back} - v; }}\n\
                 {ty} wob_fw(int k, {ty} v) {{ return abs(wob_ddx(k, v)) + abs(wob_ddy(k, v)); }}"
            );
        }
        let note = "uses screen derivatives: runs three times per pixel".to_string();
        if !notes.contains(&note) {
            notes.push(note);
        }
    }

    let prelude_lines = w.matches('\n').count();
    w.push_str(&body);
    if !w.ends_with('\n') {
        w.push('\n');
    }
    w.push_str("void wob_run(ivec2 wob_px, ivec2 wob_dim, float dx, float dy) {\n");
    w.push_str(
        "  wob_FragCoord = vec4(float(wob_px.x) + 0.5 + dx, float(wob_dim.y - wob_px.y) - 0.5 + dy, 0.0, 1.0);\n\
         wob_TexCoord = vec4(wob_FragCoord.xy / max(wob_P.c0.xy, vec2(1.0)), 0.0, 1.0);\n",
    );
    w.push_str(&init);
    for (name, init) in &hoisted {
        let _ = writeln!(w, "  {name} = {init};");
    }
    w.push_str("  wob_FragColor = vec4(0.0, 0.0, 0.0, 1.0);\n  wob_main();\n}\n");
    w.push_str(
        "void main() {\n  ivec2 wob_px = ivec2(gl_GlobalInvocationID.xy);\n\
         ivec2 wob_dim = imageSize(wob_out);\n\
         if (wob_px.x >= wob_dim.x || wob_px.y >= wob_dim.y) return;\n",
    );
    if sites > 0 {
        w.push_str(
            "  wob_dmode = 1; wob_run(wob_px, wob_dim, 1.0, 0.0);\n\
             wob_dmode = 2; wob_run(wob_px, wob_dim, 0.0, 1.0);\n\
             wob_dmode = 0;\n",
        );
    }
    w.push_str("  wob_run(wob_px, wob_dim, 0.0, 0.0);\n");
    // Matchbox output is RGB unless the preset provides a matte: whatever a
    // shader leaves in alpha is not an image's alpha.
    if keep_alpha {
        match inputs.first() {
            Some(i) if pass + 1 == n_passes => {
                let _ = writeln!(
                    w,
                    "  wob_FragColor.a = (wob_P.c3.z & 1) != 0 ? texelFetch(sampler2D({}, wob_samp), wob_px, 0).a : 1.0;",
                    i.name
                );
            }
            _ => {}
        }
    }
    w.push_str("  imageStore(wob_out, wob_px, wob_FragColor);\n}\n");

    let remap = |l: usize| (l > prelude_lines).then(|| l - prelude_lines + line0 - 1);
    let opts = naga::front::glsl::Options::from(naga::ShaderStage::Compute);
    let mut module = match naga::front::glsl::Frontend::default().parse(&opts, &w) {
        Ok(m) => m,
        Err(e) => {
            for err in e.errors.iter().take(8) {
                let line = err.meta.location(&w).line_number as usize;
                errors.push(ShaderError { line: remap(line), msg: err.kind.to_string() });
            }
            return None;
        }
    };
    for ep in &mut module.entry_points {
        ep.name = ShaderDef::entry(pass);
    }
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    );
    let info = match v.validate(&module) {
        Ok(i) => i,
        Err(e) => {
            errors.push(ShaderError { line: None, msg: e.as_inner().to_string() });
            return None;
        }
    };
    match naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty()) {
        // The writer's namer guards names ending in a digit with `_`.
        Ok(wgsl) => {
            let entry = ShaderDef::entry(pass);
            Some(PassOut { wgsl: wgsl.replace(&format!("fn {entry}_("), &format!("fn {entry}(")) })
        }
        Err(e) => {
            errors.push(ShaderError { line: None, msg: e.to_string() });
            None
        }
    }
}

/// Compile the GLSL dialect into the same `ShaderDef` WGSL produces.
pub fn compile(src: &str) -> ShaderDef {
    let (xml, sections) = split(src);
    let preset = xml.as_deref().map(parse_preset).unwrap_or_default();
    let mut errors = Vec::new();
    let mut notes = Vec::new();

    let scans: Vec<(Scan, usize)> = sections
        .iter()
        .map(|(text, line0)| (scan(text), *line0))
        .collect();
    let n_passes = scans.len();
    for (i, (s, line0)) in scans.iter().enumerate() {
        let has_main = s.toks.iter().any(|t| t.text == "wob_main");
        if !has_main {
            errors.push(ShaderError {
                line: Some(*line0),
                msg: format!("pass {} has no `void main()`", i + 1),
            });
        }
    }

    // Inputs: every sampler that is not another pass's result.
    let xml_of = |name: &str| preset.uniforms.iter().find(|u| u.name == name);
    let mut inputs: Vec<(Option<i32>, usize, Input)> = Vec::new();
    for (s, _) in &scans {
        for u in s.uniforms.iter().filter(|u| u.ty == "sampler2D") {
            if result_index(&u.name).is_some()
                || u.name.starts_with("adsk_")
                || inputs.iter().any(|i| i.2.name == u.name)
            {
                continue;
            }
            let x = xml_of(&u.name);
            let order = inputs.len();
            inputs.push((
                x.and_then(|x| x.index),
                order,
                Input {
                    name: u.name.clone(),
                    matte: x.is_some_and(|x| x.input_type.eq_ignore_ascii_case("matte")),
                    white: x.is_some_and(|x| x.white),
                    wrap: x.map_or(Wrap::Clamp, |x| x.wrap),
                    nearest: x.is_some_and(|x| x.nearest),
                },
            ));
        }
    }
    inputs.sort_by_key(|(idx, order, _)| (idx.unwrap_or(i32::MAX), *order));
    let mut inputs: Vec<Input> = inputs.into_iter().map(|x| x.2).collect();
    if inputs.len() > MAX_INPUTS {
        errors.push(ShaderError {
            line: None,
            msg: format!("{} image inputs, at most {MAX_INPUTS}", inputs.len()),
        });
    }
    let generator = inputs.is_empty();
    if generator {
        // Same rule as WGSL: every node gets a port. The binding is declared
        // under a name no shader uses.
        inputs.push(Input { name: "wob_in".into(), matte: false, white: false, wrap: Wrap::Clamp, nearest: false });
    }

    // Params: the union across passes, ordered as the preset lays them out.
    let mut params: Vec<(i32, i32, i32, usize, ShaderParam)> = Vec::new();
    for (s, _) in &scans {
        for u in &s.uniforms {
            if u.array || u.ty == "sampler2D" || builtin_expr(&u.name, &inputs).is_some() {
                continue;
            }
            if u.name.starts_with("adsk_") || params.iter().any(|p| p.4.name == u.name) {
                continue;
            }
            let Some(ty) = glsl_scalar_ty(&u.ty) else { continue };
            let x = xml_of(&u.name);
            let order = params.len();
            let key = x.map_or((i32::MAX, 0, 0), |x| (x.page, x.col, x.row));
            params.push((key.0, key.1, key.2, order, make_param(u, ty, x, &preset)));
        }
    }
    params.sort_by_key(|p| (p.0, p.1, p.2, p.3));
    let params: Vec<ShaderParam> = params.into_iter().map(|p| p.4).collect();

    // Texture slots each pass reads beyond the inputs: other passes' results,
    // and Flame-only images, which read as black.
    let slots: Vec<Vec<String>> = scans
        .iter()
        .map(|(s, _)| {
            let mut v: Vec<String> = Vec::new();
            for u in s.uniforms.iter().filter(|u| u.ty == "sampler2D") {
                let other = result_index(&u.name).is_some() || u.name.starts_with("adsk_");
                if other && !v.contains(&u.name) {
                    v.push(u.name.clone());
                }
            }
            v
        })
        .collect();
    for name in slots.iter().flatten().filter(|n| result_index(n).is_none()) {
        let note = format!("`{name}` is not available here and reads as black");
        if !notes.contains(&note) {
            notes.push(note);
        }
    }
    let n_slots = slots.iter().map(Vec::len).max().unwrap_or(0);
    let executions: Vec<u32> = (0..n_passes)
        .map(|i| {
            preset.repeats.iter().find(|(p, _)| *p == i).map_or(1, |(_, n)| *n)
        })
        .collect();
    let steps = plan_steps(&executions, &slots);
    if steps.len() > MAX_STEPS {
        errors.push(ShaderError {
            line: None,
            msg: format!("{} passes with repeats, at most {MAX_STEPS}", steps.len()),
        });
    }

    let mut pass_wgsl = Vec::new();
    if errors.is_empty() {
        for (i, (s, line0)) in scans.iter().enumerate() {
            let keep = !preset.matte_provider && !generator;
            if let Some(p) = build_pass(
                i, n_passes, s, *line0, &inputs, &slots[i], n_slots, &params, keep, &mut errors,
                &mut notes,
            ) {
                // Our own output must survive the path wgpu will take.
                if let Err(e) = usershader::validate(&p.wgsl, usize::MAX) {
                    errors.extend(e);
                }
                pass_wgsl.push(p.wgsl);
            }
        }
    }

    let label = if preset.name.is_empty() { "GLSL".to_string() } else { preset.name.clone() };
    ShaderDef {
        label,
        inputs: inputs.into_iter().map(|i| i.name).collect(),
        passes: steps.len().max(1),
        params,
        wgsl: pass_wgsl.first().cloned().unwrap_or_default(),
        pass_wgsl,
        extra_sources: n_slots,
        steps,
        notes,
        non_commercial: preset.non_commercial,
        errors,
        hash: usershader::source_hash(src),
    }
}

/// Lay the passes out as dispatches, and give each one a texture to write,
/// reusing any whose result nothing will read again. A 29-pass shader at 4K
/// would otherwise hold 2 GB of results it mostly never looks at twice.
fn plan_steps(executions: &[u32], slots: &[Vec<String>]) -> Vec<PassStep> {
    // (logical pass, execution), in order.
    let order: Vec<(usize, u32)> = executions
        .iter()
        .enumerate()
        .flat_map(|(l, n)| (0..*n).map(move |e| (l, e)))
        .collect();
    // Which dispatch each slot reads: the latest run of that pass so far. A
    // pass reading itself sees its previous execution, and black on its first.
    let reads: Vec<Vec<Option<usize>>> = order
        .iter()
        .enumerate()
        .map(|(p, (l, _))| {
            slots[*l]
                .iter()
                .map(|name| {
                    let want = result_index(name)?.checked_sub(1)?;
                    (0..p).rev().find(|&q| order[q].0 == want)
                })
                .collect()
        })
        .collect();
    let mut last_use: Vec<usize> = (0..order.len()).collect();
    for (p, r) in reads.iter().enumerate() {
        for q in r.iter().flatten() {
            last_use[*q] = p;
        }
    }
    let mut target = vec![0usize; order.len()];
    let mut live: Vec<(usize, usize)> = Vec::new(); // (texture, last reader)
    let mut free: Vec<usize> = Vec::new();
    let mut count = 0;
    for p in 0..order.len() {
        live.retain(|&(t, until)| {
            let keep = until >= p;
            if !keep {
                free.push(t);
            }
            keep
        });
        target[p] = free.pop().unwrap_or_else(|| {
            count += 1;
            count - 1
        });
        live.push((target[p], last_use[p]));
    }
    // The compositor takes the LAST texture as the node's output.
    if let Some(&out) = target.last() {
        let top = count - 1;
        for t in &mut target {
            *t = if *t == out { top } else if *t == top { out } else { *t };
        }
    }
    order
        .iter()
        .enumerate()
        .map(|(p, (l, e))| PassStep {
            module: *l,
            execution: *e,
            target: target[p],
            reads: reads[p].iter().map(|r| r.map(|q| target[q])).collect(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(src: &str) -> std::sync::Arc<ShaderDef> {
        let d = usershader::compile(src);
        assert!(d.ok(), "{:?}", d.errors);
        d
    }

    #[test]
    fn starter_compiles() {
        let d = ok(STARTER);
        assert_eq!(d.inputs, vec!["front"]);
        assert_eq!(d.params.len(), 1);
        assert_eq!(d.params[0].name, "amount");
    }

    #[test]
    fn wgsl_is_not_mistaken_for_glsl() {
        assert!(!is_glsl(usershader::STARTER));
        assert!(is_glsl("void main() { gl_FragColor = vec4(1.0); }"));
    }
}
