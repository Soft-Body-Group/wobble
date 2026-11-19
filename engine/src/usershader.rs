//! User-authored shader nodes — the Matchbox-shaped hole in the node set.
//!
//! A Shader node is WGSL the user types into the app. Unlike a Tier 0 plugin
//! (`crate::plugins`), which is one expression fused into the main pass, this
//! node gets its **own passes and its own textures**: every image input is
//! resolved to a real texture first, so the shader may sample any input at
//! any coordinate, as many times as it likes, and may run up to
//! [`MAX_PASSES`] passes that read each other's results. That is exactly the
//! contract Flame's Matchbox offers, and it is what a blur, a warp, a
//! kaleidoscope or an iterative solver actually needs.
//!
//! It is therefore a SLOW PATH by construction — resolving inputs costs a
//! full-canvas pass each, and the shader never fuses with its neighbours.
//! Manifesto rule 8 applies: the node is visibly its own thing, not a quiet
//! detour inside the fused pass.
//!
//! # The dialect
//!
//! Metadata rides in `//!` pragma lines at the head of the source, so a
//! shader is ONE file with no sidecar XML:
//!
//! ```text
//! //! name: Radial Blur
//! //! input: front
//! //! passes: 2
//! //! group: Shape
//! //! param: float amount = 0.5 [0, 2] "Amount"
//! //! param: vec2 centre = (0.5, 0.5)
//! //! param: color tint = (1, 1, 1)
//! //! param: bool invert = false
//! //! param: int steps = 16 [1, 64]
//!
//! fn pass1(uv: vec2f, frag: vec2f) -> vec4f {
//!     return tex(front, uv) * P.amount;
//! }
//! fn pass2(uv: vec2f, frag: vec2f) -> vec4f {
//!     return tex(result1, uv);
//! }
//! ```
//!
//! Params land in a generated uniform struct under their own names (`P.amount`),
//! so the port from a Matchbox `uniform float Amount;` is one `P.` prefix.
//! Inputs and previous-pass results are texture bindings named by the
//! manifest (`front`) and by index (`result1`), sampled through the helpers
//! below. Every param is a uniform: moving a slider NEVER recompiles.
//!
//! # Failure is not fatal
//!
//! The source is parsed and naga-validated here, before it can become
//! topology. A shader that does not compile leaves the node a PASSTHROUGH
//! with its errors attached — the comp keeps rendering while you type, which
//! is the only way a live text editor over a live comp is usable at all.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

/// Image inputs a Shader node may declare. Matches the 4 the graph's port UI
/// can lay out, and each one costs a full-canvas resolve pass.
pub const MAX_INPUTS: usize = 4;

/// Passes a Shader node may run. Each is a full-canvas texture kept live for
/// the whole chain (pass 4 may read pass 1), so this is a memory cap as much
/// as a time one: 4 passes at 4K is ~283 MB of rgba16f.
pub const MAX_PASSES: usize = 4;

/// Built-in uniform names. A param may not take one of these.
pub const RESERVED: &[&str] =
    &["res", "time", "frame", "pass_index", "aspect", "luma_w"];

/// Bytes every uniform member occupies. Every member is `@align(16)`, so the
/// CPU side is a flat array of 16-byte cells and the layout rules cannot be
/// got subtly wrong — which, for a struct whose shape is authored by the
/// user at runtime, is worth 12 wasted bytes per scalar.
pub const CELL: usize = 16;

/// Built-in cells ahead of the user's params (res, time, frame, pass_index,
/// aspect, luma_w).
pub const BUILTIN_CELLS: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamTy {
    Float,
    Int,
    Bool,
    Vec2,
    /// vec3 shown as a colour swatch.
    Color,
    Vec4,
}

impl ParamTy {
    pub fn wgsl(self) -> &'static str {
        match self {
            ParamTy::Float => "f32",
            ParamTy::Int => "i32",
            ParamTy::Bool => "u32",
            ParamTy::Vec2 => "vec2f",
            ParamTy::Color => "vec3f",
            ParamTy::Vec4 => "vec4f",
        }
    }

    /// How many floats of the value cell this type actually uses.
    pub fn lanes(self) -> usize {
        match self {
            ParamTy::Vec2 => 2,
            ParamTy::Color => 3,
            ParamTy::Vec4 => 4,
            _ => 1,
        }
    }

    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "float" | "f32" => ParamTy::Float,
            "int" | "i32" => ParamTy::Int,
            "bool" => ParamTy::Bool,
            "vec2" | "vec2f" => ParamTy::Vec2,
            "color" | "colour" | "vec3" | "vec3f" => ParamTy::Color,
            "vec4" | "vec4f" => ParamTy::Vec4,
            _ => return None,
        })
    }
}

/// One dispatch of a GLSL node. A pass a preset repeats is several of these.
#[derive(Clone, Debug, PartialEq)]
pub struct PassStep {
    /// Index into `ShaderDef::pass_wgsl`.
    pub module: usize,
    /// Which repeat of its pass this is (`adsk_result_execution`).
    pub execution: u32,
    /// Result texture written. Textures are shared between steps whose
    /// results are never needed at the same time.
    pub target: usize,
    /// Result texture behind each extra binding; `None` binds the dummy.
    pub reads: Vec<Option<usize>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShaderParam {
    pub name: String,
    pub label: String,
    pub ty: ParamTy,
    /// Authored default, always 4 floats (unused lanes are 0).
    pub default: [f32; 4],
    pub min: f32,
    pub max: f32,
    /// Properties-panel grouping, from the preceding `group:` pragma.
    pub group: Option<String>,
    /// Named values for an int (a Matchbox popup). Empty for a plain number.
    pub choices: Vec<(String, i32)>,
}

/// One diagnostic, with a line number in the USER's source (1-based) where
/// one is known — naga reports against the generated text, and a line number
/// pointing at a prelude the user never wrote is worse than none at all.
#[derive(Clone, Debug, PartialEq)]
pub struct ShaderError {
    pub line: Option<usize>,
    pub msg: String,
}

impl std::fmt::Display for ShaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.line {
            Some(l) => write!(f, "line {l}: {}", self.msg),
            None => write!(f, "{}", self.msg),
        }
    }
}

/// A parsed, validated shader: everything the engine needs to build passes,
/// and everything the UI needs to build widgets.
#[derive(Clone, Debug)]
pub struct ShaderDef {
    pub label: String,
    /// Declared image input names, in port order. Port 0 is what a bypassed
    /// (or broken) node passes through.
    pub inputs: Vec<String>,
    pub passes: usize,
    pub params: Vec<ShaderParam>,
    /// The full generated WGSL — prelude, user source, entry points.
    pub wgsl: String,
    /// One module per pass, for the GLSL dialect, whose passes are separate
    /// translation units. Empty for WGSL, where `wgsl` holds every pass.
    pub pass_wgsl: Vec<String>,
    /// The dispatches of a GLSL node, `passes` long. Empty for WGSL.
    pub steps: Vec<PassStep>,
    /// Texture bindings after the inputs, for a GLSL node.
    pub extra_sources: usize,
    /// Things the user should know that are not errors.
    pub notes: Vec<String>,
    /// The shader's own preset says it is not cleared for commercial use.
    pub non_commercial: bool,
    /// Errors from pragma parsing or naga. Non-empty means the node renders
    /// as a passthrough and `wgsl` must not be compiled.
    pub errors: Vec<ShaderError>,
    /// Identity of the BEHAVIOR, part of the topology hash.
    pub hash: u64,
}

impl ShaderDef {
    pub fn ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// Uniform buffer size for this shader.
    pub fn uniform_bytes(&self) -> u64 {
        ((BUILTIN_CELLS + self.params.len()) * CELL) as u64
    }

    /// Texture bindings one pass sees: every declared input, then one slot
    /// per pass result. Constant across passes so all passes share a layout
    /// and a module — the pass writing `result_i` binds a dummy there.
    pub fn source_count(&self) -> usize {
        if self.steps.is_empty() {
            self.inputs.len() + self.passes
        } else {
            self.inputs.len() + self.extra_sources
        }
    }

    /// Full-canvas result textures the node keeps.
    pub fn result_textures(&self) -> usize {
        self.steps.iter().map(|s| s.target + 1).max().unwrap_or(self.passes)
    }

    /// Every distinct WGSL module.
    pub fn modules(&self) -> Vec<&str> {
        if self.pass_wgsl.is_empty() {
            vec![self.wgsl.as_str()]
        } else {
            self.pass_wgsl.iter().map(String::as_str).collect()
        }
    }

    pub fn param(&self, name: &str) -> Option<&ShaderParam> {
        self.params.iter().find(|p| p.name == name)
    }

    /// Entry point name for pass `i` (0-based).
    pub fn entry(i: usize) -> String {
        format!("wob_pass{i}")
    }

    /// Pack the uniform buffer: built-ins, then one 16-byte cell per param
    /// taken from `values` (falling back to the authored default).
    ///
    /// `res` is the output size in pixels, `time`/`frame` the clock reaching
    /// this node, `pass` which pass is about to run.
    #[allow(clippy::too_many_arguments)]
    pub fn pack(
        &self,
        values: &std::collections::BTreeMap<String, [f32; 4]>,
        res: [f32; 2],
        time: f32,
        frame: f32,
        pass: usize,
        luma_w: [f32; 3],
    ) -> Vec<u8> {
        let mut out = vec![0u8; (BUILTIN_CELLS + self.params.len()) * CELL];
        let put_f = |out: &mut [u8], cell: usize, lane: usize, v: f32| {
            let o = cell * CELL + lane * 4;
            out[o..o + 4].copy_from_slice(&v.to_le_bytes());
        };
        let put_i = |out: &mut [u8], cell: usize, v: i32| {
            let o = cell * CELL;
            out[o..o + 4].copy_from_slice(&v.to_le_bytes());
        };
        put_f(&mut out, 0, 0, res[0]);
        put_f(&mut out, 0, 1, res[1]);
        put_f(&mut out, 1, 0, time);
        put_f(&mut out, 2, 0, frame);
        put_i(&mut out, 3, pass as i32);
        if let Some(s) = self.steps.get(pass) {
            out[3 * CELL + 4..3 * CELL + 8].copy_from_slice(&(s.execution as i32).to_le_bytes());
        }
        put_f(&mut out, 4, 0, if res[1] > 0.0 { res[0] / res[1] } else { 1.0 });
        // Luminance weights of the comp's WORKING primaries, not a constant:
        // `luma()` has to mean the same thing here as it does in every
        // built-in node, and the working space is a graph-level setting the
        // shader cannot see at compile time.
        for (l, w) in luma_w.iter().enumerate() {
            put_f(&mut out, 5, l, *w);
        }
        for (i, p) in self.params.iter().enumerate() {
            let cell = BUILTIN_CELLS + i;
            let v = values.get(&p.name).copied().unwrap_or(p.default);
            match p.ty {
                // Ints and bools live in the buffer as their WGSL type, so
                // the shader can compare them without a cast.
                ParamTy::Int => put_i(&mut out, cell, v[0].round() as i32),
                ParamTy::Bool => put_i(&mut out, cell, (v[0] > 0.5) as i32),
                _ => {
                    for l in 0..p.ty.lanes() {
                        put_f(&mut out, cell, l, v[l]);
                    }
                }
            }
        }
        out
    }

    /// Record which inputs have a wire (bit per port) in a packed buffer. A
    /// GLSL shader needs it: its preset can ask for white when nothing is
    /// connected.
    pub fn set_wired(bytes: &mut [u8], mask: u32) {
        bytes[3 * CELL + 8..3 * CELL + 12].copy_from_slice(&mask.to_le_bytes());
    }

    /// Defaults for every declared param, for a node that has never been
    /// edited (or has just gained a param).
    pub fn defaults(&self) -> std::collections::BTreeMap<String, [f32; 4]> {
        self.params.iter().map(|p| (p.name.clone(), p.default)).collect()
    }
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// Source-text hash — the node's topology tag. Cheap enough to call from
/// `topology_hash` on every graph compare; `compile` is not.
pub fn source_hash(src: &str) -> u64 {
    fnv64(src.as_bytes())
}

/// Parse + generate + validate, memoized on the source text.
///
/// Called from codegen and from the UI on every frame a Shader node is
/// selected, so the memo is not an optimization detail: naga validation is
/// milliseconds and the graph is compared many times a frame.
pub fn compile(src: &str) -> Arc<ShaderDef> {
    static MEMO: OnceLock<RwLock<HashMap<u64, Arc<ShaderDef>>>> = OnceLock::new();
    let memo = MEMO.get_or_init(Default::default);
    let key = source_hash(src);
    if let Some(d) = memo.read().unwrap().get(&key) {
        return d.clone();
    }
    let def = Arc::new(compile_uncached(src));
    let mut w = memo.write().unwrap();
    // A runaway memo would be a leak in a long session of live typing; the
    // cost of a miss is one parse, so a crude cap is the right trade.
    if w.len() > 512 {
        w.clear();
    }
    w.insert(key, def.clone());
    def
}

fn compile_uncached(src: &str) -> ShaderDef {
    if crate::matchbox::is_glsl(src) {
        return crate::matchbox::compile(src);
    }
    let (mut def, user_line0) = parse(src);
    def.hash = source_hash(src);
    if def.errors.is_empty() {
        if let Err(errs) = validate(&def.wgsl, user_line0) {
            def.errors = errs;
        }
    }
    def
}

/// naga parse + validate over the generated text. Line numbers are remapped
/// back into the user's own source where they land in it.
pub(crate) fn validate(wgsl: &str, user_line0: usize) -> Result<(), Vec<ShaderError>> {
    let remap = |line: Option<usize>| -> Option<usize> {
        // Generated lines before the user block (prelude) and after it
        // (entry points) are ours, not theirs — report them unlocated
        // rather than pointing at a line the user cannot see.
        line.and_then(|l| (l > user_line0).then(|| l - user_line0))
    };
    let module = match naga::front::wgsl::parse_str(wgsl) {
        Ok(m) => m,
        Err(e) => {
            let line = e.location(wgsl).map(|l| l.line_number as usize);
            return Err(vec![ShaderError { line: remap(line), msg: e.message().to_string() }]);
        }
    };
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    );
    match v.validate(&module) {
        Ok(_) => Ok(()),
        Err(e) => {
            let line = e.location(wgsl).map(|l| l.line_number as usize);
            let msg = e
                .as_inner()
                .to_string()
                // naga's inner display is terse; the emitted form carries
                // the offending span, which is what a shader author needs.
                .to_string();
            Err(vec![ShaderError { line: remap(line), msg }])
        }
    }
}

/// Split the source into pragmas + body, and assemble the full WGSL.
/// Returns the def and the line offset at which the user's own text starts
/// in the generated output.
fn parse(src: &str) -> (ShaderDef, usize) {
    let mut label = String::new();
    let mut inputs: Vec<String> = Vec::new();
    let mut passes = 1usize;
    let mut params: Vec<ShaderParam> = Vec::new();
    let mut group: Option<String> = None;
    let mut errors: Vec<ShaderError> = Vec::new();

    for (i, raw) in src.lines().enumerate() {
        let line = raw.trim_start();
        let Some(rest) = line.strip_prefix("//!") else { continue };
        let rest = rest.trim();
        if rest.is_empty() {
            continue;
        }
        let n = i + 1;
        let (key, val) = match rest.split_once(':') {
            Some((k, v)) => (k.trim(), v.trim()),
            None => {
                errors.push(ShaderError {
                    line: Some(n),
                    msg: format!("pragma needs `key: value`, got `{rest}`"),
                });
                continue;
            }
        };
        match key {
            "name" | "label" => label = val.to_string(),
            "input" => {
                let name = val.split_whitespace().next().unwrap_or("").to_string();
                if !is_ident(&name) {
                    errors.push(ShaderError {
                        line: Some(n),
                        msg: format!("`{name}` is not a valid input name"),
                    });
                } else if inputs.len() >= MAX_INPUTS {
                    errors.push(ShaderError {
                        line: Some(n),
                        msg: format!("at most {MAX_INPUTS} inputs"),
                    });
                } else if inputs.contains(&name) {
                    errors.push(ShaderError {
                        line: Some(n),
                        msg: format!("duplicate input `{name}`"),
                    });
                } else {
                    inputs.push(name);
                }
            }
            "passes" => match val.trim().parse::<usize>() {
                Ok(p) if (1..=MAX_PASSES).contains(&p) => passes = p,
                _ => errors.push(ShaderError {
                    line: Some(n),
                    msg: format!("passes must be 1..={MAX_PASSES}, got `{val}`"),
                }),
            },
            "group" => group = (!val.is_empty()).then(|| val.to_string()),
            "param" => match parse_param(val, group.clone()) {
                Ok(p) => {
                    if RESERVED.contains(&p.name.as_str()) {
                        errors.push(ShaderError {
                            line: Some(n),
                            msg: format!("`{}` is a built-in uniform name", p.name),
                        });
                    } else if params.iter().any(|q| q.name == p.name) {
                        errors.push(ShaderError {
                            line: Some(n),
                            msg: format!("duplicate param `{}`", p.name),
                        });
                    } else {
                        params.push(p);
                    }
                }
                Err(msg) => errors.push(ShaderError { line: Some(n), msg }),
            },
            _ => errors.push(ShaderError {
                line: Some(n),
                msg: format!("unknown pragma `{key}`"),
            }),
        }
    }

    // A shader with no declared input still gets one port: a generator is a
    // legitimate shader, but a node you cannot wire anything into is a dead
    // end in the editor, and bypass needs something to pass through.
    if inputs.is_empty() {
        inputs.push("front".into());
    }
    if label.is_empty() {
        label = "Shader".into();
    }

    let body_has = |name: &str| contains_fn(src, name);
    if !body_has("pass1") && !body_has("main_image") {
        errors.push(ShaderError {
            line: None,
            msg: "no entry point: define `fn pass1(uv: vec2f, frag: vec2f) -> vec4f`".into(),
        });
    }
    for p in 2..=passes {
        if !body_has(&format!("pass{p}")) {
            errors.push(ShaderError {
                line: None,
                msg: format!("`passes: {passes}` but `fn pass{p}` is missing"),
            });
        }
    }

    let mut def = ShaderDef {
        label,
        inputs,
        passes,
        params,
        wgsl: String::new(),
        pass_wgsl: Vec::new(),
        steps: Vec::new(),
        extra_sources: 0,
        notes: Vec::new(),
        non_commercial: false,
        errors,
        hash: 0,
    };
    let (wgsl, user_line0) = generate(&def, src);
    def.wgsl = wgsl;
    (def, user_line0)
}

/// `param: <type> <name> = <default> [min, max] "Label"`
fn parse_param(spec: &str, group: Option<String>) -> Result<ShaderParam, String> {
    // The quoted label first, so its contents can never be mistaken for
    // syntax (a label with a bracket in it is not an error).
    let (spec, label) = match spec.split_once('"') {
        Some((head, tail)) => {
            let l = tail.split('"').next().unwrap_or("").to_string();
            (head.trim(), Some(l))
        }
        None => (spec.trim(), None),
    };
    // Then the range, which is the only other bracketed thing.
    let (spec, range) = match spec.split_once('[') {
        Some((head, tail)) => {
            let inner = tail.split(']').next().ok_or("unclosed `[`")?;
            let mut it = inner.split(',');
            let lo = it.next().unwrap_or("").trim();
            let hi = it.next().ok_or("range needs `[min, max]`")?.trim();
            let lo: f32 = lo.parse().map_err(|_| format!("bad range min `{lo}`"))?;
            let hi: f32 = hi.parse().map_err(|_| format!("bad range max `{hi}`"))?;
            (head.trim(), Some((lo, hi)))
        }
        None => (spec.trim(), None),
    };
    let (decl, default) = match spec.split_once('=') {
        Some((d, v)) => (d.trim(), v.trim()),
        None => (spec, ""),
    };
    let mut words = decl.split_whitespace();
    let ty_s = words.next().ok_or("param needs a type")?;
    let ty = ParamTy::parse(ty_s).ok_or_else(|| {
        format!("unknown type `{ty_s}` (float, int, bool, vec2, color)")
    })?;
    let name = words.next().ok_or("param needs a name")?.to_string();
    if !is_ident(&name) {
        return Err(format!("`{name}` is not a valid param name"));
    }
    let value = parse_default(ty, default)?;
    let (min, max) = range.unwrap_or(match ty {
        ParamTy::Int => (0.0, 100.0),
        ParamTy::Bool => (0.0, 1.0),
        ParamTy::Vec2 | ParamTy::Vec4 => (-1.0, 2.0),
        _ => (0.0, 1.0),
    });
    Ok(ShaderParam {
        label: label.unwrap_or_else(|| name.clone()),
        name,
        ty,
        default: value,
        min,
        max,
        group,
        choices: Vec::new(),
    })
}

fn parse_default(ty: ParamTy, s: &str) -> Result<[f32; 4], String> {
    let s = s.trim();
    if s.is_empty() {
        return Ok([0.0; 4]);
    }
    match ty {
        ParamTy::Bool => Ok(match s {
            "true" | "1" => [1.0, 0.0, 0.0, 0.0],
            "false" | "0" => [0.0; 4],
            _ => return Err(format!("bool default must be true/false, got `{s}`")),
        }),
        ParamTy::Float | ParamTy::Int => {
            let v: f32 = s.parse().map_err(|_| format!("bad default `{s}`"))?;
            Ok([v, 0.0, 0.0, 0.0])
        }
        ParamTy::Vec2 | ParamTy::Color | ParamTy::Vec4 => {
            let inner = s.trim_start_matches(['(', '{']).trim_end_matches([')', '}']);
            let mut out = [0.0f32; 4];
            let want = ty.lanes();
            let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
            // One scalar splats — `color tint = 1` is unambiguous and saves
            // typing the same number three times.
            if parts.len() == 1 {
                let v: f32 = parts[0].parse().map_err(|_| format!("bad default `{s}`"))?;
                for o in out.iter_mut().take(want) {
                    *o = v;
                }
                return Ok(out);
            }
            if parts.len() != want {
                return Err(format!("expected {want} components, got {}", parts.len()));
            }
            for (i, p) in parts.iter().enumerate() {
                out[i] = p.parse().map_err(|_| format!("bad component `{p}`"))?;
            }
            Ok(out)
        }
    }
}

fn is_ident(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with(|c: char| c.is_ascii_digit())
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Whether the source defines `fn <name>`. Deliberately textual: the point is
/// a useful error BEFORE naga sees a call to a function that isn't there.
fn contains_fn(src: &str, name: &str) -> bool {
    src.match_indices("fn ").any(|(i, _)| {
        let rest = src[i + 3..].trim_start();
        rest.strip_prefix(name)
            .is_some_and(|r| r.trim_start().starts_with('('))
    })
}

/// Assemble prelude + user source + entry points. Returns the text and the
/// number of generated lines preceding the user's own.
fn generate(def: &ShaderDef, body: &str) -> (String, usize) {
    use std::fmt::Write;
    let mut w = String::with_capacity(body.len() + 2048);

    w.push_str("struct WobParams {\n");
    w.push_str("    @align(16) res: vec2f,\n");
    w.push_str("    @align(16) time: f32,\n");
    w.push_str("    @align(16) frame: f32,\n");
    w.push_str("    @align(16) pass_index: i32,\n");
    w.push_str("    @align(16) aspect: f32,\n");
    w.push_str("    @align(16) luma_w: vec3f,\n");
    for p in &def.params {
        writeln!(w, "    @align(16) {}: {},", p.name, p.ty.wgsl()).unwrap();
    }
    w.push_str("};\n");
    w.push_str("@group(0) @binding(0) var<uniform> P: WobParams;\n");
    w.push_str("@group(0) @binding(1) var wob_samp: sampler;\n");
    let mut b = 2u32;
    for name in &def.inputs {
        writeln!(w, "@group(0) @binding({b}) var {name}: texture_2d<f32>;").unwrap();
        b += 1;
    }
    for i in 1..=def.passes {
        writeln!(w, "@group(0) @binding({b}) var result{i}: texture_2d<f32>;").unwrap();
        b += 1;
    }
    writeln!(
        w,
        "@group(0) @binding({b}) var wob_out: texture_storage_2d<rgba16float, write>;"
    )
    .unwrap();

    // Sampling helpers. `tex` is clamp-to-edge (the sampler's own mode), so
    // it behaves like a Matchbox `texture2D`; `tex_b` is the comp's own
    // convention, transparent black outside the frame. Having both named
    // means a port never silently changes its edge behaviour.
    w.push_str(
        r#"
fn tex_size(t: texture_2d<f32>) -> vec2f { return vec2f(textureDimensions(t)); }
fn tex(t: texture_2d<f32>, uv: vec2f) -> vec4f {
    return textureSampleLevel(t, wob_samp, uv, 0.0);
}
fn tex_b(t: texture_2d<f32>, uv: vec2f) -> vec4f {
    let inside = all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0));
    return select(vec4f(0.0), textureSampleLevel(t, wob_samp, uv, 0.0), inside);
}
fn tex_px(t: texture_2d<f32>, p: vec2f) -> vec4f { return tex(t, p / tex_size(t)); }
fn texel(t: texture_2d<f32>, p: vec2i) -> vec4f {
    let hi = vec2i(textureDimensions(t)) - vec2i(1);
    return textureLoad(t, clamp(p, vec2i(0), hi), 0);
}
fn luma(c: vec3f) -> f32 { return dot(c, P.luma_w); }
"#,
    );
    // `main_image` is the one-pass spelling; alias it so the entry points
    // below only ever have one name to call.
    if !contains_fn(body, "pass1") && contains_fn(body, "main_image") {
        w.push_str(
            "fn pass1(uv: vec2f, frag: vec2f) -> vec4f { return main_image(uv, frag); }\n",
        );
    }
    w.push_str("// ---- user source ----\n");
    let user_line0 = w.lines().count();
    w.push_str(body);
    if !body.ends_with('\n') {
        w.push('\n');
    }
    for i in 0..def.passes {
        writeln!(
            w,
            r#"
@compute @workgroup_size(8, 8)
fn {entry}(@builtin(global_invocation_id) gid: vec3u) {{
    let dims = textureDimensions(wob_out);
    if (gid.x >= dims.x || gid.y >= dims.y) {{ return; }}
    let frag = vec2f(gid.xy) + vec2f(0.5);
    let uv = frag / max(P.res, vec2f(1.0));
    textureStore(wob_out, gid.xy, pass{n}(uv, frag));
}}"#,
            entry = ShaderDef::entry(i),
            n = i + 1
        )
        .unwrap();
    }
    (w, user_line0)
}

/// What a fresh Shader node starts as: a working, editable shader rather
/// than an empty buffer and a compile error.
pub const STARTER: &str = r#"//! name: Shader
//! input: front
//! param: float amount = 1.0 [0, 4] "Amount"
//! param: color tint = (1, 1, 1) "Tint"

fn pass1(uv: vec2f, frag: vec2f) -> vec4f {
    let c = tex(front, uv);
    return vec4f(c.rgb * P.tint * P.amount, c.a);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starter_compiles() {
        let d = compile(STARTER);
        assert!(d.ok(), "{:?}", d.errors);
        assert_eq!(d.inputs, vec!["front"]);
        assert_eq!(d.passes, 1);
        assert_eq!(d.params.len(), 2);
        assert_eq!(d.params[0].ty, ParamTy::Float);
        assert_eq!(d.params[1].ty, ParamTy::Color);
        assert_eq!(d.params[1].default, [1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn params_land_in_named_uniform_fields() {
        let d = compile(STARTER);
        assert!(d.wgsl.contains("@align(16) amount: f32"), "{}", d.wgsl);
        assert!(d.wgsl.contains("@align(16) tint: vec3f"), "{}", d.wgsl);
    }

    #[test]
    fn bad_wgsl_is_an_error_not_a_panic() {
        let d = compile("//! input: front\nfn pass1(uv: vec2f, frag: vec2f) -> vec4f { return nope(uv); }");
        assert!(!d.ok());
        assert!(!d.errors.is_empty());
    }

    #[test]
    fn error_lines_point_at_user_source() {
        // Line 3 of the user's text is the broken one.
        let src = "//! input: front\n\nfn pass1(uv: vec2f, frag: vec2f) -> vec4f { return nope(uv); }\n";
        let d = compile(src);
        assert!(!d.ok());
        assert_eq!(d.errors[0].line, Some(3), "{:?}", d.errors);
    }

    #[test]
    fn multi_pass_declares_result_textures() {
        let src = r#"//! input: front
//! passes: 2
fn pass1(uv: vec2f, frag: vec2f) -> vec4f { return tex(front, uv); }
fn pass2(uv: vec2f, frag: vec2f) -> vec4f { return tex(result1, uv) * 2.0; }
"#;
        let d = compile(src);
        assert!(d.ok(), "{:?}", d.errors);
        assert_eq!(d.passes, 2);
        assert_eq!(d.source_count(), 1 + 2);
        assert!(d.wgsl.contains("var result1: texture_2d<f32>"));
        assert!(d.wgsl.contains("fn wob_pass1"));
    }

    #[test]
    fn missing_pass_fn_is_reported_before_naga() {
        let src = "//! input: front\n//! passes: 2\nfn pass1(uv: vec2f, frag: vec2f) -> vec4f { return tex(front, uv); }\n";
        let d = compile(src);
        assert!(!d.ok());
        assert!(d.errors.iter().any(|e| e.msg.contains("pass2")), "{:?}", d.errors);
    }

    #[test]
    fn main_image_is_accepted_for_one_pass() {
        let src = "//! input: front\nfn main_image(uv: vec2f, frag: vec2f) -> vec4f { return tex(front, uv); }\n";
        let d = compile(src);
        assert!(d.ok(), "{:?}", d.errors);
    }

    #[test]
    fn reserved_names_are_rejected() {
        let src = "//! param: float time = 1.0\nfn pass1(uv: vec2f, frag: vec2f) -> vec4f { return vec4f(1.0); }";
        let d = compile(src);
        assert!(d.errors.iter().any(|e| e.msg.contains("built-in")), "{:?}", d.errors);
    }

    #[test]
    fn param_spec_forms() {
        let p = parse_param(r#"float amount = 0.5 [0, 2] "The Amount""#, None).unwrap();
        assert_eq!(p.name, "amount");
        assert_eq!(p.label, "The Amount");
        assert_eq!(p.default[0], 0.5);
        assert_eq!((p.min, p.max), (0.0, 2.0));

        let p = parse_param("int steps = 16 [1, 64]", None).unwrap();
        assert_eq!(p.ty, ParamTy::Int);
        assert_eq!(p.default[0], 16.0);

        let p = parse_param("bool invert = true", None).unwrap();
        assert_eq!(p.default[0], 1.0);

        let p = parse_param("vec2 centre = (0.25, 0.75)", None).unwrap();
        assert_eq!([p.default[0], p.default[1]], [0.25, 0.75]);

        // One scalar splats a colour.
        let p = parse_param("color tint = 0.5", None).unwrap();
        assert_eq!([p.default[0], p.default[1], p.default[2]], [0.5, 0.5, 0.5]);
    }

    #[test]
    fn packing_matches_the_generated_layout() {
        let d = compile(STARTER);
        let mut vals = d.defaults();
        vals.insert("amount".into(), [2.0, 0.0, 0.0, 0.0]);
        let bytes = d.pack(&vals, [1920.0, 1080.0], 1.5, 36.0, 0, [0.3, 0.6, 0.1]);
        assert_eq!(bytes.len() as u64, d.uniform_bytes());
        let f = |cell: usize, lane: usize| {
            let o = cell * CELL + lane * 4;
            f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap())
        };
        assert_eq!((f(0, 0), f(0, 1)), (1920.0, 1080.0));
        assert_eq!(f(1, 0), 1.5); // time
        assert_eq!(f(2, 0), 36.0); // frame
        assert_eq!(f(4, 0), 1920.0 / 1080.0); // aspect
        assert_eq!((f(5, 0), f(5, 1), f(5, 2)), (0.3, 0.6, 0.1)); // luma_w
        assert_eq!(f(BUILTIN_CELLS, 0), 2.0); // amount
        assert_eq!(f(BUILTIN_CELLS + 1, 1), 1.0); // tint.g
    }

    #[test]
    fn ints_and_bools_pack_as_integers() {
        let src = "//! param: int steps = 7 [1, 16]\n//! param: bool flip = true\nfn pass1(uv: vec2f, frag: vec2f) -> vec4f { return vec4f(f32(P.steps) * f32(P.flip)); }";
        let d = compile(src);
        assert!(d.ok(), "{:?}", d.errors);
        let bytes = d.pack(&d.defaults(), [2.0, 1.0], 0.0, 0.0, 0, [0.3, 0.6, 0.1]);
        let i = |cell: usize| {
            let o = cell * CELL;
            i32::from_le_bytes(bytes[o..o + 4].try_into().unwrap())
        };
        assert_eq!(i(BUILTIN_CELLS), 7);
        assert_eq!(i(BUILTIN_CELLS + 1), 1);
    }

    #[test]
    fn pass_index_reaches_the_shader() {
        let d = compile(STARTER);
        let b = d.pack(&d.defaults(), [1.0, 1.0], 0.0, 0.0, 3, [0.3, 0.6, 0.1]);
        assert_eq!(i32::from_le_bytes(b[3 * CELL..3 * CELL + 4].try_into().unwrap()), 3);
    }

    #[test]
    fn a_generator_still_gets_a_port() {
        let src = "fn pass1(uv: vec2f, frag: vec2f) -> vec4f { return vec4f(uv, 0.0, 1.0); }";
        let d = compile(src);
        assert!(d.ok(), "{:?}", d.errors);
        assert_eq!(d.inputs.len(), 1);
    }

    #[test]
    fn source_hash_tracks_the_text() {
        assert_ne!(source_hash(STARTER), source_hash(&format!("{STARTER}\n// x")));
    }

    #[test]
    fn every_pass_entry_point_exists() {
        let src = r#"//! passes: 3
fn pass1(uv: vec2f, frag: vec2f) -> vec4f { return vec4f(1.0); }
fn pass2(uv: vec2f, frag: vec2f) -> vec4f { return tex(result1, uv); }
fn pass3(uv: vec2f, frag: vec2f) -> vec4f { return tex(result2, uv); }
"#;
        let d = compile(src);
        assert!(d.ok(), "{:?}", d.errors);
        for i in 0..3 {
            assert!(d.wgsl.contains(&format!("fn {}", ShaderDef::entry(i))), "{}", d.wgsl);
        }
    }
}
