//! Tier 0 plugin nodes: a node authored OUTSIDE the build as a WGSL
//! expression plus a JSON manifest. The registry maps a `type_id` to its
//! definition; `NodeKind::Custom` carries only the type id and param VALUES,
//! so a setup loaded without its plugin keeps every value and renders as a
//! passthrough instead of failing.
//!
//! The contract keeps the fast path the only path: a plugin emits WGSL into
//! the fused pass and declares float params (uniforms — edits never
//! recompile). There is no CPU pixel access. naga validation and the
//! failed-topology memo already fence off a broken shader.
//!
//! Expression substitutions:
//!   `$in` `$in2` `$in3` `$in4` — the wired inputs (vec4f, unwired = 0)
//!   `$<param>`                 — a declared param (f32 uniform)
//!   `$pos`                     — pixel coords (vec2f)
//!   `$size`                    — output size (vec2f)
//! `helpers` may define WGSL functions, emitted once per shader; prefix
//! their names with the type id to keep them collision-proof.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

#[derive(Clone, Debug, PartialEq)]
pub struct ParamSpec {
    pub name: String,
    pub label: String,
    pub default: f32,
    pub min: f32,
    pub max: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShaderNodeDef {
    pub type_id: String,
    pub label: String,
    pub color: [u8; 3],
    /// Image inputs (1..=4). Input 0 is what a bypass passes through.
    pub inputs: usize,
    /// Properties-panel width in whole tiles.
    pub tiles: usize,
    pub params: Vec<ParamSpec>,
    pub expr: String,
    pub helpers: String,
    /// Identity of the BEHAVIOR: hashes expr + helpers + input count + param
    /// names. Part of the topology hash, so editing a plugin recompiles its
    /// consumers and correctly invalidates the render cache.
    pub source_hash: u64,
    /// This is a TOOL (a graph macro), not a shader: its behaviour is an
    /// inner node graph the app expands into the flat graph before the
    /// engine ever sees the instance. Only the INTERFACE — arity, params,
    /// colour, source hash — is registered here, because that is all the
    /// engine needs to size ports, key channels and hash topology.
    ///
    /// `expr` is `$in` for a tool, so an instance that reaches codegen
    /// unexpanded (definition missing, expansion depth exceeded) degrades to
    /// a passthrough instead of emitting nonsense.
    pub tool: bool,
    /// The input that carries a MATTE, if any — a tool whose body has a
    /// matte input proxy. `NodeKind::mask_input` reads it, so the instance's
    /// socket is typed, drawn and defaulted (unwired = coverage 1) like
    /// every built-in node's mask port.
    pub mask_input: Option<usize>,
    /// The OUTPUT is a matte rather than an image: the matte input proxy
    /// inside a tool body, so it wires into a mask port there.
    pub mask_out: bool,
    /// The pipe each INPUT carries, where a port is not a plain picture —
    /// a subwob's ports are typed by its input nodes (image, matte, data,
    /// channels). Empty = images, with `mask_input` the matte.
    pub input_kinds: Vec<crate::graph::PortKind>,
    /// The pipe the OUTPUT carries when it is not a picture: a subwob's
    /// input node emits the pipe it is the portal for.
    pub out_kind: Option<crate::graph::PortKind>,
    /// The node FOLLOWS ITS PIPE: wired a matte, it takes it and gives a
    /// matte (an Invert of a matte is a matte), like the built-in pointwise
    /// ops (`NodeKind::follows_pipe`). Manifest `"mattes": true`; only
    /// sensible for an expression that treats r, g and b alike.
    pub mattes: bool,
}

fn fnv64(parts: &[&[u8]]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for p in parts {
        for b in *p {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
        h ^= 0xff; // part separator
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

impl ShaderNodeDef {
    pub fn new(
        type_id: String,
        label: String,
        color: [u8; 3],
        inputs: usize,
        tiles: usize,
        params: Vec<ParamSpec>,
        expr: String,
        helpers: String,
    ) -> Self {
        let names: Vec<u8> =
            params.iter().flat_map(|p| p.name.bytes().chain([0])).collect();
        let source_hash = fnv64(&[
            expr.as_bytes(),
            helpers.as_bytes(),
            &[inputs as u8],
            &names,
        ]);
        Self {
            type_id,
            label,
            color,
            inputs,
            tiles,
            params,
            expr,
            helpers,
            source_hash,
            tool: false,
            mask_input: None,
            mask_out: false,
            input_kinds: Vec::new(),
            out_kind: None,
            mattes: false,
        }
    }

    /// A CONTROL node: it produces VALUES down patch cables and has no image
    /// behaviour at all, so it emits transparent black. `new_tool`'s `$in`
    /// would be a live grenade here — a control declares no inputs, so `$in`
    /// would never be substituted and the shader would not compile if anyone
    /// wired one into an image chain.
    pub fn new_control(
        type_id: String,
        label: String,
        color: [u8; 3],
        params: Vec<ParamSpec>,
        source_hash: u64,
    ) -> Self {
        Self {
            type_id,
            label,
            color,
            inputs: 0,
            tiles: 1,
            params,
            expr: "vec4f(0.0)".into(),
            helpers: String::new(),
            source_hash,
            tool: true,
            mask_input: None,
            mask_out: false,
            input_kinds: Vec::new(),
            out_kind: None,
            mattes: false,
        }
    }

    /// Register a TOOL's interface. `body_hash` identifies the inner graph;
    /// it lands in `source_hash` so editing a tool recompiles its consumers
    /// and invalidates their cached renders exactly like editing a plugin's
    /// WGSL does.
    #[allow(clippy::too_many_arguments)]
    pub fn new_tool(
        type_id: String,
        label: String,
        color: [u8; 3],
        inputs: usize,
        tiles: usize,
        params: Vec<ParamSpec>,
        body_hash: u64,
    ) -> Self {
        let names: Vec<u8> =
            params.iter().flat_map(|p| p.name.bytes().chain([0])).collect();
        Self {
            type_id,
            label,
            color,
            inputs,
            tiles,
            params,
            // A leaked instance passes its first input through.
            expr: "$in".into(),
            helpers: String::new(),
            source_hash: body_hash ^ fnv64(&[&names, &[inputs as u8]]),
            tool: true,
            mask_input: None,
            mask_out: false,
            input_kinds: Vec::new(),
            out_kind: None,
            mattes: false,
        }
    }

    /// Declare input `i` a MATTE port. Part of the identity: a port changing
    /// type changes what may be wired to it.
    pub fn with_mask_input(mut self, i: Option<usize>) -> Self {
        if let Some(i) = i {
            self.source_hash ^= fnv64(&[b"mask-input", &[i as u8]]);
        }
        self.mask_input = i;
        self
    }

    /// Declare the output a MATTE (see `mask_out`).
    pub fn with_mask_out(mut self) -> Self {
        self.source_hash ^= fnv64(&[b"mask-out"]);
        self.mask_out = true;
        self
    }

    /// Declare each input's pipe (see `input_kinds`). Part of the identity:
    /// a port changing pipe changes what may be wired to it.
    pub fn with_input_kinds(mut self, kinds: Vec<crate::graph::PortKind>) -> Self {
        let code: Vec<u8> = kinds.iter().map(|k| *k as u8).collect();
        self.source_hash ^= fnv64(&[b"input-kinds", &code]);
        self.input_kinds = kinds;
        self
    }

    /// Declare that the node follows its pipe (see `mattes`). Part of the
    /// identity: it changes what may be wired to input 0.
    pub fn with_mattes(mut self, on: bool) -> Self {
        if on {
            self.source_hash ^= fnv64(&[b"mattes"]);
        }
        self.mattes = on;
        self
    }

    /// Declare the output's pipe (see `out_kind`).
    pub fn with_output_kind(mut self, k: crate::graph::PortKind) -> Self {
        self.source_hash ^= fnv64(&[b"out-kind", &[k as u8]]);
        self.mask_out = k == crate::graph::PortKind::Mask;
        self.out_kind = Some(k);
        self
    }

    /// Instantiate the expression for one node: inputs are already-emitted
    /// child expressions, params resolve to uniform accessors at `slot`.
    pub fn instantiate(&self, inputs: &[String], slot: usize, pos: &str) -> String {
        let mut s = self.expr.clone();
        // Longest names first so `$radius` never half-matches `$rad`.
        let mut subs: Vec<(String, String)> = Vec::new();
        for (k, p) in self.params.iter().enumerate() {
            let comp = ["x", "y", "z", "w"][k % 4];
            subs.push((
                format!("${}", p.name),
                format!("P.v[{}].{comp}", slot + k / 4),
            ));
        }
        for i in (0..self.inputs).rev() {
            let tag = if i == 0 { "$in".to_string() } else { format!("$in{}", i + 1) };
            let val = inputs.get(i).cloned().unwrap_or_else(|| "vec4f(0.0)".into());
            subs.push((tag, val));
        }
        subs.push(("$pos".into(), pos.to_string()));
        subs.push(("$size".into(), "P.v[0].xy".to_string()));
        subs.sort_by_key(|(k, _)| std::cmp::Reverse(k.len()));
        for (from, to) in &subs {
            s = s.replace(from.as_str(), to);
        }
        s
    }
}

fn registry() -> &'static RwLock<HashMap<String, Arc<ShaderNodeDef>>> {
    static REG: OnceLock<RwLock<HashMap<String, Arc<ShaderNodeDef>>>> = OnceLock::new();
    REG.get_or_init(Default::default)
}

pub fn register(def: ShaderNodeDef) {
    registry().write().unwrap().insert(def.type_id.clone(), Arc::new(def));
}

pub fn get(type_id: &str) -> Option<Arc<ShaderNodeDef>> {
    registry().read().unwrap().get(type_id).cloned()
}

/// Every registered definition, sorted by label (for menus).
pub fn all() -> Vec<Arc<ShaderNodeDef>> {
    let mut v: Vec<_> = registry().read().unwrap().values().cloned().collect();
    v.sort_by(|a, b| a.label.cmp(&b.label));
    v
}

pub fn clear() {
    registry().write().unwrap().clear();
}

#[cfg(feature = "serde")]
mod manifest {
    use super::*;
    use anyhow::{ensure, Context, Result};
    use std::path::Path;

    #[derive(serde::Deserialize)]
    struct ParamJson {
        name: String,
        #[serde(default)]
        label: Option<String>,
        #[serde(default)]
        default: f32,
        #[serde(default)]
        min: f32,
        #[serde(default = "one")]
        max: f32,
    }

    fn one() -> f32 {
        1.0
    }
    fn one_usize() -> usize {
        1
    }

    #[derive(serde::Deserialize)]
    struct ManifestJson {
        type_id: String,
        label: String,
        #[serde(default)]
        color: Option<[u8; 3]>,
        #[serde(default = "one_usize")]
        inputs: usize,
        #[serde(default = "one_usize")]
        tiles: usize,
        #[serde(default)]
        params: Vec<ParamJson>,
        /// Inline WGSL expression, or `expr_file` relative to the manifest.
        #[serde(default)]
        expr: Option<String>,
        #[serde(default)]
        expr_file: Option<String>,
        #[serde(default)]
        helpers: Option<String>,
        #[serde(default)]
        helpers_file: Option<String>,
        /// Takes a matte and gives one back (see `ShaderNodeDef::mattes`).
        #[serde(default)]
        mattes: bool,
    }

    /// Parse one `plugin.json` (WGSL possibly in sibling files).
    pub fn load_manifest(path: &Path) -> Result<ShaderNodeDef> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let m: ManifestJson = serde_json::from_str(&text)
            .with_context(|| format!("parsing {}", path.display()))?;
        let dir = path.parent().unwrap_or(Path::new("."));
        let file = |rel: &str| -> Result<String> {
            std::fs::read_to_string(dir.join(rel))
                .with_context(|| format!("reading {}", dir.join(rel).display()))
        };
        let expr = match (&m.expr, &m.expr_file) {
            (Some(e), _) => e.clone(),
            (None, Some(f)) => file(f)?,
            (None, None) => anyhow::bail!("manifest needs `expr` or `expr_file`"),
        };
        let helpers = match (&m.helpers, &m.helpers_file) {
            (Some(h), _) => h.clone(),
            (None, Some(f)) => file(f)?,
            (None, None) => String::new(),
        };
        ensure!(!m.type_id.is_empty(), "empty type_id");
        ensure!(
            (1..=4).contains(&m.inputs),
            "inputs must be 1..=4, got {}",
            m.inputs
        );
        ensure!(m.tiles >= 1 && m.tiles <= 4, "tiles must be 1..=4");
        let params = m
            .params
            .into_iter()
            .map(|p| ParamSpec {
                label: p.label.unwrap_or_else(|| p.name.clone()),
                name: p.name,
                default: p.default,
                min: p.min,
                max: p.max,
            })
            .collect();
        Ok(ShaderNodeDef::new(
            m.type_id,
            m.label,
            m.color.unwrap_or([0x9a, 0x7a, 0xc8]),
            m.inputs,
            m.tiles,
            params,
            expr,
            helpers,
        )
        .with_mattes(m.mattes))
    }

    /// Load every `*/plugin.json` under `dir` into the registry. Returns
    /// (loaded type_ids, per-plugin errors) — one broken plugin never blocks
    /// the rest.
    pub fn load_dir(dir: &Path) -> (Vec<String>, Vec<String>) {
        let mut loaded = Vec::new();
        let mut errors = Vec::new();
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return (loaded, errors), // no plugin dir is not an error
        };
        for e in entries.flatten() {
            let manifest = e.path().join("plugin.json");
            if !manifest.is_file() {
                continue;
            }
            match load_manifest(&manifest) {
                Ok(def) => {
                    loaded.push(def.type_id.clone());
                    register(def);
                }
                Err(err) => errors.push(format!("{}: {err:#}", e.path().display())),
            }
        }
        loaded.sort();
        (loaded, errors)
    }
}

#[cfg(feature = "serde")]
pub use manifest::{load_dir, load_manifest};

#[cfg(test)]
mod tests {
    use super::*;

    fn def() -> ShaderNodeDef {
        ShaderNodeDef::new(
            "invert".into(),
            "Invert".into(),
            [200, 120, 60],
            1,
            1,
            vec![
                ParamSpec {
                    name: "mix".into(),
                    label: "mix".into(),
                    default: 1.0,
                    min: 0.0,
                    max: 1.0,
                },
                ParamSpec {
                    name: "mix_extra".into(),
                    label: "extra".into(),
                    default: 0.0,
                    min: 0.0,
                    max: 1.0,
                },
            ],
            "mix($in, vec4f(1.0 - $in.rgb, $in.a), $mix + $mix_extra)".into(),
            String::new(),
        )
    }

    #[test]
    fn substitution_targets_uniform_slots() {
        let out = def().instantiate(&["child0".into()], 7, "p0");
        assert!(out.contains("P.v[7].x"), "{out}");
        assert!(out.contains("P.v[7].y"), "{out}");
        assert!(out.contains("child0"), "{out}");
        assert!(!out.contains('$'), "unsubstituted: {out}");
    }

    #[test]
    fn longer_param_names_win() {
        // `$mix_extra` must not be half-eaten by `$mix`.
        let out = def().instantiate(&["c".into()], 1, "p0");
        assert!(!out.contains("P.v[1].x_extra"), "{out}");
    }

    #[test]
    fn source_hash_tracks_the_shader() {
        let a = def();
        let mut b = def();
        b.expr.push_str(" * 2.0");
        let b = ShaderNodeDef::new(
            b.type_id, b.label, b.color, b.inputs, b.tiles, b.params, b.expr, b.helpers,
        );
        assert_ne!(a.source_hash, b.source_hash);
    }

    #[test]
    fn unwired_inputs_are_transparent_black() {
        let mut d = def();
        d.inputs = 2;
        let out = d.instantiate(&["c".into()], 1, "p0");
        assert!(!out.contains("$in2"), "{out}");
    }
}
