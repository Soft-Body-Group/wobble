//! The document model: what a setup IS, with no drawing in it. The nodes
//! and their authored data (`ENode`), the patch bay (`PSrc`/`PDst`/
//! `PatchBay`), the graph that holds them (`EditorGraph`) with its lowering
//! to an engine graph, and the graph edits that are not gestures (copy and
//! paste, bridges, loops, swipe ops).
//!
//! Moved out of editor.rs unchanged, and re-exported from there, so
//! `editor::EditorGraph` stays the path everyone uses. It still leans on a
//! few of the editor's helpers; cutting those is what stands between this
//! module and an egui-free `doc` crate (CLAUDE.md, "Direction already
//! decided").

use crate::editor::*;
use egui::{Pos2, Rect, Vec2};
use engine::graph::{GradeParams, Graph, NodeKind, PortKind, TransformParams};

fn one() -> f32 {
    1.0
}

fn is_one(v: &f32) -> bool {
    *v == 1.0
}

impl ENode {
    /// The kind as it stands at tick `t`, with this node's animation applied.
    /// Callers get a transient copy for gizmos/key-capture; never stored back.
    pub fn kind_at(&self, t: u64) -> NodeKind {
        let mut kind = self.kind.clone();
        crate::anim::apply(&mut kind, &self.anim, t);
        kind
    }
}

#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ENode {
    pub kind: NodeKind,
    pub pos: Pos2,
    /// Bypassed: input 0 flows through unchanged (Reads go black).
    #[serde(default)]
    pub disabled: bool,
    /// Per-node visual scale (selection scale handles change it); shapes,
    /// stalks and hit areas all follow.
    #[serde(default = "one")]
    pub size: f32,
    /// Blend only: fixed output resolution. None = inherit from the
    /// background (B) input. Lives on ENode (not NodeKind) so old setups with
    /// unit `Merge` variants keep deserializing. The field was called
    /// `merge_res` before the node was renamed, so setups written then still
    /// carry their override.
    #[serde(default, alias = "merge_res")]
    pub blend_res: Option<(u32, u32)>,
    /// Keyframe curves per channel. Authored data only — playback evaluates
    /// into a transient engine Graph and never writes values back here
    /// (write-back would flood undo and thrash the render-cache key).
    #[serde(default, deserialize_with = "crate::anim::de_anim_map")]
    pub anim: std::collections::BTreeMap<crate::anim::Channel, crate::anim::AnimCurve>,
    /// Grade only: show the four adjustments as rgb sliders instead of
    /// wheels. Same params either way — it is which instrument suits the
    /// hand, so it lives here rather than in `NodeKind`, like `blend_res`.
    #[serde(default)]
    pub grade_sliders: bool,
    /// Read only: drawn as a little CRT showing its frame (shift+H —
    /// Blender's "toggle node preview"). Presentation, like `grade_sliders`:
    /// the engine never learns it. See `crate::tv`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub tv: bool,
    /// Load only: the project asset this Load reads, by NAME (project code,
    /// shot, asset — `crate::assets`). When set it is the truth and `path`
    /// is derived from it at every load, so a project moved under the
    /// managed root keeps loading. Absent for an ordinary Load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset: Option<crate::assets::AssetRef>,
    /// Load only: a PROJECT Load, which reads only footage a managed project
    /// tracks and never shows a path or a file browser (`crate::projectui`).
    /// Made by the "Project Load" menu entry; presentation, like
    /// `grade_sliders` — the engine sees a plain Load.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub project_only: bool,
    /// Which rgb rows this node has UNGANGED, by row label. Ganged is the
    /// default, so the common case stores nothing. It lives here rather than
    /// in `NodeKind` for the same reason `blend_res` does: it is how the
    /// panel is being driven, not what the pixels do.
    #[serde(default)]
    pub ungang: std::collections::BTreeSet<String>,
    /// ---- tool CONTROL nodes (`ui.*`) ------------------------------------
    /// A control's caption on the panel it appears on. Presentation, so it
    /// lives here for the same reason `grade_sliders` does — and separate
    /// from the param NAME, which is derived from the node id and therefore
    /// survives relabelling.
    #[serde(default)]
    pub ui_label: String,
    /// Output frame range set by hand, overriding whatever this node would
    /// otherwise report (media for a Read, the script for a generator, the
    /// input for everything else). See `crate::ranges`.
    #[serde(default)]
    pub range: Option<(u64, u64)>,
    /// Its cell on the panel's tile grid. `None` until it is placed, which is
    /// how a freshly added control finds the first free cell.
    #[serde(default)]
    pub ui_cell: Option<[i32; 2]>,
    /// How many whole cells it occupies. Controls are sized in cells because
    /// the panel is a grid: a widget that straddled one would tear it.
    #[serde(default)]
    pub ui_span: Option<[u8; 2]>,
    /// Dot only: how hard the wires pull through the bead — its handle
    /// length as a multiple of the automatic one. 0 meets in a corner, 1 is
    /// the automatic curve, higher swings wider. Presentation, like
    /// `grade_sliders`: a dot never reaches the engine. Not written while it
    /// is 1, so setups do not grow a field on every node.
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub tension: f32,
    /// The colour this node's viewer HUD is drawn in — its manipulators,
    /// floating sliders and their labels — as linear Rec.709, the value the
    /// colour picker holds. `None` (the default) is the adaptive ink that
    /// picks light or dark against the footage. Set from the swatch on the
    /// node's HUD chip; presentation, like `grade_sliders`, and kept on the
    /// node so it survives deselect, undo and a save.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hud_color: Option<[f32; 3]>,
}

// ---- Expression patching ------------------------------------------------
/// Value links wired in the patch bay's link mode: any float channel can
/// drive any other, optionally through small value nodes (mult, merges, an
/// expression). Like animation, links apply TRANSIENTLY at evaluation —
/// authored values are never overwritten.

/// A value source: a node channel's output, or a value node's output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum PSrc {
    Chan(u64, crate::anim::Channel),
    Mod(u64),
    /// A TOOL PARAMETER, inside a tool's own body: the knob the tool exposes
    /// on its instances, patched here into whatever inner node it should
    /// drive. Always a `Channel::Custom` (the param's interned name), which
    /// is what makes it serialize by NAME — an interner index would not
    /// survive a restart. It resolves against [`EditorGraph::params`], so it
    /// reads 0 anywhere but a body being expanded for an instance.
    Param(crate::anim::Channel),
}

/// A value destination: a node channel's input, or a value node's input slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum PDst {
    Chan(u64, crate::anim::Channel),
    Mod(u64, u8),
}

#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PatchLink {
    pub from: PSrc,
    pub to: PDst,
}

/// A small value-processing node living on the patch canvas. Every merge
/// takes two inputs, and its constant stands in for `b` while that jack is
/// unplugged — so "times 2" and "times another channel" are the same node.
#[derive(Clone, PartialEq, serde::Serialize)]
pub enum ModKind {
    Mult(f32),
    Add(f32),
    Max(f32),
    Min(f32),
    /// exprs.rs expression over inputs `a`, `b` and the frame `t`.
    Expr(String),
}

/// Early patch setups stored parameterless merges (`Add2`, `Mult2`, ...);
/// they load as the constant-carrying forms.
#[derive(serde::Deserialize)]
enum ModKindDe {
    Mult(f32),
    Add(f32),
    Max(f32),
    Min(f32),
    Add2,
    Mult2,
    Max2,
    Min2,
    Expr(String),
}

impl<'de> serde::Deserialize<'de> for ModKind {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(match ModKindDe::deserialize(d)? {
            ModKindDe::Mult(k) => ModKind::Mult(k),
            ModKindDe::Add(k) => ModKind::Add(k),
            ModKindDe::Max(k) => ModKind::Max(k),
            ModKindDe::Min(k) => ModKind::Min(k),
            ModKindDe::Add2 => ModKind::Add(0.0),
            ModKindDe::Mult2 => ModKind::Mult(1.0),
            ModKindDe::Max2 => ModKind::Max(0.0),
            ModKindDe::Min2 => ModKind::Min(0.0),
            ModKindDe::Expr(x) => ModKind::Expr(x),
        })
    }
}

impl ModKind {
    pub fn label(&self) -> &'static str {
        match self {
            ModKind::Mult(_) => "mult",
            ModKind::Add(_) => "add",
            ModKind::Max(_) => "max",
            ModKind::Min(_) => "min",
            ModKind::Expr(_) => "expr",
        }
    }

    pub fn inputs(&self) -> u8 {
        2
    }
}

#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ModNode {
    pub id: u64,
    pub kind: ModKind,
    /// Patch-canvas position in points (origin-relative).
    pub pos: [f32; 2],
}

#[derive(Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PatchBay {
    pub mods: Vec<ModNode>,
    pub links: Vec<PatchLink>,
    pub next: u64,
}

impl PatchBay {
    /// Wire `from` into `to`, replacing whatever fed `to` before (an input
    /// takes exactly one cable).
    pub fn link(&mut self, from: PSrc, to: PDst) {
        if let (PSrc::Chan(sn, sc), PDst::Chan(dn, dc)) = (from, to) {
            if sn == dn && sc == dc {
                return; // a channel feeding itself is a no-op loop
            }
        }
        self.links.retain(|l| l.to != to);
        self.links.push(PatchLink { from, to });
    }

    pub fn unlink(&mut self, to: PDst) {
        self.links.retain(|l| l.to != to);
    }

    pub fn link_to(&self, to: PDst) -> Option<&PatchLink> {
        self.links.iter().find(|l| l.to == to)
    }

    pub fn add_mod(&mut self, kind: ModKind, pos: [f32; 2]) -> u64 {
        let id = self.next;
        self.next += 1;
        self.mods.push(ModNode { id, kind, pos });
        id
    }

    /// Remove a value node and every cable touching it.
    pub fn remove_mod(&mut self, id: u64) {
        self.mods.retain(|m| m.id != id);
        self.links.retain(|l| {
            !matches!(l.from, PSrc::Mod(m) if m == id)
                && !matches!(l.to, PDst::Mod(m, _) if m == id)
        });
    }
}

/// `EditorGraph::edge_out` on the wire. JSON keys are strings, so a
/// tuple-keyed map cannot be written as an object — serde_json refuses and
/// the whole save fails the moment any wire takes output 1+. It goes as a
/// list of `((dst, input), output)`, sorted so a save is deterministic.
/// Nothing non-empty was ever written the old way (it could not be), and an
/// empty object reads back as no entries.
mod edge_out_map {
    use std::collections::HashMap;

    pub fn serialize<S: serde::Serializer>(
        m: &HashMap<(u64, usize), u32>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let mut v: Vec<((u64, usize), u32)> = m.iter().map(|(k, v)| (*k, *v)).collect();
        v.sort_unstable();
        serde::Serialize::serialize(&v, s)
    }

    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        d: D,
    ) -> Result<HashMap<(u64, usize), u32>, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(untagged)]
        enum Wire {
            List(Vec<((u64, usize), u32)>),
            Legacy(serde::de::IgnoredAny),
        }
        Ok(match serde::Deserialize::deserialize(d)? {
            Wire::List(v) => v.into_iter().collect(),
            Wire::Legacy(_) => HashMap::new(),
        })
    }
}

#[derive(Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EditorGraph {
    pub nodes: std::collections::BTreeMap<u64, ENode>,
    /// (dst node, dst input, src node) — vec keeps it JSON-friendly.
    pub edges: Vec<(u64, usize, u64)>,
    /// Which OUTPUT of the source an edge takes, for the few kinds that have
    /// more than one (a multi-band LumaKey). A SPARSE side table, exactly
    /// like the engine's: absent means output 0, so `edges` keeps its shape,
    /// every destructure of it stays valid, and every setup ever written
    /// loads unchanged.
    #[serde(default, with = "edge_out_map")]
    pub edge_out: std::collections::HashMap<(u64, usize), u32>,
    pub next_id: u64,
    /// Viewer slot A's node (render root when A is active). Set with V / 1.
    #[serde(default)]
    pub view: Option<u64>,
    /// Viewer slot B's node. Set with 2.
    #[serde(default)]
    pub view_b: Option<u64>,
    /// Which slot the viewer shows (bare 1/2 flips without assigning).
    #[serde(default)]
    pub view_active_b: bool,
    /// Which OUTPUT of each viewer slot's node is shown. Only a multi-band
    /// LumaKey has more than one; 0 everywhere else, which is why no setup
    /// ever written carries these.
    #[serde(default)]
    pub view_out: u32,
    #[serde(default)]
    pub view_b_out: u32,
    /// DUAL VIEW: the second viewer's own pair of slots, worked exactly like
    /// `view`/`view_b` — keys 3 and 4 rather than 1 and 2.
    #[serde(default)]
    pub view_c: Option<u64>,
    #[serde(default)]
    pub view_d: Option<u64>,
    #[serde(default)]
    pub view2_active_d: bool,
    #[serde(default)]
    pub view_c_out: u32,
    #[serde(default)]
    pub view_d_out: u32,
    /// Which viewer the number keys last aimed at, and so which one is LIVE
    /// in dual view: 0 = the first, 1 = the second.
    #[serde(default)]
    pub view_pane: u8,
    /// Expression links + value nodes (the patch bay's link mode).
    #[serde(default)]
    pub patch: PatchBay,
    /// Only meaningful for a TOOL BODY: the exposed parameters bound to one
    /// instance's values, which is what a `PSrc::Param` cable reads. A comp's
    /// own graph leaves this empty — `crate::tools` fills it in on the
    /// throwaway copy it expands.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub params: std::collections::BTreeMap<String, f32>,
    /// Backdrops: titled rectangles drawn under the nodes. Decoration
    /// only — they own nothing, so they are not in the render-cache key and
    /// a setup without them loads exactly as it did.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<crate::nodeux::Group>,
    /// Regions: render-layer-scale grouping (FG/ENV/CHAR). Unlike backdrops
    /// they OWN things — a member list and the common sources the backtick
    /// pie offers — so they are a separate entity, not a fatter backdrop.
    /// See `crate::regions`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub regions: Vec<crate::regions::Region>,
    /// Inputs whose wire is DRAWN HIDDEN (node-patch mode's behind-the-PCB
    /// noodles; a source chip's feed). Presentation only: the edge itself
    /// lives in `edges` like any other, so the engine, the taints and the
    /// cycle checks never learn the concept. Keyed (dst, input) — an input
    /// takes exactly one wire, so that names the edge.
    #[serde(default, skip_serializing_if = "std::collections::BTreeSet::is_empty")]
    pub hidden_wires: std::collections::BTreeSet<(u64, usize)>,
    /// Nodes whose WHOLE properties panel the comp publishes to its host
    /// (Scrub's wob controls, hosted.rs), under its own panel.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub host_panels: Vec<u64>,
    /// Inside a 2dini: the z and blend set on a noodle, keyed (dst, input)
    /// like the hidden layer. A wire with nothing set is absent; one whose
    /// edge is gone is ignored (and dropped on the way to the engine).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub wire_layer: Vec<((u64, usize), engine::geo::Layering)>,
    /// Common sources the backtick pie offers ANYWHERE in this comp — the
    /// crypto read, the plate, whatever every region borrows. A region's
    /// own roster (`regions::Region::sources`) is the same idea scoped to
    /// its wash; this is the one that needs no region at all.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub global_sources: Vec<u64>,
    /// The space every node's pixels are in, between the Reads' input
    /// transforms and the Writes' output transforms.
    ///
    /// `serde(default)` lands on Linear Rec.709 for a setup saved before
    /// this existed, which is what those comps were actually working in — so
    /// they open looking the same. New comps are made in ACEScg by
    /// `default_graph`.
    #[serde(default)]
    pub working: engine::color::Colorspace,
    /// The least mature node the add menus offer — a PROJECT setting, saved
    /// with the setup rather than in prefs, so a comp that was built on
    /// production nodes only stays that way on every machine it opens on.
    /// Defaults to alpha (everything), which is what every setup written
    /// before it existed gets. Presentation: not in the render-cache key,
    /// and cleared out of a tool body's identity hash.
    #[serde(default)]
    pub node_status_min: NodeStatus,
    /// SUBWOBS defined in this graph — made by collapsing a selection
    /// (`tools::collapse`), keyed by type id (`subwob.N`). The definition
    /// lives in the document rather than in a file, so it saves with the
    /// setup, undoes like any edit and nests (a body may hold its own).
    /// Registered for rendering by `tools::register_locals`.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub subwobs: std::collections::BTreeMap<String, crate::tools::SubwobRef>,
}

/// A Read made by hand starts as a TV (shift+H turns it back into the
/// plain triangle). `ENode::new` stays plain: tests and suites build their
/// graphs through it and measure the triangle.
pub fn new_read_tv(kind: &NodeKind) -> bool {
    matches!(kind, NodeKind::Read { .. })
}

impl ENode {
    /// A node as the editor makes them. NOT a `Default` impl: `size` has to
    /// be 1.0, and a derived Default would quietly give 0 and render
    /// nothing.
    pub fn new(mut kind: NodeKind, pos: Pos2) -> Self {
        // The add menus only make a NodeKind: "Project Load" marks its Load
        // with a path no file has, turned into the flag here, where every
        // new node passes.
        let mut project_only = false;
        if let NodeKind::Read { path, .. } = &mut kind {
            if path == crate::assets::PROJECT_LOAD {
                path.clear();
                project_only = true;
            }
        }
        Self {
            kind,
            project_only,
            pos,
            disabled: false,
            size: 1.0,
            blend_res: None,
            anim: std::collections::BTreeMap::new(),
            grade_sliders: false,
            tv: false,
            asset: None,
            ungang: std::collections::BTreeSet::new(),
            ui_label: String::new(),
            ui_cell: None,
            ui_span: None,
            range: None,
            tension: 1.0,
            hud_color: None,
        }
    }
}

impl EditorGraph {
    pub fn add(&mut self, kind: NodeKind, pos: Pos2) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, ENode::new(kind, pos));
        id
    }

    pub fn remove(&mut self, id: u64) {
        // Heal the chain: consumers of this node reconnect to its image-input
        // source (kinds permitting), so a mid-chain delete doesn't orphan
        // everything downstream.
        if let Some((_, parent)) = self.image_in_edge(id) {
            let consumers: Vec<(u64, usize)> = self
                .edges
                .iter()
                .filter(|(_, _, s)| *s == id)
                .map(|(d, i, _)| (*d, *i))
                .collect();
            let pk = self.nodes.get(&parent).map(|n| n.kind.output_kind());
            for (d, i) in consumers {
                let ok = self
                    .nodes
                    .get(&d)
                    .map(|n| n.kind.input_kind(i))
                    .zip(pk)
                    .is_some_and(|(a, b)| a == b);
                if ok && d != parent {
                    self.connect(d, i, parent);
                }
            }
        }
        // A harp comes apart cleanly: deleting a Breakout takes its hidden
        // kids with it and unpairs the receiver (which keeps working as a
        // plain n-way sum); deleting the receiver just unpairs.
        let (kids, pair) = match self.nodes.get(&id).map(|n| &n.kind) {
            Some(NodeKind::Breakout(p)) => (
                p.kids.iter().chain(p.remainder_kids.iter()).copied().collect(),
                p.pair,
            ),
            Some(NodeKind::Recombine(p)) => (Vec::new(), p.pair),
            _ => (Vec::new(), 0),
        };
        self.nodes.remove(&id);
        self.edges.retain(|(d, _, s)| *d != id && *s != id);
        for k in kids {
            self.remove(k);
        }
        if pair != 0 {
            match self.nodes.get_mut(&pair).map(|n| &mut n.kind) {
                Some(NodeKind::Breakout(p)) => p.pair = 0,
                Some(NodeKind::Recombine(p)) => p.pair = 0,
                _ => {}
            }
        }
        // Flags follow their edges; anything the retain above cut loses its.
        let mut hw = std::mem::take(&mut self.hidden_wires);
        hw.retain(|(d, i)| self.input_of(*d, *i).is_some());
        self.hidden_wires = hw;
        self.prune_groups();
        crate::regions::prune(self);
        if self.view == Some(id) {
            self.view = None;
        }
        if self.view_b == Some(id) {
            self.view_b = None;
        }
    }

    /// The render root: whichever viewer slot is active.
    pub fn active_view(&self) -> Option<u64> {
        if self.view_active_b { self.view_b } else { self.view }
    }

    /// The SECOND viewer's showing slot (dual view).
    pub fn active_view2(&self) -> Option<u64> {
        if self.view2_active_d { self.view_d } else { self.view_c }
    }

    /// Which output of [`Self::active_view2`] is on screen.
    pub fn active_view2_out(&self) -> u32 {
        if self.view2_active_d { self.view_d_out } else { self.view_c_out }
    }

    /// The node the LIVE viewer shows: the second one when the number keys
    /// last aimed there, the first otherwise. In single-view mode only the
    /// first is ever live.
    pub fn live_view(&self, dual: bool) -> Option<u64> {
        if dual && self.view_pane == 1 { self.active_view2() } else { self.active_view() }
    }

    /// Which output of [`Self::active_view`] is on screen.
    pub fn active_view_out(&self) -> u32 {
        if self.view_active_b { self.view_b_out } else { self.view_out }
    }

    /// Point the ACTIVE viewer slot at one of `node`'s outputs.
    pub fn view_output(&mut self, node: u64, out: u32) {
        if self.view_active_b {
            self.view_b = Some(node);
            self.view_b_out = out;
        } else {
            self.view = Some(node);
            self.view_out = out;
        }
    }

    pub fn connect(&mut self, dst: u64, input: usize, src: u64) {
        self.connect_out(dst, input, src, 0);
    }

    /// Wire `src`'s output `out` into `dst`'s `input`. Output 0 leaves no
    /// side-table entry, so an ordinary connection stays exactly what it was.
    pub fn connect_out(&mut self, dst: u64, input: usize, src: u64, out: u32) {
        self.edges.retain(|(d, i, _)| !(*d == dst && *i == input));
        self.edges.push((dst, input, src));
        if out == 0 {
            self.edge_out.remove(&(dst, input));
        } else {
            self.edge_out.insert((dst, input), out);
        }
        // An ordinary connect SURFACES the wire: hidden is a property of how
        // an edge was made, and remaking it visibly means visible.
        self.hidden_wires.remove(&(dst, input));
        // A node that follows its pipe (a Bake, a SoftCrop, an Invert or a
        // Gamma...) is retyped to what it was wired: into a matte chain its
        // ports read Mask on both sides (the Dot pattern). Every wiring
        // path lands here, so the flag can never go stale.
        if self.node(dst).is_some_and(|n| n.kind.follows_pipe(input)) {
            let is_mask = self
                .node(src)
                .map(|n| n.kind.output_kind_at(out as usize) == PortKind::Mask)
                .unwrap_or(false);
            self.retype_pipe(dst, is_mask);
        }
    }

    /// Put pipe-follower `id` in the matte pipe (or out of it), and carry
    /// the change down a run of followers it feeds: an Invert -> Gamma
    /// chain rewired onto a matte becomes a matte chain end to end, rather
    /// than leaving the Gamma an image node fed a matte.
    fn retype_pipe(&mut self, id: u64, is_mask: bool) {
        let mut stack = vec![id];
        let mut seen = std::collections::HashSet::new();
        while let Some(at) = stack.pop() {
            if !seen.insert(at) {
                continue;
            }
            let Some(n) = self.node_mut(at) else { continue };
            if n.kind.matte_mode() == is_mask && at != id {
                continue; // already in this pipe: nothing below changes
            }
            n.kind.set_matte_mode(is_mask);
            stack.extend(
                self.edges
                    .iter()
                    .filter(|(d, i, s)| *s == at && *i == 0 && self.out_of(*d, *i) == 0)
                    .filter(|(d, i, _)| self.node(*d).is_some_and(|n| n.kind.follows_pipe(*i)))
                    .map(|(d, ..)| *d),
            );
        }
    }

    /// Wire it, and DRAW IT HIDDEN — the patch-mode connect, and what a
    /// source chip's feed uses. The edge is an ordinary edge (the engine,
    /// the taints and the cycle checks see it like any other); only the
    /// editor's drawing and hit tests consult the flag.
    pub fn connect_hidden(&mut self, dst: u64, input: usize, src: u64) {
        self.connect(dst, input, src);
        self.hidden_wires.insert((dst, input));
    }

    pub fn is_hidden(&self, dst: u64, input: usize) -> bool {
        self.hidden_wires.contains(&(dst, input))
    }

    /// The Read a multi-channel wire comes from: walk a ChannelSelect's
    /// input 0 upstream through Dots to the first real node, which must be
    /// a Read. None = unwired or mis-wired (the select stays black).
    pub fn channel_source(&self, cs: u64) -> Option<u64> {
        let mut at = self.edges.iter().find(|(d, i, _)| *d == cs && *i == 0)?.2;
        for _ in 0..64 {
            match self.node(at).map(|n| &n.kind) {
                Some(NodeKind::Dot { .. }) => {
                    at = self.edges.iter().find(|(d, i, _)| *d == at && *i == 0)?.2;
                }
                Some(NodeKind::Read { .. }) => return Some(at),
                // A Blender node with its passes on the fat cable.
                Some(NodeKind::Blender(p)) if p.fat => return Some(at),
                _ => return None,
            }
        }
        None
    }

    /// Which output of its source an edge takes. 0 unless recorded.
    pub fn out_of(&self, dst: u64, input: usize) -> u32 {
        self.edge_out.get(&(dst, input)).copied().unwrap_or(0)
    }

    pub fn disconnect_input(&mut self, dst: u64, input: usize) {
        self.edges.retain(|(d, i, _)| !(*d == dst && *i == input));
        self.edge_out.remove(&(dst, input));
        self.hidden_wires.remove(&(dst, input));
    }

    /// The z and blend set on the noodle into (dst, input).
    pub fn wire_layering(&self, dst: u64, input: usize) -> engine::geo::Layering {
        self.wire_layer.iter().find(|(k, _)| *k == (dst, input)).map(|(_, l)| *l).unwrap_or_default()
    }

    /// Set them; the default is stored as nothing.
    pub fn set_wire_layering(&mut self, dst: u64, input: usize, l: engine::geo::Layering) {
        self.wire_layer.retain(|(k, _)| *k != (dst, input));
        if l != engine::geo::Layering::default() {
            self.wire_layer.push(((dst, input), l));
            self.wire_layer.sort_by_key(|(k, _)| *k);
        }
    }

    pub fn input_of(&self, dst: u64, input: usize) -> Option<u64> {
        self.edges
            .iter()
            .find(|(d, i, _)| *d == dst && *i == input)
            .map(|(_, _, s)| *s)
    }

    pub fn node(&self, id: u64) -> Option<&ENode> {
        self.nodes.get(&id)
    }

    pub fn node_mut(&mut self, id: u64) -> Option<&mut ENode> {
        self.nodes.get_mut(&id)
    }

    /// (connected inputs, connected outputs) of a node.
    fn io_counts(&self, id: u64) -> (usize, usize) {
        // Mask wires don't count: chain ops (swap etc.) see the image path.
        let ins = self
            .edges
            .iter()
            .filter(|(d, i, _)| {
                *d == id
                    && self
                        .nodes
                        .get(d)
                        .is_some_and(|n| is_chain_input(&n.kind, *i))
            })
            .count();
        let outs = self.edges.iter().filter(|(_, _, s)| *s == id).count();
        (ins, outs)
    }

    /// The (input index, source) of the node's image-input edge, if wired.
    fn image_in_edge(&self, id: u64) -> Option<(usize, u64)> {
        self.edges
            .iter()
            .find(|(d, i, _)| {
                *d == id
                    && self
                        .nodes
                        .get(d)
                        .is_some_and(|n| is_chain_input(&n.kind, *i))
            })
            .map(|(_, i, s)| (*i, *s))
    }

    /// Follow a wire back through routing dots to the node that actually
    /// makes the pixels. `None` when the chain runs out at an unwired dot.
    /// Bounded rather than recursive: a dot chain that somehow closed on
    /// itself would otherwise hang every conversion to an engine graph.
    pub fn resolve_dot(&self, mut id: u64) -> Option<u64> {
        for _ in 0..1024 {
            match self.nodes.get(&id) {
                // A Time Range is metadata only: the pixels are its input's
                // (unless its "start at" slips them — then it is a Slip).
                Some(n) if matches!(&n.kind, NodeKind::Dot { .. })
                    || matches!(&n.kind, NodeKind::TimeRange(p) if p.slip().is_none()) =>
                {
                    id = self.input_of(id, 0)?;
                }
                Some(_) => return Some(id),
                None => return None,
            }
        }
        None
    }

    pub fn is_dot(&self, id: u64) -> bool {
        matches!(self.nodes.get(&id).map(|n| &n.kind), Some(NodeKind::Dot { .. }))
    }

    /// A dot wears the port kind of whatever reaches it, so a dot cut into
    /// a mask wire draws in the mask colour and courts mask outputs. Cheap
    /// enough to just re-derive every frame — nothing to keep in sync.
    /// Splice a converter into every wire whose two ends no longer agree
    /// about their pipe — a setup saved before a port changed type — and
    /// return how many it bridged. Run once, on load.
    ///
    /// Today that is the depth port becoming DATA. Its old image wire gets
    /// a ToData carrying the channel the Defocus used to pick for itself,
    /// and the Defocus goes to R: the ToData broadcasts its value to rgb
    /// (and the two nodes' luma weights are the same Rec.709 row), so the
    /// Defocus reads exactly the number it read before and the picture does
    /// not move. A wire with no bridge (data into a matte port) is left as
    /// it was, still rendering as it always did.
    pub fn bridge_mistyped(&mut self) -> usize {
        use engine::graph::{DepthChannel, MatteChannel};
        // A Blur briefly had mask (1) AND modulate (2); its one matte port
        // is now 1 and modulates. A matte on the old 2 moves down, and wins
        // over an old mask — modulate was the one asked for.
        let blurs: Vec<u64> = self
            .edges
            .iter()
            .filter(|(d, i, _)| *i == 2 && matches!(self.node(*d).map(|n| &n.kind), Some(NodeKind::Blur(_))))
            .map(|(d, ..)| *d)
            .collect();
        for d in blurs {
            self.edges.retain(|(dd, i, _)| !(*dd == d && *i == 1));
            for e in self.edges.iter_mut().filter(|(dd, i, _)| *dd == d && *i == 2) {
                e.1 = 1;
            }
        }
        let wrong: Vec<(u64, usize, u64, u32)> = self
            .edges
            .iter()
            .filter_map(|(d, i, s)| {
                let out = self.out_of(*d, *i);
                let gives = self.node(*s)?.kind.output_kind_at(out as usize);
                (!self.node(*d)?.kind.accepts(*i, gives)).then_some((*d, *i, *s, out))
            })
            .collect();
        let mut made = 0;
        for (dst, input, src, out) in wrong {
            let (Some(sk), Some(dk)) =
                (self.node(src).map(|n| n.kind.clone()), self.node(dst).map(|n| n.kind.clone()))
            else {
                continue;
            };
            let Some(mut bridge) = bridge_for(&sk, out, &dk, input) else { continue };
            // The channel moves from the consumer onto the converter.
            if let (NodeKind::ToData(p), NodeKind::Defocus(d)) = (&mut bridge, &dk) {
                p.channel = match d.depth_channel {
                    DepthChannel::R => MatteChannel::R,
                    DepthChannel::G => MatteChannel::G,
                    DepthChannel::B => MatteChannel::B,
                    DepthChannel::A => MatteChannel::A,
                    DepthChannel::Luma => MatteChannel::Luma,
                };
                if let Some(NodeKind::Defocus(d)) = self.node_mut(dst).map(|n| &mut n.kind) {
                    d.depth_channel = DepthChannel::R;
                }
            }
            let hidden = self.is_hidden(dst, input);
            let (sp, dp) = (self.node(src).map(|n| n.pos), self.node(dst).map(|n| n.pos));
            let at = match (sp, dp) {
                (Some(a), Some(b)) => a + (b - a) * 0.5,
                _ => Pos2::ZERO,
            };
            let sz = self.node(dst).map(|n| n.size).unwrap_or(1.0);
            let id = self.add(bridge, at);
            if let Some(n) = self.nodes.get_mut(&id) {
                n.size = sz;
            }
            self.connect_out(id, 0, src, out);
            self.connect(dst, input, id);
            if hidden {
                self.hidden_wires.insert((dst, input));
            }
            made += 1;
        }
        made
    }

    /// Layers used to carry their own transform; placement is an upstream
    /// Transform's job now. Splice a Transform in front of every layer whose
    /// old transform (or its curves) did anything, carrying the values and
    /// the keys, and clear the legacy field. Run once, on load; returns how
    /// many it spliced.
    ///
    /// Exact but for one case: a layer's transform anchored its pivot in the
    /// STACK's format, a Transform anchors in its input's. Where a plate's
    /// format differs from the stack's, a rotate or scale now turns about
    /// that plate's centre instead.
    pub fn lift_layer_xforms(&mut self) -> usize {
        use crate::anim::{legacy_layer_transform, TRANSFORM};
        use engine::graph::TransformParams;
        let stacks: Vec<u64> = self
            .nodes
            .iter()
            .filter(|(_, n)| matches!(n.kind, NodeKind::Layer(_)))
            .map(|(id, _)| *id)
            .collect();
        let mut made = 0;
        for stack in stacks {
            let Some(n) = self.nodes.get_mut(&stack) else { continue };
            let (pos, size) = (n.pos, n.size);
            let ENode { kind, anim, .. } = n;
            let NodeKind::Layer(p) = kind else { continue };
            let mut lifts = Vec::new();
            for (i, l) in p.layers.iter_mut().enumerate() {
                let xf = l.legacy_xform.take().unwrap_or_default();
                let curves: std::collections::BTreeMap<_, _> = legacy_layer_transform(i as u16)
                    .iter()
                    .zip(TRANSFORM)
                    .filter_map(|(old, new)| anim.remove(old).map(|c| (*new, c)))
                    .collect();
                // The gang flag is UI, not placement.
                let still = TransformParams {
                    scale_gang: TransformParams::default().scale_gang,
                    matte: false,
                    ..xf
                } == TransformParams::default();
                if !still || !curves.is_empty() {
                    lifts.push((i, xf, curves));
                }
            }
            for (i, xf, curves) in lifts {
                let src = self.input_of(stack, i);
                let out = self.out_of(stack, i);
                let hidden = self.is_hidden(stack, i);
                let at = match src.and_then(|s| self.node(s)).map(|n| n.pos) {
                    Some(a) => a + (pos - a) * 0.5,
                    None => pos - Vec2::new(0.0, 80.0),
                };
                let id = self.add(NodeKind::Transform(xf), at);
                if let Some(n) = self.nodes.get_mut(&id) {
                    n.size = size;
                    n.anim = curves;
                }
                if let Some(s) = src {
                    self.connect_out(id, 0, s, out);
                }
                self.connect(stack, i, id);
                if hidden {
                    self.hidden_wires.insert((stack, i));
                }
                made += 1;
            }
        }
        made
    }

    pub fn sync_dot_kinds(&mut self) {
        let want: Vec<(u64, PortKind)> = self
            .nodes
            .iter()
            .filter(|(_, n)| matches!(n.kind, NodeKind::Dot { .. }))
            .filter_map(|(id, _)| {
                let src = self.resolve_dot(self.input_of(*id, 0)?)?;
                let out = self.out_of(*id, 0) as usize;
                let k = self.nodes.get(&src)?.kind.output_kind_at(out);
                Some((*id, k))
            })
            .collect();
        for (id, k) in want {
            if let Some(kind @ NodeKind::Dot { .. }) =
                self.nodes.get_mut(&id).map(|n| &mut n.kind)
            {
                *kind = NodeKind::dot(k);
            }
        }
    }

    /// Remove a routing dot and heal every consumer straight through to
    /// its source, carrying the source-OUTPUT index across — the thing
    /// `remove`'s generic chain-heal drops. The tidy pass dissolves dots
    /// before re-routing; a dot with no source leaves its consumers
    /// unplugged, which is what that wire meant. No-op on non-dots.
    pub fn dissolve_dot(&mut self, id: u64) -> bool {
        if !self.is_dot(id) {
            return false;
        }
        let src = self.input_of(id, 0);
        let out = self.out_of(id, 0);
        let consumers: Vec<(u64, usize)> = self
            .edges
            .iter()
            .filter(|(_, _, s)| *s == id)
            .map(|(d, i, _)| (*d, *i))
            .collect();
        for (d, i) in consumers {
            match src {
                Some(s) => self.connect_out(d, i, s, out),
                None => self.disconnect_input(d, i),
            }
        }
        self.nodes.remove(&id);
        self.edges.retain(|(d, _, s)| *d != id && *s != id);
        if self.view == Some(id) {
            self.view = src;
        }
        if self.view_b == Some(id) {
            self.view_b = src;
        }
        self.prune_groups();
        true
    }

    /// Exchange the wires on two of a node's inputs — a Blend's + and −,
    /// a Mix's from and to. An unplugged port swaps too (it takes the
    /// other's wire and leaves its own port free), which is what makes this
    /// usable as "put it on the other input" rather than only as a flip.
    pub fn swap_inputs(&mut self, id: u64, a: usize, b: usize) -> bool {
        if a == b || self.nodes.get(&id).is_none() {
            return false;
        }
        let (sa, sb) = (self.input_of(id, a), self.input_of(id, b));
        if sa.is_none() && sb.is_none() {
            return false;
        }
        self.disconnect_input(id, a);
        self.disconnect_input(id, b);
        if let Some(s) = sb {
            self.connect(id, a, s);
        }
        if let Some(s) = sa {
            self.connect(id, b, s);
        }
        true
    }

    /// Close the gaps in a multi-input node's wiring: its wired inputs
    /// renumber 0..k in their order, their side tables following. Returns
    /// k. What keeps a Houdini-style Merge one socket with no holes in it
    /// after a wire is pulled out of the middle.
    pub fn pack_inputs(&mut self, id: u64) -> usize {
        let mut wired: Vec<usize> =
            self.edges.iter().filter(|(d, _, _)| *d == id).map(|(_, i, _)| *i).collect();
        wired.sort_unstable();
        wired.dedup();
        if wired.iter().enumerate().all(|(k, i)| k == *i) {
            return wired.len();
        }
        let to = |i: usize| wired.iter().position(|w| *w == i).unwrap_or(i);
        for e in self.edges.iter_mut().filter(|e| e.0 == id) {
            e.1 = to(e.1);
        }
        let outs: Vec<_> = self.edge_out.keys().filter(|k| k.0 == id).copied().collect();
        for k in outs {
            let v = self.edge_out.remove(&k).unwrap();
            self.edge_out.insert((id, to(k.1)), v);
        }
        let hid: Vec<_> = self.hidden_wires.iter().filter(|k| k.0 == id).copied().collect();
        for k in hid {
            self.hidden_wires.remove(&k);
            self.hidden_wires.insert((id, to(k.1)));
        }
        wired.len()
    }

    /// The single downstream consumer, if the node has exactly one.
    pub fn single_consumer(&self, id: u64) -> Option<u64> {
        let mut it = self.edges.iter().filter(|(_, _, s)| *s == id);
        let first = it.next().map(|(d, _, _)| *d);
        if it.next().is_some() { None } else { first }
    }

    /// Swap `n` with its upstream parent (⌘↑): G→P→N→C becomes G→N→P→C.
    /// Only nodes with exactly one connected input AND one connected output
    /// qualify (both n and the parent), so the rewire is unambiguous.
    /// Positions are exchanged; returns [(id, drawn-pos delta); 2] so the
    /// editor can animate the exchange.
    pub fn swap_up(&mut self, n: u64) -> Option<[(u64, Vec2); 2]> {
        if self.io_counts(n) != (1, 1) {
            return None;
        }
        let (ni, parent) = self.image_in_edge(n)?;
        let (c_dst, c_in) =
            self.edges.iter().find(|(_, _, s)| *s == n).map(|(d, i, _)| (*d, *i))?;
        if self.io_counts(parent) != (1, 1) {
            return None;
        }
        let (pi, g) = self.image_in_edge(parent)?;
        // Malformed-graph guards (cycles / self edges).
        if g == n || g == parent || c_dst == n || c_dst == parent {
            return None;
        }
        self.connect(n, ni, g);
        self.connect(parent, pi, n);
        self.connect(c_dst, c_in, parent);
        let (pa, pb) = (self.nodes[&n].pos, self.nodes[&parent].pos);
        self.nodes.get_mut(&n).unwrap().pos = pb;
        self.nodes.get_mut(&parent).unwrap().pos = pa;
        Some([(n, pa - pb), (parent, pb - pa)])
    }

    pub fn to_engine_graph(&self) -> Graph {
        self.to_engine_graph_at(None)
    }

    /// Mirror into the engine graph, with animated channels evaluated at
    /// timeline frame `t` (None = authored values as stored). The result is
    /// TRANSIENT: the engine consumes it, nothing writes it back.
    pub fn to_engine_graph_at(&self, t: Option<u64>) -> Graph {
        self.to_engine_graph_at_f(t.map(|t| t as f32))
    }

    /// The same, at a FRACTIONAL tick. This is what motion blur is built on:
    /// a sub-frame sample is the whole comp evaluated a fraction of a frame
    /// early or late, which moves every animated parameter and nothing else.
    ///
    /// Things that are facts about a whole frame rather than a moment — a
    /// layer's trim, which frame a retime decodes — are resolved at the
    /// ROUNDED tick, because there is no such thing as a plate's half frame.
    /// The words on a text-pipe port: the String node wired into `port`
    /// of `node`, through any dots on the way.
    pub fn prompt_on(&self, node: u64, port: usize) -> Option<String> {
        let mut src = self.input_of(node, port)?;
        for _ in 0..64 {
            match &self.node(src)?.kind {
                NodeKind::Prompt(p) => return Some(p.text.clone()),
                NodeKind::Dot { .. } => src = self.input_of(src, 0)?,
                _ => return None,
            }
        }
        None
    }

    pub fn to_engine_graph_at_f(&self, t: Option<f32>) -> Graph {
        let t_whole = t.map(|t| t.round().max(0.0) as u64);
        // Tool instances are graph macros: unfold them into ordinary nodes
        // FIRST, so everything downstream of here — anim, patch, time maps,
        // codegen, the pipeline cache — only ever sees a flat graph and needs
        // to know nothing about tools. `expand` returns None when there is
        // nothing to unfold, which is both the common case and what
        // guarantees this recursion is one level deep.
        if let Some(flat) = crate::tools::expand(self, t_whole) {
            return flat.to_engine_graph_at_f(t);
        }
        // Breakouts lower the same way: consumers re-pointed at the hidden
        // ChannelSelect kids, the wide node dropped — the engine sees the
        // select ladder it already knows. One level deep, like tools.
        if let Some(flat) = crate::breakout::expand(self) {
            return flat.to_engine_graph_at_f(t);
        }
        let mut g = Graph::default();
        // Dots are wiring, not nodes: the viewer parked on one shows what
        // reaches it, exactly as the wire it sits on would.
        g.view = self.active_view().and_then(|v| self.resolve_dot(v));
        g.view_out = self.active_view_out();
        // Every viewer slot, lowered the same way (tools and harps have
        // already been unfolded by the time we get here): what the player
        // keeps decoded so flipping to any of them is instant.
        for v in [self.view, self.view_b, self.view_c, self.view_d].into_iter().flatten() {
            if let Some(r) = self.resolve_dot(v) {
                if Some(r) != g.view && !g.hot_roots.contains(&r) {
                    g.hot_roots.push(r);
                }
            }
        }
        g.working = self.working;
        let mut kinds: std::collections::BTreeMap<u64, NodeKind> = Default::default();
        for (id, n) in &self.nodes {
            // A dot never reaches the engine — no node, no stage, no slot,
            // nothing in the topology hash. `resolve_dot` below wires each
            // consumer straight to the pixels behind it.
            if matches!(&n.kind, NodeKind::Dot { .. })
                || matches!(&n.kind, NodeKind::TimeRange(p) if p.slip().is_none())
            {
                continue;
            }
            // A Time Range that starts elsewhere IS a Slip to the engine.
            let mut kind = match &n.kind {
                NodeKind::TimeRange(p) => NodeKind::Slip(engine::graph::SlipParams {
                    delay: -(p.slip().unwrap_or(0) as i32),
                }),
                k => k.clone(),
            };
            if let Some(t) = t {
                crate::anim::apply_at(&mut kind, &n.anim, t);
            }
            // A ChannelSelect lowers to a Read of its own reader slot with
            // the chosen layer — the engine, the player, both cache tiers
            // and eviction then treat the selected channel exactly like a
            // plate, and UNSELECTED channels of the multi wire cost nothing
            // anywhere. Unlowered (unwired / slot not minted yet) it stays
            // itself and renders conspicuous black via the codegen safety
            // net.
            if let NodeKind::ChannelSelect(p) = &kind {
                // Off a Blender node's fat cable: the select IS one pass
                // of that node (or its combined picture), lowered exactly
                // as its pass outputs are — no reader, no slot.
                if let Some((bid, bp)) = self.channel_source(*id).and_then(|r| {
                    match self.node(r).map(|n| &n.kind) {
                        Some(NodeKind::Blender(bp)) => Some((r, bp.clone())),
                        _ => None,
                    }
                }) {
                    kind = NodeKind::Blender(engine::blender::BlenderParams {
                        pass: engine::blender::Pass::by_key(&p.layer),
                        host: Some(bid),
                        passes: Vec::new(),
                        fat: false,
                        inputs: 0,
                        ..bp
                    });
                } else if p.source != usize::MAX {
                    if let Some(read) =
                        self.channel_source(*id).and_then(|r| self.node(r))
                    {
                        if let NodeKind::Read {
                            path, range, ram_cache, nvme_cache, input, start, first, last,
                            edges, cache_on_play, scale, ..
                        } = &read.kind
                        {
                            kind = NodeKind::Read {
                                source: p.source,
                                path: path.clone(),
                                range: *range,
                                ram_cache: *ram_cache,
                                nvme_cache: *nvme_cache,
                                layer: (!p.layer.is_empty()).then(|| p.layer.clone()),
                                input: input.clone(),
                                multi: false,
                                // The channel plays where its plate plays:
                                // without the Read's placement a harp string
                                // and the Read itself showed different frames
                                // at the same comp frame.
                                start: *start,
                                // Same frame range: the channel is the plate.
                                first: *first,
                                last: *last,
                                // ...and answers past its ends the same way.
                                edges: *edges,
                                cache_on_play: *cache_on_play,
                                // A layer of the plate, stored whole: the
                                // Load's own packing is for its own pixels.
                                single: Default::default(),
                                eight_bit: false,
                                scale: *scale,
                            };
                        }
                    }
                }
            }
            // A linked Uncrop takes its Crop's box as it is at this tick.
            if let NodeKind::Uncrop(p) = &mut kind {
                crate::nodes::uncrop::lower(self, p, t);
            }
            // "default" input = the per-format table's answer; the engine
            // only ever sees Working / Explicit.
            if let NodeKind::Read { input, path, .. } = &mut kind {
                *input = crate::colorui::resolve_input(*input, path);
            }
            // A TimeRemap's animated frame IS its retime curve, but time has
            // to be evaluable at every tick a reader prefetches — not just
            // the one on screen. Sample the curve into the node's lookup so
            // the engine can ask it for any tick. Unanimated stays empty:
            // that reads as a freeze on the single authored frame.
            if let NodeKind::TimeRemap(p) = &mut kind {
                p.lut = n
                    .anim
                    .get(&crate::anim::Channel::TimeRemapFrame)
                    .filter(|c| c.keys.len() > 1)
                    .map(sample_time_curve)
                    .unwrap_or_default();
            }
            // A keyed Freeze holds whatever its curve says, at every tick.
            if let NodeKind::Freeze(p) = &mut kind {
                p.lut = n
                    .anim
                    .get(&crate::anim::Channel::FreezeFrame)
                    .filter(|c| c.keys.len() > 1)
                    .map(|c| {
                        // Held flat past the end keys (a lookup extrapolates
                        // its end slope, and a freeze must not drift).
                        let mut l = sample_time_curve(c);
                        if let (Some(&a), Some(&b)) = (l.first(), l.last()) {
                            l.insert(0, [a[0] - 1.0, a[1]]);
                            l.push([b[0] + 1.0, b[1]]);
                        }
                        l
                    })
                    .unwrap_or_default();
            }
            // A layer's TRIM is a fact about the tick, and the engine has no
            // idea which tick it is rendering — so it is resolved here, into
            // the transient gate the packer folds into opacity. Same trick a
            // blending retime's `blend_w` uses.
            // A field that evolves has to know what time it is, and the
            // engine does not. Same transient trick as a Layer's gate.
            if let NodeKind::Noise(p) = &mut kind {
                p.time = t.unwrap_or(0.0);
            }
            // A 2dini's own noise nodes evolve too.
            if let NodeKind::Geo(p) = &mut kind {
                p.time = t.unwrap_or(0.0);
            }
            // An SDF's ripples travel and its displacement evolves.
            if let NodeKind::Sdf(p) = &mut kind {
                p.time = t.unwrap_or(0.0);
            }
            // A planet turns and churns on the clock.
            if let NodeKind::Planet(p) = &mut kind {
                p.time = t.unwrap_or(0.0);
            }
            if let NodeKind::Grain(p) = &mut kind {
                p.time = t.unwrap_or(0.0);
            }
            // A Plasma's particles are a function of the frame.
            if let NodeKind::Plasma(p) = &mut kind {
                p.time = t.unwrap_or(0.0);
            }
            // A Particles node simulates up to the frame.
            if let NodeKind::Particles(p) = &mut kind {
                p.time = t.unwrap_or(0.0);
            }
            // A Text's wiggles run on the clock, and a String wired into
            // its words port says what it says.
            if let NodeKind::Text(p) = &mut kind {
                p.time = t.unwrap_or(0.0);
                if let Some(words) = self.prompt_on(*id, engine::nodes::text::WORDS_INPUT) {
                    p.text = words;
                }
            }
            // A Burnin prints the tick it is rendering.
            if let NodeKind::Burnin(p) = &mut kind {
                p.time = t.unwrap_or(0.0);
            }
            // A camera or card read from a file takes the file's pose at this
            // frame (a card keeps its own transform on top). A file that cannot
            // be read leaves the node as keyed; its panel says why.
            if matches!(kind, NodeKind::Camera3D(_) | NodeKind::Card3D(_)) {
                let _ = engine::scene3d::apply_file_source(&mut kind, t.unwrap_or(0.0));
            }
            // An expression's `frame`.
            if let NodeKind::Expression(p) = &mut kind {
                p.time = t.unwrap_or(0.0);
            }
            // A Stabilize's correction is a fact about the tick: resolved
            // here into the transient pin the packer reads.
            if let NodeKind::Stabilize(p) = &mut kind {
                p.pin = p.pin_at(t_whole.unwrap_or(0) as i64);
            }
            if let NodeKind::Layer(p) = &mut kind {
                let tick = t_whole.unwrap_or(0) as i64;
                for l in &mut p.layers {
                    l.gate = if l.visible_at(tick) { 1.0 } else { 0.0 };
                }
            }
            kinds.insert(*id, kind);
            if n.disabled {
                g.disabled.insert(*id);
            }
        }
        // Expression links: driven channels take their patched value.
        // Sources read post-anim values; a source that is itself driven
        // chains (cycle-guarded, falling back to the raw value).
        if !self.patch.links.is_empty() {
            let base = kinds.clone();
            let tf = t.unwrap_or(0.0);
            for l in &self.patch.links {
                if let PDst::Chan(n, ch) = l.to {
                    let v = self.patch_value(l.from, &base, tf, &mut Vec::new());
                    if let Some(k) = kinds.get_mut(&n) {
                        crate::anim::set(k, ch, v);
                    }
                }
            }
        }
        for (id, mut kind) in kinds {
            // A CONTROL is a knob, not a picture. It had to stay in the map
            // above — patch cables READ it, which is the whole point of one —
            // but it contributes no pixels, so it never becomes an engine
            // node: no stage, no slot, nothing in the topology hash. Inside a
            // tool `expand` has already consumed it; in a comp it is the
            // script's own panel, published to an edit through `To Scrub`.
            if crate::tools::control_of(&kind).is_some() {
                continue;
            }
            // A colour keyer shows the source image (matte in alpha) only
            // while IT is the view target, and only on the transient graph —
            // exactly like GraphBake's `baked`. Authored data never carries
            // it, so a preview can never reach a render or a downstream
            // consumer, which always sees the matte.
            if let NodeKind::Keyer(k) = &mut kind {
                k.previewing = Some(id) == g.view;
            }
            // Same contract for the Cryptomatte's ids/preview views — and
            // its ranks are readers of the plate behind the cable, so they
            // play where that plate plays.
            // A Grade's ranges preview replaces its picture with the zone it
            // is showing, under the same contract as the keyer above: the
            // transient graph only, the view target only. Authored data
            // carries WHICH zone and HOW (`GradeZones::preview`, a knob that
            // undoes and saves); it carries no license to act.
            if let NodeKind::Grade(gr) = &mut kind {
                if let Some(z) = gr.zones.as_mut() {
                    z.previewing = Some(id) == g.view;
                }
            }
            // ColorSlice's highlight, under the same contract.
            if let NodeKind::ColorSlice(p) = &mut kind {
                p.previewing = Some(id) == g.view;
            }
            // Defocus's focal-plane view, under the same contract.
            if let NodeKind::Defocus(d) = &mut kind {
                d.previewing = Some(id) == g.view;
            }
            if let NodeKind::Cryptomatte(c) = &mut kind {
                c.previewing = Some(id) == g.view;
                c.start = self
                    .channel_source(id)
                    .and_then(|r| self.node(r))
                    .and_then(|r| r.kind.placement());
            }
            g.nodes.insert(id, kind);
        }
        for (d, i, s) in &self.edges {
            // Wires THROUGH dots: the consumer sees the real source. An
            // unwired dot drops the edge, which is what an unwired input is.
            if self.is_dot(*d) {
                continue;
            }
            // A Cryptomatte's cable is a lazy reference the engine never
            // samples: the ranks are the node's OWN reader slots (see
            // `CryptomatteParams::sources`), so the wire stops here — the
            // multi Read behind it is not decoded on the node's account.
            if *i == 0
                && matches!(self.node(*d).map(|n| &n.kind), Some(NodeKind::Cryptomatte(_)))
            {
                continue;
            }
            if let Some(src) = self.resolve_dot(*s) {
                g.edges.insert((*d, *i), src);
                // Which OUTPUT of the source, for the kinds that have
                // several. Sparse both sides: output 0 records nothing, so a
                // graph with no multi-output node is byte-identical. A wire
                // resolved THROUGH a dot keeps the output it left the real
                // source on — the dot is routing, not a socket of its own.
                let out = self.out_of(*d, *i);
                if out != 0 {
                    g.edge_out.insert((*d, *i), out);
                }
            }
        }
        // Upstream of a time node, animation runs on THAT node's clock: the
        // graph is complete now, so ask it what time reaches each node and
        // re-evaluate the animated ones there. `apply_f` is unrounded, so a
        // warp landing between frames still moves curves smoothly.
        if let Some(t) = t {
            let patched: std::collections::HashSet<u64> = self
                .patch
                .links
                .iter()
                .filter_map(|l| match l.to {
                    PDst::Chan(n, _) => Some(n),
                    #[allow(unreachable_patterns)]
                    _ => None,
                })
                .collect();
            // A blending retime dissolves between the two frames its
            // sub-frame time falls between; the weight is a function of the
            // tick, so it is filled per build like the remap's lookup.
            let maps = g.node_time_maps();
            let blend_w: Vec<(u64, f32)> = maps
                .iter()
                .filter_map(|(id, outer)| {
                    let n = self.nodes.get(id)?;
                    if n.kind.time_interp() != engine::graph::TimeInterp::Blend {
                        return None;
                    }
                    // The fraction the node's INPUT sits at, i.e. its own map
                    // stacked on whatever clock reaches it (a Frame Fix's:
                    // how far the bad frame is between its good ones).
                    // Which two plate frames a retime dissolves is decided
                    // per WHOLE tick: the readers decode integer frames, so
                    // there is no half frame for a sub-sample to land on.
                    Some((*id, g.nodes.get(id)?.blend_weight_at(outer, t_whole.unwrap_or(0))))
                })
                .collect();
            for (id, w) in blend_w {
                match g.nodes.get_mut(&id) {
                    Some(NodeKind::Speed(p)) => p.blend_w = w,
                    Some(NodeKind::RangeRemap(p)) => p.blend_w = w,
                    Some(NodeKind::TimeRemap(p)) => p.blend_w = w,
                    Some(NodeKind::FrameFix(p)) => p.blend_w = w,
                    _ => {}
                }
            }
            for (id, map) in maps {
                if map.is_identity() {
                    continue;
                }
                let Some(n) = self.nodes.get(&id) else { continue };
                // Patched channels are driven, not keyed; leave them on the
                // global clock rather than clobbering the patch.
                if !n.anim.values().any(|c| !c.keys.is_empty()) || patched.contains(&id) {
                    continue;
                }
                // Re-evaluate IN PLACE. Cloning the authored kind here would
                // throw away everything the build already derived for this
                // tick — a TimeRemap's sampled lut, a Noise's time, a Layer's
                // gate — and those are exactly the nodes people put above a
                // time node.
                let st = map.apply_f(t as f32);
                let Some(kind) = g.nodes.get_mut(&id) else { continue };
                crate::anim::apply_at(kind, &n.anim, st);
                // Time-dependent derived state follows the warped clock too,
                // or a warped branch would animate on one clock and evolve on
                // another.
                match kind {
                    NodeKind::Noise(p) => p.time = st,
                    NodeKind::Sdf(p) => p.time = st,
                    NodeKind::Planet(p) => p.time = st,
                    NodeKind::Grain(p) => p.time = st,
                    NodeKind::Plasma(p) => p.time = st,
                    NodeKind::Particles(p) => p.time = st,
                    NodeKind::Burnin(p) => p.time = st,
                    NodeKind::Expression(p) => p.time = st,
                    NodeKind::Layer(p) => {
                        let tick = st.round() as i64;
                        for l in &mut p.layers {
                            l.gate = if l.visible_at(tick) { 1.0 } else { 0.0 };
                        }
                    }
                    _ => {}
                }
            }
        }
        // A multi-band LumaKey is N mattes on N output ports up here; the
        // engine only ever sees single-output nodes, so lower it the same
        // way tools and dots are lowered — before anything downstream looks.
        // Inert while every node has one output, which is every graph that
        // does not use the feature.
        engine::lumabands::expand(&mut g);
        // A Blender node's passes are N pictures on N outputs; same trick.
        engine::blender::expand(&mut g);
        // A Layer stack's per-layer outputs are one-layer stacks; same trick.
        engine::layerouts::expand(&mut g);
        // An Extract is N Masks sharing an image; same trick.
        engine::extract::expand(&mut g);
        // A roto container is its shapes and ops as plain Splines and
        // MatteCombines; same trick, after animation has been applied.
        engine::roto::expand(&mut g);
        // A Render3D is one stage over its cards' pictures; the cameras and
        // cards on its scene ports become its uniforms. Motion blur asks for
        // the poses across the shutter: the curves' travel from this frame,
        // added to the pose this build already resolved (patches included).
        // A Card / Blend 3D on the viewer becomes one first: a Render3D
        // through the viewer's 3D eye (`view3dui`).
        crate::view3dui::lower(&mut g, self.nodes.keys().copied().max().unwrap_or(0));
        if g.nodes.values().any(|k| matches!(k, NodeKind::Render3D(_))) {
            let now: std::collections::HashMap<u64, NodeKind> = g
                .nodes
                .iter()
                .filter(|(_, k)| matches!(k, NodeKind::Camera3D(_) | NodeKind::Card3D(_)))
                .map(|(id, k)| (*id, k.clone()))
                .collect();
            let tf = t.unwrap_or(0.0);
            let pose = |id: u64, dt: f32| -> Option<NodeKind> {
                let n = self.nodes.get(&id)?;
                // Read from a source: the source at the sample's own time.
                let sourced = match &n.kind {
                    NodeKind::Camera3D(p) => !p.source.is_wobble(),
                    NodeKind::Card3D(p) => !p.source.is_wobble(),
                    _ => false,
                };
                if sourced {
                    let mut k = n.kind.clone();
                    crate::anim::apply_at(&mut k, &n.anim, tf + dt);
                    engine::scene3d::apply_file_source(&mut k, tf + dt).ok()?;
                    return Some(k);
                }
                if !n.anim.values().any(|c| !c.keys.is_empty()) {
                    return None;
                }
                let chans = match n.kind {
                    NodeKind::Camera3D(_) => crate::nodes::scene3d::CAMERA_CHANNELS,
                    NodeKind::Card3D(_) => crate::nodes::scene3d::CARD_CHANNELS,
                    _ => return None,
                };
                let (mut a, mut b) = (n.kind.clone(), n.kind.clone());
                crate::anim::apply_at(&mut a, &n.anim, tf);
                crate::anim::apply_at(&mut b, &n.anim, tf + dt);
                let mut out = now.get(&id)?.clone();
                for ch in chans {
                    if let (Some(v0), Some(v1), Some(v)) =
                        (crate::anim::get(&a, *ch), crate::anim::get(&b, *ch), crate::anim::get(&out, *ch))
                    {
                        crate::anim::set(&mut out, *ch, v + (v1 - v0));
                    }
                }
                Some(out)
            };
            engine::scene3d::expand(&mut g, &pose);
        }
        // An Env's camera port: the Camera's rotation and lens (sources
        // applied above) land on the node, and the scene wire goes.
        engine::nodes::env::lower(&mut g);
        // A Refract's, the same.
        engine::nodes::refract::lower(&mut g);
        // A Starfield's camera becomes its uniforms, the wire dropped.
        engine::nodes::starfield::lower(&mut g);
        // A Splat's camera lands on the node, the wire dropped; its depth
        // output becomes a node of its own where it is used.
        engine::nodes::splat::lower(&mut g);
        lower_crop_bakes(&mut g);
        g
    }

    /// A tool body's exposed parameter, as bound for the instance currently
    /// being expanded. Anywhere else `params` is empty and this reads 0 —
    /// a Param jack means nothing in a comp's own graph.
    fn param_value(&self, ch: crate::anim::Channel) -> f32 {
        match ch {
            crate::anim::Channel::Custom(i) => self
                .params
                .get(&crate::anim::interned(i))
                .copied()
                .unwrap_or(0.0),
            _ => 0.0,
        }
    }

    /// Evaluate one value source against the anim-evaluated graph. `stack`
    /// guards cycles: a source seen twice yields its raw (unpatched) value.
    pub fn patch_value(
        &self,
        src: PSrc,
        base: &std::collections::BTreeMap<u64, NodeKind>,
        t: f32,
        stack: &mut Vec<PSrc>,
    ) -> f32 {
        let raw = |src: PSrc| match src {
            PSrc::Chan(n, ch) => base
                .get(&n)
                .and_then(|k| crate::anim::get(k, ch))
                .unwrap_or(0.0),
            PSrc::Mod(_) => 0.0,
            PSrc::Param(ch) => self.param_value(ch),
        };
        if stack.contains(&src) {
            return raw(src);
        }
        stack.push(src);
        let v = match src {
            // A param is a leaf: it is fed by the instance, not by the body.
            PSrc::Param(ch) => self.param_value(ch),
            PSrc::Chan(n, ch) => match self.patch.link_to(PDst::Chan(n, ch)) {
                Some(l) => self.patch_value(l.from, base, t, stack),
                None => raw(src),
            },
            PSrc::Mod(id) => match self.patch.mods.iter().find(|m| m.id == id) {
                None => 0.0,
                Some(m) => {
                    let inp = |slot: u8, stack: &mut Vec<PSrc>| {
                        self.patch
                            .link_to(PDst::Mod(id, slot))
                            .map(|l| self.patch_value(l.from, base, t, stack))
                            .unwrap_or(0.0)
                    };
                    // `b` is its jack when wired, the node's constant when
                    // not.
                    let b_or = |k: f32, stack: &mut Vec<PSrc>| {
                        self.patch
                            .link_to(PDst::Mod(id, 1))
                            .map(|l| self.patch_value(l.from, base, t, stack))
                            .unwrap_or(k)
                    };
                    match &m.kind {
                        ModKind::Mult(k) => inp(0, stack) * b_or(*k, stack),
                        ModKind::Add(k) => inp(0, stack) + b_or(*k, stack),
                        ModKind::Max(k) => inp(0, stack).max(b_or(*k, stack)),
                        ModKind::Min(k) => inp(0, stack).min(b_or(*k, stack)),
                        ModKind::Expr(x) => {
                            let (a, b) = (inp(0, stack), inp(1, stack));
                            // `../node/param` references read through the
                            // patch too, so a driven channel chains (and the
                            // same stack guards the cycle). See `chanexpr`.
                            let here = crate::chanexpr::here_of(self, id);
                            crate::exprs::eval_with(x, a, b, t, &mut |p| {
                                let (n, ch) = crate::chanexpr::resolve(self, p, here).ok()?;
                                Some(self.patch_value(PSrc::Chan(n, ch), base, t, stack))
                            })
                            .unwrap_or(0.0)
                        }
                    }
                }
            },
        };
        stack.pop();
        v
    }

    /// Default comp, laid out top -> down.
    pub fn default_graph(path_a: &str, path_b: &str, filter: Option<NodeKind>) -> Self {
        let mut g = Self::default();
        // A NEW comp works in ACEScg. Existing setups keep whatever they
        // recorded (or Linear Rec.709 if they predate the field).
        g.working = engine::color::Colorspace::ACESCG;
        let ra = g.add(NodeKind::read(0, path_a), Pos2::new(-140.0, -240.0));
        let rb = g.add(NodeKind::read(1, path_b), Pos2::new(140.0, -240.0));
        let tf = g.add(NodeKind::Transform(TransformParams::default()), Pos2::new(-140.0, -120.0));
        let gr = g.add(NodeKind::Grade(GradeParams::default()), Pos2::new(-140.0, 0.0));
        let mg = g.add(NodeKind::blend(), Pos2::new(0.0, 150.0));

        g.connect(tf, 0, ra);
        g.connect(gr, 0, tf);
        if let Some(f) = filter {
            let fl = g.add(f, Pos2::new(-140.0, 80.0));
            g.connect(fl, 0, gr);
            g.connect(mg, 0, fl);
        } else {
            g.connect(mg, 0, gr);
        }
        g.connect(mg, 1, rb);
        g.view = Some(mg);
        g
    }
}

/// Sample an animation curve into (timeline frame, source frame) points.
///
/// Quarter-frame steps: readers only ever ask for whole frames, but animation
/// upstream of the remap is sampled between them, and a linear walk over
/// quarter frames follows a smoothed or bezier curve closely enough that the
/// difference is far below a frame. Long spans fall back to whole frames so
/// the table can never blow up.
fn sample_time_curve(c: &crate::anim::AnimCurve) -> Vec<[f32; 2]> {
    let (Some(first), Some(last)) = (c.keys.first(), c.keys.last()) else {
        return Vec::new();
    };
    let (f0, f1) = (first.frame as f32, last.frame as f32);
    let span = (f1 - f0).max(0.0);
    let step = if span * 4.0 <= 4096.0 { 0.25 } else { (span / 4096.0).ceil() };
    let n = (span / step).ceil() as usize;
    let mut out = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let f = (f0 + i as f32 * step).min(f1);
        out.push([f, c.eval_at(f).unwrap_or(f)]);
    }
    out
}

/// Is `to` reachable by walking UPSTREAM from `from`? Used to reject
/// magnet pairings that would close a loop.
pub(crate) fn reaches(g: &EditorGraph, from: u64, to: u64) -> bool {
    // Index the feeds once: scanning every edge per visited node made one
    // call O(nodes x edges), and the magnet used to make thousands a frame.
    let mut feeds: std::collections::HashMap<u64, Vec<u64>> = std::collections::HashMap::new();
    for (d, _, s) in &g.edges {
        feeds.entry(*d).or_default().push(*s);
    }
    let mut stack = vec![from];
    let mut seen = std::collections::HashSet::new();
    while let Some(cur) = stack.pop() {
        if cur == to {
            return true;
        }
        if !seen.insert(cur) {
            continue;
        }
        if let Some(f) = feeds.get(&cur) {
            stack.extend(f.iter().copied());
        }
    }
    false
}

/// Does connecting `src`'s output into `dst`'s `input` need a converter
/// spliced in between, and if so which node?
///
/// Mask inputs only accept mask outputs, so dragging an image at one used to
/// simply refuse. Rather than teach every user that the answer is a LumaKey,
/// the editor drops a ToMatte in the gap: the gesture you tried is the
/// gesture that works.
pub fn bridge_for(
    src: &NodeKind,
    out: u32,
    dst: &NodeKind,
    input: usize,
) -> Option<NodeKind> {
    // Per SOCKET, not per node: a multi-output node is judged by the output
    // the wire actually left.
    let sk = src.output_kind_at(out as usize);
    if dst.accepts(input, sk) {
        return None; // already compatible
    }
    match (sk, dst.input_kind(input)) {
        (PortKind::Image, PortKind::Mask) => {
            Some(NodeKind::ToMatte(engine::graph::ToMatteParams::default()))
        }
        // A multi-channel cable at an image input: pick a channel. The slot
        // is minted lazily by the app (usize::MAX until then).
        (PortKind::Channels, PortKind::Image) => {
            Some(NodeKind::ChannelSelect(Default::default()))
        }
        // An image at a DATA port: pick a channel, at full depth. The
        // same gesture as the ToMatte bridge and a different node, because
        // the two differ in exactly the thing that matters here — ToMatte
        // would clamp the measurement to 0..1 on its way through.
        (PortKind::Image, PortKind::Data) => {
            Some(NodeKind::ToData(engine::graph::ToDataParams::default()))
        }
        // Everything else stays refused. A mask at an image input has no
        // obvious right answer, and DATA at a matte input has one that is
        // wrong on somebody's shot — which is the range to squeeze it into
        // (see `PortKind::can_feed`).
        _ => None,
    }
}

/// Wire `src` into `dst`'s `input`, inserting a converter in the gap when the
/// ports don't match but a conversion exists. Returns the node it made, if
/// any. Refuses self-connections and pairs with no bridge, exactly as the
/// plain connect did.
/// Cut a routing dot into `edge` at graph point `at`, preserving which
/// OUTPUT of the real source the wire left on (both new hops carry it, so
/// the engine resolver and a later re-drag both still see it). Returns the
/// dot's id. Shared by `.`/shift-click/the context menu, so the three ways
/// of asking agree exactly.
pub(crate) fn cut_dot(graph: &mut EditorGraph, edge: (u64, usize, u64), at: Pos2) -> u64 {
    let (dst, input, src) = edge;
    let kind = graph.node(dst).map(|n| n.kind.input_kind(input)).unwrap_or(PortKind::Image);
    // Match the wire's own scale, so a dot on a shrunken corner of the
    // graph stays proportionate.
    let sz = graph.node(dst).map(|d| d.size).unwrap_or(1.0);
    let out = graph.out_of(dst, input);
    let id = graph.add(NodeKind::dot(kind), at);
    if let Some(n) = graph.nodes.get_mut(&id) {
        n.size = sz;
    }
    graph.connect_out(id, 0, src, out);
    graph.connect_out(dst, input, id, out);
    id
}

/// Nodes on the clipboard: what a copy writes and a paste rebuilds. JSON, so
/// it survives a trip through the system clipboard — between sessions, into
/// a text file and back — and anything else on the clipboard fails the parse
/// and is left alone.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct NodeClip {
    /// Format tag and version.
    wobble_nodes: u32,
    nodes: Vec<(u64, ENode)>,
    /// Wires BETWEEN copied nodes: (dst, input, src, src output). Wires in
    /// from outside the copy are not carried: a paste is a fresh branch.
    edges: Vec<(u64, usize, u64, u32)>,
    /// Which of those were drawn hidden (the region layer).
    hidden: Vec<(u64, usize)>,
}

/// Serialize `ids` (plus a harp's other bar, so a pair travels whole) for
/// the clipboard. `None` when there is nothing to copy.
pub(crate) fn copy_nodes(graph: &EditorGraph, ids: &[u64]) -> Option<String> {
    let kids = crate::breakout::kid_set(graph);
    let mut set: Vec<u64> = ids.to_vec();
    set.extend(crate::breakout::companions(graph, ids));
    set.sort_unstable();
    set.dedup();
    set.retain(|id| graph.node(*id).is_some() && !kids.contains(id));
    if set.is_empty() {
        return None;
    }
    let clip = NodeClip {
        wobble_nodes: 1,
        nodes: set.iter().map(|id| (*id, graph.nodes[id].clone())).collect(),
        edges: graph
            .edges
            .iter()
            .filter(|(d, _, s)| set.contains(d) && set.contains(s))
            .map(|(d, i, s)| (*d, *i, *s, graph.out_of(*d, *i)))
            .collect(),
        hidden: graph
            .hidden_wires
            .iter()
            .filter(|(d, _)| set.contains(d))
            .copied()
            .collect(),
    };
    serde_json::to_string(&clip).ok()
}

/// Where a baking Crop's LIVE crop sits in the engine graph: its own id
/// goes to the Bake that plays the cache (`lower_crop_bakes`).
pub const CROP_LIVE: u64 = 1 << 62;

pub fn crop_live_id(crop: u64) -> u64 {
    CROP_LIVE | crop
}

/// The Bake a baking Crop lowers to: the Bake node's own kind, so its
/// codegen, cost and reader handling are the Bake's. A matte crop is a
/// matte bake.
pub fn crop_bake_kind(slot: usize, mask: bool) -> NodeKind {
    NodeKind::GraphBake { source: slot, baked: false, start: None, mask, window: None, matte_half: false, single: false }
}

/// A Crop that bakes itself (`SoftCropParams::bake`) is, to the engine, a
/// live Crop (moved to `crop_live_id`) under a Bake that keeps the Crop's
/// id. Every consumer and the viewer already point at that id, so they
/// read the Bake; the app's bake state (`loaded`, the bake job, the
/// transients `apply_bake_state` sets) is keyed by the Crop's id and so
/// lands on the Bake with no mapping. A bypassed Crop is left alone: its
/// pixels are its input's, and so would a bake's be.
fn lower_crop_bakes(g: &mut Graph) {
    let hosts: Vec<(u64, usize, bool)> = g
        .nodes
        .iter()
        .filter(|(id, _)| !g.disabled.contains(id))
        .filter_map(|(id, k)| match k {
            NodeKind::SoftCrop(p) => p.bake.map(|s| (*id, s, p.matte)),
            _ => None,
        })
        .collect();
    for (id, slot, mask) in hosts {
        let live = crop_live_id(id);
        let Some(crop) = g.nodes.insert(id, crop_bake_kind(slot, mask)) else { continue };
        g.nodes.insert(live, crop);
        if let Some(src) = g.edges.remove(&(id, 0)) {
            g.edges.insert((live, 0), src);
            if let Some(o) = g.edge_out.remove(&(id, 0)) {
                g.edge_out.insert((live, 0), o);
            }
        }
        g.edges.insert((id, 0), live);
    }
}

/// A pasted node must not share a reader slot with the node it was copied
/// from — two nodes on one slot fight over one texture. Plain readers take
/// the next free slot, as an add does; the kinds the app MINTS (a
/// ChannelSelect, a Cryptomatte's ranks) go back to unminted.
pub(crate) fn reslot(kind: &mut NodeKind, next_read_slot: &mut usize) {
    match kind {
        // A baking Crop's copy bakes on its own slot.
        NodeKind::SoftCrop(p) if p.bake.is_some() => {
            p.bake = Some(*next_read_slot);
            *next_read_slot += 1;
        }
        NodeKind::Read { source, .. }
        | NodeKind::GraphBake { source, .. } => {
            *source = *next_read_slot;
            *next_read_slot += 1;
        }
        NodeKind::GenAi(p) => {
            p.source = *next_read_slot;
            *next_read_slot += 1;
        }
        NodeKind::CorridorKey(p) => {
            p.source = *next_read_slot;
            *next_read_slot += 1;
        }
        NodeKind::Sammie(p) => {
            p.source = *next_read_slot;
            *next_read_slot += 1;
        }
        NodeKind::Flow(p) => {
            p.source = *next_read_slot;
            *next_read_slot += 1;
        }
        NodeKind::MorphDissolve(p) => {
            p.source = *next_read_slot;
            *next_read_slot += 1;
        }
        NodeKind::Rifetime(p) => {
            p.source = *next_read_slot;
            *next_read_slot += 1;
        }
        NodeKind::FrameFix(p) => {
            p.source = *next_read_slot;
            *next_read_slot += 1;
        }
        NodeKind::ChannelSelect(p) => p.source = usize::MAX,
        NodeKind::Cryptomatte(c) => {
            c.sources.iter_mut().for_each(|s| *s = usize::MAX);
            c.matte_source = usize::MAX;
        }
        _ => {}
    }
}

/// Rebuild a clip in `graph`, centred on `at`: fresh ids, the copied wires
/// between them, fresh reader slots. Returns the new ids, or `None` when
/// `text` is not a node clip.
pub(crate) fn paste_nodes(
    graph: &mut EditorGraph,
    text: &str,
    at: Pos2,
    next_read_slot: &mut usize,
) -> Option<Vec<u64>> {
    let clip: NodeClip = serde_json::from_str(text).ok()?;
    if clip.wobble_nodes != 1 || clip.nodes.is_empty() {
        return None;
    }
    let bb = clip
        .nodes
        .iter()
        .fold(Rect::NOTHING, |r, (_, n)| r.union(Rect::from_center_size(n.pos, Vec2::ZERO)));
    let shift = at - bb.center();
    let mut map = std::collections::HashMap::new();
    let mut ids = Vec::new();
    for (old, n) in clip.nodes {
        let mut n = n;
        n.pos += shift;
        reslot(&mut n.kind, next_read_slot);
        let id = graph.next_id;
        graph.next_id += 1;
        graph.nodes.insert(id, n);
        map.insert(old, id);
        ids.push(id);
    }
    for (d, i, s, o) in clip.edges {
        if let (Some(d), Some(s)) = (map.get(&d), map.get(&s)) {
            graph.connect_out(*d, i, *s, o);
        }
    }
    for (d, i) in clip.hidden {
        if let Some(d) = map.get(&d) {
            if graph.input_of(*d, i).is_some() {
                graph.hidden_wires.insert((*d, i));
            }
        }
    }
    Some(ids)
}

/// What a held-key swipe of a dragged node across another makes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SwipeOp {
    /// B: the dragged node OVER the swiped-onto one.
    Blend,
    /// M with an image in hand: a Blend like B's, set to MULTIPLY.
    Multiply,
    /// I (and M with a matte in hand): the swiped-onto image ISOLATED by
    /// the dragged matte — the usual "mask", image × matte.
    In,
    /// O: the dragged matte CUT OUT of the swiped-onto image — the
    /// usual "stencil", image × (1 − matte).
    Out,
}

impl SwipeOp {
    /// The swipe needs a matte in hand (I, O) — the dragged node must
    /// output one.
    pub(crate) fn wants_matte(self) -> bool {
        !matches!(self, SwipeOp::Blend | SwipeOp::Multiply)
    }
}

/// Make the swipe's node at `at`, wired from `dragged` and `under`, and let
/// it TAKE `under`'s place downstream — a merge onto the B pipe, so
/// the comp shows the result at once. Returns the new node.
///
/// - Blend: a Blend, `dragged` on + (over), `under` on − (under).
/// - In / Out: a Blend in REPLACE with the matte on its matte input, whose
///   maths is `mix(B, A, matte)`: image on + with − empty is image × matte
///   (mask), image on − with + empty is image × (1 − matte) (stencil). One
///   node for both, so the graph shows the same shape for the same idea,
///   labelled with the name a compositor's hand looks for.
///
/// A consumer is left alone when rerouting it would close a loop (it feeds
/// `dragged`), when it reads another of `under`'s outputs, or when it
/// wants a matte or data rather than a picture.
pub(crate) fn swipe_op(
    graph: &mut EditorGraph,
    op: SwipeOp,
    dragged: u64,
    under: u64,
    at: Pos2,
    zoom: f32,
) -> u64 {
    swipe_op_on(graph, op, dragged, under, None, at, zoom)
}

/// `swipe_op`, but with `only = Some((dst, input))` the swipe landed on ONE
/// noodle out of `under`: just that consumer is rerouted, `under`'s other
/// consumers keep the plain image.
pub(crate) fn swipe_op_on(
    graph: &mut EditorGraph,
    op: SwipeOp,
    dragged: u64,
    under: u64,
    only: Option<(u64, usize)>,
    at: Pos2,
    zoom: f32,
) -> u64 {
    let consumers: Vec<(u64, usize, u32)> = graph
        .edges
        .iter()
        .filter(|(d, i, s)| *s == under && only.is_none_or(|o| o == (*d, *i)))
        .map(|(d, i, _)| (*d, *i, graph.out_of(*d, *i)))
        .collect();
    let size = graph.node(under).map(|n| n.size).unwrap_or(1.0);
    let kind = match op {
        SwipeOp::Blend => NodeKind::blend(),
        SwipeOp::Multiply => {
            NodeKind::Blend { op: engine::graph::BlendOp::Multiply, mix: 1.0, gain: [1.0, 1.0] }
        }
        SwipeOp::In | SwipeOp::Out => {
            NodeKind::Blend { op: engine::graph::BlendOp::Replace, mix: 1.0, gain: [1.0, 1.0] }
        }
    };
    let b = graph.add(kind, at);
    if let Some(n) = graph.nodes.get_mut(&b) {
        n.size = size;
        n.ui_label = match op {
            SwipeOp::Blend | SwipeOp::Multiply => String::new(),
            SwipeOp::In => "mask".into(),
            SwipeOp::Out => "stencil".into(),
        };
    }
    match op {
        SwipeOp::Blend | SwipeOp::Multiply => {
            connect_or_bridge(graph, dragged, 0, b, 0, zoom);
            connect_or_bridge(graph, under, 0, b, 1, zoom);
        }
        SwipeOp::In => {
            connect_or_bridge(graph, under, 0, b, 0, zoom);
            connect_or_bridge(graph, dragged, 0, b, 2, zoom);
        }
        SwipeOp::Out => {
            connect_or_bridge(graph, under, 0, b, 1, zoom);
            connect_or_bridge(graph, dragged, 0, b, 2, zoom);
        }
    }
    for (d, i, out) in consumers {
        let picture = graph.node(d).map(|n| n.kind.input_kind(i)) == Some(PortKind::Image);
        if out != 0 || !picture || d == dragged || reaches(graph, dragged, d) {
            continue;
        }
        graph.connect(d, i, b);
    }
    b
}

/// Land a noodle grabbed with `G` on the port it was dropped over. Two
/// outputs (or two inputs) are not a wire, so those simply do nothing —
/// dropping on the wrong sort of port should not silently rewire the comp.
pub(crate) fn land_grab(graph: &mut EditorGraph, from: Port, to: Port, zoom: f32) -> bool {
    let (src, out, dst, input) = match (from, to) {
        (Port::Out(s, o), Port::In(d, i)) | (Port::In(d, i), Port::Out(s, o)) => {
            (s, o, d, i)
        }
        _ => return false,
    };
    // A would-be cycle is not refused any more: connect_or_bridge lowers it
    // to a LoopIn/LoopOut pair (see `insert_loop`).
    connect_or_bridge(graph, src, out, dst, input, zoom);
    true
}

/// The wire the DAG refuses is the loop the user meant: `src`'s output fed
/// back into an UPSTREAM `dst.input` lowers to a LoopIn/LoopOut pair — the
/// tap replaces the back-edge at `dst.input` (whatever fed it becomes the
/// SEED), and everything that consumed `src` now consumes the LoopOut.
/// Image ports only; returns (loop_in, loop_out).
pub fn insert_loop(
    graph: &mut EditorGraph,
    src: u64,
    dst: u64,
    input: usize,
) -> Option<(u64, u64)> {
    use engine::graph::LoopParams;
    let dk = graph.node(dst)?.kind.clone();
    let sk = graph.node(src)?.kind.clone();
    if !dk.input_kind(input).is_picture() || !sk.output_kind().is_picture() {
        return None;
    }
    let sp = graph.node(src)?.pos;
    let dp = graph.node(dst)?.pos;
    let li = graph.add(NodeKind::LoopIn, Pos2::new(dp.x - 70.0, dp.y - 60.0));
    let lo = graph.add(
        NodeKind::LoopOut(LoopParams { pair: li, ..Default::default() }),
        Pos2::new(sp.x, sp.y + 80.0),
    );
    // Whatever fed the port becomes the seed.
    let seed = graph.edges.iter().find(|(d, i, _)| *d == dst && *i == input).map(|e| e.2);
    if let Some(seed) = seed {
        graph.connect(lo, 1, seed);
    }
    // Everything that consumed src now consumes the loop's result.
    let consumers: Vec<(u64, usize)> = graph
        .edges
        .iter()
        .filter(|(d, _, s)| *s == src && *d != lo)
        .map(|(d, i, _)| (*d, *i))
        .collect();
    for (d, i) in consumers {
        let out = graph.out_of(d, i);
        graph.connect_out(d, i, lo, out.min(0));
    }
    graph.connect(lo, 0, src);
    graph.connect(dst, input, li);
    Some((li, lo))
}

pub fn connect_or_bridge(
    graph: &mut EditorGraph,
    src: u64,
    out: u32,
    dst: u64,
    input: usize,
    zoom: f32,
) -> Option<u64> {
    // A wire back into the upstream is not refused — it becomes a loop.
    if src == dst || reaches(graph, src, dst) {
        return insert_loop(graph, src, dst, input).map(|(_, lo)| lo);
    }
    // A node that follows its pipe (a Bake, a SoftCrop, an Invert...)
    // takes whichever pixel pipe it is dropped into — image or matte — and
    // `connect_out` retypes it to match.
    if let (Some(sk), Some(dk)) = (graph.node(src), graph.node(dst)) {
        let s_kind = sk.kind.output_kind_at(out as usize);
        if dk.kind.follows_pipe(input) && matches!(s_kind, PortKind::Image | PortKind::Mask) {
            graph.connect_out(dst, input, src, out);
            return None;
        }
    }
    if src == dst {
        return None;
    }
    let (sk, sp) = graph.node(src).map(|n| (n.kind.clone(), n.pos))?;
    let (dk, dp) = graph.node(dst).map(|n| (n.kind.clone(), n.pos))?;
    if dk.accepts(input, sk.output_kind_at(out as usize)) {
        graph.connect_out(dst, input, src, out);
        return None;
    }
    let bridge = bridge_for(&sk, out, &dk, input)?;
    // Halfway along the wire the user just drew, so it lands where they were
    // already looking.
    let at = sp + (dp - sp) * 0.5;
    let id = graph.add(bridge, at);
    let size = new_node_size(graph, at, zoom, &[id]);
    if let Some(n) = graph.nodes.get_mut(&id) {
        n.size = size;
    }
    graph.connect(id, 0, src);
    graph.connect(dst, input, id);
    Some(id)
}
