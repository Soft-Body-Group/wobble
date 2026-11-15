//! The app's side of the scripting protocol (`api` crate): every method in
//! `api::methods()` answered against the live document.
//!
//! Calls arrive on a channel from the socket threads and are drained at the
//! TOP of `ui()` — before the cost flow, the editor and the panels — so an
//! edit made by a script is drawn, coloured and rendered in the same frame a
//! hand edit would be. Nothing here runs unless a call is waiting.
//!
//! The rules the protocol promises, and where they are kept:
//! - **Authored data only.** Values go in through `anim::set` on the
//!   authored kind or as keys on its curves — never into the transient
//!   engine graph.
//! - **One call, one undo step.** `api_drain` records any hand edit still
//!   pending, runs the call, and pushes the graph as it was before it.
//! - **Locks and cables hold.** A locked channel takes nothing and a
//!   cable-driven one is refused, as the gizmos refuse them.
//!
//! This is `impl Gfx` (the app state) from a child module, so it reads the app's private
//! state without lib.rs growing another thousand lines.

use crate::anim::{self, Channel};
use crate::editor::{self, EditorGraph, ModKind, PDst, PSrc};
use crate::Gfx;
use api::server::{Call, Host};
use api::ApiError;
use engine::graph::NodeKind;
use serde_json::{json, Map, Value};
use web_time::{Duration, Instant};

#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod deliver;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod farm;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod peers;

/// The user has sharing on (`peers`): the UI pass must run even while the
/// window is covered, because that pass opens the network API.
pub(crate) fn wants_network() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    return peers::wants_share();
    #[cfg(target_arch = "wasm32")]
    false
}
#[cfg(not(target_arch = "wasm32"))]
mod remote;

/// A `viewer_capture` waiting for its frame.
struct Capture {
    call: Call,
    what: What,
    /// UI passes seen since the request: the edit that preceded it has to
    /// reach a render before the picture means anything.
    passes: u32,
    deadline: Instant,
}

/// What a waiting call wants from the rendered frame.
enum What {
    /// `viewer_capture`: the display picture, longest side at most this.
    Png(u32),
    /// `pixel_sample`: scene-linear statistics over x, y, w, h.
    Sample([Option<u64>; 4]),
}

/// Redraws a capture waits at least: the pass that made the edit, the one
/// that rendered it, and one for luck on a rig rebuild.
const SETTLE_PASSES: u32 = 3;
/// How long a `node_run` waits for the playback rig before running anyway.
const RIG_TIMEOUT: Duration = Duration::from_secs(10);
/// A capture that never settles (a plate that will not decode) is taken
/// anyway after this, and says so.
const SETTLE_TIMEOUT: Duration = Duration::from_secs(20);

/// An `addon_events` long-poll: answered the moment its addon has
/// something to say, or empty at its deadline.
struct Poll {
    call: Call,
    addon: String,
    deadline: Instant,
}

pub struct ApiServe {
    host: Host,
    captures: Vec<Capture>,
    polls: Vec<Poll>,
    /// `node_run` calls waiting for the playback rig: an edit rebuilds it,
    /// and a render started in the same breath as the edit that wired it
    /// would be refused with "load a sequence first".
    runs: Vec<(Call, Instant)>,
    /// The properties panel drawn for another screen (`remote`).
    #[cfg(not(target_arch = "wasm32"))]
    remote: Option<remote::Remote>,
    /// The API opened to the network (`peers`, the user's switch): its
    /// calls, and the switch that closes it.
    #[cfg(not(target_arch = "wasm32"))]
    net: Option<(Host, api::server::NetHandle)>,
}

impl ApiServe {
    /// Listen on `port`. A port already taken is logged and the app runs
    /// without the API rather than refusing to start.
    pub fn start(port: u16) -> Option<Self> {
        match api::server::serve(port) {
            Ok(host) => {
                log::warn!("scripting API listening on 127.0.0.1:{}", host.port);
                API_PORT.store(host.port, std::sync::atomic::Ordering::Relaxed);
                // The remote panel rides with the API: the next port up.
                #[cfg(not(target_arch = "wasm32"))]
                let remote = {
                    let rport = std::env::var("WOBBLE_REMOTE_PORT")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(if port == 0 { 0 } else { host.port.wrapping_add(1) });
                    remote::Remote::start(rport)
                };
                Some(Self {
                    host,
                    captures: Vec::new(),
                    polls: Vec::new(),
                    runs: Vec::new(),
                    #[cfg(not(target_arch = "wasm32"))]
                    remote,
                    #[cfg(not(target_arch = "wasm32"))]
                    net: None,
                })
            }
            Err(e) => {
                log::error!("scripting API: cannot listen on port {port}: {e}");
                None
            }
        }
    }

    /// The port it actually listens on (a start on 0 picks one).
    #[allow(dead_code)]
    pub fn port(&self) -> u16 {
        self.host.port
    }
}

/// This wobble's loopback API port once it serves (0 before): its own
/// share of a shared render goes through it.
static API_PORT: std::sync::atomic::AtomicU16 = std::sync::atomic::AtomicU16::new(0);

type R = Result<Value, ApiError>;

fn int(p: &Value, k: &str) -> Option<u64> {
    p.get(k).and_then(|v| v.as_u64())
}

fn need_id(p: &Value, k: &str) -> Result<u64, ApiError> {
    int(p, k).ok_or_else(|| ApiError::invalid(format!("'{k}' must be a node id")))
}

fn channel(name: &str) -> Result<Channel, ApiError> {
    Channel::from_key(name).ok_or_else(|| ApiError::invalid(format!("unknown channel '{name}'")))
}

fn interp_name(i: anim::Interp) -> String {
    match i {
        anim::Interp::Hold => "hold".into(),
        anim::Interp::Linear => "linear".into(),
        anim::Interp::Smooth => "smooth".into(),
        anim::Interp::Bezier => "bezier".into(),
        // "ease-bounce-out": the family, then the end it eases.
        anim::Interp::Ease(e, d) => format!("ease-{}-{}", e.name(), d.name()),
    }
}

fn interp_of(s: &str) -> Result<anim::Interp, ApiError> {
    Ok(match s {
        "hold" => anim::Interp::Hold,
        "linear" => anim::Interp::Linear,
        "smooth" => anim::Interp::Smooth,
        "bezier" => anim::Interp::Bezier,
        o => {
            let ease = o.strip_prefix("ease-").and_then(|rest| {
                let e = anim::Ease::ALL.into_iter().find(|e| rest.starts_with(&format!("{}-", e.name())))?;
                let d = rest[e.name().len() + 1..].to_string();
                let d = anim::EaseDir::ALL.into_iter().find(|x| x.name() == d)?;
                Some(anim::Interp::Ease(e, d))
            });
            match ease {
                Some(i) => i,
                None => {
                    return Err(ApiError::invalid(format!(
                        "unknown interp '{o}' (hold, linear, smooth, bezier, ease-<sine|quad|cubic|quart|quint|expo|circ|back|bounce|elastic>-<in|out|in-out>)"
                    )))
                }
            }
        }
    })
}

/// A patch end as the protocol writes it.
fn src_json(s: PSrc) -> Value {
    match s {
        PSrc::Chan(n, c) => json!({ "node": n, "channel": c.key() }),
        PSrc::Mod(m) => json!({ "mod": m }),
        PSrc::Param(c) => json!({ "param": c.key() }),
    }
}

fn dst_json(d: PDst) -> Value {
    match d {
        PDst::Chan(n, c) => json!({ "node": n, "channel": c.key() }),
        PDst::Mod(m, slot) => json!({ "mod": m, "slot": slot }),
    }
}

fn mod_json(m: &editor::ModNode) -> Value {
    let (kind, value, expr) = match &m.kind {
        ModKind::Mult(k) => ("mult", json!(k), Value::Null),
        ModKind::Add(k) => ("add", json!(k), Value::Null),
        ModKind::Max(k) => ("max", json!(k), Value::Null),
        ModKind::Min(k) => ("min", json!(k), Value::Null),
        ModKind::Expr(x) => ("expr", Value::Null, json!(x)),
    };
    json!({ "id": m.id, "kind": kind, "value": value, "expr": expr, "x": m.pos[0], "y": m.pos[1] })
}

/// `"$N"` anywhere in a batch call's params becomes the id call N returned.
fn subst(v: &mut Value, results: &[Value]) -> Result<(), ApiError> {
    match v {
        Value::String(s) => {
            let Some(n) = s.strip_prefix('$').and_then(|d| d.parse::<usize>().ok()) else { return Ok(()) };
            let id = results
                .get(n)
                .and_then(|r| r.get("id"))
                .cloned()
                .ok_or_else(|| ApiError::invalid(format!("\"{s}\": call {n} has not run or returned no id")))?;
            *v = id;
            Ok(())
        }
        Value::Array(a) => a.iter_mut().try_for_each(|x| subst(x, results)),
        Value::Object(o) => o.values_mut().try_for_each(|x| subst(x, results)),
        _ => Ok(()),
    }
}

/// The name a node type goes by in the protocol: its catalog label for a
/// built-in, its type id for a plugin or tool.
/// Why a wire from `src`'s output `out` into `dst`'s input `input` would not
/// be made — neither accepted (`NodeKind::accepts`, the one wiring rule) nor
/// bridged by a converter (`editor::bridge_for`). None: it would be.
fn wire_refusal(src: &NodeKind, out: u32, dst: &NodeKind, input: usize) -> Option<String> {
    let gives = src.output_kind_at(out as usize);
    if dst.accepts(input, gives) || editor::bridge_for(src, out, dst, input).is_some() {
        return None;
    }
    Some(format!(
        "{} gives {gives:?}, {} input {input} takes {:?}, and nothing converts one to the other",
        type_name(src),
        type_name(dst),
        dst.input_kind(input)
    ))
}

fn type_name(kind: &NodeKind) -> String {
    match kind {
        NodeKind::Custom { type_id, .. } => type_id.clone(),
        k => {
            let name = k.name();
            editor::node_catalog()
                .into_iter()
                .find(|e| (e.make)(0).name() == name)
                .map(|e| e.label.to_string())
                .unwrap_or_else(|| name.to_string())
        }
    }
}

/// RFC 7386 merge: objects merge key by key, `null` deletes a key, anything
/// else replaces. The delete is what lets a script switch an enum's variant
/// in one call: `{"GeoNode": {"Scatter": null, "Noise": {…}}}`.
fn merge(into: &mut Value, patch: &Value) {
    match (into, patch) {
        (Value::Object(a), Value::Object(b)) => {
            for (k, v) in b {
                if v.is_null() {
                    a.remove(k);
                    continue;
                }
                match a.get_mut(k) {
                    Some(slot) if v.is_object() => merge(slot, v),
                    _ => {
                        a.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (slot, v) => *slot = v.clone(),
    }
}

/// The catalog the open graph takes, as `editor::offered` reads it: inside
/// a Noise, a 2dini or an SDF its own node types; anywhere else the comp's.
fn cats_for(dive: Option<crate::DiveKind>) -> (bool, bool, bool, bool, bool, bool) {
    match dive {
        Some(crate::DiveKind::Noise) => (true, false, false, false, false, false),
        Some(crate::DiveKind::Geo) => (false, true, false, false, false, false),
        Some(crate::DiveKind::Sdf) => (false, false, true, false, false, false),
        Some(crate::DiveKind::Particles) => (false, false, false, true, false, false),
        Some(crate::DiveKind::GenAi) => (false, false, false, false, true, false),
        Some(crate::DiveKind::Roto) => (false, false, false, false, false, true),
        _ => (false, false, false, false, false, false),
    }
}

/// The catalog entry a type name means HERE: the open graph's own types
/// first, so inside a 2dini `Noise` is the 2dini's Noise and `Transform`
/// its Transform, not the comp nodes of the same name.
fn find_entry(want: &str, (n, g, s, p, a, r): (bool, bool, bool, bool, bool, bool)) -> Option<editor::NodeEntry> {
    let mut lower = want.to_ascii_lowercase();
    // The names these types had before they were renamed in the UI.
    if let Some(now) = [("write", "render"), ("shuffle", "reorder"), ("read", "load"), ("scene", "blend 3d"), ("softcrop", "crop"), ("soft crop", "crop")].iter().find(|(was, _)| *was == lower) {
        lower = now.1.to_string();
    }
    let hit = |e: &editor::NodeEntry| {
        e.label.to_ascii_lowercase() == lower || (e.make)(0).name().to_ascii_lowercase() == lower
    };
    let all = editor::node_catalog();
    let own = all.iter().position(|e| editor::entry_offered(e, n, g, s, p, a, r) && hit(e));
    let any = all.iter().position(hit);
    own.or(any).map(|i| all.into_iter().nth(i).expect("indexed"))
}

/// How the graphs that live INSIDE a node are written, for a script that
/// builds one through `node_patch` on its owner (or node by node, dived in).
const SUBGRAPH_NOTES: &str = "A 2dini (type 2dini, kind.Geo.graph), Noise (kind.Noise.graph) or SDF \
(kind.Sdf.graph) owns a graph of its own: {nodes: {\"<id>\": <node kind>}, edges: \
[[[consumer_id, input_index], producer_id], ...], out: <id or null>, layout: {\"<id>\": [x, y]}, \
next_id: <one past the highest id>}. Each node kind is the 'kind' listed below WITHOUT its outer \
wrapper (GeoNode / Field / SdfNode); a unit variant is a bare string (\"Output\"). A 2dini draws \
what its Output node's input 0 carries. 2dini GEOMETRY IS IN CANVAS WIDTHS, not pixels: x runs \
0..1 across the frame, y runs 0..height/width down it (0..0.5625 at 16:9), sizes likewise. \
When the user has dived into one of these graphs, graph_get/node_create/node_connect work on \
its nodes directly (app_info.dive says which), and view_set its Output to see the result.";

/// One sub-graph catalog: each type with its input names and the default
/// node JSON `node_patch` / a graph's `nodes` take.
fn subgraph_types(cats: (bool, bool, bool, bool, bool, bool)) -> Vec<Value> {
    editor::node_catalog()
        .into_iter()
        .filter(|e| editor::entry_offered(e, cats.0, cats.1, cats.2, cats.3, cats.4, cats.5))
        .map(|e| {
            let kind = (e.make)(0);
            let inputs: Vec<&str> = match &kind {
                NodeKind::GeoNode(n) => n.inputs().iter().map(|i| i.name).collect(),
                NodeKind::Field(n) => n.inputs().iter().map(|i| i.name).collect(),
                NodeKind::SdfNode(n) => n.inputs().iter().map(|i| i.name).collect(),
                _ => Vec::new(),
            };
            json!({
                "type": e.label,
                "inputs": inputs,
                "kind": serde_json::to_value(&kind).unwrap_or(Value::Null),
            })
        })
        .collect()
}

/// Box-filter an RGBA8 image down so its longest side is at most `max`.
fn shrink(w: u32, h: u32, rgba: &[u8], max: u32) -> (u32, u32, Vec<u8>) {
    let k = w.max(h).div_ceil(max.max(1)).max(1);
    if k == 1 {
        return (w, h, rgba.to_vec());
    }
    let (ow, oh) = (w / k, h / k);
    let mut out = Vec::with_capacity((ow * oh * 4) as usize);
    for y in 0..oh {
        for x in 0..ow {
            let mut acc = [0u32; 4];
            for sy in y * k..(y + 1) * k {
                for sx in x * k..(x + 1) * k {
                    let i = ((sy * w + sx) * 4) as usize;
                    for c in 0..4 {
                        acc[c] += rgba[i + c] as u32;
                    }
                }
            }
            out.extend(acc.iter().map(|a| (a / (k * k)) as u8));
        }
    }
    (ow, oh, out)
}

fn png(w: u32, h: u32, rgba: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut enc = png::Encoder::new(&mut bytes, w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(rgba)?;
    Ok(bytes)
}

impl Gfx {
    /// Answer every call waiting on the socket, and any capture whose frame
    /// has now rendered. Called once per UI pass, at the top of `ui()`.
    /// Sharing on the network (`peers`): opens the API if no `--api` did,
    /// opens it to the network behind the pairing code, announces it, and
    /// closes all of that when the switch goes off.
    #[cfg(not(target_arch = "wasm32"))]
    fn network_tick(&mut self) {
        let want = peers::wants_share();
        if want && self.api.is_none() {
            self.api = ApiServe::start(api::port_from_env());
        }
        if let Some(a) = self.api.as_mut() {
            let restart = peers::take_restart();
            if a.net.is_some() && (!want || restart) {
                if let Some((_, h)) = a.net.take() {
                    h.close();
                }
                peers::stopped();
            }
            if want && a.net.is_none() {
                match api::server::serve_network(0, peers::code()) {
                    Ok((host, h)) => {
                        peers::listening(host.port);
                        a.net = Some((host, h));
                    }
                    Err(e) => peers::set_error(format!("cannot open the network port: {e}")),
                }
            }
        }
        peers::poll();
    }

    /// `nodes_receive`: a clip from another wobble, pasted beside the
    /// graph, its footage looked for here.
    fn api_receive(&mut self, p: &Value) -> R {
        let clip = p["clip"].as_str().ok_or_else(|| ApiError::invalid("'clip' must be a node clip"))?;
        let from = p["from"].as_str().unwrap_or("another wobble");
        let right = self.egraph.nodes.values().map(|n| n.pos.x).fold(0.0, f32::max) + 200.0;
        let ids = editor::paste_nodes(&mut self.egraph, clip, egui::pos2(right, 0.0), &mut self.next_read_slot)
            .ok_or_else(|| ApiError::invalid("not a node clip"))?;
        self.selected = ids.clone();
        self.adopt_loads();
        self.playback_dirty = true;
        let missing: Vec<Value> = ids
            .iter()
            .filter_map(|id| self.load_errors.get(id).map(|e| json!({ "id": id, "error": e })))
            .collect();
        log::info!("received {} nodes from {from}", ids.len());
        Ok(json!({ "ids": ids, "missing": missing }))
    }

    /// The network window, and what it asks for: send the selected nodes,
    /// or render the selected Render node with paired wobbles.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn network_window(&mut self, ctx: &egui::Context) {
        let render = self.selected.iter().find_map(|id| {
            let n = self.egraph.node(*id)?;
            matches!(n.kind, NodeKind::Write(_)).then(|| (*id, editor::display_label(n)))
        });
        let here = peers::Here { selected: self.selected.len(), render: render.as_ref().map(|r| r.1.clone()) };
        for ask in peers::window(ctx, &here) {
            match ask {
                peers::Ask::Send(id) => {
                    let Some(clip) = editor::copy_nodes(&self.egraph, &self.selected) else { continue };
                    let n = self.selected.len();
                    crate::bg::spawn(move || {
                        let r = (|| {
                            let (p, code) = peers::peer(&id).ok_or("not paired")?;
                            let mut c = peers::Conn::open(&p.addrs, p.port, Some(&code))?;
                            let v = c.call("nodes_receive", json!({ "clip": clip, "from": peers::name() }))?;
                            let missing = v["missing"].as_array().map(|m| m.len()).unwrap_or(0);
                            Ok::<_, String>(if missing == 0 {
                                format!("sent {n} node{}", if n == 1 { "" } else { "s" })
                            } else {
                                format!("sent {n}; {missing} Load{} missing footage there", if missing == 1 { "" } else { "s" })
                            })
                        })();
                        peers::note_peer(&id, r.unwrap_or_else(|e| format!("send failed: {e}")));
                    });
                }
                peers::Ask::Render(ids) => {
                    if let Err(e) = self.farm_start(render.as_ref().map(|r| r.0), ids) {
                        farm::set_note(e);
                    }
                }
            }
        }
    }

    /// Start a shared render of `node` with the paired wobbles `ids`.
    #[cfg(not(target_arch = "wasm32"))]
    fn farm_start(&mut self, node: Option<u64>, ids: Vec<String>) -> Result<(), String> {
        let node = node.ok_or("select a Render node")?;
        let Some(NodeKind::Write(params)) = self.egraph.node(node).map(|n| n.kind.clone()) else {
            return Err("select a Render node".into());
        };
        if !params.is_set() {
            return Err("set the Render node's location first".into());
        }
        // (`self.api` is out for the drain while a script asks this.)
        let port = Some(API_PORT.load(std::sync::atomic::Ordering::Relaxed)).filter(|p| *p != 0).ok_or("turn sharing on first")?;
        // The helpers render the FILE: it must be saved, and be what is on
        // screen.
        let setup = self.current_setup.clone().ok_or("save the setup first (in a folder the helpers can see)")?;
        let on_disk = std::fs::read_to_string(&setup).map_err(|e| format!("{}: {e}", setup.display()))?;
        let now = serde_json::to_string_pretty(&self.setup_save()).map_err(|e| e.to_string())?;
        if on_disk != now {
            return Err("save the setup first: the helpers render the saved file".into());
        }
        let (first, last) = self.write_range(node, &params);
        if last < first {
            return Err("empty frame range".into());
        }
        let local = farm::Helper { name: farm::local_name(), addrs: vec![std::net::Ipv4Addr::LOCALHOST], port, code: None };
        let helpers: Vec<farm::Helper> = ids
            .iter()
            .filter_map(|id| peers::peer(id))
            .map(|(p, code)| farm::Helper { name: p.name, addrs: p.addrs, port: p.port, code: Some(code) })
            .collect();
        let order = farm::Order { setup, node, pattern: params.pattern(), start: params.start, first, last };
        farm::start(order, local, helpers);
        Ok(())
    }

    pub(crate) fn api_drain(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        self.network_tick();
        let Some(mut serve) = self.api.take() else { return };
        // Calls from this machine, then from paired ones (already unlocked
        // by their code in the connection; the app treats them alike).
        let mut incoming: Vec<Call> = serve.host.rx.try_iter().collect();
        #[cfg(not(target_arch = "wasm32"))]
        if let Some((h, _)) = &serve.net {
            for call in h.rx.try_iter() {
                // Rendering for another machine is its own switch.
                if call.method == "render_take" && !peers::render_allowed() {
                    call.answer(Err(ApiError::app(
                        "this wobble does not render for others (network… > let paired wobbles render here)",
                    )));
                    continue;
                }
                incoming.push(call);
            }
        }
        for call in incoming {
            if matches!(call.method.as_str(), "viewer_capture" | "pixel_sample") {
                let what = if call.method == "viewer_capture" {
                    What::Png(int(&call.params, "max_size").unwrap_or(768).clamp(16, 8192) as u32)
                } else {
                    What::Sample(["x", "y", "w", "h"].map(|k| int(&call.params, k)))
                };
                serve.captures.push(Capture {
                    call,
                    what,
                    passes: 0,
                    deadline: Instant::now() + SETTLE_TIMEOUT,
                });
                continue;
            }
            if call.method == "addon_events" {
                let addon = call.params.get("addon").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                let ms = int(&call.params, "timeout_ms").unwrap_or(20_000).min(30_000);
                serve.polls.push(Poll { call, addon, deadline: Instant::now() + Duration::from_millis(ms) });
                continue;
            }
            if call.method == "node_run" {
                serve.runs.push((call, Instant::now() + RIG_TIMEOUT));
                continue;
            }
            let mutates = api::method(&call.method).is_some_and(|m| m.mutates);
            let r = if mutates && !matches!(call.method.as_str(), "undo" | "redo" | "setup_load") {
                self.api_recorded(&call.method, &call.params)
            } else {
                self.api_call(&call.method, &call.params)
            };
            call.answer(r);
        }
        // A run starts once the rig the last edit asked for is up (or says
        // why it cannot, after waiting long enough to be sure).
        // ... and the frame the script last sought is the one on screen: a
        // track starts from the DISPLAYED frame, so one begun in the same
        // breath as a `time_set` started from wherever the playhead was.
        let ready = !self.playback_dirty
            && self.playback.is_some()
            && self.shown_tick == self.seek_target.unwrap_or(self.timeline_pos);
        for (call, deadline) in std::mem::take(&mut serve.runs) {
            if ready || Instant::now() > deadline {
                let r = self.api_call("node_run", &call.params);
                call.answer(r);
            } else {
                serve.runs.push((call, deadline));
            }
        }
        let mut waiting = Vec::new();
        for mut c in std::mem::take(&mut serve.captures) {
            c.passes += 1;
            let settled = c.passes >= SETTLE_PASSES
                && self.shown_tick == self.seek_target.unwrap_or(self.timeline_pos);
            if settled || Instant::now() > c.deadline {
                let r = match c.what {
                    What::Png(max) => self.api_capture(max, settled),
                    What::Sample(rect) => self.api_sample(rect, settled),
                };
                c.call.answer(r);
            } else {
                waiting.push(c);
            }
        }
        serve.captures = waiting;
        let mut polls = Vec::new();
        for p in std::mem::take(&mut serve.polls) {
            let evs = self.addons.take_events(&p.addon);
            if !evs.is_empty() || Instant::now() > p.deadline {
                p.call.answer(Ok(json!({ "events": evs })));
            } else {
                polls.push(p);
            }
        }
        serve.polls = polls;
        self.api = Some(serve);
        #[cfg(not(target_arch = "wasm32"))]
        self.remote_tick();
    }

    /// The remote screen's pass, when one is due: the same panes the dock
    /// draws (`properties_ui`, the graph), in the remote screen's own egui
    /// context, or the viewer's picture.
    #[cfg(not(target_arch = "wasm32"))]
    fn remote_tick(&mut self) {
        let Some(mut r) = self.api.as_mut().and_then(|a| a.remote.take()) else { return };
        if let Some(raw) = r.poll() {
            let ctx = r.ctx.clone();
            crate::theme::sync(&ctx);
            let pane = r.pane;
            // The graph keeps its own view on this screen: swap a
            // NodeEditor of its own in for the pass, and the Mac's
            // pan / zoom / drag state back out after.
            let mut ed = r.editor.take().unwrap_or_else(|| {
                let mut e = editor::NodeEditor::default();
                e.system_clipboard = false;
                e
            });
            if pane == remote::Pane::Nodes {
                std::mem::swap(&mut self.node_editor, &mut ed);
            }
            // So does the viewer: its pan, zoom and rects. And the brush
            // reads the pen queue: the Mac's samples step aside for the
            // iPad's Pencil (with its pressure) and come back after.
            let mut mac_view = None;
            let mut mac_pen = None;
            if pane == remote::Pane::Viewer {
                mac_view = Some(RemoteView::grab(self));
                r.view.take().unwrap_or_default().put(self);
                mac_pen = Some((crate::pen::drain(), crate::pen::pressure()));
                for s in r.pens.drain(..) {
                    crate::pen::push(s);
                }
                crate::pen::set_pressure(r.pressure);
            }
            // The viewer's marks are drawn by a wgpu callback (ink.rs), which
            // has nothing to run on over there: keep the mesh, send it.
            crate::ink::capture_begin();
            let keep = (self.graph_panel_rect, self.bottom_rect);
            let frame = if pane == remote::Pane::Viewer { egui::Frame::NONE } else { egui::Frame::central_panel(&ctx.global_style()) };
            let out = ctx.run_ui(raw, |ui| {
                egui::CentralPanel::default().frame(frame).show(ui, |ui| match pane {
                    remote::Pane::Nodes => {
                        ui.horizontal(|ui| self.graph_bar_items(ui));
                        self.pane_ui(ui, crate::PaneKind::Nodes);
                    }
                    // The dock's viewer tab, as it draws it: the bar, then
                    // the viewer in what is left. The picture is not egui's;
                    // it goes out separately into `image_rect`.
                    remote::Pane::Viewer => {
                        let pane = ui.max_rect();
                        self.viewer_bar_ui(ui);
                        let r = ui.available_rect_before_wrap().intersect(pane);
                        ui.scope_builder(egui::UiBuilder::new().max_rect(r), |ui| self.viewer_ui(ui));
                    }
                    remote::Pane::Properties => {
                        ui.horizontal(|ui| {
                            self.prop_layout_bar(ui);
                            self.props_keys_strip(ui);
                        });
                        ui.separator();
                        self.properties_ui(ui, false);
                    }
                });
            });
            (self.graph_panel_rect, self.bottom_rect) = keep;
            if pane == remote::Pane::Nodes {
                std::mem::swap(&mut self.node_editor, &mut ed);
            }
            if let Some(v) = mac_view {
                r.view = Some(RemoteView::grab(self));
                v.put(self);
            }
            if let Some((samples, p)) = mac_pen {
                crate::pen::drain();
                for s in samples {
                    crate::pen::push(s);
                }
                crate::pen::set_pressure(p);
            }
            r.editor = Some(ed);
            let ink = crate::ink::capture_take().map(|(mesh, palette)| remote::Ink {
                mesh,
                palette,
                darken: self.prefs.frost_darken.unwrap_or(0.0),
            });
            r.finish(out, ink);
        }
        if r.pane == remote::Pane::Viewer {
            self.remote_picture(&mut r);
        }
        if let Some(a) = self.api.as_mut() {
            a.remote = Some(r);
        }
    }

    /// The viewer's picture for the remote screen: collect the readback in
    /// flight if it has landed, and start the next one when it is due.
    #[cfg(not(target_arch = "wasm32"))]
    fn remote_picture(&mut self, r: &mut remote::Remote) {
        let _ = self.ctx.device.poll(wgpu::PollType::Poll);
        r.picture.collect(r.sender());
        let Some((sw, sh)) = r.screen_px() else { return };
        if !r.picture.due() {
            return;
        }
        let Some(src) = self.comp.output_view() else { return };
        const FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
        // The screen, capped; the picture lands where the remote viewer's
        // pass put it (its own pan and zoom), in that target's pixels.
        let (tw, th, s) = remote::Picture::target(sw, sh);
        let Some(img) = r.view.as_ref().map(|v| v.image_rect) else { return };
        let k = s * sw as f32 / r.screen_points().map(|(w, _)| w).unwrap_or(sw as f32).max(1.0);
        let dest = [img.min.x * k, img.min.y * k, img.width() * k, img.height() * k];
        let pic = &mut r.picture;
        if pic.target.as_ref().is_none_or(|t| (t.2, t.3) != (tw, th)) {
            let tex = self.ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("remote-picture"),
                size: wgpu::Extent3d { width: tw, height: th, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FMT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let bpr = (tw * 4).div_ceil(256) * 256;
            let buf = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("remote-picture"),
                size: bpr as u64 * th as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            pic.target = Some((tex, buf, tw, th, bpr));
        }
        let blit = pic.blit.get_or_insert_with(|| engine::blit::Blit::new(&self.ctx.device, FMT));
        let view = self.shown_view();
        blit.set_view(&self.ctx.device, &view);
        blit.set_channel(self.view_channel);
        let Some((tex, buf, _, _, bpr)) = &pic.target else { return };
        let target_view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.ctx.device.create_command_encoder(&Default::default());
        blit.encode(
            &self.ctx.device,
            &self.ctx.queue,
            &mut encoder,
            src,
            &target_view,
            None,
            dest,
            self.view_alpha,
            &view,
        );
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo {
                buffer: buf,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(*bpr), rows_per_image: Some(th) },
            },
            wgpu::Extent3d { width: tw, height: th, depth_or_array_layers: 1 },
        );
        self.ctx.queue.submit([encoder.finish()]);
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let f = flag.clone();
        buf.slice(..).map_async(wgpu::MapMode::Read, move |res| {
            if res.is_ok() {
                f.store(true, std::sync::atomic::Ordering::Release);
            }
        });
        pic.mapping = Some(flag);
        pic.last_start = Some(Instant::now());
    }

    /// Run a mutating call as exactly one undo step.
    fn api_recorded(&mut self, method: &str, p: &Value) -> R {
        // A hand edit not yet recorded (the per-frame check waits for the
        // mouse to come up) is its own step, not folded into the script's.
        let hand = self.egraph != self.undo_last;
        if hand {
            self.api_push_undo(self.undo_last.clone());
        }
        // A drag (`node_set` with drag=true) is ONE step however many calls
        // it takes: a call continuing the last one, with nothing between,
        // adds to that step instead of pushing its own.
        let drag = drag_key(method, p);
        let continuing = {
            let mut last = API_DRAG.lock().unwrap_or_else(|e| e.into_inner());
            let c = !hand && drag.is_some() && *last == drag;
            *last = drag;
            c
        };
        let before = self.egraph.clone();
        let r = self.api_call(method, p);
        if self.egraph != before && !continuing {
            self.api_push_undo(before);
        }
        self.undo_last = self.egraph.clone();
        r
    }

    fn api_push_undo(&mut self, g: EditorGraph) {
        self.undo_stack.push(g);
        if self.undo_stack.len() > 100 {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
    }

    pub(crate) fn api_call(&mut self, method: &str, p: &Value) -> R {
        match method {
            // A connection from this machine needs no code; a network one
            // was unlocked (or refused) before its calls got here.
            "auth" => Ok(json!({ "ok": true })),
            "nodes_receive" => self.api_receive(p),
            #[cfg(not(target_arch = "wasm32"))]
            "render_take" => farm::take(p).map_err(ApiError::app),
            #[cfg(not(target_arch = "wasm32"))]
            "render_shared" => {
                let ids = p["peers"].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(String::from)).collect();
                if farm::busy() {
                    return Err(ApiError::app("a shared render is already running"));
                }
                self.farm_start(Some(need_id(p, "id")?), ids).map_err(ApiError::app)?;
                Ok(json!({ "started": true }))
            }
            #[cfg(not(target_arch = "wasm32"))]
            "network_render_status" => Ok(farm::snapshot()),
            #[cfg(target_arch = "wasm32")]
            "render_shared" | "network_render_status" => Err(ApiError::app("a page cannot share a render")),
            #[cfg(not(target_arch = "wasm32"))]
            "render_status" => farm::status(p).map_err(ApiError::app),
            #[cfg(not(target_arch = "wasm32"))]
            "render_fetch" => farm::fetch(p).map_err(ApiError::app),
            #[cfg(not(target_arch = "wasm32"))]
            "render_release" => farm::release(p).map_err(ApiError::app),
            #[cfg(target_arch = "wasm32")]
            "render_take" | "render_status" | "render_fetch" | "render_release" => {
                Err(ApiError::app("a page cannot render for others"))
            }
            #[cfg(not(target_arch = "wasm32"))]
            "render_receive" => deliver::receive(p).map_err(ApiError::app),
            #[cfg(not(target_arch = "wasm32"))]
            "render_send" => {
                let id = need_id(p, "id")?;
                let peer = p["peer"].as_str().ok_or_else(|| ApiError::invalid("'peer' must be a network_peers id"))?;
                let Some(NodeKind::Write(wp)) = self.egraph.node(id).map(|n| n.kind.clone()) else {
                    return Err(ApiError::invalid("not a Render node"));
                };
                let files = deliver::render_files(self.written.get(&id).map(Vec::as_slice).unwrap_or(&[]));
                deliver::start(peer.to_string(), deliver::send_name(&wp.clean_name()), files).map_err(ApiError::app)?;
                Ok(json!({ "started": true }))
            }
            #[cfg(not(target_arch = "wasm32"))]
            "render_send_status" => Ok(deliver::snapshot()),
            #[cfg(target_arch = "wasm32")]
            "render_receive" | "render_send" | "render_send_status" => {
                Err(ApiError::app("a page cannot send or receive renders"))
            }
            #[cfg(not(target_arch = "wasm32"))]
            "network_peers" => Ok(peers::snapshot()),
            #[cfg(target_arch = "wasm32")]
            "network_peers" => Ok(json!({ "sharing": false, "peers": [] })),
            "app_info" => Ok(json!({
                "protocol": api::VERSION,
                "playhead": self.timeline_pos,
                "shown_frame": self.shown_tick,
                "frame_range": self.comp_range,
                "fps": self.fps,
                "working_space": format!("{:?}", self.egraph.working),
                "setup": self.current_setup.as_ref().map(|p| p.display().to_string()),
                "view": self.egraph.active_view(),
                // A subwob is what the code calls a tool; the old key stays
                // for scripts written before the rename.
                "inside_subwob": !self.dive.is_empty(),
                "inside_tool": !self.dive.is_empty(),
                // Which graph the node calls work on right now.
                "dive": self.dive.last().map(|d| match d.kind {
                    crate::DiveKind::Tool => "subwob",
                    crate::DiveKind::Noise => "noise",
                    crate::DiveKind::Geo => "2dini",
                    crate::DiveKind::Sdf => "sdf",
                    crate::DiveKind::Particles => "particles",
                    crate::DiveKind::GenAi => "genai",
                    crate::DiveKind::Roto => "roto",
                }),
                "dive_depth": self.dive.len(),
                "nodes": self.egraph.nodes.len(),
                "playing": self.playing,
                "selection": self.selected,
                "script_range": self.comp_range,
                "viewer": {
                    "exposure": self.view.exposure,
                    "gamma": self.view.gamma,
                    "channel": match (self.view_alpha, self.view_channel) {
                        (true, _) => "a",
                        (_, Some(0)) => "r",
                        (_, Some(1)) => "g",
                        (_, Some(2)) => "b",
                        _ => "rgb",
                    },
                    "proxy": self.proxy < 1.0,
                },
            })),
            "node_types" => Ok(self.api_node_types()),
            "addon_panel_set" => {
                let s = |k: &str| p.get(k).and_then(|v| v.as_str()).unwrap_or_default();
                let empty = Map::new();
                let values = p.get("values").and_then(|v| v.as_object()).unwrap_or(&empty);
                self.addons
                    .panel_set(s("addon"), s("panel"), values, p.get("open").and_then(|v| v.as_bool()))
                    .map_err(ApiError::invalid)?;
                Ok(json!({ "ok": true }))
            }
            "addon_list" => Ok(json!({ "addons": self.addons.list.iter().map(|a| json!({
                "id": a.m.id,
                "label": a.m.label,
                "version": a.m.version,
                "description": a.m.description,
                "enabled": a.enabled,
                "runs": !a.m.run.is_empty(),
                "running": a.running(),
                "exit": a.exit,
                "status": a.status,
                "commands": a.m.commands.iter().map(|c| json!({ "id": c.id, "label": c.label })).collect::<Vec<_>>(),
                "panels": a.m.panels.iter().map(|p| json!({ "id": p.id, "label": p.label, "open": a.open_panels.contains(&p.id) })).collect::<Vec<_>>(),
            })).collect::<Vec<_>>() })),
            "addon_command" => {
                let s = |k: &str| p.get(k).and_then(|v| v.as_str()).unwrap_or_default();
                let (addon, cmd) = (s("addon"), s("command"));
                let a = self.addons.get_mut(addon).ok_or_else(|| ApiError::invalid(format!("no addon '{addon}'")))?;
                if !a.enabled {
                    return Err(ApiError::app(format!("addon '{addon}' is not enabled")));
                }
                if !a.m.commands.iter().any(|c| c.id == cmd) {
                    return Err(ApiError::invalid(format!("addon '{addon}' has no command '{cmd}'")));
                }
                self.addons.command(addon, cmd);
                Ok(json!({ "queued": true }))
            }
            "blender_session" => {
                let id = need_id(p, "id")?;
                self.api_exists(id)?;
                if !matches!(self.egraph.node(id).map(|n| &n.kind), Some(engine::graph::NodeKind::Blender(_))) {
                    return Err(ApiError::invalid(format!("node {id} is not a Blender node")));
                }
                let action = p.get("action").and_then(|v| v.as_str()).unwrap_or_default();
                let a = match action {
                    "open" => crate::blenderui::Action::Open,
                    "connect" => {
                        let port = int(p, "port").unwrap_or(0);
                        if !(1..=65535).contains(&port) {
                            return Err(ApiError::invalid("connect needs a port (1..65535)".to_string()));
                        }
                        crate::blenderui::Action::Connect(port as u16)
                    }
                    "detach" => crate::blenderui::Action::Detach,
                    "quit" => crate::blenderui::Action::Quit,
                    "refresh" => crate::blenderui::Action::Refresh,
                    other => return Err(ApiError::invalid(format!("unknown action '{other}'"))),
                };
                self.blender_action(id, a);
                match self.load_errors.get(&id) {
                    Some(e) if action == "open" => Err(ApiError::app(e.clone())),
                    _ => Ok(json!({ "ok": true, "session": self.blender.contains_key(&id) })),
                }
            }
            "blender_view" => {
                let id = need_id(p, "id")?;
                self.api_exists(id)?;
                let Some(l) = self.blender.get_mut(&id) else {
                    return Err(ApiError::app(format!("node {id} has no Blender session")));
                };
                let mut v = l.view.unwrap_or_default();
                let f3 = |k: &str| p.get(k).and_then(|a| a.as_array()).and_then(|a| {
                    let n: Vec<f32> = a.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect();
                    (n.len() == 3).then(|| [n[0], n[1], n[2]])
                });
                if let Some(loc) = f3("loc") {
                    v.loc = loc;
                }
                if let Some(a) = p.get("rot").and_then(|a| a.as_array()) {
                    let n: Vec<f32> = a.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect();
                    if n.len() == 4 {
                        v.rot = [n[0], n[1], n[2], n[3]];
                    }
                }
                if let Some(d) = p.get("dist").and_then(|x| x.as_f64()) {
                    v.dist = d as f32;
                }
                if let Some(d) = p.get("lens").and_then(|x| x.as_f64()) {
                    v.lens = d as f32;
                }
                if let Some(o) = p.get("ortho").and_then(|x| x.as_bool()) {
                    v.ortho = o;
                }
                if let Some(a) = p.get("orbit").and_then(|a| a.as_array()) {
                    let n: Vec<f32> = a.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect();
                    if n.len() == 2 {
                        v.orbit(n[0], n[1]);
                    }
                }
                let moving = p.get("moving").and_then(|x| x.as_bool()).unwrap_or(false);
                l.view = Some(v);
                l.nav_active = moving;
                l.session.view(&v, moving);
                Ok(json!({ "eye": { "loc": v.loc, "rot": v.rot, "dist": v.dist, "lens": v.lens, "ortho": v.ortho }, "moving": moving }))
            }
            "blender_status" => {
                let id = need_id(p, "id")?;
                self.api_exists(id)?;
                Ok(self.blender_status_json(id))
            }
            "addon_status" => {
                let s = |k: &str| p.get(k).and_then(|v| v.as_str()).unwrap_or_default();
                self.addons.set_status(s("addon"), s("text")).map_err(ApiError::invalid)?;
                Ok(json!({ "ok": true }))
            }
            "graph_get" => Ok(self.api_graph(p.get("include_kind").and_then(|v| v.as_bool()).unwrap_or(false))),
            "node_get" => {
                let id = need_id(p, "id")?;
                self.api_node(id, true)
            }
            "node_create" => self.api_create(p),
            "load_set" => self.api_load_set(p),
            "node_delete" => {
                let id = need_id(p, "id")?;
                self.api_exists(id)?;
                self.egraph.remove(id);
                self.selected.retain(|s| *s != id);
                self.playback_dirty = true;
                Ok(json!({ "deleted": id }))
            }
            "node_connect" => {
                let (src, dst) = (need_id(p, "src")?, need_id(p, "dst")?);
                self.api_exists(src)?;
                self.api_exists(dst)?;
                let input = int(p, "input").unwrap_or(0) as usize;
                let output = int(p, "output").unwrap_or(0) as u32;
                let n_in = self.egraph.node(dst).map(|n| n.kind.input_count()).unwrap_or(0);
                if input >= n_in {
                    return Err(ApiError::invalid(format!("node {dst} has {n_in} input(s); no input {input}")));
                }
                let bridge = editor::connect_or_bridge(&mut self.egraph, src, output, dst, input, 1.0);
                // connect_or_bridge declines quietly (a pipe nothing converts,
                // a loop that cannot be made); a script must hear about it.
                if bridge.is_none() && self.egraph.input_of(dst, input) != Some(src) {
                    let why = {
                        let (s, d) = (self.egraph.node(src).expect("checked"), self.egraph.node(dst).expect("checked"));
                        wire_refusal(&s.kind, output, &d.kind, input)
                            .unwrap_or_else(|| "the wire would make a loop that cannot be built here".into())
                    };
                    return Err(ApiError::invalid(format!("node {src} output {output} -> node {dst} input {input}: not connected: {why}")));
                }
                self.playback_dirty = true;
                Ok(json!({ "connected": [src, dst, input], "bridge": bridge }))
            }
            "node_disconnect" => {
                let dst = need_id(p, "dst")?;
                self.api_exists(dst)?;
                self.egraph.disconnect_input(dst, int(p, "input").unwrap_or(0) as usize);
                self.playback_dirty = true;
                Ok(json!({ "disconnected": dst }))
            }
            "node_set" => self.api_set(p),
            "node_key" => self.api_key(p),
            "node_unkey" => self.api_unkey(p),
            "node_patch" => self.api_patch(p),
            "node_key_delete" => self.api_key_delete(p),
            "node_lock" => {
                let id = need_id(p, "id")?;
                self.api_exists(id)?;
                let on = p["locked"].as_bool().unwrap_or(true);
                let mut chans = Vec::new();
                for name in p["channels"].as_array().into_iter().flatten() {
                    let ch = channel(name.as_str().unwrap_or_default())?;
                    if anim::get(&self.egraph.node(id).expect("checked").kind, ch).is_none() {
                        return Err(ApiError::invalid(format!("node {id} has no channel {}", ch.key())));
                    }
                    chans.push(ch);
                }
                let n = self.egraph.node_mut(id).expect("checked");
                for ch in &chans {
                    anim::set_locked(&mut n.anim, *ch, on);
                }
                Ok(json!({ "id": id, "locked": on, "channels": chans.len() }))
            }
            "expr_set" => self.api_expr(p),
            "patch_get" => Ok(json!({
                "mods": self.egraph.patch.mods.iter().map(mod_json).collect::<Vec<_>>(),
                "links": self.egraph.patch.links.iter().map(|l| json!({
                    "from": src_json(l.from), "to": dst_json(l.to),
                })).collect::<Vec<_>>(),
            })),
            "patch_link" => {
                let from = self.api_src(&p["from"])?;
                let to = self.api_dst(&p["to"])?;
                if let PDst::Chan(n, c) = to {
                    if anim::is_locked(&self.egraph.node(n).expect("checked").anim, c) {
                        return Err(ApiError::app(format!("{} is locked", c.key())));
                    }
                }
                if matches!((from, to), (PSrc::Chan(a, x), PDst::Chan(b, y)) if a == b && x == y) {
                    return Err(ApiError::invalid("a channel cannot drive itself"));
                }
                self.egraph.patch.link(from, to);
                Ok(json!({ "from": src_json(from), "to": dst_json(to) }))
            }
            "patch_unlink" => {
                let to = self.api_dst(&p["to"])?;
                let had = self.egraph.patch.link_to(to).is_some();
                self.egraph.patch.unlink(to);
                Ok(json!({ "unlinked": had }))
            }
            "patch_mod_create" => {
                let k = p.get("value").and_then(|v| v.as_f64()).map(|v| v as f32);
                let kind = match p["kind"].as_str().unwrap_or_default() {
                    "mult" => ModKind::Mult(k.unwrap_or(1.0)),
                    "add" => ModKind::Add(k.unwrap_or(0.0)),
                    "max" => ModKind::Max(k.unwrap_or(0.0)),
                    "min" => ModKind::Min(k.unwrap_or(1.0)),
                    _ => ModKind::Expr(p.get("expr").and_then(|v| v.as_str()).unwrap_or("a").to_string()),
                };
                let f = |k: &str| p.get(k).and_then(|v| v.as_f64()).map(|v| v as f32);
                // Default: under the value nodes already there.
                let y = f("y").unwrap_or_else(|| {
                    self.egraph.patch.mods.iter().map(|m| m.pos[1] + 80.0).fold(0.0, f32::max)
                });
                let id = self.egraph.patch.add_mod(kind, [f("x").unwrap_or(-200.0), y]);
                Ok(json!({ "id": id }))
            }
            "patch_mod_set" => {
                let id = need_id(p, "id")?;
                let m = self
                    .egraph
                    .patch
                    .mods
                    .iter_mut()
                    .find(|m| m.id == id)
                    .ok_or_else(|| ApiError::invalid(format!("no value node {id} (see patch_get)")))?;
                let k = p.get("value").and_then(|v| v.as_f64()).map(|v| v as f32);
                let x = p.get("expr").and_then(|v| v.as_str());
                match (&mut m.kind, k, x) {
                    (ModKind::Expr(t), _, Some(x)) => *t = x.to_string(),
                    (ModKind::Mult(c) | ModKind::Add(c) | ModKind::Max(c) | ModKind::Min(c), Some(k), _) => *c = k,
                    (ModKind::Expr(_), _, None) => return Err(ApiError::invalid("an expr value node takes 'expr'")),
                    _ => return Err(ApiError::invalid("this value node takes 'value'")),
                }
                Ok(mod_json(m))
            }
            "patch_mod_delete" => {
                let id = need_id(p, "id")?;
                if !self.egraph.patch.mods.iter().any(|m| m.id == id) {
                    return Err(ApiError::invalid(format!("no value node {id} (see patch_get)")));
                }
                self.egraph.patch.remove_mod(id);
                Ok(json!({ "deleted": id }))
            }
            "dive_enter" => {
                let id = need_id(p, "id")?;
                self.api_exists(id)?;
                self.dive_into(id).map_err(ApiError::app)?;
                if let Some(e) = self.tool_error.take() {
                    return Err(ApiError::app(e));
                }
                Ok(json!({ "depth": self.dive.len(), "nodes": self.egraph.nodes.len() }))
            }
            "dive_exit" => {
                let all = p.get("all").and_then(|v| v.as_bool()).unwrap_or(false);
                if self.dive.is_empty() {
                    return Err(ApiError::app("not inside a node: this is the comp"));
                }
                loop {
                    self.dive_out().map_err(ApiError::app)?;
                    if !all || self.dive.is_empty() {
                        break;
                    }
                }
                self.playback_dirty = true;
                Ok(json!({ "depth": self.dive.len(), "nodes": self.egraph.nodes.len() }))
            }
            "subwob_collapse" => {
                let ids: Vec<u64> =
                    p["ids"].as_array().into_iter().flatten().filter_map(|v| v.as_u64()).collect();
                if ids.is_empty() {
                    return Err(ApiError::invalid("'ids' names no nodes"));
                }
                for id in &ids {
                    self.api_exists(*id)?;
                }
                let was = std::mem::replace(&mut self.selected, ids);
                self.collapse_selection();
                if let Some(e) = self.tool_error.take() {
                    self.selected = was;
                    return Err(ApiError::app(e));
                }
                Ok(json!({ "id": self.selected.first() }))
            }
            "selection_set" => {
                let ids: Vec<u64> =
                    p["ids"].as_array().into_iter().flatten().filter_map(|v| v.as_u64()).collect();
                for id in &ids {
                    self.api_exists(*id)?;
                }
                // As the summary says: their panels open (pinned in the
                // rack, the way a Z-click pins them).
                for id in &ids {
                    if !self.prop.order.contains(id) {
                        self.prop.order.push(*id);
                    }
                }
                self.selected = ids;
                Ok(json!({ "selection": self.selected }))
            }
            "viewer_set" => {
                if let Some(v) = p.get("exposure").and_then(|v| v.as_f64()) {
                    self.view.exposure = v as f32;
                }
                if let Some(v) = p.get("gamma").and_then(|v| v.as_f64()) {
                    if v <= 0.0 {
                        return Err(ApiError::invalid("gamma must be above 0"));
                    }
                    self.view.gamma = v as f32;
                }
                if let Some(c) = p.get("channel").and_then(|v| v.as_str()) {
                    self.view_alpha = c == "a";
                    self.view_channel = ["r", "g", "b"].iter().position(|k| *k == c);
                }
                if let Some(on) = p.get("proxy").and_then(|v| v.as_bool()) {
                    if on != (self.proxy < 1.0) {
                        // Exactly what the toolbar chip does.
                        self.proxy = if on { 0.5 } else { 1.0 };
                        self.playback_dirty = true;
                        self.last_engine_graph = None;
                    }
                }
                Ok(json!({ "ok": true }))
            }
            "comp_set" => {
                if let Some(r) = p.get("range").and_then(|v| v.as_array()) {
                    let (Some(a), Some(b)) = (r.first().and_then(|v| v.as_u64()), r.get(1).and_then(|v| v.as_u64()))
                    else {
                        return Err(ApiError::invalid("'range' is [first, last], whole frames"));
                    };
                    self.comp_range = Some((a, b.max(a)));
                }
                if p.get("auto_range").and_then(|v| v.as_bool()) == Some(true) {
                    self.comp_range = None;
                }
                if let Some(f) = p.get("fps").and_then(|v| v.as_f64()) {
                    if !(1.0..=240.0).contains(&f) {
                        return Err(ApiError::invalid("fps must be 1..240"));
                    }
                    self.fps = f as f32;
                }
                if let Some(on) = p.get("playing").and_then(|v| v.as_bool()) {
                    self.playing = on;
                }
                Ok(json!({ "script_range": self.comp_range, "fps": self.fps, "playing": self.playing }))
            }
            "node_run" => self.api_run(p),
            "jobs_get" => Ok(self.api_jobs()),
            "batch" => self.api_batch(p),
            "view_set" => {
                let id = need_id(p, "id")?;
                self.api_exists(id)?;
                if p.get("slot").and_then(|s| s.as_str()) == Some("b") {
                    self.egraph.view_b = Some(id);
                    self.egraph.view_active_b = true;
                } else {
                    self.egraph.view = Some(id);
                    self.egraph.view_active_b = false;
                }
                // The view is saved with the setup but is not an edit: keep
                // it out of the undo history, as the number keys do.
                self.undo_last.view = self.egraph.view;
                self.undo_last.view_b = self.egraph.view_b;
                self.undo_last.view_active_b = self.egraph.view_active_b;
                Ok(json!({ "view": id }))
            }
            "time_set" => {
                let f = int(p, "frame").ok_or_else(|| ApiError::invalid("'frame' must be a frame number"))?;
                self.seek_to(f);
                Ok(json!({ "playhead": f }))
            }
            "record" => self.api_record(p),
            "ui_set" => {
                if let Some(b) = p["bottom"].as_str() {
                    self.bottom = match b {
                        "properties" => crate::BottomTab::Properties,
                        "curves" => crate::BottomTab::Curves,
                        _ => return Err(ApiError::invalid("'bottom' is properties or curves")),
                    };
                    self.props_visible = true;
                }
                if let Some(ids) = p["hud_pin"].as_array() {
                    let mut pins = Vec::new();
                    for v in ids {
                        let id = v.as_u64().ok_or_else(|| ApiError::invalid("'hud_pin' holds node ids"))?;
                        self.api_exists(id)?;
                        pins.push(id);
                    }
                    self.gizmo_pinned = pins;
                }
                self.egui_ctx.request_repaint();
                Ok(json!({ "ok": true }))
            }
            "graph_op" => {
                let act = match p["action"].as_str() {
                    Some("tidy") => crate::keymap::Act::GraphTidy,
                    Some("frame") => crate::keymap::Act::GraphFrame,
                    _ => return Err(ApiError::invalid("'action' is tidy or frame")),
                };
                self.node_editor.pick_op(act);
                self.egui_ctx.request_repaint();
                Ok(json!({ "ok": true }))
            }
            "perf_get" => {
                let g = self.timer.averages();
                let ring = self.playback.as_ref().map(|pb| {
                    json!({
                        "slots": pb.ring.slot_count(),
                        "created": pb.ring.created(),
                        "free": pb.ring.free_count(),
                        "readers": pb.slots.len(),
                        // What a tick costs: the readers it decodes and
                        // uploads, and their bytes.
                        "active": (0..pb.slots.len())
                            .filter(|r| pb.player.is_active(*r))
                            .map(|r| {
                                let s = &pb.player.sequences()[r];
                                let (res, _) = pb.player.ram_residency(r);
                                json!([
                                    pb.slots[r],
                                    s.info.width,
                                    s.info.height,
                                    s.slot_bytes() >> 20,
                                    res.iter().filter(|x| **x).count(),
                                    res.len()
                                ])
                            })
                            .collect::<Vec<_>>(),
                        "ram": pb.player.cache_stats(),
                    })
                });
                let nvme = self.disk_cache.as_ref().map(|d| {
                    json!({
                        "hits": d.hits.load(std::sync::atomic::Ordering::Relaxed),
                        "decodes": d.misses.load(std::sync::atomic::Ordering::Relaxed),
                    })
                });
                Ok(json!({
                    "playing": self.playing,
                    "fps_shown": self.displayed_fps_counter.2,
                    "fps_target": self.fps,
                    "skipped": self.skipped,
                    "gpu_ms": { "upload": g.upload_ms, "prepass": g.prepass_ms, "fused": g.fused_ms, "blit": g.blit_ms },
                    "cpu_loop_ms": self.frame_cpu_ms.avg(),
                    "output": self.comp.output_size(),
                    "vram_mb": self
                        .comp
                        .vram_report()
                        .into_iter()
                        .map(|(k, b)| (k.to_string(), json!(b >> 20)))
                        .collect::<serde_json::Map<String, Value>>(),
                    "ring": ring,
                    "nvme": nvme,
                }))
            }
            "cost_get" => {
                let ids = |s: &std::collections::HashSet<u64>| {
                    let mut v: Vec<u64> = s.iter().copied().collect();
                    v.sort();
                    v
                };
                Ok(json!({
                    "slow": ids(&self.cost_flow.slow),
                    "heavy": ids(&self.cost_flow.heavy),
                    "noncommercial": ids(&self.cost_flow.nc),
                    "noncommercial_models": crate::costviz::nc_sources(&self.egraph),
                }))
            }
            "setup_load" => {
                let path = p["path"].as_str().unwrap_or_default();
                let path = engine::paths::expand_home(path);
                self.load_setup(&path).map_err(|e| ApiError::app(format!("{e:#}")))?;
                Ok(json!({ "loaded": path.display().to_string(), "nodes": self.egraph.nodes.len() }))
            }
            "setup_save" => {
                let path = match p.get("path").and_then(|s| s.as_str()) {
                    Some(s) => engine::paths::expand_home(s),
                    None => self
                        .current_setup
                        .clone()
                        .ok_or_else(|| ApiError::invalid("no setup file is open: give a 'path'"))?,
                };
                self.save_setup(&path).map_err(|e| ApiError::app(format!("{e:#}")))?;
                Ok(json!({ "saved": path.display().to_string() }))
            }
            "undo" => {
                let had = !self.undo_stack.is_empty();
                self.undo();
                Ok(json!({ "undone": had }))
            }
            "redo" => {
                let had = !self.redo_stack.is_empty();
                self.redo();
                Ok(json!({ "redone": had }))
            }
            other => Err(ApiError { code: -32601, message: format!("the app does not implement {other}") }),
        }
    }

    fn api_exists(&self, id: u64) -> Result<(), ApiError> {
        if self.egraph.node(id).is_some() {
            Ok(())
        } else {
            Err(ApiError::invalid(format!("no node {id}")))
        }
    }

    fn api_node_types(&self) -> Value {
        let cats = cats_for(self.dive.last().map(|d| d.kind));
        let dived = cats != (false, false, false, false, false, false);
        let mut types: Vec<Value> = editor::node_catalog()
            .into_iter()
            .filter(|e| editor::entry_offered(e, cats.0, cats.1, cats.2, cats.3, cats.4, cats.5))
            .map(|e| {
                let kind = (e.make)(0);
                json!({
                    "type": e.label,
                    "category": format!("{:?}", e.cat),
                    "status": e.status.name(),
                    "inputs": kind.input_count(),
                    "channels": editor::node_channels(&kind).iter().map(|c| c.key()).collect::<Vec<_>>(),
                })
            })
            .collect();
        // Plugins and subwobs are comp nodes; a sub-graph holds none.
        for d in engine::plugins::all().into_iter().filter(|_| !dived) {
            if crate::tools::is_proxy_id(&d.type_id) {
                continue;
            }
            let kind = NodeKind::Custom { type_id: d.type_id.clone(), params: Default::default() };
            types.push(json!({
                "type": d.type_id,
                "label": d.label,
                "category": if d.tool { "Subwob" } else { "Plugin" },
                "status": editor::status_of(&kind).name(),
                "inputs": d.inputs,
                "channels": editor::node_channels(&kind).iter().map(|c| c.key()).collect::<Vec<_>>(),
            }));
        }
        json!({
            "types": types,
            // What goes INSIDE a 2dini / Noise / SDF node.
            "subgraphs": {
                "notes": SUBGRAPH_NOTES,
                "2dini": subgraph_types((false, true, false, false, false, false)),
                "noise": subgraph_types((true, false, false, false, false, false)),
                "sdf": subgraph_types((false, false, true, false, false, false)),
                "particles": subgraph_types((false, false, false, true, false, false)),
                "genai": subgraph_types((false, false, false, false, true, false)),
                "roto": subgraph_types((false, false, false, false, false, true)),
            },
        })
    }

    fn api_graph(&self, include_kind: bool) -> Value {
        let nodes: Vec<Value> = self
            .egraph
            .nodes
            .keys()
            .filter_map(|id| self.api_node(*id, include_kind).ok())
            .collect();
        json!({
            "nodes": nodes,
            "view": self.egraph.view,
            "view_b": self.egraph.view_b,
            "showing_b": self.egraph.view_active_b,
            "playhead": self.timeline_pos,
        })
    }

    fn api_node(&self, id: u64, full: bool) -> R {
        let n = self.egraph.node(id).ok_or_else(|| ApiError::invalid(format!("no node {id}")))?;
        let t = self.timeline_pos;
        let live = n.kind_at(t);
        let mut values = Map::new();
        // In panel order, with the label the panel shows: a client that
        // draws its own panel (the iPad's) needs both, and `values` is a map.
        let mut order = Vec::new();
        let mut labels = Map::new();
        for ch in editor::node_channels(&n.kind) {
            if let Some(v) = anim::get(&live, ch) {
                values.insert(ch.key(), json!(v));
                order.push(ch.key());
                labels.insert(ch.key(), json!(crate::curves::ch_label_for(&n.kind, ch)));
            }
        }
        let animated: Vec<String> =
            n.anim.iter().filter(|(_, c)| !c.keys.is_empty()).map(|(ch, _)| ch.key()).collect();
        let locked: Vec<String> =
            n.anim.keys().filter(|ch| anim::is_locked(&n.anim, **ch)).map(|ch| ch.key()).collect();
        let inputs: Vec<Value> = (0..n.kind.input_count())
            .map(|i| match self.egraph.input_of(id, i) {
                Some(src) => json!({ "input": i, "src": src, "output": self.egraph.out_of(id, i) }),
                None => json!({ "input": i, "src": null }),
            })
            .collect();
        let mut v = json!({
            "id": id,
            "type": type_name(&n.kind),
            "name": editor::display_label(n),
            "x": n.pos.x,
            "y": n.pos.y,
            "bypass": n.disabled,
            "inputs": inputs,
            "values": values,
            "channels": order,
            "labels": labels,
            "animated": animated,
            "locked": locked,
            // What an expression calls this node: `../<path>/<param>`.
            "path": crate::chanexpr::node_name(&self.egraph, id),
        });
        if let Some(a) = &n.asset {
            v["asset"] = serde_json::to_value(a).unwrap_or(Value::Null);
        }
        // Channels something else drives: writing them is refused, and
        // their value above is the DRIVEN one.
        let mut driven = Map::new();
        for l in &self.egraph.patch.links {
            let PDst::Chan(n2, ch) = l.to else { continue };
            if n2 != id {
                continue;
            }
            let expr = match l.from {
                PSrc::Mod(m) => self.egraph.patch.mods.iter().find(|x| x.id == m).and_then(|x| match &x.kind {
                    ModKind::Expr(t) => Some(t.clone()),
                    _ => None,
                }),
                _ => None,
            };
            let mut d = json!({ "from": src_json(l.from) });
            if let Some(t) = expr {
                d["expr"] = json!(t);
            }
            driven.insert(ch.key(), d);
        }
        if !driven.is_empty() {
            // Read them the way the render does: through the cables.
            let base: std::collections::BTreeMap<u64, NodeKind> =
                self.egraph.nodes.iter().map(|(i, n)| (*i, n.kind_at(t))).collect();
            for name in driven.keys() {
                if let Some(ch) = Channel::from_key(name) {
                    let x = self.egraph.patch_value(PSrc::Chan(id, ch), &base, t as f32, &mut Vec::new());
                    v["values"][name] = json!(x);
                }
            }
            v["driven"] = driven.into();
        }
        if let Some(e) = self.load_errors.get(&id) {
            v["error"] = json!(e);
        }
        if full {
            // Results that ride on a node (a camera solve, a stabilise path)
            // are MB of numbers: the stripped kind is what a script wants.
            v["kind"] = serde_json::to_value(&*Gfx::key_kind(&n.kind)).unwrap_or(Value::Null);
            v["keys"] = n
                .anim
                .iter()
                .filter(|(_, c)| !c.keys.is_empty())
                .map(|(ch, c)| (ch.key(), json!(c.keys.iter().map(|k| [k.frame as f64, k.value as f64]).collect::<Vec<_>>())))
                .collect::<Map<String, Value>>()
                .into();
            // Beside 'keys', index for index (a third element there would
            // break every script that unpacks [frame, value]).
            v["interp"] = n
                .anim
                .iter()
                .filter(|(_, c)| !c.keys.is_empty())
                .map(|(ch, c)| (ch.key(), json!(c.keys.iter().map(|k| interp_name(k.interp)).collect::<Vec<_>>())))
                .collect::<Map<String, Value>>()
                .into();
        }
        Ok(v)
    }

    fn api_create(&mut self, p: &Value) -> R {
        let want = p["type"].as_str().unwrap_or_default();
        let lower = want.to_ascii_lowercase();
        let entry = find_entry(want, cats_for(self.dive.last().map(|d| d.kind)));
        let kind = if let Some(e) = entry {
            let slot = self.next_read_slot;
            if e.reader {
                self.next_read_slot += 1;
            }
            (e.make)(slot)
        } else if let Some(d) = engine::plugins::all()
            .into_iter()
            .find(|d| d.type_id.to_ascii_lowercase() == lower || d.label.to_ascii_lowercase() == lower)
        {
            NodeKind::Custom { type_id: d.type_id.clone(), params: Default::default() }
        } else {
            return Err(ApiError::invalid(format!("unknown node type '{want}' (see node_types)")));
        };
        // Default placement: right of everything, so a script's nodes never
        // land on top of the user's.
        let x = p.get("x").and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or_else(|| {
            self.egraph.nodes.values().map(|n| n.pos.x).fold(0.0, f32::max) + 120.0
        });
        let y = p.get("y").and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(0.0);
        let path = p.get("path").and_then(|v| v.as_str()).map(str::to_string);
        if path.is_some() && !matches!(kind, NodeKind::Read { .. }) {
            return Err(ApiError::invalid("'path' is for a Load"));
        }
        let asset = match p.get("asset").filter(|v| !v.is_null()) {
            None => None,
            Some(_) if !matches!(kind, NodeKind::Read { .. }) => {
                return Err(ApiError::invalid("'asset' is for a Load"));
            }
            Some(_) if path.is_some() => {
                return Err(ApiError::invalid("give a Load a 'path' or an 'asset', not both"));
            }
            Some(v) => Some(
                serde_json::from_value::<crate::assets::AssetRef>(v.clone())
                    .map_err(|e| ApiError::invalid(format!("asset: {e}")))?,
            ),
        };
        let srcs: Vec<u64> = p
            .get("inputs")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
            .unwrap_or_default();
        for s in &srcs {
            self.api_exists(*s)?;
        }
        if srcs.len() > kind.input_count() {
            return Err(ApiError::invalid(format!("{want} has {} input(s)", kind.input_count())));
        }
        // Every wire is checked BEFORE the node exists, so a refusal leaves
        // nothing half-made behind (connect_or_bridge declines quietly).
        for (i, src) in srcs.iter().enumerate() {
            let sk = &self.egraph.node(*src).expect("checked").kind;
            if let Some(why) = wire_refusal(sk, 0, &kind, i) {
                return Err(ApiError::invalid(format!("inputs[{i}] (node {src}) -> {want} input {i}: {why}")));
            }
        }
        let id = self.egraph.add(kind, egui::pos2(x, y));
        if let Some(name) = p.get("name").and_then(|v| v.as_str()) {
            if let Some(n) = self.egraph.node_mut(id) {
                n.ui_label = name.to_string();
            }
        }
        let mut bridges = Vec::new();
        for (i, src) in srcs.into_iter().enumerate() {
            bridges.extend(editor::connect_or_bridge(&mut self.egraph, src, 0, id, i, 1.0));
        }
        if let Some(path) = path {
            if let Some(NodeKind::Read { path: p, .. }) = self.egraph.node_mut(id).map(|n| &mut n.kind) {
                *p = path.clone();
            }
            self.load_read(id, &path);
        }
        if let Some(a) = asset {
            if let Some(n) = self.egraph.node_mut(id) {
                n.asset = Some(a);
            }
            self.load_asset(id);
        }
        self.playback_dirty = true;
        let mut v = json!({ "id": id, "bridges": bridges });
        if let Some(e) = self.load_errors.get(&id) {
            v["load_error"] = json!(e.to_string());
        }
        Ok(v)
    }

    /// `load_set`: point an existing Load at other footage — a path, or an
    /// organisicator asset by id — as a new clip (its own range). What an
    /// addon's Load picker answers with (`addons::PickerDef`).
    fn api_load_set(&mut self, p: &Value) -> Result<Value, ApiError> {
        let id = need_id(p, "id")?;
        self.api_exists(id)?;
        if !matches!(self.egraph.node(id).map(|n| &n.kind), Some(NodeKind::Read { .. })) {
            return Err(ApiError::invalid(format!("node {id} is not a Load")));
        }
        let path = p.get("path").and_then(|v| v.as_str()).map(str::to_string);
        let asset = match p.get("asset").filter(|v| !v.is_null()) {
            None => None,
            Some(v) => Some(
                serde_json::from_value::<crate::assets::AssetRef>(v.clone())
                    .map_err(|e| ApiError::invalid(format!("asset: {e}")))?,
            ),
        };
        match (path, asset) {
            (Some(_), Some(_)) => Err(ApiError::invalid("give a Load a 'path' or an 'asset', not both")),
            (None, None) => Err(ApiError::invalid("give 'path' or 'asset'")),
            (None, Some(a)) => {
                if let Some(n) = self.egraph.node_mut(id) {
                    n.asset = Some(a);
                }
                self.load_asset_fresh(id);
                Ok(self.load_set_answer(id))
            }
            (Some(path), None) => {
                if let Some(n) = self.egraph.node_mut(id) {
                    if n.project_only {
                        return Err(ApiError::invalid(format!("node {id} is a Project Load: give it an 'asset'")));
                    }
                    n.asset = None;
                    n.ui_label.clear();
                }
                if let Some(NodeKind::Read { path: rp, start, first, last, range, .. }) =
                    self.egraph.node_mut(id).map(|n| &mut n.kind)
                {
                    *rp = path.clone();
                    (*start, *first, *last, *range) = (None, None, None, None);
                }
                self.loaded.remove(&id);
                self.load_errors.remove(&id);
                self.load_read(id, &path);
                Ok(self.load_set_answer(id))
            }
        }
    }

    fn load_set_answer(&mut self, id: u64) -> Value {
        self.playback_dirty = true;
        let mut v = json!({ "id": id });
        if let Some(e) = self.load_errors.get(&id) {
            v["load_error"] = json!(e.to_string());
        }
        v
    }

    /// Why a channel on `id` may not be written, if it may not.
    fn api_refuse(&self, id: u64, ch: Channel) -> Option<String> {
        let n = self.egraph.node(id)?;
        if anim::is_locked(&n.anim, ch) {
            return Some(format!("{} is locked", ch.key()));
        }
        if self.egraph.patch.links.iter().any(|l| l.to == PDst::Chan(id, ch)) {
            return Some(format!("{} is driven by a patch cable", ch.key()));
        }
        if anim::get(&n.kind, ch).is_none() {
            return Some(format!("node {id} has no channel {}", ch.key()));
        }
        None
    }

    fn api_set(&mut self, p: &Value) -> R {
        let id = need_id(p, "id")?;
        self.api_exists(id)?;
        let t = int(p, "frame").unwrap_or(self.timeline_pos);
        let vals = p["values"].as_object().cloned().unwrap_or_default();
        // Check everything before writing anything: a call is one step, so
        // it either lands whole or not at all.
        let mut todo = Vec::new();
        for (name, v) in &vals {
            let ch = channel(name)?;
            let v = v.as_f64().ok_or_else(|| ApiError::invalid(format!("{name} must be a number")))? as f32;
            if let Some(why) = self.api_refuse(id, ch) {
                return Err(ApiError::app(why));
            }
            todo.push((ch, v));
        }
        let mut keyed = Vec::new();
        let n = self.egraph.node_mut(id).expect("checked");
        for (ch, v) in todo {
            if n.anim.get(&ch).is_some_and(|c| !c.keys.is_empty()) {
                n.anim.entry(ch).or_default().set_key(t, v);
                keyed.push(ch.key());
            } else {
                anim::set(&mut n.kind, ch, v);
            }
        }
        Ok(json!({ "id": id, "keyed_at": if keyed.is_empty() { Value::Null } else { json!(t) }, "keyed": keyed }))
    }

    fn api_key(&mut self, p: &Value) -> R {
        let id = need_id(p, "id")?;
        self.api_exists(id)?;
        let keys = p["keys"].as_object().cloned().unwrap_or_default();
        let mut todo = Vec::new();
        for (name, list) in &keys {
            let ch = channel(name)?;
            if let Some(why) = self.api_refuse(id, ch) {
                return Err(ApiError::app(why));
            }
            for k in list.as_array().into_iter().flatten() {
                let (Some(f), Some(v)) = (k.get(0).and_then(|f| f.as_f64()), k.get(1).and_then(|v| v.as_f64())) else {
                    return Err(ApiError::invalid(format!("{name}: each key is [frame, value]")));
                };
                if f < 0.0 || f.fract() != 0.0 {
                    return Err(ApiError::invalid(format!("{name}: frame {f} is not a whole comp frame")));
                }
                let interp = match k.get(2) {
                    Some(i) => Some(interp_of(i.as_str().unwrap_or_default())?),
                    None => None,
                };
                todo.push((ch, f as u64, v as f32, interp));
            }
        }
        let n = self.egraph.node_mut(id).expect("checked");
        for (ch, f, v, _) in &todo {
            n.anim.entry(*ch).or_default().set_key(*f, *v);
        }
        // Interpolation after every key is down: a bezier seeds its handles
        // from the curve's slope, which needs the neighbours.
        for (ch, f, _, interp) in &todo {
            if let (Some(i), Some(c)) = (interp, n.anim.get_mut(ch)) {
                c.set_interp(*f, *i);
            }
        }
        Ok(json!({ "id": id, "keys_set": todo.len() }))
    }

    fn api_unkey(&mut self, p: &Value) -> R {
        let id = need_id(p, "id")?;
        self.api_exists(id)?;
        let t = self.timeline_pos;
        let mut chans = Vec::new();
        for name in p["channels"].as_array().into_iter().flatten() {
            let ch = channel(name.as_str().unwrap_or_default())?;
            if let Some(why) = self.api_refuse(id, ch) {
                return Err(ApiError::app(why));
            }
            chans.push(ch);
        }
        let n = self.egraph.node_mut(id).expect("checked");
        let live = n.kind_at(t);
        for ch in chans {
            if let Some(v) = anim::get(&live, ch) {
                anim::set(&mut n.kind, ch, v);
            }
            if let Some(c) = n.anim.get_mut(&ch) {
                c.keys.clear();
            }
        }
        Ok(json!({ "id": id }))
    }

    fn api_patch(&mut self, p: &Value) -> R {
        let id = need_id(p, "id")?;
        self.api_exists(id)?;
        let old = self.egraph.node(id).expect("checked").kind.clone();
        let kind = match p.get("kind") {
            Some(patch) => {
                let mut v = serde_json::to_value(&old).map_err(|e| ApiError::app(e.to_string()))?;
                merge(&mut v, patch);
                let k: NodeKind = serde_json::from_value(v)
                    .map_err(|e| ApiError::invalid(format!("kind does not fit this node: {e}")))?;
                if std::mem::discriminant(&k) != std::mem::discriminant(&old) {
                    return Err(ApiError::invalid("node_patch cannot change a node's type: delete it and create another"));
                }
                Some(k)
            }
            None => None,
        };
        let n = self.egraph.node_mut(id).expect("checked");
        if let Some(k) = kind {
            n.kind = k;
        }
        if let Some(s) = p.get("name").and_then(|v| v.as_str()) {
            n.ui_label = s.to_string();
        }
        if let Some(b) = p.get("bypass").and_then(|v| v.as_bool()) {
            n.disabled = b;
        }
        if let Some(x) = p.get("x").and_then(|v| v.as_f64()) {
            n.pos.x = x as f32;
        }
        if let Some(y) = p.get("y").and_then(|v| v.as_f64()) {
            n.pos.y = y as f32;
        }
        // A Read whose path changed loads the new footage, as typing it in
        // the panel does; any Read edit may move the playback rig.
        let new_path = match (&old, &n.kind) {
            (NodeKind::Read { path: a, .. }, NodeKind::Read { path: b, .. }) if a != b => Some(b.clone()),
            _ => None,
        };
        if matches!(old, NodeKind::Read { .. } | NodeKind::GraphBake { .. }) {
            self.playback_dirty = true;
        }
        if let Some(path) = new_path {
            self.loaded.remove(&id);
            self.load_errors.remove(&id);
            self.load_read(id, &path);
        }
        self.api_node(id, true)
    }

    fn api_key_delete(&mut self, p: &Value) -> R {
        let id = need_id(p, "id")?;
        self.api_exists(id)?;
        let t = self.timeline_pos;
        let mut todo = Vec::new();
        for (name, list) in p["keys"].as_object().cloned().unwrap_or_default() {
            let ch = channel(&name)?;
            if let Some(why) = self.api_refuse(id, ch) {
                return Err(ApiError::app(why));
            }
            let frames: Vec<f64> = list.as_array().into_iter().flatten().filter_map(|f| f.as_f64()).collect();
            todo.push((ch, frames));
        }
        let n = self.egraph.node_mut(id).expect("checked");
        let live = n.kind_at(t);
        let mut removed = 0;
        for (ch, frames) in todo {
            let Some(c) = n.anim.get_mut(&ch) else { continue };
            let had = c.keys.len();
            for f in frames {
                c.remove_key(f);
            }
            removed += had - c.keys.len();
            if had > 0 && c.keys.is_empty() {
                // The last key went: hold what the playhead showed.
                if let Some(v) = anim::get(&live, ch) {
                    anim::set(&mut n.kind, ch, v);
                }
            }
        }
        Ok(json!({ "id": id, "removed": removed }))
    }

    fn api_src(&self, v: &Value) -> Result<PSrc, ApiError> {
        if let Some(m) = int(v, "mod") {
            if !self.egraph.patch.mods.iter().any(|x| x.id == m) {
                return Err(ApiError::invalid(format!("no value node {m} (see patch_get)")));
            }
            return Ok(PSrc::Mod(m));
        }
        let (n, c) = self.api_chan_end(v)?;
        Ok(PSrc::Chan(n, c))
    }

    fn api_dst(&self, v: &Value) -> Result<PDst, ApiError> {
        if let Some(m) = int(v, "mod") {
            if !self.egraph.patch.mods.iter().any(|x| x.id == m) {
                return Err(ApiError::invalid(format!("no value node {m} (see patch_get)")));
            }
            let slot = int(v, "slot").unwrap_or(0);
            if slot > 1 {
                return Err(ApiError::invalid("a value node has jacks 0 (a) and 1 (b)"));
            }
            return Ok(PDst::Mod(m, slot as u8));
        }
        let (n, c) = self.api_chan_end(v)?;
        Ok(PDst::Chan(n, c))
    }

    /// `{node, channel}`, checked: the node exists and has that channel.
    fn api_chan_end(&self, v: &Value) -> Result<(u64, Channel), ApiError> {
        let (Some(n), Some(c)) = (int(v, "node"), v.get("channel").and_then(|c| c.as_str())) else {
            return Err(ApiError::invalid("a patch end is {node, channel}, {mod} or {mod, slot}"));
        };
        self.api_exists(n)?;
        let ch = channel(c)?;
        if anim::get(&self.egraph.node(n).expect("checked").kind, ch).is_none() {
            return Err(ApiError::invalid(format!("node {n} has no channel {c}")));
        }
        Ok((n, ch))
    }

    fn api_expr(&mut self, p: &Value) -> R {
        let id = need_id(p, "id")?;
        self.api_exists(id)?;
        let text = p["expr"].as_str().unwrap_or_default().trim().to_string();
        let mut chans = Vec::new();
        for name in p["channels"].as_array().into_iter().flatten() {
            let ch = channel(name.as_str().unwrap_or_default())?;
            let n = self.egraph.node(id).expect("checked");
            if anim::get(&n.kind, ch).is_none() {
                return Err(ApiError::invalid(format!("node {id} has no channel {}", ch.key())));
            }
            if anim::is_locked(&n.anim, ch) {
                return Err(ApiError::app(format!("{} is locked", ch.key())));
            }
            chans.push(ch);
        }
        if chans.is_empty() {
            return Err(ApiError::invalid("'channels' names no channels"));
        }
        if !text.is_empty() {
            // Say what is wrong now: a broken expression evaluates to 0 at
            // render time without a word.
            for path in crate::exprs::refs(&text) {
                match crate::chanexpr::resolve(&self.egraph, &path, Some(id)) {
                    Ok((n, c)) if n == id && chans.contains(&c) => {
                        return Err(ApiError::invalid(format!("{path} is the channel it drives")));
                    }
                    Ok(_) => {}
                    Err(e) => return Err(ApiError::invalid(e)),
                }
            }
            if crate::exprs::eval_with(&text, 1.0, 1.0, 1.0, &mut |_| Some(1.0)).is_none() {
                return Err(ApiError::invalid(format!("'{text}' is not an expression")));
            }
        }
        let t = self.timeline_pos;
        crate::chanexpr::set_expr(&mut self.egraph, id, &chans, &text, t);
        self.api_node(id, false)
    }

    /// Press a node's run button, by the same call its panel makes.
    /// The screen recorder, driven the way its button is: `start` /
    /// `stop` are `toggle_recording`, `save` is the review's save button.
    /// Every answer is the status, so a client polls `status` after `save`
    /// until `saved` (or `error`) appears.
    fn api_record(&mut self, p: &Value) -> R {
        let action = p["action"].as_str().unwrap_or("status");
        let ctx = self.egui_ctx.clone();
        match action {
            "start" => {
                if self.recorder.recording() {
                    return Err(ApiError::app("already recording"));
                }
                if let Some(a) = p["area"].as_str() {
                    self.recorder.opts.area = match a {
                        "viewer" => crate::recorder::Area::Viewer,
                        "window" => crate::recorder::Area::Window,
                        _ => return Err(ApiError::invalid("'area' is 'window' or 'viewer'")),
                    };
                }
                for (k, slot) in [("cursor", 0), ("watermark", 1), ("gif", 2)] {
                    if let Some(b) = p[k].as_bool() {
                        match slot {
                            0 => self.recorder.opts.cursor = b,
                            1 => self.recorder.opts.watermark = b,
                            _ => self.recorder.opts.gif = b,
                        }
                    }
                }
                self.toggle_recording(&ctx);
                if !self.recorder.recording() {
                    return Err(ApiError::app(self.recorder.error.clone().unwrap_or_else(|| "the take did not start".into())));
                }
            }
            "stop" => {
                if !self.recorder.recording() {
                    return Err(ApiError::app("not recording"));
                }
                self.toggle_recording(&ctx);
                if self.recorder.review_frames().is_none() {
                    return Err(ApiError::app(self.recorder.error.clone().unwrap_or_else(|| "the take did not reach review".into())));
                }
            }
            "save" => {
                if self.recorder.review_frames().is_none() {
                    return Err(ApiError::app("nothing under review: stop a take first"));
                }
                self.recorder.save_all();
            }
            "status" => {}
            _ => return Err(ApiError::invalid("'action' is start, stop, save or status")),
        }
        ctx.request_repaint();
        let saved = self.recorder.save_result();
        Ok(json!({
            "recording": self.recorder.recording(),
            "seconds": self.recorder.elapsed(),
            "review_frames": self.recorder.review_frames(),
            "saving": self.recorder.review_frames().is_some() && saved.is_none() && self.recorder.saving(),
            "saved": saved.as_ref().and_then(|r| r.as_ref().ok()).map(|s| json!({
                "mp4": s.mp4, "gif": s.gif, "note": s.note,
            })),
            "error": saved.as_ref().and_then(|r| r.as_ref().err().cloned()).or_else(|| self.recorder.error.clone()),
        }))
    }

    fn api_run(&mut self, p: &Value) -> R {
        let id = need_id(p, "id")?;
        self.api_exists(id)?;
        let kind = self.egraph.node(id).expect("checked").kind.clone();
        let default = match &kind {
            NodeKind::Write(_) => "render",
            NodeKind::GraphBake { .. } | NodeKind::SoftCrop(_) => "bake",
            NodeKind::Tracker(_) | NodeKind::PlanarTrack(_) => "track",
            NodeKind::CorridorKey(_) => "key",
            NodeKind::Sammie(_) => "track",
            NodeKind::Flow(_) | NodeKind::MorphDissolve(_) => "analyse",
            NodeKind::Stabilize(_) => "analyse",
            NodeKind::CamTrack(_) => "solve",
            NodeKind::Rifetime(_) => "render",
            NodeKind::FrameFix(_) => "render",
            NodeKind::Upscale(_) => "render",
            NodeKind::HdrCnn(_) => "render",
            NodeKind::SeedVr(_) => "render",
            NodeKind::Depth(_) => "render",
            NodeKind::LtxHdr(_) => "render",
            NodeKind::DiffLight(_) => "render",
            NodeKind::GenAi(_) => "quote",
            // A model step inside a GenAI pipeline runs from its own node.
            NodeKind::GenAiStep(genai::Op::Generate { .. } | genai::Op::Comfy { .. }) => "quote",
            // Inside a 2dini (dive_enter it first): the simulation bakes.
            NodeKind::GeoNode(engine::geo::GeoNode::Physics { .. } | engine::geo::GeoNode::Ragdoll { .. }) => "bake",
            _ => return Err(ApiError::invalid(format!("{} has nothing to run", type_name(&kind)))),
        };
        let action = p.get("action").and_then(|a| a.as_str()).unwrap_or(default);
        let busy = |me: &Self| me.graph_bake.as_ref().map(|b| b.node);
        let fits = |me: &Self| -> Result<(), ApiError> {
            match busy(me) {
                Some(n) => Err(ApiError::app(format!("node {n} is still rendering: wait (jobs_get) or cancel it"))),
                None => Ok(()),
            }
        };
        match (&kind, action) {
            (NodeKind::Write(wp), "render") => {
                fits(self)?;
                self.load_errors.remove(&id);
                self.start_write(id, wp);
            }
            (NodeKind::GraphBake { .. }, "bake") => {
                fits(self)?;
                self.load_errors.remove(&id);
                self.start_graph_bake(id);
            }
            // A Crop bakes itself (its first bake mints its reader slot and
            // starts once the rig is up): the panel's button.
            (NodeKind::SoftCrop(_), "bake") => {
                fits(self)?;
                self.load_errors.remove(&id);
                self.press_bake(id);
            }
            (NodeKind::Write(_) | NodeKind::GraphBake { .. } | NodeKind::SoftCrop(_), "cancel") => {
                if busy(self) == Some(id) {
                    self.graph_bake = None;
                }
            }
            (NodeKind::Tracker(tp), "track") => {
                self.render_status = None;
                self.start_track(id, tp.clone());
                if !self.track_job.as_ref().is_some_and(|(n, _)| *n == id) {
                    return Err(ApiError::app(self.render_status.clone().unwrap_or_else(|| "the track did not start".into())));
                }
            }
            (NodeKind::Tracker(_), "cancel") => {
                if let Some((_, j)) = self.track_job.take_if(|(n, _)| *n == id) {
                    j.cancel();
                }
            }
            (NodeKind::PlanarTrack(_), "track" | "track_back") => {
                self.render_status = None;
                self.start_planar(id, action == "track_back");
                if !self.planar_job.as_ref().is_some_and(|(n, _)| *n == id) {
                    return Err(ApiError::app(self.render_status.clone().unwrap_or_else(|| "the track did not start".into())));
                }
            }
            (NodeKind::PlanarTrack(_), "cancel") => {
                if let Some((_, j)) = self.planar_job.take_if(|(n, _)| *n == id) {
                    j.cancel();
                }
            }
            (NodeKind::CorridorKey(_), "key") => {
                self.corridor_action(id, crate::corridorkey::Action::Key);
                if let Some(crate::corridorkey::Phase::Failed(e)) = self.jobs.corridor.run.as_ref().map(|r| &r.phase) {
                    return Err(ApiError::app(e.clone()));
                }
            }
            (NodeKind::CorridorKey(_), "cancel") => self.corridor_action(id, crate::corridorkey::Action::Cancel),
            (NodeKind::Sammie(_), "track") => {
                self.sammie_action(id, crate::sammie::Action::Track);
                if let Some(crate::sammie::Phase::Failed(e)) = self.jobs.sammie.run.as_ref().map(|r| &r.phase) {
                    return Err(ApiError::app(e.clone()));
                }
            }
            (NodeKind::Sammie(_), "cancel") => self.sammie_action(id, crate::sammie::Action::Cancel),
            (NodeKind::Flow(_) | NodeKind::MorphDissolve(_), "analyse") => {
                self.flow_action(id, crate::flowui::Action::Analyse);
                if let Some(crate::flowui::Phase::Failed(e)) = self.jobs.flow.run.as_ref().map(|r| &r.phase) {
                    return Err(ApiError::app(e.clone()));
                }
            }
            (NodeKind::Rifetime(_) | NodeKind::FrameFix(_), "render") => {
                self.flow_action(id, crate::flowui::Action::Analyse);
                if let Some(crate::flowui::Phase::Failed(e)) = self.jobs.flow.run.as_ref().map(|r| &r.phase) {
                    return Err(ApiError::app(e.clone()));
                }
            }
            (NodeKind::Rifetime(_) | NodeKind::FrameFix(_), "cancel") => self.flow_action(id, crate::flowui::Action::Cancel),
            (NodeKind::Flow(_) | NodeKind::MorphDissolve(_), "cancel") => self.flow_action(id, crate::flowui::Action::Cancel),
            (NodeKind::Stabilize(_), "analyse") => {
                self.stab_action(id, crate::flowui::Action::Analyse);
                if let Some(crate::flowui::Phase::Failed(e)) = self.jobs.stab.run.as_ref().map(|r| &r.phase) {
                    return Err(ApiError::app(e.clone()));
                }
            }
            (NodeKind::Stabilize(_), "cancel") => self.stab_action(id, crate::flowui::Action::Cancel),
            (NodeKind::CamTrack(_), "solve") => {
                self.ct_action(id, crate::flowui::Action::Analyse);
                if let Some(crate::flowui::Phase::Failed(e)) = self.jobs.camtrack.run.as_ref().map(|r| &r.phase) {
                    return Err(ApiError::app(e.clone()));
                }
            }
            (NodeKind::CamTrack(_), "cancel") => self.ct_action(id, crate::flowui::Action::Cancel),
            (NodeKind::Upscale(_), "render") => {
                self.upscale_action(id, crate::flowui::Action::Analyse);
                if let Some(crate::flowui::Phase::Failed(e)) = self.jobs.upscale.run.as_ref().map(|r| &r.phase) {
                    return Err(ApiError::app(e.clone()));
                }
            }
            (NodeKind::Upscale(_), "cancel") => self.upscale_action(id, crate::flowui::Action::Cancel),
            (NodeKind::HdrCnn(_), "render") => {
                self.hdrcnn_action(id, crate::flowui::Action::Analyse);
                if let Some(crate::flowui::Phase::Failed(e)) = self.jobs.hdrcnn.run.as_ref().map(|r| &r.phase) {
                    return Err(ApiError::app(e.clone()));
                }
            }
            (NodeKind::HdrCnn(_), "cancel") => self.hdrcnn_action(id, crate::flowui::Action::Cancel),
            (NodeKind::SeedVr(_), "render") => {
                self.seedvr_action(id, crate::seedvr::Action::Run);
                if let Some(crate::seedvr::Phase::Failed(e)) = self.jobs.seedvr.run.as_ref().map(|r| &r.phase) {
                    return Err(ApiError::app(e.clone()));
                }
            }
            (NodeKind::SeedVr(_), "cancel") => self.seedvr_action(id, crate::seedvr::Action::Cancel),
            (NodeKind::Depth(_), "render") => {
                self.depth_action(id, crate::depth::Action::Run);
                if let Some(crate::depth::Phase::Failed(e)) = self.jobs.depth.run.as_ref().map(|r| &r.phase) {
                    return Err(ApiError::app(e.clone()));
                }
            }
            (NodeKind::Depth(_), "cancel") => self.depth_action(id, crate::depth::Action::Cancel),
            // Runs on what is installed and fetched; installing and the
            // gated fetch stay clicks.
            (NodeKind::LtxHdr(_), "render") => {
                self.ltxhdr_action(id, crate::ltxhdr::Action::Run);
                if let Some(crate::ltxhdr::Phase::Failed(e)) = self.jobs.ltxhdr.run.as_ref().map(|r| &r.phase) {
                    return Err(ApiError::app(e.clone()));
                }
            }
            (NodeKind::LtxHdr(_), "cancel") => self.ltxhdr_action(id, crate::ltxhdr::Action::Cancel),
            // Runs on what is installed and fetched; those stay clicks.
            (NodeKind::DiffLight(_), "render") => {
                self.difflight_action(id, crate::difflight::Action::Run);
                if let Some(crate::difflight::Phase::Failed(e)) = self.jobs.difflight.run.as_ref().map(|r| &r.phase) {
                    return Err(ApiError::app(e.clone()));
                }
            }
            (NodeKind::DiffLight(_), "cancel") => self.difflight_action(id, crate::difflight::Action::Cancel),
            // A quote only: `Confirm` is what bills, and that stays a click.
            (NodeKind::GenAi(_), "quote") => self.genai_action(id, crate::genaiui::Action::Prepare),
            (NodeKind::GenAi(_), "cancel") => self.genai_action(id, crate::genaiui::Action::Cancel),
            (NodeKind::GenAiStep(_), a @ ("quote" | "cancel")) => {
                let owner = self.dive.last().filter(|d| d.kind == crate::DiveKind::GenAi).and_then(|d| d.from);
                let step = self.egraph.nodes.iter().filter(|(_, n)| crate::genaiui::is_step(&n.kind)).position(|(k, _)| *k == id);
                let (Some(owner), Some(step)) = (owner, step) else {
                    return Err(ApiError::invalid("not a step of an open GenAI pipeline".to_string()));
                };
                let act = if a == "quote" {
                    crate::genaiui::Action::PrepareStep(step)
                } else {
                    crate::genaiui::Action::Cancel
                };
                self.genai_in_comp(|me| me.genai_action(owner, act));
            }
            (NodeKind::GeoNode(g @ (engine::geo::GeoNode::Physics { .. } | engine::geo::GeoNode::Ragdoll { .. })), a @ ("bake" | "clear")) => {
                if !self.dive.last().is_some_and(|d| d.kind == crate::DiveKind::Geo) {
                    return Err(ApiError::invalid(
                        "a 2dini's nodes bake from inside it: dive_enter the 2dini first".to_string(),
                    ));
                }
                let clear = a == "clear";
                if matches!(g, engine::geo::GeoNode::Physics { .. }) {
                    self.physics_bake(id, clear);
                    if !clear && self.physics_job.as_ref().is_none_or(|j| j.node != id) {
                        return Err(ApiError::app("the Physics node has nothing to bake".to_string()));
                    }
                } else {
                    // A few bones: done before this returns.
                    self.ragdoll_bake(id, clear);
                }
            }
            (NodeKind::GeoNode(engine::geo::GeoNode::Physics { .. }), "cancel") => {
                if self.physics_job.as_ref().is_some_and(|j| j.node == id) {
                    self.physics_cancel();
                }
            }
            (k, a) => {
                let more = match k {
                    NodeKind::GeoNode(engine::geo::GeoNode::Physics { .. }) => "clear, cancel",
                    NodeKind::GeoNode(engine::geo::GeoNode::Ragdoll { .. }) => "clear",
                    _ => "cancel",
                };
                return Err(ApiError::invalid(format!("{} has no '{a}' (it takes: {default}, {more})", type_name(k))))
            }
        }
        if let Some(e) = self.load_errors.get(&id) {
            if busy(self) != Some(id) {
                return Err(ApiError::app(e.clone()));
            }
        }
        let mut v = self.api_jobs();
        v["started"] = json!(action);
        Ok(v)
    }

    fn api_jobs(&self) -> Value {
        let mut jobs = Vec::new();
        if let Some(b) = &self.graph_bake {
            let what = match &b.sink {
                crate::BakeSink::Cache => "bake",
                crate::BakeSink::Write { .. } => "render",
                crate::BakeSink::Capture { .. } => "capture",
                crate::BakeSink::Corridor { .. } => "key-inputs",
                crate::BakeSink::Sammie => "track-plate",
                crate::BakeSink::Flow => "flow-input",
                crate::BakeSink::Morph { .. } => "morph-input",
                crate::BakeSink::Upscale => "upscale-input",
                crate::BakeSink::HdrCnn => "hdrcnn-input",
                crate::BakeSink::SeedVr => "seedvr-input",
                crate::BakeSink::Depth => "depth-input",
                crate::BakeSink::LtxHdr => "ltxhdr-input",
                crate::BakeSink::DiffLight => "difflight-input",
                crate::BakeSink::Stabilize { .. } => "stabilize-input",
                crate::BakeSink::CamTrack { .. } => "camtrack-input",
                crate::BakeSink::Planar => "planar-matte",
            };
            jobs.push(json!({
                "node": b.node, "kind": what,
                "first": b.from, "last": b.len.saturating_sub(1), "frame": b.next,
            }));
        }
        if self.render_bake.is_some() {
            jobs.push(json!({ "kind": "viewer-cache" }));
        }
        if let Some(j) = &self.physics_job {
            let done = j.bake.as_ref().map_or(0, |b| b.frames);
            jobs.push(json!({ "node": j.node, "kind": "physics", "done": done, "total": j.total }));
        }
        if let Some((n, j)) = &self.track_job {
            jobs.push(json!({ "node": n, "kind": "track", "done": j.done, "total": j.total }));
        }
        if let Some((n, _)) = &self.planar_job {
            jobs.push(json!({ "node": n, "kind": "planar-track" }));
        }
        if let Some(r) = self.jobs.corridor.run.as_ref().filter(|r| r.active()) {
            jobs.push(json!({ "node": r.node, "kind": "key" }));
        }
        if let Some(r) = self.jobs.sammie.run.as_ref().filter(|r| r.active()) {
            jobs.push(json!({ "node": r.node, "kind": "roto-track" }));
        }
        if let Some(r) = self.jobs.genai.run.as_ref().filter(|r| r.active()) {
            // "ready" = quoted and waiting for pay & run (a click in the
            // app): nothing more will happen until then.
            jobs.push(json!({ "node": r.node, "kind": "genai", "state": self.genai_state_words(), "step": r.to_step }));
        }
        // Paid runs going beside it, any number.
        for r in self.jobs.genai.background.iter().filter(|r| r.active()) {
            jobs.push(json!({ "node": r.node, "kind": "genai", "state": "running", "step": r.to_step }));
        }
        if let Some(r) = self.jobs.flow.run.as_ref().filter(|r| r.active()) {
            let (done, total) = match &r.phase {
                crate::flowui::Phase::Solving { done, total, .. } => (*done, *total),
                _ => (0, 0),
            };
            jobs.push(json!({ "node": r.node, "kind": "flow", "done": done, "total": total }));
        }
        if let Some(r) = self.jobs.stab.run.as_ref().filter(|r| r.active()) {
            let (done, total) = match &r.phase {
                crate::flowui::Phase::Solving { done, total, .. } => (*done, *total),
                _ => (0, 0),
            };
            jobs.push(json!({ "node": r.node, "kind": "stabilize", "done": done, "total": total }));
        }
        if let Some(r) = self.jobs.camtrack.run.as_ref().filter(|r| r.active()) {
            let (done, total) = match &r.phase {
                crate::flowui::Phase::Solving { done, total, .. } => (*done, *total),
                _ => (0, 0),
            };
            let kind = if self.jobs.camtrack.tracking { "camtrack-track" } else { "camtrack-solve" };
            jobs.push(json!({ "node": r.node, "kind": kind, "done": done, "total": total }));
        }
        if let Some(r) = self.jobs.upscale.run.as_ref().filter(|r| r.active()) {
            let (done, total) = match &r.phase {
                crate::flowui::Phase::Solving { done, total, .. } => (*done, *total),
                _ => (0, 0),
            };
            jobs.push(json!({ "node": r.node, "kind": "upscale", "done": done, "total": total }));
        }
        if let Some(r) = self.jobs.hdrcnn.run.as_ref().filter(|r| r.active()) {
            let (done, total) = match &r.phase {
                crate::flowui::Phase::Solving { done, total, .. } => (*done, *total),
                _ => (0, 0),
            };
            jobs.push(json!({ "node": r.node, "kind": "hdrcnn", "done": done, "total": total }));
        }
        if let Some(r) = self.jobs.seedvr.run.as_ref().filter(|r| r.active()) {
            let (done, total) = match &r.phase {
                crate::seedvr::Phase::Running { done, total, .. } => (*done, *total),
                _ => (0, 0),
            };
            jobs.push(json!({ "node": r.node, "kind": "seedvr", "done": done, "total": total }));
        }
        if let Some(r) = self.jobs.depth.run.as_ref().filter(|r| r.active()) {
            let (done, total) = match &r.phase {
                crate::depth::Phase::Running { done, total, .. } => (*done, *total),
                _ => (0, 0),
            };
            jobs.push(json!({ "node": r.node, "kind": "depth", "done": done, "total": total }));
        }
        if let Some(r) = self.jobs.ltxhdr.run.as_ref().filter(|r| r.active()) {
            let (done, total) = match &r.phase {
                crate::ltxhdr::Phase::Running { done, total, .. } => (*done, *total),
                _ => (0, 0),
            };
            jobs.push(json!({ "node": r.node, "kind": "ltxhdr", "done": done, "total": total }));
        }
        if let Some(r) = self.jobs.difflight.run.as_ref().filter(|r| r.active()) {
            let (done, total) = match &r.phase {
                crate::difflight::Phase::Running { done, total, .. } => (*done, *total),
                _ => (0, 0),
            };
            jobs.push(json!({ "node": r.node, "kind": "difflight", "done": done, "total": total }));
        }
        let errors: Map<String, Value> =
            self.load_errors.iter().map(|(id, e)| (id.to_string(), json!(e))).collect();
        let corridor = self.jobs.corridor.run.as_ref().map(|r| match &r.phase {
            crate::corridorkey::Phase::Failed(e) => json!({ "node": r.node, "failed": e }),
            crate::corridorkey::Phase::Done(d) => json!({ "node": r.node, "done": d }),
            _ => json!({ "node": r.node, "running": true }),
        });
        json!({
            "busy": !jobs.is_empty(),
            "jobs": jobs,
            "status": self.render_status,
            "errors": errors,
            "key": corridor,
            "flow": self.jobs.flow.run.as_ref().map(|r| match &r.phase {
                crate::flowui::Phase::Failed(e) => json!({ "node": r.node, "failed": e }),
                crate::flowui::Phase::Done(d) => json!({ "node": r.node, "done": d }),
                _ => json!({ "node": r.node, "running": true }),
            }),
            "upscale": self.jobs.upscale.run.as_ref().map(|r| match &r.phase {
                crate::flowui::Phase::Failed(e) => json!({ "node": r.node, "failed": e }),
                crate::flowui::Phase::Done(d) => json!({ "node": r.node, "done": d }),
                _ => json!({ "node": r.node, "running": true }),
            }),
            "seedvr": self.jobs.seedvr.run.as_ref().map(|r| match &r.phase {
                crate::seedvr::Phase::Failed(e) => json!({ "node": r.node, "failed": e }),
                crate::seedvr::Phase::Done(d) => json!({ "node": r.node, "done": d }),
                _ => json!({ "node": r.node, "running": true }),
            }),
            "depth": self.jobs.depth.run.as_ref().map(|r| match &r.phase {
                crate::depth::Phase::Failed(e) => json!({ "node": r.node, "failed": e }),
                crate::depth::Phase::Done(d) => json!({ "node": r.node, "done": d }),
                _ => json!({ "node": r.node, "running": true }),
            }),
            "ltxhdr": self.jobs.ltxhdr.run.as_ref().map(|r| match &r.phase {
                crate::ltxhdr::Phase::Failed(e) => json!({ "node": r.node, "failed": e }),
                crate::ltxhdr::Phase::Done(d) => json!({ "node": r.node, "done": d }),
                _ => json!({ "node": r.node, "running": true }),
            }),
            "difflight": self.jobs.difflight.run.as_ref().map(|r| match &r.phase {
                crate::difflight::Phase::Failed(e) => json!({ "node": r.node, "failed": e }),
                crate::difflight::Phase::Done(d) => json!({ "node": r.node, "done": d }),
                _ => json!({ "node": r.node, "running": true }),
            }),
            "roto": self.jobs.sammie.run.as_ref().map(|r| match &r.phase {
                crate::sammie::Phase::Failed(e) => json!({ "node": r.node, "failed": e }),
                crate::sammie::Phase::Done(d) => json!({ "node": r.node, "done": d }),
                _ => json!({ "node": r.node, "running": true }),
            }),
        })
    }

    /// Several calls, one undo step (the caller, `api_recorded`, takes the
    /// step), all or nothing.
    fn api_batch(&mut self, p: &Value) -> R {
        let calls = p["calls"].as_array().cloned().unwrap_or_default();
        let before = self.egraph.clone();
        let mut results: Vec<Value> = Vec::new();
        for (i, c) in calls.iter().enumerate() {
            let step = (|| {
                let name = c.get("method").and_then(|m| m.as_str()).ok_or_else(|| ApiError::invalid("no 'method'"))?;
                let m = api::method(name).ok_or_else(|| ApiError::invalid(format!("unknown method {name}")))?;
                if api::DEFERRED.contains(&name)
                    || matches!(name, "batch" | "undo" | "redo" | "setup_load" | "dive_enter" | "dive_exit" | "api_schema")
                {
                    return Err(ApiError::invalid(format!("{name} cannot ride in a batch")));
                }
                let mut params = c.get("params").cloned().unwrap_or_else(|| json!({}));
                subst(&mut params, &results)?;
                api::validate(&m, &params)?;
                self.api_call(name, &params)
            })();
            match step {
                Ok(v) => results.push(v),
                Err(e) => {
                    self.egraph = before;
                    self.selected.retain(|s| self.egraph.nodes.contains_key(s));
                    self.playback_dirty = true;
                    return Err(ApiError { code: e.code, message: format!("call {i}: {} — nothing was changed", e.message) });
                }
            }
        }
        Ok(json!({ "results": results }))
    }

    /// Scene-linear statistics of the viewed output over a rectangle.
    fn api_sample(&self, rect: [Option<u64>; 4], settled: bool) -> R {
        let (bytes, w, h) =
            self.comp.read_output_rgba16().ok_or_else(|| ApiError::app("nothing is rendered yet"))?;
        let [x, y, rw, rh] = rect;
        let (x, y) = (x.unwrap_or(0) as u32, y.unwrap_or(0) as u32);
        if x >= w || y >= h {
            return Err(ApiError::invalid(format!("({x}, {y}) is outside the {w}x{h} output")));
        }
        let rw = (rw.unwrap_or((w - x) as u64).max(1) as u32).min(w - x);
        let rh = (rh.unwrap_or((h - y) as u64).max(1) as u32).min(h - y);
        let (mut sum, mut lo, mut hi) = ([0f64; 4], [f32::INFINITY; 4], [f32::NEG_INFINITY; 4]);
        for py in y..y + rh {
            let row = (py * w + x) as usize * 8;
            for px in bytes[row..row + rw as usize * 8].chunks_exact(8) {
                for c in 0..4 {
                    let v = half::f16::from_le_bytes([px[c * 2], px[c * 2 + 1]]).to_f32();
                    sum[c] += v as f64;
                    lo[c] = lo[c].min(v);
                    hi[c] = hi[c].max(v);
                }
            }
        }
        let n = (rw as f64) * (rh as f64);
        Ok(json!({
            "mean": sum.map(|s| s / n),
            "min": lo,
            "max": hi,
            "rect": [x, y, rw, rh],
            "size": [w, h],
            "channels": "rgba, scene-linear, working space",
            "frame": self.shown_tick,
            "view": self.egraph.active_view(),
            "settled": settled,
        }))
    }

    fn api_capture(&self, max_size: u32, settled: bool) -> R {
        let (w, h, rgba) = self.read_output_rgba8().map_err(|e| ApiError::app(format!("capture: {e:#}")))?;
        let (ow, oh, small) = shrink(w, h, &rgba, max_size);
        let bytes = png(ow, oh, &small).map_err(|e| ApiError::app(format!("png: {e:#}")))?;
        Ok(json!({
            "png_base64": api::b64(&bytes),
            "width": ow,
            "height": oh,
            "source_size": [w, h],
            "frame": self.shown_tick,
            "view": self.egraph.active_view(),
            "settled": settled,
        }))
    }
}

/// The viewer's per-screen state: what the remote screen's viewer pass
/// swaps in for the Mac's (`remote_tick`), so each screen pans and zooms
/// on its own. Everything else about the viewer (the comp, the gizmos'
/// drags, the brush) is the one app's.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct RemoteView {
    view_pan: egui::Vec2,
    view_zoom: f32,
    last_viewer_center: Option<egui::Pos2>,
    last_pane_key: (bool, u8),
    viewer_rect: egui::Rect,
    hold_rect: Option<egui::Rect>,
    pub(crate) image_rect: egui::Rect,
    pending_view_nav: Option<(f32, egui::Vec2, Option<egui::Pos2>)>,
    dual_view: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for RemoteView {
    fn default() -> Self {
        Self {
            view_pan: egui::Vec2::ZERO,
            view_zoom: 1.0,
            last_viewer_center: None,
            last_pane_key: (false, 0),
            viewer_rect: egui::Rect::NOTHING,
            hold_rect: None,
            image_rect: egui::Rect::NOTHING,
            pending_view_nav: None,
            // One pane on a tablet; the Mac keeps its dual view.
            dual_view: false,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl RemoteView {
    fn grab(g: &mut Gfx) -> Self {
        Self {
            view_pan: g.view_pan,
            view_zoom: g.view_zoom,
            last_viewer_center: g.last_viewer_center,
            last_pane_key: g.last_pane_key,
            viewer_rect: g.viewer_rect,
            hold_rect: g.hold_rect,
            image_rect: g.image_rect,
            pending_view_nav: g.pending_view_nav.take(),
            dual_view: g.dual_view,
        }
    }

    fn put(self, g: &mut Gfx) {
        g.view_pan = self.view_pan;
        g.view_zoom = self.view_zoom;
        g.last_viewer_center = self.last_viewer_center;
        g.last_pane_key = self.last_pane_key;
        g.viewer_rect = self.viewer_rect;
        g.hold_rect = self.hold_rect;
        g.image_rect = self.image_rect;
        g.pending_view_nav = self.pending_view_nav;
        g.dual_view = self.dual_view;
    }
}

/// The drag the last recorded call belonged to (node, channels), if any.
static API_DRAG: std::sync::Mutex<Option<(u64, Vec<String>)>> = std::sync::Mutex::new(None);

/// What a call continues, if it is one step of a drag: the node and the
/// channels it moves. Any other call ends the drag.
fn drag_key(method: &str, p: &Value) -> Option<(u64, Vec<String>)> {
    if method != "node_set" || p.get("drag").and_then(|v| v.as_bool()) != Some(true) {
        return None;
    }
    let id = p.get("id")?.as_u64()?;
    let mut chans: Vec<String> = p.get("values")?.as_object()?.keys().cloned().collect();
    chans.sort();
    Some((id, chans))
}

#[cfg(test)]
mod tests {
    #[test]
    fn only_a_drag_on_the_same_channels_continues() {
        let a = drag_key("node_set", &json!({ "id": 3, "values": { "BlurSize": 2.0 }, "drag": true }));
        let b = drag_key("node_set", &json!({ "id": 3, "values": { "BlurSize": 5.0 }, "drag": true }));
        assert!(a.is_some() && a == b, "the same channel dragged again is the same step");
        let other = drag_key("node_set", &json!({ "id": 3, "values": { "BlurSizeY": 5.0 }, "drag": true }));
        assert_ne!(a, other, "another channel is another step");
        assert_eq!(drag_key("node_set", &json!({ "id": 3, "values": { "BlurSize": 5.0 } })), None);
        assert_eq!(drag_key("node_key", &json!({ "id": 3, "drag": true })), None);
    }

    use super::*;

    #[test]
    fn merge_is_rfc7386_for_objects_and_replaces_the_rest() {
        let mut v = json!({ "Blur": { "size": 4.0, "mode": "Box", "list": [1, 2] } });
        merge(&mut v, &json!({ "Blur": { "size": 9.0, "list": [3] } }));
        assert_eq!(v, json!({ "Blur": { "size": 9.0, "mode": "Box", "list": [3] } }));
    }

    #[test]
    fn shrink_averages_blocks_and_keeps_small_images() {
        let img = [0u8, 0, 0, 255, 255, 255, 255, 255].repeat(2); // 2x2
        let (w, h, px) = shrink(2, 2, &img, 1);
        assert_eq!((w, h), (1, 1));
        assert_eq!(px, vec![127, 127, 127, 255]);
        assert_eq!(shrink(2, 2, &img, 8).2, img);
    }

    /// Every method in the protocol table is answered by the app. A method
    /// added to `api::methods()` without an arm here fails this, not a
    /// user's script.
    #[test]
    fn a_batch_reference_becomes_the_id_that_call_made() {
        let results = vec![json!({ "id": 7 }), json!({ "ok": true })];
        let mut p = json!({ "src": "$0", "inputs": ["$0", 3], "name": "$F * 2", "expr": "$x" });
        subst(&mut p, &results).unwrap();
        assert_eq!(p, json!({ "src": 7, "inputs": [7, 3], "name": "$F * 2", "expr": "$x" }));
        // A call that made no node, or has not run, is an error — never a 0.
        assert!(subst(&mut json!({ "src": "$1" }), &results).is_err());
        assert!(subst(&mut json!({ "src": "$5" }), &results).is_err());
    }

    #[test]
    fn interp_names_round_trip() {
        for i in [anim::Interp::Hold, anim::Interp::Linear, anim::Interp::Smooth, anim::Interp::Bezier] {
            assert_eq!(interp_of(&interp_name(i)).unwrap(), i);
        }
        for e in anim::Ease::ALL {
            for d in anim::EaseDir::ALL {
                let i = anim::Interp::Ease(e, d);
                assert_eq!(interp_of(&interp_name(i)).unwrap(), i);
            }
        }
        assert!(interp_of("ease").is_err());
        assert!(interp_of("ease-bounce-sideways").is_err());
    }

    #[test]
    fn the_app_implements_every_method() {
        let src = include_str!("apiserve.rs");
        for m in api::methods() {
            if m.name == "api_schema" {
                continue; // the server answers it without the app
            }
            assert!(src.contains(&format!("\"{}\"", m.name)), "apiserve.rs does not handle {}", m.name);
        }
    }

    /// A script that guessed the edge shape used to get silence and a graph
    /// with nothing wired; now it gets told the shape.
    #[test]
    fn malformed_subgraph_edges_are_an_error_not_silence() {
        let mut g = engine::graph::GeoParams::default();
        g.graph.nodes.insert(1, engine::geo::GeoNode::Output);
        g.graph.nodes.insert(2, engine::geo::GeoNode::Output);
        let mut v = serde_json::to_value(NodeKind::Geo(g)).unwrap();
        v["Geo"]["graph"]["edges"] = json!([[[2, 0], 1]]);
        let k: NodeKind = serde_json::from_value(v.clone()).unwrap();
        let NodeKind::Geo(p) = k else { panic!() };
        assert_eq!(p.graph.edges.get(&(2, 0)), Some(&1));
        for bad in [json!([{"from": 1, "to": 2}]), json!({"2": 1}), json!([[2, 0, 1]])] {
            v["Geo"]["graph"]["edges"] = bad.clone();
            let e = serde_json::from_value::<NodeKind>(v.clone()).err().map(|e| e.to_string());
            assert!(e.as_deref().is_some_and(|e| e.contains("[[consumer, input], producer]")), "{bad}: {e:?}");
        }
        // The legacy empty object still reads as no wires.
        v["Geo"]["graph"]["edges"] = json!({});
        assert!(serde_json::from_value::<NodeKind>(v).is_ok());
    }

    /// `null` deletes, so one patch switches an enum's variant.
    #[test]
    fn a_null_in_a_patch_deletes_the_key() {
        let mut v = json!({"GeoNode": {"Scatter": {"count": 1}}});
        merge(&mut v, &json!({"GeoNode": {"Scatter": null, "Output": {}}}));
        assert_eq!(v, json!({"GeoNode": {"Output": {}}}));
    }

    /// Inside a 2dini, `Noise` and `Transform` mean the 2dini's own nodes.
    #[test]
    fn a_dive_prefers_its_own_node_types() {
        let geo = cats_for(Some(crate::DiveKind::Geo));
        for t in ["Noise", "Transform", "Merge"] {
            let k = (find_entry(t, geo).unwrap().make)(0);
            assert!(matches!(k, NodeKind::GeoNode(_)), "{t} inside a 2dini made {}", k.name());
        }
        let k = (find_entry("Noise", cats_for(None)).unwrap().make)(0);
        assert!(matches!(k, NodeKind::Noise(_)), "Noise in a comp made {}", k.name());
        // The sub-graph catalog carries default JSON and input names.
        let types = subgraph_types(geo);
        let scatter = types.iter().find(|t| t["type"] == "Scatter").expect("Scatter listed");
        assert_eq!(scatter["inputs"], json!(["region", "density"]));
        assert!(scatter["kind"]["GeoNode"]["Scatter"]["count"].is_number());
    }

    /// Scripts written before a node was renamed in the UI still create it.
    #[test]
    fn old_type_names_still_resolve() {
        for (was, now) in [("Read", "Load"), ("Write", "Render"), ("Shuffle", "Reorder")] {
            let e = find_entry(was, (false, false, false, false, false, false)).unwrap_or_else(|| panic!("{was} resolves to nothing"));
            assert_eq!(e.label, now);
        }
    }

    /// `node_connect` and `node_create`'s `inputs` used to answer
    /// "connected" when `connect_or_bridge` quietly made no wire; the check
    /// they now run says why, and agrees with the wiring rule.
    #[test]
    fn a_wire_nothing_takes_is_refused_by_name() {
        use engine::graph::{DepthParams, PromptParams, WriteParams};
        let grade = NodeKind::Grade(Default::default());
        let prompt = NodeKind::Prompt(PromptParams::default());
        let why = wire_refusal(&prompt, 0, &grade, 0).expect("text into a Grade is not a wire");
        assert!(why.contains("Text") && why.contains("Image"), "{why}");
        let depth = NodeKind::Depth(DepthParams::default());
        assert_eq!(wire_refusal(&depth, 0, &NodeKind::Write(WriteParams::default()), 0), None, "a Render takes data");
        assert_eq!(wire_refusal(&NodeKind::read(0, ""), 0, &grade, 0), None);
    }

    /// Every built-in type round-trips through the protocol's type name, so
    /// `graph_get`'s "type" is always something `node_create` accepts.
    #[test]
    fn type_names_resolve_back_to_their_catalog_entry() {
        for e in editor::node_catalog().into_iter().filter(|e| editor::offered(e.cat, false, false, false, false)) {
            let name = type_name(&(e.make)(0));
            let lower = name.to_ascii_lowercase();
            assert!(
                editor::node_catalog().iter().any(|f| f.label.to_ascii_lowercase() == lower
                    || (f.make)(0).name().to_ascii_lowercase() == lower),
                "{} reports type {name}, which node_create would not find",
                e.label
            );
        }
    }
}
