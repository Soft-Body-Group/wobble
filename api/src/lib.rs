//! wobble's scripting protocol.
//!
//! ONE language-neutral protocol (CLAUDE.md, "Direction already decided"):
//! newline-delimited JSON-RPC 2.0 over a LOCAL TCP socket, described by the
//! method table in [`methods`]. The app serves it (`wobble --api`); every
//! client — the Python module, the MCP bridge (`wobble-mcp`), a Rust tool —
//! is a thin wrapper that learns the methods from `api_schema` rather than
//! hard-coding them, so no client is privileged and none can drift.
//!
//! Rules the protocol carries, whoever the caller is:
//! - It edits AUTHORED data, never the engine graph; one call is one undo step.
//! - Nothing here runs per frame or per pixel. A call is a command the app
//!   drains at the top of its UI pass, between frames.
//! - Localhost only, and off unless the user starts the app with `--api`.

pub mod client;
pub mod mcp;
pub mod server;

use serde_json::{json, Value};

/// The port the app listens on and clients dial when nothing says otherwise.
/// `WOBBLE_API_PORT` overrides it on both sides.
pub const DEFAULT_PORT: u16 = 7979;

/// Protocol revision. Bumped on a breaking change to a method's shape;
/// adding a method or an optional param is not breaking.
pub const VERSION: u32 = 1;

/// The port from `WOBBLE_API_PORT`, else [`DEFAULT_PORT`].
pub fn port_from_env() -> u16 {
    std::env::var("WOBBLE_API_PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_PORT)
}

/// One callable method.
#[derive(Clone, Debug)]
pub struct Method {
    /// snake_case, and a valid MCP tool name (`^[a-zA-Z0-9_-]{1,64}$`), so
    /// the bridge can expose the table unchanged.
    pub name: &'static str,
    pub summary: &'static str,
    /// A JSON Schema object for the params.
    pub params: Value,
    /// Changes the document (one undo step), as opposed to reading it or
    /// moving the viewer.
    pub mutates: bool,
}

/// Methods that leave the DOCUMENT alone but still act: they move the
/// viewer or the playhead, write a file, start a render or a process. Not
/// `mutates` (no undo step), and not read-only either — a client that
/// auto-approves read-only calls must not auto-approve these.
pub const ACTS: &[&str] = &[
    "view_set", "time_set", "setup_save", "addon_panel_set", "addon_command", "addon_status",
    "blender_session", "blender_view", "dive_enter", "dive_exit", "viewer_set", "comp_set",
    "node_run", "selection_set", "render_take", "render_release",
    "render_shared", "record", "graph_op", "ui_set", "render_receive", "render_send",
];

/// Methods that can throw work away: a client should confirm these.
pub const DESTRUCTIVE: &[&str] = &["node_delete", "setup_load", "patch_mod_delete"];

/// Methods answered on a LATER UI pass (they wait for a frame or an event),
/// so they cannot ride inside a `batch`.
pub const DEFERRED: &[&str] = &["viewer_capture", "pixel_sample", "addon_events", "node_run"];

impl Method {
    /// Reads and nothing else: safe to call without asking.
    pub fn read_only(&self) -> bool {
        !self.mutates && !ACTS.contains(&self.name)
    }
    pub fn destructive(&self) -> bool {
        DESTRUCTIVE.contains(&self.name)
    }
}

fn obj(props: Value, required: &[&str]) -> Value {
    json!({ "type": "object", "properties": props, "required": required, "additionalProperties": false })
}

/// The whole protocol. The app implements every entry (a test in the app
/// checks that); clients generate their surface from it.
pub fn methods() -> Vec<Method> {
    let id = json!({ "type": "integer", "description": "node id, as graph_get reports it" });
    vec![
        Method {
            name: "api_schema",
            summary: "This table: every method, its params as JSON Schema, and whether it edits the comp.",
            params: obj(json!({}), &[]),
            mutates: false,
        },
        Method {
            name: "auth",
            summary: "Unlock a connection from another machine with that wobble's pairing code (shown in its network window). Every other call on a network connection is refused until this succeeds; on this machine it is a no-op.",
            params: obj(json!({ "key": { "type": "string" } }), &["key"]),
            mutates: false,
        },
        Method {
            name: "network_peers",
            summary: "Other wobbles heard on the local network (their id, machine name, version, addresses and network port, and whether a pairing code for them is remembered), and whether this one is shared and on which port.",
            params: obj(json!({}), &[]),
            mutates: false,
        },
        Method {
            name: "nodes_receive",
            summary: "Paste a node clip (the JSON a copy makes) into this graph as new nodes, wired as they were among themselves, beside what is there. 'from' names the sender for the note. Returns the new ids and any Load whose footage is not on this machine.",
            params: obj(json!({
                "clip": { "type": "string" },
                "from": { "type": "string" }
            }), &["clip"]),
            mutates: true,
        },
        Method {
            name: "render_take",
            summary: "Help render: render frames first..last of a Render node of a SAVED setup this machine can read (a shared folder), in a hidden worker, files numbered from 'start' under the node's file name pattern. Refused for setups with Blender, process-backed or GenAI nodes, and from another machine unless 'let paired wobbles render here' is on. Returns a job id.",
            params: obj(json!({
                "setup": { "type": "string" },
                "node": { "type": "integer" },
                "first": { "type": "integer" },
                "last": { "type": "integer" },
                "start": { "type": "integer" },
                "pattern": { "type": "string" }
            }), &["setup", "node", "first", "last", "start", "pattern"]),
            mutates: false,
        },
        Method {
            name: "render_shared",
            summary: "Render a Render node with paired wobbles helping (network_peers ids): the frames are split between this wobble and them, each renders in the background and the frames land where the node writes. The setup must be saved, in a folder the helpers can see. Progress in the network window and network_render_status.",
            params: obj(json!({
                "id": { "type": "integer" },
                "peers": { "type": "array", "items": { "type": "string" } }
            }), &["id", "peers"]),
            mutates: false,
        },
        Method {
            name: "network_render_status",
            summary: "The shared render this wobble started: running or not, each share's helper, frames and state, and the closing note.",
            params: obj(json!({}), &[]),
            mutates: false,
        },
        Method {
            name: "render_status",
            summary: "A render_take job: starting / running / done / failed, the frame files finished so far, the total.",
            params: obj(json!({ "job": { "type": "string" } }), &["job"]),
            mutates: false,
        },
        Method {
            name: "render_fetch",
            summary: "One finished frame file of a render_take job, base64.",
            params: obj(json!({ "job": { "type": "string" }, "file": { "type": "string" } }), &["job", "file"]),
            mutates: false,
        },
        Method {
            name: "render_send",
            summary: "Send the frames a Render node last wrote to a paired wobble (a network_peers id), which saves them under its inbox in <this machine>/<render name>. It must have 'accept renders from paired wobbles' on. Runs in the background; progress in render_send_status.",
            params: obj(json!({ "id": id, "peer": { "type": "string" } }), &["id", "peer"]),
            mutates: false,
        },
        Method {
            name: "render_send_status",
            summary: "The render this wobble is sending (render_send): whether it runs, to whom, files done of total, and the closing note.",
            params: obj(json!({}), &[]),
            mutates: false,
        },
        Method {
            name: "render_receive",
            summary: "One chunk of one file of a render another wobble sends: 'from' (its machine), 'name' (the render) and 'file' are plain names, 'offset' and 'size' in bytes, 'data' base64. Saved under this wobble's inbox once whole; pictures and movies only. Refused unless 'accept renders from paired wobbles' is on.",
            params: obj(json!({
                "from": { "type": "string" },
                "name": { "type": "string" },
                "file": { "type": "string" },
                "offset": { "type": "integer" },
                "size": { "type": "integer" },
                "data": { "type": "string" }
            }), &["from", "name", "file", "offset", "size", "data"]),
            mutates: false,
        },
        Method {
            name: "render_release",
            summary: "Stop a render_take job if it runs and delete its frames here.",
            params: obj(json!({ "job": { "type": "string" } }), &["job"]),
            mutates: false,
        },
        Method {
            name: "app_info",
            summary: "Protocol version, playhead, frame range, fps, playing, working colour space, the open setup file, the viewed node, the selection, the viewer's exposure/gamma/channel/proxy and which graph is open (dive).",
            params: obj(json!({}), &[]),
            mutates: false,
        },
        Method {
            name: "node_types",
            summary: "Every node type the app can create: its type name (what node_create takes), category, maturity (alpha/beta/production), input count and animatable channels.",
            params: obj(json!({}), &[]),
            mutates: false,
        },
        Method {
            name: "graph_get",
            summary: "The comp graph: each node's id, type, name, position, bypass, input wires, channel values at the playhead and which channels are animated; plus the viewed nodes.",
            params: obj(json!({
                "include_kind": { "type": "boolean", "description": "also return each node's full parameter record (as node_get does)" }
            }), &[]),
            mutates: false,
        },
        Method {
            name: "node_get",
            summary: "One node in full: its parameter record ('kind', the same JSON a setup file stores — node_patch takes this shape), channel values (in panel order, 'channels', with a display label each, 'labels'), every keyframe with its interpolation, what drives each driven channel (an expression or a patch cable), and its error if it has one.",
            params: obj(json!({ "id": id }), &["id"]),
            mutates: false,
        },
        Method {
            name: "node_create",
            summary: "Add a node by type name (see node_types). Optionally wire sources into its inputs in order, and for a Load give the image-sequence path (or a project 'asset') to load. Returns the new id.",
            params: obj(json!({
                "type": { "type": "string" },
                "x": { "type": "number" },
                "y": { "type": "number" },
                "inputs": { "type": "array", "items": { "type": "integer" }, "description": "source node ids for inputs 0, 1, …" },
                "path": { "type": "string", "description": "Load only: a sequence path (dir/name.####.exr, a directory, or a movie)" },
                "asset": { "type": "object", "description": "Load only, instead of path: an organisicator asset by database id, {id: N} (ids from organisicator's API, ~/.organisicator/api.json). Re-found through organisicator on every load, so a moved project or renamed shot keeps loading; organisicator must be running." },
                "name": { "type": "string" }
            }), &["type"]),
            mutates: true,
        },
        Method {
            name: "node_delete",
            summary: "Remove a node and its wires.",
            params: obj(json!({ "id": id }), &["id"]),
            mutates: true,
        },
        Method {
            name: "node_connect",
            summary: "Wire src's output into dst's input. A mistyped pipe gets a converter spliced in, and a wire back upstream becomes a loop, exactly as dragging it in the graph would.",
            params: obj(json!({
                "src": { "type": "integer" },
                "dst": { "type": "integer" },
                "input": { "type": "integer", "description": "default 0" },
                "output": { "type": "integer", "description": "default 0; only multi-output nodes have more" }
            }), &["src", "dst"]),
            mutates: true,
        },
        Method {
            name: "node_disconnect",
            summary: "Remove the wire into one input.",
            params: obj(json!({ "dst": { "type": "integer" }, "input": { "type": "integer" } }), &["dst", "input"]),
            mutates: true,
        },
        Method {
            name: "node_set",
            summary: "Set channel values by name (see node_types / graph_get). An animated channel gets a key at 'frame' (default: the playhead); a static one has its value set. A locked or cable-driven channel is refused. drag=true marks one step of a continuous gesture: consecutive drag calls on the same node and channels are ONE undo step.",
            params: obj(json!({
                "id": id,
                "values": { "type": "object", "additionalProperties": { "type": "number" } },
                "frame": { "type": "integer" },
                "drag": { "type": "boolean", "description": "part of a drag: folds into the previous drag call's undo step" }
            }), &["id", "values"]),
            mutates: true,
        },
        Method {
            name: "node_key",
            summary: "Set keyframes on channels. 'keys' maps channel name to a list of [frame, value] or [frame, value, interp] (interp: hold, linear, smooth, bezier; default keeps the key's own, linear for a new one). A locked channel is refused.",
            params: obj(json!({
                "id": id,
                "keys": { "type": "object", "additionalProperties": { "type": "array", "items": { "type": "array", "minItems": 2, "maxItems": 3 } } }
            }), &["id", "keys"]),
            mutates: true,
        },
        Method {
            name: "node_unkey",
            summary: "Remove all animation from channels, leaving each at its value at the playhead.",
            params: obj(json!({ "id": id, "channels": { "type": "array", "items": { "type": "string" } } }), &["id", "channels"]),
            mutates: true,
        },
        Method {
            name: "node_patch",
            summary: "Edit anything else about a node: 'kind' is merged into its parameter record (the JSON node_get returns — paths, modes, enums, lists; objects merge key by key, anything else replaces), plus its name, bypass and position.",
            params: obj(json!({
                "id": id,
                "kind": { "type": "object" },
                "name": { "type": "string" },
                "bypass": { "type": "boolean" },
                "x": { "type": "number" },
                "y": { "type": "number" }
            }), &["id"]),
            mutates: true,
        },
        Method {
            name: "view_set",
            summary: "Show a node in the viewer (slot 'a', the default, or 'b').",
            params: obj(json!({ "id": id, "slot": { "type": "string", "enum": ["a", "b"] } }), &["id"]),
            mutates: false,
        },
        Method {
            name: "time_set",
            summary: "Move the playhead to a comp frame.",
            params: obj(json!({ "frame": { "type": "integer" } }), &["frame"]),
            mutates: false,
        },
        Method {
            name: "viewer_capture",
            summary: "A PNG of what the viewer shows, through the display transform, once the frame at the playhead has rendered with every edit so far. Returned as base64 ('png_base64').",
            params: obj(json!({
                "max_size": { "type": "integer", "description": "longest side in pixels, default 768" }
            }), &[]),
            mutates: false,
        },
        Method {
            name: "record",
            summary: "The built-in screen recorder, as the record button and ⌘⇧R drive it: 'start' begins a take (of the whole window, or 'area': 'viewer'), 'stop' ends it and opens the review, 'save' writes the take under review to the clips folder (~/Movies/Wobble, or WOBBLE_RECORD_DIR) and 'status' says where things stand — recording, seconds so far, frames under review, the saved mp4/gif once a save lands, or the error. Nothing here is undoable.",
            params: obj(json!({
                "action": { "type": "string", "enum": ["start", "stop", "save", "status"] },
                "area": { "type": "string", "enum": ["window", "viewer"], "description": "start only; default: the recorder's own setting" },
                "cursor": { "type": "boolean", "description": "start only: draw the pointer into the clip" },
                "watermark": { "type": "boolean", "description": "start only: the \"made with wobble\" mark" },
                "gif": { "type": "boolean", "description": "start only: also save a GIF" }
            }), &["action"]),
            mutates: false,
        },
        Method {
            name: "graph_op",
            summary: "A graph-layout op, as the ` pie fires it: 'tidy' lays the selection (or the whole comp when nothing is selected) out as a top-to-bottom flow, 'frame' fits the selection (or every node) in the graph pane. Moves nodes and the view, never pixels.",
            params: obj(json!({ "action": { "type": "string", "enum": ["tidy", "frame"] } }), &["action"]),
            mutates: false,
        },
        Method {
            name: "ui_set",
            summary: "What the window shows around the picture, for a clip or a screenshot: 'bottom' picks the properties or the curve editor in the lower pane; 'hud_pin' pins these nodes' viewer HUDs and manipulators (X-click pins: shown whatever the hud-follow setting says; [] clears). Never touches prefs.",
            params: obj(json!({
                "bottom": { "type": "string", "enum": ["properties", "curves"] },
                "hud_pin": { "type": "array", "items": id }
            }), &[]),
            mutates: false,
        },
        Method {
            name: "cost_get",
            summary: "What the graph's colours say: nodes that take their own pass or sit downstream of one (slow), the pass chains that dominate a frame (heavy), and nodes carrying a non-commercial model licence.",
            params: obj(json!({}), &[]),
            mutates: false,
        },
        Method {
            name: "setup_load",
            summary: "Open a setup file (.wbl).",
            params: obj(json!({ "path": { "type": "string" } }), &["path"]),
            mutates: true,
        },
        Method {
            name: "setup_save",
            summary: "Save the comp to a setup file (default: the file it was opened from).",
            params: obj(json!({ "path": { "type": "string" } }), &[]),
            mutates: false,
        },
        Method {
            name: "undo",
            summary: "Undo one step (an API call is one step).",
            params: obj(json!({}), &[]),
            mutates: true,
        },
        Method {
            name: "redo",
            summary: "Redo one step.",
            params: obj(json!({}), &[]),
            mutates: true,
        },
        // Addons (`addon.json`): the app launches an addon's process with
        // WOBBLE_API_PORT and WOBBLE_ADDON set; the process waits on
        // `addon_events` for what the user did with the commands and panels
        // its manifest declared, and answers with ordinary calls.
        Method {
            name: "addon_events",
            summary: "Wait for what the user did with an addon's commands, panels and pickers: [{type: 'command', command}, {type: 'panel', panel, row, value, activate?}, {type: 'pick', picker, for: 'load', node}]. Returns as soon as there is one, or empty after timeout_ms (default 20000, max 30000).",
            params: obj(json!({
                "addon": { "type": "string", "description": "the addon's id (WOBBLE_ADDON)" },
                "timeout_ms": { "type": "integer" }
            }), &["addon"]),
            mutates: false,
        },
        Method {
            name: "load_set",
            summary: "Point an existing Load at other footage, as a new clip with its own frame range: a 'path' (image sequence, movie or still), or an organisicator 'asset' by database id ({id: N}). What an addon's Load picker answers a {type: 'pick', node} event with.",
            params: obj(json!({
                "id": id,
                "path": { "type": "string" },
                "asset": { "type": "object", "description": "an organisicator asset by database id, {id: N}; re-found through organisicator on every load" }
            }), &["id"]),
            mutates: true,
        },
        Method {
            name: "addon_panel_set",
            summary: "Set rows of an addon's panel: values maps a row id to its value (number for a slider, bool for a toggle, string for text or a field, string for a button's label; a choice or a list takes the picked string, or {options: [..], value} to replace its options). open shows or hides the panel.",
            params: obj(json!({
                "addon": { "type": "string" },
                "panel": { "type": "string" },
                "values": { "type": "object" },
                "open": { "type": "boolean" }
            }), &["addon", "panel"]),
            mutates: false,
        },
        Method {
            name: "addon_list",
            summary: "Every addon found: id, label, version, description, whether it is enabled and running, its status line, commands and panels.",
            params: obj(json!({}), &[]),
            mutates: false,
        },
        Method {
            name: "addon_command",
            summary: "Run an enabled addon's command, as if picked from the addons menu (the addon's process receives it through addon_events).",
            params: obj(json!({
                "addon": { "type": "string" },
                "command": { "type": "string" }
            }), &["addon", "command"]),
            mutates: false,
        },
        Method {
            name: "blender_session",
            summary: "A Blender node's live link: open launches Blender on the node's .blend with the wobble bridge; connect joins a Blender already listening on a port (its wobble addon tab); detach drops the link and leaves Blender running; quit asks Blender to close; refresh re-sends the frame, spec and textures.",
            params: obj(json!({
                "id": id,
                "action": { "type": "string", "enum": ["open", "connect", "detach", "quit", "refresh"] },
                "port": { "type": "integer", "description": "for connect: the port the addon listens on" }
            }), &["id", "action"]),
            mutates: false,
        },
        Method {
            name: "blender_view",
            summary: "Move a Blender node's eye from a script: the pivot, the view->world rotation quaternion (w,x,y,z), the distance, the lens; the 3D view in free mode, the camera in camera mode. Omitted fields keep their value; moving=true draws at one sample and half resolution until a call with moving=false.",
            params: obj(json!({
                "id": id,
                "loc": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                "rot": { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 },
                "dist": { "type": "number" },
                "lens": { "type": "number" },
                "ortho": { "type": "boolean" },
                "orbit": { "type": "array", "items": { "type": "number" }, "minItems": 2, "maxItems": 2, "description": "a turntable step in screen pixels (dx, dy), applied after the fields above" },
                "moving": { "type": "boolean" }
            }), &["id"]),
            mutates: false,
        },
        Method {
            name: "blender_status",
            summary: "A Blender node's session as its panel shows it: connected, Blender's version and scene range, the frame on the canvas (Blender frame, scene version, settled/final) against the frame the playhead asks for, the eye (pivot, rotation, distance, lens), counts and the last log lines. Poll it after a seek to know when the frame has landed.",
            params: obj(json!({ "id": id }), &["id"]),
            mutates: false,
        },
        Method {
            name: "addon_status",
            summary: "Show a one-line status for an addon (in the addons window and under its panels).",
            params: obj(json!({
                "addon": { "type": "string" },
                "text": { "type": "string" }
            }), &["addon", "text"]),
            mutates: false,
        },
        // ---- Values that drive values: the patch bay and expressions ----
        Method {
            name: "expr_set",
            summary: "Drive channels of a node with an expression, as typing it into the value field does: `../grade12/gainr * 2` (another node's param; names come from graph_get's 'path'), `./mix` (this node), `$F` or `t` (the frame), arithmetic and sin/cos/abs/min/max/clamp/smoothstep… An empty expr removes it; constant maths (`10/2`) is written as a number. Refused if a path names nothing.",
            params: obj(json!({
                "id": id,
                "channels": { "type": "array", "items": { "type": "string" } },
                "expr": { "type": "string" }
            }), &["id", "channels", "expr"]),
            mutates: true,
        },
        Method {
            name: "patch_get",
            summary: "The patch bay: every value node (mult/add/max/min with a constant, or an expression over its jacks a and b) and every cable. An end is {node, channel} (a node's channel), {mod} (a value node's output) or {mod, slot} (its input jack 0 = a, 1 = b).",
            params: obj(json!({}), &[]),
            mutates: false,
        },
        Method {
            name: "patch_link",
            summary: "Cable one value into another: from {node, channel} or {mod}, to {node, channel} or {mod, slot}. Replaces whatever fed 'to'. A locked channel is refused.",
            params: obj(json!({ "from": { "type": "object" }, "to": { "type": "object" } }), &["from", "to"]),
            mutates: true,
        },
        Method {
            name: "patch_unlink",
            summary: "Remove the cable into 'to' ({node, channel} or {mod, slot}). The channel keeps its authored value.",
            params: obj(json!({ "to": { "type": "object" } }), &["to"]),
            mutates: true,
        },
        Method {
            name: "patch_mod_create",
            summary: "Add a value node to the patch bay: kind mult/add/max/min (with 'value', the constant standing in for jack b) or expr (with 'expr' over a, b and t). Returns its id for patch_link.",
            params: obj(json!({
                "kind": { "type": "string", "enum": ["mult", "add", "max", "min", "expr"] },
                "value": { "type": "number" },
                "expr": { "type": "string" },
                "x": { "type": "number" },
                "y": { "type": "number" }
            }), &["kind"]),
            mutates: true,
        },
        Method {
            name: "patch_mod_set",
            summary: "Change a value node's constant ('value') or expression text ('expr').",
            params: obj(json!({
                "id": { "type": "integer" },
                "value": { "type": "number" },
                "expr": { "type": "string" }
            }), &["id"]),
            mutates: true,
        },
        Method {
            name: "patch_mod_delete",
            summary: "Remove a value node and every cable touching it.",
            params: obj(json!({ "id": { "type": "integer" } }), &["id"]),
            mutates: true,
        },
        // ---- Keys and locks ----
        Method {
            name: "node_key_delete",
            summary: "Remove single keyframes: 'keys' maps channel name to a list of frames. A channel left with no keys holds its value at the playhead.",
            params: obj(json!({
                "id": id,
                "keys": { "type": "object", "additionalProperties": { "type": "array", "items": { "type": "number" } } }
            }), &["id", "keys"]),
            mutates: true,
        },
        Method {
            name: "node_lock",
            summary: "Lock or unlock channels against keying (the red rows in the panel).",
            params: obj(json!({
                "id": id,
                "channels": { "type": "array", "items": { "type": "string" } },
                "locked": { "type": "boolean" }
            }), &["id", "channels", "locked"]),
            mutates: true,
        },
        // ---- Graphs inside nodes ----
        Method {
            name: "dive_enter",
            summary: "Open the graph INSIDE a node (a subwob, Noise, 2dini or SDF), as Enter on it does. Every node call then works on that graph (app_info.dive says which); the viewer keeps showing the comp with the live body in place.",
            params: obj(json!({ "id": id }), &["id"]),
            mutates: false,
        },
        Method {
            name: "dive_exit",
            summary: "Step out of the open inner graph, writing it back to its node. 'all' steps out to the comp.",
            params: obj(json!({ "all": { "type": "boolean" } }), &[]),
            mutates: false,
        },
        Method {
            name: "subwob_collapse",
            summary: "Collapse nodes into ONE subwob node defined in this comp (the picture must not change). Returns the new node's id; refused, with the reason, when the set cannot collapse (e.g. it holds a Load).",
            params: obj(json!({ "ids": { "type": "array", "items": { "type": "integer" } } }), &["ids"]),
            mutates: true,
        },
        // ---- Looking ----
        Method {
            name: "pixel_sample",
            summary: "SCENE-LINEAR values of the viewed node's output (working space, before the display transform), once the playhead's frame has rendered: mean, min and max RGBA over a rectangle in output pixels (origin top-left; default the whole frame; w=h=1 for one pixel). The numbers to judge exposure, a matte's coverage or a key by — viewer_capture is display-referred 8-bit.",
            params: obj(json!({
                "x": { "type": "integer" },
                "y": { "type": "integer" },
                "w": { "type": "integer" },
                "h": { "type": "integer" }
            }), &[]),
            mutates: false,
        },
        Method {
            name: "viewer_set",
            summary: "How the viewer SHOWS the picture (never what it is): exposure in stops, gamma, one channel as grey (rgb/r/g/b/a — 'a' to look at a matte), proxy (half-res) on or off. viewer_capture shows the result.",
            params: obj(json!({
                "exposure": { "type": "number" },
                "gamma": { "type": "number" },
                "channel": { "type": "string", "enum": ["rgb", "r", "g", "b", "a"] },
                "proxy": { "type": "boolean" }
            }), &[]),
            mutates: false,
        },
        Method {
            name: "perf_get",
            summary: "How playback is doing right now: frames shown per second against the target, the GPU's time per frame (upload, prepass stages, fused pass, blit), the CPU loop, the decode ring and the cache tiers' hits. The numbers the stats window shows.",
            params: obj(json!({}), &[]),
            mutates: false,
        },
        Method {
            name: "selection_set",
            summary: "Select nodes (their panels open in the properties pane).",
            params: obj(json!({ "ids": { "type": "array", "items": { "type": "integer" } } }), &["ids"]),
            mutates: false,
        },
        // ---- The comp itself ----
        Method {
            name: "comp_set",
            summary: "Comp settings: the script frame 'range' [first, last] (auto_range=true goes back to spanning the loaded plates), fps, and playing (start/stop playback).",
            params: obj(json!({
                "range": { "type": "array", "items": { "type": "integer" }, "minItems": 2, "maxItems": 2 },
                "auto_range": { "type": "boolean" },
                "fps": { "type": "number" },
                "playing": { "type": "boolean" }
            }), &[]),
            mutates: false,
        },
        // ---- Work that takes time ----
        Method {
            name: "node_run",
            summary: "Press a node's run button. Render (Write): render (its frames to its path). GraphBake: bake. Tracker: track. PlanarTrack: track | track_back. CorridorKey: key. Sammie: track. GenAI: quote (prices the run; PAYING is only ever a click in the app). Physics / Ragdoll (inside a 2dini: dive_enter first): bake | clear. Any of them: cancel. Returns at once — poll jobs_get until it is idle, then check the node's error.",
            params: obj(json!({
                "id": id,
                "action": { "type": "string", "enum": ["render", "bake", "clear", "track", "track_back", "key", "quote", "cancel"] }
            }), &["id"]),
            mutates: false,
        },
        Method {
            name: "jobs_get",
            summary: "What is running: renders and bakes (node, frame reached, first, last), Physics bakes (node, done, total), tracks, keys and generations; 'busy' is false when nothing is; 'status' is the app's status line and 'errors' maps node id to why it failed or cannot run.",
            params: obj(json!({}), &[]),
            mutates: false,
        },
        Method {
            name: "batch",
            summary: "Several calls as ONE undo step, in order: calls is [{method, params}, …]. A later call may use a node an earlier one made: the string \"$N\" anywhere in its params is replaced by the id call N returned (0-based). All or nothing: the first error undoes the whole batch and is returned with its index. Not for undo/redo, setup_load, dives or the waiting calls (viewer_capture, pixel_sample, node_run).",
            params: obj(json!({ "calls": { "type": "array", "items": { "type": "object" } } }), &["calls"]),
            mutates: true,
        },
    ]
}

/// Look one method up.
pub fn method(name: &str) -> Option<Method> {
    methods().into_iter().find(|m| m.name == name)
}

/// `api_schema`'s answer: the table as data.
pub fn schema() -> Value {
    json!({
        "version": VERSION,
        "methods": methods().iter().map(|m| json!({
            "name": m.name,
            "summary": m.summary,
            "params": m.params,
            "mutates": m.mutates,
            "read_only": m.read_only(),
            "destructive": m.destructive(),
        })).collect::<Vec<_>>(),
    })
}

/// An error a call returns. `code` follows JSON-RPC: -32601 unknown method,
/// -32602 bad params, -32000 the app refused or failed.
#[derive(Clone, Debug, PartialEq)]
pub struct ApiError {
    pub code: i64,
    pub message: String,
}

impl ApiError {
    pub fn invalid(msg: impl Into<String>) -> Self {
        Self { code: -32602, message: msg.into() }
    }
    pub fn app(msg: impl Into<String>) -> Self {
        Self { code: -32000, message: msg.into() }
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.message, self.code)
    }
}

impl std::error::Error for ApiError {}

/// Check params against a method's schema: an object, required keys
/// present, no unknown keys, and top-level primitive types right. Deeper
/// shapes are the handler's job — it has to parse them anyway.
pub fn validate(m: &Method, params: &Value) -> Result<(), ApiError> {
    let Some(p) = params.as_object() else {
        return Err(ApiError::invalid(format!("{}: params must be an object", m.name)));
    };
    let props = m.params["properties"].as_object().cloned().unwrap_or_default();
    for r in m.params["required"].as_array().into_iter().flatten() {
        let r = r.as_str().unwrap_or_default();
        if !p.contains_key(r) {
            return Err(ApiError::invalid(format!("{}: missing '{r}'", m.name)));
        }
    }
    for (k, v) in p {
        let Some(s) = props.get(k) else {
            let known: Vec<&str> = props.keys().map(|s| s.as_str()).collect();
            return Err(ApiError::invalid(format!(
                "{}: unknown param '{k}' (takes: {})",
                m.name,
                known.join(", ")
            )));
        };
        let ok = match s["type"].as_str() {
            Some("integer") => v.is_i64() || v.is_u64(),
            Some("number") => v.is_number(),
            Some("string") => v.is_string(),
            Some("boolean") => v.is_boolean(),
            Some("array") => v.is_array(),
            Some("object") => v.is_object(),
            _ => true,
        };
        if !ok {
            return Err(ApiError::invalid(format!(
                "{}: '{k}' must be {}",
                m.name,
                s["type"].as_str().unwrap_or("?")
            )));
        }
        if let (Some(opts), Some(v)) = (s["enum"].as_array(), v.as_str()) {
            if !opts.iter().any(|o| o.as_str() == Some(v)) {
                return Err(ApiError::invalid(format!("{}: '{k}' must be one of {}", m.name, s["enum"])));
            }
        }
    }
    Ok(())
}

/// Base64 for PNG payloads (`viewer_capture`), so the app need not carry
/// the dependency.
pub fn b64(bytes: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// The other way (`render_fetch`'s frames).
pub fn unb64(s: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.decode(s).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_unique_and_valid_mcp_tool_names() {
        let ms = methods();
        let mut seen = std::collections::HashSet::new();
        for m in &ms {
            assert!(seen.insert(m.name), "duplicate method {}", m.name);
            assert!(
                !m.name.is_empty()
                    && m.name.len() <= 64
                    && m.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
                "{} is not a valid MCP tool name",
                m.name
            );
            assert_eq!(m.params["type"], "object", "{}", m.name);
            for r in m.params["required"].as_array().unwrap() {
                assert!(m.params["properties"].get(r.as_str().unwrap()).is_some(), "{}: required {r} undeclared", m.name);
            }
        }
    }

    #[test]
    fn hint_lists_name_real_methods_and_never_a_read() {
        for n in ACTS.iter().chain(DESTRUCTIVE).chain(DEFERRED) {
            assert!(method(n).is_some(), "{n} is in a hint list but not in the table");
        }
        // What a client may auto-approve: nothing that writes, launches,
        // moves the view or edits.
        for n in ["setup_save", "blender_session", "node_run", "view_set", "node_set", "batch"] {
            assert!(!method(n).unwrap().read_only(), "{n} must not be read-only");
        }
        for n in ["graph_get", "node_get", "jobs_get", "pixel_sample", "viewer_capture", "patch_get"] {
            assert!(method(n).unwrap().read_only(), "{n} only reads");
        }
        assert!(method("setup_load").unwrap().destructive());
    }

    #[test]
    fn validate_catches_missing_unknown_and_mistyped() {
        let m = method("node_connect").unwrap();
        assert!(validate(&m, &json!({ "src": 1, "dst": 2 })).is_ok());
        assert!(validate(&m, &json!({ "src": 1 })).unwrap_err().message.contains("missing 'dst'"));
        assert!(validate(&m, &json!({ "src": 1, "dst": 2, "nope": 0 })).unwrap_err().message.contains("unknown param"));
        assert!(validate(&m, &json!({ "src": "1", "dst": 2 })).unwrap_err().message.contains("integer"));
        let v = method("view_set").unwrap();
        assert!(validate(&v, &json!({ "id": 1, "slot": "c" })).is_err());
        assert!(validate(&v, &json!([])).is_err());
    }
}
