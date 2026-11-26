//! Scrub's scripting protocol, the same shape as wobble's (its `api`
//! crate): newline-delimited JSON-RPC 2.0 over a LOCAL TCP socket,
//! described by the method table in [`methods`]. The app serves it; every
//! client — the MCP bridge (`scrub-mcp`), the app's own `--test-api`, a
//! script — is a thin wrapper that learns the methods from `api_schema`
//! rather than hard-coding them, so no client is privileged and none can
//! drift.
//!
//! Rules the protocol carries, whoever the caller is:
//! - Every edit goes through the same edit-model operations the keys and
//!   the mouse use, and one mutating call is ONE undo step (a `batch` too).
//! - Nothing here runs per frame. A call is a command the app drains at the
//!   top of its UI pass, between frames.
//! - Localhost only. The app listens on [`DEFAULT_PORT`] (or
//!   `EDIFY_API_PORT`) unless started with `--no-api`.
//!
//! Tracks are named as the timeline names them: "V1" "V2" "V3" "A1" "A2"
//! "A3" "M". A clip is named by its track and the record frame it starts on
//! (`timeline_get` lists them); frames are record frames at the sequence
//! rate unless a param says "source".

pub mod client;
pub mod mcp;
pub mod server;

use serde_json::{json, Value};

/// The port the app listens on and clients dial when nothing says otherwise
/// (wobble's is 7979). `EDIFY_API_PORT` overrides it on both sides.
pub const DEFAULT_PORT: u16 = 7980;

/// Protocol revision. Bumped on a breaking change to a method's shape;
/// adding a method or an optional param is not breaking.
pub const VERSION: u32 = 1;

/// The port from `EDIFY_API_PORT`, else [`DEFAULT_PORT`].
pub fn port_from_env() -> u16 {
    std::env::var("SCRUB_API_PORT").or_else(|_| std::env::var("EDIFY_API_PORT")).ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_PORT)
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
    /// Changes the sequence (one undo step), as opposed to reading it or
    /// moving a playhead.
    pub mutates: bool,
}

/// Methods that leave the SEQUENCE alone but still act: they move a
/// playhead, set marks or targets, load media, open or save a project, or
/// start a render. Not `mutates` (no undo step), and not read-only either —
/// a client that auto-approves read-only calls must not auto-approve these.
pub const ACTS: &[&str] = &[
    "project_open",
    "project_new",
    "project_save",
    "post_open",
    "media_open",
    "bin_add",
    "marks_set",
    "playhead_set",
    "transport",
    "targets_set",
    "select",
    "tool_set",
    "render_start",
    "render_cancel",
    "srt_export",
    "mode_set",
    "clip_rate",
    "subclip_add",
    "subclip_remove",
];

/// Methods that can throw work away: a client should confirm these.
pub const DESTRUCTIVE: &[&str] = &["project_open", "project_new", "post_open", "clip_delete"];

/// Methods answered later (on a worker), so they cannot ride inside a
/// `batch`.
pub const DEFERRED: &[&str] = &["viewer_capture"];

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
    let track = json!({ "type": "string", "enum": ["V1", "V2", "V3", "A1", "A2", "A3", "M"] });
    let start = json!({ "type": "integer", "description": "the record frame the clip starts on (timeline_get)" });
    let at = json!({ "type": "integer", "description": "the cut: the record frame where one clip ends and the next begins" });
    let clip_list = json!({ "type": "array", "items": { "type": "object", "properties": { "track": track, "start": start }, "required": ["track", "start"] }, "description": "clips as {track, start}" });
    let m = |name, summary, params, mutates| Method { name, summary, params, mutates };
    vec![
        m("api_schema", "This table: every method, its params as JSON Schema, and whether it edits the sequence.", obj(json!({}), &[]), false),
        m(
            "app_info",
            "Protocol version; the project (path, name, unsaved changes), the organisicator post if any; the sequence's fps, raster and duration; the record playhead, marks, targets, tool, trim mode, snapping and selection; the source clip (path, frames, fps, playhead, marks); whether anything plays; a render running.",
            obj(json!({}), &[]),
            false,
        ),
        m(
            "timeline_get",
            "The whole sequence as JSON: fps, raster, duration, the media list, and every track (name, kind, mute, solo, gain_db) with its clips (start, len, end, src_in, speed, media index and name, link, reframe, handles before/after in record frames) and dissolves (at, before, after).",
            obj(json!({}), &[]),
            false,
        ),
        m("bin_list", "The bin: every item's path, name, detail and group (bins by shoot day / post).", obj(json!({}), &[]), false),
        m(
            "bin_add",
            "Add media to the bin: a file (movie or image sequence frame) or a folder (scanned for media, image sequences collapsed). 'group' names the bin it lands in.",
            obj(json!({ "path": { "type": "string" }, "group": { "type": "string" } }), &["path"]),
            false,
        ),
        m(
            "media_open",
            "Load a file into the source viewer (a movie, or any frame of an image sequence). It becomes what insert / overwrite edit from; its marks start clear.",
            obj(json!({ "path": { "type": "string" } }), &["path"]),
            false,
        ),
        m("project_open", "Open a .scrub (or older .edify) project in place of the current one (save that first with project_save: its unsaved changes are dropped).", obj(json!({ "path": { "type": "string" } }), &["path"]), false),
        m(
            "project_new",
            "Start a new, empty project in place of the current one (unsaved changes are dropped). The first edit sets the rate and raster from its clip unless sequence_set chose them.",
            obj(json!({}), &[]),
            false,
        ),
        m(
            "project_save",
            "Save the project: to 'path' (a .scrub file), or where it already lives. A project that was never saved needs a path.",
            obj(json!({ "path": { "type": "string" } }), &[]),
            false,
        ),
        m(
            "post_open",
            "Open (or start) the edit of an organisicator marketing post, as organisicator launches it: 'post' is <marketing>/posts/<slug>; each 'rushes' folder fills the bins.",
            obj(json!({ "post": { "type": "string" }, "rushes": { "type": "array", "items": { "type": "string" } } }), &["post"]),
            false,
        ),
        m(
            "sequence_set",
            "Sequence settings: the raster [width, height] (any time), the frame rate as [num, den] (only while the sequence is empty), and conform (new edits of clips at another rate play one source frame per sequence frame).",
            obj(json!({ "raster": { "type": "array", "items": { "type": "integer" } }, "fps": { "type": "array", "items": { "type": "integer" } }, "conform": { "type": "boolean" } }), &[]),
            true,
        ),
        m(
            "marks_set",
            "Set in / out marks on the 'source' or 'record' side. 'in' and 'out' are frames (out is the LAST frame included); null clears one; clear=true clears both first.",
            obj(
                json!({
                    "side": { "type": "string", "enum": ["source", "record"] },
                    "in": { "type": ["integer", "null"] },
                    "out": { "type": ["integer", "null"] },
                    "clear": { "type": "boolean" }
                }),
                &["side"],
            ),
            false,
        ),
        m(
            "playhead_set",
            "Move a playhead ('record', the default, or 'source') to a frame. The record playhead may go past the last clip (an edit there leaves a gap).",
            obj(json!({ "frame": { "type": "integer" }, "side": { "type": "string", "enum": ["source", "record"] } }), &["frame"]),
            false,
        ),
        m(
            "transport",
            "Drive a viewer's transport: play, stop, toggle, shuttle_forward / shuttle_back (J / L), step by 'frames', start, end.",
            obj(
                json!({
                    "action": { "type": "string", "enum": ["play", "stop", "toggle", "shuttle_forward", "shuttle_back", "step", "start", "end"] },
                    "side": { "type": "string", "enum": ["source", "record"] },
                    "frames": { "type": "integer", "description": "step only (negative: back); default 1" }
                }),
                &["action"],
            ),
            false,
        ),
        m(
            "targets_set",
            "Which tracks edits land on: 'video' (V1-V3, or null for none) and 'audio' (A1-A3, M, or null). Setting only video moves a paired sound target with it (V2 -> A2), as the keys do.",
            obj(json!({ "video": { "type": ["string", "null"] }, "audio": { "type": ["string", "null"] } }), &[]),
            false,
        ),
        m(
            "track_set",
            "A track's mute, solo and gain (dB; audio tracks).",
            obj(json!({ "track": track, "mute": { "type": "boolean" }, "solo": { "type": "boolean" }, "gain_db": { "type": "number" } }), &["track"]),
            true,
        ),
        m(
            "edit",
            "Insert or overwrite the source into the record side on the targeted tracks, by the three-point rules (source in/out or playhead, record in/out or playhead). With all four marks set the source is FITTED to the record range (its speed changes; 'fitted' says so). The record playhead lands after the edit and the record marks are used up, as in the app.",
            obj(json!({ "mode": { "type": "string", "enum": ["insert", "overwrite"] } }), &["mode"]),
            true,
        ),
        m(
            "fit_to_fill",
            "Fit to fill: the source in..out stretched over the record in..out (all four marks), overwritten on the targeted tracks.",
            obj(json!({}), &[]),
            true,
        ),
        m(
            "select",
            "Set the record selection: 'clips' (a list of {track, start}), or an 'edit' point {track, at, side: out|in|roll}, or nothing (both left out).",
            obj(
                json!({
                    "clips": clip_list,
                    "edit": { "type": "object", "properties": { "track": track, "at": at, "side": { "type": "string", "enum": ["out", "in", "roll"] } }, "required": ["track", "at"] }
                }),
                &[],
            ),
            false,
        ),
        m(
            "tool_set",
            "The timeline tool the mouse uses in the app: select, razor, roll, rate, slip, slide.",
            obj(json!({ "tool": { "type": "string", "enum": ["select", "razor", "roll", "rate", "slip", "slide"] } }), &["tool"]),
            false,
        ),
        m(
            "trim",
            "Trim the cut at 'at' on a track by 'frames' (positive: later): side 'out' moves the outgoing clip's end, 'in' the incoming clip's start, 'roll' both. ripple (default: the app's trim mode) moves what follows on every track. Clamped to handles and neighbours; returns the frames actually moved.",
            obj(
                json!({
                    "track": track, "at": at,
                    "side": { "type": "string", "enum": ["out", "in", "roll"] },
                    "frames": { "type": "integer" },
                    "ripple": { "type": "boolean" }
                }),
                &["track", "at", "side", "frames"],
            ),
            true,
        ),
        m(
            "slip",
            "Slip a clip (with its linked sound): it shows 'frames' SOURCE frames later (negative: earlier) in the same place. Clamped to the media; returns the frames slipped.",
            obj(json!({ "track": track, "start": start, "frames": { "type": "integer" } }), &["track", "start", "frames"]),
            true,
        ),
        m(
            "slide",
            "Slide a clip 'frames' along the timeline showing the same source; the clips either side (or the gap) give and take. Returns the frames slid.",
            obj(json!({ "track": track, "start": start, "frames": { "type": "integer" } }), &["track", "start", "frames"]),
            true,
        ),
        m(
            "razor",
            "Cut at record frame 'at': on 'tracks' (each clip with its linked sound), or on every track (all=true, or no tracks given).",
            obj(json!({ "at": { "type": "integer" }, "tracks": { "type": "array", "items": track }, "all": { "type": "boolean" } }), &["at"]),
            true,
        ),
        m(
            "clips_move",
            "Move clips 'frames' along and 'rows' up (+: V1 toward V3; their sound follows to the paired track) as dragging them does. Each clip takes its linked sound unless alone=true. insert=true opens a gap where they land (every track) instead of overwriting; duplicate=true moves copies and leaves the originals. Returns the frames and rows applied.",
            obj(
                json!({
                    "clips": clip_list,
                    "frames": { "type": "integer" },
                    "rows": { "type": "integer" },
                    "insert": { "type": "boolean" },
                    "duplicate": { "type": "boolean" },
                    "alone": { "type": "boolean" }
                }),
                &["clips"],
            ),
            true,
        ),
        m(
            "clip_delete",
            "Delete a clip with its linked sound: leaving a gap (lift), or ripple=true closing it (refused when another track has material under it).",
            obj(json!({ "track": track, "start": start, "ripple": { "type": "boolean" } }), &["track", "start"]),
            true,
        ),
        m(
            "clip_speed",
            "Set a clip's speed as a multiplier (2 = twice as fast, 0.5 half, negative backwards). Its source stays and its length follows (never growing into the next clip). Returns the new length.",
            obj(json!({ "track": track, "start": start, "speed": { "type": "number" } }), &["track", "start", "speed"]),
            true,
        ),
        m(
            "clip_conform",
            "Conform a clip (on=true, the default) or put it back to real time (on=false). Conformed, a clip at another frame rate plays one source frame per sequence frame: no drops or repeats; its source stays, its length follows (never growing into the next clip), its sound is muted. Returns the new length.",
            obj(json!({ "track": track, "start": start, "on": { "type": "boolean" } }), &["track", "start"]),
            true,
        ),
        m(
            "clip_stretch",
            "Rate stretch: give a clip a new 'length' in record frames and its speed changes so the same source fills it. side 'out' (default) keeps the start, 'in' keeps the end. Never overlaps a neighbour. Returns the length and speed.",
            obj(json!({ "track": track, "start": start, "length": { "type": "integer" }, "side": { "type": "string", "enum": ["out", "in"] } }), &["track", "start", "length"]),
            true,
        ),
        m(
            "dissolve_add",
            "A dissolve on the cut at 'at' on a track ('length' frames, default 24, centred), clamped to the handles both clips have.",
            obj(json!({ "track": track, "at": at, "length": { "type": "integer" } }), &["track", "at"]),
            true,
        ),
        m(
            "dissolve_set",
            "Resize a dissolve: frames 'before' and 'after' its cut, each clamped to the handle there is.",
            obj(json!({ "track": track, "at": at, "before": { "type": "integer" }, "after": { "type": "integer" } }), &["track", "at", "before", "after"]),
            true,
        ),
        m("dissolve_remove", "Remove the dissolve on a cut.", obj(json!({ "track": track, "at": at }), &["track", "at"]), true),
        m(
            "reframe_set",
            "How a clip's picture sits in the raster: fill (cover, crop the overflow) or fit, zoom (1 = as fitted/filled), x / y (centre offset as a fraction of the raster's width / height). reset=true goes back to fit, 1, 0, 0 first. Applies to the clip's linked group.",
            obj(
                json!({
                    "track": track, "start": start,
                    "fill": { "type": "boolean" }, "zoom": { "type": "number" },
                    "x": { "type": "number" }, "y": { "type": "number" },
                    "reset": { "type": "boolean" }
                }),
                &["track", "start"],
            ),
            true,
        ),
        m(
            "viewer_capture",
            "A PNG of a viewer's picture ('record', the default, or 'source') at its playhead or at 'frame', composed as a render composes it (reframes, dissolves, speed). Returned as base64 ('png_base64').",
            obj(
                json!({
                    "viewer": { "type": "string", "enum": ["record", "source"] },
                    "frame": { "type": "integer" },
                    "max_size": { "type": "integer", "description": "longest side in pixels, default 768" }
                }),
                &[],
            ),
            false,
        ),
        m(
            "text_create",
            "A text clip (title / subtitle): 'text' at record frame 'at' (default: the playhead) for 'length' frames (default 3 s) on a video track (default: the free track above the target), in 'style' (a name from style_list; default the last used). Returns its track and start.",
            obj(json!({ "text": { "type": "string" }, "track": track, "at": { "type": "integer" }, "length": { "type": "integer" }, "style": { "type": "string" } }), &["text"]),
            true,
        ),
        m(
            "text_set",
            "Change a text clip's words and / or its style (by name). Its style keeps its look and position: retyping never moves it.",
            obj(json!({ "track": track, "start": start, "text": { "type": "string" }, "style": { "type": "string" } }), &["track", "start"]),
            true,
        ),
        m("style_list", "The text styles: each one's fields (font, size, colour, alignment, line spacing, outline, shadow, box, position) and how many clips use it.", obj(json!({}), &[]), false),
        m(
            "style_set",
            "Edit a text style by name — every clip using it changes — or make it (create=true, copying 'from' if given). 'fields' takes any of: name, font (a font file path, or null for the built-in face), font_index, size (fraction of the frame height), color / outline_color / shadow_color / box_color ([r,g,b,a] 0-255), align (Left/Center/Right), line_spacing, outline (fraction of size), shadow, boxed, box_pad, x / y (fractions of the frame), valign (Top/Middle/Bottom), place (centred, title top, lower third, subtitle, top left).",
            obj(json!({ "style": { "type": "string" }, "fields": { "type": "object" }, "create": { "type": "boolean" }, "from": { "type": "string" } }), &["style"]),
            true,
        ),
        m(
            "srt_import",
            "SubRip subtitles from a file as text clips on a video track (default: the target, else V2) in a style (default 'subtitle').",
            obj(json!({ "path": { "type": "string" }, "track": track, "style": { "type": "string" } }), &["path"]),
            true,
        ),
        m(
            "srt_export",
            "Write a video track's text clips (default: the track with the most) as a SubRip file.",
            obj(json!({ "path": { "type": "string" }, "track": track }), &["path"]),
            false,
        ),
        m("render_presets", "The render presets (id, name, size, bitrate) and which matches the sequence's shape.", obj(json!({}), &[]), false),
        m(
            "render_start",
            "Render the sequence to an H.264 + AAC MP4 in the background: a preset id (render_presets), fit (bars) or fill (crop), the whole sequence or in..out. Inside an organisicator post it goes to the post's out/ (path ignored); otherwise 'path' (a .mp4; never overwritten: a free name is picked). Progress in render_status.",
            obj(
                json!({
                    "preset": { "type": "string" },
                    "fill": { "type": "boolean" },
                    "inout": { "type": "boolean" },
                    "path": { "type": "string" }
                }),
                &["preset"],
            ),
            false,
        ),
        m("render_status", "The render running (file, frames done of total, fps) or the last one's result.", obj(json!({}), &[]), false),
        m("render_cancel", "Stop the running render (its partial file is removed).", obj(json!({}), &[]), false),
        // Selects mode: verdicts on whole clips, subclips in the selects bin.
        m("mode_set", "Switch the window between 'edit' (bin, two-up, timeline) and 'selects' (one viewer, the clips with their verdicts, the selects bin of subclips).", obj(json!({ "mode": { "type": "string", "enum": ["edit", "selects"] } }), &["mode"]), false),
        m(
            "selects_list",
            "The bin's whole clips with their selects-mode verdict ('select', 'reject' or null), remembered source marks and subclip count, filtered by 'filter' (default all); with the counts per filter.",
            obj(json!({ "filter": { "type": "string", "enum": ["all", "selects", "unmarked", "rejects"] } }), &[]),
            false,
        ),
        m(
            "clip_rate",
            "Set a whole clip's verdict: 'select', 'reject' or 'none'. 'path' names the clip (a bin item; default: the source clip).",
            obj(json!({ "path": { "type": "string" }, "rating": { "type": "string", "enum": ["select", "reject", "none"] } }), &["rating"]),
            false,
        ),
        m(
            "subclip_add",
            "Put a subclip into the selects bin: source frames 'in'..'out' (both included, 0-based source frame indices as marks_set side 'source' counts them) of 'path' (default: the source clip; a path is loaded into the source). Missing in / out: the source marks, else the clip's first / last frame. A clip with no verdict becomes a select. Repeatable per clip; the very same range twice is refused.",
            obj(json!({ "path": { "type": "string" }, "in": { "type": "integer" }, "out": { "type": "integer" } }), &[]),
            false,
        ),
        m("subclip_list", "The selects bin, in order: each subclip's path, name, in, out (source frames, both included) and length.", obj(json!({}), &[]), false),
        m(
            "subclip_remove",
            "Take a subclip out of the selects bin, named by its path, in and out (subclip_list).",
            obj(json!({ "path": { "type": "string" }, "in": { "type": "integer" }, "out": { "type": "integer" } }), &["path", "in", "out"]),
            false,
        ),
        m(
            "subclips_place",
            "Lay subclips onto the timeline one after another, each with its source in..out and its linked sound: 'subclips' as [{path, in, out}] (default: the whole selects bin, in order), on 'track' (a video track takes picture + sound, default V1) from record frame 'at' (default: the end of the sequence, an append); 'insert' pushes what follows along instead of overwriting. One undo step.",
            obj(
                json!({
                    "subclips": { "type": "array", "items": { "type": "object", "properties": { "path": { "type": "string" }, "in": { "type": "integer" }, "out": { "type": "integer" } }, "required": ["path", "in", "out"] } },
                    "track": track,
                    "at": { "type": "integer" },
                    "insert": { "type": "boolean" }
                }),
                &[],
            ),
            true,
        ),
        m("undo", "Undo the last sequence change (one call, one batch, one drag or one key edit).", obj(json!({}), &[]), true),
        m("redo", "Redo what undo undid.", obj(json!({}), &[]), true),
        m(
            "batch",
            "Several calls as ONE undo step, in order: calls is [{method, params}, …]. All or nothing: the first error puts the sequence back as it was and is returned with its index. Not for undo/redo, project_open/new, post_open or viewer_capture.",
            obj(
                json!({ "calls": { "type": "array", "items": { "type": "object", "properties": { "method": { "type": "string" }, "params": { "type": "object" } }, "required": ["method"] } } }),
                &["calls"],
            ),
            true,
        ),
    ]
}

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
/// present, no unknown keys, and top-level primitive types right (a type
/// list like ["integer", "null"] takes either). Deeper shapes are the
/// handler's job — it has to parse them anyway.
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
            return Err(ApiError::invalid(format!("{}: unknown param '{k}' (takes: {})", m.name, known.join(", "))));
        };
        let types: Vec<&str> = match &s["type"] {
            Value::String(t) => vec![t.as_str()],
            Value::Array(ts) => ts.iter().filter_map(|t| t.as_str()).collect(),
            _ => vec![],
        };
        let fits = |t: &str| match t {
            "integer" => v.is_i64() || v.is_u64(),
            "number" => v.is_number(),
            "string" => v.is_string(),
            "boolean" => v.is_boolean(),
            "array" => v.is_array(),
            "object" => v.is_object(),
            "null" => v.is_null(),
            _ => true,
        };
        if !types.is_empty() && !types.iter().any(|t| fits(t)) {
            return Err(ApiError::invalid(format!("{}: '{k}' must be {}", m.name, types.join(" or "))));
        }
        if let (Some(e), Some(s)) = (s["enum"].as_array(), v.as_str()) {
            if !e.iter().any(|x| x.as_str() == Some(s)) {
                let opts: Vec<&str> = e.iter().filter_map(|x| x.as_str()).collect();
                return Err(ApiError::invalid(format!("{}: '{k}' must be one of {}", m.name, opts.join(", "))));
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
                !m.name.is_empty() && m.name.len() <= 64 && m.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
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
        // What a client may auto-approve: nothing that writes, opens,
        // renders, moves a playhead or edits.
        for n in ["project_save", "render_start", "playhead_set", "edit", "batch", "media_open", "undo"] {
            assert!(!method(n).unwrap().read_only(), "{n} must not be read-only");
        }
        for n in ["app_info", "timeline_get", "bin_list", "viewer_capture", "render_status", "render_presets", "api_schema", "selects_list", "subclip_list"] {
            assert!(method(n).unwrap().read_only(), "{n} only reads");
        }
        assert!(method("project_open").unwrap().destructive());
    }

    #[test]
    fn validate_catches_missing_unknown_mistyped_and_off_list() {
        let m = method("trim").unwrap();
        assert!(validate(&m, &json!({ "track": "V1", "at": 12, "side": "out", "frames": -2 })).is_ok());
        assert!(validate(&m, &json!({ "track": "V1", "at": 12, "side": "out" })).unwrap_err().message.contains("missing 'frames'"));
        assert!(validate(&m, &json!({ "track": "V1", "at": 12, "side": "out", "frames": 1, "nope": 0 })).unwrap_err().message.contains("unknown param"));
        assert!(validate(&m, &json!({ "track": "V1", "at": "12", "side": "out", "frames": 1 })).unwrap_err().message.contains("integer"));
        assert!(validate(&m, &json!({ "track": "V9", "at": 12, "side": "out", "frames": 1 })).unwrap_err().message.contains("one of"));
        let mk = method("marks_set").unwrap();
        assert!(validate(&mk, &json!({ "side": "record", "in": null, "out": 20 })).is_ok());
        assert!(validate(&mk, &json!([])).is_err());
    }
}
