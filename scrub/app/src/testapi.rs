//! `--test-api`: the scripting API end to end, through the real socket
//! while the app runs its UI passes — the calls a script or an agent makes:
//! open media, mark, edit (one batch, one undo step), dissolves, razor,
//! trims, slip / slide, speed / rate stretch, reframe, moves and copies,
//! delete, tracks and targets, a viewer capture (a real PNG), four-mark fit
//! to fill, a failed batch that leaves nothing behind, save / reopen, a
//! render to an MP4, and the MCP bridge listing and calling the same tools.

use crate::App;
use api::client::Client;
use serde_json::{json, Value};

impl App {
    pub(crate) fn test_api(&mut self) {
        let (Some(gpu), Some(win)) = (&self.gpu, &mut self.win) else { return };
        let ed = &mut self.editor;
        let port = match ed.api_listen(0) {
            Ok(p) => p,
            Err(e) => {
                println!("TEST FAIL {e}");
                return;
            }
        };
        let file = ed.first_media();
        let tmp = std::env::temp_dir().join(format!("scrub-api-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let t2 = tmp.clone();
        let script = std::thread::spawn(move || script(port, file, &t2));
        // The app keeps drawing (and so answering) until the script is done.
        let t0 = std::time::Instant::now();
        while !script.is_finished() && t0.elapsed().as_secs() < 180 {
            let _ = win.offscreen(gpu, ed);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let fails = script.join().unwrap_or_else(|_| vec!["the script panicked".into()]);
        let _ = std::fs::remove_dir_all(&tmp);
        if fails.is_empty() {
            println!("TEST PASS scripting API: media, marks, batch edit (one undo), dissolves, razor, trim, slip, slide, speed, stretch, reframe, move/copy, delete, tracks, capture PNG, fit to fill, failed batch rolls back, save/reopen, render, MCP bridge");
        } else {
            for f in &fails {
                println!("TEST FAIL {f}");
            }
        }
    }
}

fn script(port: u16, file: Option<std::path::PathBuf>, tmp: &std::path::Path) -> Vec<String> {
    let mut fails: Vec<String> = Vec::new();
    let mut c = match Client::connect(port) {
        Ok(c) => c,
        Err(e) => return vec![format!("connect: {e}")],
    };
    let mut call = |m: &str, p: Value| -> Result<Value, String> { c.call(m, p).map_err(|e| e.message) };
    let mut check = |what: String, ok: bool| {
        if !ok {
            fails.push(what);
        }
    };
    let Some(file) = file else { return vec!["give the suite a movie".into()] };
    let ok = |r: &Result<Value, String>| r.as_ref().map(|v| v.clone()).unwrap_or(Value::Null);
    let tl = |call: &mut dyn FnMut(&str, Value) -> Result<Value, String>| call("timeline_get", json!({})).unwrap_or(Value::Null);
    let clips = |t: &Value, track: &str| -> Vec<(i64, i64)> {
        t["tracks"].as_array().into_iter().flatten().filter(|x| x["name"] == track).flat_map(|x| x["clips"].as_array().cloned().unwrap_or_default()).map(|c| (c["start"].as_i64().unwrap_or(-1), c["len"].as_i64().unwrap_or(-1))).collect()
    };

    let r = call("project_new", json!({}));
    check(format!("project_new: {r:?}"), r.is_ok());
    let r = call("media_open", json!({ "path": file }));
    check(format!("media_open: {r:?}"), r.is_ok());
    let audio = ok(&r)["has_audio"].as_bool().unwrap_or(false);

    // Three 12-frame clips on V1 at 0, 12 and 24 in ONE batch: one undo step.
    let mut calls = Vec::new();
    for (k, src) in [10, 60, 110].iter().enumerate() {
        calls.push(json!({ "method": "marks_set", "params": { "side": "source", "in": src, "out": src + 11 } }));
        calls.push(json!({ "method": "playhead_set", "params": { "frame": k * 12 } }));
        calls.push(json!({ "method": "edit", "params": { "mode": "overwrite" } }));
    }
    let r = call("batch", json!({ "calls": calls }));
    check(format!("batch edit: {r:?}"), r.is_ok());
    let t = tl(&mut call);
    check(format!("three clips: {:?}", clips(&t, "V1")), clips(&t, "V1") == vec![(0, 12), (12, 12), (24, 12)]);
    check(format!("their sound: {:?}", clips(&t, "A1")), !audio || clips(&t, "A1").len() == 3);
    let u = call("undo", json!({}));
    check(format!("one undo takes the whole batch: {:?}", clips(&tl(&mut call), "V1")), u.is_ok() && clips(&tl(&mut call), "V1").is_empty());
    let _ = call("redo", json!({}));
    check("redo".into(), clips(&tl(&mut call), "V1").len() == 3);

    // Dissolves.
    let r = call("dissolve_add", json!({ "track": "V1", "at": 12, "length": 8 }));
    check(format!("dissolve_add: {r:?}"), ok(&r)["length"] == 8);
    let r = call("dissolve_set", json!({ "track": "V1", "at": 12, "before": 2, "after": 6 }));
    check(format!("dissolve_set: {r:?}"), ok(&r)["before"] == 2 && ok(&r)["after"] == 6);
    let r = call("dissolve_remove", json!({ "track": "V1", "at": 12 }));
    check(format!("dissolve_remove: {r:?}"), r.is_ok());

    // Razor, trim, slip, slide.
    let r = call("razor", json!({ "at": 6, "tracks": ["V1"] }));
    check(format!("razor: {r:?} {:?}", clips(&tl(&mut call), "V1")), r.is_ok() && clips(&tl(&mut call), "V1")[..2] == [(0, 6), (6, 6)]);
    let r = call("trim", json!({ "track": "V1", "at": 6, "side": "out", "frames": -2, "ripple": false }));
    check(format!("trim: {r:?}"), ok(&r)["moved"] == -2 && clips(&tl(&mut call), "V1")[0] == (0, 4));
    let r = call("slip", json!({ "track": "V1", "start": 12, "frames": 5 }));
    check(format!("slip: {r:?}"), ok(&r)["slipped"] == 5);
    let r = call("slide", json!({ "track": "V1", "start": 12, "frames": 2 }));
    check(format!("slide: {r:?} {:?}", clips(&tl(&mut call), "V1")), ok(&r)["slid"] == 2 && clips(&tl(&mut call), "V1").contains(&(14, 12)));

    // Speed, rate stretch, reframe.
    let r = call("clip_speed", json!({ "track": "V1", "start": 26, "speed": 2.0 }));
    check(format!("clip_speed: {r:?}"), r.is_ok());
    let r = call("clip_stretch", json!({ "track": "V1", "start": 26, "length": 20 }));
    check(format!("clip_stretch: {r:?}"), ok(&r)["length"] == 20 && ok(&r)["speed"].as_f64().is_some_and(|s| s < 1.0));
    let r = call("reframe_set", json!({ "track": "V1", "start": 14, "fill": true, "zoom": 1.2, "x": 0.1 }));
    check(format!("reframe_set: {r:?}"), ok(&r)["fill"] == true);

    // Move up a row (sound to A2), copy, delete.
    let r = call("clips_move", json!({ "clips": [{ "track": "V1", "start": 14 }], "frames": 100, "rows": 1 }));
    check(format!("clips_move: {r:?} {:?}", clips(&tl(&mut call), "V2")), r.is_ok() && clips(&tl(&mut call), "V2") == vec![(114, 12)] && (!audio || clips(&tl(&mut call), "A2") == vec![(114, 12)]));
    let r = call("clips_move", json!({ "clips": [{ "track": "V2", "start": 114 }], "frames": 20, "duplicate": true }));
    check(format!("duplicate: {r:?}"), r.is_ok() && clips(&tl(&mut call), "V2") == vec![(114, 12), (134, 12)]);
    let r = call("clip_delete", json!({ "track": "V2", "start": 134 }));
    check(format!("clip_delete: {r:?}"), r.is_ok() && clips(&tl(&mut call), "V2") == vec![(114, 12)]);

    // Tracks and targets.
    let r = call("track_set", json!({ "track": "A2", "mute": true, "gain_db": -3.0 }));
    check(format!("track_set: {r:?}"), ok(&r)["mute"] == true);
    let r = call("track_set", json!({ "track": "V1", "mute": true }));
    check("a picture track has no mute".into(), r.is_err());
    let r = call("targets_set", json!({ "video": "V3" }));
    check(format!("targets follow: {r:?}"), ok(&r)["video"] == "V3" && ok(&r)["audio"] == "A3");
    let r = call("targets_set", json!({ "video": "V1", "audio": "A1" }));
    check(format!("targets_set: {r:?}"), r.is_ok());

    // Selection, tools, transport.
    let r = call("select", json!({ "clips": [{ "track": "V1", "start": 0 }, { "track": "V1", "start": 6 }] }));
    check(format!("select: {r:?}"), ok(&r)["clips"].as_array().is_some_and(|v| v.len() == 2));
    let r = call("tool_set", json!({ "tool": "razor" }));
    check(format!("tool_set: {r:?}"), r.is_ok());
    let _ = call("tool_set", json!({ "tool": "select" }));
    let r = call("transport", json!({ "action": "step", "frames": 3 }));
    check(format!("transport: {r:?}"), r.is_ok());
    let r = call("playhead_set", json!({ "frame": 2000 }));
    check(format!("past the end: {r:?}"), ok(&r)["frame"] == 2000);

    // A capture is a real PNG.
    let r = call("viewer_capture", json!({ "frame": 2, "max_size": 320 }));
    let png = ok(&r)["png_base64"].as_str().unwrap_or_default().to_string();
    // EDIFY_API_SHOT=<file.png>: keep the capture to look at.
    if let Some(shot) = media::sys::var_os("API_SHOT") {
        use base64::Engine as _;
        let _ = std::fs::write(shot, base64::engine::general_purpose::STANDARD.decode(&png).unwrap_or_default());
    }
    check(format!("viewer_capture: {} bytes, {}x{}", png.len(), ok(&r)["width"], ok(&r)["height"]), png.starts_with("iVBORw0KGgo") && ok(&r)["width"].as_u64().is_some_and(|w| w <= 320));
    let r = call("viewer_capture", json!({ "viewer": "source", "max_size": 200 }));
    check(format!("source capture: {r:?}"), ok(&r)["png_base64"].as_str().is_some_and(|s| s.starts_with("iVBORw0KGgo")));

    // Four marks: fitted.
    let _ = call("marks_set", json!({ "side": "source", "in": 200, "out": 223 }));
    let _ = call("marks_set", json!({ "side": "record", "in": 300, "out": 311 }));
    let r = call("edit", json!({ "mode": "overwrite" }));
    check(format!("fit to fill: {r:?}"), ok(&r)["fitted"].as_str().is_some_and(|s| s.contains("2.000x")));

    // A batch that fails half way leaves nothing behind (and no undo step).
    let before = tl(&mut call);
    let depth = call("app_info", json!({})).map(|v| v["record"]["undo"].clone()).unwrap_or_default();
    let r = call("batch", json!({ "calls": [
        { "method": "razor", "params": { "at": 3, "tracks": ["V1"] } },
        { "method": "trim", "params": { "track": "V3", "at": 999, "side": "out", "frames": 1 } },
    ] }));
    let after = tl(&mut call);
    let depth2 = call("app_info", json!({})).map(|v| v["record"]["undo"].clone()).unwrap_or_default();
    check(format!("failed batch: {r:?}"), r.as_ref().is_err_and(|e| e.contains("call 1")) && before == after && depth == depth2);
    check("an unknown track is refused".into(), call("slip", json!({ "track": "V9", "start": 0, "frames": 1 })).is_err());
    check("a missing clip is refused in words".into(), call("slip", json!({ "track": "V1", "start": 5555, "frames": 1 })).is_err_and(|e| e.contains("timeline_get")));

    // Save, start over, reopen.
    let proj = tmp.join("api.scrub");
    let r = call("project_save", json!({ "path": proj }));
    check(format!("project_save: {r:?}"), r.is_ok() && proj.is_file());
    let _ = call("project_new", json!({}));
    let r = call("project_open", json!({ "path": proj }));
    check(format!("project_open: {r:?}"), r.is_ok() && tl(&mut call) == after);

    // Titles: a capture without and with a subtitle differs; a style edit
    // reaches every clip of it; SRT out and back in.
    let plain = call("viewer_capture", json!({ "frame": 5, "max_size": 320 })).map(|v| v["png_base64"].clone()).unwrap_or_default();
    let r = call("text_create", json!({ "text": "Subtitle one", "track": "V3", "at": 0, "length": 10, "style": "subtitle" }));
    check(format!("text_create: {r:?}"), ok(&r)["start"] == 0 && ok(&r)["track"] == "V3");
    let r = call("text_create", json!({ "text": "Subtitle two", "track": "V3", "at": 10, "length": 10, "style": "subtitle" }));
    check(format!("text_create 2: {r:?}"), ok(&r)["start"] == 10);
    let titled = call("viewer_capture", json!({ "frame": 5, "max_size": 320 })).map(|v| v["png_base64"].clone()).unwrap_or_default();
    check("the subtitle is in the capture".into(), titled != plain && titled.as_str().is_some_and(|s| s.len() > 100));
    let r = call("text_set", json!({ "track": "V3", "start": 10, "text": "Second" }));
    check(format!("text_set: {r:?}"), ok(&r)["text"] == "Second" && ok(&r)["style"] == "subtitle");
    let r = call("style_set", json!({ "style": "subtitle", "fields": { "size": 0.07, "color": [255, 220, 0, 255], "place": "lower third" } }));
    check(format!("style_set: {r:?}"), ok(&r)["size"].as_f64().is_some_and(|v| (v - 0.07).abs() < 1e-6) && ok(&r)["align"] == "Left");
    let r = call("style_list", json!({}));
    let sub = ok(&r)["styles"].as_array().and_then(|v| v.iter().find(|s| s["name"] == "subtitle").cloned()).unwrap_or_default();
    check(format!("style_list: {sub}"), sub["clips"] == 2 && sub["x"].as_f64().is_some_and(|x| (x - 0.1).abs() < 1e-6));
    check("an unknown style field is refused".into(), call("style_set", json!({ "style": "subtitle", "fields": { "glow": 1 } })).is_err());
    let r = call("style_set", json!({ "style": "caption", "create": true, "from": "subtitle", "fields": { "boxed": false } }));
    check(format!("style create: {r:?}"), ok(&r)["name"] == "caption");
    let srt = tmp.join("subs.srt");
    let r = call("srt_export", json!({ "path": srt, "track": "V3" }));
    check(format!("srt_export: {r:?}"), ok(&r)["exported"] == 2 && std::fs::read_to_string(&srt).is_ok_and(|t| t.contains("Subtitle one") && t.contains("Second")));
    let r = call("srt_import", json!({ "path": srt, "track": "V2", "style": "caption" }));
    let t = tl(&mut call);
    let v2_texts = t["tracks"][1]["clips"].as_array().map(|v| v.iter().filter(|c| c["text"]["style"] == "caption").count()).unwrap_or(0);
    check(format!("srt_import: {r:?} {v2_texts}"), ok(&r)["imported"] == 2 && v2_texts == 2);
    // Dissolve between the two subtitles.
    let r = call("dissolve_add", json!({ "track": "V3", "at": 10, "length": 6 }));
    check(format!("a dissolve between text clips: {r:?}"), ok(&r)["length"] == 6);

    // Render the first 24 frames.
    let _ = call("marks_set", json!({ "side": "record", "in": 0, "out": 23 }));
    let out = tmp.join("api.mp4");
    let r = call("render_start", json!({ "preset": "youtube-1080", "inout": true, "path": out }));
    check(format!("render_start: {r:?}"), r.is_ok());
    let t0 = std::time::Instant::now();
    let mut last = Value::Null;
    while t0.elapsed().as_secs() < 120 {
        last = call("render_status", json!({})).unwrap_or_default();
        if last["running"] == false {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    let made = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    check(format!("render: {last} ({made} bytes)"), last["last"].as_str().is_some_and(|s| s.contains("rendered 24 frames")) && made > 10_000);

    // The MCP bridge: the same table as tools, and a call through it.
    let mut b = api::mcp::Bridge::new(port);
    let tools = b.handle(&json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/list" })).unwrap_or_default();
    let n = tools["result"]["tools"].as_array().map_or(0, |t| t.len());
    check(format!("mcp tools: {n}"), n == api::methods().len());
    let r = b.handle(&json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": { "name": "app_info", "arguments": {} } })).unwrap_or_default();
    check(format!("mcp call: {r}"), r["result"]["content"][0]["text"].as_str().is_some_and(|s| s.contains("\"protocol\"")));
    let r = b.handle(&json!({ "jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": { "name": "viewer_capture", "arguments": { "max_size": 64 } } })).unwrap_or_default();
    check(format!("mcp image: {}", r["result"]["content"][0]["type"]), r["result"]["content"][0]["type"] == "image");
    fails
}
