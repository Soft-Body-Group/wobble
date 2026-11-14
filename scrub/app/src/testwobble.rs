//! `--test-wobble` (the `wobble` feature): a Wobble clip end to end with
//! the real Wobble, built with `--features app/movie` (a hidden worker;
//! `EDIFY_WOBBLE` or the usual places): the selected clip opened in Wobble
//! renders v001 straight from the original movie, which then plays (an EXR
//! render, the same picture as the source; nothing exported); a
//! "save in Wobble" (a worker adding a Mult x0.25 to the working copy, as a
//! person would in the app) becomes v002, rendered and playing (darker);
//! v001 can be picked again; and a render of the sequence includes it.
//! With `EDIFY_WOBBLE_STOCK` naming a Wobble WITHOUT the movie feature, a
//! clip opened with it is refused with the build command.

use crate::{editor, App};
use api::client::Client;
use serde_json::json;
use std::time::{Duration, Instant};

fn mean(png: &[u8]) -> f64 {
    let dec = png::Decoder::new(std::io::Cursor::new(png));
    let Ok(mut r) = dec.read_info() else { return -1.0 };
    let mut buf = vec![0; r.output_buffer_size().unwrap_or(0)];
    let Ok(info) = r.next_frame(&mut buf) else { return -1.0 };
    let px = &buf[..info.buffer_size()];
    px.iter().map(|&v| v as f64).sum::<f64>() / px.len().max(1) as f64
}

impl App {
    pub(crate) fn test_wobble(&mut self) {
        let (Some(gpu), Some(win)) = (&self.gpu, &mut self.win) else { return };
        let ed = &mut self.editor;
        let mut fails: Vec<String> = Vec::new();
        if crate::wobbleclip::wobble_bin().is_none() {
            println!("TEST SKIP no Wobble binary (EDIFY_WOBBLE_BIN or ~/src/devl/wobble/target/release/wobble)");
            return;
        }
        ed.build_cuts(4, false);
        ed.focus_record();
        ed.wobble_test_setup();
        for _ in 0..3 {
            let _ = win.offscreen(gpu, ed);
        }
        let wait = |win: &mut imfram::shell::UiWindow, ed: &mut editor::Editor, what: &dyn Fn(&editor::Editor) -> bool, secs: u64| -> bool {
            let t0 = Instant::now();
            while t0.elapsed() < Duration::from_secs(secs) {
                let _ = win.offscreen(gpu, ed);
                if what(ed) {
                    return true;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            false
        };
        let src_mean = mean(&ed.capture_record(3));
        // 1. Open the V1 clip at 0 in Wobble: v001 made and playing.
        ed.select_clip_at(model::V1, 0);
        ed.wobble_open();
        let ok = wait(win, ed, &|e| e.wobble_state(model::V1, 0).is_some_and(|s| s.1 == vec![1] && !s.2), 240);
        let st = ed.wobble_state(model::V1, 0);
        if !ok {
            fails.push(format!("v001 did not render: {st:?} {}", ed.last_notice()));
        }
        let v1_mean = mean(&ed.capture_record(3));
        if (v1_mean - src_mean).abs() > 6.0 {
            fails.push(format!("v001 should look like the source: {v1_mean:.1} vs {src_mean:.1}"));
        }
        // 2. A save in Wobble: the working copy gets a Mult x0.25.
        ed.wobble_open(); // the session (no app in a test) watching work.wbl
        let _ = win.offscreen(gpu, ed);
        let work = ed.wobble_root().join(st.as_ref().map(|s| s.0.clone()).unwrap_or_default()).join("work.wbl");
        let edit = std::thread::spawn(move || -> Result<(), String> {
            let bin = crate::wobbleclip::wobble_bin().ok_or("no wobble")?;
            let port = std::net::TcpListener::bind(("127.0.0.1", 0)).map_err(|e| e.to_string())?.local_addr().map_err(|e| e.to_string())?.port();
            let mut child = std::process::Command::new(bin).env("WOBBLE_WORKER", "1").arg("--worker").arg(port.to_string()).arg(&work).spawn().map_err(|e| e.to_string())?;
            let r = (|| -> Result<(), String> {
                let t0 = Instant::now();
                let mut c = loop {
                    match Client::connect(port) {
                        Ok(c) => break c,
                        Err(e) if t0.elapsed() > Duration::from_secs(60) => return Err(e.to_string()),
                        Err(_) => std::thread::sleep(Duration::from_millis(200)),
                    }
                };
                let mut call = |m: &str, p: serde_json::Value| c.call(m, p).map_err(|e| e.message);
                let mut nodes = serde_json::Value::Null;
                for _ in 0..100 {
                    nodes = call("graph_get", json!({}))?;
                    if nodes["nodes"].as_array().is_some_and(|n| n.len() >= 2) {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(200));
                }
                let id = |name: &str| nodes["nodes"].as_array().into_iter().flatten().find(|n| n["name"] == name).and_then(|n| n["id"].as_u64());
                let (l, o) = (id("SCRUB SOURCE").ok_or("no source node")?, id("SCRUB OUT").ok_or("no out node")?);
                let m = call("node_create", json!({ "type": "Mult", "inputs": [l] }))?["id"].as_u64().ok_or("no mult")?;
                call("node_connect", json!({ "src": m, "dst": o, "input": 0 }))?;
                call("node_set", json!({ "id": m, "values": { "MultR": 0.25, "MultG": 0.25, "MultB": 0.25 } }))?;
                call("setup_save", json!({ "path": work }))?;
                Ok(())
            })();
            let _ = child.kill();
            r
        });
        let t0 = Instant::now();
        while !edit.is_finished() && t0.elapsed() < Duration::from_secs(120) {
            let _ = win.offscreen(gpu, ed);
            std::thread::sleep(Duration::from_millis(20));
        }
        if let Ok(Err(e)) = edit.join() {
            fails.push(format!("editing in Wobble: {e}"));
        }
        let ok = wait(win, ed, &|e| e.wobble_state(model::V1, 0).is_some_and(|s| s.1.contains(&2) && s.3 == 2 && !s.2), 240);
        if !ok {
            fails.push(format!("v002 did not become active: {:?} {}", ed.wobble_state(model::V1, 0), ed.last_notice()));
        }
        let v2_mean = mean(&ed.capture_record(3));
        if !(v2_mean < v1_mean * 0.8) {
            fails.push(format!("v002 (x0.25) should be darker: {v2_mean:.1} vs {v1_mean:.1}"));
        }
        // 3. Back to v001.
        ed.wobble_pick(model::V1, 0, 1);
        let back = mean(&ed.capture_record(3));
        if (back - v1_mean).abs() > 1.0 {
            fails.push(format!("picking v001 again: {back:.1} vs {v1_mean:.1}"));
        }
        // 4. A render of the sequence includes the Wobble clip.
        let out = std::env::temp_dir().join(format!("scrub-wobble-{}.mp4", std::process::id()));
        match ed.render_now(&out) {
            Ok(n) if n > 0 => {}
            r => fails.push(format!("render with a Wobble clip: {r:?}")),
        }
        let _ = std::fs::remove_file(&out);
        // Nothing of the source was copied into the comp's folder.
        let uid = ed.wobble_state(model::V1, 0).map(|s| s.0).unwrap_or_default();
        let copied = std::fs::read_dir(ed.wobble_root().join(&uid)).into_iter().flatten().flatten().any(|e| e.file_name() == "source");
        if copied {
            fails.push("a source folder was exported".into());
        }
        // 5. A Wobble that cannot read movies: refused, with the command.
        if let Some(stock) = media::sys::var_os("WOBBLE_STOCK") {
            let keep = media::sys::var_os("WOBBLE");
            std::env::set_var("SCRUB_WOBBLE", &stock);
            ed.select_clip_at(model::V2, 12);
            ed.wobble_open();
            let refused = wait(win, ed, &|e| e.last_notice().contains("features app/movie"), 120);
            if !refused {
                fails.push(format!("a stock Wobble should be refused with the build command: {}", ed.last_notice()));
            }
            match keep {
                Some(k) => std::env::set_var("SCRUB_WOBBLE", k),
                None => std::env::remove_var("SCRUB_WOBBLE"),
            }
        }
        let _ = std::fs::remove_dir_all(ed.wobble_root());
        if fails.is_empty() {
            println!("TEST PASS Wobble clips: v001 rendered by a Wobble worker straight from the movie (nothing exported) and playing, a save in Wobble became v002 (darker), versions switch, the sequence renders");
        } else {
            for f in &fails {
                println!("TEST FAIL {f}");
            }
        }
    }
}
