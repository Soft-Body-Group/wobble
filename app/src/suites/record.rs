//! `--test-record`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-dual: left pane (1) views the plate, right pane (3) a Grade of
    /// it at gain 0.5. With the LEFT pane live, the right one is the second
    /// compositor's: it must read half the plate — and after the Grade is
    /// set to 0.25 (left still live) a quarter. A held still would not move.
    /// `--test-record`: ⌘⇧R through the real keymap starts a take of the
    /// window, a pointer circles the viewer while the comp plays, ⌘⇧R
    /// stops it, and the whole take saves to an MP4 (the OS encoder) with
    /// a GIF beside it. Needs a window that draws: run it with
    /// `WOBBLE_TEST_FOCUS=1` (an accessory window behind others produces
    /// no frames). `WOBBLE_RECORD_DIR` picks where the clip lands.
    pub(crate) fn drive_test_record(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, at)) = self.test_record else { return };
        self.egui_ctx.request_repaint();
        let fail = |m: String| -> ! {
            println!("RECORD TEST FAIL: {m}");
            std::process::exit(1);
        };
        if at.elapsed() > std::time::Duration::from_secs(40) {
            fail(format!("timeout at stage {stage}"));
        }
        let chord = egui::Modifiers { shift: true, command: true, mac_cmd: cfg!(target_os = "macos"), ctrl: !cfg!(target_os = "macos"), ..Default::default() };
        let tap = |raw: &mut egui::RawInput| {
            raw.events.push(egui::Event::ModifiersChanged(chord));
            for pressed in [true, false] {
                raw.events.push(egui::Event::Key { key: egui::Key::R, physical_key: None, pressed, repeat: false, modifiers: chord });
            }
            raw.events.push(egui::Event::ModifiersChanged(Default::default()));
        };
        let secs = at.elapsed().as_secs_f32();
        let mut next = stage;
        match stage {
            0 if self.layout_settled() && secs > 1.5 => {
                self.recorder.opts.area = recorder::Area::Window;
                self.recorder.opts.gif = true;
                self.playing = true;
                tap(raw);
                next = 1;
            }
            1 => {
                if !self.recorder.recording() {
                    fail(format!("⌘⇧R did not start a take: {:?}", self.recorder.error));
                }
                next = 2;
            }
            2 => {
                // A pointer circling the viewer, so the clip has a cursor
                // to draw and the HUD something to answer.
                let r = self.viewer_rect;
                let a = secs * 2.0;
                let p = r.center() + egui::vec2(a.cos(), a.sin()) * r.height().min(r.width()) * 0.3;
                raw.events.push(egui::Event::PointerMoved(p));
                let len = std::env::var("WOBBLE_RECORD_TEST_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(2.5f32);
                if secs > 1.5 + len {
                    tap(raw);
                    next = 3;
                }
            }
            3 => {
                let Some(n) = self.recorder.review_frames() else {
                    if self.recorder.recording() {
                        self.test_record = Some((stage, at));
                        return;
                    }
                    fail(format!("the take did not reach review: {:?}", self.recorder.error));
                };
                // 2.5 s at 30 fps is 75; a window that does not draw gets ~none.
                if n < 45 {
                    fail(format!("only {n} frames captured (is the window drawing? WOBBLE_TEST_FOCUS=1)"));
                }
                self.recorder.save_all();
                next = 4;
            }
            4 => match self.recorder.save_result() {
                None => {}
                Some(Err(e)) => fail(format!("saving: {e}")),
                Some(Ok(saved)) => {
                    let size = |p: &Option<std::path::PathBuf>| p.as_ref().and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len()).unwrap_or(0);
                    if saved.mp4.is_none() && saved.note.is_none() {
                        fail("no MP4 and no reason why".into());
                    }
                    #[cfg(any(target_os = "macos", target_os = "windows"))]
                    if size(&saved.mp4) < 20_000 {
                        fail(format!("the MP4 is {} bytes", size(&saved.mp4)));
                    }
                    if size(&saved.gif) < 20_000 {
                        fail(format!("the GIF is {} bytes", size(&saved.gif)));
                    }
                    println!(
                        "record: {} frames, mp4 {:?} ({} bytes), gif {:?} ({} bytes)",
                        self.recorder.review_frames().unwrap_or(0),
                        saved.mp4,
                        size(&saved.mp4),
                        saved.gif,
                        size(&saved.gif)
                    );
                    // Then the screenshot button: one PNG of the window.
                    let ctx = self.egui_ctx.clone();
                    self.take_screenshot(&ctx);
                    next = 5;
                }
            },
            5 => match &self.shot.note {
                None => {}
                Some((Err(e), _)) => fail(format!("screenshot: {e}")),
                Some((Ok(path), _)) => {
                    let path = path.clone();
                    let dec = std::fs::File::open(&path).map_err(|e| e.to_string()).and_then(|f| {
                        let mut r = png::Decoder::new(std::io::BufReader::new(f)).read_info().map_err(|e| e.to_string())?;
                        let mut buf = vec![0u8; r.output_buffer_size().unwrap_or(0)];
                        let info = r.next_frame(&mut buf).map_err(|e| e.to_string())?;
                        buf.truncate(info.buffer_size());
                        Ok((info.width, info.height, buf))
                    });
                    let (w, h, px) = dec.unwrap_or_else(|e| fail(format!("reading {}: {e}", path.display())));
                    if (w, h) != (self.config.width, self.config.height) {
                        fail(format!("the PNG is {w}x{h}, the window {}x{}", self.config.width, self.config.height));
                    }
                    let first = &px[..3];
                    if px.chunks_exact(3).all(|p| p == first) {
                        fail("the PNG is one flat colour".into());
                    }
                    println!("RECORD TEST PASS: clip saved, screenshot {} ({w}x{h})", path.display());
                    std::process::exit(0);
                }
            },
            _ => {}
        }
        self.test_record = Some((next, at));
    }
}
