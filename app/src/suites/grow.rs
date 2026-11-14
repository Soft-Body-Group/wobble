//! `--test-grow`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// A Read whose range runs PAST its source: the timeline is the range,
    /// the frames that do not exist yet read as black without stopping
    /// anything, and a frame written while the app is running appears on
    /// its own. That last part is the feature — watching a render arrive —
    /// and it is the part a static test would miss entirely.
    pub(crate) fn drive_test_grow(&mut self) {
        let Some((stage, node, frames)) = self.test.grow else { return };
        let t0 = *self.test.grow_t0.get_or_insert_with(web_time::Instant::now);
        if t0.elapsed().as_secs_f32() > 60.0 {
            println!("GROW TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let dir = std::env::temp_dir().join("wobble-grow-test");
        // Four frames now; a fifth appears later, mid-run.
        let write = |n: u64| -> bool {
            let path = dir.join(format!("plate.{n:04}.png"));
            let (w, h) = (32u32, 16u32);
            // Frame N is a flat grey of N*40 — so which frame is on screen
            // is readable from one pixel.
            let px = vec![(n as u8).saturating_mul(40); (w * h * 4) as usize];
            let Ok(f) = std::fs::File::create(&path) else { return false };
            let mut enc = png::Encoder::new(f, w, h);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            enc.write_header()
                .and_then(|mut wr| wr.write_image_data(&px))
                .is_ok()
        };
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                let _ = std::fs::remove_dir_all(&dir);
                if std::fs::create_dir_all(&dir).is_err() {
                    println!("GROW TEST FAIL: could not make {}", dir.display());
                    std::process::exit(1);
                }
                for n in 0..4 {
                    if !write(n) {
                        println!("GROW TEST FAIL: could not write frame {n}");
                        std::process::exit(1);
                    }
                }
                let id = self.egraph.add(
                    NodeKind::read(self.next_read_slot, &dir.display().to_string()),
                    egui::pos2(320.0, -240.0),
                );
                self.next_read_slot += 1;
                self.load_read(id, &dir.display().to_string());
                self.test.grow = Some((1, id, 0));
            }
            // Ask for eight frames when only four exist.
            1 => {
                let Some(l) = self.loaded.get(&node) else {
                    println!("GROW TEST FAIL: sequence did not load");
                    std::process::exit(1);
                };
                if l.seq.frames.len() != 4 {
                    println!(
                        "GROW TEST FAIL: {} frames on disk, want 4",
                        l.seq.frames.len()
                    );
                    std::process::exit(1);
                }
                // LAST runs four frames past the files, the way a user
                // watches a render arrive.
                if let Some(editor::ENode { kind: NodeKind::Read { first, last, .. }, .. }) =
                    self.egraph.node_mut(node)
                {
                    *last = first.map(|f| f + 7);
                }
                self.playback_dirty = true;
                self.playing = true;
                self.test.grow = Some((2, node, 0));
            }
            // This Read's own sequence must BE the range, not the footage.
            // (The app also has the two CLI plates loaded, so the GLOBAL
            // timeline is theirs — it is this reader that must grow.)
            2 if frames == 0 => {
                let Some(seq) = self.grow_seq(&dir) else {
                    // The rig rebuilds asynchronously; keep waiting.
                    self.test.grow = Some((2, node, 0));
                    return;
                };
                if seq.frames.len() != 8 {
                    self.test.grow = Some((2, node, 0));
                    return;
                }
                // The extrapolated path for the frame that does not exist
                // yet has to be the one the renderer will actually write.
                let want = dir.join("plate.0005.png");
                if seq.frames.get(5) != Some(&want) {
                    println!(
                        "GROW TEST FAIL: frame 5 of the range resolves to {:?}, \
                         want {want:?} — the renderer's next file",
                        seq.frames.get(5)
                    );
                    std::process::exit(1);
                }
                // And the tail really is absent — otherwise "it kept
                // playing" would prove nothing.
                if seq.frames[5..].iter().any(|f| f.exists()) {
                    println!("GROW TEST FAIL: the tail frames already exist on disk");
                    std::process::exit(1);
                }
                self.test.grow = Some((3, node, self.displayed_total));
            }
            // Play across the empty tail: it must keep running.
            3 if self.displayed_total >= frames + 40 => {
                // Now write frame 4 WHILE the app is playing. Nothing is
                // reloaded and no button is pressed; it must simply appear.
                if !write(4) {
                    println!("GROW TEST FAIL: could not write the late frame");
                    std::process::exit(1);
                }
                self.test.grow = Some((4, node, self.displayed_total));
            }
            // Give the loop time to come back round to tick 4 and read it.
            4 if self.displayed_total >= frames + 60 => {
                let Some(seq) = self.grow_seq(&dir) else {
                    println!("GROW TEST FAIL: the grown sequence vanished");
                    std::process::exit(1);
                };
                // Read frame 4 back through the decoder the player uses: it
                // was NOT on disk when the range was set, and no cache
                // entry may be standing in for it.
                let stride = seq.info.row_stride_f16();
                let mut buf = vec![half::f16::ZERO; stride * seq.info.height as usize];
                let path = seq.frames[4].clone();
                if let Err(e) =
                    self.decoder.decode_into(&path, seq.info, stride, &mut buf)
                {
                    println!("GROW TEST FAIL: late frame did not decode: {e:#}");
                    std::process::exit(1);
                }
                let v = buf[0].to_f32();
                let want = (4u8.saturating_mul(40)) as f32 / 255.0;
                if (v - want).abs() > 0.01 {
                    println!(
                        "GROW TEST FAIL: late frame read {v:.3}, want {want:.3}"
                    );
                    std::process::exit(1);
                }
                let _ = std::fs::remove_dir_all(&dir);
                println!(
                    "GROW TEST PASS: 4 frames on disk read as an 8-frame range; the \
                     4 tail frames were absent and playback still advanced 40+ \
                     frames across them; frame 5 resolves to the renderer's next \
                     path; a frame written mid-run was picked up with no reload"
                );
                std::process::exit(0);
            }
            _ => {}
        }
    }
}
