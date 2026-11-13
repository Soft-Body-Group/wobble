//! `--test-still`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// PNG and JPEG through the real Read path: a mixed still sequence
    /// loads, reports the right length and size, and — the part that is
    /// easy to get wrong invisibly — decodes to CODE VALUES. A decoder
    /// that helpfully linearised sRGB would still look like a picture, so
    /// the assertion is a specific number, not "looks plausible".
    pub(crate) fn drive_test_still(&mut self) {
        let Some((stage, mut frames, node)) = self.test_still else { return };
        frames += 1;
        if frames > 2400 {
            println!("STILL TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let dir = std::env::temp_dir().join("wobble-still-suite");
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                let _ = std::fs::remove_dir_all(&dir);
                if std::fs::create_dir_all(&dir).is_err() {
                    println!("STILL TEST FAIL: could not make {}", dir.display());
                    std::process::exit(1);
                }
                // Column 0 is 8-bit 128 — the value an sRGB decode would
                // turn into 0.216 and a code-value decode leaves at 0.502.
                // Column 1 is white, so a wholly dead decode is also caught.
                // Frame 3 is a white PORTRAIT, a photo of another size:
                // it must be fitted into the 64x32 frame, not refused.
                for f in 0..4 {
                    let (w, h) = if f == 3 { (32u32, 64u32) } else { (64u32, 32u32) };
                    let path = dir.join(format!("plate.{f:04}.png"));
                    let mut px = vec![0u8; (w * h * 4) as usize];
                    for y in 0..h as usize {
                        for x in 0..w as usize {
                            let i = (y * w as usize + x) * 4;
                            let v = match x {
                                _ if f == 3 => 255,
                                0 => 128,
                                1 => 255,
                                _ => 0,
                            };
                            px[i] = v;
                            px[i + 1] = v;
                            px[i + 2] = v;
                            px[i + 3] = 255;
                        }
                    }
                    let Ok(file) = std::fs::File::create(&path) else {
                        println!("STILL TEST FAIL: could not write {}", path.display());
                        std::process::exit(1);
                    };
                    let mut enc = png::Encoder::new(file, w, h);
                    enc.set_color(png::ColorType::Rgba);
                    enc.set_depth(png::BitDepth::Eight);
                    if enc
                        .write_header()
                        .and_then(|mut wr| wr.write_image_data(&px))
                        .is_err()
                    {
                        println!("STILL TEST FAIL: encoding {}", path.display());
                        std::process::exit(1);
                    }
                }
                let id = self.egraph.add(
                    NodeKind::read(self.next_read_slot, &dir.display().to_string()),
                    egui::pos2(320.0, -240.0),
                );
                self.next_read_slot += 1;
                self.load_read(id, &dir.display().to_string());
                self.test_still = Some((1, frames, id));
            }
            1 if frames > 20 => {
                let Some(l) = self.loaded.get(&node) else {
                    let err = self
                        .load_errors
                        .get(&node)
                        .cloned()
                        .unwrap_or_else(|| "no error recorded".into());
                    println!("STILL TEST FAIL: sequence did not load ({err})");
                    std::process::exit(1);
                };
                if l.seq.frames.len() != 4 {
                    println!(
                        "STILL TEST FAIL: {} frames found, want 4",
                        l.seq.frames.len()
                    );
                    std::process::exit(1);
                }
                if (l.seq.info.width, l.seq.info.height) != (64, 32) {
                    println!(
                        "STILL TEST FAIL: header says {}x{}, want 64x32",
                        l.seq.info.width, l.seq.info.height
                    );
                    std::process::exit(1);
                }
                if l.seq.movie {
                    println!("STILL TEST FAIL: a PNG sequence is not a movie");
                    std::process::exit(1);
                }
                let stride = l.seq.info.row_stride_f16();
                let mut buf = vec![half::f16::ZERO; stride * 32];
                let path = l.seq.frames[0].clone();
                if let Err(e) =
                    self.decoder.decode_into(&path, l.seq.info, stride, &mut buf)
                {
                    println!("STILL TEST FAIL: decode: {e:#}");
                    std::process::exit(1);
                }
                let (grey, white) = (buf[0].to_f32(), buf[4].to_f32());
                if (grey - 0.502).abs() > 0.01 {
                    println!(
                        "STILL TEST FAIL: code 128 decoded to {grey:.3}; expected \
                         0.502 (a code value). 0.216 would mean the decoder \
                         applied sRGB, which is InputSpace's job."
                    );
                    std::process::exit(1);
                }
                if (white - 1.0).abs() > 0.01 {
                    println!("STILL TEST FAIL: code 255 decoded to {white:.3}, want 1.0");
                    std::process::exit(1);
                }
                // The portrait: fitted whole (16x32, centred at x 24..40),
                // transparent black either side, and counted on the panel.
                let path = l.seq.frames[3].clone();
                if let Err(e) =
                    self.decoder.decode_into(&path, l.seq.info, stride, &mut buf)
                {
                    println!("STILL TEST FAIL: a frame of another size failed: {e:#}");
                    std::process::exit(1);
                }
                let at = |x: usize, y: usize| buf[y * stride + x * 4 + 3].to_f32();
                if (at(32, 16) - 1.0).abs() > 0.01 || at(4, 16) != 0.0 || at(60, 16) != 0.0 {
                    println!(
                        "STILL TEST FAIL: portrait not fitted: alpha {:.2} in the \
                         middle, {:.2} / {:.2} at the sides",
                        at(32, 16),
                        at(4, 16),
                        at(60, 16)
                    );
                    std::process::exit(1);
                }
                let t0 = web_time::Instant::now();
                let fitted = loop {
                    let n = l.fitted.count.load(std::sync::atomic::Ordering::Relaxed);
                    if n >= 1 || t0.elapsed().as_secs_f32() > 2.0 {
                        break n;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(10));
                };
                if fitted != 1 {
                    println!("STILL TEST FAIL: {fitted} frames counted as fitted, want 1");
                    std::process::exit(1);
                }
                // The offered input space is a GUESS, shown not applied.
                if engine::color::InputSpace::guess_for_extension("png")
                    != Some(engine::color::Colorspace::SRGB)
                {
                    println!("STILL TEST FAIL: png should offer sRGB as its guess");
                    std::process::exit(1);
                }
                // A `~` path must load exactly like the absolute one: it is
                // what people type, and the field takes text, not a dialog.
                // Copy the fixture out BEFORE the source directory goes.
                let home = engine::paths::home().expect("HOME is set");
                let under_home = home.join("wobble-tilde-suite");
                let _ = std::fs::remove_dir_all(&under_home);
                std::fs::create_dir_all(&under_home).ok();
                for f in 0..3 {
                    let src = dir.join(format!("plate.{f:04}.png"));
                    if std::fs::copy(&src, under_home.join(format!("p.{f:04}.png"))).is_err() {
                        println!("STILL TEST FAIL: could not stage the tilde fixture");
                        std::process::exit(1);
                    }
                }
                let _ = std::fs::remove_dir_all(&dir);
                self.test_still = Some((2, frames, node));
            }
            2 => {
                let home = engine::paths::home().expect("HOME is set");
                let under_home = home.join("wobble-tilde-suite");
                let id = self.egraph.add(
                    NodeKind::read(self.next_read_slot, "~/wobble-tilde-suite"),
                    egui::pos2(520.0, -240.0),
                );
                self.next_read_slot += 1;
                self.load_read(id, "~/wobble-tilde-suite");
                let ok = self.loaded.get(&id).map(|l| l.seq.frames.len());
                let err = self.load_errors.get(&id).cloned();
                let _ = std::fs::remove_dir_all(&under_home);
                match ok {
                    Some(3) => {}
                    other => {
                        println!(
                            "STILL TEST FAIL: `~/wobble-tilde-suite` loaded {other:?} \
                             frames, want 3 ({err:?}) — a typed ~ must mean home"
                        );
                        std::process::exit(1);
                    }
                }
                let (grey, white) = (0.502f32, 1.0f32);
                println!(
                    "STILL TEST PASS: 4-frame 64x32 RGBA PNG loaded via Read; code \
                     values preserved (128 -> {grey:.3}, 255 -> {white:.3}); sRGB \
                     offered as a guess, not applied; and `~/…` resolved to home"
                );
                std::process::exit(0);
            }
            _ => self.test_still = Some((stage, frames, node)),
        }
    }
}
