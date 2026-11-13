//! `--test-dpx`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// DPX end to end: a written 10-bit log DPX sequence loads through a
    /// Read node, resolves its real dimensions, and reaches the compositor
    /// with the film curve linearized.
    pub(crate) fn drive_test_dpx(&mut self) {
        let Some((stage, mut frames, node)) = self.test_dpx else { return };
        frames += 1;
        if frames > 2400 {
            println!("DPX TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let dir = std::env::temp_dir().join("wobble-dpx-test");
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                let _ = std::fs::remove_dir_all(&dir);
                if std::fs::create_dir_all(&dir).is_err() {
                    println!("DPX TEST FAIL: could not make {}", dir.display());
                    std::process::exit(1);
                }
                // A ramp, plus known black (95) and white (685) code values.
                for f in 0..4 {
                    let path = dir.join(format!("plate.{f:04}.dpx"));
                    if let Err(e) = engine::dpx::write_test_dpx(&path, 64, 32, true, |x, _| {
                        match x {
                            0 => [95, 95, 95],
                            1 => [685, 685, 685],
                            _ => {
                                let v = (x * 16).min(1023) as u16;
                                [v, v, v]
                            }
                        }
                    }) {
                        println!("DPX TEST FAIL: writing fixture: {e:#}");
                        std::process::exit(1);
                    }
                }
                let id = self.egraph.add(
                    NodeKind::read(self.next_read_slot, &dir.display().to_string()),
                    egui::pos2(320.0, -240.0),
                );
                self.next_read_slot += 1;
                self.load_read(id, &dir.display().to_string());
                self.test_dpx = Some((1, frames, id));
            }
            1 if frames > 20 => {
                let Some(l) = self.loaded.get(&node) else {
                    let err = self
                        .load_errors
                        .get(&node)
                        .cloned()
                        .unwrap_or_else(|| "no error recorded".into());
                    println!("DPX TEST FAIL: sequence did not load ({err})");
                    std::process::exit(1);
                };
                if l.seq.frames.len() != 4 {
                    println!(
                        "DPX TEST FAIL: {} frames found, want 4",
                        l.seq.frames.len()
                    );
                    std::process::exit(1);
                }
                if (l.seq.info.width, l.seq.info.height) != (64, 32) {
                    println!(
                        "DPX TEST FAIL: header says {}x{}, want 64x32",
                        l.seq.info.width, l.seq.info.height
                    );
                    std::process::exit(1);
                }
                // Decode one directly: CODE VALUES come back (no curve in
                // the decoder), and the Read — left on "default" — lowers to
                // the dpx row's Cineon transform, which lands them on the film
                // curve's black and white.
                let mut buf =
                    vec![half::f16::ZERO; l.seq.info.row_stride_f16() * 32];
                let path = l.seq.frames[0].clone();
                let stride = l.seq.info.row_stride_f16();
                if let Err(e) =
                    self.decoder.decode_into(&path, l.seq.info, stride, &mut buf)
                {
                    println!("DPX TEST FAIL: decode: {e:#}");
                    std::process::exit(1);
                }
                let (c95, c685) = (buf[0].to_f32(), buf[4].to_f32());
                if (c95 - 95.0 / 1023.0).abs() > 2e-3 || (c685 - 685.0 / 1023.0).abs() > 2e-3 {
                    println!(
                        "DPX TEST FAIL: decoder should return codes (95 -> {c95:.4}, 685 -> {c685:.4})"
                    );
                    std::process::exit(1);
                }
                let lowered = self.egraph.to_engine_graph();
                let input = match lowered.nodes.get(&node) {
                    Some(NodeKind::Read { input, .. }) => *input,
                    _ => engine::color::InputSpace::Working,
                };
                if input != engine::color::InputSpace::Explicit(engine::color::Colorspace::CINEON) {
                    println!("DPX TEST FAIL: default input lowered to {input:?}, want Cineon");
                    std::process::exit(1);
                }
                let black = engine::color::cineon_to_linear(c95 * 1023.0);
                let white = engine::color::cineon_to_linear(c685 * 1023.0);
                if black.abs() > 0.01 || (white - 1.0).abs() > 0.03 {
                    println!(
                        "DPX TEST FAIL: Cineon transform off (95 -> {black:.3}, 685 -> {white:.3})"
                    );
                    std::process::exit(1);
                }
                let _ = std::fs::remove_dir_all(&dir);
                println!(
                    "DPX TEST PASS: 4-frame 64x32 10-bit log DPX loaded via Read; codes decoded, default input = Cineon (95 -> {black:.3}, 685 -> {white:.3})"
                );
                std::process::exit(0);
            }
            _ => self.test_dpx = Some((stage, frames, node)),
        }
    }
}
