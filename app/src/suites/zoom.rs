//! `--test-zoom`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// Properties zoom: the tile (and everything sized off it) scales,
    /// while the type does not — so a panel at 2x is twice as wide but its
    /// labels are still 1x.
    pub(crate) fn drive_test_zoom(&mut self) {
        let Some((stage, mut frames, w1)) = self.test_zoom else { return };
        frames += 1;
        if frames > 1200 {
            println!("ZOOM TEST FAIL: timeout at stage {stage} (w1 {w1})");
            std::process::exit(1);
        }
        let font_px = |s: &Self| {
            // What a Body label actually renders at, zoom_factor included.
            s.egui_ctx.options(|o| o.zoom_factor)
                * s.egui_ctx.style_of(egui::Theme::Dark).text_styles
                    [&egui::TextStyle::Body]
                    .size
        };
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                self.prop_order = self.egraph.nodes.keys().copied().collect();
                self.props_visible = true;
                self.bottom = BottomTab::Properties;
                self.set_prop_zoom(1.0);
                self.test_zoom = Some((1, frames, font_px(self)));
            }
            // Settle, then bank the 1x width (stashing the 1x font size in
            // the slot until the width replaces it).
            1 if frames > 12 && self.last_panel_w > 0.0 => {
                let f1 = w1;
                let w = self.last_panel_w;
                self.zoom_x_1x = self.last_panel_x;
                self.zoom_h_1x = self.last_panel_h;
                self.set_prop_zoom(2.0);
                // Anchor well inside the strip: keeping that point still
                // demands a real scroll shift, unlike anchoring the edge.
                self.prop_zoom_anchor = Some(egui::pos2(
                    self.prop_origin.x + 300.0,
                    self.last_panel_rect_y,
                ));
                self.test_zoom = Some((2, frames, w));
                self.zoom_font_1x = f1;
            }
            2 if frames > 24 => {
                let w2 = self.last_panel_w;
                let ratio = w2 / w1.max(1.0);
                let f2 = font_px(self);
                if (ratio - 2.0).abs() > 0.05 {
                    println!(
                        "ZOOM TEST FAIL: panel {w1:.0} -> {w2:.0} (ratio {ratio:.2}, want 2.0)"
                    );
                    std::process::exit(1);
                }
                if (f2 - self.zoom_font_1x).abs() > 0.01 {
                    println!(
                        "ZOOM TEST FAIL: body text scaled {:.1} -> {f2:.1}",
                        self.zoom_font_1x
                    );
                    std::process::exit(1);
                }
                // The content under the anchor must not move: with the
                // strip's first panel starting at content x = 0, that means
                // it slides left by exactly anchor_offset * (ratio - 1).
                // Height is the STRIP's to give, not the tile's: a zoomed
                // tile scrolls inside the docked strip rather than growing
                // the bottom pane. Overflowing the strip by even a pixel is
                // the bug — egui persists the pane at its content height,
                // so an overflow ratchets it upward every frame.
                let h2 = self.last_panel_h;
                let strip = self.last_strip_h;
                if h2 > strip + 0.5 {
                    println!(
                        "ZOOM TEST FAIL: panel height {:.0} -> {h2:.0} overflows the {strip:.0}px strip (should scroll, not grow)",
                        self.zoom_h_1x
                    );
                    std::process::exit(1);
                }
                let want_shift = 300.0 * (2.0 - 1.0);
                let got_shift = self.zoom_x_1x - self.last_panel_x;
                let drift = (got_shift - want_shift).abs();
                if drift > 6.0 {
                    println!(
                        "ZOOM TEST FAIL: anchor slipped {drift:.0}px (panel slid {got_shift:.0}, want {want_shift:.0})"
                    );
                    std::process::exit(1);
                }
                println!(
                    "ZOOM TEST PASS: panel {w1:.0} -> {w2:.0} (x{ratio:.2}), height {:.0} -> {h2:.0} in a {strip:.0}px strip, body text {f2:.1}px unchanged, anchored point held (slid {got_shift:.0}px, want {want_shift:.0}px)",
                    self.zoom_h_1x
                );
                // On to per-panel width, back at 1x.
                self.set_prop_zoom(1.0);
                self.test_zoom = None;
                self.test_panelw = Some(PanelWTest { start: Some(web_time::Instant::now()), ..Default::default() });
            }
            _ => self.test_zoom = Some((stage, frames, w1)),
        }
    }
}
