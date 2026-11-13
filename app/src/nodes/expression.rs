//! Expression's app half: the panel and the glyph. The language, the WGSL
//! and the params are `engine::nodes::expression`.
//!
//! A field is a BUFFER while you type — the Shader node's contract: the
//! verdict under it is about what you are typing, and nothing reaches the
//! graph (a recompile) until Enter or leaving the field. Escape walks away
//! from the edit.

use egui::{Color32, Painter, Pos2, Stroke};
use engine::graph::ExpressionParams;
use engine::nodes::expression::{check, CHANNELS as LANES, FUNCTIONS};

/// Two tiles wide, so an expression has room to be read; one tall.
pub const TILES: usize = 2;

/// The four fields, the verdicts under them, and the help line.
pub fn panel(ui: &mut egui::Ui, node: u64, p: &mut ExpressionParams) {
    let w = ui.available_width();
    for (i, src) in p.exprs.iter_mut().enumerate() {
        field(ui, node, i, src, w);
    }
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new("r g b a · x y width height frame · inf nan pi")
            .small()
            .weak(),
    )
    .on_hover_text(format!(
        "variables: r g b a (the input), x y (pixels from the top left), \
         width height, frame\nconstants: pi tau e inf nan\noperators: + - * / % ^ \
         == != < > <= >= && || ! and c ? a : b\nfunctions: {FUNCTIONS}\n\n\
         r == 0 ? 1 : 0        a matte where red is exactly zero\n\
         isinf(r) ? 0 : r       infinities out\n\
         k = r * 2; k > 1 ? 1 : k   a line can name a value"
    ));
    ui.label(egui::RichText::new("empty passes a channel through · enter applies").small().weak());
}

/// One channel's field. `authored` is what renders; the buffer lives in
/// egui's memory only while the field has focus.
fn field(ui: &mut egui::Ui, node: u64, i: usize, authored: &mut String, w: f32) {
    let id = egui::Id::new(("expression-field", node, i));
    let buf_id = id.with("buf");
    // Evaluated before any data closure: egui's lock is not re-entrant.
    let editing = ui.memory(|m| m.has_focus(id));
    let held: Option<String> = if editing { ui.data(|d| d.get_temp(buf_id)) } else { None };
    let mut buf = held.unwrap_or_else(|| authored.clone());
    ui.horizontal(|ui| {
        ui.add_sized(
            [14.0, 18.0],
            egui::Label::new(egui::RichText::new(LANES[i].to_string()).monospace()),
        );
        let r = ui.add(
            egui::TextEdit::singleline(&mut buf)
                .id(id)
                .font(egui::TextStyle::Monospace)
                .desired_width((w - 24.0).max(60.0))
                .hint_text(format!("{} (unchanged)", LANES[i])),
        );
        if r.changed() || r.gained_focus() {
            ui.data_mut(|d| d.insert_temp(buf_id, buf.clone()));
        }
        if r.lost_focus() {
            let walked_away = ui.input(|inp| inp.key_pressed(egui::Key::Escape));
            if !walked_away && buf != *authored {
                *authored = buf.clone();
            }
            ui.data_mut(|d| d.remove::<String>(buf_id));
        }
    });
    if let Some(err) = check(&buf) {
        ui.horizontal(|ui| {
            ui.add_space(18.0);
            ui.label(
                egui::RichText::new(format!("{} — {} passes through", err.describe(&buf), LANES[i]))
                    .small()
                    .color(crate::theme::error()),
            );
        });
    }
}

/// The graph glyph: an equals sign — a pixel is what you write.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.14).max(1.0), crate::theme::node_ink());
    let (w, gap) = (r * 0.42, r * 0.17);
    for dy in [-gap, gap] {
        painter.line_segment([Pos2::new(c.x - w, c.y + dy), Pos2::new(c.x + w, c.y + dy)], ink);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The panel WITH its text, on the GPU, to `target/expression-panel-gpu.png`
    /// — the plain snapshot draws no glyphs. For a look, not a check:
    /// `cargo test -p app expression_panel_render -- --ignored`.
    #[test]
    #[ignore]
    fn expression_panel_render() {
        let mut p = ExpressionParams {
            exprs: ["r == 0 ? 1 : 0".into(), "isinf(g) ? 0 : g".into(), String::new(), "r == inf 0: 1 : r".into()],
            time: 0.0,
        };
        let tiles =
            |n: usize| crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0);
        crate::uishot::render("expression-panel-gpu", tiles(TILES) - 14.0, tiles(1) - 14.0 - 22.0, 2.0, 3, |ui, _| {
            panel(ui, 1, &mut p);
        });
    }

    /// The panel fits the tiles it asks for, both ways, with every field
    /// showing an error (the tallest it gets).
    #[test]
    fn the_panel_fits_its_tiles() {
        let mut p = ExpressionParams {
            exprs: ["r ==".into(), "g +".into(), "b ?".into(), "a ^".into()],
            time: 0.0,
        };
        let (over, wide) = (std::cell::Cell::new(0.0f32), std::cell::Cell::new(0.0f32));
        let tiles =
            |n: usize| crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0);
        let shot = crate::snapshot::shoot(tiles(TILES) - 14.0, tiles(1) - 14.0 - 22.0, |ui| {
            let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                panel(ui, 1, &mut p);
            });
            over.set(out.content_size.y - out.inner_rect.height());
            wide.set(out.content_size.x - out.inner_rect.width());
        });
        shot.save("expression-panel");
        assert!(over.get() <= 0.5, "overflows its tile DOWN by {}pt", over.get());
        assert!(wide.get() <= 0.5, "overflows its tiles SIDEWAYS by {}pt", wide.get());
    }
}
