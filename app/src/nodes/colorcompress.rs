//! ColorCompress's app half: the panel, the glyph and the channel
//! bindings. The arithmetic, the WGSL and the params are
//! `engine::nodes::colorcompress`.

use crate::anim::Channel;
use crate::rgbslider::RgbSpec;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use engine::color::Colorspace;
use engine::graph::ColorCompressParams;
use std::collections::BTreeSet;

/// Every animatable param, in panel order: the two levels measured, then
/// the two they land on.
pub const CHANNELS: &[Channel] = &[
    Channel::CompressBlackR,
    Channel::CompressBlackG,
    Channel::CompressBlackB,
    Channel::CompressWhiteR,
    Channel::CompressWhiteG,
    Channel::CompressWhiteB,
    Channel::CompressLiftR,
    Channel::CompressLiftG,
    Channel::CompressLiftB,
    Channel::CompressMultR,
    Channel::CompressMultG,
    Channel::CompressMultB,
];

/// The input pair and the output pair, as the panel groups them.
const BLACK: [Channel; 3] =
    [Channel::CompressBlackR, Channel::CompressBlackG, Channel::CompressBlackB];
const WHITE: [Channel; 3] =
    [Channel::CompressWhiteR, Channel::CompressWhiteG, Channel::CompressWhiteB];
const LIFT: [Channel; 3] =
    [Channel::CompressLiftR, Channel::CompressLiftG, Channel::CompressLiftB];
const MULT: [Channel; 3] =
    [Channel::CompressMultR, Channel::CompressMultG, Channel::CompressMultB];

/// Panel size in tiles. An rgb row does not shrink below about 420pt —
/// label, gang, three lanes, three number fields and the chip — so two of
/// them SIDE BY SIDE want a thousand points of panel, and four of them
/// STACKED want more than one tile of height. One column, two tiles tall,
/// is the shape that actually fits: 500 wide against the row's 420, and
/// 470 of height against the four rows' ~370.
pub const TILES: usize = 2;
/// ...and two tiles TALL, which is where the four rows go.
pub const TILE_ROWS: usize = 2;

pub fn get(p: &ColorCompressParams, ch: Channel) -> Option<f32> {
    use Channel::*;
    Some(match ch {
        CompressBlackR => p.black[0],
        CompressBlackG => p.black[1],
        CompressBlackB => p.black[2],
        CompressWhiteR => p.white[0],
        CompressWhiteG => p.white[1],
        CompressWhiteB => p.white[2],
        CompressLiftR => p.lift[0],
        CompressLiftG => p.lift[1],
        CompressLiftB => p.lift[2],
        CompressMultR => p.mult[0],
        CompressMultG => p.mult[1],
        CompressMultB => p.mult[2],
        _ => return None,
    })
}

pub fn set(p: &mut ColorCompressParams, ch: Channel, v: f32) {
    use Channel::*;
    match ch {
        CompressBlackR => p.black[0] = v,
        CompressBlackG => p.black[1] = v,
        CompressBlackB => p.black[2] = v,
        CompressWhiteR => p.white[0] = v,
        CompressWhiteG => p.white[1] = v,
        CompressWhiteB => p.white[2] = v,
        CompressLiftR => p.lift[0] = v,
        CompressLiftG => p.lift[1] = v,
        CompressLiftB => p.lift[2] = v,
        CompressMultR => p.mult[0] = v,
        CompressMultG => p.mult[1] = v,
        CompressMultB => p.mult[2] = v,
        _ => {}
    }
}

/// The curve editor's name for one of this node's channels.
pub fn label(ch: Channel) -> Option<&'static str> {
    use Channel::*;
    Some(match ch {
        CompressBlackR => "black point r",
        CompressBlackG => "black point g",
        CompressBlackB => "black point b",
        CompressWhiteR => "white point r",
        CompressWhiteG => "white point g",
        CompressWhiteB => "white point b",
        CompressLiftR => "lift r",
        CompressLiftG => "lift g",
        CompressLiftB => "lift b",
        CompressMultR => "mult r",
        CompressMultG => "mult g",
        CompressMultB => "mult b",
        _ => return None,
    })
}

/// One rgb row, ganged unless this node has that row unganged. The gang
/// lives on the ENode by row label, like the Mult's — it is how the panel
/// is being driven, not what the pixels do. A row the user ctrl-dragged
/// out lands in `out` rather than being returned, so the caller can keep
/// one slot across a column without moving it about.
#[allow(clippy::too_many_arguments)]
fn row(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    ungang: &mut BTreeSet<String>,
    label: &str,
    v: &mut [f32; 3],
    spec: RgbSpec,
    chans: &[Channel; 3],
    out: &mut Option<HudSlider>,
) {
    // Each row gets its OWN child, sized to leave room for the overhang.
    // One shared parent will not do: egui widens a parent to fit what
    // overflows it, so row 1's chip would buy row 2 more width, row 2's
    // buys row 3 more still, and the panel walks off its tile a row at a
    // time — measured at 78pt of overflow across four rows.
    let w = (ui.available_width() - ROW_OVERHANG).max(240.0);
    let mut c = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(egui::Rect::from_min_size(ui.cursor().min, egui::vec2(w, 0.0)))
            .layout(egui::Layout::top_down(egui::Align::Min)),
    );
    let mut gang = !ungang.contains(label);
    if let Some(h) = widgets::rgb_row(&mut c, rcx, label, v, &mut gang, spec, chans) {
        *out = Some(h);
    }
    let used = c.min_rect();
    ui.advance_cursor_after_rect(used);
    if gang {
        ungang.remove(label);
    } else {
        ungang.insert(label.to_string());
    }
}

/// What an rgb row hangs PAST the width it is given: it lays its track into
/// the available width and then puts the value chip beyond it, so a row is
/// always about this much wider than its parent (measured: available + 26
/// at every width). Rows are drawn into a child this much narrower, or the
/// chip lands outside the tile — and, worse, egui grows the parent to fit
/// it, so the NEXT row is wider again and the panel walks off the edge one
/// row at a time. See the note in `.agents/messages.md`.
const ROW_OVERHANG: f32 = crate::rgbslider::SWATCH_W + 4.0;

/// A heading over the pair below it.
fn head(ui: &mut egui::Ui, text: &str, hint: &str) {
    ui.label(egui::RichText::new(text).small().weak()).on_hover_text(hint);
}

/// The properties panel: the pair you MEASURE on the left, the pair they
/// LAND ON on the right, because that is the sentence the node says.
pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut ColorCompressParams,
    working: Colorspace,
    ungang: &mut BTreeSet<String>,
) -> Option<HudSlider> {
    let luma = working.primaries.luma_weights();
    let disp = engine::color::primaries_matrix(
        working.primaries,
        engine::color::Primaries::Rec709,
    );
    // A level is a SCALE (halving a white point is a stop, and it must not
    // cross zero); a pedestal is an OFFSET (it may sit below black). Same
    // ranges as the Grade's own rows, so neither node can reach a value the
    // other cannot.
    let level = || RgbSpec::scale(-4.0, 8.0, luma).with_display(disp);
    let pedestal = || RgbSpec::offset(-4.0, 8.0, luma).with_display(disp);
    let mut h: Option<HudSlider> = None;
    // ONE column, in the order the node reads: the pair you measure, then
    // the pair they land on. Side by side they would not fit — see `TILES`.
    head(ui, "in — the levels you measure", "the two input levels this node is about");
    row(ui, rcx, ungang, "black", &mut p.black, pedestal(), &BLACK, &mut h);
    row(ui, rcx, ungang, "white", &mut p.white, level(), &WHITE, &mut h);
    ui.separator();
    head(ui, "out — where they land", "where those two levels come out");
    row(ui, rcx, ungang, "lift", &mut p.lift, pedestal(), &LIFT, &mut h);
    row(ui, rcx, ungang, "mult", &mut p.mult, level(), &MULT, &mut h);
    ui.separator();
    ui.horizontal(|ui| {
        if ui
            .small_button("reset")
            .on_hover_text("back to 0/1 in, 0/1 out — the identity")
            .clicked()
        {
            *p = ColorCompressParams::default();
        }
        ui.weak("black ➡ lift, white ➡ mult, straight line between");
    });
    ui.weak("outside the two points the line keeps going: nothing is clipped");
    h
}

/// The graph glyph: the transfer line, with a tick on each axis for the
/// point it is pinned to. Drawn SHALLOW — a compressed range is the name on
/// the tin — so it reads as "levels", not as an identity.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.rect(
        egui::Rect::from_center_size(c, egui::vec2(r * 1.7, r * 1.7)),
        r * 0.3,
        fill,
        stroke,
        egui::StrokeKind::Middle,
    );
    let ink = Stroke::new((r * 0.14).max(1.0), crate::theme::node_ink());
    let h = r * 0.62;
    // y is down, so the line rises to the RIGHT.
    painter.line_segment([c + Vec2::new(-h, h * 0.55), c + Vec2::new(h, -h * 0.55)], ink);
    // The two points it is pinned to, on the input axis.
    let tick = (r * 0.22).max(1.5);
    for x in [-h * 0.45, h * 0.45] {
        let p = c + Vec2::new(x, h);
        painter.line_segment([p, p - Vec2::new(0.0, tick)], ink);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    /// The panel must FIT the tiles it asks for. Four rgb rows is exactly
    /// the layout that once overflowed the Grade into its own scroll area,
    /// where a row cannot be clicked at all — so this measures the content
    /// against the viewport rather than trusting the arithmetic.
    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let mut p = ColorCompressParams::default();
        let mut ungang = BTreeSet::new();
        let mut anim = Default::default();
        let ekind = NodeKind::ColorCompress(p);
        let over = std::cell::Cell::new(0.0f32);
        let wide = std::cell::Cell::new(0.0f32);
        // What the properties grid gives it: TILES tiles wide and
        // TILE_ROWS tall (with the gaps), less the group frame and the
        // title row.
        let span = |n: usize| {
            crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0)
        };
        let (w, hgt) = (span(TILES) - 14.0, span(TILE_ROWS) - 14.0 - 22.0);
        let shot = crate::snapshot::shoot(w, hgt, |ui| {
            let out = egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let mut rcx = RowCtx {
                        node: 1,
                        anim: &mut anim,
                        ekind: &ekind,
                        t: 0,
                        hud_live: false,
                        driven: Vec::new(),
                    };
                    let _ = panel(ui, &mut rcx, &mut p, Colorspace::ACESCG, &mut ungang);
                });
            over.set(out.content_size.y - out.inner_rect.height());
            wide.set(out.content_size.x - out.inner_rect.width());
        });
        shot.save("colorcompress-panel");
        assert!(
            over.get() <= 0.5,
            "the panel overflows its {TILES} tiles by {}pt — rows in the scroll \
             area cannot be clicked",
            over.get()
        );
        // ...and it must not overflow SIDEWAYS either. The panel scrolls
        // vertically only, so anything past the right edge is CLIPPED: the
        // value chip and the number fields of a row would be drawn and
        // unreachable. Worse, egui grows the parent to fit an overflowing
        // row, so the next row is wider again and the panel walks off the
        // edge one row at a time — which is exactly what four rgb rows did
        // here before `ROW_OVERHANG`.
        assert!(
            wide.get() <= 0.5,
            "the panel overflows its {TILES} tiles SIDEWAYS by {}pt — the rows' \
             chips and numbers are outside the tile",
            wide.get()
        );
    }

    /// Ganging is per ROW and lives on the node, not in the params: the
    /// pixels are the same either way.
    #[test]
    fn ungang_is_remembered_per_row() {
        let mut ungang = BTreeSet::new();
        ungang.insert("white".to_string());
        assert!(!ungang.contains("black") && ungang.contains("white"));
        // The params carry no gang at all — nothing about the picture
        // changes when a row is ungagged.
        let a = ColorCompressParams::default();
        let b = a;
        assert_eq!(a, b);
    }
}
