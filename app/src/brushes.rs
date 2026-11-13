//! Brush presets: a named brush — mode, size, edge, shape, opacity, blend
//! and how the pen's pressure shapes it — kept app-wide in prefs, so a
//! brush you like is there in every comp. Colour is NOT part of a brush
//! (that is the colour picker's palette), and neither is a clone anchor.
//!
//! The row works like the colour picker's palette: click a chip to use it,
//! shift-click to overwrite it with the brush as it is now, alt-click to
//! remove it, `+` to keep the current brush as a new one.

use engine::graph::{BrushMode, PaintBlend, PressureMap};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BrushPreset {
    pub name: String,
    pub mode: BrushMode,
    pub radius: f32,
    pub soft: f32,
    pub opacity: f32,
    pub aspect: f32,
    pub angle: f32,
    /// The mode's second knob (blur px, drag px, stops); unused by colour
    /// and clone, and then left at whatever the brush last had.
    #[serde(default)]
    pub strength: Option<f32>,
    #[serde(default)]
    pub blend: PaintBlend,
    #[serde(default)]
    pub press: PressureMap,
}

impl BrushPreset {
    fn new(name: &str, mode: BrushMode, radius: f32, soft: f32, opacity: f32, press: PressureMap) -> Self {
        Self {
            name: name.into(),
            mode,
            radius,
            soft,
            opacity,
            aspect: 1.0,
            angle: 0.0,
            strength: None,
            blend: PaintBlend::Normal,
            press,
        }
    }
}

/// What a fresh install starts with. Each one is a different answer to
/// "what does pressure do", which is most of what makes a brush feel like
/// itself.
pub fn builtins() -> Vec<BrushPreset> {
    let p = |size, opacity, gamma| PressureMap { size, opacity, gamma };
    vec![
        // Hard and pressure-thin, like a pencil: size alone.
        BrushPreset::new("pencil", BrushMode::Color, 4.0, 0.1, 1.0, p(1.0, 0.0, 1.0)),
        // Ink: size, with a firm curve so a light touch still lays a line.
        BrushPreset::new("ink", BrushMode::Color, 10.0, 0.2, 1.0, p(0.8, 0.0, 0.6)),
        // Airbrush: soft and wide, pressure is FLOW (opacity), not size.
        BrushPreset::new("airbrush", BrushMode::Color, 80.0, 1.0, 0.3, p(0.0, 1.0, 1.5)),
        // A matte brush: hard, opaque, constant — what a roto touch-up wants.
        BrushPreset::new("matte", BrushMode::Color, 30.0, 0.15, 1.0, PressureMap::OFF),
        BrushPreset {
            blend: PaintBlend::Erase,
            ..BrushPreset::new("eraser", BrushMode::Color, 30.0, 0.5, 1.0, p(0.5, 0.0, 1.0))
        },
        // Dust busting: a small clone that tapers.
        BrushPreset::new("dust", BrushMode::Clone, 8.0, 0.6, 1.0, p(0.6, 0.0, 1.0)),
        BrushPreset {
            strength: Some(0.3),
            ..BrushPreset::new("dodge", BrushMode::Dodge, 60.0, 1.0, 0.5, p(0.0, 1.0, 1.0))
        },
    ]
}

/// What a click on the row asked for.
pub enum RowAct {
    Use(usize),
    Replace(usize),
    Remove(usize),
    Keep,
}

/// The preset row. `current` is the index of a preset the brush still
/// matches exactly, drawn selected.
pub fn row(ui: &mut egui::Ui, presets: &[BrushPreset], current: Option<usize>) -> Option<RowAct> {
    let mut act = None;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 3.0;
        for (i, b) in presets.iter().enumerate() {
            let r = ui
                .selectable_label(current == Some(i), egui::RichText::new(&b.name).small())
                .on_hover_text(format!(
                    "{} · {:.0}px · {}{}\nclick use · shift-click replace with this brush · alt-click remove",
                    b.mode.name(),
                    b.radius,
                    b.blend.name(),
                    pressure_words(&b.press),
                ));
            if r.clicked() {
                let m = ui.input(|i| i.modifiers);
                act = Some(if m.alt {
                    RowAct::Remove(i)
                } else if m.shift {
                    RowAct::Replace(i)
                } else {
                    RowAct::Use(i)
                });
            }
        }
        if ui
            .small_button("+")
            .on_hover_text("keep this brush as a preset")
            .clicked()
        {
            act = Some(RowAct::Keep);
        }
    });
    act
}

/// " · pressure ➡ size" and the like, for a tooltip.
pub fn pressure_words(p: &PressureMap) -> String {
    match (p.size > 0.0, p.opacity > 0.0) {
        (false, false) => String::new(),
        (true, false) => " · pressure ➡ size".into(),
        (false, true) => " · pressure ➡ opacity".into(),
        (true, true) => " · pressure ➡ size + opacity".into(),
    }
}

/// A name for a kept brush that is not taken yet.
pub fn fresh_name(presets: &[BrushPreset], mode: BrushMode) -> String {
    let base = mode.name();
    (1..)
        .map(|n| format!("{base} {n}"))
        .find(|n| presets.iter().all(|p| &p.name != n))
        .unwrap()
}

/// The pressure response, drawn: x is pressure, the line is the radius
/// factor, the dashed one opacity. `live` marks where the pen is now.
pub fn curve_plot(ui: &mut egui::Ui, p: &PressureMap, live: Option<f32>, size: egui::Vec2) {
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    let vis = ui.visuals();
    painter.rect_filled(rect, 2.0, vis.extreme_bg_color);
    let at = |x: f32, y: f32| {
        egui::pos2(rect.left() + x * rect.width(), rect.bottom() - y * rect.height())
    };
    const N: usize = 32;
    let line = |f: &dyn Fn(f32) -> f32| -> Vec<egui::Pos2> {
        (0..=N).map(|i| i as f32 / N as f32).map(|x| at(x, f(x))).collect()
    };
    let ink = vis.text_color();
    painter.add(egui::Shape::line(line(&|x| p.factors(x).0), egui::Stroke::new(1.5, ink)));
    painter.add(egui::Shape::dashed_line(
        &line(&|x| p.factors(x).1),
        egui::Stroke::new(1.0, vis.weak_text_color()),
        4.0,
        3.0,
    ));
    if let Some(x) = live {
        let x = x.clamp(0.0, 1.0);
        painter.line_segment(
            [at(x, 0.0), at(x, 1.0)],
            egui::Stroke::new(1.0, vis.selection.stroke.color),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_have_unique_names_and_round_trip_through_prefs() {
        let b = builtins();
        let mut names: Vec<_> = b.iter().map(|p| p.name.as_str()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), b.len());
        let json = serde_json::to_string(&b).unwrap();
        let back: Vec<BrushPreset> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, b);
        // A preset saved before blend/press existed still loads.
        let old = r#"[{"name":"x","mode":"Color","radius":3.0,"soft":0.5,"opacity":1.0,"aspect":1.0,"angle":0.0}]"#;
        let back: Vec<BrushPreset> = serde_json::from_str(old).unwrap();
        assert_eq!(back[0].blend, PaintBlend::Normal);
        assert_eq!(fresh_name(&b, BrushMode::Color), "colour 1");
    }
}
