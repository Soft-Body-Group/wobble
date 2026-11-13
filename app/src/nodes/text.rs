//! Text's app half: the four pages (type, style, animate, path), the
//! channel binding and the glyph. The params, layout, animators and the
//! layer are `engine::nodes::text`, `engine::textfx` and `engine::textgpu`.
//!
//! Position and size keep their old channels (`TextX`/`TextY`/`TextSize`,
//! so old setups' curves land); every other number is `text.<name>` on
//! `Channel::Custom` (the 2dini look's precedent), animators included as
//! `text.a<i>.<name>`. Removing an animator slides the later animators'
//! curves down, so a key stays with the animator it was set on.

use crate::anim::{self, AnimCurve, Channel};
use crate::rgbslider::RgbSpec;
use crate::widgets::{self, HudSlider, PanelTab, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use engine::color::{Colorspace, Primaries, primaries_matrix};
use engine::geotext::{TextAlign, TextFont};
use engine::graph::TextParams;
use engine::nodes::text::PREFIX;
use engine::textfx::{RangeShape, TextAnimator, TextCase, TextUnit, VAlign};
use std::collections::BTreeMap;

pub const CHANNELS: &[Channel] = &[Channel::TextX, Channel::TextY, Channel::TextSize];

/// Three tiles wide, two tall: a text box, a type page's dozen rows, and
/// an animator's range and deltas side by side.
pub const TILES: usize = 3;
pub const TILE_ROWS: usize = 2;

pub fn tabs() -> Vec<PanelTab> {
    vec![
        PanelTab::new("type", "type"),
        PanelTab::new("style", "style"),
        PanelTab::new("fx", "shadow + glow"),
        PanelTab::new("animate", "animate"),
        PanelTab::new("path", "path"),
    ]
}

/// The channel of a Text number by its name after `text.`.
pub fn chan(name: &str) -> Channel {
    Channel::Custom(anim::intern(&format!("{PREFIX}{name}")))
}

/// Every Text channel this node has now, beyond [`CHANNELS`].
pub fn dyn_channels(p: &TextParams) -> Vec<Channel> {
    p.names().iter().map(|n| chan(n)).collect()
}

pub fn get(p: &TextParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::TextX => Some(p.position[0]),
        Channel::TextY => Some(p.position[1]),
        Channel::TextSize => Some(p.size),
        Channel::Custom(i) => p.get(anim::interned(i).strip_prefix(PREFIX)?),
        _ => None,
    }
}

pub fn set(p: &mut TextParams, ch: Channel, v: f32) {
    match ch {
        Channel::TextX => p.position[0] = v,
        Channel::TextY => p.position[1] = v,
        Channel::TextSize => p.size = v.max(0.0),
        Channel::Custom(i) => {
            if let Some(n) = anim::interned(i).strip_prefix(PREFIX) {
                p.set(n, v);
            }
        }
        _ => {}
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    match ch {
        Channel::TextX => Some("text x"),
        Channel::TextY => Some("text y"),
        Channel::TextSize => Some("text size"),
        _ => None,
    }
}

/// Sans / mono / a file — the 2dini Text node's font row.
pub fn font_row(ui: &mut egui::Ui, font: &mut TextFont) {
    ui.horizontal(|ui| {
        ui.add_sized(
            [widgets::LABEL_W, 16.0],
            egui::Label::new(egui::RichText::new("font").small()),
        );
        if ui
            .selectable_label(*font == TextFont::Sans, "sans")
            .clicked()
        {
            *font = TextFont::Sans;
        }
        if ui
            .selectable_label(*font == TextFont::Mono, "mono")
            .clicked()
        {
            *font = TextFont::Mono;
        }
        let file = matches!(font, TextFont::File(_));
        let name = if file { font.name() } else { "file…".into() };
        if ui
            .selectable_label(file, name)
            .on_hover_text("a .ttf / .otf on disk")
            .clicked()
        {
            if let Some(p) = crate::filedialog::FileDialog::new()
                .add_filter("font", &["ttf", "otf", "ttc"])
                .pick_file()
            {
                *font = TextFont::File(p.to_string_lossy().into_owned());
            }
        }
    });
}

/// A labelled row of choices.
fn choices<T: PartialEq + Copy>(
    ui: &mut egui::Ui,
    label: &str,
    v: &mut T,
    all: &[T],
    name: impl Fn(T) -> &'static str,
) {
    ui.horizontal(|ui| {
        ui.add_sized(
            [widgets::LABEL_W, 16.0],
            egui::Label::new(egui::RichText::new(label).small()),
        );
        for a in all {
            if ui
                .selectable_label(*v == *a, egui::RichText::new(name(*a)).small())
                .clicked()
            {
                *v = *a;
            }
        }
    });
}

struct Rows<'a, 'b> {
    rcx: &'a mut RowCtx<'b>,
    working: Colorspace,
    out: Option<HudSlider>,
    /// A number's name -> its channel (`text.<name>` here, a 2dini Text's
    /// literal index inside a dive).
    ch: &'a dyn Fn(&str) -> Channel,
}

impl Rows<'_, '_> {
    fn slider(
        &mut self,
        ui: &mut egui::Ui,
        label: &str,
        v: &mut f32,
        lo: f32,
        hi: f32,
        name: &str,
    ) {
        if let Some(h) = widgets::slider(ui, self.rcx, label, v, lo, hi, &[(self.ch)(name)]) {
            self.out = Some(h);
        }
    }

    fn xy(
        &mut self,
        ui: &mut egui::Ui,
        label: &str,
        v: &mut [f32; 2],
        speed: f32,
        hud: (f32, f32),
        names: [&str; 2],
    ) {
        let [x, y] = v;
        if let Some(h) = widgets::drags(
            ui,
            self.rcx,
            label,
            &mut [x, y],
            speed,
            hud,
            &[(self.ch)(names[0]), (self.ch)(names[1])],
        ) {
            self.out = Some(h);
        }
    }

    /// An rgb row, and its alpha when `alpha` names one.
    fn color(
        &mut self,
        ui: &mut egui::Ui,
        label: &str,
        c: &mut [f32; 4],
        head: &str,
        alpha: Option<&str>,
    ) {
        let mut rgb = [c[0], c[1], c[2]];
        let mut gang = true;
        let spec = RgbSpec::scale(0.0, 4.0, self.working.primaries.luma_weights())
            .with_display(primaries_matrix(self.working.primaries, Primaries::Rec709));
        let chans = [
            (self.ch)(&format!("{head}.r")),
            (self.ch)(&format!("{head}.g")),
            (self.ch)(&format!("{head}.b")),
        ];
        if let Some(h) = widgets::rgb_row(ui, self.rcx, label, &mut rgb, &mut gang, spec, &chans) {
            self.out = Some(h);
        }
        c[..3].copy_from_slice(&rgb);
        if let Some(a) = alpha {
            self.slider(ui, a, &mut c[3], 0.0, 1.0, &format!("{head}.a"));
        }
    }
}

/// A 2dini Text's style page: the standalone node's typography and styling
/// (anchor, kerning, italic, scale, weight, outline, drop shadow), keyed by
/// the node's literal indices (`ch` names a `GEO_STYLE_PARAMS` number). The
/// rest of a 2dini shape's look — glow, soft shadow, bevel, light — is the
/// drawing's own effects pages, which take every shape at once.
pub fn geo_style_page(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    fx: &mut engine::textfx::GeoTextFx,
    working: Colorspace,
    ch: &dyn Fn(&str) -> Channel,
) -> Option<HudSlider> {
    let mut r = Rows { rcx, working, out: None, ch };
    choices(ui, "anchor", &mut fx.valign, &VAlign::ALL, VAlign::name);
    ui.checkbox(&mut fx.kerning, egui::RichText::new("kerning").small());
    r.slider(ui, "italic", &mut fx.skew, -30.0, 30.0, "italic");
    r.xy(ui, "scale", &mut fx.scale, 0.01, (0.0, 4.0), ["scale.x", "scale.y"]);
    r.slider(ui, "weight", &mut fx.weight, 0.0, 0.15, "weight");
    r.slider(ui, "outline", &mut fx.outline, 0.0, 0.2, "outline");
    if fx.outline > 0.0 {
        r.color(ui, "outline", &mut fx.outline_color, "outline", Some("ring α"));
    }
    ui.checkbox(&mut fx.shadow, egui::RichText::new("drop shadow").small());
    if fx.shadow {
        r.xy(ui, "offset px", &mut fx.shadow_offset, 0.5, (-60.0, 60.0), ["shadow.x", "shadow.y"]);
        r.color(ui, "colour", &mut fx.shadow_color, "shadow", Some("opacity"));
    }
    ui.weak("weight and outline are ems · fill, stroke and gradient are on the shape tab · glow, a soft shadow, bevel and light are the drawing's effects pages");
    r.out
}

/// The animate page: the animator chips, "+ add" with presets, and the
/// picked animator's range (left) and what it drives (right). `ch` names
/// animator `i`'s number `s` — the Text node's `text.a<i>.<s>`, or a 2dini
/// Text's literal index. Shared by both, so the two cannot drift.
pub fn animators_page(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    anims: &mut Vec<TextAnimator>,
    node: u64,
    working: Colorspace,
    ch: &dyn Fn(usize, &str) -> Channel,
) -> Option<HudSlider> {
    let mut out: Option<HudSlider> = None;
    let nm = |s: &str| s.to_string();
    let id = picked_id(node);
    let mut pick = ui.data(|d| d.get_temp::<usize>(id)).unwrap_or(0);
    let mut remove = None;
    ui.horizontal_wrapped(|ui| {
        for (i, a) in anims.iter().enumerate() {
            let label = format!("{} {}", i + 1, a.name);
            if ui
                .selectable_label(pick == i, egui::RichText::new(label).small())
                .clicked()
            {
                pick = i;
            }
        }
        ui.menu_button(egui::RichText::new("+ add").small(), |ui| {
            for k in ["animator", "rise in", "typewriter", "wave", "jitter"] {
                if ui.button(k).clicked() {
                    anims.push(preset(k));
                    pick = anims.len() - 1;
                    ui.close();
                }
            }
        });
    });
    if anims.is_empty() {
        ui.weak("an animator selects a range of letters, words or lines and moves, scales, turns, tints or fades them — key its offset to sweep it");
    } else {
        pick = pick.min(anims.len() - 1);
        let i = pick;
        let a = &mut anims[i];
        ui.horizontal(|ui| {
            ui.checkbox(&mut a.on, "");
            ui.add(egui::TextEdit::singleline(&mut a.name).desired_width(120.0));
            if ui.small_button("remove").clicked() {
                remove = Some(i);
            }
        });
        let cof = |s: &str| ch(i, s);
        let out = &mut out;
        ui.columns(2, |cols| {
            let mut r = Rows {
                rcx: &mut *rcx,
                working,
                out: None,
                ch: &cof,
            };
            {
                let ui = &mut cols[0];
                ui.horizontal(|ui| {
                    ui.add_sized(
                        [widgets::LABEL_W, 16.0],
                        egui::Label::new(egui::RichText::new("by").small()),
                    );
                    egui::ComboBox::from_id_salt(("text-unit", node, i))
                        .selected_text(egui::RichText::new(a.unit.name()).small())
                        .width(72.0)
                        .show_ui(ui, |ui| {
                            for u in TextUnit::ALL {
                                ui.selectable_value(&mut a.unit, u, u.name());
                            }
                        });
                    egui::ComboBox::from_id_salt(("text-shape", node, i))
                        .selected_text(egui::RichText::new(a.shape.name()).small())
                        .width(80.0)
                        .show_ui(ui, |ui| {
                            for sh in RangeShape::ALL {
                                ui.selectable_value(&mut a.shape, sh, sh.name());
                            }
                        });
                });
                r.slider(ui, "start %", &mut a.start, 0.0, 100.0, &nm("start"));
                r.slider(ui, "end %", &mut a.end, 0.0, 100.0, &nm("end"));
                r.slider(ui, "offset %", &mut a.offset, -100.0, 100.0, &nm("offset"));
                if a.shape == RangeShape::Square {
                    r.slider(
                        ui,
                        "softness %",
                        &mut a.softness,
                        0.0,
                        100.0,
                        &nm("softness"),
                    );
                }
                r.slider(ui, "amount %", &mut a.amount, -100.0, 200.0, &nm("amount"));
                ui.horizontal(|ui| {
                    ui.checkbox(&mut a.random, egui::RichText::new("random order").small());
                    if a.random {
                        ui.add(egui::DragValue::new(&mut a.seed).prefix("seed "));
                    }
                });
                r.slider(ui, "wiggle", &mut a.wiggle, 0.0, 1.0, &nm("wiggle"));
                if a.wiggle > 0.0 {
                    r.slider(
                        ui,
                        "per second",
                        &mut a.wiggle_rate,
                        0.0,
                        10.0,
                        &nm("wiggle_rate"),
                    );
                }
            }
            let ui = &mut cols[1];
            let [px, py] = &mut a.position;
            if let Some(h) = widgets::drags(
                ui,
                r.rcx,
                "offset px",
                &mut [px, py],
                0.5,
                (-200.0, 200.0),
                &[chan(&nm("position.x")), chan(&nm("position.y"))],
            ) {
                r.out = Some(h);
            }
            let [sx, sy] = &mut a.scale;
            if let Some(h) = widgets::drags(
                ui,
                r.rcx,
                "scale",
                &mut [sx, sy],
                0.01,
                (0.0, 4.0),
                &[chan(&nm("scale.x")), chan(&nm("scale.y"))],
            ) {
                r.out = Some(h);
            }
            r.slider(
                ui,
                "rotation",
                &mut a.rotation,
                -180.0,
                180.0,
                &nm("rotation"),
            );
            r.slider(ui, "skew", &mut a.skew, -45.0, 45.0, &nm("skew"));
            r.slider(
                ui,
                "tracking",
                &mut a.tracking,
                -200.0,
                1000.0,
                &nm("tracking"),
            );
            r.slider(ui, "opacity", &mut a.opacity, 0.0, 1.0, &nm("opacity"));
            r.slider(ui, "blur px", &mut a.blur, 0.0, 20.0, &nm("blur"));
            r.slider(ui, "weight", &mut a.weight, -0.08, 0.15, &nm("weight"));
            let mut tint = a.tint;
            r.color(ui, "tint", &mut tint, &nm("tint"), Some("tint mix"));
            a.tint = tint;
            if r.out.is_some() {
                *out = r.out.take();
            }
        });
    }
    if let Some(i) = remove {
        remove_animator_in(anims, rcx.anim, i, ch);
        pick = i.saturating_sub(1);
    }
    ui.data_mut(|d| d.insert_temp(id, pick));
    out
}

/// Remove animator `i` of `anims`: its curves go, and the later
/// animators' curves slide down one so every key stays with the animator it
/// was set on. `ch` names the channels, as for [`animators_page`].
pub fn remove_animator_in(
    anims: &mut Vec<TextAnimator>,
    anim: &mut BTreeMap<Channel, AnimCurve>,
    i: usize,
    ch: &dyn Fn(usize, &str) -> Channel,
) {
    if i >= anims.len() {
        return;
    }
    anims.remove(i);
    let n = anims.len() + 1;
    for name in engine::textfx::ANIM_PARAMS {
        anim.remove(&ch(i, name));
        for j in i + 1..n {
            if let Some(c) = anim.remove(&ch(j, name)) {
                anim.insert(ch(j - 1, name), c);
            }
        }
    }
}

/// The animator page's pick, per node, in egui's frame memory.
fn picked_id(node: u64) -> egui::Id {
    egui::Id::new(("text-animator", node))
}

/// Remove animator `i`: its curves go, and the later animators' curves
/// slide down one so every key stays with the animator it was set on.
#[cfg(test)]
fn remove_animator(p: &mut TextParams, anim: &mut BTreeMap<Channel, AnimCurve>, i: usize) {
    remove_animator_in(&mut p.animators, anim, i, &|j, s| {
        chan(&format!("a{j}.{s}"))
    });
}

/// A preset for a new animator: what a title designer reaches for first.
fn preset(kind: &str) -> TextAnimator {
    let base = TextAnimator {
        name: kind.into(),
        ..Default::default()
    };
    match kind {
        // Letters rise and fade in as the window sweeps left to right:
        // key `offset` from -20 to 100.
        "rise in" => TextAnimator {
            unit: TextUnit::Letters,
            shape: RangeShape::RampUp,
            start: 0.0,
            end: 20.0,
            offset: -20.0,
            position: [0.0, 40.0],
            opacity: 0.0,
            ..base
        },
        // Letters in the window are hidden: key `start` from 0 to 100 and
        // they type on in order.
        "typewriter" => TextAnimator {
            unit: TextUnit::Letters,
            opacity: 0.0,
            ..base
        },
        "wave" => TextAnimator {
            unit: TextUnit::Letters,
            shape: RangeShape::Smooth,
            start: 0.0,
            end: 30.0,
            position: [0.0, -30.0],
            scale: [1.3, 1.3],
            ..base
        },
        "jitter" => TextAnimator {
            wiggle: 1.0,
            wiggle_rate: 3.0,
            position: [6.0, 6.0],
            rotation: 8.0,
            ..base
        },
        _ => base,
    }
}

pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut TextParams,
    tab: &str,
    words: Option<&str>,
    working: Colorspace,
) -> Option<HudSlider> {
    let node = rcx.node;
    let mut r = Rows {
        rcx,
        working,
        out: None,
        ch: &chan,
    };
    match tab {
        "style" => {
            r.color(ui, "fill", &mut p.color, "color", Some("opacity"));
            r.slider(ui, "weight", &mut p.weight, -0.08, 0.15, "weight");
            r.slider(ui, "softness", &mut p.softness, 0.0, 20.0, "softness");
            r.slider(ui, "outline", &mut p.outline, 0.0, 0.2, "outline");
            if p.outline > 0.0 {
                r.color(
                    ui,
                    "outline",
                    &mut p.outline_color,
                    "outline",
                    Some("ring α"),
                );
            }
            ui.weak("weight, outline, glow and shadow softness are ems and share a quarter em");
        }
        "fx" => {
            ui.checkbox(&mut p.shadow, egui::RichText::new("drop shadow").small());
            if p.shadow {
                r.xy(
                    ui,
                    "offset",
                    &mut p.shadow_offset,
                    0.5,
                    (-60.0, 60.0),
                    ["shadow.x", "shadow.y"],
                );
                r.slider(ui, "softness", &mut p.shadow_soft, 0.0, 0.2, "shadow.soft");
                r.color(ui, "colour", &mut p.shadow_color, "shadow", Some("opacity"));
            }
            ui.checkbox(&mut p.glow, egui::RichText::new("glow").small());
            if p.glow {
                r.slider(ui, "size", &mut p.glow_size, 0.0, 0.2, "glow.size");
                r.color(ui, "colour", &mut p.glow_color, "glow", Some("opacity"));
            }
        }
        "animate" => {
            let ch = |i: usize, s: &str| chan(&format!("a{i}.{s}"));
            if let Some(h) = animators_page(ui, r.rcx, &mut p.animators, node, working, &ch) {
                r.out = Some(h);
            }
        }
        "path" => {
            ui.checkbox(&mut p.path.on, egui::RichText::new("follow a path").small());
            ui.horizontal(|ui| {
                ui.checkbox(
                    &mut p.path.perpendicular,
                    egui::RichText::new("turn with it").small(),
                );
                ui.checkbox(&mut p.path.closed, egui::RichText::new("closed").small());
                ui.checkbox(&mut p.path.reverse, egui::RichText::new("reverse").small());
            });
            r.slider(
                ui,
                "offset px",
                &mut p.path.offset,
                -2000.0,
                2000.0,
                "path.offset",
            );
            ui.horizontal(|ui| {
                if ui
                    .small_button("arc")
                    .on_hover_text("a half circle under the text")
                    .clicked()
                {
                    p.path.points = arc(p.position, p.size * 4.0);
                    p.path.closed = false;
                    p.path.on = true;
                }
                if ui.small_button("circle").clicked() {
                    p.path.points = circle(p.position, p.size * 3.0);
                    p.path.closed = true;
                    p.path.on = true;
                }
                if ui.small_button("wave").clicked() {
                    p.path.points = wave(p.position, p.size * 8.0, p.size);
                    p.path.closed = false;
                    p.path.on = true;
                }
                if ui.small_button("clear").clicked() {
                    p.path.points.clear();
                }
            });
            ui.label(egui::RichText::new(format!("{} points", p.path.points.len())).small());
            ui.weak("in the viewer: S-click adds a point, drag moves one, alt-drag pulls its handles, D-click removes it · the text's alignment picks where on the path it starts");
        }
        _ => {
            match words {
                // The words come off a String wired into the words port:
                // shown, not edited here (the authored text waits under
                // it for when the wire comes out).
                Some(w) => {
                    let mut shown = w.to_string();
                    ui.add_enabled(
                        false,
                        egui::TextEdit::multiline(&mut shown)
                            .desired_rows(3)
                            .desired_width(f32::INFINITY),
                    )
                    .on_disabled_hover_text("from the String wired into the words input");
                }
                None => {
                    ui.add(
                        egui::TextEdit::multiline(&mut p.text)
                            .desired_rows(3)
                            .desired_width(f32::INFINITY),
                    );
                }
            }
            font_row(ui, &mut p.font);
            ui.horizontal(|ui| {
                ui.add_sized(
                    [widgets::LABEL_W, 16.0],
                    egui::Label::new(egui::RichText::new("align").small()),
                );
                for a in TextAlign::ALL {
                    ui.selectable_value(&mut p.align, a, egui::RichText::new(a.name()).small());
                }
                ui.checkbox(&mut p.justify, egui::RichText::new("justify").small())
                    .on_hover_text("wrapped lines fill the box (set a wrap width)");
            });
            choices(ui, "anchor", &mut p.valign, &VAlign::ALL, VAlign::name);
            ui.horizontal(|ui| {
                ui.add_sized(
                    [widgets::LABEL_W, 16.0],
                    egui::Label::new(egui::RichText::new("case").small()),
                );
                for c in TextCase::ALL {
                    ui.selectable_value(&mut p.case, c, egui::RichText::new(c.name()).small());
                }
                ui.checkbox(&mut p.kerning, egui::RichText::new("kerning").small());
            });
            let [x, y] = &mut p.position;
            if let Some(h) = widgets::drags(
                ui,
                r.rcx,
                "position",
                &mut [x, y],
                1.0,
                (0.0, 1920.0),
                &[Channel::TextX, Channel::TextY],
            ) {
                r.out = Some(h);
            }
            if let Some(h) = widgets::slider(
                ui,
                r.rcx,
                "size",
                &mut p.size,
                4.0,
                400.0,
                &[Channel::TextSize],
            ) {
                r.out = Some(h);
            }
            r.slider(ui, "tracking", &mut p.tracking, -200.0, 1000.0, "tracking");
            r.slider(ui, "leading", &mut p.leading, 0.5, 3.0, "leading");
            r.slider(ui, "wrap px", &mut p.wrap, 0.0, 1920.0, "wrap");
            r.slider(ui, "rotation", &mut p.rotation, -180.0, 180.0, "rotation");
            r.xy(
                ui,
                "scale",
                &mut p.scale,
                0.01,
                (0.0, 4.0),
                ["scale.x", "scale.y"],
            );
            r.slider(ui, "italic", &mut p.skew, -30.0, 30.0, "skew");
        }
    }
    r.out
}

/// A half circle below `c` (the text arcs over its top).
fn arc(c: [f32; 2], r: f32) -> Vec<engine::textfx::PathPoint> {
    let k = 0.5523 * r;
    let cy = c[1] + r;
    vec![
        engine::textfx::PathPoint {
            p: [c[0] - r, cy],
            h_in: [0.0, k],
            h_out: [0.0, -k],
        },
        engine::textfx::PathPoint {
            p: [c[0], cy - r],
            h_in: [-k, 0.0],
            h_out: [k, 0.0],
        },
        engine::textfx::PathPoint {
            p: [c[0] + r, cy],
            h_in: [0.0, -k],
            h_out: [0.0, k],
        },
    ]
}

/// A circle round `c`, clockwise from the top.
fn circle(c: [f32; 2], r: f32) -> Vec<engine::textfx::PathPoint> {
    let k = 0.5523 * r;
    let pt = |x: f32, y: f32, hi: [f32; 2], ho: [f32; 2]| engine::textfx::PathPoint {
        p: [c[0] + x, c[1] + y],
        h_in: hi,
        h_out: ho,
    };
    vec![
        pt(0.0, -r, [-k, 0.0], [k, 0.0]),
        pt(r, 0.0, [0.0, -k], [0.0, k]),
        pt(0.0, r, [k, 0.0], [-k, 0.0]),
        pt(-r, 0.0, [0.0, k], [0.0, -k]),
    ]
}

/// A sine-ish wave through `c`, `w` wide and `a` tall.
fn wave(c: [f32; 2], w: f32, a: f32) -> Vec<engine::textfx::PathPoint> {
    let n = 5;
    let step = w / (n - 1) as f32;
    let h = step * 0.4;
    (0..n)
        .map(|i| {
            let y = c[1] + if i % 2 == 0 { a } else { -a };
            engine::textfx::PathPoint {
                p: [c[0] - w * 0.5 + step * i as f32, y],
                h_in: [-h, 0.0],
                h_out: [h, 0.0],
            }
        })
        .collect()
}

/// A viewer gesture's verdict on a Text's path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathEdit {
    Add([f32; 2]),
    Remove(usize),
    Move(usize, [f32; 2]),
    /// Point i's outgoing handle (the incoming one mirrors it).
    Pull(usize, [f32; 2]),
    /// One handle alone: (point, outgoing?, handle).
    Handle(usize, bool, [f32; 2]),
}

/// What the edit does to the path.
pub fn apply(path: &mut engine::textfx::TextPath, e: PathEdit) {
    let n = path.points.len();
    match e {
        PathEdit::Add(q) => {
            path.points.push(engine::textfx::PathPoint {
                p: q,
                ..Default::default()
            });
            path.on = true;
        }
        PathEdit::Remove(i) if i < n => {
            path.points.remove(i);
        }
        PathEdit::Move(i, q) if i < n => path.points[i].p = q,
        PathEdit::Pull(i, h) if i < n => {
            path.points[i].h_out = h;
            path.points[i].h_in = [-h[0], -h[1]];
        }
        PathEdit::Handle(i, out, h) if i < n => {
            if out {
                path.points[i].h_out = h;
            } else {
                path.points[i].h_in = h;
            }
        }
        _ => {}
    }
}

/// The path over the viewer (canvas px through `to_scr`), editable: S-click
/// adds a point at the end, a drag moves a point, alt-drag on a point pulls
/// its handles out (mirrored), a handle's dot drags alone, D-click removes
/// a point. Drags are absolute from the press, never summed deltas.
#[allow(clippy::too_many_arguments)]
pub fn gizmo(
    ui: &mut egui::Ui,
    ink: &mut crate::ink::InkShapes,
    node: u64,
    p: &TextParams,
    nav: &egui::Response,
    to_scr: &dyn Fn([f32; 2]) -> egui::Pos2,
    from_screen: &dyn Fn(egui::Pos2) -> [f32; 2],
    pointer_pos: Option<egui::Pos2>,
) -> Option<PathEdit> {
    let path = &p.path;
    let mut edit = None;
    let (line, _) = path.flatten();
    let alpha = if path.on { 0.9 } else { 0.35 };
    for w in line.windows(2) {
        ink.seg(to_scr(w[0]), to_scr(w[1]), 1.2, alpha, false);
    }
    let d_held = ui.input(|i| i.key_down(egui::Key::D));
    let alt = ui.input(|i| i.modifiers.alt);
    let origin = ui.input(|i| i.pointer.press_origin());
    let delta = |id: egui::Id, base: [f32; 2]| -> Option<[f32; 2]> {
        let pp = pointer_pos?;
        let b0 = ui.data(|d| d.get_temp::<[f32; 2]>(id)).unwrap_or(base);
        let a = from_screen(origin.unwrap_or(pp));
        let b = from_screen(pp);
        Some([b0[0] + b[0] - a[0], b0[1] + b[1] - a[1]])
    };
    for (i, q) in path.points.iter().enumerate() {
        let sp = to_scr(q.p);
        // The handles first, so a point on top of its own handle wins.
        for (out, h) in [(false, q.h_in), (true, q.h_out)] {
            if h == [0.0, 0.0] {
                continue;
            }
            let hp = to_scr([q.p[0] + h[0], q.p[1] + h[1]]);
            ink.seg(sp, hp, 0.8, alpha * 0.8, false);
            let id = ui.id().with(("text-path-h", node, i, out));
            let hr = ui.interact(
                egui::Rect::from_center_size(hp, egui::vec2(12.0, 12.0)),
                id,
                egui::Sense::drag(),
            );
            if hr.drag_started() {
                ui.data_mut(|d| d.insert_temp::<[f32; 2]>(id, h));
            }
            ink.circle_stroke(hp, 3.0, 1.0, alpha, hr.hovered());
            if hr.dragged() {
                if let Some(v) = delta(id, h) {
                    edit = Some(PathEdit::Handle(i, out, v));
                }
            }
        }
        let id = ui.id().with(("text-path-pt", node, i));
        let hr = ui.interact(
            egui::Rect::from_center_size(sp, egui::vec2(16.0, 16.0)),
            id,
            egui::Sense::click_and_drag(),
        );
        if hr.drag_started() {
            ui.data_mut(|d| d.insert_temp::<[f32; 2]>(id, if alt { [0.0, 0.0] } else { q.p }));
        }
        ink.circle_fill(sp, 4.0, alpha, hr.hovered());
        if hr.clicked() && d_held {
            edit = Some(PathEdit::Remove(i));
        } else if hr.dragged() {
            if alt {
                if let Some(v) = delta(id, [0.0, 0.0]) {
                    edit = Some(PathEdit::Pull(i, v));
                }
            } else if let Some(v) = delta(id, q.p) {
                edit = Some(PathEdit::Move(i, v));
            }
        }
    }
    let s_held = ui.input(|i| i.key_down(egui::Key::S));
    if edit.is_none() && s_held && nav.clicked_by(egui::PointerButton::Primary) {
        if let Some(pp) = nav.interact_pointer_pos() {
            edit = Some(PathEdit::Add(from_screen(pp)));
        }
    }
    edit
}

/// The graph glyph: a capital T.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.16).max(1.0), crate::theme::node_ink());
    let s = r * 0.42;
    painter.line_segment([c + Vec2::new(-s, -s), c + Vec2::new(s, -s)], ink);
    painter.line_segment([c + Vec2::new(0.0, -s), c + Vec2::new(0.0, s)], ink);
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    /// A 2dini Text's style page, everything on, fits the Text's tiles
    /// (it borrows them, `widgets::tiles_for`).
    #[test]
    fn the_2dini_style_page_fits() {
        let mut fx = engine::textfx::GeoTextFx { outline: 0.05, shadow: true, ..Default::default() };
        let mut anim = Default::default();
        let ekind = NodeKind::Text(TextParams::default());
        let over = std::cell::Cell::new(0.0f32);
        let wide = std::cell::Cell::new(0.0f32);
        let span = |n: usize| crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0);
        let (w, hgt) = (span(TILES) - 14.0, span(TILE_ROWS) - 14.0 - 22.0);
        let ch = |s: &str| {
            let k = engine::textfx::GEO_STYLE_PARAMS.iter().position(|p| *p == s).unwrap();
            Channel::Field(engine::geo::text_style_index(k) as u16)
        };
        crate::snapshot::shoot(w, hgt, |ui| {
            let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let mut rcx =
                    RowCtx { node: 1, anim: &mut anim, ekind: &ekind, t: 0, hud_live: false, driven: Vec::new() };
                let _ = geo_style_page(ui, &mut rcx, &mut fx, Colorspace::LINEAR_REC709, &ch);
            });
            over.set(out.content_size.y - out.inner_rect.height());
            wide.set(out.content_size.x - out.inner_rect.width());
        })
        .save("text-panel-geo-style");
        assert!(over.get() <= 0.5, "overflows by {}pt", over.get());
        assert!(wide.get() <= 0.5, "overflows sideways by {}pt", wide.get());
    }

    /// Every page must fit the tiles it asks for, both ways (the
    /// ColorCompress fit test) — the animate page with an animator open,
    /// the style page with every effect on.
    #[test]
    fn every_page_fits_the_tiles_it_asks_for() {
        for tab in ["type", "style", "fx", "animate", "path"] {
            let mut p = TextParams {
                shadow: true,
                glow: true,
                outline: 0.05,
                ..Default::default()
            };
            p.animators.push(TextAnimator {
                wiggle: 0.5,
                ..Default::default()
            });
            let mut anim = Default::default();
            let ekind = NodeKind::Text(p.clone());
            let over = std::cell::Cell::new(0.0f32);
            let wide = std::cell::Cell::new(0.0f32);
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
                        let _ = panel(
                            ui,
                            &mut rcx,
                            &mut p,
                            tab,
                            Some("from a prompt"),
                            Colorspace::LINEAR_REC709,
                        );
                    });
                over.set(out.content_size.y - out.inner_rect.height());
                wide.set(out.content_size.x - out.inner_rect.width());
            });
            shot.save(&format!("text-panel-{tab}"));
            assert!(
                over.get() <= 0.5,
                "{tab}: the panel overflows its tiles by {}pt",
                over.get()
            );
            assert!(
                wide.get() <= 0.5,
                "{tab}: the panel overflows its tiles SIDEWAYS by {}pt",
                wide.get()
            );
        }
    }

    #[test]
    fn removing_an_animator_slides_the_later_curves_down() {
        let mut p = TextParams::default();
        p.animators = vec![
            TextAnimator::default(),
            TextAnimator::default(),
            TextAnimator::default(),
        ];
        let mut anim: BTreeMap<Channel, AnimCurve> = BTreeMap::new();
        anim.insert(chan("a0.offset"), AnimCurve::default());
        let mut c2 = AnimCurve::default();
        c2.set_key(5, 42.0);
        anim.insert(chan("a2.offset"), c2.clone());
        remove_animator(&mut p, &mut anim, 0);
        assert_eq!(p.animators.len(), 2);
        assert!(!anim.contains_key(&chan("a2.offset")));
        assert_eq!(
            anim.get(&chan("a1.offset")),
            Some(&c2),
            "animator 3's key now drives animator 2"
        );
    }

    #[test]
    fn path_edits_do_what_they_say() {
        let mut path = engine::textfx::TextPath::default();
        apply(&mut path, PathEdit::Add([1.0, 2.0]));
        apply(&mut path, PathEdit::Add([5.0, 2.0]));
        assert!(path.on && path.points.len() == 2);
        apply(&mut path, PathEdit::Pull(0, [3.0, 0.0]));
        assert_eq!(path.points[0].h_in, [-3.0, 0.0]);
        apply(&mut path, PathEdit::Handle(0, false, [0.0, 4.0]));
        assert_eq!(
            (path.points[0].h_in, path.points[0].h_out),
            ([0.0, 4.0], [3.0, 0.0])
        );
        apply(&mut path, PathEdit::Move(1, [9.0, 9.0]));
        apply(&mut path, PathEdit::Remove(0));
        assert_eq!(path.points[0].p, [9.0, 9.0]);
        apply(&mut path, PathEdit::Remove(7));
    }

    #[test]
    fn channels_route_to_the_params() {
        let mut p = TextParams::default();
        p.animators.push(TextAnimator::default());
        set(&mut p, chan("a0.rotation"), 33.0);
        assert_eq!(p.animators[0].rotation, 33.0);
        assert_eq!(get(&p, chan("a0.rotation")), Some(33.0));
        set(&mut p, chan("outline.g"), 0.5);
        assert_eq!(p.outline_color[1], 0.5);
        assert!(dyn_channels(&p).contains(&chan("a0.tint.a")));
    }
}
