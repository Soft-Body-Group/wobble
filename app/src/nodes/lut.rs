//! LUT's app half: the panel, the glyph and the channel bindings. The
//! parsers are `engine::lut`; the shaping, the WGSL and the params are
//! `engine::nodes::lut`.

use crate::anim::Channel;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use engine::color::Colorspace;
use engine::graph::{LutOut, LutParams, LutShaper, LutSource};
use engine::invview::InverseView;
use engine::lut::Interp;
use engine::viewtf::{Eotf, Gamut};

pub const CHANNELS: &[Channel] = &[
    Channel::LutMix,
    Channel::LutLogMin,
    Channel::LutLogMax,
    Channel::LutDither,
    Channel::LutLimit,
];

/// The path field and three space pickers want the width of two tiles.
pub const TILES: usize = 2;

pub fn get(p: &LutParams, ch: Channel) -> Option<f32> {
    Some(match ch {
        Channel::LutMix => p.mix,
        Channel::LutLogMin => p.log_min,
        Channel::LutLogMax => p.log_max,
        Channel::LutDither => p.dither,
        Channel::LutLimit => p.limit,
        _ => return None,
    })
}

pub fn set(p: &mut LutParams, ch: Channel, v: f32) {
    match ch {
        Channel::LutMix => p.mix = v,
        Channel::LutLogMin => p.log_min = v,
        Channel::LutLogMax => p.log_max = v,
        Channel::LutDither => p.dither = v.max(0.0),
        Channel::LutLimit => p.limit = v.max(0.0),
        _ => {}
    }
}

/// The curve editor's name for one of this node's channels.
pub fn label(ch: Channel) -> Option<&'static str> {
    Some(match ch {
        Channel::LutMix => "lut mix",
        Channel::LutLogMin => "lut log2 min",
        Channel::LutLogMax => "lut log2 max",
        Channel::LutDither => "inverse view dither",
        Channel::LutLimit => "inverse view limit",
        _ => return None,
    })
}

/// What the loaded file holds, in words.
fn describe(l: &engine::lut::Lut) -> String {
    let mut parts = Vec::new();
    if let Some(s) = &l.shaper {
        parts.push(format!("1D {}", s.data.len()));
    }
    if let Some(c) = &l.cube {
        parts.push(format!("3D {}³", c.size));
    }
    format!(".{} · {}", l.format, parts.join(" + "))
}

/// The path, typed or picked. Typing is buffered and lands on Enter or
/// when the field lets go, so a half-typed path is never a render.
fn path_row(ui: &mut egui::Ui, node: u64, path: &mut String) {
    let id = egui::Id::new(("lut-path", node));
    let buf_id = id.with("buf");
    // Read before any data closure: egui's lock is not re-entrant.
    let editing = ui.memory(|m| m.has_focus(id));
    let held: Option<String> = if editing { ui.data(|d| d.get_temp(buf_id)) } else { None };
    let mut buf = held.unwrap_or_else(|| path.clone());
    ui.horizontal(|ui| {
        ui.add_sized([widgets::LABEL_W, 18.0], egui::Label::new("file"));
        let w = (ui.available_width() - 30.0).max(40.0);
        let r = ui.add(
            egui::TextEdit::singleline(&mut buf)
                .id(id)
                .desired_width(w)
                .hint_text(".cube .3dl .csp .spi1d .spi3d"),
        );
        if r.changed() || r.gained_focus() {
            ui.data_mut(|d| d.insert_temp(buf_id, buf.clone()));
        }
        if r.lost_focus() {
            let walked_away = ui.input(|i| i.key_pressed(egui::Key::Escape));
            if !walked_away && buf != *path {
                *path = buf.trim().to_string();
            }
            ui.data_mut(|d| d.remove::<String>(buf_id));
        }
        if ui.small_button("…").on_hover_text("pick a LUT file").clicked() {
            if let Some(f) = crate::filedialog::FileDialog::new()
                .add_filter("LUT", &engine::lut::EXTS)
                .pick_file()
            {
                *path = f.to_string_lossy().into_owned();
            }
        }
    });
}

fn space_combo(ui: &mut egui::Ui, salt: (&str, u64), label: &str, cs: &mut Colorspace) {
    ui.horizontal(|ui| {
        ui.add_sized([widgets::LABEL_W, 18.0], egui::Label::new(label));
        egui::ComboBox::from_id_salt(salt).width(180.0).selected_text(cs.label()).show_ui(ui, |ui| {
            for (name, c) in Colorspace::PRESETS {
                ui.selectable_value(cs, c, name);
            }
        });
    });
}

fn combo<T: PartialEq + Copy>(
    ui: &mut egui::Ui,
    salt: (&str, u64),
    label: &str,
    v: &mut T,
    all: &[T],
    name: impl Fn(T) -> String,
) {
    ui.horizontal(|ui| {
        ui.add_sized([widgets::LABEL_W, 18.0], egui::Label::new(label));
        egui::ComboBox::from_id_salt(salt).width(180.0).selected_text(name(*v)).show_ui(ui, |ui| {
            for &o in all {
                ui.selectable_value(v, o, name(o));
            }
        });
    });
}

/// The Inverse View's panel: which rendering to undo, how the plate was
/// decoded, and the two knobs that make an 8-bit source behave.
fn inverse_panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut LutParams, v: &mut InverseView) -> Option<HudSlider> {
    let node = rcx.node;
    combo(ui, ("iv-tf", node), "undo", &mut v.transform, &InverseView::TRANSFORMS, |t| t.label().to_string());
    if !v.transform.looks().contains(&v.look) {
        v.look = engine::viewtf::Look::None;
    }
    ui.horizontal(|ui| {
        ui.add_sized([widgets::LABEL_W, 18.0], egui::Label::new("display"));
        egui::ComboBox::from_id_salt(("iv-disp", node)).width(86.0).selected_text(v.display.label()).show_ui(
            ui,
            |ui| {
                for g in Gamut::DISPLAYS {
                    ui.selectable_value(&mut v.display, g, g.label());
                }
            },
        );
        egui::ComboBox::from_id_salt(("iv-eotf", node)).width(86.0).selected_text(v.eotf.label()).show_ui(ui, |ui| {
            for c in Eotf::ALL {
                ui.selectable_value(&mut v.eotf, c, c.label());
            }
        });
    });
    if v.transform.looks().len() > 1 {
        let looks = v.transform.looks();
        combo(ui, ("iv-look", node), "look", &mut v.look, looks, |l| l.label().to_string());
    }
    space_combo(ui, ("lut-in", node), "plate as", &mut p.input);
    let line = match v.check() {
        Ok(()) => egui::RichText::new("set \"plate as\" to the Load's colour; view through the same transform")
            .small()
            .weak(),
        Err(e) => egui::RichText::new(format!("{e}: passing through")).small().color(crate::theme::error()),
    };
    ui.horizontal(|ui| {
        ui.add_space(widgets::LABEL_W + 8.0);
        ui.add(egui::Label::new(line).truncate());
    });
    let mut h = widgets::slider(ui, rcx, "mix", &mut p.mix, 0.0, 1.0, &[Channel::LutMix]);
    h = h.or(widgets::slider(ui, rcx, "dither", &mut p.dither, 0.0, 4.0, &[Channel::LutDither]));
    h = h.or(widgets::slider(ui, rcx, "limit", &mut p.limit, 0.0, 200.0, &[Channel::LutLimit]));
    h
}

/// The properties panel.
pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut LutParams) -> Option<HudSlider> {
    // The source picks the panel: a file LUT, or a view transform undone.
    let mut inv = p.inverse().is_some();
    ui.horizontal(|ui| {
        ui.add_sized([widgets::LABEL_W, 18.0], egui::Label::new("table"));
        ui.selectable_value(&mut inv, false, "file");
        ui.selectable_value(&mut inv, true, "inverse view");
    });
    if inv != p.inverse().is_some() {
        *p = if inv {
            LutParams { path: p.path.clone(), ..LutParams::inverse_view() }
        } else {
            LutParams { path: p.path.clone(), ..Default::default() }
        };
    }
    if let LutSource::InverseView(mut v) = p.source {
        let h = inverse_panel(ui, rcx, p, &mut v);
        p.source = LutSource::InverseView(v);
        return h;
    }
    path_row(ui, rcx.node, &mut p.path);
    let status = p.status();
    let line = match &status {
        Ok(l) => egui::RichText::new(describe(l)).small().weak(),
        Err(e) if e.is_empty() => egui::RichText::new("no file: the picture passes through").small().weak(),
        Err(e) => egui::RichText::new(format!("{e} — passing through")).small().color(crate::theme::error()),
    };
    ui.horizontal(|ui| {
        ui.add_space(widgets::LABEL_W + 8.0);
        ui.add(egui::Label::new(line).truncate());
    });
    let mut h = widgets::slider(ui, rcx, "mix", &mut p.mix, 0.0, 1.0, &[Channel::LutMix]);
    // The shaping is shader shape, so these are combos, not sliders.
    space_combo(ui, ("lut-in", rcx.node), "expects", &mut p.input);
    ui.horizontal(|ui| {
        ui.add_sized([widgets::LABEL_W, 18.0], egui::Label::new("shaper"));
        egui::ComboBox::from_id_salt(("lut-shaper", rcx.node))
            .width(180.0)
            .selected_text(p.shaper.label())
            .show_ui(ui, |ui| {
                for s in LutShaper::ALL {
                    ui.selectable_value(&mut p.shaper, s, s.label());
                }
            })
            .response
            .on_hover_text(
                "log2: the expected space's primaries, then log2 about 18% grey \
                 over the stops below — for LUTs built on an lg2 allocation",
            );
    });
    if p.shaper == LutShaper::Log2 {
        h = h.or(widgets::drags(
            ui,
            rcx,
            "log2 stops",
            {
                let LutParams { log_min, log_max, .. } = p;
                &mut [log_min, log_max]
            },
            0.05,
            (-20.0, 20.0),
            &[Channel::LutLogMin, Channel::LutLogMax],
        ));
    }
    let out_label = match p.output {
        LutOut::Same => "as it went in".to_string(),
        LutOut::Raw => "raw (display values)".to_string(),
        LutOut::Space(cs) => cs.label(),
    };
    ui.horizontal(|ui| {
        ui.add_sized([widgets::LABEL_W, 18.0], egui::Label::new("output"));
        egui::ComboBox::from_id_salt(("lut-out", rcx.node))
            .width(180.0)
            .selected_text(out_label)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut p.output, LutOut::Same, "as it went in")
                    .on_hover_text("undo the shaping: a look stays scene-linear");
                ui.selectable_value(&mut p.output, LutOut::Raw, "raw (display values)")
                    .on_hover_text("the table's numbers as they are: view with the view transform off");
                ui.separator();
                for (name, c) in Colorspace::PRESETS {
                    ui.selectable_value(&mut p.output, LutOut::Space(c), name);
                }
            });
    });
    if status.as_ref().is_ok_and(|l| l.cube.is_some()) {
        combo(ui, ("lut-interp", rcx.node), "interp", &mut p.interp, &Interp::ALL, |i| i.label().to_string());
        ui.horizontal(|ui| {
            ui.add_space(widgets::LABEL_W + 8.0);
            ui.add(
                egui::Label::new(
                    egui::RichText::new("match Resolve's 3D LUT interpolation setting; out of range clamps")
                        .small()
                        .weak(),
                )
                .truncate(),
            );
        });
    } else {
        ui.weak("values outside the table's range clamp");
    }
    h
}

/// The graph glyph: a cube in outline — a 3D table.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.11).max(1.0), crate::theme::node_ink());
    let s = r * 0.36;
    let d = Vec2::new(r * 0.2, -r * 0.2);
    let f = [Vec2::new(-s, -s), Vec2::new(s, -s), Vec2::new(s, s), Vec2::new(-s, s)]
        .map(|v| c + v - d * 0.5);
    let b = f.map(|p| p + d);
    for i in 0..4 {
        painter.line_segment([f[i], f[(i + 1) % 4]], ink);
        painter.line_segment([f[i], b[i]], ink);
    }
    painter.line_segment([b[0], b[1]], ink);
    painter.line_segment([b[1], b[2]], ink);
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    #[test]
    fn the_channels_round_trip() {
        let mut p = LutParams::default();
        for (i, ch) in CHANNELS.iter().enumerate() {
            set(&mut p, *ch, i as f32 + 0.5);
            assert_eq!(get(&p, *ch), Some(i as f32 + 0.5));
            assert!(label(*ch).is_some());
        }
    }

    /// The panel fits its tiles, both ways, with the log2 row showing and
    /// an error line as long as a real one.
    #[test]
    fn the_panel_fits_its_tiles() {
        fits(LutParams {
            path: "/a/very/long/path/that/does/not/exist/anywhere/at/all/look.cube".into(),
            shaper: LutShaper::Log2,
            ..Default::default()
        });
        // A cube that loads, so the interpolation row shows.
        let cube = std::env::temp_dir().join(format!("wobble-lut-panel-{}.cube", std::process::id()));
        std::fs::write(&cube, "LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n").unwrap();
        fits(LutParams { path: cube.to_string_lossy().into_owned(), shaper: LutShaper::Log2, ..Default::default() });
        let _ = std::fs::remove_file(&cube);
        // The Inverse View, with AgX's look row showing.
        let mut iv = LutParams::inverse_view();
        if let LutSource::InverseView(v) = &mut iv.source {
            v.transform = engine::viewtf::ViewTransform::AgX;
        }
        fits(iv);
    }

    fn fits(mut p: LutParams) {
        let mut anim = Default::default();
        let ekind = NodeKind::Lut(p.clone());
        let (over, wide) = (std::cell::Cell::new(0.0f32), std::cell::Cell::new(0.0f32));
        let t = crate::widgets::TILE;
        let w = t * TILES as f32 + crate::widgets::TILE_GAP * (TILES as f32 - 1.0);
        let shot = crate::snapshot::shoot(w - 14.0, t - 14.0 - 22.0, |ui| {
            let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let mut rcx = RowCtx {
                    node: 1,
                    anim: &mut anim,
                    ekind: &ekind,
                    t: 0,
                    hud_live: false,
                    driven: Vec::new(),
                };
                let _ = panel(ui, &mut rcx, &mut p);
            });
            over.set(out.content_size.y - out.inner_rect.height());
            wide.set(out.content_size.x - out.inner_rect.width());
        });
        shot.save("lut-panel");
        assert!(over.get() <= 0.5, "overflows its tiles DOWN by {}pt", over.get());
        assert!(wide.get() <= 0.5, "overflows its tiles SIDEWAYS by {}pt", wide.get());
    }
}
