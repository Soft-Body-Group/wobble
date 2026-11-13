//! The LTX HDR node's panel, channels and glyph. The params live in
//! `engine::nodes::ltxhdr`; the runner (install, fetch, render, process,
//! adopt) in `crate::ltxhdr`.

use crate::anim::Channel;
use crate::ltxhdr::{weights_bytes, Action, View, ViewState, LICENCE_URL, MODEL_PAGES, REPO, WHERE};
use crate::theme;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Ui, Vec2};
use engine::graph::{LtxHdrParams, LtxOffload, LtxScale};

pub const CHANNELS: &[Channel] = &[Channel::LtxClip, Channel::LtxSoftness];

pub const TILES: usize = 2;
pub const TILE_ROWS: usize = 2;

pub fn get(p: &LtxHdrParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::LtxClip => Some(p.clip),
        Channel::LtxSoftness => Some(p.softness),
        _ => None,
    }
}

pub fn set(p: &mut LtxHdrParams, ch: Channel, v: f32) {
    match ch {
        Channel::LtxClip => p.clip = v,
        Channel::LtxSoftness => p.softness = v,
        _ => {}
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    match ch {
        Channel::LtxClip => Some("ltx hdr clip level"),
        Channel::LtxSoftness => Some("ltx hdr softness"),
        _ => None,
    }
}

fn gb(b: u64) -> String {
    format!("{:.1} GB", b as f64 / 1e9)
}

/// The properties panel. The click it hands back is acted on after the
/// node borrow ends.
pub fn panel(ui: &mut Ui, rcx: &mut RowCtx, node: u64, p: &mut LtxHdrParams, v: &View) -> (Option<Action>, Option<HudSlider>) {
    let mut act = None;
    ui.horizontal_wrapped(|ui| {
        ui.weak("LTX-2.5 SDR to HDR by Lightricks ·");
        ui.hyperlink_to("LTX-2 Community Licence", LICENCE_URL).on_hover_text(
            "free commercial use under $10M annual revenue, with use restrictions and a provenance clause: read it before delivering anything made with it. Tagged, and a Write warns",
        );
    });
    // --- install --------------------------------------------------------
    if v.installed {
        ui.weak(format!("installed at {}", v.dir)).on_hover_text(WHERE);
    } else if let Some(line) = &v.installing {
        ui.horizontal(|ui| {
            ui.add(egui::Spinner::new().size(12.0));
            ui.add(egui::Label::new(egui::RichText::new(format!("installing… {line}")).weak()).truncate());
        });
    } else {
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(theme::error(), "not installed");
            let b = ui.add_enabled(v.uv, egui::Button::new("install…")).on_hover_text(format!(
                "fetch {REPO} (pinned) into {} and uv sync there (torch: a few GB). Nothing is fetched until you press this. {WHERE}",
                v.dir
            ));
            if b.clicked() {
                act = Some(Action::Install);
            }
            if !v.uv {
                ui.weak("needs uv: brew install uv");
            }
        });
    }
    if let Some(Err(e)) = &v.install_result {
        ui.colored_label(theme::error(), format!("install failed: {e}"));
    }
    // --- weights (gated) --------------------------------------------------
    if v.weights {
        ui.weak("weights fetched");
    } else if let Some((line, done, total)) = &v.fetching {
        ui.horizontal(|ui| {
            ui.add(egui::Spinner::new().size(12.0));
            ui.add(egui::Label::new(egui::RichText::new(format!("fetching {done}/{total}… {line}")).weak()).truncate());
        });
    } else {
        ui.horizontal_wrapped(|ui| {
            let b = ui.add_enabled(v.installed, egui::Button::new(format!("fetch weights… ({})", gb(weights_bytes())))).on_hover_text(
                "GATED on Hugging Face: first accept the LTX-2 Community Licence on BOTH model pages (links beside this), then log in once (hf auth login, or set HF_TOKEN). The fetch uses your login; nothing is downloaded until you press this",
            );
            if b.clicked() {
                act = Some(Action::Fetch);
            }
            ui.hyperlink_to("base", MODEL_PAGES[0]);
            ui.hyperlink_to("HDR LoRA", MODEL_PAGES[1]);
        });
    }
    if let Some(Err(e)) = &v.fetch_result {
        ui.add(egui::Label::new(egui::RichText::new(format!("fetch failed: {e}")).color(theme::error())).wrap());
    }
    // --- the composite (uniforms: no re-run) ------------------------------
    let mut h = widgets::slider(ui, rcx, "clip", &mut p.clip, 0.05, 4.0, &[Channel::LtxClip]);
    h = h.or(widgets::slider(ui, rcx, "softness", &mut p.softness, 0.0, 0.5, &[Channel::LtxSoftness]));
    ui.checkbox(&mut p.raw, "raw (the result everywhere)").on_hover_text(
        "off: the result only where the plate nears the clip, so the pixels the plate recorded properly stay exactly as they were (the model's VAE shifts every pixel a little). On: every pixel through the model",
    );
    // --- the recipe (a new directory) -------------------------------------
    ui.horizontal(|ui| {
        ui.label("size");
        egui::ComboBox::from_id_salt(("ltx-scale", node)).width(90.0).selected_text(p.scale.label()).show_ui(ui, |ui| {
            for s in LtxScale::ALL {
                ui.selectable_value(&mut p.scale, s, s.label());
            }
        }).response.on_hover_text("run the model on a smaller copy; the result is stretched back over the plate (a highlight is smooth). What makes HD fit a smaller GPU or a Mac");
        ui.label("seed");
        ui.add(egui::DragValue::new(&mut p.seed).speed(1.0));
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut p.high_quality, "high quality").on_hover_text("twice the frames internally, every other one kept: steadier, ~2x the time");
        ui.checkbox(&mut p.keyframes, "keyframes").on_hover_text("seam keyframes (the tool's default); off is plain IC-LoRA");
    });
    ui.horizontal(|ui| {
        ui.label("offload");
        egui::ComboBox::from_id_salt(("ltx-offload", node)).width(130.0).selected_text(p.offload.label()).show_ui(ui, |ui| {
            for o in LtxOffload::ALL {
                ui.selectable_value(&mut p.offload, o, o.label());
            }
        }).response.on_hover_text("where the weights sit while it runs: cpu or disk to fit less memory, slower. Not part of the result's key");
    });
    // --- the run --------------------------------------------------------
    let running = matches!(v.state, ViewState::Rendering(..) | ViewState::Running(..));
    match &v.state {
        ViewState::Idle => {}
        ViewState::Rendering(done, total) => {
            let frac = if *total > 0 { *done as f32 / *total as f32 } else { 0.0 };
            ui.add(egui::ProgressBar::new(frac).text(format!("rendering the input {done}/{total}")));
        }
        ViewState::Running(done, total) => {
            let frac = if *total > 0 { *done as f32 / *total as f32 } else { 0.0 };
            let words = if *done > 0 { format!("writing {done}/{total}") } else { "generating (the whole clip at once)…".into() };
            ui.add(egui::ProgressBar::new(frac).text(words));
        }
        ViewState::Failed(e) => {
            ui.add(egui::Label::new(egui::RichText::new(format!("failed: {e}")).color(theme::error())).wrap());
        }
        ViewState::Done(s) => {
            ui.weak(s);
        }
    }
    match v.loaded {
        Some((n, true)) => {
            ui.monospace(format!("HDR — up to date  ({n} frames)"));
        }
        Some((_, false)) => {
            ui.colored_label(theme::error(), "STALE — the input or a setting changed (still playing the old result)");
        }
        None if !running => {
            ui.weak(if v.on_disk { "a result for this input is on disk — load it" } else { "not run: passes the plate through" });
        }
        None => {}
    }
    ui.horizontal(|ui| {
        if running {
            if ui.button("cancel").clicked() {
                act = Some(Action::Cancel);
            }
        } else {
            let label = match v.loaded {
                Some((_, true)) => "re-run",
                _ if v.on_disk => "load result",
                _ => "run",
            };
            let can = v.has_input && ((v.installed && v.weights) || v.on_disk) && !v.busy_elsewhere;
            let b = ui.add_enabled(can, egui::Button::new(label)).on_hover_text(
                "renders the input over its range (ACEScg), runs LTX-2.5's SDR to HDR over the whole clip, then composites the result here where the plate clipped. Keyed on everything upstream: an edit shows as STALE, nothing re-runs behind your back",
            );
            if b.clicked() {
                act = Some(Action::Run);
            }
            if v.busy_elsewhere {
                ui.weak("another LTX HDR is running");
            }
            if matches!(v.state, ViewState::Failed(_) | ViewState::Done(_)) && ui.small_button("dismiss").clicked() {
                act = Some(Action::Dismiss);
            }
        }
    });
    if !v.log.is_empty() {
        egui::ScrollArea::vertical().max_height(60.0).id_salt(("ltx-log", node)).show(ui, |ui| {
            for l in &v.log {
                ui.add(egui::Label::new(egui::RichText::new(l).weak().small()).truncate());
            }
        });
    }
    (act, h)
}

/// A sun over a horizon, its rays running long: the highlight, extended.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let sq = egui::Rect::from_center_size(c, Vec2::splat(r * 1.7));
    painter.rect(sq, r * 0.35, fill, stroke, egui::StrokeKind::Middle);
    let ink = theme::node_ink();
    let sun = c + Vec2::new(0.0, r * 0.18);
    painter.circle_filled(sun, r * 0.3, ink);
    for k in 0..5 {
        let a = std::f32::consts::PI * (1.0 + k as f32 / 4.0);
        let d = Vec2::new(a.cos(), a.sin());
        painter.line_segment([sun + d * r * 0.42, sun + d * r * 0.72], Stroke::new((r * 0.1).max(1.0), ink));
    }
    let y = c.y + r * 0.5;
    painter.line_segment([Pos2::new(c.x - r * 0.7, y), Pos2::new(c.x + r * 0.7, y)], Stroke::new((r * 0.1).max(1.0), ink));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let span = |n: usize| crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0);
        let (w, hgt) = (span(TILES) - 14.0, span(TILE_ROWS) - 14.0 - 22.0);
        for (name, state, installed) in [
            ("ltxhdr-panel-install", ViewState::Failed("LTX-2 is not installed: press install…".into()), false),
            ("ltxhdr-panel-running", ViewState::Running(22, 48), true),
        ] {
            let v = View {
                dir: "/Users/someone/.wobble/LTX-2".into(),
                installed,
                uv: true,
                installing: None,
                install_result: (!installed).then(|| Err("uv sync: exited with status 1".into())),
                weights: false,
                fetching: None,
                fetch_result: Some(Err("Lightricks/LTX-2.5 is gated: accept the licence at https://huggingface.co/Lightricks/LTX-2.5 and log in (hf auth login, or set HF_TOKEN), then fetch again".into())),
                state,
                log: vec!["loading LTX-2.5 + the HDR IC-LoRA on Apple Silicon (MPS)".into(); 3],
                has_input: true,
                loaded: Some((48, false)),
                on_disk: false,
                busy_elsewhere: false,
            };
            let mut p = LtxHdrParams::new(0);
            let mut anim = std::collections::BTreeMap::new();
            let ekind = engine::graph::NodeKind::LtxHdr(p.clone());
            let over = std::cell::Cell::new(0.0f32);
            let wide = std::cell::Cell::new(0.0f32);
            let shot = crate::snapshot::shoot(w, hgt, |ui| {
                let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    let mut rcx = RowCtx { node: 1, anim: &mut anim, ekind: &ekind, t: 0, hud_live: false, driven: Vec::new() };
                    let _ = panel(ui, &mut rcx, 1, &mut p, &v);
                });
                over.set(out.content_size.y - out.inner_rect.height());
                wide.set(out.content_size.x - out.inner_rect.width());
            });
            shot.save(name);
            assert!(over.get() <= 0.5, "{name} overflows its tiles by {}pt", over.get());
            assert!(wide.get() <= 0.5, "{name} overflows sideways by {}pt", wide.get());
        }
    }
}
