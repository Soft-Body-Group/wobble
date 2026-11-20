//! The Depth node's panel and glyph. The params live in
//! `engine::nodes::depth`; the runner (install, render, process, adopt) in
//! `crate::depth`. The download gate for the picked model's weights is
//! drawn ABOVE this panel by `properties_ui` (the node declares them in
//! `NodeKind::models`), so nothing here offers a download.

use crate::depth::{Action, View, ViewState};
use crate::theme;
use egui::{Color32, Painter, Pos2, Stroke, Ui, Vec2};
use engine::graph::{DepthModel, DepthOut, DepthParams};

pub const TILES: usize = 2;
pub const TILE_ROWS: usize = 2;

/// The picked model's weights licence, from its catalog row (the one source
/// of truth, rule 10), and whether it allows commercial use.
pub fn licence(model: DepthModel) -> (&'static str, bool) {
    match engine::models::spec(model.model_id()) {
        Some(m) => (m.license.split(" (").next().unwrap_or(m.license), m.commercial),
        None => ("unknown licence", false),
    }
}

/// The properties panel. The click it hands back is acted on after the
/// node borrow ends.
pub fn panel(ui: &mut Ui, node: u64, p: &mut DepthParams, v: &View) -> Option<Action> {
    let mut act = None;
    let (weights, commercial) = licence(p.model);
    let (who, code_licence, code_url) = v.tool.credit();
    let about = v.tool.about();
    ui.horizontal_wrapped(|ui| {
        ui.weak(format!("{who} · code"));
        ui.hyperlink_to(code_licence, code_url);
        ui.weak("· weights");
        if commercial {
            ui.weak(weights);
        } else {
            ui.colored_label(theme::error(), weights)
                .on_hover_text("research and non-commercial use only: the node is tagged, a Write below it warns, and a bake does not clear that");
        }
    });
    // --- install --------------------------------------------------------
    if v.installed {
        ui.weak(format!("installed at {}", v.dir)).on_hover_text(about);
    } else if let Some(line) = &v.installing {
        ui.horizontal(|ui| {
            ui.add(egui::Spinner::new().size(12.0));
            ui.add(egui::Label::new(egui::RichText::new(format!("installing… {line}")).weak()).truncate());
        });
    } else {
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(theme::error(), "not installed");
            let b = ui.add_enabled(v.uv, egui::Button::new("install…")).on_hover_text(format!(
                "fetch {} (pinned) into {} and uv sync there (torch: a few GB). The weights come from the button above, not from the tool. Nothing is fetched until you press this. {about}",
                v.tool.repo(),
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
    // --- settings -------------------------------------------------------
    ui.horizontal(|ui| {
        ui.label("model");
        egui::ComboBox::from_id_salt(("depth-model", node)).width(130.0).selected_text(p.model.label()).show_ui(ui, |ui| {
            for m in DepthModel::ALL {
                ui.selectable_value(&mut p.model, m, m.label());
            }
        });
        ui.label("size");
        let (m, lo, hi) = p.model.size_range();
        ui.add(egui::DragValue::new(&mut p.size).range(lo..=hi).speed(2.0)).on_hover_text(format!(
            "the network's input, short side (a multiple of {m}; never more than the plate's): larger is finer and slower; the result comes back at the plate's size"
        ));
    });
    ui.horizontal(|ui| {
        ui.label("output");
        egui::ComboBox::from_id_salt(("depth-out", node)).width(130.0).selected_text(p.output.label()).show_ui(ui, |ui| {
            for o in DepthOut::ALL {
                ui.selectable_value(&mut p.output, o, o.label());
            }
        }).response.on_hover_text("all three are on disk: switching never re-runs");
    });
    ui.weak(match (p.model.metric(), p.output) {
        (false, DepthOut::Depth) => "RELATIVE, not metres: the nearest thing in the clip reads 1, further reads more",
        (false, DepthOut::Disparity) => "relative inverse depth, as the network made it: for a Defocus with inverse depth on",
        (true, DepthOut::Depth) if p.model.pro2() => "metres along the optical axis (estimated, with a focal length estimated for the clip)",
        (true, DepthOut::Depth) => "metres (estimated: trained on driving and indoor scenes)",
        (true, DepthOut::Disparity) => "1 / metres: for a Defocus with inverse depth on",
        (_, DepthOut::Near) => "near 1, far 0, over the clip",
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
            let words = if *total > 0 { format!("depth {done}/{total}") } else { "loading the model…".into() };
            ui.add(egui::ProgressBar::new(frac).text(words));
        }
        ViewState::Failed(e) => {
            ui.colored_label(theme::error(), format!("failed: {e}"));
        }
        ViewState::Done(s) => {
            ui.weak(s);
        }
    }
    match v.loaded {
        Some((n, true)) => {
            ui.monospace(format!("depth — up to date  ({n} frames)"));
        }
        Some((_, false)) => {
            ui.colored_label(theme::error(), "STALE — the input or a setting changed (still playing the old depth)");
        }
        None if !running => {
            ui.weak(if v.on_disk { "a result for this input is on disk — load it" } else { "not run: no depth (outputs 0)" });
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
            let can = v.has_input && ((v.installed && v.weights.is_ok()) || v.on_disk) && !v.busy_elsewhere;
            let b = ui.add_enabled(can, egui::Button::new(label)).on_hover_text(format!(
                "renders the input over its range, runs {} over the clip, then plays the depth here. Keyed on everything upstream: an edit shows as STALE, nothing re-runs behind your back",
                v.tool.name()
            ));
            if b.clicked() {
                act = Some(Action::Run);
            }
            if v.busy_elsewhere {
                ui.weak("another Depth is running");
            }
            if matches!(v.state, ViewState::Failed(_) | ViewState::Done(_)) && ui.small_button("dismiss").clicked() {
                act = Some(Action::Dismiss);
            }
        }
    });
    if let (Err(e), false) = (&v.weights, running) {
        ui.add(egui::Label::new(egui::RichText::new(e).weak()).truncate());
    }
    if !v.log.is_empty() {
        egui::ScrollArea::vertical().max_height(60.0).id_salt(("depth-log", node)).show(ui, |ui| {
            for l in &v.log {
                ui.add(egui::Label::new(egui::RichText::new(l).weak().small()).truncate());
            }
        });
    }
    act
}

/// Three stacked planes receding: near big and bright, far small and dim.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let sq = egui::Rect::from_center_size(c, Vec2::splat(r * 1.7));
    painter.rect(sq, r * 0.35, fill, stroke, egui::StrokeKind::Middle);
    let ink = theme::node_ink();
    for (i, (w, dy)) in [(0.95, 0.42), (0.68, 0.02), (0.42, -0.34)].into_iter().enumerate() {
        let a = 1.0 - i as f32 * 0.3;
        let y = c.y + r * dy;
        let s = Stroke::new((r * 0.14 * a).max(1.0), ink.gamma_multiply(a));
        painter.line_segment([Pos2::new(c.x - r * w * 0.5, y), Pos2::new(c.x + r * w * 0.5, y)], s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_licence_line_follows_the_picked_model() {
        assert_eq!(licence(DepthModel::Small), ("Apache-2.0", true));
        assert_eq!(licence(DepthModel::SmallMetric), ("Apache-2.0", true));
        assert_eq!(licence(DepthModel::Large), ("CC-BY-NC-4.0", false));
        assert_eq!(licence(DepthModel::LargeMetric), ("CC-BY-NC-4.0", false));
        for m in [DepthModel::Pro2Small, DepthModel::Pro2Base, DepthModel::Pro2Large, DepthModel::Pro2Huge] {
            assert_eq!(licence(m), ("Apple ML Research Model licence, research only", false), "{m:?}");
        }
    }

    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let span = |n: usize| crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0);
        let (w, hgt) = (span(TILES) - 14.0, span(TILE_ROWS) - 14.0 - 22.0);
        for (name, state, installed) in [
            ("depth-panel-install", ViewState::Failed("Video Depth Anything is not installed: press install…".into()), false),
            ("depth-panel-running", ViewState::Running(22, 48), true),
        ] {
            let v = View {
                tool: crate::depth::Tool::Vda,
                dir: "/Users/someone/.wobble/VideoDepthAnything".into(),
                installed,
                uv: true,
                installing: None,
                install_result: (!installed).then(|| Err("uv sync: exited with status 1".into())),
                weights: Err("Video Depth Anything small is not downloaded (116.4 MB) — the node's panel has the button".into()),
                state,
                log: vec!["running on Apple Silicon (MPS), vits, input 518".into(); 3],
                has_input: true,
                loaded: Some((48, false)),
                on_disk: false,
                busy_elsewhere: false,
            };
            let mut p = DepthParams::new(0);
            let over = std::cell::Cell::new(0.0f32);
            let wide = std::cell::Cell::new(0.0f32);
            let shot = crate::snapshot::shoot(w, hgt, |ui| {
                let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    let _ = panel(ui, 1, &mut p, &v);
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
