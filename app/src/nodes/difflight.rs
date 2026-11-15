//! The DiffusionLight node's panel and glyph. The params live in
//! `engine::nodes::difflight`; the runner (install, fetch, render, process,
//! adopt) in `crate::difflight`.

use crate::difflight::{Action, View, ViewState, LICENCE_URL, REPO, SDXL_LICENCE_URL, WEIGHTS_BYTES, WHERE};
use crate::theme;
use egui::{Color32, Painter, Pos2, Stroke, Ui, Vec2};
use engine::graph::{DiffLightParams, DiffLightSize};

pub const TILES: usize = 2;
pub const TILE_ROWS: usize = 2;

/// The properties panel. The click it hands back is acted on after the
/// node borrow ends.
pub fn panel(ui: &mut Ui, node: u64, p: &mut DiffLightParams, v: &View) -> Option<Action> {
    let mut act = None;
    ui.horizontal_wrapped(|ui| {
        ui.weak("DiffusionLight-Turbo ·");
        ui.hyperlink_to("MIT", LICENCE_URL).on_hover_text("the method's code and both of its LoRAs are MIT");
        ui.weak("· on SDXL ·");
        ui.hyperlink_to("Open RAIL++-M", SDXL_LICENCE_URL).on_hover_text(
            "Stable Diffusion XL and its depth ControlNet: commercial use is allowed, with restrictions on what the model may be used for. The depth estimator (DPT-Large) is Apache-2.0. Nothing here is non-commercial, so the node is not tagged",
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
                ui.weak("needs uv (astral.sh/uv)");
            }
        });
    }
    if let Some(Err(e)) = &v.install_result {
        ui.colored_label(theme::error(), format!("install failed: {e}"));
    }
    // --- weights --------------------------------------------------------
    if v.weights {
        ui.weak("weights fetched");
    } else if let Some((line, done, total)) = &v.fetching {
        ui.horizontal(|ui| {
            ui.add(egui::Spinner::new().size(12.0));
            ui.add(egui::Label::new(egui::RichText::new(format!("fetching {done}/{total}… {line}")).weak()).truncate());
        });
    } else {
        let b = ui
            .add_enabled(v.installed, egui::Button::new(format!("fetch weights… ({:.1} GB)", WEIGHTS_BYTES as f64 / 1e9)))
            .on_hover_text(format!(
                "SDXL base, its depth ControlNet, the Turbo and Exposure LoRAs and DPT-Large from Hugging Face, at pinned revisions, into {}/models. Public: no login. Nothing is downloaded until you press this, and a run is offline",
                v.dir
            ));
        if b.clicked() {
            act = Some(Action::Fetch);
        }
    }
    if let Some(Err(e)) = &v.fetch_result {
        ui.add(egui::Label::new(egui::RichText::new(format!("fetch failed: {e}")).color(theme::error())).wrap());
    }
    // --- the estimate (a new one when these change) -----------------------
    ui.horizontal(|ui| {
        ui.label("exposure");
        ui.add(egui::DragValue::new(&mut p.exposure).speed(0.05).range(-12.0..=12.0).suffix(" st")).on_hover_text(
            "stops applied to the plate before the model sees it. It is shown an ordinary sRGB picture clipped at 1, so set this until the plate would look like a reasonably exposed photograph. It is divided back out: the latlong stays in the plate's units",
        );
        ui.label("seed");
        ui.add(egui::DragValue::new(&mut p.seed).speed(1.0)).on_hover_text(
            "one seed for every frame. The method is sensitive to it: if the ball comes out wrong (a pattern, not a reflection), try another",
        );
    });
    ui.horizontal(|ui| {
        ui.label("size");
        egui::ComboBox::from_id_salt(("difflight-size", node)).width(96.0).selected_text(p.size.label()).show_ui(ui, |ui| {
            for s in DiffLightSize::ALL {
                ui.selectable_value(&mut p.size, s, s.label());
            }
        }).response.on_hover_text("the latlong's size. It is unwrapped from a 256 px ball, so a bigger one is smoother, not more detailed");
        ui.label("steps");
        ui.add(egui::DragValue::new(&mut p.steps).speed(0.2).range(4..=100)).on_hover_text("denoising steps: 30 is the authors'. Time is proportional");
    });
    // --- over time (a second pass: no estimate is made again) -------------
    ui.horizontal(|ui| {
        ui.label("smooth");
        ui.add(egui::DragValue::new(&mut p.smooth).speed(0.05).range(0.0..=50.0).suffix(" fr")).on_hover_text(
            "every frame is its own guess at what is behind the camera, so the raw latlongs flicker. This averages them over time (a Gaussian, sigma in frames; 0 is off). A second pass over estimates kept on disk: changing it re-runs in seconds",
        );
        ui.label("every");
        ui.add(egui::DragValue::new(&mut p.stride).speed(0.1).range(1..=64)).on_hover_text(
            "estimate every Nth frame and fill the ones between from their neighbours. With 'follow the plate' on, the frames between still get their own level and colour",
        );
    });
    ui.checkbox(&mut p.follow, "follow the plate").on_hover_text(
        "for lighting that really changes every frame (strobes, colour washes): only the PICTURE of the surroundings is smoothed; how bright and what colour each frame's latlong is comes from that frame of the plate, unsmoothed. Off: a plain average over time, which smears a strobe across its neighbours",
    );
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
            let words = if *done > 0 { format!("{done}/{total}") } else { "loading the model…".into() };
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
            ui.monospace(format!("latlong — up to date  ({n} frames)"));
        }
        Some((_, false)) => {
            ui.colored_label(theme::error(), "STALE — the input or a setting changed (still playing the old result)");
        }
        None if !running => {
            ui.weak(if v.on_disk { "a result for this input is on disk — load it" } else { "not run: black" });
        }
        None => {}
    }
    ui.horizontal(|ui| {
        if running {
            if ui.button("cancel").on_hover_text("the estimates made so far are kept: the next run carries on from them").clicked() {
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
                "renders the input over its range, estimates an HDR latlong per frame (slow: see the hover on 'installed'), smooths them and plays the result here. Wire it into an Env's latlong input. The centre of the latlong is where the camera looks, on every frame: it is relative to the camera, not the world. Keyed on everything upstream: an edit shows as STALE, nothing re-runs behind your back",
            );
            if b.clicked() {
                act = Some(Action::Run);
            }
            if v.busy_elsewhere {
                ui.weak("another DiffusionLight is running");
            }
            if matches!(v.state, ViewState::Failed(_) | ViewState::Done(_)) && ui.small_button("dismiss").clicked() {
                act = Some(Action::Dismiss);
            }
        }
    });
    if !v.log.is_empty() {
        egui::ScrollArea::vertical().max_height(60.0).id_salt(("difflight-log", node)).show(ui, |ui| {
            for l in &v.log {
                ui.add(egui::Label::new(egui::RichText::new(l).weak().small()).truncate());
            }
        });
    }
    act
}

/// A chrome ball: the horizon bent across it, and a glint.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let sq = egui::Rect::from_center_size(c, Vec2::splat(r * 1.7));
    painter.rect(sq, r * 0.35, fill, stroke, egui::StrokeKind::Middle);
    let ink = theme::node_ink();
    let w = (r * 0.1).max(1.0);
    painter.circle_stroke(c, r * 0.58, Stroke::new(w, ink));
    let n = 8;
    let pts: Vec<Pos2> = (0..=n)
        .map(|i| {
            let t = i as f32 / n as f32 * 2.0 - 1.0;
            c + Vec2::new(t * r * 0.52, (1.0 - t * t) * r * 0.2 + r * 0.02)
        })
        .collect();
    painter.add(egui::Shape::line(pts, Stroke::new(w, ink)));
    painter.circle_filled(c + Vec2::new(-r * 0.2, -r * 0.26), r * 0.1, ink);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let span = |n: usize| crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0);
        let (w, hgt) = (span(TILES) - 14.0, span(TILE_ROWS) - 14.0 - 22.0);
        for (name, state, installed) in [
            ("difflight-panel-install", ViewState::Failed("DiffusionLight-Turbo is not installed: press install…".into()), false),
            ("difflight-panel-running", ViewState::Running(22, 96), true),
        ] {
            let v = View {
                dir: "/Users/someone/.wobble/DiffusionLight-Turbo".into(),
                installed,
                uv: true,
                installing: None,
                install_result: (!installed).then(|| Err("uv sync: exited with status 1".into())),
                weights: false,
                fetching: None,
                fetch_result: Some(Err("fetching stabilityai/stable-diffusion-xl-base-1.0: the connection dropped".into())),
                state,
                log: vec!["loading SDXL + the depth ControlNet + the DiffusionLight LoRAs on CUDA".into(); 3],
                has_input: true,
                loaded: Some((48, false)),
                on_disk: false,
                busy_elsewhere: false,
            };
            let mut p = DiffLightParams::new(0);
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
