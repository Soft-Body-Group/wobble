//! TimeRange: the clip bar (trim, slip, where it sits).
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// TimeRange: the clip bar (trim, slip, where it sits).
#[allow(clippy::too_many_arguments)]
pub(crate) fn panel(
    ui: &mut egui::Ui,
    tr: &mut engine::graph::TimeRangeParams,
    cur_t: u64,
    input_range: Option<(u64, u64)>,
    node: u64,
    script: (i64, i64),
    play_range: &mut Option<(u64, u64)>,
    load_mirror: &Option<(u64, Option<(u64, u64)>)>,
) {
    // The noodle's frames, set here. No remap: frames
    // past the source's own come from its before/after.
    // The Load's in/out widget: a bar over the input's
    // frames, then the numbers with their reset, then
    // "start at" (the frame the in plays at; a slip).
    let mut next = (tr.first, tr.last);
    let mut at = tr.at;
    // Input frame minus this noodle's frame.
    let off = tr.slip().unwrap_or(0);
    {
        let mut span = next;
        // Its own view, kept: the main bar's hands, R and
        // pie, and it never re-fits to the drag.
        let vid = ui.make_persistent_id(("timerange-view", node));
        let shown = ui
            .data(|d| d.get_temp::<(i64, i64)>(vid.with("shown")))
            .unwrap_or_else(|| {
                imfram::widgets::span::fit(
                    span,
                    input_range.map(|(a, b)| (a as i64, b as i64)),
                )
            });
        let out = imfram::widgets::span_bar(
            ui,
            ("timerange-span", node),
            &mut span,
            imfram::widgets::SpanOpts {
                view: Some(shown),
                solid: input_range.map(|(a, b)| (a as i64, b as i64)),
                marker: Some(cur_t as i64 + off),
                labels: true,
                color: Some(theme::range_load()),
                hint: Some(
                    "the input's frames this noodle plays · drag an end or \
                     slide it whole (with \"start at\" set, a slip) \
                     · grey = the input's frames\n\
                     wheel or ctrl+MMB zooms · MMB or alt-drag pans · \
                     R or double-click frames the in–out · ` for the range pie",
                ),
                ..Default::default()
            },
        );
        if out.changed {
            next = span;
        }
        let node_range = match input_range {
            Some((a, b)) => (next.0.min(a as i64), next.1.max(b as i64)),
            None => next,
        };
        let mv = minibar::view(
            ui,
            vid,
            out.rect,
            // The bar is in the INPUT's frames.
            &minibar::Ranges {
                comp: (script.0 + off, script.1 + off),
                node: node_range,
                inout: play_range
                    .map(|(a, b)| (a as i64 + off, b as i64 + off)),
            },
            out.double_clicked,
            false,
        );
        let shown = (mv.view.0.round() as i64, mv.view.1.round() as i64);
        ui.data_mut(|d| d.insert_temp(vid.with("shown"), shown));
    }
    ui.horizontal(|ui| {
        for (label, v, tip) in [
            ("in", &mut next.0, "the first frame of this noodle"),
            ("out", &mut next.1, "the last frame of this noodle"),
        ] {
            ui.label(label);
            crate::widgets::num_field_w(
                ui,
                widgets::NUM_W + 14.0,
                egui::DragValue::new(v)
                    .update_while_editing(false)
                    .speed(widgets::rated(ui, 0.5)),
            )
            .on_hover_text(tip);
        }
        if let Some((a, b)) = input_range {
            let auto = (a as i64, b as i64);
            if ui
                .add_enabled(next != auto, egui::Button::new("reset").small())
                .on_hover_text(format!("in/out back to the input's frames: {a}–{b}"))
                .clicked()
            {
                next = auto;
            }
            let off = (auto.0 - next.0).max(0) + (next.1 - auto.1).max(0);
            if off > 0 {
                ui.colored_label(theme::ok(), format!("{off} past the input"))
                    .on_hover_text(
                        "past the input's frames, its before/after answers \
                         (a Load's hold / loop / bounce / black)",
                    );
            }
        }
        // The timeline's in/out: take those frames.
        if let Some((i, o)) = play_range.filter(|_| load_mirror.is_none()) {
            let (i, o) = (i as i64, o as i64);
            if (i, o) != tr.output()
                && ui
                    .small_button("timeline in/out")
                    .on_hover_text(format!(
                        "set this range to the timeline's in/out: {i}–{o}"
                    ))
                    .clicked()
            {
                // Those frames of THIS noodle: no slip.
                next = (i + off, o.max(i) + off);
                if at.is_some() {
                    at = Some(i);
                }
                *play_range = None;
            }
        }
    });
    // "Start at": optional. Unset, the in plays at its
    // own frame and moving it moves the noodle; set, it
    // holds and moving in/out slips the input under it.
    ui.horizontal(|ui| {
        ui.label("start at");
        let mut v = at.unwrap_or(next.0);
        let r = crate::widgets::num_field_w(
            ui,
            widgets::NUM_W + 14.0,
            egui::DragValue::new(&mut v)
                .update_while_editing(false)
                .speed(widgets::rated(ui, 0.5)),
        )
        .on_hover_text(
            "the frame the in point plays at — like a Load's \"start at\". \
             Unset, it plays at its own frame",
        );
        if r.changed() {
            at = Some(v);
        }
        if ui
            .add_enabled(at.is_some(), egui::Button::new("reset").small())
            .on_hover_text("unset: the in plays at its own frame, no slip")
            .clicked()
        {
            at = None;
        }
        if let Some(o) = at.map(|a| next.0 - a).filter(|o| *o != 0) {
            ui.weak(format!("slip {o:+}"));
        }
    });
    next.1 = next.1.max(next.0);
    (tr.first, tr.last) = next;
    tr.at = at;
}
