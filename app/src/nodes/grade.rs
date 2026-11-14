//! Grade: wheels or sliders, the zones and their ranges.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Everything the grade panel borrows from the properties pass.
pub(crate) struct Cx<'a> {
    pub g: &'a mut engine::graph::GradeParams,
    pub grade_sliders: &'a mut bool,
    pub key: u64,
    pub node: u64,
    pub tab: &'a String,
    pub viewing: bool,
    pub working: engine::color::Colorspace,
    pub zoom: f32,
    pub hud_hidden: &'a mut std::collections::HashSet<u64>,
}

/// Grade: wheels or sliders, the zones and their ranges.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    cx: Cx,
) -> Option<widgets::HudSlider> {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    let Cx {
        g,
        grade_sliders,
        key,
        node,
        tab,
        viewing,
        working,
        zoom,
        hud_hidden,
    } = cx;
    // Tabs (`widgets::panel_tabs`): the master, one per
    // zone, and the ranges that cut the luma axis into
    // them. `tab` is the id the title row's strip chose.
    let zone: Option<usize> =
        tab.strip_prefix("zone:").and_then(|s| s.parse().ok());
    if tab == "ranges" {
        match &mut g.zones {
            None => {
                ui.weak("no zones: one correction for the whole frame");
                ui.weak(
                    "zones are shadows / midtones / highlights, \
                     generalised — bands of the luma axis, each with a \
                     grade of its own, cut like a LumaKey's bands",
                );
                let add = ui.button("add zones");
                // For --test-zones to aim at.
                widgets::note_rect(
                    ui,
                    egui::Id::new("grade-add-zones-rect"),
                    add.rect,
                );
                if add
                    .on_hover_text(
                        "shadows / midtones / highlights with the usual \
                         default ranges; split or dissolve bands from \
                         there",
                    )
                    .clicked()
                {
                    g.zones = Some(engine::graph::GradeZones::default());
                }
            }
            Some(z) => {
                // The LumaKey's own band editor, over the
                // zones' partition — same model, same
                // gestures, zone names on the chips.
                let bid = ui.id().with(("grade-zone", node));
                let mut sel = ui
                    .data_mut(|d| d.get_temp::<usize>(bid).unwrap_or(0));
                let labels: Vec<String> =
                    (0..z.count()).map(|i| z.label(i)).collect();
                let opts = bandui::BandOpts {
                    keyer: false,
                    labels: Some(&labels),
                    name_hint: "zone name",
                    removable: true,
                    max_points: Some(engine::graph::ZONE_PTS),
                    // Nothing below the panel: the
                    // preview rides in the band's own
                    // row (the tail), because this page
                    // is one tile tall and a row of its
                    // own went below the fold.
                    reserve: 0.0,
                };
                // PREVIEW. Setting a range is a question
                // about PIXELS — which ones this zone
                // has — and the curve above answers it
                // in the abstract. These put the answer
                // on the shot: the matte for a boundary,
                // the highlight for what it lands on.
                // It follows the selected band, so the
                // picture moves with the hand from chip
                // to chip.
                use engine::graph::ZoneView;
                let preview = &mut z.preview;
                let out = bandui::luma_panel_with(
                    ui,
                    &mut z.key,
                    node,
                    &mut sel,
                    zoom,
                    &opts,
                    |ui, sel| {
                        ui.label(egui::RichText::new("preview").small());
                        for v in ZoneView::ALL {
                            let on = preview.view == v;
                            let chip = ui.add(egui::Button::selectable(
                                on,
                                egui::RichText::new(v.name()).small(),
                            ));
                            widgets::note_rect(
                                ui,
                                egui::Id::new(("zone-preview-rect", v.name())),
                                chip.rect,
                            );
                            if chip.on_hover_text(v.hint()).clicked() && !on {
                                preview.view = v;
                                preview.zone = sel as u32;
                            }
                        }
                        // A preview only ever replaces
                        // the picture of the node being
                        // VIEWED — that is what keeps it
                        // out of renders and away from
                        // consumers — so say so rather
                        // than showing nothing and
                        // letting it look broken.
                        if preview.view != ZoneView::Off && !viewing {
                            ui.label(
                                egui::RichText::new("— view this node (V)")
                                    .small()
                                    .weak(),
                            )
                            .on_hover_text(
                                "a preview replaces the picture of the \
                                 node being viewed, so it can never \
                                 reach a render or a node downstream",
                            );
                        }
                    },
                );
                ui.data_mut(|d| d.insert_temp(bid, sel));
                // The corrections follow the bands: a
                // split copies its band's, a dissolve
                // drops one.
                if let Some(new) = out.split {
                    z.band_split(new);
                }
                if let Some(i) = out.dissolved {
                    z.band_dissolved(i);
                }
                z.sync();
                if z.preview.view != ZoneView::Off {
                    z.preview.zone = sel as u32;
                }
                if out.remove_all {
                    g.zones = None;
                }
            }
        }
    } else {
        // The master's page or a zone's: ONE instrument
        // over a different set of five, so a zone is
        // graded exactly the way the whole frame is.
        let zone = zone.filter(|i| *i < g.zone_count());
        let zone16 = zone.map(|z| z as u16);
        let chans = anim::grade_channels(zone16);
        let sat_ch = anim::grade_sat(zone16);
        let [con_ch, piv_ch] = anim::grade_contrast(zone16);
        let mut gang = g.gang;
        if let Some(z) = g.zones.as_mut() {
            z.sync();
            // Leaving the ranges page puts the preview
            // away: it is an instrument for setting a
            // boundary, not a state to be left on — and
            // the page you are on is the only place that
            // explains why the picture looks like that.
            z.preview.view = engine::graph::ZoneView::Off;
        }
        if let Some(engine::graph::GradeAdjustMut {
            lift,
            gamma,
            gain,
            offset,
            saturation,
            contrast,
            pivot,
        }) = g.adjust_mut(zone)
        {
        ui.horizontal(|ui| {
            ui.checkbox(&mut gang, "gang rgb").on_hover_text(
                "gang r/g/b — the slider rows and the viewer HUD",
            );
            widgets::key_dot(ui, rcx, &chans);
            let mut shown = !hud_hidden.contains(&key);
            if ui.checkbox(&mut shown, "viewer HUD").changed() {
                if shown {
                    hud_hidden.remove(&key);
                } else {
                    hud_hidden.insert(key);
                }
            }
            // Two instruments for the same four
            // adjustments: wheels to feel for a colour,
            // sliders to state one — and to reach the
            // S/D/F axes.
            ui.separator();
            if ui
                .add(egui::Button::selectable(!*grade_sliders, "wheels"))
                .clicked()
            {
                *grade_sliders = false;
            }
            if ui
                .add(egui::Button::selectable(*grade_sliders, "sliders"))
                .on_hover_text(
                    "rgb sliders: hold G to invert the gang, S saturation, \
                     D orange<->cyan, F green<->magenta",
                )
                .clicked()
            {
                *grade_sliders = true;
            }
            // Saturation belongs to the page, not to
            // either view of it, so it rides in the
            // header — and in a FIXED-WIDTH child,
            // because the tile's height budget (TILE 250
            // = wheels 187 + header) has no room for
            // another row: the overflow would land in the
            // scroll area where it cannot even be
            // clicked. The panel is three tiles wide; the
            // room here is sideways.
            // Contrast and its pivot ride alongside it
            // for the same reason, the three sharing
            // whatever width the header has left.
            ui.separator();
            let w = ((ui.available_width() - 16.0) / 3.0).clamp(120.0, 200.0);
            let rows: [(&str, &mut f32, f32, f32, anim::Channel); 3] = [
                ("sat", saturation, 0.0, 4.0, sat_ch),
                ("contrast", contrast, 0.0, 3.0, con_ch),
                ("pivot", pivot, 0.01, 1.0, piv_ch),
            ];
            for (label, v, lo, hi, ch) in rows {
                let mut cell = ui.new_child(
                    egui::UiBuilder::new()
                        .max_rect(egui::Rect::from_min_size(
                            ui.cursor().min,
                            egui::vec2(w, 18.0),
                        ))
                        .layout(egui::Layout::top_down(egui::Align::Min)),
                );
                if let Some(h) =
                    widgets::slider(&mut cell, rcx, label, v, lo, hi, &[ch])
                {
                    hud_drag_out = Some(h);
                }
                let used = cell.min_rect();
                ui.advance_cursor_after_rect(used);
            }
        });
        if !*grade_sliders {
            if let Some(ri) = wheels::adjust_wheels_ui(
                ui,
                [lift, gamma, gain, offset],
                (node, zone.map_or(0, |z| z + 1)),
                zoom,
            ) {
                // A wheel pulled out: the same triple
                // and range the sliders view offers.
                let (lo, hi) = [(0.0, 1.0), (0.0, 1.0), (0.0, 5.0), (0.0, 5.0)][ri];
                hud_drag_out = Some(widgets::HudSlider {
                    node,
                    chans: anim::grade_triple(zone16, ri).to_vec(),
                    label: ["lift", "gamma", "gain", "offset"][ri].into(),
                    lo,
                    hi,
                    rgb: true,
                    rgb_scale: ri == 1 || ri == 2,
                    wheel: true,
                    ..Default::default()
                });
            }
        } else {
            // Slider ranges (the wheels keep their own):
            // the useful span, not the reachable one —
            // a typed value may still go past either end.
            let luma = working.primaries.luma_weights();
            let disp = engine::color::primaries_matrix(
                working.primaries,
                engine::color::Primaries::Rec709,
            );
            // TWO COLUMNS. Four stacked rgb rows are
            // taller than a tile, and the overflow ends
            // up inside the tile's scroll area where it
            // cannot even be clicked; the Grade panel is
            // three tiles WIDE, so the room is sideways.
            let col_w = ((ui.available_width() - 12.0) * 0.5).max(210.0);
            let mut col: Option<egui::Ui> = None;
            ui.horizontal_top(|ui| {
            for ri in 0..4usize {
                let name = ["lift", "gamma", "gain", "offset"][ri];
                let spec = match ri {
                    // The chosen defaults: lift and gamma 0..1,
                    // gain and offset 0..5.
                    0 => rgbslider::RgbSpec::offset(0.0, 1.0, luma),
                    1 => rgbslider::RgbSpec::scale(0.0, 1.0, luma),
                    2 => rgbslider::RgbSpec::scale(0.0, 5.0, luma),
                    _ => rgbslider::RgbSpec::offset(0.0, 5.0, luma),
                };
                let chans = anim::grade_triple(zone16, ri);
                let v: &mut [f32; 3] = match ri {
                    0 => &mut *lift,
                    1 => &mut *gamma,
                    2 => &mut *gain,
                    _ => &mut *offset,
                };
                if ri % 2 == 0 {
                    col = Some(ui.new_child(
                        egui::UiBuilder::new()
                            .max_rect(egui::Rect::from_min_size(
                                ui.cursor().min,
                                egui::vec2(col_w, 0.0),
                            ))
                            .layout(egui::Layout::top_down(
                                egui::Align::Min,
                            )),
                    ));
                }
                let cui = col.as_mut().expect("column started");
                if let Some(h) = widgets::rgb_row(
                    cui,
                    rcx,
                    name,
                    v,
                    &mut gang,
                    spec.with_display(disp),
                    &chans,
                ) {
                    hud_drag_out = Some(h);
                }
                if ri % 2 == 1 {
                    let used = cui.min_rect();
                    ui.advance_cursor_after_rect(used.expand2(
                        egui::vec2(6.0, 0.0),
                    ));
                }
            }
            });
        }
        }
        // ONE gang for the node, so the panel and the
        // viewer HUD never disagree about it.
        g.gang = gang;
    }
    hud_drag_out
}
