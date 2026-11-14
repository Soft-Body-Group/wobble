//! The HUD chips: a column down the viewer's left edge naming every manipulator and floating slider, each with an x.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// The HUD chips: a column down the viewer's left edge naming every manipulator and floating slider, each with an x.
    pub(crate) fn viewer_hud_chips(
        &mut self,
        ui: &mut egui::Ui,
        avail: egui::Rect,
        gizmo_layer: egui::LayerId,
        scrubbing: bool,
    ) {
        // HUD chips: a discreet column down the viewer's left edge naming
        // every manipulator and floating slider on screen, each with an ×.
        // This is where a pin comes OFF — a second X-click adds, never
        // toggles — and where a selection-followed HUD is dismissed too.
        if !scrubbing && !self.prefs.hud_hide.unwrap_or(false) {
            let follow = self.prefs.hud_follow.unwrap_or(true);
            // (node, pinned): pins first, then whatever the selection adds.
            let mut items: Vec<(u64, bool)> = Vec::new();
            for id in &self.gizmo_pinned {
                if self.egraph.node(*id).is_some() {
                    items.push((*id, true));
                }
            }
            if follow {
                for id in &self.selected {
                    if !items.iter().any(|(n, _)| n == id) && self.egraph.node(*id).is_some() {
                        items.push((*id, false));
                    }
                }
            }
            if !items.is_empty() || !self.huds.is_empty() {
                let strip = egui::Rect::from_min_size(
                    avail.min + egui::vec2(6.0, 6.0),
                    egui::vec2(220.0, (avail.height() - 12.0).max(0.0)),
                );
                let mut unpin: Option<u64> = None;
                let mut drop_hud: Option<usize> = None;
                // (node, colour): a chip's swatch picked (Some) or cleared
                // (None) that node's HUD colour.
                let mut set_ink: Option<(u64, Option<[f32; 3]>)> = None;
                ui.scope_builder(
                    egui::UiBuilder::new().layer_id(gizmo_layer).max_rect(strip),
                    |ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(4.0, 3.0);
                        // One chip: swatch, name, ×. The swatch IS the
                        // node's HUD colour — its type colour until it picks
                        // one — and opens the colour picker; right-click
                        // puts the adaptive ink back. `salt` keeps two chips
                        // of one node (its manipulators and a floating
                        // slider) on separate pickers.
                        let chip = |ui: &mut egui::Ui,
                                    node: u64,
                                    salt: usize,
                                    text: String,
                                    hint: &str|
                         -> (bool, Option<Option<[f32; 3]>>) {
                            let mut close = false;
                            let mut pick: Option<Option<[f32; 3]>> = None;
                            egui::Frame::new()
                                .fill(ui.visuals().panel_fill.gamma_multiply(0.7))
                                .corner_radius(egui::CornerRadius::same(3))
                                .inner_margin(egui::Margin::symmetric(5, 1))
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.spacing_mut().item_spacing.x = 3.0;
                                        let n = self.egraph.node(node);
                                        let cur = n.and_then(|n| n.hud_color);
                                        let mut v = match cur {
                                            Some(c) => [c[0], c[1], c[2], 1.0],
                                            None => {
                                                let col = n
                                                    .map(|n| editor::type_color(&n.kind))
                                                    .unwrap_or(ui.visuals().weak_text_color());
                                                let [r, g, b, _] = col.to_srgba_unmultiplied();
                                                let d = |x: u8| colorpicker::dec1(x as f32 / 255.0);
                                                [d(r), d(g), d(b), 1.0]
                                            }
                                        };
                                        let spec = colorpicker::PickSpec::rgb()
                                            .with_input(widgets::sens_factor, widgets::claim_sens);
                                        let tip = if cur.is_some() {
                                            "HUD colour · click to pick · right-click for the \
                                             adaptive ink"
                                        } else {
                                            "HUD colour: adaptive, light or dark against the \
                                             footage · click to pick one"
                                        };
                                        // The swatch takes exactly the old
                                        // dot's footprint (a 6-wide slot + 3
                                        // gap) so the name and the × stay
                                        // where hands — and the layer suite's
                                        // tab click, which threads past the ×
                                        // — expect them. The dot is no
                                        // target, so the CLICK reaches across
                                        // the name: the whole chip but its ×
                                        // opens the picker.
                                        ui.spacing_mut().item_spacing.x = 1.0;
                                        let name_w = egui::WidgetText::from(
                                            egui::RichText::new(text.clone()).small(),
                                        )
                                        .into_galley(
                                            ui,
                                            Some(egui::TextWrapMode::Extend),
                                            f32::INFINITY,
                                            egui::TextStyle::Small,
                                        )
                                        .size()
                                        .x;
                                        let at = ui.cursor().min;
                                        let hit = egui::Rect::from_min_max(
                                            at - egui::vec2(4.0, 3.0),
                                            at + egui::vec2(8.0 + 1.0 + name_w, 10.0 + 3.0),
                                        );
                                        let r = colorpicker::button_sized(
                                            ui,
                                            ("hud-ink", node, salt),
                                            &mut v,
                                            spec,
                                            egui::vec2(8.0, 10.0),
                                            tip,
                                            Some(hit),
                                        );
                                        ui.spacing_mut().item_spacing.x = 3.0;
                                        if r.changed() {
                                            pick = Some(Some([v[0], v[1], v[2]]));
                                        } else if r.secondary_clicked() && cur.is_some() {
                                            pick = Some(None);
                                        }
                                        // Not selectable: a selectable label
                                        // senses clicks and would take the
                                        // one meant for the swatch's reach.
                                        ui.add(
                                            egui::Label::new(egui::RichText::new(text).small().weak())
                                                .selectable(false),
                                        );
                                        close = ui
                                            .add(
                                                egui::Button::new(
                                                    egui::RichText::new("×").small(),
                                                )
                                                .frame(false),
                                            )
                                            .on_hover_text(hint)
                                            .clicked();
                                    });
                                });
                            (close, pick)
                        };
                        for (id, pinned) in &items {
                            let Some(n) = self.egraph.node(*id) else { continue };
                            let hint = if *pinned {
                                "unpin manipulators"
                            } else {
                                "deselect (manipulators follow the selection)"
                            };
                            let (close, pick) = chip(ui, *id, 0, editor::display_label(n), hint);
                            if close {
                                unpin = Some(*id);
                            }
                            if let Some(c) = pick {
                                set_ink = Some((*id, c));
                            }
                        }
                        for (i, h) in self.huds.iter().enumerate() {
                            let (close, pick) =
                                chip(ui, h.node, i + 1, h.label.clone(), "remove floating slider");
                            if close {
                                drop_hud = Some(i);
                            }
                            if let Some(c) = pick {
                                set_ink = Some((h.node, c));
                            }
                        }
                    },
                );
                if let Some(id) = unpin {
                    self.gizmo_pinned.retain(|x| *x != id);
                    self.selected.retain(|x| *x != id);
                }
                if let Some(i) = drop_hud {
                    self.huds.remove(i);
                }
                if let Some((id, c)) = set_ink {
                    if let Some(n) = self.egraph.node_mut(id) {
                        n.hud_color = c;
                    }
                }
            }
        }
    }
}
