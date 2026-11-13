//! Viewer colour-management panel: which display rendering transform runs,
//! what the monitor is, and (with the `opendrt` feature) the full OpenDRT
//! parameter space.
//!
//! Pure UI over [`ViewSettings`] — no engine calls. Everything here is
//! display state, so an edit costs a uniform write; only the transform,
//! display gamut and EOTF rows rebuild the blit pipeline.
//!
//! Owned by instance C (see .agents/board.md).

use crate::theme;
use engine::viewtf::{Eotf, Gamut, Look, ViewSettings, ViewTransform};

/// Blender's looks for the transform in use; nothing for one with none.
fn look_row(ui: &mut egui::Ui, v: &mut ViewSettings) -> bool {
    let looks = v.transform.looks();
    if looks.len() < 2 {
        return false;
    }
    let mut changed = false;
    let shown = if looks.contains(&v.look) { v.look } else { Look::None };
    ui.horizontal(|ui| {
        ui.label("look");
        egui::ComboBox::from_id_salt("view-look")
            .selected_text(shown.label())
            .show_ui(ui, |ui| {
                for l in looks {
                    changed |= ui.selectable_value(&mut v.look, *l, l.label()).changed();
                }
            });
    })
    .response
    .on_hover_text("Blender's looks, same maths as Blender's colour config");
    if v.transform == ViewTransform::AgX
        && v.look != Look::None
        && v.transform.available()
        && engine::viewtf::agx::lumcomp().is_none()
    {
        ui.colored_label(theme::warn(), "this Blender has no AgX looks — showing None");
    }
    changed
}

/// Returns true when anything changed (caller persists prefs).
pub fn view_settings_ui(ui: &mut egui::Ui, v: &mut ViewSettings) -> bool {
    let mut changed = false;

    ui.horizontal(|ui| {
        ui.label("transform");
        egui::ComboBox::from_id_salt("view-tf")
            .selected_text(v.transform.label())
            .show_ui(ui, |ui| {
                for t in ViewTransform::ALL {
                    let mut r = ui.selectable_value(&mut v.transform, t, t.label());
                    if !t.available() {
                        r = r.on_hover_text(if t == ViewTransform::AgX {
                            "no Blender install found (AgX reads Blender's own LUTs)"
                        } else {
                            "rebuild with --features opendrt (GPLv3)"
                        });
                    }
                    changed |= r.changed();
                }
            });
    });
    if !v.transform.available() {
        ui.colored_label(
            theme::warn(),
            "not in this build — showing Filmic",
        );
    }
    changed |= look_row(ui, v);

    ui.separator();
    // The working space is a property of the COMP, not of the viewer — it
    // changes what the pixels are, not how they are shown, so it is set in
    // the colour panel and only reported here.
    ui.horizontal(|ui| {
        ui.label("working");
        ui.weak(v.working.label());
    })
    .response
    .on_hover_text("set by the comp's working space (colour panel)");
    changed |= gamut_row(ui, "display", "view-dg", &mut v.display, &Gamut::DISPLAYS);
    ui.horizontal(|ui| {
        ui.label("eotf");
        egui::ComboBox::from_id_salt("view-eotf")
            .selected_text(v.eotf.label())
            .show_ui(ui, |ui| {
                for e in Eotf::ALL {
                    changed |= ui.selectable_value(&mut v.eotf, e, e.label()).changed();
                }
            });
    });
    changed |= crate::widgets::plain_slider(ui, "exposure", &mut v.exposure, -8.0, 8.0, false)
        .on_hover_text("stops")
        .changed();
    if v.exposure != 0.0 && ui.button("reset exposure").clicked() {
        v.exposure = 0.0;
        changed = true;
    }

    #[cfg(feature = "opendrt")]
    if v.transform == ViewTransform::OpenDrt {
        ui.separator();
        changed |= opendrt_ui(ui, &mut v.opendrt);
    }

    changed
}

fn gamut_row(ui: &mut egui::Ui, label: &str, salt: &str, g: &mut Gamut, opts: &[Gamut]) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        egui::ComboBox::from_id_salt(salt)
            .selected_text(g.label())
            .show_ui(ui, |ui| {
                for o in opts {
                    changed |= ui.selectable_value(g, *o, o.label()).changed();
                }
            });
    });
    changed
}

/// One labelled slider that ORs into the panel's changed flag.
#[cfg(feature = "opendrt")]
fn sl(
    ui: &mut egui::Ui,
    changed: &mut bool,
    v: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    text: &str,
) {
    *changed |= crate::widgets::plain_slider(ui, text, v, *range.start(), *range.end(), false)
        .changed();
}

#[cfg(feature = "opendrt")]
fn opendrt_ui(ui: &mut egui::Ui, p: &mut engine::viewtf::opendrt::OpenDrtParams) -> bool {
    use engine::viewtf::opendrt::{CreativeWhite, LookPreset, OpenDrtParams};
    let mut changed = false;

    ui.horizontal(|ui| {
        ui.label("look");
        for preset in LookPreset::ALL {
            if ui.small_button(preset.label()).clicked() {
                *p = OpenDrtParams::preset(preset);
                changed = true;
            }
        }
    });
    ui.weak("presets overwrite every knob below");

    ui.collapsing("tonescale", |ui| {
        sl(ui, &mut changed, &mut p.tn_Lp, 100.0..=1000.0, "peak luminance");
        sl(ui, &mut changed, &mut p.tn_Lg, 3.0..=25.0, "grey luminance");
        sl(ui, &mut changed, &mut p.tn_con, 1.0..=2.0, "contrast");
        sl(ui, &mut changed, &mut p.tn_sh, 0.0..=1.0, "shoulder clip");
        sl(ui, &mut changed, &mut p.tn_toe, 0.0..=0.1, "toe");
        sl(ui, &mut changed, &mut p.tn_off, 0.0..=0.02, "offset");
        changed |= ui
            .checkbox(&mut p.tn_lcon_enable, "low contrast")
            .changed();
        if p.tn_lcon_enable {
            sl(ui, &mut changed, &mut p.tn_lcon, 0.0..=3.0, "amount");
            sl(ui, &mut changed, &mut p.tn_lcon_w, 0.0..=2.0, "width");
        }
        changed |= ui
            .checkbox(&mut p.tn_hcon_enable, "high contrast")
            .changed();
        if p.tn_hcon_enable {
            sl(ui, &mut changed, &mut p.tn_hcon, -1.0..=1.0, "amount");
            sl(ui, &mut changed, &mut p.tn_hcon_pv, 0.0..=4.0, "pivot");
            sl(ui, &mut changed, &mut p.tn_hcon_st, 0.0..=4.0, "strength");
        }
    });

    ui.collapsing("rendering space", |ui| {
        sl(ui, &mut changed, &mut p.rs_sa, 0.0..=0.6, "strength");
        sl(ui, &mut changed, &mut p.rs_rw, 0.0..=0.8, "weight R");
        sl(ui, &mut changed, &mut p.rs_bw, 0.0..=0.8, "weight B");
    });

    ui.collapsing("purity", |ui| {
        sl(ui, &mut changed, &mut p.pt_lml, 0.0..=1.0, "limit low");
        sl(ui, &mut changed, &mut p.pt_lml_r, 0.0..=1.0, "limit low R");
        sl(ui, &mut changed, &mut p.pt_lml_g, 0.0..=1.0, "limit low G");
        sl(ui, &mut changed, &mut p.pt_lml_b, 0.0..=1.0, "limit low B");
        sl(ui, &mut changed, &mut p.pt_lmh, 0.0..=1.0, "limit high");
        sl(ui, &mut changed, &mut p.pt_lmh_r, 0.0..=1.0, "limit high R");
        sl(ui, &mut changed, &mut p.pt_lmh_b, 0.0..=1.0, "limit high B");
        changed |= ui.checkbox(&mut p.ptm_enable, "mid-range purity").changed();
        if p.ptm_enable {
            sl(ui, &mut changed, &mut p.ptm_low, 0.0..=2.0, "low");
            sl(ui, &mut changed, &mut p.ptm_low_rng, 0.0..=1.0, "low range");
            sl(ui, &mut changed, &mut p.ptm_low_st, 0.1..=1.0, "low strength");
            sl(ui, &mut changed, &mut p.ptm_high, -0.9..=0.0, "high");
            sl(ui, &mut changed, &mut p.ptm_high_rng, 0.0..=1.0, "high range");
            sl(ui, &mut changed, &mut p.ptm_high_st, 0.1..=1.0, "high strength");
        }
        changed |= ui.checkbox(&mut p.ptl_enable, "softclip").changed();
        if p.ptl_enable {
            sl(ui, &mut changed, &mut p.ptl_c, 0.0..=0.25, "C");
            sl(ui, &mut changed, &mut p.ptl_m, 0.0..=0.25, "M");
            sl(ui, &mut changed, &mut p.ptl_y, 0.0..=0.25, "Y");
        }
    });

    ui.collapsing("brilliance", |ui| {
        changed |= ui.checkbox(&mut p.brl_enable, "brilliance").changed();
        if p.brl_enable {
            sl(ui, &mut changed, &mut p.brl, -6.0..=2.0, "amount");
            sl(ui, &mut changed, &mut p.brl_r, -6.0..=2.0, "R");
            sl(ui, &mut changed, &mut p.brl_g, -6.0..=2.0, "G");
            sl(ui, &mut changed, &mut p.brl_b, -6.0..=2.0, "B");
            sl(ui, &mut changed, &mut p.brl_rng, 0.0..=1.0, "range");
            sl(ui, &mut changed, &mut p.brl_st, 0.0..=1.0, "strength");
        }
        changed |= ui.checkbox(&mut p.brlp_enable, "post brilliance").changed();
        if p.brlp_enable {
            sl(ui, &mut changed, &mut p.brlp, -1.0..=0.0, "amount");
            sl(ui, &mut changed, &mut p.brlp_r, -3.0..=0.0, "R");
            sl(ui, &mut changed, &mut p.brlp_g, -3.0..=0.0, "G");
            sl(ui, &mut changed, &mut p.brlp_b, -3.0..=0.0, "B");
        }
    });

    ui.collapsing("hue", |ui| {
        changed |= ui.checkbox(&mut p.hc_enable, "hue contrast R").changed();
        if p.hc_enable {
            sl(ui, &mut changed, &mut p.hc_r, 0.0..=2.0, "amount");
            sl(ui, &mut changed, &mut p.hc_r_rng, 0.0..=1.0, "range");
        }
        changed |= ui.checkbox(&mut p.hs_rgb_enable, "hue shift RGB").changed();
        if p.hs_rgb_enable {
            sl(ui, &mut changed, &mut p.hs_r, 0.0..=1.0, "R");
            sl(ui, &mut changed, &mut p.hs_r_rng, 0.0..=2.0, "R range");
            sl(ui, &mut changed, &mut p.hs_g, 0.0..=1.0, "G");
            sl(ui, &mut changed, &mut p.hs_g_rng, 0.0..=2.0, "G range");
            sl(ui, &mut changed, &mut p.hs_b, 0.0..=1.0, "B");
            sl(ui, &mut changed, &mut p.hs_b_rng, 0.0..=4.0, "B range");
        }
        changed |= ui.checkbox(&mut p.hs_cmy_enable, "hue shift CMY").changed();
        if p.hs_cmy_enable {
            sl(ui, &mut changed, &mut p.hs_c, 0.0..=1.0, "C");
            sl(ui, &mut changed, &mut p.hs_c_rng, 0.0..=2.0, "C range");
            sl(ui, &mut changed, &mut p.hs_m, 0.0..=1.0, "M");
            sl(ui, &mut changed, &mut p.hs_m_rng, 0.0..=2.0, "M range");
            sl(ui, &mut changed, &mut p.hs_y, 0.0..=1.0, "Y");
            sl(ui, &mut changed, &mut p.hs_y_rng, 0.0..=2.0, "Y range");
        }
    });

    ui.collapsing("creative white", |ui| {
        ui.horizontal(|ui| {
            for w in CreativeWhite::ALL {
                changed |= ui.selectable_value(&mut p.cwp, w, w.label()).changed();
            }
        });
        sl(ui, &mut changed, &mut p.cwp_lm, 0.0..=1.0, "range");
    });

    changed
}
