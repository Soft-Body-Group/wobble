//! Write (Render): the file, the format, the range and the render buttons.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Everything the write panel borrows from the properties pass.
pub(crate) struct Cx<'a> {
    pub wp: &'a mut engine::graph::WriteParams,
    pub key: u64,
    pub node: u64,
    pub nc_upstream: &'a Option<(Vec<String>, bool)>,
    pub shot_span_now: (u64, u64),
    pub working: engine::color::Colorspace,
    pub write_data: bool,
    pub write_spans: Option<((u64, u64), (u64, u64))>,
    pub browser: &'a mut Option<browser::SeqBrowser>,
    pub decoder: &'a Arc<dyn ExrDecoder>,
    pub graph_bake: &'a Option<SubtreeBake>,
    pub load_errors: &'a HashMap<u64, String>,
    pub render_status: &'a Option<String>,
    pub written: &'a std::collections::HashMap<u64, Vec<std::path::PathBuf>>,
    pub blog_click: &'a mut Option<engine::graph::WriteParams>,
    pub make_load_click: &'a mut Option<engine::graph::WriteParams>,
    pub write_click: &'a mut Option<engine::graph::WriteParams>,
    pub graph_bake_cancel: &'a mut bool,
    /// Proxy is on: a render is half res and named `_prx`.
    pub proxy: bool,
}

/// Write (Render): the file, the format, the range and the render buttons.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    cx: Cx,
) {
    let Cx {
        wp,
        key,
        node,
        nc_upstream,
        shot_span_now,
        working,
        write_data,
        write_spans,
        browser,
        decoder,
        graph_bake,
        load_errors,
        render_status,
        written,
        blog_click,
        make_load_click,
        write_click,
        graph_bake_cancel,
        proxy,
    } = cx;
    // Rendering is a BUTTON. Nothing here writes to disk
    // as you scrub, and the image passes through
    // untouched so a Write can sit mid-chain.
    //
    // Manifesto rule 10: this is the moment output
    // LEAVES, so it is the moment to say what made it.
    // A warning, never a block — the render still runs.
    if let Some((srcs, nc)) = &nc_upstream {
        ui.colored_label(
            theme::cost_nc(),
            if *nc {
                "⚠ non-commercial weights upstream"
            } else {
                "⚠ licence-restricted weights upstream"
            },
        );
        ui.weak(if *nc {
            format!(
                "via {}. these frames are research-only; the \
                 render is not blocked.",
                srcs.join(", ")
            )
        } else {
            format!(
                "via {}. commercial use only under the model's \
                 terms (revenue cap, use restrictions): read them \
                 before delivery. the render is not blocked.",
                srcs.join(", ")
            )
        });
    }
    let running = matches!(
        graph_bake.as_ref().map(|b| b.node),
        Some(n) if n == key
    );
    // Location, name, padding, subfolder: the path is
    // compiled from them (`WriteParams::pattern`).
    let lab = |ui: &mut egui::Ui, t: &str| {
        ui.add_sized(
            [widgets::LABEL_W, 16.0],
            egui::Label::new(egui::RichText::new(t).small()).truncate(),
        );
    };
    let text_w = |ui: &egui::Ui, trailing: f32| {
        (ui.available_width() - trailing).max(60.0)
    };
    if !wp.path.trim().is_empty() {
        // A setup from before the split whose path the
        // fields cannot say (`WriteParams::migrate_legacy`).
        ui.horizontal(|ui| {
            lab(ui, "path");
            let w = text_w(ui, 0.0);
            ui.add(
                egui::TextEdit::singleline(&mut wp.path)
                    .hint_text("out/comp.####.exr")
                    .desired_width(w),
            );
        });
        ui.horizontal(|ui| {
            ui.weak("an old single-path render");
            if ui
                .button("use location + name")
                .on_hover_text("start from this path's folder and name")
                .clicked()
            {
                let dir = std::path::Path::new(wp.path.trim())
                    .parent()
                    .map(|d| d.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let stem = std::path::Path::new(wp.path.trim())
                    .file_name()
                    .map(|f| f.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let stem = stem.split(['.', '#']).next().unwrap_or("").trim_matches('_');
                wp.location = dir;
                wp.name = if stem.is_empty() { "comp".into() } else { stem.into() };
                wp.subfolder = false;
                wp.path.clear();
            }
        });
    } else {
        ui.horizontal(|ui| {
            lab(ui, "location");
            let w = text_w(ui, 64.0);
            ui.add(
                egui::TextEdit::singleline(&mut wp.location)
                    .hint_text("~/renders")
                    .desired_width(w),
            )
            .on_hover_text("the folder the render goes in");
            if ui
                .button("browse…")
                .on_hover_text("pick the folder in wobble's browser")
                .clicked()
            {
                let typed = engine::paths::expand_home(wp.location.trim());
                let start = if !wp.location.trim().is_empty() && typed.is_dir() {
                    typed
                } else {
                    std::env::var("HOME")
                        .or_else(|_| std::env::var("USERPROFILE"))
                        .map(std::path::PathBuf::from)
                        .unwrap_or_else(|_| std::path::PathBuf::from("/"))
                };
                let mut b = browser::SeqBrowser::new(Some(key), start, decoder.clone());
                b.pick_folder = true;
                *browser = Some(b);
            }
        });
        ui.horizontal(|ui| {
            lab(ui, "name");
            let w = text_w(ui, 0.0);
            ui.add(
                egui::TextEdit::singleline(&mut wp.name)
                    .hint_text("comp")
                    .desired_width(w),
            )
            .on_hover_text("one plain name: the files are name.####.exr");
        });
        let mut pad = wp.padding as f32;
        if widgets::plain_slider(ui, "padding", &mut pad, 1.0, 8.0, true)
            .on_hover_text("digits in the frame number: 4 writes 1001 or 0001")
            .changed()
        {
            wp.padding = pad.round().clamp(1.0, 12.0) as u32;
        }
        ui.horizontal(|ui| {
            ui.add_space(widgets::LABEL_W + ui.spacing().item_spacing.x);
            ui.checkbox(&mut wp.subfolder, "enclose in subfolder of same name")
                .on_hover_text("location/name/name.####.exr rather than location/name.####.exr");
        });
    }
    // The file format: an EXR sequence, or one movie file.
    ui.horizontal(|ui| {
        lab(ui, "format");
        egui::ComboBox::from_id_salt(("write-fmt", node))
            .selected_text(wp.format.label())
            .width(130.0)
            .show_ui(ui, |ui| {
                for f in engine::graph::WriteFormat::ALL {
                    ui.selectable_value(&mut wp.format, f, f.label()).on_hover_text(f.hint());
                }
            })
            .response
            .on_hover_text(wp.format.hint());
    });
    let movie = wp.format.is_movie();
    if movie {
        if write_data {
            ui.colored_label(theme::error(), egui::RichText::new("data renders are EXR only").small());
        } else if let Err(e) = movieout::MovieOut::can_encode(wp.format) {
            ui.colored_label(theme::error(), egui::RichText::new(e).small());
        }
    }
    if write_data {
        // Data is written as data: no output space to pick.
        ui.weak("data in: one float32 channel Z per frame, no colour transform");
    } else {
        let pattern = wp.pattern();
        colorui::output_space_row(
            ui,
            &format!("write-cs-{node}"),
            &mut wp.output,
            &mut wp.output_default,
            working,
            &pattern,
        );
        // One channel instead of rgba (a matte, a
        // luminance pass): half float, named as asked.
        ui.horizontal(|ui| {
            lab(ui, "channels");
            let shown = wp.channel.map_or("rgba", |c| c.label());
            egui::ComboBox::from_id_salt(("write-chan", node))
                .selected_text(shown)
                .width(78.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut wp.channel, None, "rgba")
                        .on_hover_text("all four channels");
                    for c in engine::graph::WriteChannel::ALL {
                        ui.selectable_value(&mut wp.channel, Some(c), c.label())
                            .on_hover_text(match c {
                                engine::graph::WriteChannel::Luma => {
                                    "one channel: luminance in the output space's primaries"
                                }
                                _ => "one channel: this one alone",
                            });
                    }
                })
                .response
                .on_hover_text("write every channel, or just one");
            if let Some(c) = wp.channel.filter(|_| !movie) {
                ui.label(egui::RichText::new("as").small());
                let w = text_w(ui, 0.0).min(90.0);
                ui.add(
                    egui::TextEdit::singleline(&mut wp.channel_name)
                        .hint_text(c.default_name())
                        .desired_width(w),
                )
                .on_hover_text("the channel's name in the file");
            }
        });
    }
    // Compression: every method the EXR writer has. A movie's is its codec.
    if !movie {
    ui.horizontal(|ui| {
        lab(ui, "compression");
        egui::ComboBox::from_id_salt(("write-comp", node))
            .selected_text(wp.compression.label())
            .width(110.0)
            .show_ui(ui, |ui| {
                for c in engine::graph::ExrCompression::ALL {
                    ui.selectable_value(&mut wp.compression, c, c.label())
                        .on_hover_text(c.hint());
                }
            })
            .response
            .on_hover_text(wp.compression.hint());
    });
    if wp.compression.lossy(write_data) {
        ui.colored_label(
            theme::warn(),
            egui::RichText::new(if write_data {
                "lossy here: float32 data is rounded to 24 bits"
            } else {
                "lossy: the frames will not match the comp exactly"
            })
            .small(),
        );
    }
    // A movie has no frame numbers in its name.
    ui.horizontal(|ui| {
        lab(ui, "start");
        crate::widgets::num_field_w(ui, widgets::NUM_W + 6.0, egui::DragValue::new(&mut wp.start).update_while_editing(false).speed(widgets::rated(ui, 1.0)),
        )
        .on_hover_text("frame number the first written file gets");
    });
    }
    let (input_span, global_span) =
        write_spans.unwrap_or((shot_span_now, shot_span_now));
    ui.horizontal(|ui| {
        // Comp frames, like every other frame number in
        // the app.
        ui.label("frames");
        let mode = wp.mode();
        for m in engine::graph::WriteRange::ALL {
            let hover = match m {
                engine::graph::WriteRange::Input => {
                    "the range the noodle into this node carries"
                }
                engine::graph::WriteRange::Range => "a start and end set here",
                engine::graph::WriteRange::Global => "the script's frame range",
            };
            if ui
                .add(egui::Button::selectable(mode == m, m.label()))
                .on_hover_text(hover)
                .clicked()
            {
                wp.range_mode = Some(m);
                // A first switch to range starts from
                // what it was rendering, not from zero.
                if m == engine::graph::WriteRange::Range && wp.range.is_none() {
                    wp.range = Some(wp.resolve_range(input_span, global_span));
                }
            }
        }
    });
    if wp.mode() == engine::graph::WriteRange::Range {
        ui.horizontal(|ui| {
            let (a, b) = wp.range.get_or_insert(input_span);
            crate::widgets::num_field_w(ui, widgets::NUM_W + 8.0, egui::DragValue::new(a).update_while_editing(false)
                    .speed(widgets::rated(ui, 0.5))
                    .prefix("start "),
            );
            crate::widgets::num_field_w(ui, widgets::NUM_W + 8.0, egui::DragValue::new(b).update_while_editing(false)
                    .speed(widgets::rated(ui, 0.5))
                    .prefix("end "),
            );
            *b = (*b).max(*a);
        });
    }
    // What a render will actually cover, in words.
    {
        let (a, b) = wp.resolve_range(input_span, global_span);
        ui.label(
            egui::RichText::new(format!(
                "{a} to {b} ({})",
                wp.mode().label()
            ))
            .weak()
            .small(),
        );
    }
    // The full path the render will write, wrapped
    // rather than cut: the first file, and the last.
    if proxy {
        ui.colored_label(
            theme::warn(),
            egui::RichText::new("proxy is on: renders at proxy res, named _prx").small(),
        );
    }
    if wp.is_set() {
        let (a, b) = wp.resolve_range(input_span, global_span);
        let n = b.saturating_sub(a) + 1;
        // What a render NOW writes: under proxy, the `_prx` name.
        let shown = if proxy { wp.proxied() } else { wp.clone() };
        let first = shown.resolve_frame(0);
        let last = shown.resolve_frame(n - 1);
        ui.add(
            egui::Label::new(
                egui::RichText::new(first.display().to_string()).small().monospace(),
            )
            .wrap(),
        )
        .on_hover_text("the full path of the first file this render writes");
        let last_name = last
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default();
        let new_dir = !shown.dir().is_dir();
        ui.label(
            egui::RichText::new(format!(
                "{}{n} frame{}{}",
                if movie {
                    "one file, ".to_string()
                } else if n > 1 {
                    format!("through {last_name}, ")
                } else {
                    String::new()
                },
                if n == 1 { "" } else { "s" },
                if new_dir { " · the folder will be made" } else { "" },
            ))
            .weak()
            .small(),
        );
    }
    ui.horizontal(|ui| {
        if running {
            if ui.button("cancel").clicked() {
                *graph_bake_cancel = true;
            }
        } else if ui
            .add_enabled(wp.is_set(), egui::Button::new("render"))
            .on_hover_text(
                "renders the comp above this node frame by frame; the viewer follows along",
            )
            .on_disabled_hover_text("set a location first")
            .clicked()
        {
            *write_click = Some(wp.clone());
        }
        if ui
            .add_enabled(wp.is_set(), egui::Button::new("make Load"))
            .on_hover_text(
                "a Load beside this node reading what it renders, over the render's frames",
            )
            .on_disabled_hover_text("set a location first")
            .clicked()
        {
            *make_load_click = Some(wp.clone());
        }
    });
    if let Some(b) = graph_bake.as_ref().filter(|b| b.node == key)
    {
        let span = b.len.saturating_sub(b.first_tick()).max(1);
        let done = b.next.saturating_sub(b.first_tick());
        ui.add(
            egui::ProgressBar::new(done as f32 / span as f32)
                .text(format!("rendering {done}/{span}")),
        );
    }
    if let Some(e) = load_errors.get(&key) {
        ui.colored_label(theme::error(), e);
    } else if let Some(st) =
        render_status.as_ref().filter(|s| s.starts_with("wrote "))
    {
        ui.colored_label(theme::ok(), st);
    }
    // Always there, so it can be found before there is
    // anything to send; live once a render has landed.
    let rendered = !running
        && written
            .get(&key)
            .and_then(|f| f.first())
            .is_some_and(|f| f.is_file());
    if ui
        .add_enabled(rendered, egui::Button::new("submit to the wobble blog…"))
        .on_hover_text(
            "send this render to wobble.ing/wobbles/ as an MP4 \
             (a PNG for one frame), seen through the viewer's \
             transform; it appears once approved",
        )
        .on_disabled_hover_text("render first, then send it to wobble.ing/wobbles/")
        .clicked()
    {
        *blog_click = Some(wp.clone());
    }
    // Copy the frames to a paired wobble (`apiserve::deliver`).
    #[cfg(not(target_arch = "wasm32"))]
    apiserve::deliver::send_row(
        ui,
        &wp.clean_name(),
        if running { &[] } else { written.get(&key).map(Vec::as_slice).unwrap_or(&[]) },
    );
    ui.weak(if movie {
        "one movie file, display-referred — the image passes through untouched"
    } else {
        "EXR, RGBA half — the image passes through untouched"
    });
}
