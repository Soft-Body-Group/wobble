//! Read: the path, the frame range, the caches and what the file holds.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Everything the read panel borrows from the properties pass.
pub(crate) struct Cx<'a> {
    pub path: &'a mut String,
    pub ram_cache: &'a mut bool,
    pub nvme_cache: &'a mut bool,
    pub source: &'a mut usize,
    pub layer: &'a mut Option<String>,
    pub input: &'a mut engine::color::InputSpace,
    pub multi: &'a mut bool,
    pub start: &'a mut Option<i64>,
    pub first: &'a mut Option<i64>,
    pub last: &'a mut Option<i64>,
    pub edges: &'a mut engine::graph::ReadEdges,
    pub cache_on_play: &'a mut bool,
    pub single: &'a mut engine::decode::SingleChannel,
    pub eight_bit: &'a mut bool,
    pub asset: &'a mut Option<assets::AssetRef>,
    pub node: u64,
    pub key: u64,
    pub comp0: i64,
    pub working: engine::color::Colorspace,
    pub timeline_pos: u64,
    pub playback_dirty: &'a mut bool,
    pub setup_error: &'a mut Option<String>,
    pub browser: &'a mut Option<browser::SeqBrowser>,
    pub aov_dialog: &'a mut Option<(u64, Vec<(String, String, bool)>)>,
    pub play_range: &'a mut Option<(u64, u64)>,
    pub fps: &'a mut f32,
    pub asset_ingest: &'a Option<(u64, std::sync::mpsc::Receiver<Result<assets::AssetRef, String>>)>,
    pub asset_ingest_msg: &'a Option<(u64, String)>,
    pub decoder: &'a Arc<dyn ExrDecoder>,
    pub disk_cache: &'a Option<Arc<DiskCache>>,
    pub loaded: &'a HashMap<u64, LoadedSeq>,
    pub load_errors: &'a HashMap<u64, String>,
    pub load_mirror: &'a Option<(u64, Option<(u64, u64)>)>,
    pub asset_act: &'a mut Option<AssetAct>,
    pub cache_meter: &'a mut Option<usize>,
    pub deferred_load: &'a mut Option<String>,
    pub deferred_refresh: &'a mut bool,
    pub layer_changed: &'a mut bool,
    pub load_cache_act: &'a mut Option<LoadCacheAct>,
}

/// Read: the path, the frame range, the caches and what the file holds.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    cx: Cx,
) {
    #[allow(unused_variables)] // setup_error is the web build's
    let Cx {
        path,
        ram_cache,
        nvme_cache,
        source,
        layer,
        input,
        multi,
        start,
        first,
        last,
        edges,
        cache_on_play,
        single,
        eight_bit,
        asset,
        node,
        key,
        comp0,
        working,
        timeline_pos,
        playback_dirty,
        setup_error,
        browser,
        aov_dialog,
        play_range,
        fps,
        asset_ingest,
        asset_ingest_msg,
        decoder,
        disk_cache,
        loaded,
        load_errors,
        load_mirror,
        asset_act,
        cache_meter,
        deferred_load,
        deferred_refresh,
        layer_changed,
        load_cache_act,
    } = cx;
    let reader_slot = *source;
    let mut load_clicked = false;
    let mut refresh_clicked = false;
    if let Some(a) = asset.as_ref() {
        // A project asset: what it IS, never where.
        egui::Grid::new(("read-asset", key)).num_columns(2).show(ui, |ui| {
            ui.label("project");
            ui.strong(a.project_label());
            ui.end_row();
            ui.label("shot");
            ui.strong(a.shot_label());
            ui.end_row();
            ui.label("asset");
            ui.strong(a.asset_label());
            ui.end_row();
        });
        ui.horizontal(|ui| {
            if ui.button("pick asset…").on_hover_text("choose another asset from the project tree").clicked() {
                *asset_act = Some(AssetAct::Pick);
            }
            if ui.button("reload").on_hover_text("find the asset again (after a move) and load it").clicked() {
                *asset_act = Some(AssetAct::Reload);
            }
            if ui.small_button("use a path").on_hover_text("forget the asset and edit this Load's path by hand").clicked() {
                *asset_act = Some(AssetAct::Forget);
            }
        });
    } else {
    ui.horizontal(|ui| {
        ui.label("dir");
        ui.add(
            egui::TextEdit::singleline(path).desired_width(f32::INFINITY),
        );
    });
    ui.horizontal(|ui| {
        #[cfg(not(target_arch = "wasm32"))]
        if ui.button("pick asset…").on_hover_text("load an asset from an organisicator project by name — keeps working when the project moves").clicked() {
            *asset_act = Some(AssetAct::Pick);
        }
        #[cfg(not(target_arch = "wasm32"))]
        if !path.is_empty()
            && ui
                .add_enabled(asset_ingest.is_none(), egui::Button::new("ingest…"))
                .on_hover_text("copy this footage into a shot in organisicator; the Load then reads the copy by id, so moves and renames keep working")
                .clicked()
        {
            *asset_act = Some(AssetAct::Ingest);
        }
        // In a page: the browser's file picker, and what it
        // picks loads into THIS Read (`webmain::drain_opens`).
        #[cfg(target_arch = "wasm32")]
        if ui.button("pick files…").on_hover_text("images or an image sequence from your computer: EXR, DPX, PNG, JPEG").clicked() {
            if let Err(e) = webmain::pick_for(key) {
                *setup_error = Some(format!("{e:#}"));
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        if ui.button("browse…").clicked() {
            // Start where the node points, else home.
            let expanded = engine::paths::expand_home(path);
            let p = expanded.as_path();
            let start = if p.is_dir() {
                p.to_path_buf()
            } else if let Some(parent) =
                p.parent().filter(|d| d.is_dir())
            {
                parent.to_path_buf()
            } else {
                std::env::var("HOME")
                    .or_else(|_| std::env::var("USERPROFILE"))
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(|_| std::path::PathBuf::from("/"))
            };
            *browser = Some(browser::SeqBrowser::new(
                Some(key),
                start,
                decoder.clone(),
            ));
        }
        load_clicked |= ui.button("load").clicked();
        refresh_clicked |= ui
            .button("⟲ refresh")
            .on_hover_text(
                "read the files again from disk: picks up \
                 re-rendered or added frames and empties \
                 the caches holding the old ones",
            )
            .clicked();
    });
    }
    if asset_ingest.as_ref().is_some_and(|(n, _)| *n == key) {
        ui.weak("copying into organisicator…");
    } else if let Some((_, m)) = asset_ingest_msg.as_ref().filter(|(n, _)| *n == key) {
        ui.weak(m.as_str());
    }
    let path_now = path.clone();
    match (loaded.get(&key), load_errors.get(&key)) {
        (Some(l), _) => {
            ui.monospace(format!(
                "{} frames  {}x{} half",
                l.seq.frames.len(),
                l.seq.info.width,
                l.seq.info.height,
            ));
            // Manifesto rule 8: a slow path is visible.
            // A movie cannot be random-accessed — a
            // scrub outside the warm run costs a seek
            // and a decode from the nearest keyframe —
            // and that is worth knowing BEFORE the
            // playback feels wrong.
            if l.seq.movie {
                ui.colored_label(
                    theme::cost_slow(),
                    "♦ movie — decoded from keyframes, \
                     slower to scrub than a sequence",
                )
                .on_hover_text(
                    "Frames live inside one container, so \
                     reaching an arbitrary one means \
                     seeking to a keyframe and decoding \
                     forward. Playing forward is cheap; \
                     scrubbing is not. Cache to RAM or \
                     NVMe, or transcode to a frame \
                     sequence, to get scrub back.",
                );
            }
            if let Some(rate) = l.movie_fps {
                movie_rate_row(ui, rate, fps);
            }
            if l.missing > 0 {
                ui.colored_label(
                    theme::warn(),
                    format!("⚠ {} missing frames in range", l.missing),
                );
            }
            let fitted = l.fitted.count.load(std::sync::atomic::Ordering::Relaxed);
            if fitted > 0 {
                ui.colored_label(
                    theme::warn(),
                    format!(
                        "⚠ {fitted} frames of another size, fitted to {}x{}",
                        l.seq.info.width, l.seq.info.height
                    ),
                )
                .on_hover_text(
                    "These stills are not the sequence's size (the \
                     first frame's). Each is scaled to fit whole, \
                     aspect kept, centred, with transparent black \
                     round it. Put a Reformat downstream to change \
                     the frame.",
                );
            }
            if l.path != path_now {
                ui.weak("path edited — load to apply");
            }
        }
        (None, Some(e)) => {
            ui.colored_label(theme::error(), e);
        }
        (None, None) => {
            ui.weak("not loaded");
        }
    }
    // EXR layer picker. Only shown when the file has
    // more than the base layer to offer; a DPX or a plain
    // rgba EXR has nothing to choose.
    if let Some(l) = loaded.get(&key) {
        if l.layers.len() > 1 {
            let current = layer.clone().unwrap_or_default();
            let shown = l
                .layers
                .iter()
                .find(|d| d.name == current)
                .map(|d| d.label().to_string())
                .unwrap_or_else(|| format!("{current} (missing)"));
            let mut pick: Option<Option<String>> = None;
            ui.horizontal(|ui| {
                ui.label("layer");
                egui::ComboBox::from_id_salt(("read-layer", key))
                    .selected_text(shown)
                    .width(150.0)
                    .show_ui(ui, |ui| {
                        for d in &l.layers {
                            let sel = d.name == current;
                            let text = if d.part > 0 {
                                format!("{} (part {})", d.label(), d.part)
                            } else {
                                format!(
                                    "{}  [{}]",
                                    d.label(),
                                    d.suffixes().join("")
                                )
                            };
                            if ui.selectable_label(sel, text).clicked()
                                && !sel
                            {
                                pick = Some(
                                    (!d.name.is_empty())
                                        .then(|| d.name.clone()),
                                );
                            }
                        }
                    });
            });
            if let Some(new) = pick {
                *layer = new;
                // A different layer is different pixels:
                // rebuild the rig and drop the bake.
                *playback_dirty = true;
                *layer_changed = true;
            }
            // Fat-cable output: the wire carries EVERY
            // layer, lazily — ChannelSelects downstream
            // pick them off, and only picked layers
            // ever decode or cache.
            if ui
                .checkbox(multi, "all-channels output")
                .on_hover_text(
                    "output becomes a multi-channel cable; \
                     wire it into ChannelSelect nodes (or \
                     drop it on any image input and one is \
                     inserted for you)",
                )
                .changed()
            {
                *playback_dirty = true;
            }
            // The python ritual as a button: tick
            // AOVs, get one Read per layer summed back
            // into the beauty (see `aovs`).
            if ui.button("rebuild from AOVs…").clicked() {
                *aov_dialog = Some((
                    key,
                    l.layers
                        .iter()
                        .map(|d| {
                            (
                                d.name.clone(),
                                d.label().to_string(),
                                !d.name.is_empty(),
                            )
                        })
                        .collect(),
                ));
            }
        }
    }
    // What the plate answers OUTSIDE its own frames —
    // when the viewer range, a retime or a Write asks
    // for more than the clip has. The usual before/after.
    ui.horizontal(|ui| {
        for (label, edge, tip) in [
            ("before", &mut edges.before, "frames before the clip's first"),
            ("after", &mut edges.after, "frames after the clip's last"),
        ] {
            ui.label(label);
            egui::ComboBox::from_id_salt(("read-edge", key, label))
                .width(64.0)
                .selected_text(edge.label())
                .show_ui(ui, |ui| {
                    for e in engine::graph::ReadEdge::ALL {
                        ui.selectable_value(edge, e, e.label());
                    }
                })
                .response
                .on_hover_text(format!(
                    "{tip}: hold the end frame, loop the clip, \
                     bounce it, or black"
                ));
        }
    });
    // IN / OUT and IN AT: the range in the CLIP's own
    // frame numbers, and the comp frame the in point
    // lands on. Stored as start/first/last (comp frames,
    // `ranges::ClipRange` translates), so moving in or out
    // with "in at" held SLIPS the footage under the comp.
    // A frame in range with no file is black and never
    // cached, so running OUT past the end watches a
    // render arrive.
    let files = loaded.get(&key).map(|l| l.seq.frames.len() as i64);
    let f0 = loaded.get(&key).and_then(|l| l.first_frame).unwrap_or(0);
    let s0 = start.unwrap_or(0);
    let cur = ranges::ClipRange::of(
        f0,
        s0,
        first.unwrap_or(s0),
        last.unwrap_or(s0 + files.unwrap_or(1) - 1),
    );
    let mut next = cur;
    // The bar: the in/out over the clip's files. Drag an
    // end, or slide it whole (a slip: "in at" holds).
    {
        let mut span = (cur.inp, cur.out);
        let out = imfram::widgets::span_bar(
            ui,
            ("read-span", key),
            &mut span,
            imfram::widgets::SpanOpts {
                solid: files.map(|n| (f0, f0 + n.max(1) - 1)),
                marker: Some(f0 + timeline_pos as i64 - s0),
                max_len: Some(MAX_READ_OUT as i64),
                labels: true,
                color: Some(theme::range_load()),
                hint: Some(
                    "clip frames · sliding is a slip: \
                     \"start at\" stays put · grey = the files on disk",
                ),
                ..Default::default()
            },
        );
        if out.changed {
            (next.inp, next.out) = span;
        }
    }
    // Two lines: in / out with their reset (all the
    // files), then in at with its (the file numbers).
    ui.horizontal(|ui| {
        for (label, v, tip) in [
            ("in", &mut next.inp, "the clip's first frame used (its own numbering)"),
            ("out", &mut next.out, "the clip's last frame used"),
        ] {
            ui.label(label);
            ui.add_sized(
                [widgets::NUM_W + 14.0, widgets::NUM_H],
                egui::DragValue::new(v)
                    .update_while_editing(false)
                    .speed(widgets::rated(ui, 0.5)),
            )
            .on_hover_text(tip);
        }
        if let Some(n) = files {
            let auto = (f0, f0 + n - 1);
            if ui
                .add_enabled((next.inp, next.out) != auto, egui::Button::new("reset").small())
                .on_hover_text(format!("in/out back to the files: {}–{}", auto.0, auto.1))
                .clicked()
            {
                (next.inp, next.out) = auto;
            }
            // Say plainly that part of the range is not on
            // disk; otherwise the black reads as a broken
            // read.
            let off = (auto.0 - next.inp).max(0) + (next.out - auto.1).max(0);
            if off > 0 {
                ui.colored_label(theme::ok(), format!("{off} not on disk"))
                    .on_hover_text(
                        "frames in the range with no file read as \
                         black and are never cached — a render \
                         landing there fills in on its own",
                    );
            }
        }
        // The timeline's in/out, while viewing something
        // else: trim to those comp frames, no slip.
        if let Some((i, o)) = play_range.filter(|_| load_mirror.is_none()) {
            let (i, o) = (i as i64, o as i64);
            if (i, o) != (cur.at, cur.at + cur.out - cur.inp)
                && ui
                    .small_button("timeline in/out")
                    .on_hover_text(format!(
                        "trim this Load to the timeline's in/out: comp {i}–{o}"
                    ))
                    .clicked()
            {
                next = ranges::ClipRange::of(f0, s0, i, o.max(i));
                *play_range = None;
            }
        }
    });
    ui.horizontal(|ui| {
        ui.label("start at");
        ui.add_sized(
            [widgets::NUM_W + 14.0, widgets::NUM_H],
            egui::DragValue::new(&mut next.at)
                .update_while_editing(false)
                .speed(widgets::rated(ui, 0.5)),
        )
        .on_hover_text(format!(
            "the comp frame the in point plays at — the Load's \"start at\" (the comp starts at {comp0})"
        ));
        if ui
            .add_enabled(files.is_some() && next.at != next.inp, egui::Button::new("reset").small())
            .on_hover_text("place by the file numbers: in 1001 plays at 1001")
            .clicked()
        {
            next.at = next.inp;
        }
    });
    if next != cur {
        next.out = next.out.clamp(next.inp, next.inp + MAX_READ_OUT as i64);
        let (st, fi, la) = next.stored(f0);
        *start = Some(st);
        *first = Some(fi);
        *last = Some(la);
        *playback_dirty = true;
    }
    if colorui::input_space_row(
        ui,
        &format!("read-cs-{node}"),
        input,
        working,
        path,
    ) {
        // The transform is compiled INTO the shader, so
        // this is a topology change, not a knob.
        *playback_dirty = true;
    }
    ui.horizontal(|ui| {
        if ui
            .checkbox(ram_cache, "RAM cache")
            .on_hover_text("decoded frames may use the RAM budget")
            .changed()
        {
            *playback_dirty = true;
        }
        let have_disk = disk_cache.is_some();
        let r = ui
            .add_enabled(
                have_disk,
                egui::Checkbox::new(nvme_cache, "NVMe cache"),
            )
            .on_hover_text(if have_disk {
                "decoded frames persist to the NVMe cache dir"
            } else {
                "set a cache dir first (cache… in the menubar)"
            });
        if r.changed() {
            *playback_dirty = true;
        }
        if ui
            .checkbox(cache_on_play, "cache on play")
            .on_hover_text(
                "frames decoded while playing go into the cache; \
                 off, only the cache buttons fill it",
            )
            .changed()
        {
            *playback_dirty = true;
        }
    });
    // How the frames are STORED (`decode::Pack`): every
    // cache tier, the upload and the texture shrink with
    // it. Its own cache entries, so a switch re-caches.
    ui.horizontal(|ui| {
        ui.label("single channel");
        let was = *single;
        egui::ComboBox::from_id_salt(("read-single", node))
            .selected_text(single.label())
            .show_ui(ui, |ui| {
                for c in engine::decode::SingleChannel::ALL {
                    ui.selectable_value(single, c, c.label());
                }
            })
            .response
            .on_hover_text(
                "keep one channel, shown in all four (a white element \
                 over black): a quarter of the memory, cache and upload",
            );
        if *single != was {
            *playback_dirty = true;
        }
        if ui
            .checkbox(eight_bit, "8-bit")
            .on_hover_text(
                "store 8-bit (0..1, clamped): half the memory, cache and \
                 upload again; lossless for PNG/JPEG, lossy for float plates",
            )
            .changed()
        {
            *playback_dirty = true;
        }
    });
    if loaded.contains_key(&key) {
        ui.horizontal(|ui| {
            if ui
                .button("cache in–out")
                .on_hover_text("decode the in..out into the cache now")
                .clicked_by(egui::PointerButton::Primary)
            {
                // A mouse click only: a button keeps focus after
                // one, and Space (play) or Enter on a focused
                // button would start a run nobody asked for.
                *load_cache_act = Some(LoadCacheAct::InOut);
            }
            if ui
                .button("cache all")
                .on_hover_text(
                    "every file of the clip: the in..out to RAM and \
                     NVMe, the rest to NVMe (RAM holds what plays)",
                )
                .clicked_by(egui::PointerButton::Primary)
            {
                *load_cache_act = Some(LoadCacheAct::All);
            }
            if ui
                .button("clear cache")
                .on_hover_text("drop this Load's frames from RAM and NVMe")
                .clicked_by(egui::PointerButton::Primary)
            {
                *load_cache_act = Some(LoadCacheAct::Clear);
            }
        });
    }
    if load_clicked {
        *deferred_load = Some(path_now.clone());
    }
    if refresh_clicked {
        *deferred_refresh = true;
    }
    *cache_meter = Some(reader_slot);
}
