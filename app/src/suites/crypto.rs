//! `--test-crypto`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-crypto driver: a Read on the generated cryptomatte plate, a
    /// Cryptomatte node whose rank Reads are built from the file's own
    /// metadata, a pick by name that mattes the cube, invert, the ids view
    /// carrying rank 0's chunks exactly, and a viewer click that resolves
    /// through the manifest to "cube" (and alt-click that removes it).
    pub(crate) fn drive_test_crypto(&mut self, raw: &mut egui::RawInput) {
        use engine::cryptomatte as cm;
        let Some((stage, mut frames, sid)) = self.test_crypto else { return };
        frames += 1;
        // Persist the count FIRST: the stage arms `return` early while they
        // wait, and a count that only lands on the fall-through path is a
        // timeout that never fires (the run just sits there).
        self.test_crypto = Some((stage, frames, sid));
        let since = self.test_crypto_since.get_or_insert_with(web_time::Instant::now).elapsed();
        let budget = web_time::Duration::from_secs(20);
        // A click whose readback caught a frame in flight lands nothing;
        // a person just clicks again, and so does the suite (the 60s cap
        // above is what catches a click that never lands).
        let retry = web_time::Duration::from_secs(3);
        // A REAL pointer click on the image, at a comp pixel: moved there,
        // pressed, and (a frame later) released with no travel — the still
        // click egui reports as `clicked`, never `drag_started`.
        let ir = self.image_rect;
        let (ow, oh) = self.comp.output_size().unwrap_or((512, 288));
        let at = |px: (i32, i32)| {
            egui::pos2(
                ir.min.x + (px.0 as f32 + 0.5) / ow as f32 * ir.width(),
                ir.min.y + (px.1 as f32 + 0.5) / oh as f32 * ir.height(),
            )
        };
        let click = |raw: &mut egui::RawInput, pos: egui::Pos2, modifiers: egui::Modifiers, down: bool| {
            raw.events.insert(0, egui::Event::ModifiersChanged(modifiers));
            raw.events.push(egui::Event::PointerMoved(pos));
            raw.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers,
            });
        };
        let total = self.test_crypto_t0.get_or_insert_with(web_time::Instant::now).elapsed();
        if total > web_time::Duration::from_secs(60) {
            println!("CRYPTO TEST FAIL: timeout at stage {stage} after {total:?}");
            std::process::exit(1);
        }
        let mut next = stage;
        // A ONE-frame plate, written here if absent: the transport runs
        // under test and the discs orbit with the frame, while every pixel
        // below is a frame-0 pixel. One frame = the frame-0 picture, still.
        let plate = std::path::Path::new("data/crypto_still");
        // Frame 0 of the plate: the cube disc is centred at (0.8, 0.5) with
        // an 82px radius, so this box sits well inside it.
        let cube_box = (394u32, 128u32, 32u32, 32u32);
        let cube_px = (410i32, 144i32);
        // ...and the sphere's centre at frame 0 (0.35, 0.62).
        let sphere_px = (179i32, 178i32);
        let none = egui::Modifiers::NONE;
        let cmd = egui::Modifiers::COMMAND;
        let cmd_shift = egui::Modifiers::COMMAND | egui::Modifiers::SHIFT;
        let picked = |g: &editor::EditorGraph| -> Vec<String> {
            match g.node(sid).map(|n| &n.kind) {
                Some(NodeKind::Cryptomatte(c)) => c.picks.iter().map(|p| p.name.clone()).collect(),
                _ => Vec::new(),
            }
        };
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                if !plate.join("frame.0000.exr").is_file() {
                    if let Err(e) = cm::write_plate(plate, 512, 288, 1) {
                        println!("CRYPTO TEST FAIL: could not write {}: {e:#}", plate.display());
                        std::process::exit(1);
                    }
                }
                // BOTH positional Reads play the plate: the canvas follows the
                // largest loaded source, and every pixel below is a plate
                // pixel — with a 2K plate still loaded the canvas would be
                // 2048x1080 and the crypto plate stretched across it.
                self.load_read(0, &plate.to_string_lossy());
                self.load_read(1, &plate.to_string_lossy());
                next = 1;
            }
            1 if frames > 20 => {
                let Some(frame) = self.crypto_frame_of_read(0) else { return };
                let defs = self.crypto.defs(&frame).to_vec();
                let cube = cm::id_of_name("cube");
                if defs.len() != 1
                    || defs[0].name != "CryptoObject"
                    || defs[0].manifest.len() != 3
                    || defs[0].name_of(cube) != Some("cube")
                {
                    println!("CRYPTO TEST FAIL: header read as {defs:?}");
                    std::process::exit(1);
                }
                // The Read flips to all-channels output; the Cryptomatte
                // takes the cable on its ONE input and owns the ranks as
                // its reader set, minted by the rebuild.
                if let Some(editor::ENode { kind: NodeKind::Read { multi, .. }, .. }) =
                    self.egraph.node_mut(0)
                {
                    *multi = true;
                }
                let cid = self.egraph.add(
                    NodeKind::Cryptomatte(Default::default()),
                    egui::pos2(300.0, 0.0),
                );
                self.egraph.connect(cid, 0, 0);
                if self.egraph.channel_source(cid) != Some(0) {
                    println!("CRYPTO TEST FAIL: the cable does not reach the Read");
                    std::process::exit(1);
                }
                if let Some(editor::ENode { kind: NodeKind::Cryptomatte(c), .. }) =
                    self.egraph.node_mut(cid)
                {
                    c.layer = "CryptoObject".into();
                    c.toggle("cube", cube, true);
                }
                self.egraph.view = Some(cid);
                self.playback_dirty = true;
                self.test_crypto_since = None;
                self.test_crypto = Some((2, 0, cid));
                return;
            }
            // The cube is picked: solid matte inside its disc. Viewed, the
            // node shows its ids picture, so the matte is read off ALPHA.
            2 if frames > 60 => {
                let Some(m) = self.probe_output_box(cube_box.0, cube_box.1, cube_box.2, cube_box.3, 3)
                else {
                    return;
                };
                if m < 0.99 {
                    if since > budget {
                        println!("CRYPTO TEST FAIL: cube matte probed {m:.4} (wanted 1.0)");
                        std::process::exit(1);
                    }
                    return;
                }
                self.test_dump("crypto-cube");
                // The rebuild minted the node's reader set and the rig
                // holds one reader per source, all off the one loaded file.
                let minted = matches!(
                    self.egraph.node(sid).map(|n| &n.kind),
                    Some(NodeKind::Cryptomatte(c)) if c.minted()
                );
                if !minted || self.loaded.len() != 2 {
                    println!(
                        "CRYPTO TEST FAIL: minted={minted}, {} loaded files (wanted 2: \
                         the two positional plates; the ranks read the first)",
                        self.loaded.len()
                    );
                    std::process::exit(1);
                }
                // Now look at it from DOWNSTREAM: a MatteIn over the plate,
                // viewed. The node is no longer the view target, so its id
                // layers leave the rig and its one matte reader comes in.
                // Unbaked, it GLOWS: every frame costs six rank decodes,
                // and that is what the bake below takes away.
                if !self.cost_flow.is_heavy(sid) {
                    println!("CRYPTO TEST FAIL: an unbaked cryptomatte must glow (Speed::Heavy)");
                    std::process::exit(1);
                }
                let mid = self.egraph.add(NodeKind::MatteIn, egui::pos2(300.0, 160.0));
                self.egraph.connect(mid, 0, 1);
                self.egraph.connect(mid, 1, sid);
                self.egraph.view = Some(mid);
                self.test_crypto_since = None;
                self.test_crypto = Some((21, 0, sid));
                return;
            }
            // Downstream gets the same matte — from the decode-side bake.
            21 if frames > 30 => {
                let Some(m) = self.probe_output_box(cube_box.0, cube_box.1, cube_box.2, cube_box.3, 3)
                else {
                    return;
                };
                let subs: Vec<usize> = self
                    .reader_slots
                    .iter()
                    .filter(|(i, _)| i.node == sid)
                    .map(|(i, _)| i.sub)
                    .collect();
                if m < 0.99 || subs != [cm::MATTE_SUB] {
                    if since > budget {
                        println!(
                            "CRYPTO TEST FAIL: downstream matte probed {m:.4} with the node's \
                             readers at subs {subs:?} (wanted 1.0 off the matte reader alone)"
                        );
                        std::process::exit(1);
                    }
                    return;
                }
                // The frame that was just asked for baked itself: it is in
                // the NVMe tier under THIS selection's name. (No cache dir,
                // nothing to look for.)
                if let Some(disk) = self.disk_cache.clone() {
                    let layer = match self.egraph.node(sid).map(|n| &n.kind) {
                        Some(NodeKind::Cryptomatte(c)) => c.matte_layer(),
                        _ => String::new(),
                    };
                    let landed = self
                        .crypto_matte_seq(sid)
                        .and_then(|seq| cryptoui::matte_file(&disk, &seq, &layer, 0))
                        .is_some_and(|f| f.exists());
                    if !landed {
                        if since > budget {
                            println!("CRYPTO TEST FAIL: the requested frame did not bake itself");
                            std::process::exit(1);
                        }
                        return;
                    }
                    // ...and with the whole matte on disk (this plate is
                    // one frame) the glow goes out. The count behind it is
                    // throttled, so it gets the same budget as the probe.
                    if self.cost_flow.is_heavy(sid) {
                        if since > budget {
                            println!(
                                "CRYPTO TEST FAIL: the cryptomatte still glows with its matte \
                                 baked into the NVMe tier"
                            );
                            std::process::exit(1);
                        }
                        return;
                    }
                }
                // Back onto the node for the rest.
                self.egraph.view = Some(sid);
                if let Some(editor::ENode { kind: NodeKind::Cryptomatte(c), .. }) =
                    self.egraph.node_mut(sid)
                {
                    c.invert = true;
                }
                self.test_crypto_since = None;
                self.test_crypto = Some((3, 0, sid));
                return;
            }
            // Inverted: the same box goes black — and it is a uniform, so
            // the topology hash must not have moved.
            3 if frames > 30 => {
                let Some(m) = self.probe_output_box(cube_box.0, cube_box.1, cube_box.2, cube_box.3, 3)
                else {
                    return;
                };
                if m > 0.01 {
                    if since > budget {
                        println!("CRYPTO TEST FAIL: inverted cube matte probed {m:.4}");
                        std::process::exit(1);
                    }
                    return;
                }
                if let Some(editor::ENode { kind: NodeKind::Cryptomatte(c), .. }) =
                    self.egraph.node_mut(sid)
                {
                    c.invert = false;
                    c.picks.clear();
                }
                self.test_crypto_since = None;
                self.test_crypto = Some((4, 0, sid));
                return;
            }
            // Viewed: the ids picture — red is the cube id's low chunk over
            // 2048, exactly.
            4 if frames > 30 => {
                let Some(m) = self.probe_output_box(cube_box.0, cube_box.1, cube_box.2, cube_box.3, 0)
                else {
                    return;
                };
                let want = cm::encode_id(cm::id_of_name("cube"))[0].to_f32() / cm::CHUNK_SCALE;
                if (m - want).abs() > 1e-3 {
                    if since > budget {
                        println!("CRYPTO TEST FAIL: ids view probed {m:.4}, wanted {want:.4}");
                        std::process::exit(1);
                    }
                    return;
                }
                self.test_dump("crypto-ids");
                // A PLAIN click on the cube: not a pick, nothing changes.
                click(raw, at(cube_px), none, true);
                self.test_crypto_since = None;
                self.test_crypto = Some((41, 0, sid));
                return;
            }
            41 if frames > 2 => {
                click(raw, at(cube_px), none, false);
                self.test_crypto_since = None;
                self.test_crypto = Some((5, 0, sid));
                return;
            }
            // Give a readback every chance to land, then: still nothing.
            5 if since > web_time::Duration::from_millis(600) => {
                let picks = picked(&self.egraph);
                if !picks.is_empty() {
                    println!("CRYPTO TEST FAIL: a plain click picked {picks:?}");
                    std::process::exit(1);
                }
                // Cmd-click the cube: ADDS.
                click(raw, at(cube_px), cmd, true);
                self.test_crypto_since = None;
                self.test_crypto = Some((51, 0, sid));
                return;
            }
            51 if frames > 2 => {
                click(raw, at(cube_px), cmd, false);
                raw.events.insert(0, egui::Event::ModifiersChanged(none));
                self.test_crypto_since = None;
                self.test_crypto = Some((52, 0, sid));
                return;
            }
            52 if frames > 10 => {
                let picks = picked(&self.egraph);
                if picks.is_empty() {
                    if since > retry {
                        self.test_crypto_since = None;
                        self.test_crypto = Some((5, 0, sid)); // click again
                    }
                    return;
                }
                if picks != ["cube"] {
                    println!("CRYPTO TEST FAIL: cmd-click picked {picks:?}");
                    std::process::exit(1);
                }
                // Cmd-click the sphere: adds to it.
                click(raw, at(sphere_px), cmd, true);
                self.test_crypto_since = None;
                self.test_crypto = Some((53, 0, sid));
                return;
            }
            53 if frames > 2 => {
                click(raw, at(sphere_px), cmd, false);
                raw.events.insert(0, egui::Event::ModifiersChanged(none));
                self.test_crypto_since = None;
                self.test_crypto = Some((54, 0, sid));
                return;
            }
            54 if frames > 10 => {
                let picks = picked(&self.egraph);
                if picks.len() < 2 {
                    if since > retry {
                        self.test_crypto_since = None;
                        self.test_crypto = Some((52, 0, sid)); // click again
                    }
                    return;
                }
                if picks != ["cube", "sphere"] {
                    println!("CRYPTO TEST FAIL: the second cmd-click gave {picks:?}");
                    std::process::exit(1);
                }
                // Cmd+shift-click the lit cube: REMOVES it, sphere stays.
                click(raw, at(cube_px), cmd_shift, true);
                self.test_crypto_since = None;
                self.test_crypto = Some((55, 0, sid));
                return;
            }
            55 if frames > 2 => {
                click(raw, at(cube_px), cmd_shift, false);
                raw.events.insert(0, egui::Event::ModifiersChanged(none));
                self.test_crypto_since = None;
                self.test_crypto = Some((6, 0, sid));
                return;
            }
            6 if frames > 10 => {
                let picks = picked(&self.egraph);
                if picks.len() != 1 {
                    if since > retry {
                        self.test_crypto_since = None;
                        self.test_crypto = Some((54, 0, sid)); // click again
                    }
                    return;
                }
                if picks != ["sphere"] {
                    println!("CRYPTO TEST FAIL: cmd+shift-click on the cube left {picks:?}");
                    std::process::exit(1);
                }
                // ...and the lit sphere.
                click(raw, at(sphere_px), cmd_shift, true);
                self.test_crypto_since = None;
                self.test_crypto = Some((61, 0, sid));
                return;
            }
            61 if frames > 2 => {
                click(raw, at(sphere_px), cmd_shift, false);
                raw.events.insert(0, egui::Event::ModifiersChanged(none));
                self.test_crypto_since = None;
                self.test_crypto = Some((7, 0, sid));
                return;
            }
            7 if frames > 10 => {
                let picks = picked(&self.egraph);
                if !picks.is_empty() {
                    if since > retry {
                        self.test_crypto_since = None;
                        self.test_crypto = Some((6, 0, sid)); // click again
                    }
                    return;
                }
                println!(
                    "CRYPTO TEST PASS: unbaked it glows and a baked matte clears it, \
                     manifest read, one cable in, reader slots minted and decoded, \
                     cube matte solid, downstream reads it off the ONE matte reader and the \
                     requested frame baked itself, invert clears it, ids picture exact, a plain click \
                     picked nothing, cmd-click added 'cube' then 'sphere' through the \
                     manifest, and cmd+shift-click on the lit objects removed them"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        let frames = if next == stage { frames } else { 0 };
        if next != stage {
            self.test_crypto_since = None;
        }
        self.test_crypto = Some((next, frames, sid));
    }
}
