//! `--test-harp`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-harp: viewing a harp string costs what viewing the
    /// equivalent ChannelSelect costs, and shows the same picture.
    ///
    /// The rig holds every reader the graph could show, but the player
    /// decodes only what the viewer slots sample (`Player::set_active`).
    /// Measured structurally — frames delivered per frame displayed over a
    /// steady window, and rig rebuilds — because throughput on a small
    /// plate proves nothing and wobbles with load. Pixels are checked too:
    /// a mask that activated the WRONG reader would be just as cheap.
    ///   A  select on `diffuse` viewed: one reader decodes, not all three.
    ///   B  a harp over the same plate, its `diffuse` string viewed: still
    ///      one decode, the rig grew by every string, and the picture is
    ///      the select's picture; its `N` string is a different picture.
    ///   C  select on A, harp `N` string parked on slot B: two decode (the
    ///      other slot is kept hot), and neither assignment rebuilt the rig.
    ///   D  flip to B: no rebuild, and the `N` picture is up.
    pub(crate) fn drive_test_harp(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, t0, mut m)) = self.test_harp else { return };
        if t0.elapsed().as_secs_f32() > 45.0 {
            println!("HARP TEST FAIL: timed out at stage {stage}");
            std::process::exit(1);
        }
        let fail = |msg: String| -> ! {
            println!("HARP TEST FAIL: {msg}");
            std::process::exit(1);
        };
        // Reader frames delivered to the compositor per shown tick — the
        // uploads a set carries, which is exactly the ACTIVE readers. Not
        // decodes: with the ring resident a second pass over eight frames
        // decodes nothing, and the number this suite is about is how many
        // readers a view keeps fed, cached or not.
        let delivered = self.uploaded_total;
        let rig = self.playback.as_ref().map(|pb| pb.slots.len()).unwrap_or(0);
        let el = t0.elapsed().as_secs_f32();
        let (settle, window) = (1.5f32, 2.5f32);
        // Frames delivered per frame shown, over the window since `m` was
        // marked — None if the rig was rebuilt inside it (the player, and
        // its counter, would have been replaced).
        let ratio = |m: &[u64; 8], rig_builds: u64, displayed: u64| -> Option<f32> {
            (rig_builds == m[0]).then(|| {
                delivered.saturating_sub(m[1]) as f32
                    / displayed.saturating_sub(m[2]).max(1) as f32
            })
        };
        let mark = |m: &mut [u64; 8], rb: u64, shown: u64| {
            m[0] = rb;
            m[1] = delivered;
            m[2] = shown;
        };
        let probe = || self.probe_output_box(64, 128, 32, 32, 0);
        let named: Vec<String> = self
            .loaded
            .get(&0)
            .map(|l| l.layers.iter().map(|d| d.name.clone()).filter(|n| !n.is_empty()).collect())
            .unwrap_or_default();
        let idx = |name: &str| named.iter().position(|n| n == name);
        let breakout = self
            .egraph
            .nodes
            .iter()
            .find(|(_, n)| matches!(n.kind, NodeKind::Breakout(_)))
            .map(|(id, _)| *id);
        let select = {
            let kids = breakout::kid_set(&self.egraph);
            self.egraph
                .nodes
                .iter()
                .find(|(id, n)| {
                    matches!(n.kind, NodeKind::ChannelSelect(_)) && !kids.contains(id)
                })
                .map(|(id, _)| *id)
        };
        let (rb, shown) = (self.rig_builds, self.displayed_total);
        match stage {
            0 if el > 0.5 => {
                if idx("diffuse").is_none() || idx("N").is_none() {
                    return; // layers not probed yet
                }
                if let Some(NodeKind::Read { multi, .. }) =
                    self.egraph.node_mut(0).map(|n| &mut n.kind)
                {
                    *multi = true;
                }
                let pos = self.egraph.node(0).map(|n| n.pos).unwrap_or_default();
                let cs = self.egraph.add(
                    NodeKind::ChannelSelect(engine::graph::ChannelSelectParams {
                        source: usize::MAX,
                        layer: "diffuse".into(),
                    }),
                    pos + egui::vec2(-300.0, 150.0),
                );
                self.egraph.connect(cs, 0, 0);
                self.egraph.view = Some(cs);
                self.egraph.view_out = 0;
                self.egraph.view_active_b = false;
                self.playback_dirty = true;
                self.test_harp = Some((1, Instant::now(), m));
            }
            1 if el > settle => {
                mark(&mut m, rb, shown);
                self.test_harp = Some((2, Instant::now(), m));
            }
            2 if el > window => {
                let Some(r) = ratio(&m, rb, shown) else {
                    fail("window A: the rig rebuilt during steady playback".into())
                };
                // Three readers are instanced (both default plates and the
                // select); only the viewed select may decode.
                if r > 1.3 {
                    fail(format!("A: {r:.2} frames delivered per shown, want ~1 (rig {rig})"));
                }
                let Some(p_sel) = probe() else { return };
                m[6] = p_sel.to_bits() as u64;
                m[5] = rig as u64;
                let pos = self.egraph.node(0).map(|n| n.pos).unwrap_or_default();
                let (b, _r) = breakout::spawn_pair(
                    &mut self.egraph,
                    pos + egui::vec2(0.0, 200.0),
                    named.clone(),
                );
                self.egraph.connect(b, 0, 0);
                breakout::sync(&mut self.egraph);
                self.egraph.view = Some(b);
                self.egraph.view_out = idx("N").unwrap_or(0) as u32;
                self.playback_dirty = true;
                println!("HARP DIAG A: rig {rig}, {r:.2} delivered/shown, probe {p_sel:.4}");
                self.test_harp = Some((3, Instant::now(), m));
            }
            // The N string first, just long enough to read its picture.
            3 if el > settle => {
                let Some(p_n) = probe() else { return };
                m[7] = p_n.to_bits() as u64;
                self.egraph.view_out = idx("diffuse").unwrap_or(0) as u32;
                self.test_harp = Some((4, Instant::now(), m));
            }
            4 if el > 0.8 => {
                mark(&mut m, rb, shown);
                self.test_harp = Some((5, Instant::now(), m));
            }
            5 if el > window => {
                let Some(r) = ratio(&m, rb, shown) else {
                    fail("window B: a view flip between strings rebuilt the rig".into())
                };
                let rig_a = m[5] as usize;
                if rig < rig_a + named.len() {
                    fail(format!("B: rig {rig} readers, want the harp's {} strings instanced on top of {rig_a}", named.len()));
                }
                if r > 1.3 {
                    fail(format!("B: {r:.2} frames delivered per shown with one string viewed (rig {rig})"));
                }
                let Some(p_d) = probe() else { return };
                let p_sel = f32::from_bits(m[6] as u32);
                let p_n = f32::from_bits(m[7] as u32);
                if (p_d - p_sel).abs() > 2e-3 || p_d < 0.01 {
                    fail(format!("B: the diffuse string shows {p_d:.4}, its select showed {p_sel:.4}"));
                }
                if (p_n - p_d).abs() < 0.05 {
                    fail(format!("B: the N string ({p_n:.4}) looks like the diffuse one ({p_d:.4})"));
                }
                println!("HARP DIAG B: rig {rig}, {r:.2} delivered/shown, diffuse {p_d:.4}, N {p_n:.4}");
                // C: the select back on slot A, the N string parked on B.
                self.egraph.view = select;
                self.egraph.view_out = 0;
                self.egraph.view_b = breakout;
                self.egraph.view_b_out = idx("N").unwrap_or(0) as u32;
                self.egraph.view_active_b = false;
                m[3] = rb;
                m[4] = shown;
                self.test_harp = Some((6, Instant::now(), m));
            }
            6 if el > settle => {
                mark(&mut m, rb, shown);
                self.test_harp = Some((7, Instant::now(), m));
            }
            7 if el > window => {
                if rb != m[3] {
                    fail(format!("C: assigning the viewer slots rebuilt the rig {} times", rb - m[3]));
                }
                let Some(r) = ratio(&m, rb, shown) else { return };
                if !(1.6..=2.6).contains(&r) {
                    fail(format!("C: {r:.2} delivered per shown with a select on A and a string on B, want ~2 (slot B hot)"));
                }
                println!("HARP DIAG C: rig {rig}, {r:.2} delivered/shown (A + hot B)");
                self.egraph.view_active_b = true;
                m[3] = rb;
                m[4] = shown;
                self.test_harp = Some((8, Instant::now(), m));
            }
            8 if shown >= m[4] + 3 => {
                if rb != m[3] {
                    fail("D: flipping to slot B rebuilt the rig".into());
                }
                let Some(p) = probe() else { return };
                let p_n = f32::from_bits(m[7] as u32);
                if (p - p_n).abs() > 2e-3 {
                    fail(format!("D: slot B shows {p:.4}, the N string is {p_n:.4}"));
                }
                // E: PAUSED, view a reader nothing was decoding (the second
                // plate). The tick on screen was taken before it woke, so
                // unless something re-shows that tick the view renders a
                // stale texture forever.
                self.playing = false;
                self.test_harp = Some((9, Instant::now(), m));
            }
            9 if el > 0.6 => {
                m[3] = rb;
                m[4] = shown;
                m[5] = self.shown_tick;
                self.egraph.view = Some(1);
                self.egraph.view_out = 0;
                self.egraph.view_active_b = false;
                self.test_harp = Some((10, Instant::now(), m));
            }
            10 => {
                if rb != m[3] {
                    fail("E: a paused view change rebuilt the rig".into());
                }
                if shown <= m[4] {
                    if el > 5.0 {
                        fail("E: a paused view change onto a cold reader never re-showed the tick".into());
                    }
                    return;
                }
                if self.shown_tick != m[5] {
                    fail(format!("E: the re-show moved the paused playhead {} -> {}", m[5], self.shown_tick));
                }
                // F: a string nodule PICKED, then the cursor walked away.
                // The view keys follow the selection, not the hover — a
                // nodule you have to keep hovering is a nodule you cannot
                // aim a viewer at.
                let Some(bar) = breakout else { return };
                let Some((_, r)) = self.node_rects.iter().find(|(id, _)| *id == bar)
                else {
                    return;
                };
                let (pos, size) = match self.egraph.node(bar) {
                    Some(n) => (n.pos, n.size),
                    None => return,
                };
                let Some(gx) = breakout::string_x(
                    &self.egraph.node(bar).unwrap().kind,
                    pos,
                    1,
                    size,
                ) else {
                    return;
                };
                let z = self.node_editor.zoom;
                // A bar's recorded rect is centred on its SLAB, which
                // leans by its porch — the node itself is dx back from it.
                let dx = breakout::wide_rect(&self.egraph.node(bar).unwrap().kind)
                    .map(|(_, dx)| dx)
                    .unwrap_or(0.0);
                let node_x = r.center().x - dx * size * z;
                let nodule = egui::pos2(
                    node_x + (gx - pos.x) * z,
                    r.center().y + (editor::NODE_R * 0.55 + editor::STALK) * size * z,
                );
                // Somewhere with nothing under it, near enough to stay in
                // the graph pane: the keys only assign while it is hovered.
                let rects: Vec<egui::Rect> =
                    self.node_rects.iter().map(|(_, r)| r.expand(30.0)).collect();
                let Some(away) = [
                    egui::vec2(0.0, 110.0),
                    egui::vec2(0.0, 160.0),
                    egui::vec2(-90.0, 110.0),
                    egui::vec2(90.0, 160.0),
                    egui::vec2(0.0, 220.0),
                ]
                .into_iter()
                .map(|d| nodule + d)
                .find(|p| !rects.iter().any(|r| r.contains(*p)))
                else {
                    fail("F: nowhere empty to park the cursor".into())
                };
                m[5] = nodule.x.to_bits() as u64;
                m[6] = nodule.y.to_bits() as u64;
                m[7] = ((away.x.to_bits() as u64) << 32) | away.y.to_bits() as u64;
                raw.events.push(egui::Event::PointerMoved(nodule));
                self.test_harp = Some((11, Instant::now(), m));
            }
            11 => {
                let nodule = egui::pos2(
                    f32::from_bits(m[5] as u32),
                    f32::from_bits(m[6] as u32),
                );
                for pressed in [true, false] {
                    raw.events.push(egui::Event::PointerButton {
                        pos: nodule,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: Default::default(),
                    });
                }
                self.test_harp = Some((12, Instant::now(), m));
            }
            12 => {
                // Away from every nodule and every node, then press 1.
                let away = egui::pos2(
                    f32::from_bits((m[7] >> 32) as u32),
                    f32::from_bits(m[7] as u32),
                );
                raw.events.push(egui::Event::PointerMoved(away));
                self.test_harp = Some((13, Instant::now(), m));
            }
            13 => {
                for pressed in [true, false] {
                    raw.events.push(egui::Event::Key {
                        key: egui::Key::Num1,
                        physical_key: None,
                        pressed,
                        repeat: false,
                        modifiers: Default::default(),
                    });
                }
                self.test_harp = Some((14, Instant::now(), m));
            }
            14 => {
                let Some(bar) = breakout else { return };
                if self.egraph.view != Some(bar) || self.egraph.view_out != 1 {
                    fail(format!(
                        "F: 1 with the cursor off the nodule viewed {:?} out {}, want the picked string 1 of {bar}",
                        self.egraph.view, self.egraph.view_out
                    ));
                }
                println!(
                    "HARP TEST PASS: one string decodes like its select (same picture, ~1 read per frame), \
                     the rig holds the whole harp without rebuilding on view flips, slot B stays hot, \
                     a paused view onto a cold reader re-shows its tick in place, \
                     and the view keys follow a PICKED nodule with the cursor elsewhere"
                );
                std::process::exit(0);
            }
            _ => {}
        }
    }
}
