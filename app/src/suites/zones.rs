//! `--test-zones`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-tab driver: Tab over the graph background opens the search,
    /// typing filters, Enter adds the first hit.
    /// --test-zones: a Grade's ranges, driven through the REAL panel — the
    /// ranges tab, "add zones", every zone tab, split, dissolve and the matte
    /// preview. "Adding ranges to grade node causes crash" is a claim about
    /// that path; the engine half has its own GPU tests. A panic anywhere
    /// kills the process before the PASS line, and each step says on stderr
    /// what it is about to press, so a crash names the step.
    pub(crate) fn drive_test_zones(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, frames, gid)) = self.test.zones else { return };
        let frames = frames + 1;
        let t0 = *self.test.zones_t0.get_or_insert_with(Instant::now);
        if t0.elapsed() > web_time::Duration::from_secs(90) {
            println!("ZONES TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let fail = |m: String| -> ! {
            println!("ZONES TEST FAIL: {m}");
            std::process::exit(1);
        };
        let rect_of = |s: &Self, id: egui::Id| widgets::noted_rect(&s.egui_ctx, id);
        let zones = |s: &Self| match s.egraph.node(gid).map(|n| &n.kind) {
            Some(NodeKind::Grade(g)) => g.zone_count(),
            _ => usize::MAX,
        };
        let press = |raw: &mut egui::RawInput, p: egui::Pos2, down: bool| {
            raw.events.push(egui::Event::PointerMoved(p));
            raw.events.push(egui::Event::PointerButton {
                pos: p,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers: Default::default(),
            });
        };
        let tab = |t: &'static str| egui::Id::new(("panel-tab-rect", t));
        let plan: [(egui::Id, &str); 11] = [
            (tab("ranges"), "the ranges tab"),
            (egui::Id::new("grade-add-zones-rect"), "add zones"),
            (tab("zone:0"), "the shadows tab"),
            (tab("zone:1"), "the midtones tab"),
            (tab("zone:2"), "the highlights tab"),
            (tab("ranges"), "back to ranges"),
            (egui::Id::new("band-split-rect"), "split a band"),
            (egui::Id::new("band-dissolve-rect"), "dissolve a band"),
            (egui::Id::new(("zone-preview-rect", "matte")), "the matte preview"),
            (egui::Id::new(("zone-preview-rect", "zones")), "the zones preview"),
            (egui::Id::new(("zone-preview-rect", "hard")), "the hard preview"),
        ];
        // The preview the RENDERER sees: armed on the lowered graph, which
        // it is because this node is the view.
        let armed = |s: &Self| match s.egraph.to_engine_graph().nodes.get(&gid) {
            Some(NodeKind::Grade(g)) => g.zones.as_ref().and_then(|z| z.live_preview()).map(|p| p.0),
            _ => None,
        };
        // What must be true once step `done` has had its frames.
        let check = |s: &Self, done: usize| -> Result<(), String> {
            use engine::graph::ZoneView;
            let view = match done {
                8 => Some(ZoneView::Matte),
                9 => Some(ZoneView::Zones),
                10 => Some(ZoneView::Hard),
                _ => None,
            };
            if let Some(view) = view {
                let got = armed(s);
                return if got == Some(view) {
                    Ok(())
                } else {
                    Err(format!(
                        "after `{}` the renderer's preview is {got:?}, want {view:?} (view {:?}, grade {gid})",
                        plan[done].1,
                        s.egraph.view
                    ))
                };
            }
            let want = match done {
                1 | 7 => 3,
                6 => 4,
                _ => return Ok(()),
            };
            let got = zones(s);
            if got == want {
                Ok(())
            } else {
                Err(format!("after `{}` the grade has {got} zones, want {want}", plan[done].1))
            }
        };
        let n = plan.len() as u32;
        let mut next = stage;
        match stage {
            0 => {
                let g = self.egraph.add(
                    NodeKind::Grade(Default::default()),
                    egui::Pos2::new(460.0, 0.0),
                );
                self.egraph.connect(g, 0, 0);
                self.egraph.view = Some(g);
                self.prop.order = vec![g];
                self.test.zones = Some((1, 0, g));
                return;
            }
            s if (1..=3 * n).contains(&s) => {
                let k = ((s - 1) / 3) as usize;
                match (s - 1) % 3 {
                    // Frames are not time: the panel animates and the comp
                    // renders on the WALL clock, so a step also waits a beat.
                    0 if frames > 24
                        && self.layout_settled()
                        && self.test.zones_since.elapsed().as_secs_f32() > 0.4 =>
                    {
                        if k > 0 {
                            if let Err(e) = check(self, k - 1) {
                                fail(e);
                            }
                        }
                        // The graded picture, before any preview replaces
                        // it: what the zones map is judged against.
                        if k == 8 {
                            self.test.zones_px = self.output_mean_rgb();
                        }
                        // The BLENDED map is on screen: plainly not the graded
                        // picture it replaced (not matched to the palette —
                        // the view transform shades it like any pixel, and
                        // the engine test pins the maths), and blended: the
                        // plate's grain runs through its crossfade, so it
                        // shows far more colours than it has zones.
                        if k == 10 {
                            let (Some(graded), Some(map), Some(census)) =
                                (self.test.zones_px, self.output_mean_rgb(), self.output_census())
                            else {
                                fail("could not read the output back".into());
                            };
                            let moved =
                                (0..3).map(|c| (map[c] - graded[c]).abs()).fold(0.0, f32::max);
                            if moved < 0.08 {
                                fail(format!(
                                    "the zones map left the picture as it was: mean {map:?} vs \
                                     graded {graded:?}"
                                ));
                            }
                            if census.len() < 50 {
                                fail(format!(
                                    "the blended map shows only {} colours — no crossfade",
                                    census.len()
                                ));
                            }
                            let _ = self
                                .dump_output_png(std::path::Path::new("target/zones-preview.png"));
                        }
                        let Some((r, clip)) = rect_of(self, plan[k].0) else {
                            if frames > 600 {
                                fail(format!("`{}` was never drawn", plan[k].1));
                            }
                            self.test.zones = Some((stage, frames, gid));
                            return;
                        };
                        // A control drawn past its tile's edge has a rect
                        // and no pixels: pressable here, invisible to a
                        // hand. The preview row sat 23pt below the fold.
                        if !clip.expand(0.5).contains_rect(r) {
                            fail(format!(
                                "`{}` is drawn at {r:?}, outside its visible {clip:?}",
                                plan[k].1
                            ));
                        }
                        eprintln!("ZONES step {k}: pressing {}", plan[k].1);
                        self.test.zones_at = r.center();
                        press(raw, r.center(), true);
                        next = s + 1;
                    }
                    1 => {
                        press(raw, self.test.zones_at, false);
                        next = s + 1;
                    }
                    2 => next = s + 1,
                    _ => {}
                }
            }
            // A second on the wall clock: the uniform lands at once, but the
            // pixels only once a render has run with it.
            s if s == 3 * n + 1 && self.test.zones_since.elapsed().as_secs_f32() > 1.0 => {
                if let Err(e) = check(self, plan.len() - 1) {
                    fail(e);
                }
                // The HARD map: flat colours only — one per zone the plate
                // reaches (plus black for uncovered), whatever its grain.
                // 99% rather than all: a boundary pixel may filter at the blit.
                let Some(census) = self.output_census() else {
                    fail("could not read the output back".into());
                };
                let total: u32 = census.iter().sum();
                let flat: u32 = census.iter().take(zones(self) + 1).sum();
                if (flat as f32) < total as f32 * 0.99 {
                    fail(format!(
                        "the hard map is not flat: its {} commonest colours cover {:.1}% of \
                         {} ({} colours in all)",
                        zones(self) + 1,
                        100.0 * flat as f32 / total.max(1) as f32,
                        total,
                        census.len()
                    ));
                }
                let _ = self.dump_output_png(std::path::Path::new("target/zones-hard.png"));
                println!(
                    "ZONES TEST PASS: ranges tab, add zones (3), shadows/midtones/highlights \
                     tabs, split (4), dissolve (3), the matte preview, the blended zones map \
                     and the hard one ({} colours, flat), all through the real panel",
                    census.len()
                );
                std::process::exit(0);
            }
            _ => {}
        }
        let frames = if next != stage {
            self.test.zones_since = Instant::now();
            0
        } else {
            frames
        };
        self.test.zones = Some((next, frames, gid));
    }
}
