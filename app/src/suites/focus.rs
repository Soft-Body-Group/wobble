//! `--test-focus`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-focus: the Defocus focus pick and focal-plane view, live.
    ///
    /// Depth is a ToData off the plate, so every pixel has its own value.
    /// Arming the pick and holding the pointer at a point must land the Z
    /// texture's value there in `focus` — checked against the ToData's own
    /// output at that pixel — and the focal-plane view must then show that
    /// pixel green (in focus) while the Defocus is viewed.
    pub(crate) fn drive_test_focus(&mut self) {
        let Some((stage, mut frames, dn)) = self.test_focus else { return };
        frames += 1;
        if frames > 20000 {
            println!("FOCUS TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let fail = |m: String| -> ! {
            println!("FOCUS TEST FAIL: {m}");
            std::process::exit(1);
        };
        const AT: [f32; 2] = [0.37, 0.41];
        let focus = |app: &Self| match app.egraph.node(dn).map(|n| &n.kind) {
            Some(NodeKind::Defocus(d)) => d.focus,
            _ => f32::NAN,
        };
        let px_at = |app: &Self| -> Option<[f32; 3]> {
            let (rgb, w, h) = app.comp.read_output_rgb()?;
            let x = ((AT[0] * w as f32) as usize).min(w as usize - 1);
            let y = ((AT[1] * h as f32) as usize).min(h as usize - 1);
            let i = (y * w as usize + x) * 3;
            Some([rgb[i], rgb[i + 1], rgb[i + 2]])
        };
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                let z = self
                    .egraph
                    .add(NodeKind::ToData(Default::default()), egui::pos2(360.0, 40.0));
                let d = self
                    .egraph
                    .add(NodeKind::Defocus(Default::default()), egui::pos2(320.0, 120.0));
                // The picture is the plate at a THIRD of its value, the Z
                // the plate itself: a pick that read the rgba input would
                // land a third of the right number.
                let gr = self.egraph.add(
                    NodeKind::Grade(engine::graph::GradeParams {
                        gain: [0.3; 3],
                        ..Default::default()
                    }),
                    egui::pos2(280.0, 40.0),
                );
                self.egraph.connect(gr, 0, 0);
                self.egraph.connect(z, 0, 0);
                self.egraph.connect(d, 0, gr);
                self.egraph.connect(d, 2, z);
                self.egraph.view = Some(d);
                // One frame for every read: the plate moves, and a pick at
                // one tick checked against the pass at another is noise.
                self.playing = false;
                self.seek_to(5);
                self.focus_pick.toggle(d);
                self.test_focus = Some((1, frames, d));
            }
            1 if frames > 10 => {
                // The press, held: the viewer gesture sets exactly this.
                self.focus_pick.cursor = Some(AT);
                self.test_focus = Some((2, 0, dn));
            }
            2 => {
                if (focus(self) - 10.0).abs() > 1e-6 {
                    // Landed. Look at the depth pass itself there.
                    self.focus_pick.cursor = None;
                    self.focus_pick.armed = None;
                    let z = self.egraph.input_of(dn, 2).unwrap();
                    self.egraph.view = Some(z);
                    self.test_focus = Some((3, 0, dn));
                } else {
                    self.test_focus = Some((2, frames, dn));
                }
            }
            3 if frames > 20
                && self.test_focus_t.get_or_insert_with(web_time::Instant::now).elapsed()
                    > web_time::Duration::from_millis(1500) =>
            {
                self.test_focus_t = None;
                let Some(p) = px_at(self) else { fail("no output to read".into()) };
                let (got, want) = (focus(self), p[0]);
                if (got - want).abs() > want.abs() * 2e-3 + 1e-4 {
                    fail(format!("picked focus {got} but the depth pass there reads {want}"));
                }
                if let Some(editor::ENode { kind: NodeKind::Defocus(d), .. }) =
                    self.egraph.node_mut(dn)
                {
                    d.focal_view = true;
                }
                self.egraph.view = Some(dn);
                self.test_focus = Some((4, 0, dn));
            }
            4 if frames > 20
                && self.test_focus_t.get_or_insert_with(web_time::Instant::now).elapsed()
                    > web_time::Duration::from_millis(1500) =>
            {
                let Some(p) = px_at(self) else { fail("no output to read".into()) };
                if !(p[1] > p[0] && p[1] > p[2]) {
                    fail(format!("the picked pixel is not green in the focal-plane view: {p:?}"));
                }
                println!(
                    "FOCUS TEST PASS: the pick landed the depth pass value under the pointer \
                     ({}) in focus, and the focal-plane view shows that pixel in focus",
                    focus(self)
                );
                std::process::exit(0);
            }
            _ => self.test_focus = Some((stage, frames, dn)),
        }
    }
}
