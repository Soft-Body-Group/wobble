//! The hotkey tour: on first launch (and from help > hotkey tour) a card
//! over the window draws the keyboard, lights Scrub's keys a group at a
//! time, and under it acts the group out on a dummy timeline: a pointer
//! that holds a key, presses on a clip and drags, the way Scrub edits.
//!
//! Seen once: `Panes::tour_seen` in ui.json, set when the card closes by
//! any route. Never shown in a headless run (tests, renders, shots), nor
//! when `SCRUB_NO_TOUR` is set; `SCRUB_TOUR=<step>` forces it open.
//!
//! While the card is up the editor's own keys stand down (`Editor::ui`);
//! left / right step through, Esc or a click outside closes.

use egui::{pos2, vec2, Align2, Color32, FontId, Key, Pos2, Rect, Shape, Stroke, StrokeKind};

/// One step: the caps it lights, what it says, and what the dummy does.
struct Step {
    title: &'static str,
    line: &'static str,
    /// Lit together; with a hold animation, pressed one after another.
    keys: &'static [&'static str],
    /// The shift cap is held with them.
    shift: bool,
    /// The cmd cap is held with them.
    cmd: bool,
    anim: Anim,
}

#[derive(Clone, Copy, PartialEq)]
enum Anim {
    Intro,
    Shuttle,
    Marks,
    Edit,
    Targets,
    /// A / D: trim the in / out, the neighbour stays (a gap opens).
    Trim,
    /// Z / C: ripple trim, everything after follows.
    Ripple,
    Slip,
    Speed,
    Dissolve,
    Scrub,
    Panels,
    Done,
}

const STEPS: &[Step] = &[
    Step { title: "Scrub is driven from the keyboard", line: "your left hand stays on the keys, your right on the mouse. here are the ones that matter.", keys: &[], shift: false, cmd: false, anim: Anim::Intro },
    Step { title: "hold A or D, drag a clip   trim", line: "hold the key, press anywhere on a clip and drag: A trims its in, D its out. no aiming for edges.", keys: &["A", "D"], shift: false, cmd: false, anim: Anim::Trim },
    Step { title: "hold S, drag a clip   slip", line: "the clip stays put and shows other source: its in and out move together.", keys: &["S"], shift: false, cmd: false, anim: Anim::Slip },
    Step { title: "hold Z or C, drag a clip   ripple trim", line: "like A and D, but everything after the clip follows, so no gap opens.", keys: &["Z", "C"], shift: false, cmd: false, anim: Anim::Ripple },
    Step { title: "hold shift S, drag a clip   speed", line: "the clip's speed, scaled about the playhead. cmd R types a speed in.", keys: &["S"], shift: true, cmd: false, anim: Anim::Speed },
    Step { title: "J  K  L   shuttle", line: "back, stop, forward. tap again to go faster. space plays and stops.", keys: &["J", "K", "L", "space"], shift: false, cmd: false, anim: Anim::Shuttle },
    Step { title: "I  O   mark in and out", line: "on the source or the record side, whichever has focus (or is under the pointer).", keys: &["I", "O"], shift: false, cmd: false, anim: Anim::Marks },
    Step { title: ",  .   insert and overwrite", line: "the source in..out goes into the timeline at the record marks, on the targeted tracks.", keys: &[",", "."], shift: false, cmd: false, anim: Anim::Edit },
    Step { title: "1  2  3   target tracks", line: "V1 V2 V3 (again: off). the sound target follows. 4-7 target the audio tracks.", keys: &["1", "2", "3"], shift: false, cmd: false, anim: Anim::Targets },
    Step { title: "cmd D   dissolve", line: "on the selected edit, or the cut nearest the playhead. alt = and alt - make it longer or shorter.", keys: &["D"], shift: false, cmd: true, anim: Anim::Dissolve },
    Step { title: "drag the picture   scrub", line: "press on either viewer and drag sideways: frame by frame, relative, no playhead to find.", keys: &[], shift: false, cmd: false, anim: Anim::Scrub },
    Step { title: "Q  W  H   panels", line: "Q: the panel under the pointer alone. W closes it, shift W brings them all back. H hides the key hints.", keys: &["Q", "W", "H"], shift: false, cmd: false, anim: Anim::Panels },
    Step { title: "that's the core of it", line: "the key hints over the viewers list the rest. help > hotkey tour plays this again.", keys: &[], shift: false, cmd: false, anim: Anim::Done },
];

/// Seconds one pass of an animation takes (one key of a pair).
const PASS: f64 = 3.2 / SPEED;
/// How much faster than first drawn the animations run.
const SPEED: f64 = 1.5;
/// Loops of its animation a step plays before the tour moves on.
const LOOPS: f64 = 1.0;
/// A beat on the last frame before the next step starts.
const REST: f64 = 0.4;

/// How long a step stays before the tour moves on by itself: one loop
/// of its animation (a pair of keys loops once both have had a pass).
fn dwell(s: &Step) -> f64 {
    let one = match s.anim {
        Anim::Done => return f64::INFINITY,
        Anim::Trim | Anim::Ripple | Anim::Shuttle | Anim::Marks | Anim::Edit => 2.0 * PASS,
        Anim::Targets => 3.0 * 0.9 / SPEED,
        Anim::Panels => 3.0 * 1.6 / SPEED,
        _ => PASS,
    };
    LOOPS * one + REST
}

pub struct Tour {
    pub open: bool,
    step: usize,
    /// egui time the step started at; `None`: start it on the next frame.
    t0: Option<f64>,
    /// Stepped by hand: no more auto advance.
    manual: bool,
}

impl Tour {
    /// First launch: open unless it was seen (or this run has no window).
    /// `SCRUB_TOUR=<step>` opens it at that step whatever the flag says
    /// (headless too, for shots); `SCRUB_NO_TOUR` keeps it shut.
    pub fn new(seen: bool, headless: bool) -> Self {
        if let Some(n) = std::env::var("SCRUB_TOUR").ok().and_then(|s| s.parse::<usize>().ok()) {
            return Tour { open: true, step: n.min(STEPS.len() - 1), t0: None, manual: false };
        }
        let off = seen || headless || std::env::var_os("SCRUB_NO_TOUR").is_some();
        Tour { open: !off, step: 0, t0: None, manual: false }
    }

    /// help > hotkey tour.
    pub fn replay(&mut self) {
        *self = Tour { open: true, step: 0, t0: None, manual: false };
    }

    fn go(&mut self, step: usize) {
        self.step = step.min(STEPS.len() - 1);
        self.t0 = None;
    }

    /// Draw the card. Returns true on the frame it closes (the caller keeps
    /// the seen flag).
    pub fn show(&mut self, ctx: &egui::Context) -> bool {
        if !self.open {
            return false;
        }
        let now = ctx.input(|i| i.time);
        let t0 = *self.t0.get_or_insert(now);
        let t = now - t0;
        let step = &STEPS[self.step];
        if !self.manual && t > dwell(step) && self.step + 1 < STEPS.len() {
            self.go(self.step + 1);
        }
        let (left, right) = ctx.input_mut(|i| (i.consume_key(egui::Modifiers::NONE, Key::ArrowLeft), i.consume_key(egui::Modifiers::NONE, Key::ArrowRight) || i.consume_key(egui::Modifiers::NONE, Key::Enter)));
        if left && self.step > 0 {
            self.manual = true;
            self.go(self.step - 1);
        }
        if right {
            if self.step + 1 < STEPS.len() {
                self.manual = true;
                self.go(self.step + 1);
            } else {
                self.open = false;
            }
        }
        let mut close = false;
        let modal = egui::Modal::new(egui::Id::new("scrub-hotkey-tour")).show(ctx, |ui| {
          ui.horizontal_top(|ui| {
            // Every step by name: click one to see it again (it then loops).
            ui.vertical(|ui| {
                ui.set_width(LIST_W);
                ui.add_space(4.0);
                ui.label(egui::RichText::new("steps").size(11.0).weak());
                ui.add_space(2.0);
                for (i, s) in STEPS.iter().enumerate() {
                    let on = i == self.step;
                    let text = egui::RichText::new(short(s.anim)).size(13.0);
                    let text = if on { text.color(accent()) } else { text };
                    if ui.add_sized([LIST_W, 22.0], egui::Button::selectable(on, text)).clicked() {
                        self.manual = true;
                        self.go(i);
                    }
                }
            });
            ui.separator();
            ui.vertical(|ui| {
            ui.set_width(CARD_W);
            let step = &STEPS[self.step];
            ui.add_space(2.0);
            ui.label(egui::RichText::new(step.title).size(18.0).strong());
            ui.add_space(2.0);
            ui.label(egui::RichText::new(step.line).size(13.0).weak());
            ui.add_space(10.0);
            let (kb, _) = ui.allocate_exact_size(vec2(CARD_W, KB_H), egui::Sense::hover());
            let (stage, _) = ui.allocate_exact_size(vec2(CARD_W, STAGE_H), egui::Sense::hover());
            let painter = ui.painter_at(kb.union(stage));
            let pressed = pressed_now(step, t);
            keyboard(&painter, kb, step, &pressed, t);
            act(&painter, stage, step, t);
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let last = self.step + 1 == STEPS.len();
                    if ui.button(if last { "start editing" } else { "next" }).clicked() {
                        if last {
                            close = true;
                        } else {
                            self.manual = true;
                            self.go(self.step + 1);
                        }
                    }
                    if self.step > 0 && ui.button("back").clicked() {
                        self.manual = true;
                        self.go(self.step - 1);
                    }
                    if !last && ui.button("skip").on_hover_text("help > hotkey tour plays it again").clicked() {
                        close = true;
                    }
                });
            });
            });
          });
        });
        if modal.should_close() {
            close = true;
        }
        ctx.request_repaint();
        if close {
            self.open = false;
        }
        !self.open
    }
}

const CARD_W: f32 = 640.0;
/// The step list down the card's left.
const LIST_W: f32 = 170.0;
const KB_H: f32 = 190.0;
const STAGE_H: f32 = 150.0;

fn accent() -> Color32 {
    Color32::from_rgb(120, 220, 230)
}
const CLIP: Color32 = Color32::from_rgb(60, 110, 160);
const CLIP_HI: Color32 = Color32::from_rgb(90, 150, 205);
const TRIM: Color32 = Color32::from_rgb(230, 60, 60);
const RIPPLE: Color32 = Color32::from_rgb(240, 210, 60);
const INK: Color32 = Color32::from_rgb(225, 228, 235);
const DIM: Color32 = Color32::from_rgb(120, 124, 135);

/// A step's name in the list.
fn short(a: Anim) -> &'static str {
    match a {
        Anim::Intro => "welcome",
        Anim::Trim => "A  D   trim",
        Anim::Slip => "S   slip",
        Anim::Ripple => "Z  C   ripple trim",
        Anim::Speed => "shift S   speed",
        Anim::Shuttle => "J  K  L   shuttle",
        Anim::Marks => "I  O   marks",
        Anim::Edit => ",  .   insert, overwrite",
        Anim::Targets => "1  2  3   targets",
        Anim::Dissolve => "cmd D   dissolve",
        Anim::Scrub => "drag   scrub",
        Anim::Panels => "Q  W  H   panels",
        Anim::Done => "done",
    }
}

// ------------------------------------------------------------ the keyboard

/// (label, width in key units). "" is a blank cap.
const ROWS: &[&[(&str, f32)]] = &[
    &[("esc", 1.0), ("1", 1.0), ("2", 1.0), ("3", 1.0), ("4", 1.0), ("5", 1.0), ("6", 1.0), ("7", 1.0), ("8", 1.0), ("9", 1.0), ("0", 1.0), ("-", 1.0), ("=", 1.0), ("del", 1.5)],
    &[("tab", 1.5), ("Q", 1.0), ("W", 1.0), ("E", 1.0), ("R", 1.0), ("T", 1.0), ("Y", 1.0), ("U", 1.0), ("I", 1.0), ("O", 1.0), ("P", 1.0), ("[", 1.0), ("]", 1.0), ("\\", 1.0)],
    &[("", 1.75), ("A", 1.0), ("S", 1.0), ("D", 1.0), ("F", 1.0), ("G", 1.0), ("H", 1.0), ("J", 1.0), ("K", 1.0), ("L", 1.0), (";", 1.0), ("'", 1.0), ("return", 1.75)],
    &[("shift", 2.25), ("Z", 1.0), ("X", 1.0), ("C", 1.0), ("V", 1.0), ("B", 1.0), ("N", 1.0), ("M", 1.0), (",", 1.0), (".", 1.0), ("/", 1.0), ("shift", 2.25)],
    &[("", 1.25), ("alt", 1.25), ("cmd", 1.5), ("space", 6.0), ("cmd", 1.5), ("alt", 1.25), ("", 1.25)],
];

/// Keys that do something in Scrub (drawn a shade brighter).
const SCRUB_KEYS: &[&str] = &[
    "esc", "1", "2", "3", "4", "5", "6", "7", "-", "=", "del", "Q", "W", "R", "T", "Y", "U", "I", "O", "[", "]", "\\", "A", "S", "D", "F", "G", "H", "J", "K", "L", "Z", "X", "C", "V", "B", "N", "M", ",", ".", "space",
];

fn cap_label(l: &str) -> String {
    match l {
        "cmd" if !cfg!(target_os = "macos") => "ctrl".into(),
        "space" => String::new(),
        _ => l.into(),
    }
}

/// The caps held down at time `t` of the step.
fn pressed_now(step: &Step, t: f64) -> Vec<&'static str> {
    let mut v = Vec::new();
    let held = |p: f64| (0.08..0.9).contains(&p);
    match step.anim {
        Anim::Trim | Anim::Ripple => {
            let (k, p) = pair(t);
            if held(p) {
                v.push(step.keys[k]);
            }
        }
        Anim::Slip | Anim::Speed => {
            if held(phase(t)) {
                v.extend_from_slice(step.keys);
            }
        }
        Anim::Shuttle => {
            // J, K, L, L (faster), space.
            let i = ((t % (2.0 * PASS)) / (2.0 * PASS) * 5.0) as usize;
            let k = ["J", "K", "L", "L", "space"][i.min(4)];
            if (t * 5.0 / (2.0 * PASS)).fract() < 0.35 {
                v.push(k);
            }
        }
        Anim::Marks | Anim::Edit => {
            let (k, p) = pair(t);
            if (0.25..0.4).contains(&p) {
                v.push(step.keys[k]);
            }
        }
        Anim::Targets => {
            let i = ((t * SPEED / 0.9) as usize) % 3;
            if (t * SPEED / 0.9).fract() < 0.3 {
                v.push(step.keys[i]);
            }
        }
        Anim::Dissolve => {
            if (0.3..0.45).contains(&phase(t)) {
                v.push("D");
            }
        }
        Anim::Panels => {
            let i = ((t * SPEED / 1.6) as usize) % 3;
            if (t * SPEED / 1.6).fract() < 0.25 {
                v.push(step.keys[i]);
            }
        }
        _ => {}
    }
    v
}

/// Where in one pass, 0..1.
fn phase(t: f64) -> f64 {
    (t % PASS) / PASS
}

/// Which key of a pair is acting, and where in its pass.
fn pair(t: f64) -> (usize, f64) {
    (((t / PASS) as usize) % 2, phase(t))
}

fn keyboard(p: &egui::Painter, area: Rect, step: &Step, pressed: &[&str], t: f64) {
    let units = 15.0;
    let u = (area.width() / units).min(area.height() / ROWS.len() as f32);
    let w = u * units;
    let x0 = area.center().x - w * 0.5;
    let pulse = 0.75 + 0.25 * (t * 4.0).sin() as f32;
    for (ri, row) in ROWS.iter().enumerate() {
        let mut x = x0;
        let y = area.top() + ri as f32 * u;
        for &(label, kw) in row.iter() {
            let r = Rect::from_min_size(pos2(x, y), vec2(kw * u, u)).shrink(2.5);
            x += kw * u;
            let lit = step.keys.contains(&label) || (step.shift && label == "shift") || (step.cmd && label == "cmd");
            let down = pressed.contains(&label) || (lit && (label == "shift" || label == "cmd") && !pressed.is_empty());
            let used = SCRUB_KEYS.contains(&label);
            let r = if down { r.translate(vec2(0.0, 2.0)) } else { r };
            let (fill, ink, stroke) = if down {
                (accent(), Color32::from_rgb(20, 30, 36), Stroke::new(1.0, accent()))
            } else if lit {
                (accent().gamma_multiply(0.22 * pulse + 0.08), accent(), Stroke::new(1.5, accent().gamma_multiply(pulse)))
            } else if used {
                (Color32::from_gray(52), INK.gamma_multiply(0.75), Stroke::new(1.0, Color32::from_gray(70)))
            } else {
                (Color32::from_gray(38), DIM.gamma_multiply(0.6), Stroke::new(1.0, Color32::from_gray(50)))
            };
            if !down {
                // The cap's side, under it.
                p.rect_filled(r.translate(vec2(0.0, 2.0)), 4.0, Color32::from_gray(22));
            }
            p.rect_filled(r, 4.0, fill);
            p.rect_stroke(r, 4.0, stroke, StrokeKind::Inside);
            let text = cap_label(label);
            if !text.is_empty() {
                let size = if text.chars().count() > 1 { u * 0.24 } else { u * 0.36 };
                p.text(r.center(), Align2::CENTER_CENTER, text, FontId::proportional(size), ink);
            }
        }
    }
}

// --------------------------------------------------------------- the stage

/// The dummy: a little timeline (and a viewer for scrubbing), the pointer
/// acting on it.
fn act(p: &egui::Painter, area: Rect, step: &Step, t: f64) {
    let area = area.shrink2(vec2(24.0, 8.0));
    p.rect_filled(area.expand(6.0), 6.0, Color32::from_gray(26));
    let lane_h = 34.0;
    let ease = |x: f64| -> f32 { let x = x.clamp(0.0, 1.0); (x * x * (3.0 - 2.0 * x)) as f32 };
    // One lane: three clips, A | B | C, B being acted on.
    let lane = Rect::from_min_size(pos2(area.left(), area.center().y - lane_h * 0.5 + 18.0), vec2(area.width(), lane_h));
    let ux = area.width() / 100.0;
    let at = |x: f32| lane.left() + x * ux;
    let label_by_pointer = |p: &egui::Painter, ptr: Pos2, s: &str| {
        let g = p.layout_no_wrap(s.into(), FontId::proportional(11.0), Color32::WHITE);
        let r = Rect::from_min_size(ptr + vec2(16.0, 12.0), g.size()).expand(3.0);
        // Kept inside the stage: near its rest the pointer sits by the edge.
        let r = if r.right() > area.right() + 4.0 { r.translate(vec2(-(r.width() + 28.0), 0.0)) } else { r };
        let r = if r.bottom() > area.bottom() + 4.0 { r.translate(vec2(0.0, -(r.height() + 22.0))) } else { r };
        p.rect_filled(r, 3.0, Color32::from_black_alpha(210));
        p.galley(r.min + vec2(3.0, 3.0), g, Color32::WHITE);
    };
    let ruler = |p: &egui::Painter| {
        let y = lane.top() - 18.0;
        p.line_segment([pos2(lane.left(), y + 10.0), pos2(lane.right(), y + 10.0)], Stroke::new(1.0, Color32::from_gray(60)));
        for i in 0..=20 {
            let x = lane.left() + i as f32 * lane.width() / 20.0;
            let h = if i % 5 == 0 { 8.0 } else { 4.0 };
            p.line_segment([pos2(x, y + 10.0 - h), pos2(x, y + 10.0)], Stroke::new(1.0, Color32::from_gray(70)));
        }
    };
    let playhead = |p: &egui::Painter, x: f32| {
        p.line_segment([pos2(x, lane.top() - 18.0), pos2(x, lane.bottom() + 6.0)], Stroke::new(1.5, TRIM));
        p.add(Shape::convex_polygon(vec![pos2(x - 5.0, lane.top() - 18.0), pos2(x + 5.0, lane.top() - 18.0), pos2(x, lane.top() - 11.0)], TRIM, Stroke::NONE));
    };
    // (start, end, name, source in) in lane units.
    let clip = |p: &egui::Painter, a: f32, b: f32, name: &str, src: i64, hi: bool, edge: Option<(bool, Color32)>| {
        if b <= a {
            return;
        }
        let r = Rect::from_min_max(pos2(at(a), lane.top()), pos2(at(b), lane.bottom())).shrink2(vec2(0.5, 0.0));
        p.rect_filled(r, 3.0, if hi { CLIP_HI } else { CLIP });
        p.rect_stroke(r, 3.0, Stroke::new(1.0, Color32::from_rgb(30, 50, 75)), StrokeKind::Inside);
        let pc = p.with_clip_rect(r.shrink(1.0));
        pc.text(r.left_top() + vec2(5.0, 3.0), Align2::LEFT_TOP, name, FontId::proportional(11.0), INK);
        pc.text(r.left_bottom() + vec2(5.0, -3.0), Align2::LEFT_BOTTOM, format!("{src}"), FontId::monospace(9.5), INK.gamma_multiply(0.7));
        if let Some((left, c)) = edge {
            let x = if left { r.left() + 1.0 } else { r.right() - 1.0 };
            p.line_segment([pos2(x, r.top()), pos2(x, r.bottom())], Stroke::new(3.0, c));
        }
    };
    let rest = pos2(area.right() - 30.0, area.bottom() - 6.0);
    match step.anim {
        Anim::Trim | Anim::Ripple => {
            ruler(p);
            let ripple = step.anim == Anim::Ripple;
            let (k, ph) = pair(t);
            let out = k == 1;
            // The pointer comes in, presses on B at 0.3, drags to 0.75, lets go.
            let grab = pos2(at(50.0), lane.center().y);
            let d = ease((ph - 0.3) / 0.45) * 10.0 * if out { -1.0 } else { 1.0 };
            let dragging = (0.3..0.8).contains(&ph);
            let d = if ph >= 0.9 { 0.0 } else { d };
            let (mut b0, mut b1, mut c0, mut c1) = (35.0, 65.0, 65.0, 95.0);
            let mut src = 1001;
            if out {
                b1 += d;
                if ripple {
                    c0 += d;
                    c1 += d;
                }
            } else {
                b0 += d;
                src += (d * 2.0) as i64;
                if ripple {
                    // In-trims ripple too: B keeps its start, the rest close up.
                    b0 -= d;
                    b1 -= d;
                    c0 -= d;
                    c1 -= d;
                }
            }
            let col = if ripple { RIPPLE } else { TRIM };
            let down = (0.08..0.9).contains(&ph);
            clip(p, 5.0, 35.0, "A", 1001, false, None);
            clip(p, b0, b1, "B", src, dragging, down.then_some((!out, col)));
            clip(p, c0, c1, "C", 1001, false, None);
            let ptr = pointer_path(ph, rest, grab, pos2(grab.x + d * ux, grab.y));
            if down {
                let op = match (ripple, out) {
                    (false, false) => "trim in",
                    (false, true) => "trim out",
                    (true, false) => "ripple trim in",
                    (true, true) => "ripple trim out",
                };
                label_by_pointer(p, ptr, op);
            }
            pointer(p, ptr, dragging);
        }
        Anim::Slip => {
            ruler(p);
            let ph = phase(t);
            let grab = pos2(at(50.0), lane.center().y);
            let d = (ease((ph - 0.3) / 0.25) - ease((ph - 0.55) / 0.2)) * 12.0;
            let dragging = (0.3..0.8).contains(&ph);
            let src = 1001 - (d * 3.0) as i64;
            clip(p, 5.0, 35.0, "A", 1001, false, None);
            clip(p, 35.0, 65.0, "B", src, dragging, None);
            clip(p, 65.0, 95.0, "C", 1001, false, None);
            // The source behind B, shown sliding while slipping.
            if dragging {
                let full = Rect::from_min_max(pos2(at(35.0 - 15.0 + d), lane.bottom() + 4.0), pos2(at(65.0 + 15.0 + d), lane.bottom() + 10.0));
                p.rect_filled(full, 2.0, CLIP.gamma_multiply(0.45));
                let shown = Rect::from_min_max(pos2(at(35.0), full.top()), pos2(at(65.0), full.bottom()));
                p.rect_filled(shown, 2.0, CLIP_HI);
            }
            let ptr = pointer_path(ph, rest, grab, pos2(grab.x + d * ux, grab.y));
            if (0.08..0.9).contains(&ph) {
                label_by_pointer(p, ptr, &format!("slip  {src}–{}", src + 59));
            }
            pointer(p, ptr, dragging);
        }
        Anim::Speed => {
            ruler(p);
            let ph = phase(t);
            let grab = pos2(at(50.0), lane.center().y);
            let d = ease((ph - 0.3) / 0.45) * 12.0;
            let d = if ph >= 0.9 { 0.0 } else { d };
            let dragging = (0.3..0.8).contains(&ph);
            let len = 30.0 + d;
            let speed = 30.0 / len;
            clip(p, 5.0, 35.0, "A", 1001, false, None);
            clip(p, 35.0, 35.0 + len, &format!("B  ×{speed:.2}"), 1001, dragging, None);
            clip(p, 35.0 + len, 65.0 + d.max(0.0), "C", 1001, false, None);
            playhead(p, at(35.0));
            let ptr = pointer_path(ph, rest, grab, pos2(grab.x + d * ux, grab.y));
            if (0.08..0.9).contains(&ph) {
                label_by_pointer(p, ptr, &format!("speed ×{speed:.2}"));
            }
            pointer(p, ptr, dragging);
        }
        Anim::Shuttle => {
            ruler(p);
            clip(p, 5.0, 50.0, "A", 1001, false, None);
            clip(p, 50.0, 95.0, "B", 1001, false, None);
            // J back, K stop, L forward, L faster, space.
            let seg = (t % (2.0 * PASS)) / (2.0 * PASS) * 5.0;
            let i = seg as usize;
            let (speed, word) = [(-1.0, "◀  1×"), (0.0, "■"), (1.0, "▶  1×"), (2.0, "▶▶  2×"), (0.0, "■")][i.min(4)];
            // Integrate the playhead over the segments so far.
            let speeds = [-1.0f32, 0.0, 1.0, 2.0, 0.0];
            let seg_len = 2.0 * PASS / 5.0;
            let mut x = 50.0f32;
            for s in speeds.iter().take(i) {
                x += s * seg_len as f32 * 8.0 * SPEED as f32;
            }
            x += speed * (seg.fract() * seg_len) as f32 * 8.0 * SPEED as f32;
            playhead(p, at(x.clamp(0.0, 100.0)));
            p.text(pos2(lane.center().x, lane.bottom() + 22.0), Align2::CENTER_CENTER, word, FontId::proportional(14.0), accent());
        }
        Anim::Marks => {
            ruler(p);
            clip(p, 5.0, 95.0, "source clip", 1001, false, None);
            let (k, ph) = pair(t);
            let x_in = 30.0;
            let mark = |p: &egui::Painter, x: f32, s: &str| {
                let y = lane.top() - 18.0;
                p.text(pos2(at(x), y), Align2::CENTER_BOTTOM, s, FontId::proportional(11.0), accent());
                p.line_segment([pos2(at(x), y), pos2(at(x), lane.bottom())], Stroke::new(1.5, accent()));
            };
            let head = if k == 0 { 10.0 + ease(ph / 0.3) * 20.0 } else { 30.0 + ease(ph / 0.3) * 40.0 };
            let in_set = k == 1 || ph > 0.3;
            let out_set = k == 1 && ph > 0.3;
            if in_set {
                mark(p, x_in, "in");
            }
            if out_set {
                mark(p, 70.0, "out");
                let band = Rect::from_min_max(pos2(at(30.0), lane.top()), pos2(at(70.0), lane.bottom()));
                p.rect_filled(band, 0.0, accent().gamma_multiply(0.18));
            }
            playhead(p, at(head));
        }
        Anim::Edit => {
            ruler(p);
            let (k, ph) = pair(t);
            let ins = k == 0;
            let e = ease((ph - 0.4) / 0.3);
            let w = 20.0;
            // The record mark at 35: insert pushes B right, overwrite covers it.
            clip(p, 5.0, 35.0, "A", 1001, false, None);
            if ins {
                clip(p, 35.0 + w * e, 75.0 + w * e, "B", 1001, false, None);
            } else {
                clip(p, 35.0 + w * e, 75.0, "B", 1001 + (w * e * 2.0) as i64, false, None);
            }
            // The source piece, dropping in.
            let y = lane.top() - 46.0 + 46.0 * e;
            let r = Rect::from_min_size(pos2(at(35.0), y), vec2(w * ux * e.max(0.02), lane_h));
            if ph > 0.38 {
                p.rect_filled(r, 3.0, Color32::from_rgb(170, 130, 40));
                p.with_clip_rect(r).text(r.left_top() + vec2(5.0, 3.0), Align2::LEFT_TOP, "src", FontId::proportional(11.0), INK);
            }
            playhead(p, at(35.0));
            p.text(pos2(lane.center().x, lane.bottom() + 22.0), Align2::CENTER_CENTER, if ins { ",  insert" } else { ".  overwrite" }, FontId::proportional(14.0), accent());
        }
        Anim::Targets => {
            let i = ((t * SPEED / 0.9) as usize) % 3;
            for (row, name) in ["V3", "V2", "V1"].iter().enumerate() {
                let tr = 2 - row;
                let y = area.top() + 14.0 + row as f32 * 38.0;
                let tag = Rect::from_min_size(pos2(area.left(), y), vec2(34.0, 30.0));
                let on = tr == i;
                p.rect_filled(tag, 4.0, if on { accent() } else { Color32::from_gray(50) });
                p.text(tag.center(), Align2::CENTER_CENTER, *name, FontId::proportional(12.0), if on { Color32::from_rgb(20, 30, 36) } else { INK });
                let l = Rect::from_min_max(pos2(tag.right() + 8.0, y), pos2(area.right(), y + 30.0));
                p.rect_filled(l, 3.0, if on { Color32::from_gray(44) } else { Color32::from_gray(34) });
                if tr == 0 {
                    p.rect_filled(Rect::from_min_size(l.min, vec2(l.width() * 0.6, 30.0)), 3.0, CLIP);
                }
            }
        }
        Anim::Dissolve => {
            ruler(p);
            let ph = phase(t);
            let e = ease((ph - 0.4) / 0.25);
            clip(p, 5.0, 50.0, "A", 1001, false, None);
            clip(p, 50.0, 95.0, "B", 1001, false, None);
            if e > 0.0 {
                let w = 8.0 * e;
                let r = Rect::from_min_max(pos2(at(50.0 - w), lane.top()), pos2(at(50.0 + w), lane.bottom()));
                p.rect_filled(r, 3.0, Color32::from_rgba_unmultiplied(235, 235, 250, 200));
                p.line_segment([r.left_bottom(), r.right_top()], Stroke::new(1.0, Color32::from_rgb(40, 40, 60)));
            }
            playhead(p, at(52.0));
        }
        Anim::Scrub => {
            // A viewer: the picture, a frame counter, a playhead under it.
            let ph = phase(t);
            let view = Rect::from_center_size(pos2(area.center().x, area.center().y - 6.0), vec2(220.0, 110.0));
            let d = (ease((ph - 0.25) / 0.3) - ease((ph - 0.55) / 0.3) * 0.6) * 60.0;
            let dragging = (0.25..0.85).contains(&ph);
            let frame = 1001 + (d / 5.0 * 3.0) as i64;
            p.rect_filled(view, 2.0, Color32::from_gray(14));
            // A moving "picture": bars that slide with the frame.
            let pc = p.with_clip_rect(view);
            for i in 0..8 {
                let x = view.left() + ((i as f32 * 34.0 + frame as f32 * 3.0) % 272.0) - 26.0;
                pc.rect_filled(Rect::from_min_size(pos2(x, view.top() + 20.0 + (i % 3) as f32 * 20.0), vec2(22.0, 40.0)), 3.0, Color32::from_rgb(40 + i * 18, 70 + i * 9, 110));
            }
            p.text(view.right_bottom() + vec2(-6.0, -4.0), Align2::RIGHT_BOTTOM, format!("{frame}"), FontId::monospace(11.0), INK);
            let bar = Rect::from_min_max(pos2(view.left(), view.bottom() + 6.0), pos2(view.right(), view.bottom() + 9.0));
            p.rect_filled(bar, 1.0, Color32::from_gray(50));
            let hx = view.left() + ((frame - 1001 + 30) as f32 / 90.0).clamp(0.0, 1.0) * view.width();
            p.line_segment([pos2(hx, bar.top() - 3.0), pos2(hx, bar.bottom() + 3.0)], Stroke::new(2.0, TRIM));
            let grab = view.center();
            let ptr = pointer_path(ph, rest, grab, grab + vec2(d, 0.0));
            pointer(p, ptr, dragging);
        }
        Anim::Panels => {
            // Four panels; Q solos the hovered one, W closes it, H the hints.
            let i = ((t * SPEED / 1.6) as usize) % 3;
            let f = (t * SPEED / 1.6).fract() as f32;
            let r = area.shrink(4.0);
            let half = r.width() * 0.5;
            let panes = [
                Rect::from_min_size(r.min, vec2(half - 3.0, r.height() * 0.55)),
                Rect::from_min_size(r.min + vec2(half + 3.0, 0.0), vec2(half - 3.0, r.height() * 0.55)),
                Rect::from_min_max(pos2(r.left(), r.top() + r.height() * 0.55 + 6.0), r.max),
            ];
            let names = ["source", "record", "timeline"];
            for (j, pr) in panes.iter().enumerate() {
                let (rect, show) = match i {
                    0 if f > 0.2 => (if j == 1 { r } else { *pr }, j == 1),
                    1 if f > 0.2 => (*pr, j != 1),
                    _ => (*pr, true),
                };
                if !show {
                    continue;
                }
                p.rect_filled(rect, 4.0, Color32::from_gray(36));
                p.rect_stroke(rect, 4.0, Stroke::new(1.0, if j == 1 { accent() } else { Color32::from_gray(60) }), StrokeKind::Inside);
                p.text(rect.left_top() + vec2(8.0, 6.0), Align2::LEFT_TOP, names[j], FontId::proportional(11.0), DIM);
                if j < 2 && !(i == 2 && f > 0.2) {
                    p.text(rect.center_bottom() + vec2(0.0, -8.0), Align2::CENTER_BOTTOM, "J K L   I O   , .", FontId::monospace(10.0), INK.gamma_multiply(0.6));
                }
            }
            pointer(p, panes[1].center() + vec2(10.0, 0.0), false);
        }
        Anim::Intro => {
            // The idea in one picture: a held key and a drag.
            let ph = phase(t);
            ruler(p);
            let grab = pos2(at(50.0), lane.center().y);
            let d = ease((ph - 0.3) / 0.45) * -10.0;
            let d = if ph >= 0.9 { 0.0 } else { d };
            let down = (0.08..0.9).contains(&ph);
            clip(p, 5.0, 35.0, "A", 1001, false, None);
            clip(p, 35.0, 65.0 + d, "B", 1001, (0.3..0.8).contains(&ph), down.then_some((false, TRIM)));
            clip(p, 65.0, 95.0, "C", 1001, false, None);
            let ptr = pointer_path(ph, rest, grab, pos2(grab.x + d * ux, grab.y));
            if down {
                label_by_pointer(p, ptr, "hold D + drag: trim out");
            }
            pointer(p, ptr, (0.3..0.8).contains(&ph));
        }
        Anim::Done => {
            p.text(area.center(), Align2::CENTER_CENTER, "help > hotkey tour  ·  H toggles the key hints", FontId::proportional(14.0), DIM);
        }
    }
}

/// Rest → grab (by 0.3) → follows the drag → back to rest after 0.85.
fn pointer_path(ph: f64, rest: Pos2, grab: Pos2, dragged: Pos2) -> Pos2 {
    let s = |x: f64| -> f32 { let x = x.clamp(0.0, 1.0); (x * x * (3.0 - 2.0 * x)) as f32 };
    if ph < 0.3 {
        rest.lerp(grab, s((ph - 0.05) / 0.25))
    } else if ph < 0.85 {
        dragged
    } else {
        dragged.lerp(rest, s((ph - 0.85) / 0.15))
    }
}

/// An arrow pointer; filled darker while the button is down.
fn pointer(p: &egui::Painter, at: Pos2, down: bool) {
    let pts = [(0.0, 0.0), (0.0, 16.0), (4.2, 12.4), (7.0, 18.5), (9.6, 17.4), (6.9, 11.4), (12.0, 11.4)];
    let v: Vec<Pos2> = pts.iter().map(|(x, y)| at + vec2(*x, *y)).collect();
    if down {
        p.circle_stroke(at, 9.0, Stroke::new(1.5, accent().gamma_multiply(0.8)));
    }
    p.add(Shape::convex_polygon(v.clone(), if down { Color32::from_gray(210) } else { Color32::WHITE }, Stroke::NONE));
    p.add(Shape::closed_line(v, Stroke::new(1.0, Color32::BLACK)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_lit_key_is_on_the_board() {
        let caps: Vec<&str> = ROWS.iter().flat_map(|r| r.iter().map(|(l, _)| *l)).collect();
        for s in STEPS {
            for k in s.keys {
                assert!(caps.contains(k), "{k} is not drawn");
            }
        }
    }

    #[test]
    fn headless_and_seen_stay_shut() {
        assert!(!Tour::new(true, false).open);
        assert!(!Tour::new(false, true).open);
        let mut t = Tour::new(true, false);
        t.replay();
        assert!(t.open);
    }

    #[test]
    fn steps_end_on_done() {
        assert!(STEPS.last().is_some_and(|s| s.anim == Anim::Done));
        assert!(dwell(STEPS.last().unwrap()).is_infinite());
    }

    fn frame(ctx: &egui::Context, tour: &mut Tour, time: f64) {
        let input = egui::RawInput { time: Some(time), ..Default::default() };
        let mut out = ctx.run_ui(input, |ui| {
            tour.show(ui.ctx());
        });
        // No renderer here: drop the font atlas uploads on purpose.
        out.textures_delta.clear();
    }

    #[test]
    fn an_untouched_step_moves_on_when_its_loops_are_done() {
        for from in 0..STEPS.len() - 1 {
            let ctx = egui::Context::default();
            let mut tour = Tour { open: true, step: from, t0: None, manual: false };
            let end = dwell(&STEPS[from]);
            frame(&ctx, &mut tour, 0.0);
            frame(&ctx, &mut tour, end - 0.01);
            assert_eq!(tour.step, from, "left early");
            frame(&ctx, &mut tour, end + 0.01);
            assert_eq!(tour.step, from + 1, "did not move on");
        }
    }

    #[test]
    fn a_step_chosen_by_hand_stays() {
        let ctx = egui::Context::default();
        let mut tour = Tour { open: true, step: 1, t0: None, manual: true };
        frame(&ctx, &mut tour, 0.0);
        frame(&ctx, &mut tour, 100.0);
        assert_eq!(tour.step, 1);
    }
}
