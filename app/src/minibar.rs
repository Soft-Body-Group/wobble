//! The view of a small time bar inside a panel (the Layer node's timeline,
//! the Time Range bar): the main time bar's hands, at panel size.
//!
//! - **Navigation** is `nav::bar_nav`, the viewer's and graph's chords:
//!   wheel / pinch / ctrl+MMB / ctrl+alt+LMB zoom, MMB / alt+LMB / two
//!   fingers pan.
//! - **R** over the bar frames the in–out (the node's range without one).
//! - **The pie** (`timeline.pie`, held over the bar) zooms it to the comp,
//!   the node or the in–out range — the main bar's three slices.
//! - A pick or R GLIDES there, as on the main bar.
//!
//! And nothing else moves it. The view is kept per bar and never refits
//! itself to what is being edited: a bar that re-framed when a drag let go
//! jumped out from under the hand that had just put it somewhere.

use egui::{Id, Pos2, Rect};

/// The frames a pick can frame, inclusive comp frames.
#[derive(Clone, Copy, Debug)]
pub struct Ranges {
    /// The comp's script range.
    pub comp: (i64, i64),
    /// What this node covers (its range, its layers).
    pub node: (i64, i64),
    /// The timeline's in/out, when one is set.
    pub inout: Option<(i64, i64)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Zoom {
    Comp,
    Node,
    InOut,
}

impl Zoom {
    /// The main bar's pie, slice for slice.
    pub const ALL: [Zoom; 3] = [Zoom::Comp, Zoom::Node, Zoom::InOut];
    pub fn label(self) -> &'static str {
        match self {
            Zoom::Comp => "comp range",
            Zoom::Node => "node range",
            Zoom::InOut => "in–out range",
        }
    }
}

impl Ranges {
    /// The view a pick shows: the range with a little room either side, so
    /// an end sitting on it can still be grabbed.
    pub fn target(&self, z: Zoom) -> (f64, f64) {
        let (a, b) = match z {
            Zoom::Comp => self.comp,
            Zoom::Node => self.node,
            Zoom::InOut => self.inout.unwrap_or(self.node),
        };
        let (a, b) = imfram::widgets::span::fit((a.min(b), a.max(b)), None);
        (a as f64, b as f64)
    }
}

/// How long a pick takes to land.
const GLIDE: f32 = 0.28;

#[derive(Clone, Copy)]
struct State {
    view: (f64, f64),
    zoom: Zoom,
    /// Gliding from, to, since.
    glide: Option<((f64, f64), (f64, f64), web_time::Instant)>,
    /// The pie: centre and its spring (scale, velocity).
    pie: Option<(Pos2, f32, f32)>,
}

/// Rects that own the pie key this frame (the properties pie stands down
/// over them).
#[derive(Clone, Default)]
struct Zones {
    frame: u64,
    rects: Vec<Rect>,
}

fn zones_id() -> Id {
    Id::new("minibar-pie-zones")
}

/// A mini bar under `p` this frame takes the pie key, so the properties
/// pie must not open there too.
pub fn owns_pie(ctx: &egui::Context, p: Pos2) -> bool {
    let frame = ctx.cumulative_frame_nr();
    let z: Option<Zones> = ctx.data(|d| d.get_temp(zones_id()));
    z.is_some_and(|z| z.frame == frame && z.rects.iter().any(|r| r.contains(p)))
}

fn claim(ctx: &egui::Context, rect: Rect) {
    let frame = ctx.cumulative_frame_nr();
    ctx.data_mut(|d| {
        let z: &mut Zones = d.get_temp_mut_or_default(zones_id());
        if z.frame != frame {
            *z = Zones { frame, rects: Vec::new() };
        }
        z.rects.push(rect);
    });
}

/// What the bar shows this frame, and whether a navigation drag holds the
/// pointer (the caller's own drags then stand down: an alt-drag on a layer
/// pans, it does not slide it).
pub struct View {
    pub view: (f64, f64),
    pub nav: bool,
}

/// Run the bar's view over `rect`. `frame` asks for R's framing (a
/// double-click, say). `no_nav_press`: the press this frame belongs to the
/// caller even with a nav chord held (an alt-drag on a trim handle).
pub fn view(
    ui: &egui::Ui,
    id: Id,
    rect: Rect,
    ranges: &Ranges,
    frame: bool,
    no_nav_press: bool,
) -> View {
    let ctx = ui.ctx().clone();
    let mut st: State = ui.data(|d| d.get_temp(id)).unwrap_or(State {
        view: ranges.target(Zoom::Node),
        zoom: Zoom::Node,
        glide: None,
        pie: None,
    });
    let glide_to = |st: &mut State, z: Zoom| {
        st.glide = Some((st.view, ranges.target(z), web_time::Instant::now()));
        st.zoom = z;
    };

    // By hand. A press the caller owns starts no hold; one already held
    // keeps running.
    let held_id = id.with("nav");
    let pressing = ui.input(|i| i.pointer.any_pressed());
    let mut nav = false;
    if !(no_nav_press && pressing) {
        let mut v = st.view;
        nav = crate::nav::bar_nav(ui, held_id, rect, &mut v);
        if v != st.view {
            st.view = v;
            st.glide = None;
        }
    }

    let hover = ctx.pointer_hover_pos().is_some_and(|p| rect.contains(p));
    let keys_free = !ctx.egui_wants_keyboard_input() && !crate::keymap::capturing();
    if frame || (hover && keys_free && ui.input(|i| crate::keymap::pressed(i, crate::keymap::Act::TimelineFrame))) {
        glide_to(&mut st, Zoom::InOut);
    }

    // The pie, held over the bar.
    claim(&ctx, rect);
    let act = crate::keymap::Act::TimelinePie;
    let pointer = ctx.pointer_latest_pos();
    if st.pie.is_none() && hover && keys_free && ctx.input(|i| crate::keymap::pressed(i, act)) {
        if let Some(p) = pointer {
            st.pie = Some((p, 0.0, 0.0));
        }
    }
    if let Some((c, mut s, mut v)) = st.pie {
        let dt = ctx.input(|i| i.stable_dt).min(0.1);
        v += ((1.0 - s) * 260.0 - v * 10.0) * dt;
        s += v * dt;
        st.pie = Some((c, s, v));
        ctx.request_repaint();
        let labels: Vec<String> = Zoom::ALL
            .iter()
            .map(|z| format!("{}{}", if *z == st.zoom { "• " } else { "" }, z.label()))
            .collect();
        let hot = pointer.and_then(|p| crate::regions::pie_pick(c, p, labels.len()));
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, id.with("pie")));
        crate::regions::draw_label_pie(&painter, c, &labels, hot, 62.0 * s.max(0.0));
        // A shield: a click on a slice is the pie's, never the widget under it.
        let clicked = egui::Area::new(id.with("pie-shield"))
            .order(egui::Order::Foreground)
            .fixed_pos(c - egui::vec2(120.0, 90.0))
            .show(&ctx, |ui| ui.allocate_response(egui::vec2(240.0, 180.0), egui::Sense::click()).clicked())
            .inner;
        let released = ctx.input(|i| crate::keymap::released(i, act)) || clicked;
        let held = ctx.input(|i| crate::keymap::down(i, act));
        if released || !held {
            if let (true, Some(i)) = (released, hot) {
                glide_to(&mut st, Zoom::ALL[i]);
            }
            st.pie = None;
        }
    }

    // The glide: quick to leave, gentle to land.
    let shown = match st.glide {
        Some((from, to, t0)) => {
            let t = (t0.elapsed().as_secs_f32() / GLIDE).min(1.0);
            if t >= 1.0 {
                st.glide = None;
                st.view = to;
                to
            } else {
                ctx.request_repaint();
                let e = (1.0 - (1.0 - t).powi(3)) as f64;
                st.view = (from.0 + (to.0 - from.0) * e, from.1 + (to.1 - from.1) * e);
                st.view
            }
        }
        None => st.view,
    };
    ui.data_mut(|d| d.insert_temp(id, st));
    View { view: shown, nav }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranges() -> Ranges {
        Ranges { comp: (1001, 1100), node: (990, 1040), inout: Some((1010, 1020)) }
    }

    #[test]
    fn each_slice_frames_its_range_with_room() {
        let r = ranges();
        for (z, (a, b)) in [(Zoom::Comp, r.comp), (Zoom::Node, r.node), (Zoom::InOut, r.inout.unwrap())] {
            let (va, vb) = r.target(z);
            assert!(va < a as f64 && vb > b as f64, "{z:?}: {va}..{vb} round {a}..{b}");
        }
        // No in/out: the node's range.
        let r = Ranges { inout: None, ..r };
        assert_eq!(r.target(Zoom::InOut), r.target(Zoom::Node));
    }

    /// The point of the module: an edit that moves the node's range does
    /// not move the view. Only a hand, R or a pick does.
    #[test]
    fn the_view_never_refits_to_an_edit() {
        let ctx = egui::Context::default();
        let rect = Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(400.0, 20.0));
        let id = Id::new("bar");
        let frame = |r: Ranges, ask: bool| {
            let mut v = (0.0, 0.0);
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                v = view(ui, id, rect, &r, ask, false).view;
            });
            v
        };
        let first = frame(ranges(), false);
        for i in 1..4 {
            // The node's range grows every frame, as a dragged end would.
            let mut r = ranges();
            r.node.1 += 200 * i;
            assert_eq!(frame(r, false), first, "frame {i}: the view moved with the edit");
        }
        // R's framing asked for: it glides to the in–out, and lands there.
        frame(ranges(), true);
        std::thread::sleep(std::time::Duration::from_millis((GLIDE * 1000.0) as u64 + 30));
        assert_eq!(frame(ranges(), false), ranges().target(Zoom::InOut));
    }
}
