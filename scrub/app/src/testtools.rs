//! `--test-tools`: the editing batch through real input — audio tracks
//! (targets, mute / solo), the razor (click, shift-click, cmd+K), the
//! playhead past the end, four-mark fit to fill, rate stretch, slip and
//! slide by dragging, marquee select, alt-drag duplicate, dissolves between
//! two selected clips (and the refusal when they don't meet), the cut menu,
//! and drops from the source viewer and the transitions bin.

use crate::timelineui::{Cmd, Payload, Tool, TrackOp};
use crate::{editor, App};
use egui::{Event as E, Key as K, Modifiers as M, PointerButton as B, Pos2};
use imfram::shell::UiWindow;

impl App {
    pub(crate) fn test_tools(&mut self) {
        let (Some(gpu), Some(win)) = (&self.gpu, &mut self.win) else { return };
        let ed = &mut self.editor;
        ed.build_cuts(4, false); // V1 at 0 and 24, V2 at 12 and 36, 12 frames each
        ed.focus_record();
        ed.set_snap(false);
        for _ in 0..3 {
            let _ = win.offscreen(gpu, ed);
        }
        let audio = ed.sequence_has_audio();
        let mut fails: Vec<String> = Vec::new();
        let frame = |win: &mut UiWindow, ed: &mut editor::Editor, evs: Vec<E>| {
            win.inject(evs);
            let _ = win.offscreen(gpu, ed);
        };
        let press = |win: &mut UiWindow, ed: &mut editor::Editor, key: K, mods: M| {
            for pressed in [true, false] {
                win.inject([E::Key { key, physical_key: None, pressed, repeat: false, modifiers: mods }]);
                let _ = win.offscreen(gpu, ed);
            }
        };
        let click = |win: &mut UiWindow, ed: &mut editor::Editor, at: Pos2, mods: M| {
            frame(win, ed, vec![E::PointerMoved(at)]);
            frame(win, ed, vec![E::PointerButton { pos: at, button: B::Primary, pressed: true, modifiers: mods }]);
            frame(win, ed, vec![E::PointerButton { pos: at, button: B::Primary, pressed: false, modifiers: mods }]);
            frame(win, ed, vec![]);
        };
        let drag = |win: &mut UiWindow, ed: &mut editor::Editor, from: Pos2, to: Pos2, mods: M| {
            frame(win, ed, vec![E::PointerMoved(from)]);
            frame(win, ed, vec![E::PointerButton { pos: from, button: B::Primary, pressed: true, modifiers: mods }]);
            for k in 1..=8 {
                frame(win, ed, vec![E::PointerMoved(from + (to - from) * (k as f32 / 8.0))]);
            }
            frame(win, ed, vec![E::PointerButton { pos: to, button: B::Primary, pressed: false, modifiers: mods }]);
            frame(win, ed, vec![]);
        };
        let clips = |ed: &editor::Editor| ed.sequence_summary().clips;
        let has = |ed: &editor::Editor, c: (usize, i64, i64)| clips(ed).iter().any(|x| (x.0, x.1, x.2) == c);
        let pt = |ed: &editor::Editor, t: usize, f: f64| ed.timeline_point(t, f).expect("lane");
        let speed = |ed: &editor::Editor, t: usize, start: i64| ed.sequence_summary().speeds.iter().find(|s| s.0 == t && s.1 == start).map(|s| s.2);
        let mut check = |what: String, ok: bool| {
            if !ok {
                fails.push(what);
            }
        };
        use model::{A1, A2, A3, M as MT, V1, V2, V3};

        // 1. Audio tracks: V2's clips carry their sound on A2; 2 targets V2
        // and the sound follows to A2; mute and solo pick what is heard.
        if audio {
            check(format!("sound on A1 and A2: {:?}", clips(ed)), has(ed, (A1, 0, 12)) && has(ed, (A2, 12, 12)));
        }
        press(win, ed, K::Num2, M::NONE);
        check(format!("2 targets V2 + A2: {:?}", ed.targets()), ed.targets().video == Some(V2) && ed.targets().audio == Some(A2));
        press(win, ed, K::Num7, M::NONE);
        check(format!("7 targets M: {:?}", ed.targets()), ed.targets().audio == Some(MT));
        press(win, ed, K::Num1, M::NONE);
        press(win, ed, K::Num4, M::NONE);
        check(format!("back to V1 + A1: {:?}", ed.targets()), ed.targets() == model::Targets::default());
        ed.track_header(A2, TrackOp::Mute);
        check(format!("A2 muted: {:?}", ed.audible_tracks()), ed.audible_tracks() == vec![A1, A3, MT]);
        ed.track_header(MT, TrackOp::Solo);
        check(format!("M solo: {:?}", ed.audible_tracks()), ed.audible_tracks() == vec![MT]);
        ed.track_header(MT, TrackOp::Solo);
        ed.track_header(A2, TrackOp::Mute);
        ed.track_header(A1, TrackOp::Gain(-6.0, true));
        check(format!("all heard again: {:?}", ed.audible_tracks()), ed.audible_tracks().len() == 4);
        ed.track_header(A1, TrackOp::Gain(0.0, true));

        // 2. Razor: X, click the V1 clip at 6; shift-click at 30 cuts every
        // track there; V goes back to select.
        click(win, ed, pt(ed, V1, 60.0), M::NONE); // the pointer over the timeline
        press(win, ed, K::X, M::NONE);
        check(format!("X: razor: {:?}", ed.tool()), ed.tool() == Tool::Razor);
        click(win, ed, pt(ed, V1, 6.0), M::NONE);
        check(format!("razor at 6: {:?}", clips(ed)), has(ed, (V1, 0, 6)) && has(ed, (V1, 6, 6)) && (!audio || has(ed, (A1, 6, 6))));
        click(win, ed, pt(ed, V2, 30.0), M::SHIFT);
        check(format!("shift razor at 30: {:?}", clips(ed)), has(ed, (V1, 24, 6)) && has(ed, (V1, 30, 6)));
        press(win, ed, K::V, M::NONE);
        check(format!("V: {:?}", ed.tool()), ed.tool() == Tool::Select);
        ed.undo_once();
        ed.undo_once();
        check(format!("razors undone: {:?}", clips(ed)), has(ed, (V1, 0, 12)) && has(ed, (V1, 24, 12)));

        // 3. cmd+K at the playhead on the targeted tracks.
        ed.seek_record(18);
        press(win, ed, K::K, M::COMMAND);
        check(format!("cmd K, V1 empty there: {}", ed.last_notice()), has(ed, (V2, 12, 12)));
        press(win, ed, K::Num2, M::NONE);
        press(win, ed, K::K, M::COMMAND);
        check(format!("cmd K on V2: {:?}", clips(ed)), has(ed, (V2, 12, 6)) && has(ed, (V2, 18, 6)));
        ed.undo_once();
        press(win, ed, K::Num1, M::NONE);

        // 4. Past the end: click an empty lane at 60 (the last clip ends at
        // 48), step on, overwrite there: a gap before it.
        click(win, ed, pt(ed, V1, 60.5), M::NONE);
        let ph = ed.sequence_summary().playhead;
        check(format!("click past the end: {ph}"), ph == 60);
        press(win, ed, K::ArrowRight, M::NONE);
        check(format!("step past the end: {}", ed.sequence_summary().playhead), ed.sequence_summary().playhead == 61);
        press(win, ed, K::Num2, M::SHIFT); // focus source (its marks: 12 frames)
        press(win, ed, K::Period, M::NONE);
        check(format!("edit past the end: {:?}", clips(ed)), has(ed, (V1, 61, 12)) && ed.sequence_summary().duration == 73);
        ed.undo_once();
        ed.focus_record();

        // 5. Four marks: the 12 source frames fitted to 6 record frames.
        ed.set_record_marks(model::Marks { inp: Some(100), out: Some(105) });
        press(win, ed, K::Period, M::NONE);
        check(format!("fit to fill: {:?} {}", speed(ed, V1, 100), ed.last_notice()), speed(ed, V1, 100) == Some(2.0) && ed.last_notice().contains("fit to fill"));
        ed.undo_once();

        // 6. Rate stretch (the tool, from the menu now): drag the V1 clip's
        // out from 12 to 18.
        ed.set_tool_for_test(Tool::Rate);
        drag(win, ed, pt(ed, V1, 11.9), pt(ed, V1, 17.9), M::NONE);
        let sp = speed(ed, V1, 0).unwrap_or(1.0);
        check(format!("rate stretch: {:?} {sp}", clips(ed)), has(ed, (V1, 0, 18)) && (sp - 12.0 / 18.0).abs() < 1e-6);
        ed.undo_once();

        // 7. Slip: Y, drag the V1 clip at 24 three frames left: three later
        // source frames.
        let src = |ed: &editor::Editor| clips(ed).iter().find(|c| c.0 == V1 && c.1 == 24).map(|c| c.3);
        let before = src(ed);
        press(win, ed, K::Y, M::NONE);
        drag(win, ed, pt(ed, V1, 30.0), pt(ed, V1, 27.0), M::NONE);
        check(format!("slip: {before:?} -> {:?}", src(ed)), before.zip(src(ed)).is_some_and(|(a, b)| b == a + 3));
        ed.undo_once();

        // 8. Slide: U, drag the V1 clip at 24 four frames earlier (the gap
        // before it gives).
        press(win, ed, K::U, M::NONE);
        drag(win, ed, pt(ed, V1, 30.0), pt(ed, V1, 26.0), M::NONE);
        check(format!("slide: {:?}", clips(ed)), has(ed, (V1, 20, 12)));
        ed.undo_once();

        // 9. Marquee: V, drag from empty V3 at 2 down to V1 at 30.
        press(win, ed, K::V, M::NONE);
        drag(win, ed, pt(ed, V3, 2.0), pt(ed, V1, 30.0), M::NONE);
        check(format!("marquee: {} selected", ed.selected_clips()), ed.selected_clips() == 3);

        // 10. alt-drag: a copy of the V1 clip at 0 on V3 at 60.
        click(win, ed, pt(ed, V1, 6.0), M::NONE);
        drag(win, ed, pt(ed, V1, 6.0), pt(ed, V3, 66.0), M::ALT);
        check(format!("alt duplicate: {:?}", clips(ed)), has(ed, (V1, 0, 12)) && has(ed, (V3, 60, 12)) && (!audio || has(ed, (A3, 60, 12))));
        ed.undo_once();

        // 11. Two selected clips that don't meet: refused in words. Moved
        // back to back: cmd+D dissolves their cut.
        click(win, ed, pt(ed, V1, 6.0), M::NONE);
        click(win, ed, pt(ed, V1, 30.0), M::SHIFT);
        press(win, ed, K::D, M::COMMAND);
        check(format!("not adjacent: {}", ed.last_notice()), ed.last_notice().contains("don't meet") && ed.sequence_summary().dissolves.is_empty());
        click(win, ed, pt(ed, V1, 30.0), M::NONE);
        drag(win, ed, pt(ed, V1, 30.0), pt(ed, V1, 18.0), M::NONE); // the clip at 24 to 12
        click(win, ed, pt(ed, V1, 6.0), M::NONE);
        click(win, ed, pt(ed, V1, 18.0), M::SHIFT);
        press(win, ed, K::D, M::COMMAND);
        let ds = ed.sequence_summary().dissolves;
        check(format!("dissolve between two clips: {ds:?} {}", ed.last_notice()), ds.iter().any(|d| d.0 == V1 && d.1 == 12));

        // 12. The cut menu: right-click opens it; its commands remove and add.
        let at = pt(ed, V1, 12.0);
        frame(win, ed, vec![E::PointerMoved(at)]);
        frame(win, ed, vec![E::PointerButton { pos: at, button: B::Secondary, pressed: true, modifiers: M::NONE }]);
        frame(win, ed, vec![E::PointerButton { pos: at, button: B::Secondary, pressed: false, modifiers: M::NONE }]);
        frame(win, ed, vec![]);
        check("right-click opens the cut menu".into(), egui::Popup::is_any_open(win.ctx()));
        press(win, ed, K::Escape, M::NONE);
        frame(win, ed, vec![]);
        check("Esc closes the cut menu".into(), !egui::Popup::is_any_open(win.ctx()));
        ed.timeline_cmd(Cmd::RemoveDissolve { track: V1, at: 12 });
        check("menu: remove dissolve".into(), ed.sequence_summary().dissolves.is_empty());
        ed.timeline_cmd(Cmd::AddDissolve { track: V1, at: 12 });
        check("menu: add dissolve".into(), !ed.sequence_summary().dissolves.is_empty());
        ed.timeline_cmd(Cmd::RemoveDissolve { track: V1, at: 12 });

        // 12b. At a butt cut, a press just right of the cut is the incoming
        // clip's IN edge: a plain drag moves that cut (a gap opens), the
        // outgoing clip stays.
        drag(win, ed, pt(ed, V1, 12.3), pt(ed, V1, 15.3), M::NONE);
        check(format!("left edge at a butt cut: {:?}", clips(ed)), has(ed, (V1, 0, 12)) && has(ed, (V1, 15, 9)));
        ed.undo_once();
        drag(win, ed, pt(ed, V1, 11.7), pt(ed, V1, 9.7), M::NONE);
        check(format!("right edge at a butt cut: {:?}", clips(ed)), has(ed, (V1, 0, 10)) && has(ed, (V1, 12, 12)));
        ed.undo_once();

        // 13. Drops: the cross dissolve onto the cut, the source onto V2 at
        // 100 (overwrite) — through the timeline's own drop handling.
        let drop = |win: &mut UiWindow, ed: &mut editor::Editor, p: Payload, at: Pos2, mods: M| {
            frame(win, ed, vec![E::PointerMoved(at)]);
            egui::DragAndDrop::set_payload(win.ctx(), p);
            frame(win, ed, vec![E::PointerButton { pos: at, button: B::Primary, pressed: false, modifiers: mods }]);
            frame(win, ed, vec![]);
        };
        drop(win, ed, Payload::Dissolve, pt(ed, V1, 13.0), M::NONE);
        check(format!("dissolve dropped on the cut: {:?}", ed.sequence_summary().dissolves), ed.sequence_summary().dissolves.iter().any(|d| d.0 == V1 && d.1 == 12));
        let ph = ed.sequence_summary().playhead;
        drop(win, ed, Payload::Source { len: 12 }, pt(ed, V2, 100.0), M::NONE);
        check(format!("source dropped on V2: {:?}", clips(ed)), has(ed, (V2, 100, 12)) && (!audio || has(ed, (A2, 100, 12))));
        check("a drop leaves the playhead".into(), ed.sequence_summary().playhead == ph);
        drop(win, ed, Payload::Source { len: 12 }, pt(ed, V1, 30.0), M::COMMAND);
        check(format!("cmd drop inserts: {:?}", clips(ed)), has(ed, (V1, 30, 12)) && has(ed, (V2, 112, 12)));

        // 14. The UI batch: shift+H hides the viewer readouts; the wheel
        // over the record viewer zooms it about the pointer, R there fits
        // it, shift+R gives the source the same view and size; R over the
        // timeline is still the rate tool; the ruler's chip and the title
        // bar's button clear the marks.
        use crate::panes::Pane;
        let over = |win: &mut UiWindow, ed: &mut editor::Editor, at: Pos2, key: K, mods: M| {
            frame(win, ed, vec![E::PointerMoved(at)]);
            frame(win, ed, vec![E::Key { key, physical_key: None, pressed: true, repeat: false, modifiers: mods }]);
            frame(win, ed, vec![E::Key { key, physical_key: None, pressed: false, repeat: false, modifiers: mods }]);
        };
        let rec = ed.pane_rect(Pane::Record).expect("record viewer");
        let src_r = ed.pane_rect(Pane::Source).expect("source viewer");
        let mid = rec.center();
        over(win, ed, mid, K::H, M::SHIFT);
        check("shift H hides the readouts".into(), !ed.readouts());
        over(win, ed, mid, K::H, M::SHIFT);
        check("shift H brings them back".into(), ed.readouts());
        frame(win, ed, vec![E::PointerMoved(mid + egui::vec2(60.0, 20.0))]);
        frame(win, ed, vec![E::MouseWheel { unit: egui::MouseWheelUnit::Line, delta: egui::vec2(0.0, 3.0), modifiers: M::NONE, phase: egui::TouchPhase::Move }]);
        let (z, pan) = ed.viewer_zoom(1);
        check(format!("wheel zooms the record viewer about the pointer: {z} {pan:?}"), z > 1.3 && pan.length() > 1.0);
        over(win, ed, src_r.center(), K::R, M::SHIFT);
        check(format!("shift R over the source: the record viewer matches it: {:?}", ed.viewer_zoom(1)), ed.viewer_zoom(1).0 == ed.viewer_zoom(0).0);
        over(win, ed, mid, K::R, M::SHIFT);
        check(format!("shift R over the record: the source matches: {:?} {}", ed.viewer_zoom(0), ed.monitor_split()), ed.viewer_zoom(0) == ed.viewer_zoom(1) && ed.monitor_split() == 0.5);
        over(win, ed, mid, K::R, M::NONE);
        check(format!("R over the record viewer fits it: {:?}", ed.viewer_zoom(1)), ed.viewer_zoom(1).0 == 1.0 && ed.viewer_zoom(1).1 == egui::Vec2::ZERO);
        check("R over a viewer is not the rate tool".into(), ed.tool() == Tool::Select);
        ed.timeline_zoom(3.0);
        frame(win, ed, vec![]);
        let zoomed = !ed.timeline_fitted();
        let tl_mid = ed.pane_rect(Pane::Timeline).expect("timeline").center();
        over(win, ed, tl_mid, K::R, M::NONE);
        check(format!("R over the timeline fits it (zoomed first: {zoomed}), tool {:?}", ed.tool()), zoomed && ed.timeline_fitted() && ed.tool() == Tool::Select);
        // A plain drag across the record picture scrubs it, RV style: 50 pt
        // right is 10 frames on from where it was.
        let f0 = ed.record_frame();
        let a = mid - egui::vec2(60.0, 0.0);
        drag(win, ed, a, a + egui::vec2(10.0 * crate::editor::SCRUB_PX, 0.0), M::NONE);
        check(format!("drag-scrub on the record viewer: {f0} -> {}", ed.record_frame()), ed.record_frame() == f0 + 10);
        drag(win, ed, a + egui::vec2(40.0, 0.0), a + egui::vec2(40.0 - 4.0 * crate::editor::SCRUB_PX, 0.0), M::NONE);
        check(format!("drag-scrub back: {}", ed.record_frame()), ed.record_frame() == f0 + 6);
        // The same on the source picture.
        let s0 = ed.source_frame().unwrap_or(0);
        let sc = src_r.center();
        drag(win, ed, sc, sc + egui::vec2(7.0 * crate::editor::SCRUB_PX, 0.0), M::NONE);
        check(format!("drag-scrub on the source viewer: {s0} -> {:?}", ed.source_frame()), ed.source_frame() == Some(s0 + 7));
        // Shift-drag on the source picture drops the clip on the timeline
        // (a plain drag scrubs).
        let v3 = clips(ed).iter().filter(|c| c.0 == V3).count();
        drag(win, ed, sc, pt(ed, V3, 60.0), M::SHIFT);
        check(format!("shift-drag on the source picture lays the clip on V3: {} -> {}", v3, clips(ed).iter().filter(|c| c.0 == V3).count()), clips(ed).iter().filter(|c| c.0 == V3).count() == v3 + 1);
        press(win, ed, K::Z, M::COMMAND);
        // I / O go to the side under the pointer: over the source viewer
        // with the record focused, I marks the source.
        ed.focus_record();
        over(win, ed, sc, K::I, M::NONE);
        check(format!("I over the source viewer marks the source: {:?}", ed.source_marks()), ed.source_marks().is_some_and(|m| m.inp == Some(s0 + 7)));
        over(win, ed, sc, K::I, M::ALT);
        over(win, ed, pt(ed, V1, 3.0), K::O, M::NONE);
        check(format!("O over the timeline marks the record: {:?}", ed.record_marks()), ed.record_marks().out.is_some());
        ed.set_record_marks(model::Marks::default());
        // Focus elsewhere, keys still go to the panel under the pointer,
        // and that panel wears the key-target edge.
        let bin_c = ed.pane_rect(Pane::Bin).expect("bin").center();
        ed.focus_bin();
        over(win, ed, sc, K::O, M::NONE);
        check(format!("focus on the bin, O over the source viewer marks the source: {:?}, edge on {:?}", ed.source_marks(), ed.key_panel()), ed.source_marks().is_some_and(|m| m.out.is_some()) && ed.key_panel() == Some(Pane::Source));
        over(win, ed, sc, K::X, M::ALT);
        ed.focus_source();
        over(win, ed, mid, K::I, M::NONE);
        check(format!("focus on the source, I over the record viewer marks the record: {:?}, source {:?}, edge {:?}", ed.record_marks(), ed.source_marks(), ed.key_panel()), ed.record_marks().inp.is_some() && ed.source_marks().is_some_and(|m| m.inp.is_none()) && ed.key_panel() == Some(Pane::Record));
        ed.set_record_marks(model::Marks::default());
        let tl_c = ed.pane_rect(Pane::Timeline).expect("timeline").center();
        over(win, ed, tl_c, K::O, M::NONE);
        check(format!("focus on the source, O over the timeline marks the record: {:?}, edge {:?}", ed.record_marks(), ed.key_panel()), ed.record_marks().out.is_some() && ed.key_panel() == Some(Pane::Timeline));
        ed.set_record_marks(model::Marks::default());
        // Over the bin (no marks of its own) the focused side gets them.
        over(win, ed, bin_c, K::I, M::NONE);
        check(format!("over the bin, I goes to the focused source: {:?}", ed.source_marks()), ed.source_marks().is_some_and(|m| m.inp.is_some()));
        over(win, ed, sc, K::X, M::ALT);
        // M appends the source (marks, else the whole clip) after the last
        // clip; one undo step; the playhead lands at the new end.
        let end0 = ed.sequence().duration();
        let n0 = clips(ed).len();
        let s_now = ed.source_frame().unwrap_or(0);
        ed.set_source_marks(model::Marks { inp: Some(s_now), out: Some(s_now + 9) });
        over(win, ed, tl_c, K::M, M::NONE);
        let end1 = ed.sequence().duration();
        check(format!("M appends 10 frames at {end0}: end {end1}, playhead {}, clips {n0} -> {}", ed.record_frame(), clips(ed).len()), end1 == end0 + 10 && ed.record_frame() == end1 && clips(ed).iter().any(|c| c.0 == V1 && c.1 == end0 && c.2 == 10));
        press(win, ed, K::Z, M::COMMAND);
        check(format!("one undo takes the append back: end {}", ed.sequence().duration()), ed.sequence().duration() == end0 && clips(ed).len() == n0);
        ed.set_source_marks(model::Marks::default());
        press(win, ed, K::V, M::NONE);
        ed.set_record_marks(model::Marks { inp: Some(4), out: Some(20) });
        frame(win, ed, vec![]);
        let chip = Pos2::new(ed.timeline_point(V1, 21.0).expect("lane").x, ed.pane_rect(Pane::Timeline).expect("timeline").min.y + 20.0 + 10.0); // under its 20pt bar
        // Let the tooltip from the last hover close first (it can open over
        // the ruler) — a hand moving there takes longer than that.
        for _ in 0..12 {
            frame(win, ed, vec![E::PointerMoved(chip)]);
        }
        click(win, ed, chip, M::NONE);
        check(format!("the ruler chip clears the marks: {:?}", ed.record_marks()), ed.record_marks() == model::Marks::default());
        ed.set_record_marks(model::Marks { inp: Some(4), out: Some(20) });
        frame(win, ed, vec![]);
        click(win, ed, Pos2::new(rec.max.x - 40.0, rec.min.y + 10.0), M::NONE);
        check(format!("the title bar button clears the marks: {:?}", ed.record_marks()), ed.record_marks() == model::Marks::default());

        // 15. Hold keys: with the key held, a press anywhere on a clip
        // drags that op; D tapped alone still selects.
        press(win, ed, K::V, M::NONE);
        ed.set_snap(false);
        let hold = |win: &mut UiWindow, ed: &mut editor::Editor, key: K, mods: M, from: Pos2, to: Pos2| {
            frame(win, ed, vec![E::PointerMoved(from)]);
            frame(win, ed, vec![E::Key { key, physical_key: None, pressed: true, repeat: false, modifiers: mods }]);
            frame(win, ed, vec![E::PointerButton { pos: from, button: B::Primary, pressed: true, modifiers: mods }]);
            for k in 1..=8 {
                frame(win, ed, vec![E::PointerMoved(from + (to - from) * (k as f32 / 8.0))]);
            }
            frame(win, ed, vec![E::PointerButton { pos: to, button: B::Primary, pressed: false, modifiers: mods }]);
            frame(win, ed, vec![E::Key { key, physical_key: None, pressed: false, repeat: false, modifiers: mods }]);
            frame(win, ed, vec![]);
        };
        let src_of = |ed: &editor::Editor, t: usize, st: i64| clips(ed).iter().find(|c| c.0 == t && c.1 == st).map(|c| c.3);
        // The V1 clip at 30 (12 long, a gap before it; V2's next clip at
        // 48 is what a ripple moves), pressed mid-clip.
        let s30 = src_of(ed, V1, 30);
        let undo = |win: &mut UiWindow, ed: &mut editor::Editor| {
            ed.undo_once();
            frame(win, ed, vec![]);
        };
        check(format!("the clips step 15 expects: {:?}", clips(ed)), has(ed, (V1, 30, 12)) && has(ed, (V2, 48, 12)));
        hold(win, ed, K::A, M::NONE, pt(ed, V1, 36.0), pt(ed, V1, 39.0));
        check(format!("A: trim in +3: {:?}", clips(ed)), has(ed, (V1, 33, 9)) && src_of(ed, V1, 33) == s30.map(|s| s + 3) && has(ed, (V2, 48, 12)));
        undo(win, ed);
        hold(win, ed, K::D, M::NONE, pt(ed, V1, 36.0), pt(ed, V1, 33.0));
        check(format!("D: trim out -3: {:?}", clips(ed)), has(ed, (V1, 30, 9)) && has(ed, (V2, 48, 12)));
        undo(win, ed);
        hold(win, ed, K::Z, M::NONE, pt(ed, V1, 36.0), pt(ed, V1, 40.0));
        check(format!("Z: ripple in +4 (start stays, the rest follows): {:?}", clips(ed)), has(ed, (V1, 30, 8)) && has(ed, (V2, 44, 12)));
        undo(win, ed);
        hold(win, ed, K::C, M::NONE, pt(ed, V1, 36.0), pt(ed, V1, 38.0));
        check(format!("C: ripple out +2: {:?}", clips(ed)), has(ed, (V1, 30, 14)) && has(ed, (V2, 50, 12)));
        undo(win, ed);
        hold(win, ed, K::S, M::NONE, pt(ed, V1, 36.0), pt(ed, V1, 33.0));
        check(format!("S: slip: {:?} -> {:?}", s30, src_of(ed, V1, 30)), s30.zip(src_of(ed, V1, 30)).is_some_and(|(a, b)| b == a + 3) && has(ed, (V1, 30, 12)));
        undo(win, ed);
        hold(win, ed, K::A, M::SHIFT, pt(ed, V1, 36.0), pt(ed, V1, 34.0));
        check(format!("shift A: stretch from the in: {:?} {:?}", clips(ed), speed(ed, V1, 28)), has(ed, (V1, 28, 14)) && speed(ed, V1, 28).is_some_and(|s| s < 1.0));
        undo(win, ed);
        hold(win, ed, K::D, M::SHIFT, pt(ed, V1, 36.0), pt(ed, V1, 40.0));
        check(format!("shift D: stretch from the out: {:?}", clips(ed)), has(ed, (V1, 30, 16)) && speed(ed, V1, 30).is_some_and(|s| s < 1.0));
        undo(win, ed);
        ed.seek_record(36);
        let px120 = 120.0 * ed.timeline_fpp();
        hold(win, ed, K::S, M::SHIFT, pt(ed, V1, 36.0), pt(ed, V1, 36.0 + px120));
        let fast = clips(ed).iter().find(|c| c.0 == V1 && c.1 >= 30 && c.1 < 42).copied();
        check(format!("shift S: twice the speed about the playhead: {fast:?}"), fast.is_some_and(|c| c.2 == 6 && c.1 == 33));
        undo(win, ed);
        // Slipping shows the clip's handles (live) and the first / last
        // frame two-up: by key, and at the limit one side has none.
        ed.select_clip_for_test(V1, 30);
        press(win, ed, K::Period, M::ALT);
        let (band, two_up) = ed.slip_state();
        let src_now = src_of(ed, V1, 30).unwrap_or(0);
        check(format!("slip key: band {band:?}, two-up {two_up}"), two_up && band.is_some_and(|b| b.0 == V1 && b.2 == 30 && b.3 == 42 && b.2 - b.1 == src_now));
        for _ in 0..30 {
            press(win, ed, K::Comma, M::ALT | M::SHIFT);
        }
        let (band, _) = ed.slip_state();
        check(format!("slipped to the start: no handle before: {band:?}"), band.is_some_and(|b| b.1 == b.2));
        // Held slip keys are one undo step (a burst).
        ed.undo_once();
        frame(win, ed, vec![]);
        check(format!("one undo takes the slips back: {:?}", src_of(ed, V1, 30)), src_of(ed, V1, 30) == Some(src_now - 1));
        // A tapped alone: nothing; D tapped alone: selects the clip at the playhead.
        ed.timeline_cmd_select_none();
        ed.seek_record(36);
        press(win, ed, K::D, M::NONE);
        check(format!("D tapped: selects: {}", ed.selected_clips()), ed.selected_clips() == 1);
        let before = clips(ed);
        press(win, ed, K::A, M::NONE);
        check("A tapped: nothing".into(), clips(ed) == before);
        press(win, ed, K::T, M::NONE);
        check("T toggles snapping".into(), ed.snapping());
        press(win, ed, K::T, M::NONE);

        // 16. Titles: cmd+shift+T at the playhead makes a text clip above
        // the target and opens the text window; typing changes the words;
        // an alt-drag copy shares the style; retyping the copy leaves the
        // style alone; moving the style (cmd+arrows, a drag in the viewer)
        // moves both; the bin's T text dropped on V3 makes one there.
        ed.seek_record(150);
        press(win, ed, K::Num1, M::NONE); // target off then on: V1 is the target
        press(win, ed, K::Num1, M::NONE);
        frame(win, ed, vec![E::PointerMoved(pt(ed, V2, 180.0))]);
        press(win, ed, K::T, M::COMMAND | M::SHIFT);
        let made = ed.text_window_on();
        check(format!("cmd shift T: a text clip and its window: {made:?}"), made.is_some_and(|(t, s)| t == V2 && s == 150));
        // Type into the focused text: select all, then the words.
        for _ in 0..2 {
            frame(win, ed, vec![]);
        }
        frame(win, ed, vec![E::Key { key: K::A, physical_key: None, pressed: true, repeat: false, modifiers: M::COMMAND }]);
        frame(win, ed, vec![E::Text("Hello there".into())]);
        frame(win, ed, vec![]);
        check(format!("typed: {:?}", ed.clip_text(V2, 150)), ed.clip_text(V2, 150).is_some_and(|t| t.0 == "Hello there" && t.1 == "title"));
        press(win, ed, K::Escape, M::NONE);
        check("Esc closes the text window".into(), ed.text_window_on().is_none());
        // Enter opens it again on the selected text clip.
        click(win, ed, pt(ed, V2, 152.0), M::NONE);
        press(win, ed, K::Enter, M::NONE);
        check("Enter opens the text window".into(), ed.text_window_on() == Some((V2, 150)));
        press(win, ed, K::Escape, M::NONE);
        // The text is drawn over the picture; a drag on it moves its style.
        ed.seek_record(152);
        for _ in 0..3 {
            frame(win, ed, vec![]);
        }
        let boxes = ed.text_boxes();
        check(format!("drawn in the viewer: {boxes:?}"), boxes.iter().any(|b| b.0 == V2 && b.1 == 150));
        let p0 = ed.style_pos("title");
        if let Some(b) = boxes.first() {
            drag(win, ed, b.2.center(), b.2.center() + egui::vec2(0.0, 40.0), M::NONE);
        }
        let p1 = ed.style_pos("title");
        check(format!("a drag in the viewer moves the style: {p0:?} -> {p1:?}"), p0.zip(p1).is_some_and(|(a, b)| b.1 > a.1 + 0.01));
        // An alt-drag copy at 230, retyped: same style, still where it was.
        drag(win, ed, pt(ed, V2, 152.0), pt(ed, V2, 232.0), M::ALT);
        check(format!("copy: {:?}", ed.clip_text(V2, 230)), ed.clip_text(V2, 230).is_some_and(|t| t.1 == "title"));
        ed.timeline_cmd_select_none();
        click(win, ed, pt(ed, V2, 232.0), M::NONE);
        press(win, ed, K::Enter, M::NONE);
        frame(win, ed, vec![E::Key { key: K::A, physical_key: None, pressed: true, repeat: false, modifiers: M::COMMAND }]);
        frame(win, ed, vec![E::Text("Second line".into())]);
        frame(win, ed, vec![]);
        press(win, ed, K::Escape, M::NONE);
        check(format!("retyped copy: {:?} {:?}", ed.clip_text(V2, 230), ed.clip_text(V2, 150)), ed.clip_text(V2, 230).is_some_and(|t| t.0 == "Second line") && ed.clip_text(V2, 150).is_some_and(|t| t.0 == "Hello there"));
        check("retyping leaves the style where it was".into(), ed.style_pos("title") == p1);
        // cmd+arrow on the selected text clip moves the shared style.
        press(win, ed, K::ArrowRight, M::COMMAND);
        let p2 = ed.style_pos("title");
        check(format!("cmd+right moves the style: {p1:?} -> {p2:?}"), p1.zip(p2).is_some_and(|(a, b)| (b.0 - a.0 - 0.01).abs() < 1e-4));
        // The bin's text generator dropped on V3 at 100.
        let at = pt(ed, V3, 100.0);
        frame(win, ed, vec![E::PointerMoved(at)]);
        egui::DragAndDrop::set_payload(win.ctx(), Payload::Text);
        frame(win, ed, vec![E::PointerButton { pos: at, button: B::Primary, pressed: false, modifiers: M::NONE }]);
        frame(win, ed, vec![]);
        check(format!("T text dropped on V3: {:?}", ed.clip_text(V3, 100)), ed.clip_text(V3, 100).is_some());
        press(win, ed, K::Escape, M::NONE);

        // 17. The load-clip UI (the `clips` feature): cmd+I opens seq_manip's
        // browser as a window over the viewers; it lists a folder (timed);
        // Enter on a movie puts it in the bin and the source and closes the
        // window; two rows dragged down onto the timeline lay both clips
        // down one after the other; a file dropped on the bin joins it;
        // Scrub's letter keys stand down over the window; Esc closes it.
        #[cfg(feature = "clips")]
        {
            let bin = ed.pane_rect(Pane::Bin).expect("bin");
            frame(win, ed, vec![E::PointerMoved(bin.center())]);
            press(win, ed, K::I, M::COMMAND);
            frame(win, ed, vec![]);
            check("cmd I opens the clip browser".into(), ed.clips_open());
            let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata");
            let t0 = std::time::Instant::now();
            ed.clips_go(&dir);
            let listed = t0.elapsed();
            frame(win, ed, vec![]);
            let rows = ed.clips_rows();
            let find = |n: &str| rows.iter().find(|r| r.1.file_name().is_some_and(|f| f == n)).cloned();
            let (mov, mp4) = (find("prores_hq_4k.mov"), find("h264_1080_audio_tc.mp4"));
            check(format!("the browser lists testdata ({} media, {listed:?})", rows.len()), mov.is_some() && mp4.is_some() && listed.as_millis() < 500);
            if let (Some((id, path)), Some((id2, _))) = (mov, mp4) {
                let n0 = ed.bin_groups().len();
                ed.clips_activate(id);
                frame(win, ed, vec![]);
                frame(win, ed, vec![]);
                check(format!("Enter: source {:?}, bin {} -> {}, window closed", ed.source_path(), n0, ed.bin_groups().len()), ed.source_path().as_deref() == Some(path.as_path()) && ed.bin_groups().iter().any(|g| g.1 == path) && !ed.clips_open());
                // Open again; two rows dragged down onto V1 at 180 (overwriting).
                press(win, ed, K::I, M::COMMAND);
                frame(win, ed, vec![]);
                frame(win, ed, vec![]);
                let wr = ed.clips_rect().unwrap_or(egui::Rect::NOTHING);
                let at = pt(ed, V1, 180.0);
                check(format!("the window leaves the timeline uncovered: {wr:?} {at:?}"), !wr.contains(at));
                frame(win, ed, vec![E::PointerMoved(at)]);
                egui::DragAndDrop::set_payload(win.ctx(), imfram::gallery::Dragged(vec![id, id2]));
                frame(win, ed, vec![E::PointerMoved(at)]);
                frame(win, ed, vec![E::PointerButton { pos: at, button: B::Primary, pressed: false, modifiers: M::NONE }]);
                frame(win, ed, vec![]);
                let v1: Vec<_> = clips(ed).into_iter().filter(|c| c.0 == V1 && c.1 >= 180).collect();
                let in_order = v1.len() >= 2 && v1[0].1 == 180 && v1[1].1 == v1[0].1 + v1[0].2;
                check(format!("two dragged clips laid down in order on V1: {v1:?}"), in_order);
                // Onto the bin.
                let n1 = ed.bin_groups().len();
                frame(win, ed, vec![E::PointerMoved(bin.center())]);
                egui::DragAndDrop::set_payload(win.ctx(), Payload::Files(vec![dir.join("h264_4k.mp4"), dir.join("hevc_4k_10bit.mov")]));
                frame(win, ed, vec![E::PointerButton { pos: bin.center(), button: B::Primary, pressed: false, modifiers: M::NONE }]);
                frame(win, ed, vec![]);
                check(format!("two files dropped on the bin: {n1} -> {}", ed.bin_groups().len()), ed.bin_groups().len() == n1 + 2);
                // The to-import tray: B over the window collects the mp4
                // here, then a sequence from another folder; the tray keeps
                // both across the move, in that order, and Import puts them
                // in the bin in that order and closes the window.
                frame(win, ed, vec![E::PointerMoved(wr.center())]);
                let canon = |p: &std::path::Path| p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
                let id3 = ed.clips_rows().iter().find(|r| r.1.file_name().is_some_and(|f| f == "h264_4k_50m.mp4")).map(|r| r.0).unwrap_or(id2);
                ed.clips_select(&[id3]);
                press(win, ed, K::B, M::NONE);
                frame(win, ed, vec![]);
                ed.clips_go(&dir.join("exr"));
                frame(win, ed, vec![]);
                let seqrow = ed.clips_rows().first().cloned();
                if let Some((sid, _)) = &seqrow {
                    ed.clips_select(&[*sid]);
                }
                press(win, ed, K::B, M::NONE);
                frame(win, ed, vec![]);
                let tray = ed.clips_tray();
                let want_first = canon(&dir.join("h264_4k_50m.mp4"));
                check(format!("B collects across folders, in order: {tray:?}"), tray.len() == 2 && canon(&tray[0]) == want_first && seqrow.as_ref().is_some_and(|s| canon(&tray[1]) == canon(&s.1)));
                let n2 = ed.bin_groups().len();
                ed.clips_import();
                frame(win, ed, vec![]);
                frame(win, ed, vec![]);
                let g = ed.bin_groups();
                let tail: Vec<_> = g.iter().rev().take(2).rev().map(|x| canon(&x.1)).collect();
                let want: Vec<_> = tray.iter().map(|p| canon(p)).collect();
                check(format!("Import: bin {n2} -> {}, last two {tail:?}, tray {:?}, window {}", g.len(), ed.clips_tray(), ed.clips_open()), g.len() == n2 + 2 && tail == want && ed.clips_tray().is_empty() && !ed.clips_open());
                press(win, ed, K::I, M::COMMAND);
                frame(win, ed, vec![]);
                ed.clips_go(&dir);
                frame(win, ed, vec![]);
                // Keys over the window are the browser's; Esc closes it.
                press(win, ed, K::V, M::NONE);
                frame(win, ed, vec![E::PointerMoved(wr.center())]);
                press(win, ed, K::R, M::NONE);
                check(format!("R over the browser is not Scrub's: {:?}", ed.tool()), ed.tool() == Tool::Select);
                press(win, ed, K::Escape, M::NONE);
                frame(win, ed, vec![]);
                check("Esc closes the browser".into(), !ed.clips_open());
            }
        }

        // 18. Bin tiles: posters for the tiles on screen, hover-scrub to the
        // exact frame under the pointer (timed, on an intra and a long-GOP
        // movie and an EXR sequence), cmd click picks two and a drag lays
        // both down in the bin's order.
        {
            let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata").canonicalize().expect("testdata");
            let want = [dir.join("prores_hq_4k.mov"), dir.join("h264_4k.mp4"), dir.join("exr/shot.1001.exr")];
            for p in &want {
                ed.bin_add(p);
            }
            ed.bin_set_tiles(true);
            let bin = ed.pane_rect(Pane::Bin).expect("bin");
            frame(win, ed, vec![E::PointerMoved(bin.center() + egui::vec2(0.0, 2000.0))]);
            // Scroll the tiles we want into view one by one as needed: the
            // bin is narrow, so find each tile's rect when it is drawn.
            let t0 = std::time::Instant::now();
            let mut rects: Vec<(std::path::PathBuf, egui::Rect)> = Vec::new();
            while t0.elapsed().as_secs_f32() < 20.0 {
                frame(win, ed, vec![]);
                rects = ed.bin_tile_rects();
                let shown: Vec<_> = rects.iter().filter(|r| want.contains(&r.0)).collect();
                if !shown.is_empty() && shown.iter().all(|r| ed.bin_scrub(&r.0, 0).0.is_some()) {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            let posters = t0.elapsed();
            let shown: Vec<(std::path::PathBuf, egui::Rect)> = rects.iter().filter(|r| want.contains(&r.0)).cloned().collect();
            check(format!("bin tiles: {} wanted tiles on screen, their info in {posters:?}", shown.len()), !shown.is_empty() && shown.iter().all(|r| ed.bin_scrub(&r.0, 0).0.is_some()));
            let mut report = Vec::new();
            for (path, rect) in &shown {
                let n = ed.bin_scrub(path, 0).0.unwrap_or(1);
                let pic_w = rect.width();
                let pic_h = (pic_w * 9.0 / 16.0).round();
                let y = rect.top() + pic_h * 0.5;
                let at = |frac: f32| egui::pos2(rect.left() + pic_w * frac, y);
                let target = |frac: f32| ((frac.clamp(0.0, 1.0)) * (n - 1) as f32).round() as u64;
                // Cold: the first moves over the tile (the strip pass starts).
                let mut cold = Vec::new();
                let t_fill = std::time::Instant::now();
                for k in 0..4 {
                    let frac = 0.11 + 0.21 * k as f32;
                    let s0 = std::time::Instant::now();
                    frame(win, ed, vec![E::PointerMoved(at(frac))]);
                    while !ed.bin_scrub(path, target(frac)).1 && s0.elapsed().as_secs_f32() < 4.0 {
                        frame(win, ed, vec![E::PointerMoved(at(frac))]);
                    }
                    cold.push(s0.elapsed().as_secs_f32() * 1000.0);
                }
                // The strip fills (every frame of a clip up to 240 long).
                while ed.bin_scrub(path, 0).2 < n.min(240) as usize && t_fill.elapsed().as_secs_f32() < 30.0 {
                    frame(win, ed, vec![E::PointerMoved(at(0.5))]);
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                let fill = t_fill.elapsed().as_secs_f32();
                // Warm: a sweep across the tile, every position exact on the
                // redraw the pointer moved in.
                let mut redraws = Vec::new();
                let mut warm = Vec::new();
                for k in 0..20 {
                    let frac = 0.02 + 0.049 * k as f32;
                    let s0 = std::time::Instant::now();
                    let mut r = 1;
                    frame(win, ed, vec![E::PointerMoved(at(frac))]);
                    while !ed.bin_scrub(path, target(frac)).1 && r < 400 {
                        frame(win, ed, vec![E::PointerMoved(at(frac))]);
                        r += 1;
                    }
                    redraws.push(r);
                    warm.push(s0.elapsed().as_secs_f32() * 1000.0);
                }
                let med = |v: &mut Vec<f32>| {
                    v.sort_by(f32::total_cmp);
                    v[v.len() / 2]
                };
                let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                check(format!("{name}: warm scrub exact on the first redraw: {redraws:?}"), redraws.iter().all(|&r| r == 1));
                check(format!("{name}: strip filled ({} of {n}) in {fill:.1}s", ed.bin_scrub(path, 0).2), ed.bin_scrub(path, 0).2 >= n.min(240) as usize);
                report.push(format!("{name} ({n} f): cold exact median {:.0} ms, strip {fill:.1}s, then {:.1} ms (1 redraw)", med(&mut cold), med(&mut warm)));
            }
            let (_, dec) = ed.bin_scrub_ms();
            let mut dec = dec;
            dec.sort_by(f32::total_cmp);
            println!("INFO bin tiles: posters in {posters:?}; hover-scrub to the exact frame: {}; decode+box per frame median {:.1} ms", report.join("; "), dec.get(dec.len() / 2).copied().unwrap_or(0.0));
            // cmd click two tiles, drag them down onto V1 at 400.
            if shown.len() >= 2 {
                frame(win, ed, vec![E::PointerMoved(bin.center() + egui::vec2(0.0, 2000.0))]);
                click(win, ed, shown[0].1.center(), M::NONE);
                click(win, ed, shown[1].1.center(), M::COMMAND);
                check(format!("cmd click picks two tiles: {}", ed.bin_chosen()), ed.bin_chosen() == 2);
                let at = pt(ed, V2, 160.0);
                let from = shown[1].1.center();
                frame(win, ed, vec![E::PointerMoved(from)]);
                frame(win, ed, vec![E::PointerButton { pos: from, button: B::Primary, pressed: true, modifiers: M::NONE }]);
                for k in 1..=8 {
                    frame(win, ed, vec![E::PointerMoved(from + (at - from) * (k as f32 / 8.0))]);
                }
                let carrying = egui::DragAndDrop::payload::<Payload>(win.ctx()).map(|p| match &*p { Payload::Files(v) => v.len(), Payload::File(_) => 1, _ => 0 });
                frame(win, ed, vec![E::PointerButton { pos: at, button: B::Primary, pressed: false, modifiers: M::NONE }]);
                frame(win, ed, vec![]);
                check(format!("a tile drag carries both: {carrying:?}"), carrying == Some(2));
                let v2: Vec<_> = clips(ed).into_iter().filter(|c| c.0 == V2 && c.1 >= 160).collect();
                check(format!("two tiles dragged lay down in order on V2: {v2:?}"), v2.len() >= 2 && v2[0].1 == 160 && v2[1].1 == v2[0].1 + v2[0].2);
            }
            ed.bin_set_tiles(false);
            frame(win, ed, vec![]);
        }

        if fails.is_empty() {
            println!("TEST PASS editing tools: audio tracks + mute/solo, razor (click, shift, cmd K), past the end, four-mark fit, rate stretch, slip, slide, marquee, alt duplicate, dissolve between two clips (and refusal), cut menu, drops; readouts toggle, viewer zoom + R / shift R, context R, clear in/out buttons; hold keys A D Z C S, shift A D S, D tap, T snap; slip band + two-up; titles: new, type, Enter, viewer drag, copy + retype, cmd arrows, bin drop; clip browser + to-import tray; bin tiles + hover-scrub");
        } else {
            for f in &fails {
                println!("TEST FAIL {f}");
            }
        }
    }
}
