//! Frame ranges as metadata that flows down the graph.
//!
//! The script owns a global range (plus handles either side of it). Every
//! node then has an output range of its own, resolved the way a compositor
//! expects: a Read guesses from its media, a generator adopts the script's,
//! a Write inherits whatever feeds it, and anything else is the span its
//! inputs cover. Any node can override the answer.
//!
//! The point is that a CLIP knows how long it is independently of the
//! script, so a retime has something to remap and a Write has something to
//! render without either of them guessing.
//!
//! Owned by instance B (see .agents/board.md).

use crate::editor::EditorGraph;
use engine::graph::NodeKind;
use std::collections::BTreeMap;

/// An inclusive frame range, in the clock of the node that reports it.
pub type Range = (u64, u64);

pub fn len(r: Range) -> u64 {
    r.1.saturating_sub(r.0) + 1
}

/// Where a node's range came from, so the UI can say rather than imply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// Set by hand on this node.
    Override,
    /// A Read's media (its trim if it has one, else the whole sequence).
    Media,
    /// Nothing upstream to ask: generators run for as long as the script.
    Script,
    /// Inherited from what feeds this node.
    Upstream,
    /// Inherited, then moved by a time node.
    Retimed,
}

impl Source {
    pub fn label(self) -> &'static str {
        match self {
            Source::Override => "set here",
            Source::Media => "from media",
            Source::Script => "script range",
            Source::Upstream => "from input",
            Source::Retimed => "retimed",
        }
    }
}

/// What the resolver needs that the graph itself cannot answer.
pub struct Ctx<'a> {
    pub graph: &'a EditorGraph,
    /// The script's own range, which generators adopt.
    pub script: Range,
    /// A Read's media range, when its sequence is loaded.
    pub media: Box<dyn Fn(u64) -> Option<Range> + 'a>,
}

/// Resolve every node's output range in one walk.
pub fn all(cx: &Ctx) -> BTreeMap<u64, (Range, Source)> {
    let mut out = BTreeMap::new();
    let ids: Vec<u64> = cx.graph.nodes.keys().copied().collect();
    for id in ids {
        resolve(cx, id, &mut out, &mut Vec::new());
    }
    out
}

/// One node's output range. Memoised into `out`; `stack` breaks cycles.
pub fn resolve(
    cx: &Ctx,
    id: u64,
    out: &mut BTreeMap<u64, (Range, Source)>,
    stack: &mut Vec<u64>,
) -> (Range, Source) {
    if let Some(hit) = out.get(&id) {
        return *hit;
    }
    if stack.contains(&id) {
        return (cx.script, Source::Script); // cycle: fall back to the script
    }
    let Some(n) = cx.graph.node(id) else {
        return (cx.script, Source::Script);
    };
    // An explicit range always wins, whatever the node is.
    if let Some(r) = n.range {
        let r = (r.0, r.1.max(r.0));
        out.insert(id, (r, Source::Override));
        return (r, Source::Override);
    }
    stack.push(id);
    let answer = match &n.kind {
        // A Read is the one node that knows something nobody else does.
        // Timeline placement shifts its whole span: files numbered 1001..
        // occupy comp frames 1001.. (negative starts clamp at 0 — the
        // reader holds frame one before its start anyway).
        // With its explicit first/last (every loaded Read has them) that IS
        // the answer — in comp frames already, files or no files.
        NodeKind::Read { first: Some(f), last: Some(l), .. } => {
            ((((*f).max(0)) as u64, ((*l).max(*f).max(0)) as u64), Source::Media)
        }
        NodeKind::Read { range, start, .. } => {
            let shift = |r: Range| -> Range {
                let s = start.unwrap_or(0);
                (
                    (r.0 as i64 + s).max(0) as u64,
                    (r.1 as i64 + s).max(0) as u64,
                )
            };
            match range {
                Some(r) => (shift((r.0, r.1.max(r.0))), Source::Media),
                None => match (cx.media)(id) {
                    Some(r) => (shift(r), Source::Media),
                    None => (cx.script, Source::Script),
                },
            }
        }
        // Metadata only: the noodle says these frames, whatever is above.
        // Past its source's own, the source's before/after answers.
        // "Start at" moves them: the in point plays at `at`.
        NodeKind::TimeRange(p) => {
            let (a, b) = p.output();
            ((a.max(0) as u64, b.max(a).max(0) as u64), Source::Override)
        }
        // A GenAI node showing a VIDEO take: the frames the take really
        // has, from the frame it was made at. A model may hand back fewer
        // (its length limit) or more (its own frame rate) than it was
        // sent, and the range the noodle carries — and "node range" in
        // the pie — must be the frames there is a picture for.
        NodeKind::GenAi(p) if p.pipeline.picked().is_some_and(|t| t.is_video()) && (cx.media)(id).is_some() => {
            let at = p.pipeline.picked().map_or(0, |t| t.tick.max(0) as u64);
            let (a, b) = (cx.media)(id).unwrap_or((0, 0));
            ((at + a, at + b.max(a)), Source::Media)
        }
        kind => {
            // The PLATE's span (input 0), or the span peers cover (a
            // Blend's A and B) — never an element's: a Convolve's kernel
            // or a Defocus's depth must not stretch the shot. A Switch is
            // as long as the input it is ON. Nothing wired (a generator,
            // or a Read that never loaded) means the script decides.
            let mut span: Option<Range> = None;
            for i in kind.range_inputs(|i| cx.graph.input_of(id, i).is_some()) {
                let Some(src) = cx.graph.input_of(id, i) else { continue };
                let (r, _) = resolve(cx, src, out, stack);
                span = Some(match span {
                    None => r,
                    Some(s) => (s.0.min(r.0), s.1.max(r.1)),
                });
            }
            match span {
                None => (cx.script, Source::Script),
                Some(s) => match time_shift(&n.kind, s) {
                    Some(moved) => (moved, Source::Retimed),
                    None => (s, Source::Upstream),
                },
            }
        }
    };
    stack.pop();
    out.insert(id, answer);
    answer
}

/// Where a time node's output range lands, when that is knowable.
///
/// A time node maps OUTPUT tick to SOURCE frame, so a range of source frames
/// becomes the ticks that reach it — the inverse. That inverse is exact for
/// a shift or a constant rate, and not a function at all for a lookup curve
/// (which may visit a frame many times, or never), so a remap keeps its
/// input's range and says so.
pub fn time_shift(kind: &NodeKind, input: Range) -> Option<Range> {
    let (speed, offset) = match kind {
        NodeKind::Slip(p) => (1.0f32, p.offset as f32),
        NodeKind::Speed(p) => (p.speed, p.anchor - p.anchor * p.speed),
        NodeKind::RangeRemap(p) => p.affine(),
        // Anchored on the input's first frame: the shot starts where it
        // did and runs 1 / speed as long.
        // A curve keeps its input's range, like a TimeRemap.
        NodeKind::Rifetime(p) if p.mode == engine::graph::RifetimeMode::Speed => {
            let a = input.0 as f32;
            (p.speed(), a - a * p.speed())
        }
        _ => return None,
    };
    if speed.abs() < 1e-6 {
        return None; // frozen: every tick shows one frame
    }
    let inv = |src: f32| (src - offset) / speed;
    let (a, b) = (inv(input.0 as f32), inv(input.1 as f32));
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    Some((lo.round().max(0.0) as u64, hi.round().max(0.0) as u64))
}

/// A Load's range as the panel shows it: `inp` / `out` are CLIP frames (the
/// files' own numbers, `f0` being the first file's), `at` the comp frame the
/// in point lands on. Stored as `start` (comp frame of the first file) and
/// `first` / `last` (comp frames), so setups are unchanged and the player
/// keeps one rule; this is only the translation.
///
/// Moving in or out with `at` held is a SLIP: comp `at` shows the new in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClipRange {
    pub inp: i64,
    pub out: i64,
    pub at: i64,
}

impl ClipRange {
    pub fn of(f0: i64, start: i64, first: i64, last: i64) -> Self {
        Self { inp: f0 + first - start, out: f0 + last - start, at: first }
    }

    /// `(start, first, last)` that put this on the comp.
    pub fn stored(self, f0: i64) -> (i64, i64, i64) {
        let out = self.out.max(self.inp);
        (self.at - (self.inp - f0), self.at, self.at + (out - self.inp))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::Pos2;
    use engine::graph::{SlipParams, SpeedParams};

    const SCRIPT: Range = (0, 99);

    fn ctx<'a>(g: &'a EditorGraph, media: &'a dyn Fn(u64) -> Option<Range>) -> Ctx<'a> {
        Ctx { graph: g, script: SCRIPT, media: Box::new(media) }
    }

    #[test]
    fn a_read_guesses_from_its_media_and_a_trim_wins() {
        let mut g = EditorGraph::default();
        let r = g.add(NodeKind::read(0, ""), Pos2::ZERO);
        let media = |_: u64| Some((0u64, 52u64));
        let mut out = BTreeMap::new();
        assert_eq!(
            resolve(&ctx(&g, &media), r, &mut out, &mut Vec::new()),
            ((0, 52), Source::Media)
        );
        // A trim on the Read is the media answer, narrowed.
        if let Some(NodeKind::Read { range, .. }) = g.node_mut(r).map(|n| &mut n.kind) {
            *range = Some((10, 20));
        }
        let mut out = BTreeMap::new();
        assert_eq!(
            resolve(&ctx(&g, &media), r, &mut out, &mut Vec::new()),
            ((10, 20), Source::Media)
        );
        // Nothing loaded and no trim: the script decides.
        let mut g2 = EditorGraph::default();
        let r2 = g2.add(NodeKind::read(0, ""), Pos2::ZERO);
        let none = |_: u64| None;
        let mut out = BTreeMap::new();
        assert_eq!(
            resolve(&ctx(&g2, &none), r2, &mut out, &mut Vec::new()),
            (SCRIPT, Source::Script)
        );
    }

    #[test]
    fn a_reads_first_and_last_are_its_range_and_flow_on() {
        let mut g = EditorGraph::default();
        let r = g.add(NodeKind::read(0, ""), Pos2::ZERO);
        let grade = g.add(NodeKind::Grade(Default::default()), Pos2::ZERO);
        g.connect(grade, 0, r);
        if let Some(NodeKind::Read { first, last, .. }) = g.node_mut(r).map(|n| &mut n.kind) {
            // Wider than the files on purpose: the frames past them are black.
            *first = Some(1001);
            *last = Some(1080);
        }
        let media = |_: u64| Some((0u64, 68u64));
        let m = all(&ctx(&g, &media));
        assert_eq!(m[&r], ((1001, 1080), Source::Media));
        assert_eq!(m[&grade], ((1001, 1080), Source::Upstream));
    }

    #[test]
    fn generators_take_the_script_and_downstream_inherits() {
        let mut g = EditorGraph::default();
        let gen = g.add(NodeKind::Spline(Default::default()), Pos2::ZERO);
        let grade = g.add(NodeKind::Grade(Default::default()), Pos2::ZERO);
        g.connect(grade, 0, gen);
        let none = |_: u64| None;
        let m = all(&ctx(&g, &none));
        assert_eq!(m[&gen], (SCRIPT, Source::Script));
        assert_eq!(m[&grade], (SCRIPT, Source::Upstream));
    }

    #[test]
    fn an_override_wins_anywhere_and_flows_on() {
        let mut g = EditorGraph::default();
        let r = g.add(NodeKind::read(0, ""), Pos2::ZERO);
        let grade = g.add(NodeKind::Grade(Default::default()), Pos2::ZERO);
        g.connect(grade, 0, r);
        g.node_mut(r).unwrap().range = Some((5, 9));
        let media = |_: u64| Some((0u64, 52u64));
        let m = all(&ctx(&g, &media));
        assert_eq!(m[&r], ((5, 9), Source::Override), "override beats media");
        assert_eq!(m[&grade], ((5, 9), Source::Upstream), "and flows downstream");
        // Overriding the consumer detaches it from what feeds it.
        g.node_mut(grade).unwrap().range = Some((100, 120));
        let m = all(&ctx(&g, &media));
        assert_eq!(m[&grade], ((100, 120), Source::Override));
    }

    #[test]
    fn two_inputs_cover_the_union() {
        let mut g = EditorGraph::default();
        let a = g.add(NodeKind::read(0, ""), Pos2::ZERO);
        let b = g.add(NodeKind::read(1, ""), Pos2::ZERO);
        let blend = g.add(NodeKind::blend(), Pos2::ZERO);
        g.connect(blend, 0, a);
        g.connect(blend, 1, b);
        g.node_mut(a).unwrap().range = Some((0, 10));
        g.node_mut(b).unwrap().range = Some((20, 30));
        let none = |_: u64| None;
        let m = all(&ctx(&g, &none));
        assert_eq!(m[&blend], ((0, 30), Source::Upstream));
    }

    #[test]
    fn an_element_input_never_stretches_the_plates_range() {
        // The plate at 1001..1080; a one-frame element at frame 0 in every
        // node that reads one. The range is the plate's, whatever the port.
        let cases: Vec<(NodeKind, Vec<usize>)> = vec![
            (NodeKind::Convolve(Default::default()), vec![engine::nodes::convolve::KERNEL_INPUT]),
            (NodeKind::Defocus(Default::default()), vec![2, 3]),
            (NodeKind::Diffuse(Default::default()), vec![engine::nodes::diffuse::DEPTH_INPUT]),
            (NodeKind::VectorBlur(Default::default()), vec![1]),
            (NodeKind::VectorWarp(Default::default()), vec![1]),
            (NodeKind::Lightwrap(Default::default()), vec![1]),
            (NodeKind::Keyer(Default::default()), vec![1]),
            (NodeKind::ScreenKey(Default::default()), vec![1, 2]),
            (NodeKind::EdgeExtend(Default::default()), vec![engine::nodes::edgeextend::MATTE_INPUT]),
            (NodeKind::Godrays(Default::default()), vec![engine::nodes::godrays::SOURCE_INPUT]),
            (NodeKind::Stabilize(Default::default()), vec![1]),
            (NodeKind::CorridorKey(Default::default()), vec![1]),
        ];
        for (kind, elems) in cases {
            let name = kind.name();
            let mut g = EditorGraph::default();
            let plate = g.add(NodeKind::read(0, ""), Pos2::ZERO);
            g.node_mut(plate).unwrap().range = Some((1001, 1080));
            let node = g.add(kind, Pos2::ZERO);
            g.connect(node, 0, plate);
            for (n, port) in elems.into_iter().enumerate() {
                let e = g.add(NodeKind::read(1 + n, ""), Pos2::ZERO);
                g.node_mut(e).unwrap().range = Some((0, 0));
                g.connect(node, port, e);
            }
            let none = |_: u64| None;
            let m = all(&ctx(&g, &none));
            assert_eq!(m[&node], ((1001, 1080), Source::Upstream), "{name}");
        }
        // With no plate wired, the element is all there is to go on.
        let mut g = EditorGraph::default();
        let k = g.add(NodeKind::read(0, ""), Pos2::ZERO);
        g.node_mut(k).unwrap().range = Some((3, 4));
        let conv = g.add(NodeKind::Convolve(Default::default()), Pos2::ZERO);
        g.connect(conv, engine::nodes::convolve::KERNEL_INPUT, k);
        let none = |_: u64| None;
        assert_eq!(all(&ctx(&g, &none))[&conv], ((3, 4), Source::Upstream));
    }

    #[test]
    fn a_shift_moves_the_range_and_a_rate_stretches_it() {
        // Slip +5: the clip's frames 0..10 now arrive 5 ticks EARLIER, so
        // the ticks that reach them are -5..5, clamped at zero.
        let slip = NodeKind::Slip(SlipParams { offset: 5 });
        assert_eq!(time_shift(&slip, (10, 20)), Some((5, 15)));
        // Half speed over an anchor of 0: 20 source frames take 40 ticks.
        let half = NodeKind::Speed(SpeedParams { speed: 0.5, ..Default::default() });
        assert_eq!(time_shift(&half, (0, 20)), Some((0, 40)));
        // Reversed footage still reports a forward range.
        let rev = NodeKind::Speed(SpeedParams { speed: -1.0, ..Default::default() });
        let r = time_shift(&rev, (0, 10)).unwrap();
        assert!(r.0 <= r.1, "a reversed clip still spans forwards: {r:?}");
        // A freeze has no inverse, and a lookup curve is not a function.
        let frozen = NodeKind::Speed(SpeedParams { speed: 0.0, ..Default::default() });
        assert_eq!(time_shift(&frozen, (0, 10)), None);
        assert_eq!(time_shift(&NodeKind::Grade(Default::default()), (0, 10)), None);
    }

    #[test]
    fn a_cycle_falls_back_instead_of_hanging() {
        let mut g = EditorGraph::default();
        let a = g.add(NodeKind::Grade(Default::default()), Pos2::ZERO);
        let b = g.add(NodeKind::Grade(Default::default()), Pos2::ZERO);
        g.connect(a, 0, b);
        g.connect(b, 0, a);
        let none = |_: u64| None;
        let m = all(&ctx(&g, &none));
        assert_eq!(m[&a].0, SCRIPT);
    }

    #[test]
    fn a_clip_range_round_trips_and_an_in_move_is_a_slip() {
        // Files 1..=500; in 414 at comp 1001.
        let c = ClipRange { inp: 414, out: 450, at: 1001 };
        let (start, first, last) = c.stored(1);
        assert_eq!((first, last), (1001, 1037));
        // Comp 1001 shows clip frame 414: file index first - start.
        assert_eq!(1 + first - start, 414);
        assert_eq!(ClipRange::of(1, start, first, last), c);
        // Slide in and out by 10 with `at` held: comp 1001 now shows 424.
        let slid = ClipRange { inp: 424, out: 460, ..c }.stored(1);
        assert_eq!((slid.1, slid.2), (1001, 1037));
        assert_eq!(1 + slid.1 - slid.0, 424);
    }

    #[test]
    fn a_time_range_sets_the_noodle_and_flows_on_without_a_pass() {
        let mut g = EditorGraph::default();
        let r = g.add(NodeKind::read(0, ""), Pos2::ZERO);
        let t = g.add(
            NodeKind::TimeRange(engine::graph::TimeRangeParams { first: 990, last: 1200, at: None }),
            Pos2::ZERO,
        );
        let c = g.add(NodeKind::Grade(Default::default()), Pos2::ZERO);
        g.edges.push((t, 0, r));
        g.edges.push((c, 0, t));
        let media = |_: u64| Some((0u64, 52u64));
        let mut out = BTreeMap::new();
        assert_eq!(resolve(&ctx(&g, &media), c, &mut out, &mut Vec::new()).0, (990, 1200));
        // The engine never sees it: the Grade reads the Read straight.
        assert_eq!(g.resolve_dot(t), Some(r));
    }

    #[test]
    fn a_time_range_start_at_slips_the_input() {
        let mut g = EditorGraph::default();
        let r = g.add(NodeKind::read(0, ""), Pos2::ZERO);
        let t = g.add(
            NodeKind::TimeRange(engine::graph::TimeRangeParams { first: 10, last: 40, at: Some(1001) }),
            Pos2::ZERO,
        );
        let c = g.add(NodeKind::Grade(Default::default()), Pos2::ZERO);
        g.edges.push((t, 0, r));
        g.edges.push((c, 0, t));
        let media = |_: u64| Some((0u64, 52u64));
        let mut out = BTreeMap::new();
        // Input 10..40 plays at 1001..1031.
        assert_eq!(resolve(&ctx(&g, &media), c, &mut out, &mut Vec::new()).0, (1001, 1031));
        // To the engine it is a Slip: tick 1001 reads input frame 10.
        assert_eq!(g.resolve_dot(t), Some(t));
        let eg = g.to_engine_graph_at_f(None);
        match eg.nodes.get(&t) {
            Some(NodeKind::Slip(p)) => assert_eq!(1001 + p.offset as i64, 10),
            k => panic!("expected a Slip, got {k:?}"),
        }
        assert_eq!(eg.edges.get(&(c, 0)), Some(&t));
        // "Start at" on its own frame is no slip: metadata only again.
        if let Some(NodeKind::TimeRange(p)) = g.nodes.get_mut(&t).map(|n| &mut n.kind) {
            p.at = Some(10);
        }
        assert_eq!(g.resolve_dot(t), Some(r));
    }

    #[test]
    fn a_genai_video_take_is_as_long_as_its_frames_from_where_it_was_made() {
        let mut g = EditorGraph::default();
        let r = g.add(NodeKind::read(0, ""), egui::Pos2::ZERO);
        let mut p = engine::graph::GenAiParams::new(1);
        let take = |file: &str| genai::Take {
            key: "k".into(),
            file: file.into(),
            tick: 1231,
            commercial: true,
            usd: None,
            label: String::new(),
            when: 0,
            data: false,
        };
        p.pipeline.takes = vec![take("result.mp4"), take("result.png")];
        p.pipeline.take = 0;
        let n = g.add(NodeKind::GenAi(p), egui::Pos2::ZERO);
        g.connect(n, 0, r);
        // The plate runs 0..=202; the take loaded in the node has 150 frames.
        let media = move |id: u64| if id == n { Some((0u64, 149u64)) } else { Some((0u64, 202u64)) };
        let mut out = BTreeMap::new();
        assert_eq!(resolve(&ctx(&g, &media), n, &mut out, &mut Vec::new()), ((1231, 1380), Source::Media));
        // A still holds everywhere: it keeps its input's range.
        if let Some(NodeKind::GenAi(p)) = g.node_mut(n).map(|e| &mut e.kind) {
            p.pipeline.take = 1;
        }
        let mut out = BTreeMap::new();
        assert_eq!(resolve(&ctx(&g, &media), n, &mut out, &mut Vec::new()).0, (0, 202));
    }
}
