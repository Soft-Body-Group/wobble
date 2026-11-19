//! Script tracker and corner pin blocks pasted into the node graph become a
//! Tracker or a Corner Pin with their keyframes.
//!
//! Two block classes are understood, beside the Read blocks `pathpaste`
//! turns into Loads (a paste may mix all three):
//!
//! - **`Tracker4`**: its `tracks` table — one row per track, its columns
//!   named in the table's own header — gives each track's name, its
//!   `track_x` / `track_y` (a constant or a `{curve x<frame> v v …}`), and
//!   its pattern and search boxes. Each track becomes one Tracker point.
//! - **`CornerPin2D`**: `to1..to4` (and `from1..from4`) through
//!   `trackimport`'s script reader, keyed onto the pin channels.
//!
//! Script pixels are y UP from the bottom-left; a node's are normalised, y
//! down. The size dividing them is, in order: a `Root` block's `format` in
//! the same paste, a corner pin's `from` when it is the whole frame, the
//! comp's format (`fallback`). Frames are comp frames on both sides (a tick
//! IS a comp frame number), so keys land where they were.

use crate::anim::AnimCurve;
use crate::pathpaste::{script_blocks, tokens, Tok};
use engine::graph::{CornerPinParams, NodeKind, TrackPoint, TrackerParams, TRACK_POINTS_MAX};
use std::collections::BTreeMap;

/// A pasted block, parsed, before it is a node.
#[derive(Debug)]
pub(crate) enum Pasted {
    Tracker { params: TrackerParams, keys: Vec<[BTreeMap<i64, f32>; 2]> },
    CornerPin { params: CornerPinParams, anim: BTreeMap<crate::anim::Channel, AnimCurve> },
}

/// Every tracker / corner pin block in `text`, parsed. Empty when there are
/// none (or `text` is not a script).
pub(crate) fn parse(text: &str, fallback: [f32; 2]) -> Vec<Pasted> {
    let Some(blocks) = script_blocks(text.trim_start_matches('\u{feff}')) else {
        return Vec::new();
    };
    let size = blocks
        .iter()
        .find(|(c, _)| c == "Root")
        .and_then(|(_, b)| root_format(b))
        .unwrap_or(fallback);
    blocks
        .iter()
        .filter_map(|(class, body)| match class.as_str() {
            "Tracker4" | "Tracker3" => tracker_block(body, size),
            "CornerPin2D" => corner_pin_block(body, size),
            _ => None,
        })
        .collect()
}

/// Make the nodes `text` asks for: the Loads its Read blocks name (through
/// `pathpaste`) and a Tracker / Corner Pin per tracker block, beside them.
/// `None` when it asks for none of these.
pub(crate) fn paste(
    graph: &mut crate::editor::EditorGraph,
    text: &str,
    at: egui::Pos2,
    next_read_slot: &mut usize,
    fallback: [f32; 2],
) -> Option<Vec<u64>> {
    const GAP: f32 = 150.0;
    let mut ids = crate::pathpaste::paste_loads(graph, text, at, next_read_slot).unwrap_or_default();
    let nodes = parse(text, fallback);
    // Below the Loads when there are any: the stack reads top-down.
    let y = if ids.is_empty() { at.y } else { at.y + GAP };
    let x0 = at.x - GAP * nodes.len().saturating_sub(1) as f32 / 2.0;
    for (i, p) in nodes.into_iter().enumerate() {
        let pos = egui::pos2(x0 + GAP * i as f32, y);
        let (kind, anim) = match p {
            Pasted::Tracker { params, keys } => {
                let mut anim = BTreeMap::new();
                for (k, [x, y]) in keys.iter().enumerate() {
                    let [cx, cy] = crate::tracker::chans(k);
                    for (ch, m) in [(cx, x), (cy, y)] {
                        if m.is_empty() {
                            continue;
                        }
                        let c: &mut AnimCurve = anim.entry(ch).or_default();
                        for (f, v) in m {
                            if *f >= 0 {
                                c.set_key(*f, *v);
                            }
                        }
                    }
                }
                (NodeKind::Tracker(params), anim)
            }
            Pasted::CornerPin { params, anim } => (NodeKind::CornerPin(params), anim),
        };
        let id = graph.add(kind, pos);
        if let Some(n) = graph.nodes.get_mut(&id) {
            n.anim = anim;
        }
        ids.push(id);
    }
    (!ids.is_empty()).then_some(ids)
}

// ---- knobs ---------------------------------------------------------------

/// A block's knobs: each line's first word, and its words after it.
fn knobs(body: &str) -> BTreeMap<String, Vec<Tok>> {
    let mut out = BTreeMap::new();
    let mut line: Vec<Tok> = Vec::new();
    let mut flush = |line: &mut Vec<Tok>| {
        if let Some(Tok::Word(k)) = line.first() {
            let k = k.clone();
            out.entry(k).or_insert_with(|| line.drain(1..).collect());
        }
        line.clear();
    };
    for t in tokens(body) {
        match t {
            Tok::Newline => flush(&mut line),
            t => line.push(t),
        }
    }
    flush(&mut line);
    out
}

fn word(t: &Tok) -> Option<&str> {
    match t {
        Tok::Word(w) | Tok::Group(w) => Some(w.as_str()),
        Tok::Newline => None,
    }
}

/// `format "2048 1556 0 0 2048 1556 1 2K"` → its width and height.
fn root_format(body: &str) -> Option<[f32; 2]> {
    let k = knobs(body);
    let f = k.get("format")?.first().and_then(word)?;
    let mut it = f.split_whitespace().map(|s| s.parse::<f32>());
    let (w, h) = (it.next()?.ok()?, it.next()?.ok()?);
    (w > 0.0 && h > 0.0).then_some([w, h])
}

/// A value cell: a number, or a `{curve …}` (frame → value). An expression
/// or an empty cell is `None`.
fn cell(t: &Tok) -> Option<BTreeMap<i64, f32>> {
    match t {
        Tok::Word(w) => w.trim().parse::<f32>().ok().map(|v| BTreeMap::from([(i64::MIN, v)])),
        Tok::Group(g) => {
            let toks = crate::trackimport::tokens(g);
            match toks.first().map(String::as_str) {
                Some("curve") => {
                    let m = crate::trackimport::curve(&toks[1..]);
                    (!m.is_empty()).then_some(m)
                }
                // `{1024}` and the like.
                Some(v) if toks.len() == 1 => v.parse::<f32>().ok().map(|v| BTreeMap::from([(i64::MIN, v)])),
                _ => None,
            }
        }
        Tok::Newline => None,
    }
}

/// The value of a cell at `f` (held before its first key and after its last).
fn at(m: &BTreeMap<i64, f32>, f: i64) -> f32 {
    m.range(..=f).next_back().or_else(|| m.iter().next()).map_or(0.0, |(_, v)| *v)
}

/// Whether a cell holds keys (a constant is stored at `i64::MIN`).
fn keyed(m: &BTreeMap<i64, f32>) -> bool {
    m.keys().any(|f| *f != i64::MIN)
}

// ---- Tracker4 ------------------------------------------------------------

fn tracker_block(body: &str, size: [f32; 2]) -> Option<Pasted> {
    let k = knobs(body);
    let table = k.get("tracks")?.first().and_then(word)?;
    // { header } { columns } { rows }
    let parts: Vec<String> = tokens(table)
        .into_iter()
        .filter_map(|t| match t {
            Tok::Group(g) => Some(g),
            _ => None,
        })
        .collect();
    let [_, cols, rows] = parts.as_slice() else { return None };
    let columns: Vec<String> = tokens(cols)
        .into_iter()
        .filter_map(|t| match t {
            Tok::Group(g) => g.split_whitespace().nth(3).map(str::to_string),
            _ => None,
        })
        .collect();
    let col = |n: &str| columns.iter().position(|c| c == n);
    let (cx, cy) = (col("track_x")?, col("track_y")?);
    let reference = k
        .get("reference_frame")
        .and_then(|v| v.first())
        .and_then(word)
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map(|f| f.round() as i64);
    let (w, h) = (size[0], size[1]);
    let mut points: Vec<TrackPoint> = Vec::new();
    let mut keys: Vec<[BTreeMap<i64, f32>; 2]> = Vec::new();
    for row in tokens(rows).into_iter().filter_map(|t| match t {
        Tok::Group(g) => Some(g),
        _ => None,
    }) {
        if points.len() >= TRACK_POINTS_MAX {
            break;
        }
        let cells: Vec<Tok> = tokens(&row).into_iter().filter(|t| !matches!(t, Tok::Newline)).collect();
        let num = |c: Option<usize>, f: i64| c.and_then(|c| cells.get(c)).and_then(cell).map(|m| at(&m, f));
        let (Some(x), Some(y)) = (cells.get(cx).and_then(cell), cells.get(cy).and_then(cell)) else {
            continue;
        };
        // The reference frame: the script's, else the first tracked one.
        let f0 = reference
            .or_else(|| x.keys().chain(y.keys()).copied().filter(|f| *f != i64::MIN).min())
            .unwrap_or(0);
        let c = [at(&x, f0) / w, 1.0 - at(&y, f0) / h];
        // Pattern and search boxes are offsets from the point, in pixels.
        let pat = ["pattern_x", "pattern_y", "pattern_r", "pattern_t"].map(|n| num(col(n), f0));
        let half = match pat {
            [Some(l), Some(b), Some(r), Some(t)] if r > l && t > b => {
                [((r - l) * 0.5 / w).clamp(0.002, 0.5), ((t - b) * 0.5 / h).clamp(0.002, 0.5)]
            }
            _ => TrackerParams::default().size,
        };
        let srch = ["search_x", "search_r"].map(|n| num(col(n), f0));
        let search = match (srch, pat) {
            ([Some(l), Some(r)], [Some(pl), _, Some(pr), _]) if r > l => {
                (((r - l) - (pr - pl)).max(0.0) * 0.5 / w.max(h)).clamp(0.005, 0.4)
            }
            _ => TrackerParams::default().search,
        };
        points.push(TrackPoint { center: c, size: half, search, track: c });
        let norm = |m: &BTreeMap<i64, f32>, d: f32, flip: bool| -> BTreeMap<i64, f32> {
            if !keyed(m) {
                return BTreeMap::new();
            }
            m.iter().map(|(f, v)| (*f, if flip { 1.0 - v / d } else { v / d })).collect()
        };
        keys.push([norm(&x, w, false), norm(&y, h, true)]);
    }
    let first = points.first().copied()?;
    let mut params = TrackerParams::default();
    params.set_point(0, first);
    params.more = points[1..].to_vec();
    if keys.iter().any(|[x, y]| !x.is_empty() || !y.is_empty()) {
        params.anchors = points.iter().map(|p| p.center).collect();
        params.frame = size;
    }
    Some(Pasted::Tracker { params, keys })
}

// ---- CornerPin2D ---------------------------------------------------------

fn corner_pin_block(body: &str, size: [f32; 2]) -> Option<Pasted> {
    let track = crate::trackimport::parse(&format!("CornerPin2D {{\n{body}\n}}"), size).ok()?;
    let frames = track.normalised();
    let first = frames.first()?.1;
    let mut params = CornerPinParams::pinned(first);
    let mut anim = BTreeMap::new();
    // One frame is a still pin: set it, key nothing.
    let with_from = if frames.len() > 1 {
        crate::trackimport::apply(&mut anim, &track, 0, true).1
    } else {
        track.normalised_from()
    };
    if let Some(from) = with_from {
        params.from = from;
    }
    Some(Pasted::CornerPin { params, anim })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRACKER: &str = r#"set cut_paste_input [stack 0]
version 14.0 v5
push $cut_paste_input
Tracker4 {
 tracks { { 1 31 2 }
{ { 5 1 20 enable e 1 }
{ 3 1 75 name name 1 }
{ 2 1 58 track_x track_x 1 }
{ 2 1 58 track_y track_y 1 }
{ 2 1 63 offset_x offset_x 1 }
{ 2 1 63 offset_y offset_y 1 }
{ 4 1 27 T T 1 }
{ 4 1 27 R R 1 }
{ 4 1 27 S S 1 }
{ 2 0 45 error error 1 }
{ 1 1 0 error_min error_min 1 }
{ 1 1 0 error_max error_max 1 }
{ 1 1 0 pattern_x pattern_x 1 }
{ 1 1 0 pattern_y pattern_y 1 }
{ 1 1 0 pattern_r pattern_r 1 }
{ 1 1 0 pattern_t pattern_t 1 }
{ 1 1 0 search_x search_x 1 }
{ 1 1 0 search_y search_y 1 }
{ 1 1 0 search_r search_r 1 }
{ 1 1 0 search_t search_t 1 }
}
{
 { {curve K x1001 1} "track 1" {curve x1001 1000 1010 1020} {curve x1001 500 505 510} {} {} 1 0 0 {curve x1001 0 0.1 0.2} 1 0 -20 -20 20 20 -40 -40 40 40 }
 { {curve K x1001 1} "track 2" 300 200 {} {} 1 0 0 {} 1 0 -10 -10 10 10 -30 -30 30 30 }
}
}
 reference_frame 1001
 name Tracker7
 xpos 100
 ypos 200
}
"#;

    #[test]
    fn tracker4_tracks_become_points_with_keys() {
        let v = parse(TRACKER, [1920.0, 1080.0]);
        let [Pasted::Tracker { params, keys }] = v.as_slice() else { panic!("{v:?}") };
        assert_eq!(params.count(), 2);
        let p0 = params.point(0).unwrap();
        assert!((p0.center[0] - 1000.0 / 1920.0).abs() < 1e-6);
        assert!((p0.center[1] - (1.0 - 500.0 / 1080.0)).abs() < 1e-6, "y flips");
        assert!((p0.size[0] - 20.0 / 1920.0).abs() < 1e-6);
        assert_eq!(keys[0][0].len(), 3);
        assert!((keys[0][0][&1003] - 1020.0 / 1920.0).abs() < 1e-6);
        assert!((keys[0][1][&1002] - (1.0 - 505.0 / 1080.0)).abs() < 1e-6);
        // A constant track is placed, not keyed.
        assert!(keys[1][0].is_empty());
        let p1 = params.point(1).unwrap();
        assert!((p1.center[0] - 300.0 / 1920.0).abs() < 1e-6);
        assert_eq!(params.anchors.len(), 2);
    }

    #[test]
    fn root_format_sizes_the_paste() {
        let text = format!("Root {{\n format \"1000 1000 0 0 1000 1000 1 sq\"\n}}\n{TRACKER}");
        let v = parse(&text, [1920.0, 1080.0]);
        let [Pasted::Tracker { params, .. }] = v.as_slice() else { panic!() };
        assert!((params.point(0).unwrap().center[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn corner_pin_keys_land_on_the_pin() {
        let text = "CornerPin2D {\n to1 {{curve x1001 0 10} {curve x1001 0 0}}\n to2 {{curve x1001 100 110} 0}\n to3 {100 100}\n to4 {0 100}\n from1 {0 0}\n from2 {100 0}\n from3 {100 100}\n from4 {0 100}\n name CornerPin2D3\n}\n";
        let v = parse(text, [1920.0, 1080.0]);
        let [Pasted::CornerPin { params, anim }] = v.as_slice() else { panic!("{v:?}") };
        // `from` is the whole frame, so it sizes the pin: 100 x 100.
        assert_eq!(params.from, engine::graph::UNIT_QUAD);
        // to1 is BL, y up: TL in wobble's order is to4 = (0, 100) -> (0, 0).
        assert_eq!(params.to[0], [0.0, 0.0]);
        let bl = &anim[&crate::trackimport::PIN[3][0]];
        assert_eq!(bl.keys.len(), 2);
        assert!((bl.keys[1].value - 0.1).abs() < 1e-6);
    }

    #[test]
    fn plain_text_and_other_blocks_make_nothing() {
        assert!(parse("hello there", [1.0, 1.0]).is_empty());
        assert!(parse("Blur {\n size 3\n}\n", [1.0, 1.0]).is_empty());
    }
}
