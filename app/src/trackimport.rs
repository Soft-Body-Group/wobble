//! Planar-track import: corner pins that another tracker exported as text,
//! keyed onto a CornerPin's (or a Transform's) pin channels.
//!
//! Three plain-text shapes, named generically in the UI:
//! - **corner pin keys**: tab-separated keyframe text, one section per
//!   corner (`Effects <effect> <corner>`), with the source size in its
//!   header. Optional `Anchor Point` / `Position` / `Scale` / `Rotation`
//!   sections (the motion-blur flavour) move the corners as a layer
//!   transform would.
//! - **transform keys**: the same text with only those four sections; the
//!   frame's corners are carried through the transform.
//! - **script corner pin**: a node-script block with `to1..to4` (and
//!   optionally `from1..from4`) as `{curve x<frame> v v v ...}` lists, y up.
//!
//! Everything is parsed to pixel corners, y DOWN, order TL TR BR BL, then
//! divided by the source size — a CornerPin's quad is normalised, so a
//! track exported at one resolution lands on a plate of another unchanged.

use crate::anim::{self, AnimCurve, Channel};
use std::collections::{BTreeMap, BTreeSet};

pub type Quad = [[f32; 2]; 4];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    CornerPinKeys,
    TransformKeys,
    ScriptCornerPin,
}

impl Format {
    pub fn label(self) -> &'static str {
        match self {
            Format::CornerPinKeys => "corner pin keys",
            Format::TransformKeys => "transform keys",
            Format::ScriptCornerPin => "script corner pin",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Track {
    pub format: Format,
    /// The source size the pixels are measured in (`None`: the text did
    /// not say, and the caller's fallback was used).
    pub size: [f32; 2],
    pub size_from_text: bool,
    pub fps: Option<f32>,
    /// Per frame (the FILE's numbering), pixel corners, y down, TL TR BR BL.
    pub frames: Vec<(i64, Quad)>,
    /// A constant source quad, pixels, when the text carries one that is
    /// not simply the frame.
    pub from: Option<Quad>,
}

impl Track {
    pub fn first(&self) -> i64 {
        self.frames.first().map_or(0, |f| f.0)
    }
    pub fn last(&self) -> i64 {
        self.frames.last().map_or(0, |f| f.0)
    }
    fn norm(&self, q: Quad) -> Quad {
        q.map(|[x, y]| [x / self.size[0], y / self.size[1]])
    }
    /// Normalised corners per frame.
    pub fn normalised(&self) -> Vec<(i64, Quad)> {
        self.frames.iter().map(|(f, q)| (*f, self.norm(*q))).collect()
    }
    pub fn normalised_from(&self) -> Option<Quad> {
        self.from.map(|q| self.norm(q))
    }
    /// One line for the window: what was found.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "{}: {} frames ({}..{}), {}x{}{}",
            self.format.label(),
            self.frames.len(),
            self.first(),
            self.last(),
            self.size[0],
            self.size[1],
            if self.size_from_text { "" } else { " (node format)" },
        );
        if let Some(fps) = self.fps {
            s += &format!(", {fps} fps");
        }
        if self.from.is_some() {
            s += ", with a source quad";
        }
        s
    }
}

/// Parse any of the three shapes. `fallback` is the size used when the text
/// does not carry one (the node's format).
pub fn parse(text: &str, fallback: [f32; 2]) -> Result<Track, String> {
    if text.contains("Keyframe Data") {
        parse_keys(text, fallback)
    } else if text.contains("to1") || text.contains("CornerPin") {
        parse_script(text, fallback)
    } else if text.trim().is_empty() {
        Err("paste tracking data or load a file".into())
    } else {
        Err("not tracking data this can read: expected corner pin or transform keyframe text, or a script corner pin".into())
    }
}

// ---------------------------------------------------------------- keys

/// One sampled property: frame -> values.
type Samples = BTreeMap<i64, Vec<f32>>;

/// The value at `f`: exact, else held from the nearest earlier sample, else
/// the first.
fn sample(s: &Samples, f: i64) -> Option<&[f32]> {
    s.range(..=f).next_back().or_else(|| s.iter().next()).map(|(_, v)| v.as_slice())
}

fn corner_of(name: &str) -> Option<usize> {
    let n = name.to_ascii_lowercase();
    let has = |a: &str, b: &str| n.contains(a) || n.contains(b);
    if has("upper left", "top left") {
        Some(0)
    } else if has("upper right", "top right") {
        Some(1)
    } else if has("lower right", "bottom right") {
        Some(2)
    } else if has("lower left", "bottom left") {
        Some(3)
    } else if n.ends_with("-0002") {
        Some(0)
    } else if n.ends_with("-0003") {
        Some(1)
    } else if n.ends_with("-0004") {
        Some(3)
    } else if n.ends_with("-0005") {
        Some(2)
    } else {
        None
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Sect {
    Corner(usize),
    Anchor,
    Position,
    Scale,
    Rotation,
    Other,
}

fn parse_keys(text: &str, fallback: [f32; 2]) -> Result<Track, String> {
    let mut w = None;
    let mut h = None;
    let mut fps = None;
    let mut corners: [Samples; 4] = Default::default();
    let (mut anchor, mut pos, mut scale, mut rot) =
        (Samples::new(), Samples::new(), Samples::new(), Samples::new());
    let mut sect = Sect::Other;
    for raw in text.lines() {
        let line = raw.trim_end_matches(['\r', '\n']);
        if line.trim().is_empty() {
            continue;
        }
        let indented = line.starts_with(['\t', ' ']);
        if !indented {
            let fields: Vec<&str> = line.split('\t').map(str::trim).collect();
            sect = match fields[0] {
                "Effects" => fields.last().and_then(|n| corner_of(n)).map_or(Sect::Other, Sect::Corner),
                "Anchor Point" => Sect::Anchor,
                "Position" => Sect::Position,
                "Scale" => Sect::Scale,
                "Rotation" | "Z Rotation" => Sect::Rotation,
                _ => Sect::Other,
            };
            continue;
        }
        let fields: Vec<&str> = line.split(['\t', ' ']).filter(|s| !s.is_empty()).collect();
        let Some(first) = fields.first() else { continue };
        let num = |i: usize| fields.get(i).and_then(|s| s.parse::<f32>().ok());
        if let Ok(frame) = first.parse::<f64>() {
            let vals: Vec<f32> = fields[1..].iter().filter_map(|s| s.parse().ok()).collect();
            let f = frame.round() as i64;
            let target = match sect {
                Sect::Corner(i) => &mut corners[i],
                Sect::Anchor => &mut anchor,
                Sect::Position => &mut pos,
                Sect::Scale => &mut scale,
                Sect::Rotation => &mut rot,
                Sect::Other => continue,
            };
            if !vals.is_empty() {
                target.insert(f, vals);
            }
            continue;
        }
        // Header lines: "\tSource Width\t1920" (the label may hold spaces,
        // so read the LAST field as the number).
        let label = line.trim().rsplit_once(['\t', ' ']).map(|(l, _)| l.trim()).unwrap_or("");
        let last = fields.len().checked_sub(1).and_then(num);
        match label {
            "Source Width" => w = last,
            "Source Height" => h = last,
            "Units Per Second" => fps = last,
            _ => {}
        }
    }
    let size_from_text = w.is_some() && h.is_some();
    let size = match (w, h) {
        (Some(w), Some(h)) if w > 0.0 && h > 0.0 => [w, h],
        _ => fallback,
    };
    let have_corners = corners.iter().filter(|c| !c.is_empty()).count();
    if have_corners > 0 && have_corners < 4 {
        return Err(format!("the corner pin text has {have_corners} of its 4 corners"));
    }
    let have_xf = [&anchor, &pos, &scale, &rot].iter().any(|s| !s.is_empty());
    if have_corners == 0 && !have_xf {
        return Err("the keyframe text holds no corner pin or transform data".into());
    }
    let mut frames: BTreeSet<i64> = BTreeSet::new();
    for s in corners.iter().chain([&anchor, &pos, &scale, &rot]) {
        frames.extend(s.keys().copied());
    }
    let centre = [size[0] * 0.5, size[1] * 0.5];
    let rect: Quad = [[0.0, 0.0], [size[0], 0.0], [size[0], size[1]], [0.0, size[1]]];
    let xy = |s: &Samples, f: i64, d: [f32; 2]| {
        sample(s, f).map_or(d, |v| [v[0], v.get(1).copied().unwrap_or(v[0])])
    };
    let out = frames
        .into_iter()
        .map(|f| {
            let base: Quad = if have_corners == 4 {
                std::array::from_fn(|i| xy(&corners[i], f, rect[i]))
            } else {
                rect
            };
            // A layer transform: pos + R * S * (p - anchor). Defaults are a
            // frame-sized layer's, so a missing section is the identity.
            let a = xy(&anchor, f, centre);
            let p = xy(&pos, f, centre);
            let s = xy(&scale, f, [100.0, 100.0]);
            let r = sample(&rot, f).map_or(0.0, |v| v[0]).to_radians();
            let (sn, cs) = r.sin_cos();
            let q = base.map(|[x, y]| {
                let (dx, dy) = ((x - a[0]) * s[0] / 100.0, (y - a[1]) * s[1] / 100.0);
                [p[0] + cs * dx - sn * dy, p[1] + sn * dx + cs * dy]
            });
            (f, q)
        })
        .collect();
    Ok(Track {
        format: if have_corners == 4 { Format::CornerPinKeys } else { Format::TransformKeys },
        size,
        size_from_text,
        fps,
        frames: out,
        from: None,
    })
}

// ---------------------------------------------------------------- script

#[derive(Debug)]
enum Val {
    Const(f32),
    Curve(BTreeMap<i64, f32>),
}

impl Val {
    fn at(&self, f: i64) -> f32 {
        match self {
            Val::Const(v) => *v,
            Val::Curve(m) => m
                .range(..=f)
                .next_back()
                .or_else(|| m.iter().next())
                .map_or(0.0, |(_, v)| *v),
        }
    }
    fn frames(&self) -> impl Iterator<Item = i64> + '_ {
        match self {
            Val::Const(_) => None,
            Val::Curve(m) => Some(m.keys().copied()),
        }
        .into_iter()
        .flatten()
    }
}

fn tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        if c == '{' || c == '}' {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            out.push(c.to_string());
        } else if c.is_whitespace() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        } else {
            cur.push(c);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Read a `{curve ...}` body (tokens after `curve`, up to the closing brace).
fn curve(body: &[String]) -> BTreeMap<i64, f32> {
    let mut m = BTreeMap::new();
    let mut frame = 1i64;
    let mut i = 0;
    while i < body.len() {
        let t = body[i].as_str();
        if let Some(n) = t.strip_prefix('x').and_then(|n| n.parse::<f64>().ok()) {
            frame = n.round() as i64;
        } else if let Ok(v) = t.parse::<f32>() {
            m.insert(frame, v);
            frame += 1;
        } else if matches!(t, "s" | "t" | "u" | "v") {
            i += 1; // a tangent, and its number
        }
        i += 1;
    }
    m
}

/// The two values of `{X Y}` starting at `toks[i] == "{"`.
fn pair(toks: &[String], mut i: usize) -> Option<[Val; 2]> {
    if toks.get(i)? != "{" {
        return None;
    }
    i += 1;
    let mut vals = Vec::new();
    while vals.len() < 2 {
        let t = toks.get(i)?;
        if t == "}" {
            break;
        }
        if t == "{" {
            // An inner group: a curve, or an expression we cannot read.
            let start = i + 1;
            let mut depth = 1;
            i += 1;
            while depth > 0 {
                match toks.get(i)?.as_str() {
                    "{" => depth += 1,
                    "}" => depth -= 1,
                    _ => {}
                }
                i += 1;
            }
            let body = &toks[start..i - 1];
            if body.first().map(String::as_str) == Some("curve") {
                vals.push(Val::Curve(curve(&body[1..])));
            } else {
                vals.push(Val::Const(body.first()?.parse().ok()?));
            }
        } else {
            vals.push(Val::Const(t.parse().ok()?));
            i += 1;
        }
    }
    let mut it = vals.into_iter();
    Some([it.next()?, it.next()?])
}

fn parse_script(text: &str, fallback: [f32; 2]) -> Result<Track, String> {
    let toks = tokens(text);
    let find = |name: &str| -> Option<[Val; 2]> {
        let i = toks.iter().position(|t| t == name)?;
        pair(&toks, i + 1)
    };
    // to1..to4 are BL BR TR TL in a y-up frame.
    let to: Vec<Option<[Val; 2]>> = (1..=4).map(|k| find(&format!("to{k}"))).collect();
    let missing = to.iter().filter(|t| t.is_none()).count();
    if missing > 0 {
        return Err(format!("the script corner pin is missing {missing} of to1..to4"));
    }
    let to: Vec<[Val; 2]> = to.into_iter().map(Option::unwrap).collect();
    let from: Vec<Option<[Val; 2]>> = (1..=4).map(|k| find(&format!("from{k}"))).collect();
    let from: Option<[[f32; 2]; 4]> = if from.iter().all(Option::is_some) {
        let v: Vec<[f32; 2]> =
            from.iter().map(|p| p.as_ref().map(|[x, y]| [x.at(0), y.at(0)]).unwrap()).collect();
        Some([v[0], v[1], v[2], v[3]])
    } else {
        None
    };
    // A `from` that is the frame itself, (0,0) to (W,H), IS the size.
    let frame_from = from.and_then(|f| {
        let (w, h) = (f[2][0], f[2][1]);
        let is_rect = f[0] == [0.0, 0.0] && f[1] == [w, 0.0] && f[3] == [0.0, h];
        (is_rect && w > 0.0 && h > 0.0).then_some([w, h])
    });
    let size = frame_from.unwrap_or(fallback);
    let flip = |[x, y]: [f32; 2]| [x, size[1] - y];
    // BL BR TR TL -> TL TR BR BL.
    const ORDER: [usize; 4] = [3, 2, 1, 0];
    let mut frames: BTreeSet<i64> = BTreeSet::new();
    for [x, y] in &to {
        frames.extend(x.frames().chain(y.frames()));
    }
    if frames.is_empty() {
        frames.insert(0);
    }
    let out = frames
        .into_iter()
        .map(|f| {
            let q: Quad = std::array::from_fn(|i| {
                let [x, y] = &to[ORDER[i]];
                flip([x.at(f), y.at(f)])
            });
            (f, q)
        })
        .collect();
    let from = match (from, frame_from) {
        (Some(f), None) => Some(std::array::from_fn(|i| flip(f[ORDER[i]]))),
        _ => None,
    };
    Ok(Track {
        format: Format::ScriptCornerPin,
        size,
        size_from_text: frame_from.is_some(),
        fps: None,
        frames: out,
        from,
    })
}

// ---------------------------------------------------------------- apply

pub const PIN: [[Channel; 2]; 4] = [
    [Channel::Pin0X, Channel::Pin0Y],
    [Channel::Pin1X, Channel::Pin1Y],
    [Channel::Pin2X, Channel::Pin2Y],
    [Channel::Pin3X, Channel::Pin3Y],
];
pub const FROM: [[Channel; 2]; 4] = [
    [Channel::From0X, Channel::From0Y],
    [Channel::From1X, Channel::From1Y],
    [Channel::From2X, Channel::From2Y],
    [Channel::From3X, Channel::From3Y],
];

#[derive(Debug, Default, PartialEq)]
pub struct Applied {
    pub keys: usize,
    /// Frames dropped because the offset put them before comp frame 0.
    pub dropped: usize,
    /// Channels left alone because they are locked.
    pub locked: usize,
}

/// Key `track` onto the pin channels at `file frame + offset`, replacing
/// whatever keys those channels had. `from` is the constant source quad to
/// set (normalised), for a CornerPin only.
pub fn apply(
    anim: &mut BTreeMap<Channel, AnimCurve>,
    track: &Track,
    offset: i64,
    with_from: bool,
) -> (Applied, Option<Quad>) {
    let mut out = Applied::default();
    let frames = track.normalised();
    for (i, chans) in PIN.iter().enumerate() {
        for (axis, ch) in chans.iter().enumerate() {
            if anim::is_locked(anim, *ch) {
                out.locked += 1;
                continue;
            }
            let c = anim.entry(*ch).or_default();
            c.keys.clear();
            for (f, q) in &frames {
                let t = f + offset;
                if t < 0 {
                    continue;
                }
                c.set_key(t, q[i][axis]);
                out.keys += 1;
            }
        }
    }
    out.dropped = frames.iter().filter(|(f, _)| f + offset < 0).count();
    let from = if with_from { track.normalised_from() } else { None };
    (out, from)
}

// ---------------------------------------------------------------- window

/// The import window's state: one at a time, for one node.
pub struct Win {
    pub node: u64,
    pub name: String,
    pub text: String,
    pub offset: i64,
    /// The offset still follows the plate (nobody has typed one).
    pub auto_offset: bool,
    pub with_from: bool,
    pub status: String,
    /// The node takes a source quad (a CornerPin; a Transform has none).
    pub has_from: bool,
}

impl Win {
    pub fn new(node: u64, name: String, has_from: bool) -> Self {
        Self {
            node,
            name,
            text: String::new(),
            offset: 0,
            auto_offset: true,
            with_from: true,
            status: String::new(),
            has_from,
        }
    }
}

/// What the window asks the app to do this frame.
pub enum Act {
    Apply { track: Track, offset: i64, with_from: bool },
    LoadFile,
    Close,
}

/// Draw the window. `fallback` is the node's format, `plate_first` the
/// first comp frame of what feeds it (where file frame 0 lands by default).
pub fn window(ctx: &egui::Context, w: &mut Win, fallback: [f32; 2], plate_first: i64) -> Option<Act> {
    let mut act = None;
    let mut open = true;
    let parsed = parse(&w.text, fallback);
    if w.auto_offset {
        if let Ok(t) = &parsed {
            w.offset = plate_first - t.first();
        }
    }
    egui::Window::new(format!("import track: {}", w.name))
        .id(egui::Id::new("track-import"))
        .open(&mut open)
        .collapsible(false)
        .default_width(420.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                #[cfg(not(target_arch = "wasm32"))]
                if ui.button("paste").on_hover_text("read the clipboard").clicked() {
                    match arboard::Clipboard::new().and_then(|mut c| c.get_text()) {
                        Ok(t) => w.text = t,
                        Err(e) => w.status = format!("clipboard: {e}"),
                    }
                }
                if ui.button("load file…").clicked() {
                    act = Some(Act::LoadFile);
                }
                if ui.button("clear").clicked() {
                    w.text.clear();
                    w.auto_offset = true;
                }
            });
            egui::ScrollArea::vertical().max_height(160.0).show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(&mut w.text)
                        .font(egui::TextStyle::Monospace)
                        .desired_rows(8)
                        .desired_width(f32::INFINITY)
                        .hint_text("paste corner pin or transform keyframe text here"),
                );
            });
            match &parsed {
                Ok(t) => {
                    ui.label(t.summary());
                    ui.horizontal(|ui| {
                        ui.label("frame offset");
                        if ui.add(egui::DragValue::new(&mut w.offset).speed(0.2)).changed() {
                            w.auto_offset = false;
                        }
                        if !w.auto_offset && ui.small_button("plate start").clicked() {
                            w.auto_offset = true;
                        }
                        ui.weak(format!(
                            "frames {}..{} land on {}..{}",
                            t.first(),
                            t.last(),
                            t.first() + w.offset,
                            t.last() + w.offset
                        ));
                    });
                    if w.has_from && t.from.is_some() {
                        ui.checkbox(&mut w.with_from, "set the source quad too");
                    }
                    if ui.button("apply").on_hover_text("replaces the pin's keys").clicked() {
                        act = Some(Act::Apply {
                            track: t.clone(),
                            offset: w.offset,
                            with_from: w.has_from && w.with_from,
                        });
                    }
                }
                Err(e) => {
                    ui.weak(e);
                }
            }
            if !w.status.is_empty() {
                ui.label(&w.status);
            }
        });
    if !open {
        act = Some(Act::Close);
    }
    act
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Corner pin keys, the four-corner effect, written by hand in the
    /// shape the exports take: tab-indented, a size header, frame 0 up.
    const PIN_KEYS: &str = "Adobe After Effects 6.0 Keyframe Data\r
\r
\tUnits Per Second\t24\r
\tSource Width\t1920\r
\tSource Height\t1080\r
\tSource Pixel Aspect Ratio\t1\r
\tComp Pixel Aspect Ratio\t1\r
\r
Effects\tCC Power Pin #1\tCC Power Pin-0002\r
\tFrame\tX pixels\tY pixels\t\r
\t0\t100\t50\t\r
\t1\t110\t52\t\r
\t2\t120\t54\t\r
\r
Effects\tCC Power Pin #1\tCC Power Pin-0003\r
\tFrame\tX pixels\tY pixels\t\r
\t0\t1800\t60\t\r
\t1\t1810\t62\t\r
\t2\t1820\t64\t\r
\r
Effects\tCC Power Pin #1\tCC Power Pin-0004\r
\tFrame\tX pixels\tY pixels\t\r
\t0\t90\t1000\t\r
\t1\t100\t1002\t\r
\t2\t110\t1004\t\r
\r
Effects\tCC Power Pin #1\tCC Power Pin-0005\r
\tFrame\tX pixels\tY pixels\t\r
\t0\t1850\t1020\t\r
\t1\t1860\t1022\t\r
\t2\t1870\t1024\t\r
\r
End of Keyframe Data\r
";

    const CORNER_NAMES: &str = "Adobe After Effects 8.0 Keyframe Data

\tUnits Per Second\t25
\tSource Width\t1000
\tSource Height\t500

Effects\tCorner Pin #1\tUpper Left #2
\tFrame\tX pixels\tY pixels
\t10\t0\t0
Effects\tCorner Pin #1\tUpper Right #3
\tFrame\tX pixels\tY pixels
\t10\t1000\t0
Effects\tCorner Pin #1\tLower Left #4
\tFrame\tX pixels\tY pixels
\t10\t0\t500
Effects\tCorner Pin #1\tLower Right #5
\tFrame\tX pixels\tY pixels
\t10\t1000\t500

End of Keyframe Data
";

    const XF_KEYS: &str = "Adobe After Effects 6.0 Keyframe Data

\tUnits Per Second\t24
\tSource Width\t200
\tSource Height\t100

Anchor Point
\tFrame\tX pixels\tY pixels\tZ pixels
\t0\t100\t50\t0
Position
\tFrame\tX pixels\tY pixels\tZ pixels
\t0\t100\t50\t0
\t1\t110\t50\t0
Scale
\tFrame\tX percent\tY percent\tZ percent
\t0\t100\t100\t100
\t1\t50\t50\t100
Rotation
\tFrame\tDegrees
\t0\t0
\t1\t90

End of Keyframe Data
";

    const SCRIPT: &str = "CornerPin2D {
 to1 {{curve x1001 100 101} {curve x1001 80 81}}
 to2 {{curve x1001 1800 1801} {curve x1001 90 91}}
 to3 {{curve x1001 1820 1821} {curve x1001 1000 1001}}
 to4 {{curve x1001 120 121} {curve x1001 1010 1011}}
 from1 {0 0}
 from2 {1920 0}
 from3 {1920 1080}
 from4 {0 1080}
 name CornerPin2D1
}
";

    fn close(a: [f32; 2], b: [f32; 2]) -> bool {
        (a[0] - b[0]).abs() < 1e-3 && (a[1] - b[1]).abs() < 1e-3
    }

    #[test]
    fn corner_pin_keys_parse_in_tl_tr_br_bl() {
        let t = parse(PIN_KEYS, [1.0, 1.0]).unwrap();
        assert_eq!(t.format, Format::CornerPinKeys);
        assert_eq!(t.size, [1920.0, 1080.0]);
        assert_eq!(t.fps, Some(24.0));
        assert_eq!(t.frames.len(), 3);
        let (f, q) = t.frames[1];
        assert_eq!(f, 1);
        assert!(close(q[0], [110.0, 52.0]));
        assert!(close(q[1], [1810.0, 62.0]));
        assert!(close(q[2], [1860.0, 1022.0]), "BR is -0005");
        assert!(close(q[3], [100.0, 1002.0]), "BL is -0004");
        let n = t.normalised()[0].1;
        assert!(close(n[0], [100.0 / 1920.0, 50.0 / 1080.0]));
    }

    #[test]
    fn named_corners_and_the_frame_come_back_as_the_unit_quad() {
        let t = parse(CORNER_NAMES, [1.0, 1.0]).unwrap();
        assert_eq!(t.first(), 10);
        let n = t.normalised()[0].1;
        for (a, b) in n.iter().zip(engine::graph::UNIT_QUAD) {
            assert!(close(*a, b), "{n:?}");
        }
    }

    #[test]
    fn transform_keys_carry_the_frame_corners() {
        let t = parse(XF_KEYS, [1.0, 1.0]).unwrap();
        assert_eq!(t.format, Format::TransformKeys);
        // Frame 0 is the identity.
        let q0 = t.frames[0].1;
        assert!(close(q0[0], [0.0, 0.0]) && close(q0[2], [200.0, 100.0]), "{q0:?}");
        // Frame 1: half size, a quarter turn clockwise (y down) about the
        // centre, moved 10 px right. TL (-100,-50) -> scaled (-50,-25) ->
        // turned (25,-50) -> + (110,50).
        let q1 = t.frames[1].1;
        assert!(close(q1[0], [135.0, 0.0]), "{q1:?}");
    }

    #[test]
    fn a_script_corner_pin_flips_y_and_reorders() {
        let t = parse(SCRIPT, [1.0, 1.0]).unwrap();
        assert_eq!(t.format, Format::ScriptCornerPin);
        assert_eq!(t.size, [1920.0, 1080.0], "the frame-sized from is the size");
        assert!(t.from.is_none(), "a frame-sized from is no source quad");
        assert_eq!((t.first(), t.last()), (1001, 1002));
        let q = t.frames[0].1;
        // to4 is TL: (120, 1010) y-up -> (120, 70) y-down.
        assert!(close(q[0], [120.0, 70.0]), "{q:?}");
        assert!(close(q[1], [1820.0, 80.0]), "TR is to3");
        assert!(close(q[2], [1800.0, 990.0]), "BR is to2");
        assert!(close(q[3], [100.0, 1000.0]), "BL is to1");
    }

    #[test]
    fn a_script_without_a_size_uses_the_fallback_and_keeps_its_from() {
        let s = "to1 {0 0}\nto2 {100 0}\nto3 {100 50}\nto4 {0 50}\nfrom1 {10 10}\nfrom2 {90 10}\nfrom3 {90 40}\nfrom4 {10 40}";
        let t = parse(s, [100.0, 50.0]).unwrap();
        assert!(!t.size_from_text);
        assert_eq!(t.size, [100.0, 50.0]);
        let f = t.normalised_from().unwrap();
        assert!(close(f[0], [0.1, 0.2]), "TL is from4, flipped: {f:?}");
    }

    #[test]
    fn partial_and_foreign_text_are_errors_that_say_why() {
        let three = PIN_KEYS.replace("CC Power Pin-0005", "Something else");
        assert!(parse(&three, [1.0, 1.0]).unwrap_err().contains("3 of its 4"));
        assert!(parse("hello", [1.0, 1.0]).is_err());
        assert!(parse("", [1.0, 1.0]).is_err());
        // No other app is named in what the user reads.
        for e in [parse("hello", [1.0; 2]).unwrap_err(), parse(&three, [1.0; 2]).unwrap_err()] {
            assert!(!e.contains("After Effects") && !e.contains("Nuke"), "{e}");
        }
    }

    #[test]
    fn apply_keys_the_pin_at_the_offset_and_respects_locks() {
        let t = parse(PIN_KEYS, [1.0, 1.0]).unwrap();
        let mut anim = BTreeMap::new();
        anim.entry(Channel::Pin0X).or_insert_with(AnimCurve::default).set_key(500u64, 0.9);
        anim.entry(Channel::Pin3Y).or_insert_with(AnimCurve::default).locked = true;
        let (a, from) = apply(&mut anim, &t, 1001, true);
        assert!(from.is_none());
        assert_eq!(a.locked, 1);
        assert_eq!(a.keys, 7 * 3);
        let c = &anim[&Channel::Pin0X];
        assert_eq!(c.keys.len(), 3, "old keys replaced");
        assert_eq!(c.keys[0].frame, 1001.0);
        assert!((c.eval_at(1002.0).unwrap() - 110.0 / 1920.0).abs() < 1e-6);
        assert!(anim[&Channel::Pin3Y].keys.is_empty());
        // An offset that pushes frames before 0 drops them, and says so.
        let (a, _) = apply(&mut anim, &t, -1, false);
        assert_eq!(a.dropped, 1);
    }
}
