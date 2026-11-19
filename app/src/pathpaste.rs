//! Text pasted into the node graph that names FOOTAGE becomes Load nodes.
//!
//! Two shapes are understood:
//!
//! - **Paths**, one per line: `/plates/sh010/bg.####.exr`, `bg.%04d.exr`,
//!   `bg.@@@@.exr`, `bg.$F4.exr`, a single frame of a sequence (the
//!   sequence on disk is found and loaded whole), a movie, a directory,
//!   Windows drive paths (`P:/x`, `P:\x`), UNC paths, quoted paths and
//!   `file://` URLs.
//! - **Script Read blocks** (`Read { file … first 1001 last 1900 … }`),
//!   several to a paste; every other block is ignored. The file pattern and
//!   its frame range come across.
//!
//! Anything else is not ours: [`parse`] answers `None` and the paste falls
//! through, so pasting a sentence into the graph never makes a node.
//!
//! Sequence patterns are normalised to the `####` form `load_read` resolves,
//! and the padding is checked against the files on disk through sequitur.

/// One Load a paste asks for. `start` / `first` / `last` are COMP frames in
/// the Read's own sense (`NodeKind::Read`); `None` leaves them to the load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PastedLoad {
    pub path: String,
    pub start: Option<i64>,
    pub first: Option<i64>,
    pub last: Option<i64>,
}

impl PastedLoad {
    fn path(path: String) -> Self {
        PastedLoad { path, start: None, first: None, last: None }
    }
}

/// The Loads `text` asks for, or `None` when it names no footage.
pub(crate) fn parse(text: &str) -> Option<Vec<PastedLoad>> {
    parse_with(text, &resolve_on_disk)
}

/// [`parse`] with the disk lookup injected, so the text rules are testable
/// without files.
fn parse_with(text: &str, resolve: &dyn Fn(&str) -> String) -> Option<Vec<PastedLoad>> {
    let text = text.trim_start_matches('\u{feff}');
    if let Some(blocks) = script_blocks(text) {
        let loads: Vec<PastedLoad> = blocks
            .iter()
            .filter(|(class, _)| class == "Read")
            .filter_map(|(_, body)| read_block(body, resolve))
            .collect();
        return (!loads.is_empty()).then_some(loads);
    }
    let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    if lines.is_empty() || lines.len() > 256 {
        return None;
    }
    let mut out = Vec::new();
    for l in lines {
        let p = clean_path(l)?;
        out.push(PastedLoad::path(resolve(&normalise(&p))));
    }
    Some(out)
}

/// A pasted line as a path: quotes and a `file://` stripped, and `None`
/// unless it LOOKS like one (absolute, home-relative, a drive, UNC or a
/// bucket). A relative fragment is far more likely to be prose.
fn clean_path(line: &str) -> Option<String> {
    let mut s = line.trim();
    for q in ['"', '\''] {
        if s.len() >= 2 && s.starts_with(q) && s.ends_with(q) {
            s = &s[1..s.len() - 1];
        }
    }
    let s = s.trim();
    let s = match s.strip_prefix("file://") {
        // file:///Users/x → /Users/x; file:///C:/x → C:/x
        Some(rest) => {
            let rest = percent_decode(rest);
            let b = rest.as_bytes();
            if b.len() > 3 && b[0] == b'/' && b[1].is_ascii_alphabetic() && b[2] == b':' {
                rest[1..].to_string()
            } else {
                rest
            }
        }
        None => s.to_string(),
    };
    looks_like_path(&s).then_some(s)
}

fn looks_like_path(s: &str) -> bool {
    let b = s.as_bytes();
    if s.contains(['\n', '\t', '<', '>', '|']) || s.len() < 2 {
        return false;
    }
    s.starts_with('/')
        || s.starts_with("~/")
        || s.starts_with("\\\\")
        || s.starts_with("s3://")
        || (b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1] == b':' && (b[2] == b'/' || b[2] == b'\\'))
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Where the file name starts: after the last separator of either kind.
fn name_at(p: &str) -> usize {
    p.rfind(['/', '\\']).map(|i| i + 1).unwrap_or(0)
}

/// Separators made forward (Windows takes `/` as well as `\`, and the
/// app's pattern resolution splits on `/` everywhere) — except a UNC path,
/// whose `\\server` prefix is only reliable written the Windows way. Then
/// the frame token in the FILE NAME written as `#`s: `%04d` / `%d`,
/// `$F4` / `${F4}` / `$F`, a run of `@`.
fn normalise(p: &str) -> String {
    let p = if p.starts_with("\\\\") { p.to_string() } else { p.replace('\\', "/") };
    let at = name_at(&p);
    let (dir, name) = p.split_at(at);
    format!("{dir}{}", hash_token(name))
}

fn hash_token(name: &str) -> String {
    let name = sequitur::convert_padding_to_hashes(name);
    // $F4 / ${F4} / $F
    let mut out = String::with_capacity(name.len());
    let c: Vec<char> = name.chars().collect();
    let mut i = 0;
    while i < c.len() {
        if c[i] == '$' {
            let braced = c.get(i + 1) == Some(&'{');
            let f = i + 1 + braced as usize;
            if c.get(f) == Some(&'F') {
                let mut j = f + 1;
                while j < c.len() && c[j].is_ascii_digit() {
                    j += 1;
                }
                let close = !braced || c.get(j) == Some(&'}');
                // `$Frame` is a variable called Frame, not a frame token.
                let ends = c.get(j + braced as usize).is_none_or(|n| !n.is_ascii_alphanumeric() && *n != '_');
                if close && ends {
                    let n: usize = c[f + 1..j].iter().collect::<String>().parse().unwrap_or(1);
                    out.push_str(&"#".repeat(n.clamp(1, 16)));
                    i = j + braced as usize;
                    continue;
                }
            }
        }
        out.push(c[i]);
        i += 1;
    }
    // @@@@ — one digit per @. Only when no other token was there.
    if !out.contains('#') && out.contains('@') {
        out = out.replace('@', "#");
    }
    out
}

/// Settle a normalised path against the disk:
/// - a `#` pattern whose padding does not match the files (`$F` for a
///   4-digit render) takes the files' padding;
/// - a single frame of a sequence becomes the sequence.
///
/// Anything not on this machine is kept exactly as pasted: a Load that says
/// "no such file" is more use than no Load.
fn resolve_on_disk(p: &str) -> String {
    #[cfg(target_arch = "wasm32")]
    {
        p.to_string()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::path::Path;
        let expanded = engine::paths::expand_home(p);
        let at = name_at(p);
        let (dir_s, name) = p.split_at(at);
        let Some(dir) = expanded.parent().filter(|d| d.is_dir()) else {
            return p.to_string();
        };
        let with = |n: String| format!("{dir_s}{n}");
        if name.contains('#') {
            let Some(want) = sequitur::Components::from_sequence_string(name) else {
                return p.to_string();
            };
            let Ok(seqs) = sequitur::FileSequence::from_directory(dir, 1) else {
                return p.to_string();
            };
            let same = |s: &sequitur::FileSequence| {
                s.prefix().ok() == want.prefix.as_deref()
                    && s.delimiter().ok().flatten() == want.delimiter.as_deref()
                    && s.suffix().ok().flatten() == want.suffix.as_deref()
                    && s.extension().ok() == want.extension.as_deref()
            };
            let pick = seqs
                .iter()
                .filter(|s| same(s))
                .find(|s| Some(s.nominal_padding()) == want.padding)
                .or_else(|| seqs.iter().find(|s| same(s)));
            return match pick.and_then(|s| s.sequence_string().ok()) {
                Some(n) => with(n),
                None => p.to_string(),
            };
        }
        if !expanded.is_file() || engine::decode::is_movie(Path::new(&expanded)) {
            return p.to_string();
        }
        let Some(item) = sequitur::Item::from_path(&expanded) else {
            return p.to_string();
        };
        let file = item.filename();
        let Ok(seqs) = sequitur::FileSequence::from_directory(dir, 2) else {
            return p.to_string();
        };
        seqs.iter()
            .find(|s| s.items().iter().any(|i| i.filename() == file))
            .and_then(|s| s.sequence_string().ok())
            .map(with)
            .unwrap_or_else(|| p.to_string())
    }
}

/// Make the Loads `text` asks for, in a row centred on `at`, each with a
/// reader slot of its own. `None` (and nothing made) when the text names no
/// footage. Their footage arrives the way a pasted Read's does: the app
/// sees the paste and loads every Read with a path and nothing loaded.
pub(crate) fn paste_loads(
    graph: &mut crate::editor::EditorGraph,
    text: &str,
    at: egui::Pos2,
    next_read_slot: &mut usize,
) -> Option<Vec<u64>> {
    use engine::graph::NodeKind;
    const GAP: f32 = 150.0;
    let loads = parse(text)?;
    let x0 = at.x - GAP * (loads.len() - 1) as f32 / 2.0;
    let mut ids = Vec::new();
    for (i, l) in loads.into_iter().enumerate() {
        let mut kind = NodeKind::read(*next_read_slot, l.path);
        *next_read_slot += 1;
        if let NodeKind::Read { start, first, last, .. } = &mut kind {
            (*start, *first, *last) = (l.start, l.first, l.last);
        }
        let tv = crate::editor::new_read_tv(&kind);
        let id = graph.add(kind, egui::pos2(x0 + GAP * i as f32, at.y));
        if let Some(n) = graph.nodes.get_mut(&id) {
            n.tv = tv;
        }
        ids.push(id);
    }
    Some(ids)
}

// ---- script blocks -------------------------------------------------------

#[derive(Debug, PartialEq)]
pub(crate) enum Tok {
    Word(String),
    /// A `{…}` group, its inside verbatim.
    Group(String),
    Newline,
}

/// TCL-ish words: bare words, "quoted" (backslash escapes), {braced}
/// groups (nesting, kept verbatim) and newlines, which end a knob.
pub(crate) fn tokens(s: &str) -> Vec<Tok> {
    let c: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < c.len() {
        match c[i] {
            '\n' => {
                out.push(Tok::Newline);
                i += 1;
            }
            ch if ch.is_whitespace() => i += 1,
            '{' => {
                let mut depth = 1;
                let mut j = i + 1;
                while j < c.len() && depth > 0 {
                    match c[j] {
                        '\\' => j += 1,
                        '{' => depth += 1,
                        '}' => depth -= 1,
                        _ => {}
                    }
                    j += 1;
                }
                let end = if depth == 0 { j - 1 } else { j };
                out.push(Tok::Group(c[i + 1..end.min(c.len())].iter().collect()));
                i = j;
            }
            '"' => {
                let mut w = String::new();
                let mut j = i + 1;
                while j < c.len() && c[j] != '"' {
                    if c[j] == '\\' && j + 1 < c.len() {
                        j += 1;
                    }
                    w.push(c[j]);
                    j += 1;
                }
                out.push(Tok::Word(w));
                i = j + 1;
            }
            _ => {
                let mut w = String::new();
                while i < c.len() && !c[i].is_whitespace() && c[i] != '{' {
                    w.push(c[i]);
                    i += 1;
                }
                out.push(Tok::Word(w));
            }
        }
    }
    out
}

/// The top-level `Class { body }` blocks of a script, or `None` when the
/// text has none (so it is not a script at all).
pub(crate) fn script_blocks(text: &str) -> Option<Vec<(String, String)>> {
    let t = tokens(text);
    let mut out = Vec::new();
    for w in t.windows(2) {
        if let [Tok::Word(class), Tok::Group(body)] = w {
            if class.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
                && class.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                out.push((class.clone(), body.clone()));
            }
        }
    }
    (!out.is_empty()).then_some(out)
}

/// A Read block's knobs → one Load. The script numbers its range in FILE
/// frames (`first` / `last`) and says where they play with `frame_mode`
/// (`start at` N, or `offset` N); a Read's `start` is the comp frame its
/// FIRST FILE lands on (`origfirst`, or `first` when there is none).
fn read_block(body: &str, resolve: &dyn Fn(&str) -> String) -> Option<PastedLoad> {
    let mut knobs = std::collections::HashMap::new();
    let mut line: Vec<String> = Vec::new();
    let flush = |line: &mut Vec<String>, knobs: &mut std::collections::HashMap<String, String>| {
        if line.len() >= 2 {
            knobs.entry(line[0].clone()).or_insert_with(|| line[1].clone());
        }
        line.clear();
    };
    for t in tokens(body) {
        match t {
            Tok::Newline => flush(&mut line, &mut knobs),
            Tok::Word(w) | Tok::Group(w) => line.push(w),
        }
    }
    flush(&mut line, &mut knobs);
    let file = knobs.get("file")?.trim();
    // An expression (`[value root.something]`) needs an interpreter we
    // do not have.
    if file.is_empty() || file.contains('[') {
        return None;
    }
    let path = resolve(&normalise(file));
    let num = |k: &str| {
        knobs
            .get(k)
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite())
            .map(|v| v.round() as i64)
    };
    let (Some(first), Some(last)) = (num("first"), num("last")) else {
        return Some(PastedLoad::path(path));
    };
    let last = last.max(first);
    let mode = knobs.get("frame_mode").map(|s| s.trim().to_ascii_lowercase());
    let shift = match (mode.as_deref(), num("frame")) {
        (Some("start at"), Some(f)) => f - first,
        (Some("offset"), Some(f)) => f,
        _ => 0,
    };
    let file0 = num("origfirst").unwrap_or(first).min(first);
    Some(PastedLoad {
        path,
        start: Some(file0 + shift),
        first: Some(first + shift),
        last: Some(last + shift),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn as_is(p: &str) -> String {
        p.to_string()
    }
    fn paths(text: &str) -> Option<Vec<String>> {
        parse_with(text, &as_is).map(|v| v.into_iter().map(|l| l.path).collect())
    }

    #[test]
    fn frame_tokens_become_hashes() {
        for (inp, want) in [
            ("/shots/a/bg.####.exr", "/shots/a/bg.####.exr"),
            ("/shots/a/bg.%04d.exr", "/shots/a/bg.####.exr"),
            ("/shots/a/bg.%d.exr", "/shots/a/bg.#.exr"),
            ("/shots/a/bg.@@@@.exr", "/shots/a/bg.####.exr"),
            ("/shots/a/bg.$F4.exr", "/shots/a/bg.####.exr"),
            ("/shots/a/bg.${F3}.exr", "/shots/a/bg.###.exr"),
            ("/shots/a/bg.$F.exr", "/shots/a/bg.#.exr"),
            ("/shots/a/bg_$Frame.exr", "/shots/a/bg_$Frame.exr"),
            // only the file name: a directory with @ in it is left alone
            ("/home/me@host/bg.%04d.exr", "/home/me@host/bg.####.exr"),
        ] {
            assert_eq!(paths(inp), Some(vec![want.to_string()]), "{inp}");
        }
    }

    #[test]
    fn windows_paths_quotes_and_urls() {
        assert_eq!(paths("P:/proj/sh010/bg.####.dpx"), Some(vec!["P:/proj/sh010/bg.####.dpx".into()]));
        assert_eq!(paths(r"P:\proj\sh010\bg.%04d.dpx"), Some(vec!["P:/proj/sh010/bg.####.dpx".into()]));
        assert_eq!(paths(r#""C:\My Plates\bg.1001.exr""#), Some(vec!["C:/My Plates/bg.1001.exr".into()]));
        assert_eq!(paths(r"\\server\share\bg.####.exr"), Some(vec![r"\\server\share\bg.####.exr".into()]));
        assert_eq!(paths("'/a b/c.exr'"), Some(vec!["/a b/c.exr".into()]));
        assert_eq!(paths("file:///Users/me/My%20Plates/c.exr"), Some(vec!["/Users/me/My Plates/c.exr".into()]));
        assert_eq!(paths("file:///D:/plates/c.exr"), Some(vec!["D:/plates/c.exr".into()]));
    }

    #[test]
    fn several_lines_make_several_loads() {
        assert_eq!(
            paths("/a/bg.####.exr\r\n\n  /a/fg.####.exr  \n"),
            Some(vec!["/a/bg.####.exr".into(), "/a/fg.####.exr".into()])
        );
    }

    #[test]
    fn prose_is_not_a_path() {
        assert_eq!(paths("hello there"), None);
        assert_eq!(paths("/a/bg.exr\nand some words"), None, "one stray line rejects the paste");
        assert_eq!(paths("bg.####.exr"), None, "a bare relative name is not taken");
        assert_eq!(paths(""), None);
        assert_eq!(paths(r#"{"wobble_nodes":1}"#), None);
    }

    const SCRIPT: &str = r#"set cut_paste_input [stack 0]
version 15.0 v1
Read {
 inputs 0
 file_type dpx
 file Q:/jobs/demo/sh020/plate/bg_v003.####.dpx
 first 1001
 last 1100
 frame_mode "start at"
 frame 1001.0
 origfirst 1001
 origlast 1100
 origset true
 name Read3
 selected true
 xpos 10
 ypos 20
}
Grade {
 white 1.2
 name Grade1
}
Read {
 inputs 0
 file "/mnt/jobs/demo/sh020/fg/fg.%04d.exr"
 first 1
 last 50
 frame_mode offset
 frame 1000
 name Read4
}
Read {
 inputs 0
 file {Q:/jobs/demo/sh020/ref/still with space.exr}
 name Read5
}
"#;

    #[test]
    fn script_reads_come_across_with_their_ranges() {
        let got = parse_with(SCRIPT, &as_is).expect("a script with Reads");
        assert_eq!(
            got,
            vec![
                PastedLoad {
                    path: "Q:/jobs/demo/sh020/plate/bg_v003.####.dpx".into(),
                    start: Some(1001),
                    first: Some(1001),
                    last: Some(1100),
                },
                PastedLoad {
                    path: "/mnt/jobs/demo/sh020/fg/fg.####.exr".into(),
                    start: Some(1001),
                    first: Some(1001),
                    last: Some(1050),
                },
                PastedLoad::path("Q:/jobs/demo/sh020/ref/still with space.exr".into()),
            ]
        );
    }

    #[test]
    fn start_at_moves_the_range() {
        let s = "Read {\n file /a/b.####.exr\n first 1001\n last 1010\n frame_mode \"start at\"\n frame 1\n}\n";
        let got = parse_with(s, &as_is).unwrap();
        assert_eq!((got[0].start, got[0].first, got[0].last), (Some(1), Some(1), Some(10)));
        // a trimmed range: the files still start at origfirst
        let s = "Read {\n file /a/b.####.exr\n first 1005\n last 1010\n origfirst 1001\n}\n";
        let got = parse_with(s, &as_is).unwrap();
        assert_eq!((got[0].start, got[0].first, got[0].last), (Some(1001), Some(1005), Some(1010)));
    }

    #[test]
    fn a_script_without_reads_pastes_nothing() {
        assert_eq!(parse_with("Grade {\n white 2\n}\nBlur {\n size 3\n}\n", &as_is), None);
        assert_eq!(parse_with("Read {\n file [value root.name]\n}\n", &as_is), None);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn the_disk_settles_padding_and_single_frames() {
        let dir = std::env::temp_dir().join(format!("wobble_pathpaste_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for f in 1001..=1004 {
            std::fs::write(dir.join(format!("bg.{f}.exr")), b"").unwrap();
        }
        std::fs::write(dir.join("lonely_v002.exr"), b"").unwrap();
        let d = dir.to_string_lossy().replace('\\', "/");
        let one = |s: String| parse(&s).unwrap().remove(0).path;
        assert_eq!(one(format!("{d}/bg.1002.exr")), format!("{d}/bg.####.exr"), "a frame → its sequence");
        assert_eq!(one(format!("{d}/bg.$F.exr")), format!("{d}/bg.####.exr"), "padding from the files");
        assert_eq!(one(format!("{d}/bg.%04d.exr")), format!("{d}/bg.####.exr"));
        assert_eq!(one(format!("{d}/lonely_v002.exr")), format!("{d}/lonely_v002.exr"), "a still stays a still");
        assert_eq!(one(format!("{d}/gone.####.exr")), format!("{d}/gone.####.exr"), "missing is kept as pasted");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
