//! CHANNEL EXPRESSIONS typed straight into a parameter field, Houdini-style.
//!
//! Type `../grade12/gainr * 2` into any value field and that channel is
//! driven by the expression. It is not a second driver system: the text
//! lands in an `expr` value node wired to the channel — exactly what the
//! patch bay would build by hand — so evaluation, the driven rows, undo,
//! saving, tools and the canvas all see an ordinary patch link. The patch
//! bay is the expression VIEW; this is the expression TEXT, and each edits
//! the same node.
//!
//! Names in a path:
//! - a NODE is its hand-given label when it has one (`plate_v001`), else
//!   its kind and id (`grade12`) — stable across deletes, unique by
//!   construction. The bare kind (`grade`) also answers while there is only
//!   one of it.
//! - a PARAM is its label with the spaces squeezed out (`gain r` → `gainr`)
//!   or its channel key (`GainR`), either case.
//!
//! The rows (`crate::widgets`) never see the graph. They queue edits and
//! draw the editor; [`frame`], once per UI pass with the graph in hand,
//! applies the edits and publishes what the rows and the editor show —
//! which rows are expressions, the editor's completions and its live value.
//!
//! The editor is MULTI-LINE (`crate::exprs`: `name = …` lines, the last line
//! is the value). Enter sets, shift+Enter starts a line, and Tab completes
//! the address under the cursor: first to what every candidate shares, then
//! STEPPING through the candidates, each one written into the text as it
//! is reached (shift+Tab steps back). Enter takes the one you stopped on —
//! and sets the expression when that makes it whole. Only addresses that
//! exist are ever offered, and the ones in the text are coloured by whether
//! they resolve.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::anim::Channel;
use crate::editor::{EditorGraph, ModKind, PDst, PSrc};

// ---- Names ---------------------------------------------------------------

/// `Read 1` → `read_1`: lowercase, runs of anything else to one `_`.
fn slug(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('_') && !out.is_empty() {
            out.push('_');
        }
    }
    out.trim_end_matches('_').to_string()
}

/// `gain r` → `gainr`: a param name is its label, alphanumerics only.
fn squash(s: &str) -> String {
    s.chars().filter(char::is_ascii_alphanumeric).map(|c| c.to_ascii_lowercase()).collect()
}

/// `grade12`: kind + id. A label's own trailing number goes (`Read 1` is
/// `read`, `Layer ×3` is `layer`) so the id is the only number; an
/// underscore keeps a kind that still ends in a digit from running into it.
fn auto_name(kind: &engine::graph::NodeKind, id: u64) -> String {
    let base = slug(&crate::editor::node_label(kind));
    let base = match base.rsplit_once('_') {
        Some((stem, n)) if !stem.is_empty() && n.chars().all(|c| c.is_ascii_digit()) => stem.to_string(),
        _ => base,
    };
    let base = if base.is_empty() { "node".to_string() } else { base };
    if base.ends_with(|c: char| c.is_ascii_digit()) {
        format!("{base}_{id}")
    } else {
        format!("{base}{id}")
    }
}

/// The name a path uses for `id`: its label when renamed (and the first
/// node to hold that label), else kind + id.
pub fn node_name(g: &EditorGraph, id: u64) -> Option<String> {
    let n = g.node(id)?;
    if !n.ui_label.is_empty() {
        let s = slug(&n.ui_label);
        let first = g
            .nodes
            .iter()
            .find(|(_, o)| !o.ui_label.is_empty() && slug(&o.ui_label) == s)
            .map(|(i, _)| *i);
        if !s.is_empty() && first == Some(id) {
            return Some(s);
        }
    }
    Some(auto_name(&n.kind, id))
}

/// The node a path segment names, by the rules in the module doc.
/// Compared squeezed, so `grade12`, `grade_12` and `Grade 12` are one name.
pub fn find_node(g: &EditorGraph, name: &str) -> Option<u64> {
    let want = squash(name);
    if let Some((id, _)) = g
        .nodes
        .iter()
        .find(|(_, n)| !n.ui_label.is_empty() && squash(&n.ui_label) == want)
    {
        return Some(*id);
    }
    if let Some((id, _)) = g.nodes.iter().find(|(id, n)| squash(&auto_name(&n.kind, **id)) == want) {
        return Some(*id);
    }
    let mut kind = g
        .nodes
        .iter()
        .filter(|(_, n)| squash(&crate::editor::node_label(&n.kind)) == want);
    match (kind.next(), kind.next()) {
        (Some((id, _)), None) => Some(*id),
        _ => None,
    }
}

/// The name a path uses for one of a node's params.
pub fn param_name(kind: &engine::graph::NodeKind, ch: Channel) -> String {
    squash(&crate::curves::ch_label_for(kind, ch))
}

/// The channel a param name names on `kind`, by label or by key.
pub fn find_param(kind: &engine::graph::NodeKind, name: &str) -> Option<Channel> {
    let want = squash(name);
    crate::editor::node_channels(kind)
        .into_iter()
        .find(|c| squash(&c.key()) == want || param_name(kind, *c) == want)
}

/// A path to the channel it names. `here` is the node the expression
/// drives — what `./param` and a bare `ch("param")` read.
pub fn resolve(g: &EditorGraph, path: &str, here: Option<u64>) -> Result<(u64, Channel), String> {
    let (node, param) = if let Some(rest) = path.strip_prefix("../") {
        let (n, p) = rest
            .split_once('/')
            .ok_or_else(|| format!("{path}: a path is ../node/param"))?;
        (find_node(g, n).ok_or_else(|| format!("no node called {n}"))?, p)
    } else {
        let p = path.strip_prefix("./").unwrap_or(path);
        if p.contains('/') {
            return Err(format!("{path}: a path is ../node/param"));
        }
        (here.ok_or("./ reads the node the expression drives")?, p)
    };
    let n = g.node(node).ok_or("that node is gone")?;
    let ch = find_param(&n.kind, param).ok_or_else(|| {
        format!("{} has no param {param}", node_name(g, node).unwrap_or_default())
    })?;
    Ok((node, ch))
}

/// The path from `here` to `node`'s `ch`: `./param` on the same node.
pub fn path_to(g: &EditorGraph, node: u64, ch: Channel, here: Option<u64>) -> Option<String> {
    let p = param_name(&g.node(node)?.kind, ch);
    Some(if here == Some(node) {
        format!("./{p}")
    } else {
        format!("../{}/{p}", node_name(g, node)?)
    })
}

/// The node the expression value node `m` drives — its `./`. The first
/// channel it feeds, when it feeds several.
pub fn here_of(g: &EditorGraph, m: u64) -> Option<u64> {
    g.patch.links.iter().find_map(|l| match (l.from, l.to) {
        (PSrc::Mod(x), PDst::Chan(n, _)) if x == m => Some(n),
        _ => None,
    })
}

/// The expression value node driving `node`'s `ch`, if an expression does.
pub fn driver(g: &EditorGraph, node: u64, ch: Channel) -> Option<u64> {
    let l = g.patch.link_to(PDst::Chan(node, ch))?;
    let PSrc::Mod(m) = l.from else { return None };
    g.patch
        .mods
        .iter()
        .find(|x| x.id == m && matches!(x.kind, ModKind::Expr(_)))
        .map(|x| x.id)
}

// ---- Edits ---------------------------------------------------------------

/// Drive `chans` on `node` from `text`. Empty text takes the expression off
/// (and its value node, once nothing else reads it). Text that reads
/// nothing live — `10/2`, `2*pi` — is a number, written as one. Otherwise
/// the channel's existing expression node takes the text, or a new one is
/// made and wired, replacing whatever drove the channel before.
pub fn set_expr(g: &mut EditorGraph, node: u64, chans: &[Channel], text: &str, t: u64) {
    let text = text.trim();
    let existing = chans.iter().find_map(|c| driver(g, node, *c));
    let number = crate::exprs::eval(text, 0.0, 0.0, t as f32).filter(|v| v.is_finite());
    if text.is_empty() || (!crate::exprs::is_live(text) && number.is_some()) {
        for c in chans {
            let Some(m) = driver(g, node, *c) else { continue };
            g.patch.unlink(PDst::Chan(node, *c));
            if !g.patch.links.iter().any(|l| l.from == PSrc::Mod(m)) {
                g.patch.remove_mod(m);
            }
        }
        if let Some(v) = number {
            write_value(g, node, chans, v, t);
        }
        return;
    }
    let m = match existing {
        Some(m) => {
            if let Some(x) = g.patch.mods.iter_mut().find(|x| x.id == m) {
                x.kind = ModKind::Expr(text.to_string());
            }
            m
        }
        None => {
            let pos = place(g);
            g.patch.add_mod(ModKind::Expr(text.to_string()), pos)
        }
    };
    for c in chans {
        g.patch.link(PSrc::Mod(m), PDst::Chan(node, *c));
    }
}

/// A constant typed into the editor lands like a typed number: a key at the
/// frame on screen when the channel is animated (unless locked), the
/// authored value otherwise.
fn write_value(g: &mut EditorGraph, node: u64, chans: &[Channel], v: f32, t: u64) {
    let Some(n) = g.node_mut(node) else { return };
    for c in chans {
        match n.anim.get_mut(c) {
            Some(k) if !k.keys.is_empty() => {
                if !k.locked {
                    k.set_key(t, v);
                }
            }
            _ => crate::anim::set(&mut n.kind, *c, v),
        }
    }
}

/// Where a new expression node goes on the patch canvas: a column LEFT of
/// the origin (it feeds nodes, and the canvas flows left to right), under
/// any expression nodes already there.
fn place(g: &EditorGraph) -> [f32; 2] {
    let x = -(crate::patchbay::MOD_W + 48.0);
    let y = g
        .patch
        .mods
        .iter()
        .filter(|m| m.pos[0] < 0.0)
        .map(|m| m.pos[1] + crate::patchbay::MOD_H + 16.0)
        .fold(0.0f32, f32::max);
    [x, y]
}

// ---- Shared state between the rows and `frame` ----------------------------

enum Edit {
    Set { node: u64, chans: Vec<Channel>, text: String },
    CopyPath { node: u64, ch: Channel },
}

/// The floating editor, while one is open.
struct Open {
    node: u64,
    chans: Vec<Channel>,
    text: String,
    /// Drawn this pass by its row. A row that stops drawing (panel closed,
    /// another node selected) takes its editor with it.
    drawn: bool,
    /// Focus + cursor-to-end on the first draw.
    fresh: bool,
    /// The highlighted completion.
    pick: usize,
    /// Where the editor was last drawn, so a click that lands on it does
    /// not count as clicking away.
    rect: egui::Rect,
    /// The text cursor, as a byte offset: completions are for the address
    /// it sits in, which on line two of five is not the end of the text.
    cursor: usize,
    /// Tab is stepping through candidates (see [`Cycle`]).
    cycle: Option<Cycle>,
    /// The highlight was moved by hand (arrows), so Enter means "that one".
    navigated: bool,
    // Published by `frame` for the text and cursor as they last stood:
    completions: Vec<Completion>,
    result: Result<f32, String>,
    for_text: String,
    for_cursor: usize,
    /// The nodes the text names — ringed in the graph.
    named: Vec<u64>,
    /// Every address in the text, and whether it resolves — its colour.
    valid: Vec<(String, bool)>,
}

/// Tab stepping through candidates: the list as it stood when the stepping
/// began (each step rewrites the address, which would otherwise change what
/// is on offer), where it has got to, and the text and cursor that step
/// left — anything else there means the hand has moved on, and it is over.
#[derive(Clone)]
struct Cycle {
    cands: Vec<Completion>,
    idx: usize,
    text: String,
    cursor: usize,
}

/// A completion for the address at the cursor: what replaces the token
/// (`start` up to the cursor), a hint, and the node it names (ringed in the
/// graph while it is the pick).
#[derive(Clone, Debug)]
struct Completion {
    insert: String,
    start: usize,
    hint: String,
    node: Option<u64>,
}

#[derive(Default)]
struct State {
    /// (node, channel) → the text of the expression driving it.
    rows: HashMap<(u64, Channel), String>,
    edits: Vec<Edit>,
    open: Option<Open>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    STATE.with(|s| f(&mut s.borrow_mut()))
}

/// The expression on the first of `chans` that has one.
pub fn row_text(node: u64, chans: &[Channel]) -> Option<String> {
    with(|s| chans.iter().find_map(|c| s.rows.get(&(node, *c)).cloned()))
}

/// While an expression is being typed, the node graph NAMES its nodes the
/// way a path spells them (`grade12`), so the one you want can be read off
/// the graph — node labels there are not unique ("Grade", "Grade").
pub fn naming() -> bool {
    with(|s| s.open.is_some())
}

/// Ring `id` in the graph: the text names it, or it is the pick.
pub fn lit(id: u64) -> bool {
    with(|s| {
        s.open.as_ref().is_some_and(|o| {
            let pick = match &o.cycle {
                Some(c) => c.cands.get(c.idx),
                None => o.completions.get(o.pick),
            };
            o.named.contains(&id) || pick.is_some_and(|c| c.node == Some(id))
        })
    })
}

/// Open the editor on `chans`, seeded with `text`.
pub fn open(ctx: &egui::Context, node: u64, chans: &[Channel], text: &str) {
    with(|s| {
        s.open = Some(Open {
            node,
            chans: chans.to_vec(),
            text: text.to_string(),
            drawn: true,
            fresh: true,
            pick: 0,
            rect: egui::Rect::NOTHING,
            cursor: text.len(),
            cycle: None,
            navigated: false,
            completions: Vec::new(),
            result: Err(String::new()),
            for_text: String::from("\u{0}"),
            for_cursor: usize::MAX,
            named: Vec::new(),
            valid: Vec::new(),
        })
    });
    ctx.request_repaint();
}

// ---- Once per UI pass ------------------------------------------------------

/// Apply the edits the rows queued, then publish what they draw next.
pub fn frame(ctx: &egui::Context, g: &mut EditorGraph, t: u64) {
    let edits = with(|s| std::mem::take(&mut s.edits));
    for e in edits {
        match e {
            Edit::Set { node, chans, text } => set_expr(g, node, &chans, &text, t),
            Edit::CopyPath { node, ch } => {
                if let Some(p) = path_to(g, node, ch, None) {
                    ctx.copy_text(p);
                }
            }
        }
    }
    let rows: HashMap<(u64, Channel), String> = g
        .patch
        .links
        .iter()
        .filter_map(|l| {
            let (PSrc::Mod(m), PDst::Chan(n, c)) = (l.from, l.to) else { return None };
            match &g.patch.mods.iter().find(|x| x.id == m)?.kind {
                ModKind::Expr(x) => Some(((n, c), x.clone())),
                _ => None,
            }
        })
        .collect();
    // What the open editor needs, for its text and cursor as they stand.
    // Only when one moved: the completions walk the graph.
    let want = with(|s| {
        s.rows = rows;
        let o = s.open.as_mut()?;
        if !std::mem::replace(&mut o.drawn, false) {
            s.open = None;
            return None;
        }
        (o.for_text != o.text || o.for_cursor != o.cursor)
            .then(|| (o.node, o.chans.clone(), o.text.clone(), o.cursor))
    });
    let Some((node, chans, text, cursor)) = want else { return };
    let completions = complete(g, &text, cursor, node, t);
    let result = preview(g, &text, node, &chans, t);
    let reads: Vec<(String, Option<u64>)> = crate::exprs::refs(&text)
        .into_iter()
        .map(|p| {
            // The channel it drives is an address that exists and still
            // cannot be read from here.
            let n = resolve(g, &p, Some(node))
                .ok()
                .filter(|(n, c)| !(*n == node && chans.contains(c)))
                .map(|(n, _)| n);
            (p, n)
        })
        .collect();
    // The address still being typed, with completions on offer, is not
    // wrong yet: it gets no verdict, so it is drawn plain rather than red.
    let typing = completions
        .first()
        .and_then(|c| text.get(c.start..floor_boundary(&text, cursor)))
        .map(str::to_string);
    with(|s| {
        if let Some(o) = s.open.as_mut() {
            // A new list starts on its first candidate.
            o.pick = 0;
            o.navigated = false;
            o.completions = completions;
            o.result = result;
            o.for_text = text;
            o.for_cursor = cursor;
            o.named = reads.iter().filter_map(|(_, n)| *n).collect();
            o.valid = reads
                .into_iter()
                .filter(|(p, n)| n.is_some() || typing.as_deref() != Some(p.as_str()))
                .map(|(p, n)| (p, n.is_some()))
                .collect();
        }
    });
    ctx.request_repaint();
}

/// What the expression reads right now, or why it can't.
fn preview(g: &EditorGraph, text: &str, node: u64, chans: &[Channel], t: u64) -> Result<f32, String> {
    if text.trim().is_empty() {
        return Err("empty: Enter takes the expression off".into());
    }
    let base: std::collections::BTreeMap<u64, engine::graph::NodeKind> =
        g.nodes.iter().map(|(id, n)| (*id, n.kind_at(t))).collect();
    // The jacks, if the channel's expression node has any wired.
    let m = chans.iter().find_map(|c| driver(g, node, *c));
    let jack = |slot: u8| {
        m.and_then(|m| g.patch.link_to(PDst::Mod(m, slot)))
            .map_or(0.0, |l| g.patch_value(l.from, &base, t as f32, &mut Vec::new()))
    };
    let mut err = None;
    let v = crate::exprs::eval_report(text, jack(0), jack(1), t as f32, &mut |p| {
        match resolve(g, p, Some(node)) {
            Ok((n, c)) if n == node && chans.contains(&c) => {
                err.get_or_insert_with(|| format!("{p} is the channel it drives"));
                None
            }
            Ok((n, c)) => Some(g.patch_value(PSrc::Chan(n, c), &base, t as f32, &mut Vec::new())),
            Err(e) => {
                err.get_or_insert(e);
                None
            }
        }
    });
    // On more than one line, say which.
    let at = |line: usize, e: String| {
        if text.trim().contains('\n') { format!("line {line}: {e}") } else { e }
    };
    match (v, err) {
        (Ok(v), None) => Ok(v),
        (Err(line), Some(e)) => Err(at(line, e)),
        (Ok(_), Some(e)) => Err(e),
        (Err(line), None) => Err(at(line, "can't read that".into())),
    }
}

/// A character an address is made of.
fn path_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '/')
}

/// `at` pulled back onto a character boundary of `text`.
fn floor_boundary(text: &str, at: usize) -> usize {
    let mut at = at.min(text.len());
    while !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}

/// Completions for the address being typed AT THE CURSOR. Every one is an
/// address that exists: nodes in this graph, then that node's params.
fn complete(g: &EditorGraph, text: &str, cursor: usize, here: u64, t: u64) -> Vec<Completion> {
    let before = &text[..floor_boundary(text, cursor)];
    // Nothing to complete inside a comment.
    let line = before.rfind('\n').map_or(0, |i| i + 1);
    if before[line..].contains('#') {
        return Vec::new();
    }
    let start = before
        .char_indices()
        .rev()
        .find(|(_, c)| !path_char(*c))
        .map_or(0, |(i, c)| i + c.len_utf8());
    let tok = &before[start..];
    let mut out = Vec::new();
    let params = |id: u64, prefix: &str, path: &dyn Fn(&str) -> String, out: &mut Vec<Completion>| {
        let Some(n) = g.node(id) else { return };
        let kind = n.kind_at(t);
        let want = squash(prefix);
        for c in crate::editor::node_channels(&kind) {
            let p = param_name(&kind, c);
            if p.starts_with(&want) {
                let v = crate::anim::get(&kind, c).unwrap_or(0.0);
                out.push(Completion {
                    insert: path(&p),
                    start,
                    hint: format!("{v:.3}"),
                    node: Some(id),
                });
            }
        }
    };
    if let Some(rest) = tok.strip_prefix("../") {
        match rest.split_once('/') {
            Some((name, p)) => {
                if let Some(id) = find_node(g, name) {
                    params(id, p, &|p| format!("../{name}/{p}"), &mut out);
                }
            }
            None => {
                let want = rest.to_ascii_lowercase();
                for (id, n) in &g.nodes {
                    if matches!(n.kind, engine::graph::NodeKind::Dot { .. }) {
                        continue;
                    }
                    let Some(name) = node_name(g, *id) else { continue };
                    if name.starts_with(&want) {
                        out.push(Completion {
                            insert: format!("../{name}/"),
                            start,
                            hint: crate::editor::display_label(n),
                            node: Some(*id),
                        });
                    }
                }
            }
        }
    } else if let Some(p) = tok.strip_prefix("./") {
        params(here, p, &|p| format!("./{p}"), &mut out);
    } else if tok == "." || tok == ".." {
        out.push(Completion { insert: "../".into(), start, hint: "another node".into(), node: None });
        out.push(Completion { insert: "./".into(), start, hint: "this node".into(), node: Some(here) });
    }
    out
}

/// Put `c` where the token at the cursor is.
fn accept(text: &mut String, cursor: &mut usize, start: usize, insert: &str) {
    let end = floor_boundary(text, *cursor);
    if start > end {
        return;
    }
    text.replace_range(start..end, insert);
    *cursor = start + insert.len();
}

/// Tab, as a shell's menu-complete does it. First extend the address to
/// what every candidate shares. When that adds nothing, STEP: write the
/// highlighted candidate into the text, and on each further Tab the next
/// one (`back`: the one before), round and round.
fn tab_step(
    text: &mut String,
    cursor: &mut usize,
    cycle: &mut Option<Cycle>,
    comps: &[Completion],
    pick: usize,
    back: bool,
) -> bool {
    if let Some(c) = cycle {
        let n = c.cands.len();
        c.idx = if back { (c.idx + n - 1) % n } else { (c.idx + 1) % n };
        accept(text, cursor, c.cands[c.idx].start, &c.cands[c.idx].insert);
        c.text = text.clone();
        c.cursor = *cursor;
        return true;
    }
    let Some(first) = comps.first() else { return false };
    let typed = cursor.saturating_sub(first.start);
    let mut shared = first.insert.as_str();
    for c in comps {
        let n = shared.bytes().zip(c.insert.bytes()).take_while(|(a, b)| a == b).count();
        shared = &shared[..n];
    }
    if shared.len() > typed {
        accept(text, cursor, first.start, shared);
        return true;
    }
    let idx = if back { comps.len() - 1 } else { pick.min(comps.len() - 1) };
    accept(text, cursor, comps[idx].start, &comps[idx].insert);
    // One candidate is simply taken: nothing to step through, and what it
    // leads to (a node's params) should be on offer straight away.
    if comps.len() > 1 {
        *cycle = Some(Cycle { cands: comps.to_vec(), idx, text: text.clone(), cursor: *cursor });
    }
    true
}

/// What Enter means, given where the editor stands.
#[derive(Debug, PartialEq)]
enum Enter {
    /// Set the expression (or, empty, take it off).
    Set,
    /// Write this candidate into the text and keep editing.
    Take(usize),
    /// The stepping stops on what it wrote; keep editing.
    Settle,
    /// It cannot be read and nothing is on offer: stay, on the error.
    Stay,
}

/// Enter SETS whenever the text can be read as it stands — including when
/// Tab stepping has just written the address that makes it whole. Otherwise
/// it finishes the address: the candidate stepped to, or the highlighted
/// one. A highlight moved by hand is always what Enter takes.
fn enter_means(
    empty: bool,
    cycling: bool,
    navigated: bool,
    readable: bool,
    judged: bool,
    offered: usize,
    pick: usize,
) -> Enter {
    if empty {
        Enter::Set
    } else if cycling {
        if readable { Enter::Set } else { Enter::Settle }
    } else if navigated && pick < offered {
        Enter::Take(pick)
    } else if readable || !judged {
        Enter::Set
    } else if pick < offered {
        Enter::Take(pick)
    } else {
        Enter::Stay
    }
}

/// The editor's text, coloured: an address that resolves in the expression
/// colour, one that does not in the error colour, comments dimmed. An
/// address `frame` has not judged yet (just typed) stays plain.
fn highlight(
    text: &str,
    valid: &[(String, bool)],
    font: egui::FontId,
    [plain, ok, bad, dim]: [egui::Color32; 4],
) -> egui::text::LayoutJob {
    let mut marks: Vec<(std::ops::Range<usize>, egui::Color32)> = crate::exprs::ref_spans(text)
        .into_iter()
        .filter_map(|(r, p)| {
            let good = valid.iter().find(|(v, _)| *v == p)?.1;
            Some((r, if good { ok } else { bad }))
        })
        .collect();
    let mut at = 0;
    for line in text.split_inclusive('\n') {
        if let Some(h) = line.find('#') {
            marks.push((at + h..at + line.trim_end_matches('\n').len(), dim));
        }
        at += line.len();
    }
    marks.sort_by_key(|(r, _)| r.start);
    let mut job = egui::text::LayoutJob::default();
    let put = |job: &mut egui::text::LayoutJob, s: &str, color| {
        if !s.is_empty() {
            job.append(s, 0.0, egui::TextFormat { font_id: font.clone(), color, ..Default::default() });
        }
    };
    let mut done = 0;
    for (r, color) in marks {
        if r.start < done || r.end > text.len() {
            continue;
        }
        put(&mut job, &text[done..r.start], plain);
        put(&mut job, &text[r.clone()], color);
        done = r.end;
    }
    put(&mut job, &text[done..], plain);
    job
}

/// A multi-line expression on one line, for a row: `g = …; g * 2`.
fn one_line(text: &str) -> String {
    text.lines().map(str::trim).filter(|l| !l.is_empty()).collect::<Vec<_>>().join("; ")
}

// ---- Row hooks (called from `crate::widgets`) ------------------------------

/// Does this half-typed field text want the expression editor? A path, a
/// `$` variable or a word does; a number (`1e3` included) does not.
fn wants_editor(text: &str) -> bool {
    let t = text.trim();
    if t.contains("./") || t.contains('$') {
        return true;
    }
    t.chars().any(|c| c.is_ascii_alphabetic())
        && !t.chars().all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-'))
}

/// After a number field on an UNDRIVEN channel. Typing a path, a variable
/// or a function into it hands the typing to the expression editor (wider,
/// with completions). Constant arithmetic (`10/2`) committed in the field
/// comes back as the number to set.
pub fn field_typing(ui: &egui::Ui, r: &egui::Response, node: u64, chans: &[Channel]) -> Option<f32> {
    let stash = r.id.with("chanexpr-typed");
    let live = crate::numfield::typed_text(ui.ctx(), r.id);
    if r.has_focus() {
        let text = live?;
        if wants_editor(&text) {
            ui.memory_mut(|m| m.surrender_focus(r.id));
            crate::numfield::forget_typed(ui.ctx(), r.id);
            ui.data_mut(|d| d.remove::<String>(stash));
            open(ui.ctx(), node, chans, &text);
        } else {
            ui.data_mut(|d| d.insert_temp(stash, text));
        }
        return None;
    }
    let text = ui.data_mut(|d| d.remove_temp::<String>(stash))?;
    if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
        return None;
    }
    let text = live.unwrap_or(text);
    // A plain number the field already took.
    if text.trim().parse::<f64>().is_ok() {
        return None;
    }
    crate::exprs::eval(&text, 0.0, 0.0, 0.0).filter(|v| v.is_finite())
}

/// The value field of an EXPRESSION-driven channel: its live value in the
/// expression colour. Click to edit the expression.
pub fn value_field(
    ui: &mut egui::Ui,
    w: f32,
    node: u64,
    chans: &[Channel],
    live: f32,
    text: &str,
) -> egui::Response {
    let s = crate::numfield::fit_number(live as f64, 0..=crate::widgets::decimals(), 7);
    let r = ui
        .add_sized(
            [w, crate::widgets::NUM_H],
            egui::Button::new(
                egui::RichText::new(s).color(crate::theme::patch_expr()),
            )
            .fill(ui.visuals().extreme_bg_color),
        )
        .on_hover_text(format!("= {text}\nclick to edit the expression"));
    if r.clicked() {
        open(ui.ctx(), node, chans, text);
    }
    r
}

/// The rail of an expression-driven slider row: the expression itself,
/// clipped to the rail. Click to edit.
pub fn rail(ui: &mut egui::Ui, w: f32, node: u64, chans: &[Channel], text: &str) -> egui::Response {
    let r = ui
        .add_sized(
            [w, crate::widgets::NUM_H],
            egui::Button::new(
                egui::RichText::new(one_line(text))
                    .monospace()
                    .color(crate::theme::patch_expr()),
            )
            .frame(false)
            .truncate(),
        )
        .on_hover_text(format!("= {text}\nclick to edit the expression"));
    if r.clicked() {
        open(ui.ctx(), node, chans, text);
    }
    r
}

/// The row label's right-click menu: edit or remove the expression, per
/// channel on a multi-channel row, and copy a channel's path to paste into
/// another field.
pub fn label_menu(r: &egui::Response, node: u64, chans: &[Channel], kind: &engine::graph::NodeKind) {
    if chans.is_empty() {
        return;
    }
    r.context_menu(|ui| {
        let ctx = ui.ctx().clone();
        let row = |ui: &mut egui::Ui, cs: &[Channel], what: &str| {
            let cur = row_text(node, cs);
            let label = match (&cur, what.is_empty()) {
                (Some(_), true) => "edit expression…".to_string(),
                (None, true) => "expression…".to_string(),
                (Some(_), false) => format!("edit expression: {what}…"),
                (None, false) => format!("expression: {what}…"),
            };
            if ui.button(label).clicked() {
                open(&ctx, node, cs, cur.as_deref().unwrap_or(""));
                ui.close();
            }
        };
        if chans.len() == 1 {
            row(ui, chans, "");
        } else {
            row(ui, chans, "all");
            for c in chans {
                row(ui, &[*c], &param_name(kind, *c));
            }
        }
        if chans.iter().any(|c| row_text(node, &[*c]).is_some())
            && ui.button("remove expression").clicked()
        {
            with(|s| {
                s.edits.push(Edit::Set { node, chans: chans.to_vec(), text: String::new() })
            });
            ui.close();
        }
        ui.separator();
        for c in chans {
            let what = if chans.len() == 1 { String::new() } else { format!(": {}", param_name(kind, *c)) };
            if ui
                .button(format!("copy path{what}"))
                .on_hover_text("paste it into another field: ../node/param")
                .clicked()
            {
                with(|s| s.edits.push(Edit::CopyPath { node, ch: *c }));
                ctx.request_repaint();
                ui.close();
            }
        }
    });
}

/// Byte offset of the `n`th character.
fn byte_of(text: &str, n: usize) -> usize {
    text.char_indices().nth(n).map_or(text.len(), |(i, _)| i)
}

/// The floating editor, when it is open on channels of this row. Called
/// with the row's rect as the row finishes drawing.
pub fn editor_ui(ui: &egui::Ui, node: u64, chans: &[Channel], row: egui::Rect) {
    let mine = with(|s| {
        s.open.as_ref().is_some_and(|o| o.node == node && o.chans.iter().all(|c| chans.contains(c)))
    });
    if !mine {
        return;
    }
    let ctx = ui.ctx().clone();
    let id = egui::Id::new(("chanexpr-editor", node));
    let edit_id = id.with("text");
    struct Snap {
        text: String,
        cursor: usize,
        fresh: bool,
        pick: usize,
        completions: Vec<Completion>,
        /// The completions answer the text and cursor as they stand. Until
        /// `frame` catches up they are a keystroke stale: fine to show,
        /// not to apply — their `start` points into the old text.
        current: bool,
        result: Result<f32, String>,
        /// `result` is the verdict on this text.
        judged: bool,
        valid: Vec<(String, bool)>,
        last_rect: egui::Rect,
        cycle: Option<Cycle>,
        navigated: bool,
    }
    let Some(Snap {
        mut text,
        mut cursor,
        fresh,
        mut pick,
        completions,
        current,
        result,
        judged,
        valid,
        last_rect,
        mut cycle,
        mut navigated,
    }) = with(|s| {
        let o = s.open.as_mut()?;
        o.drawn = true;
        // Stepping the hand has moved on from (typed, clicked elsewhere in
        // the text) is over: what it wrote stays.
        if o.cycle.as_ref().is_some_and(|c| c.text != o.text || c.cursor != o.cursor) {
            o.cycle = None;
        }
        Some(Snap {
            text: o.text.clone(),
            cursor: o.cursor,
            fresh: std::mem::replace(&mut o.fresh, false),
            pick: o.pick,
            completions: o.completions.clone(),
            current: o.for_text == o.text && o.for_cursor == o.cursor,
            result: o.result.clone(),
            judged: o.for_text == o.text,
            valid: o.valid.clone(),
            last_rect: o.rect,
            cycle: o.cycle.clone(),
            navigated: o.navigated,
        })
    }) else {
        return;
    };
    let w = row.width().max(300.0);
    let mut close = false;
    let mut commit = false;
    let mut entered = false;
    let put_cursor = |ctx: &egui::Context, text: &str, at: usize| {
        let mut st = egui::TextEdit::load_state(ctx, edit_id).unwrap_or_default();
        let c = egui::text::CCursor::new(text[..floor_boundary(text, at)].chars().count());
        st.cursor.set_char_range(Some(egui::text::CCursorRange::one(c)));
        st.store(ctx, edit_id);
    };
    let area = egui::Area::new(id)
        .order(egui::Order::Foreground)
        .fixed_pos(row.left_bottom() + egui::vec2(0.0, 2.0))
        .show(&ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_width(w);
                // The editor's own keys, taken BEFORE the text edit — which
                // would otherwise indent on Tab, break the line on Enter
                // and walk the cursor on the arrows.
                let focused = ui.memory(|m| m.has_focus(edit_id));
                let (tab, up, down, enter) = ui.input_mut(|i| {
                    if !focused {
                        return (None, false, false, false);
                    }
                    let n = !completions.is_empty() || cycle.is_some();
                    // Tab, and which way: shift steps back.
                    let tab = i.events.iter().find_map(|e| match e {
                        egui::Event::Key { key: egui::Key::Tab, pressed: true, modifiers, .. } => {
                            Some(modifiers.shift)
                        }
                        _ => None,
                    });
                    i.events.retain(|e| !matches!(e, egui::Event::Key { key: egui::Key::Tab, .. }));
                    let up = n && i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp);
                    let down = n && i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown);
                    // Enter sets; shift+Enter is left for the edit, where
                    // it starts a new line.
                    let is_enter = |e: &egui::Event| {
                        matches!(e, egui::Event::Key { key: egui::Key::Enter, pressed: true, modifiers, .. } if !modifiers.shift)
                    };
                    let enter = i.events.iter().any(is_enter);
                    if enter {
                        i.events.retain(|e| !is_enter(e));
                    }
                    (tab, up, down, enter)
                });
                // While Tab is stepping, the arrows step too; otherwise
                // they move the highlight, and Enter then takes it.
                let step = match (tab, up, down, cycle.is_some()) {
                    (Some(back), ..) => Some(back),
                    (None, true, _, true) => Some(true),
                    (None, _, true, true) => Some(false),
                    _ => None,
                };
                if cycle.is_none() && (up || down) {
                    pick = if up {
                        pick.saturating_sub(1)
                    } else {
                        (pick + 1).min(completions.len().saturating_sub(1))
                    };
                    navigated = true;
                }
                let mut moved = up || down;
                if let Some(back) = step.filter(|_| current || cycle.is_some()) {
                    if tab_step(&mut text, &mut cursor, &mut cycle, &completions, pick, back) {
                        put_cursor(ui.ctx(), &text, cursor);
                        moved = true;
                    }
                }
                if enter {
                    entered = true;
                    let means = enter_means(
                        text.trim().is_empty(),
                        cycle.is_some(),
                        navigated,
                        judged && result.is_ok(),
                        judged,
                        if current { completions.len() } else { 0 },
                        pick,
                    );
                    // Whatever it means, the stepping is over.
                    cycle = None;
                    navigated = false;
                    match means {
                        Enter::Set => commit = true,
                        Enter::Take(i) => {
                            accept(&mut text, &mut cursor, completions[i].start, &completions[i].insert);
                            put_cursor(ui.ctx(), &text, cursor);
                        }
                        Enter::Settle | Enter::Stay => {}
                    }
                }
                if fresh {
                    cursor = text.len();
                    put_cursor(ui.ctx(), &text, cursor);
                }
                let font = egui::TextStyle::Monospace.resolve(ui.style());
                let colors = [
                    ui.visuals().text_color(),
                    crate::theme::patch_expr(),
                    ui.visuals().error_fg_color,
                    ui.visuals().weak_text_color(),
                ];
                let mut layouter = |ui: &egui::Ui, buf: &dyn egui::TextBuffer, wrap: f32| {
                    let mut job = highlight(buf.as_str(), &valid, font.clone(), colors);
                    job.wrap.max_width = wrap;
                    ui.fonts_mut(|f| f.layout_job(job))
                };
                let rows = text.split('\n').count().clamp(1, 14);
                let out = egui::TextEdit::multiline(&mut text)
                    .id(edit_id)
                    .font(egui::TextStyle::Monospace)
                    .hint_text("../node/param * 2")
                    .lock_focus(true)
                    .desired_width(f32::INFINITY)
                    .desired_rows(rows)
                    .return_key(egui::KeyboardShortcut::new(egui::Modifiers::SHIFT, egui::Key::Enter))
                    .layouter(&mut layouter)
                    .show(ui);
                let te = out.response.response;
                if fresh {
                    te.request_focus();
                }
                if let Some(c) = out.cursor_range {
                    cursor = byte_of(&text, c.primary.index.0);
                }
                // Tab only reaches egui while a NUMBER field has focus
                // (`keymap::take_search` strips it otherwise, for the graph's
                // search). A number field is recognised by the text egui keeps
                // under its id — so this editor keeps one too.
                ui.data_mut(|d| d.insert_temp(edit_id, text.clone()));
                match &result {
                    Ok(v) => {
                        ui.label(egui::RichText::new(format!("= {v:.4}")).small().weak());
                    }
                    // Mid-address with completions on offer is typing, not
                    // an error yet.
                    Err(e) if !e.is_empty() && completions.is_empty() => {
                        ui.label(egui::RichText::new(e).small().color(ui.visuals().error_fg_color));
                    }
                    Err(_) => {}
                }
                // While stepping, the list is the one the stepping began
                // with and the highlight is where it has got to.
                let (list, lit, usable) = match &cycle {
                    Some(c) => (c.cands.clone(), c.idx, true),
                    None => (completions.clone(), pick, current),
                };
                let pointing = ui.input(|i| i.pointer.delta() != egui::Vec2::ZERO);
                let mut take = None;
                egui::ScrollArea::vertical().max_height(190.0).id_salt(id.with("list")).show(ui, |ui| {
                    for (i, c) in list.iter().enumerate() {
                        let r = ui.horizontal(|ui| {
                            let r = ui.selectable_label(
                                i == lit,
                                egui::RichText::new(&c.insert).monospace(),
                            );
                            ui.label(egui::RichText::new(&c.hint).small().weak());
                            r
                        });
                        if moved && i == lit {
                            r.inner.scroll_to_me(None);
                        }
                        if r.inner.clicked() {
                            take = Some(i);
                        }
                        // The pointer picks too, so the graph rings what is
                        // under it — but only a pointer that MOVES: one
                        // resting on the list must not pin the highlight
                        // against the arrows.
                        if r.inner.hovered() && pointing && cycle.is_none() {
                            pick = i;
                        }
                    }
                });
                if let Some(c) = take.and_then(|i| list.get(i)).filter(|_| usable) {
                    accept(&mut text, &mut cursor, c.start, &c.insert);
                    cycle = None;
                    navigated = false;
                    pick = 0;
                    te.request_focus();
                    put_cursor(ui.ctx(), &text, cursor);
                }
                ui.label(
                    egui::RichText::new(
                        "Enter sets · shift+Enter new line · Tab steps through completions · Esc cancels",
                    )
                    .small()
                    .weak(),
                );
                if enter {
                    // Decided above, before the text edit drew.
                } else if te.lost_focus() {
                    let (esc, press) =
                        ui.input(|i| (i.key_pressed(egui::Key::Escape), i.pointer.press_origin()));
                    if esc {
                        close = true;
                    } else if press.is_some_and(|p| last_rect.contains(p)) {
                        // A click on the editor itself (a completion): keep typing.
                        te.request_focus();
                    } else {
                        // Clicked away: Houdini keeps what you typed.
                        commit = true;
                    }
                }
            });
        });
    // An expression that cannot be read is never set: the editor stays up
    // on its error, with the text, until it is fixed or Esc drops it — a
    // stray click must not throw five lines away. Only a verdict on THIS
    // text counts; `frame` may not have seen the last keystroke yet.
    if commit && !text.trim().is_empty() && judged && result.as_ref().is_err_and(|e| !e.is_empty()) {
        commit = false;
        if entered {
            ctx.memory_mut(|m| m.request_focus(edit_id));
        }
    }
    with(|s| {
        if commit {
            if let Some(o) = s.open.take() {
                s.edits.push(Edit::Set { node: o.node, chans: o.chans, text });
            }
        } else if close {
            s.open = None;
        } else if let Some(o) = s.open.as_mut() {
            o.text = text;
            o.cursor = cursor;
            o.pick = pick;
            o.cycle = cycle;
            o.navigated = navigated;
            o.rect = area.response.rect;
        }
    });
    if commit || close {
        ctx.data_mut(|d| d.remove::<String>(edit_id));
        ctx.request_repaint();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::{EditorGraph, ENode};
    use engine::graph::{GradeParams, NodeKind};

    fn grade_graph() -> (EditorGraph, u64, u64) {
        let mut g = EditorGraph::default();
        let a = g.add(NodeKind::Grade(GradeParams::default()), egui::Pos2::ZERO);
        let b = g.add(NodeKind::Grade(GradeParams::default()), egui::Pos2::ZERO);
        (g, a, b)
    }

    #[test]
    fn names_are_kind_and_id_or_the_label() {
        let (mut g, a, b) = grade_graph();
        assert_eq!(node_name(&g, a).unwrap(), format!("grade{a}"));
        assert_eq!(find_node(&g, &format!("GRADE{b}")), Some(b));
        // Two grades: the bare kind names neither.
        assert_eq!(find_node(&g, "grade"), None);
        g.node_mut(b).unwrap().ui_label = "Hero Key".into();
        assert_eq!(node_name(&g, b).unwrap(), "hero_key");
        assert_eq!(find_node(&g, "hero_key"), Some(b));
        // The auto name keeps working after a rename.
        assert_eq!(find_node(&g, &format!("grade{b}")), Some(b));
        // A Read's own number is not part of its name.
        let r = g.add(NodeKind::read(0, ""), egui::Pos2::ZERO);
        assert_eq!(node_name(&g, r).unwrap(), format!("read{r}"));
        // Spelled the way the graph draws it, or with an underscore.
        assert_eq!(find_node(&g, &format!("Grade {a}")), Some(a));
        assert_eq!(find_node(&g, &format!("grade_{a}")), Some(a));
    }

    #[test]
    fn paths_resolve_to_channels() {
        let (g, a, b) = grade_graph();
        let gain = Channel::GainR;
        let p = path_to(&g, a, gain, Some(b)).unwrap();
        assert_eq!(p, format!("../grade{a}/gainr"));
        assert_eq!(resolve(&g, &p, Some(b)), Ok((a, gain)));
        // By key, either case; `./` reads the driven node.
        assert_eq!(resolve(&g, "./GainR", Some(b)), Ok((b, gain)));
        assert_eq!(resolve(&g, "gainr", Some(b)), Ok((b, gain)));
        assert!(resolve(&g, "../nope/gainr", Some(b)).is_err());
        assert!(resolve(&g, &format!("../grade{a}/nope"), Some(b)).is_err());
    }

    #[test]
    fn a_typed_expression_is_an_expr_node_wired_to_the_channel() {
        let (mut g, a, b) = grade_graph();
        if let Some(ENode { kind: NodeKind::Grade(p), .. }) = g.node_mut(a) {
            p.gain = [0.5, 0.5, 0.5];
        }
        let src = format!("../grade{a}/gainr * 4");
        set_expr(&mut g, b, &[Channel::GainG], &src, 0);
        let m = driver(&g, b, Channel::GainG).expect("wired");
        assert!(matches!(&g.patch.mods.iter().find(|x| x.id == m).unwrap().kind, ModKind::Expr(x) if *x == src));
        // It evaluates through the patch like any cable.
        let eg = g.to_engine_graph_at(Some(0));
        let v = crate::anim::get(&eg.nodes[&b], Channel::GainG).unwrap();
        assert!((v - 2.0).abs() < 1e-6, "{v}");
        // Editing reuses the node; empty text removes it.
        set_expr(&mut g, b, &[Channel::GainG], "./gainr + 1", 0);
        assert_eq!(driver(&g, b, Channel::GainG), Some(m));
        set_expr(&mut g, b, &[Channel::GainG], "", 0);
        assert_eq!(driver(&g, b, Channel::GainG), None);
        assert!(g.patch.mods.is_empty());
    }

    #[test]
    fn constant_math_is_a_number_not_a_driver() {
        let (mut g, _, b) = grade_graph();
        set_expr(&mut g, b, &[Channel::GainB], "10 / 4", 0);
        assert_eq!(driver(&g, b, Channel::GainB), None);
        assert_eq!(crate::anim::get(&g.node(b).unwrap().kind, Channel::GainB), Some(2.5));
    }

    #[test]
    fn completions_walk_nodes_then_params() {
        let (g, a, b) = grade_graph();
        let at_end = |text: &str| complete(&g, text, text.len(), b, 0);
        let nodes = at_end("2 * ../gr");
        assert!(nodes.iter().any(|c| c.insert == format!("../grade{a}/") && c.start == 4), "{nodes:?}");
        let params = at_end(&format!("../grade{a}/gai"));
        assert!(params.iter().any(|c| c.insert == format!("../grade{a}/gainr")), "{params:?}");
        assert!(at_end("./gainr").iter().any(|c| c.insert == "./gainr"));
        // Only addresses that exist: an unknown node offers nothing, and
        // neither does a comment.
        assert!(at_end("../nope/g").is_empty());
        assert!(at_end("# ../gr").is_empty());
        // Inside ch("…") too.
        assert!(!at_end("ch(\"../gr").is_empty());
    }

    #[test]
    fn tab_extends_then_steps_through_the_candidates() {
        let (g, a, b) = grade_graph();
        // The cursor is on line one of two, mid-text.
        let mut text = "x = ../gr\nx * 2".to_string();
        let mut cursor = "x = ../gr".len();
        let mut cycle = None;
        let mut tab = |text: &mut String, cursor: &mut usize, back: bool| {
            let comps = complete(&g, text, *cursor, b, 0);
            tab_step(text, cursor, &mut cycle, &comps, 0, back)
        };
        let line1 = |text: &str| text.lines().next().unwrap().to_string();
        // Two grades: Tab goes as far as they agree...
        assert!(tab(&mut text, &mut cursor, false));
        assert_eq!(line1(&text), "x = ../grade");
        // ...then steps through them, each written into the text, round
        // and round, and back with shift.
        assert!(tab(&mut text, &mut cursor, false));
        assert_eq!(line1(&text), format!("x = ../grade{a}/"));
        assert!(tab(&mut text, &mut cursor, false));
        assert_eq!(line1(&text), format!("x = ../grade{b}/"));
        assert_eq!(cursor, format!("x = ../grade{b}/").len());
        assert!(tab(&mut text, &mut cursor, false));
        assert_eq!(line1(&text), format!("x = ../grade{a}/"));
        assert!(tab(&mut text, &mut cursor, true));
        assert_eq!(line1(&text), format!("x = ../grade{b}/"));
        // The rest of the text is never touched.
        assert!(text.ends_with("\nx * 2"));
    }

    #[test]
    fn one_candidate_is_taken_without_stepping() {
        let (mut g, a, b) = grade_graph();
        g.node_mut(a).unwrap().ui_label = "Hero".into();
        let mut text = "../he".to_string();
        let mut cursor = text.len();
        let mut cycle = None;
        let comps = complete(&g, &text, cursor, b, 0);
        assert!(tab_step(&mut text, &mut cursor, &mut cycle, &comps, 0, false));
        assert_eq!(text, "../hero/");
        // Nothing to step through — so its params are on offer at once.
        assert!(cycle.is_none());
        assert!(!complete(&g, &text, cursor, b, 0).is_empty());
        // Away from an address there is nothing to complete.
        let (mut t2, mut c2) = ("1 + ".to_string(), 4);
        assert!(!tab_step(&mut t2, &mut c2, &mut cycle, &[], 0, false));
    }

    #[test]
    fn what_enter_means() {
        // (empty, cycling, navigated, readable, judged, offered, pick)
        // Readable as it stands: set — also straight after Tab stepping.
        assert_eq!(enter_means(false, false, false, true, true, 3, 0), Enter::Set);
        assert_eq!(enter_means(false, true, false, true, true, 3, 1), Enter::Set);
        // Stepped onto a node: not whole yet, so the stepping just stops.
        assert_eq!(enter_means(false, true, false, false, true, 3, 1), Enter::Settle);
        // Half an address with candidates: Enter finishes it.
        assert_eq!(enter_means(false, false, false, false, true, 3, 0), Enter::Take(0));
        // A highlight moved by hand wins even when the text already reads.
        assert_eq!(enter_means(false, false, true, true, true, 3, 2), Enter::Take(2));
        // Unreadable, nothing offered: stay on the error.
        assert_eq!(enter_means(false, false, false, false, true, 0, 0), Enter::Stay);
        // Empty takes the expression off.
        assert_eq!(enter_means(true, false, false, false, true, 0, 0), Enter::Set);
    }

    #[test]
    fn a_multi_line_expression_drives_the_channel() {
        let (mut g, a, b) = grade_graph();
        if let Some(ENode { kind: NodeKind::Grade(p), .. }) = g.node_mut(a) {
            p.gain = [0.5, 0.5, 0.5];
        }
        let src = format!("g = ../grade{a}/gainr  # key\nlift = 1\ng * 4 + lift");
        set_expr(&mut g, b, &[Channel::GainG], &src, 0);
        assert!(driver(&g, b, Channel::GainG).is_some());
        let eg = g.to_engine_graph_at(Some(0));
        assert_eq!(crate::anim::get(&eg.nodes[&b], Channel::GainG), Some(3.0));
        // The preview names the line that fails.
        let bad = format!("g = ../grade{a}/gainr\ng * nope");
        let e = preview(&g, &bad, b, &[Channel::GainG], 0).unwrap_err();
        assert!(e.starts_with("line 2"), "{e}");
        let e = preview(&g, "g = ../nope/x\ng", b, &[Channel::GainG], 0).unwrap_err();
        assert!(e.starts_with("line 1") && e.contains("nope"), "{e}");
        // The row shows it on one line.
        assert_eq!(one_line("g = 1\n\n  g * 2\n"), "g = 1; g * 2");
    }

    #[test]
    fn addresses_are_coloured_by_whether_they_resolve() {
        let (plain, ok, bad, dim) = (
            egui::Color32::from_gray(200),
            egui::Color32::from_rgb(1, 2, 3),
            egui::Color32::from_rgb(9, 0, 0),
            egui::Color32::from_gray(90),
        );
        let text = "../grade1/gainr + ../nope/x  # ../grade1/gainr\n./new";
        let valid = vec![("../grade1/gainr".to_string(), true), ("../nope/x".to_string(), false)];
        let job = highlight(text, &valid, egui::FontId::monospace(12.0), [plain, ok, bad, dim]);
        let seen: Vec<(&str, egui::Color32)> =
            job.sections.iter().map(|s| (&job.text[s.byte_range.start.0..s.byte_range.end.0], s.format.color)).collect();
        assert_eq!(job.text, text);
        assert!(seen.contains(&("../grade1/gainr", ok)), "{seen:?}");
        assert!(seen.contains(&("../nope/x", bad)), "{seen:?}");
        assert!(seen.contains(&("# ../grade1/gainr", dim)), "{seen:?}");
        // Not judged yet: plain, never a guess.
        assert!(seen.iter().any(|(s, c)| s.contains("./new") && *c == plain), "{seen:?}");
    }

    // ---- The rows, driven through egui with real input -----------------

    /// What `properties_ui` hands a row: its node's driven channels, live.
    fn driven_for(g: &EditorGraph, node: u64) -> Vec<(Channel, f32)> {
        let base = g.nodes.iter().map(|(id, n)| (*id, n.kind_at(0))).collect();
        g.patch
            .links
            .iter()
            .filter_map(|l| match l.to {
                PDst::Chan(n, c) if n == node => {
                    Some((c, g.patch_value(l.from, &base, 0.0, &mut Vec::new())))
                }
                _ => None,
            })
            .collect()
    }

    /// One UI pass: `frame`, then a gain-g slider row for `node`, as the
    /// app runs them.
    fn pass(ctx: &egui::Context, g: &mut EditorGraph, node: u64, v: &mut f32, events: Vec<egui::Event>) {
        frame(ctx, g, 0);
        let driven = driven_for(g, node);
        let n = g.node_mut(node).unwrap();
        let ekind = n.kind_at(0);
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(420.0, 400.0))),
            events,
            ..Default::default()
        };
        let _ = ctx.run_ui(raw, |ui| {
            crate::widgets::clear_rows(ui.ctx());
            let mut cx = crate::widgets::RowCtx {
                node,
                anim: &mut n.anim,
                ekind: &ekind,
                t: 0,
                hud_live: false,
                driven: driven.clone(),
            };
            ui.set_width(360.0);
            crate::widgets::slider(ui, &mut cx, "gain g", v, 0.0, 4.0, &[Channel::GainG]);
        });
    }

    fn key(k: egui::Key) -> egui::Event {
        egui::Event::Key { key: k, physical_key: None, pressed: true, repeat: false, modifiers: egui::Modifiers::NONE }
    }

    /// Click the row's value field and type `text` into it.
    fn type_into_field(ctx: &egui::Context, g: &mut EditorGraph, node: u64, v: &mut f32, text: &str) {
        pass(ctx, g, node, v, Vec::new());
        let row = crate::widgets::rows(ctx).last().expect("the row drew").rect;
        let gap = ctx.global_style().spacing.item_spacing.x;
        let p = egui::pos2(
            row.right() - crate::widgets::DOT_W - gap - crate::widgets::NUM_W / 2.0,
            row.center().y,
        );
        let btn = |pressed| egui::Event::PointerButton {
            pos: p,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        pass(ctx, g, node, v, vec![egui::Event::PointerMoved(p), btn(true)]);
        pass(ctx, g, node, v, vec![btn(false)]);
        pass(ctx, g, node, v, Vec::new());
        pass(ctx, g, node, v, vec![egui::Event::Text(text.into())]);
    }

    #[test]
    fn typing_a_path_into_a_field_wires_an_expression() {
        let (mut g, a, b) = grade_graph();
        let ctx = egui::Context::default();
        let mut v = 1.0;
        let src = format!("../grade{a}/gainr*2");
        type_into_field(&ctx, &mut g, b, &mut v, &src);
        // The path handed the typing to the expression editor.
        assert!(with(|s| s.open.as_ref().is_some_and(|o| o.text == src)), "editor did not open");
        for _ in 0..3 {
            pass(&ctx, &mut g, b, &mut v, Vec::new());
        }
        // It reads the other grade's gain r, doubled.
        assert!(with(|s| s.open.as_ref().is_some_and(|o| o.result == Ok(2.0))));
        pass(&ctx, &mut g, b, &mut v, vec![key(egui::Key::Enter)]);
        pass(&ctx, &mut g, b, &mut v, Vec::new());
        let m = driver(&g, b, Channel::GainG).expect("Enter wired the expression");
        assert!(matches!(&g.patch.mods.iter().find(|x| x.id == m).unwrap().kind, ModKind::Expr(x) if *x == src));
        assert!(with(|s| s.open.is_none()));
        // The row now shows it as an expression.
        pass(&ctx, &mut g, b, &mut v, Vec::new());
        assert_eq!(row_text(b, &[Channel::GainG]), Some(src));
        // The authored value was never touched.
        assert_eq!(v, 1.0);
    }

    #[test]
    fn escape_leaves_the_channel_alone() {
        let (mut g, a, b) = grade_graph();
        let ctx = egui::Context::default();
        let mut v = 1.0;
        type_into_field(&ctx, &mut g, b, &mut v, &format!("../grade{a}/gainr"));
        pass(&ctx, &mut g, b, &mut v, Vec::new());
        pass(&ctx, &mut g, b, &mut v, vec![key(egui::Key::Escape)]);
        pass(&ctx, &mut g, b, &mut v, Vec::new());
        assert!(with(|s| s.open.is_none()));
        assert!(g.patch.links.is_empty());
    }

    #[test]
    fn a_bad_path_stays_open_on_its_error() {
        let (mut g, _, b) = grade_graph();
        let ctx = egui::Context::default();
        let mut v = 1.0;
        type_into_field(&ctx, &mut g, b, &mut v, "../nope/gainr");
        for _ in 0..3 {
            pass(&ctx, &mut g, b, &mut v, Vec::new());
        }
        pass(&ctx, &mut g, b, &mut v, vec![key(egui::Key::Enter)]);
        pass(&ctx, &mut g, b, &mut v, Vec::new());
        assert!(g.patch.links.is_empty());
        assert!(with(|s| s.open.as_ref().is_some_and(|o| o.result.as_ref().is_err_and(|e| e.contains("nope")))));
    }

    #[test]
    fn arithmetic_in_a_field_is_its_answer() {
        let (mut g, _, b) = grade_graph();
        let ctx = egui::Context::default();
        let mut v = 1.0;
        type_into_field(&ctx, &mut g, b, &mut v, "10/4");
        assert!(with(|s| s.open.is_none()), "plain arithmetic must stay in the field");
        pass(&ctx, &mut g, b, &mut v, vec![key(egui::Key::Enter)]);
        pass(&ctx, &mut g, b, &mut v, Vec::new());
        assert_eq!(v, 2.5);
        assert!(g.patch.links.is_empty());
    }

    fn key_mod(k: egui::Key, modifiers: egui::Modifiers) -> egui::Event {
        egui::Event::Key { key: k, physical_key: None, pressed: true, repeat: false, modifiers }
    }

    fn open_text() -> String {
        with(|s| s.open.as_ref().map(|o| o.text.clone()).unwrap_or_default())
    }

    #[test]
    fn tab_in_the_editor_completes_and_the_app_lets_it_through() {
        let (mut g, a, b) = grade_graph();
        let ctx = egui::Context::default();
        let mut v = 1.0;
        type_into_field(&ctx, &mut g, b, &mut v, "../gr");
        for _ in 0..3 {
            pass(&ctx, &mut g, b, &mut v, Vec::new());
        }
        // The app strips Tab before egui sees it (it is the graph's search
        // key) unless a number field is being typed into. The editor has
        // to count as one, or Tab would never arrive.
        let mut raw = egui::RawInput { events: vec![key(egui::Key::Tab)], ..Default::default() };
        assert!(!crate::keymap::take_search(&ctx, &mut raw), "Tab opened the node search");
        assert_eq!(raw.events.len(), 1, "Tab was stripped before the editor could have it");
        // Two grades: as far as they agree, then the pick.
        pass(&ctx, &mut g, b, &mut v, vec![key(egui::Key::Tab)]);
        assert_eq!(open_text(), "../grade");
        pass(&ctx, &mut g, b, &mut v, Vec::new());
        pass(&ctx, &mut g, b, &mut v, vec![key(egui::Key::Tab)]);
        assert_eq!(open_text(), format!("../grade{a}/"));
        // Typing carries on from the end of what Tab wrote, no tab character.
        pass(&ctx, &mut g, b, &mut v, vec![egui::Event::Text("gainr".into())]);
        assert_eq!(open_text(), format!("../grade{a}/gainr"));
        pass(&ctx, &mut g, b, &mut v, Vec::new());
        pass(&ctx, &mut g, b, &mut v, vec![key(egui::Key::Enter)]);
        pass(&ctx, &mut g, b, &mut v, Vec::new());
        assert!(driver(&g, b, Channel::GainG).is_some());
    }

    /// The whole gesture by keys alone: Tab, Tab, Tab to a node, Enter;
    /// Tab, Tab to a param, Enter — and the expression is set.
    #[test]
    fn tabbing_to_a_candidate_then_enter_sets() {
        let (mut g, a, b) = grade_graph();
        let ctx = egui::Context::default();
        let mut v = 1.0;
        type_into_field(&ctx, &mut g, b, &mut v, "../gr");
        let settle = |g: &mut EditorGraph, v: &mut f32| {
            for _ in 0..2 {
                pass(&ctx, g, b, v, Vec::new());
            }
        };
        let press = |g: &mut EditorGraph, v: &mut f32, e: egui::Event| {
            pass(&ctx, g, b, v, vec![e]);
            for _ in 0..2 {
                pass(&ctx, g, b, v, Vec::new());
            }
        };
        settle(&mut g, &mut v);
        press(&mut g, &mut v, key(egui::Key::Tab));
        assert_eq!(open_text(), "../grade");
        press(&mut g, &mut v, key(egui::Key::Tab));
        assert_eq!(open_text(), format!("../grade{a}/"));
        press(&mut g, &mut v, key(egui::Key::Tab));
        assert_eq!(open_text(), format!("../grade{b}/"));
        // The list is still the two nodes while stepping, the second lit.
        assert!(with(|s| s.open.as_ref().is_some_and(|o| o.cycle.as_ref().is_some_and(|c| c.cands.len() == 2 && c.idx == 1))));
        assert!(lit(b) || b == a);
        // shift+Tab steps back.
        press(&mut g, &mut v, key_mod(egui::Key::Tab, egui::Modifiers::SHIFT));
        assert_eq!(open_text(), format!("../grade{a}/"));
        // Enter on a node: not an expression yet, so it only settles there
        // and the node's params come on offer.
        press(&mut g, &mut v, key(egui::Key::Enter));
        assert!(with(|s| s.open.as_ref().is_some_and(|o| o.cycle.is_none() && !o.completions.is_empty())));
        assert!(g.patch.links.is_empty());
        // Tab steps through the params; Enter on one SETS.
        press(&mut g, &mut v, key(egui::Key::Tab));
        press(&mut g, &mut v, key(egui::Key::Tab));
        let landed = open_text();
        assert!(landed.starts_with(&format!("../grade{a}/")) && landed.len() > format!("../grade{a}/").len(), "{landed}");
        press(&mut g, &mut v, key(egui::Key::Enter));
        let m = driver(&g, b, Channel::GainG).expect("Enter after Tab set the expression");
        assert!(matches!(&g.patch.mods.iter().find(|x| x.id == m).unwrap().kind, ModKind::Expr(x) if *x == landed));
        assert!(with(|s| s.open.is_none()));
    }

    /// Half an address and Enter: it is finished, not refused in silence.
    /// A highlight moved with the arrows is the one Enter takes.
    #[test]
    fn enter_finishes_a_half_typed_address() {
        let (mut g, _, b) = grade_graph();
        let ctx = egui::Context::default();
        let mut v = 1.0;
        type_into_field(&ctx, &mut g, b, &mut v, "./gain");
        let press = |g: &mut EditorGraph, v: &mut f32, e: egui::Event| {
            pass(&ctx, g, b, v, vec![e]);
            for _ in 0..2 {
                pass(&ctx, g, b, v, Vec::new());
            }
        };
        press(&mut g, &mut v, egui::Event::PointerGone);
        // gainr, gaing, gainb: down twice is gain b.
        press(&mut g, &mut v, key(egui::Key::ArrowDown));
        press(&mut g, &mut v, key(egui::Key::ArrowDown));
        press(&mut g, &mut v, key(egui::Key::Enter));
        assert_eq!(open_text(), "./gainb");
        assert!(g.patch.links.is_empty(), "the first Enter only finishes the address");
        press(&mut g, &mut v, key(egui::Key::Enter));
        assert!(driver(&g, b, Channel::GainG).is_some());
    }

    #[test]
    fn shift_enter_starts_a_line_and_enter_sets_all_of_them() {
        let (mut g, a, b) = grade_graph();
        let ctx = egui::Context::default();
        let mut v = 1.0;
        type_into_field(&ctx, &mut g, b, &mut v, &format!("g = ../grade{a}/gainr"));
        pass(&ctx, &mut g, b, &mut v, Vec::new());
        pass(&ctx, &mut g, b, &mut v, vec![key_mod(egui::Key::Enter, egui::Modifiers::SHIFT)]);
        assert!(with(|s| s.open.is_some()), "shift+Enter set the expression");
        pass(&ctx, &mut g, b, &mut v, vec![egui::Event::Text("g * 3".into())]);
        let src = format!("g = ../grade{a}/gainr\ng * 3");
        assert_eq!(open_text(), src);
        for _ in 0..2 {
            pass(&ctx, &mut g, b, &mut v, Vec::new());
        }
        assert!(with(|s| s.open.as_ref().is_some_and(|o| o.result == Ok(3.0))));
        pass(&ctx, &mut g, b, &mut v, vec![key(egui::Key::Enter)]);
        pass(&ctx, &mut g, b, &mut v, Vec::new());
        let m = driver(&g, b, Channel::GainG).expect("Enter set it");
        assert!(matches!(&g.patch.mods.iter().find(|x| x.id == m).unwrap().kind, ModKind::Expr(x) if *x == src));
    }

    /// Look at it: `cargo test -p app --bin wobble chanexpr_shot -- --ignored`
    /// writes `target/chanexpr-editor.png` — a row driven by an expression,
    /// and the editor open on the next row mid-path.
    #[test]
    #[ignore]
    fn chanexpr_shot() {
        let (mut g, a, b) = grade_graph();
        set_expr(&mut g, b, &[Channel::GainR], &format!("../grade{a}/gainr * 2 + sin($F/10)"), 0);
        let text = format!("key = ../grade{a}/gainr   # the key light\nfill = ../nope/gain\nkey * 2 + ../grade{a}/g");
        let mut v = [1.0f32; 2];
        crate::uishot::render("chanexpr-editor", 460.0, 380.0, 2.0, 6, |ui, f| {
            frame(ui.ctx(), &mut g, 0);
            if f == 0 {
                open(ui.ctx(), b, &[Channel::GainB], &text);
            }
            let driven = driven_for(&g, b);
            let n = g.node_mut(b).unwrap();
            let ekind = n.kind_at(0);
            let mut cx = crate::widgets::RowCtx {
                node: b,
                anim: &mut n.anim,
                ekind: &ekind,
                t: 0,
                hud_live: false,
                driven,
            };
            egui::Frame::new().inner_margin(12.0).show(ui, |ui| {
                ui.set_width(380.0);
                crate::widgets::slider(ui, &mut cx, "gain r", &mut v[0], 0.0, 4.0, &[Channel::GainR]);
                crate::widgets::slider(ui, &mut cx, "gain b", &mut v[1], 0.0, 4.0, &[Channel::GainB]);
            });
        });
    }

    /// Look at the graph side: `cargo test -p app --bin wobble
    /// chanexpr_graph_shot -- --ignored` writes `target/chanexpr-graph.png`
    /// — two Grades that the graph normally labels alike, named by path
    /// while the editor is open, the one the text names ringed.
    #[test]
    #[ignore]
    fn chanexpr_graph_shot() {
        let mut g = EditorGraph::default();
        let r = g.add(NodeKind::read(0, ""), egui::Pos2::new(0.0, -220.0));
        let a = g.add(NodeKind::Grade(GradeParams::default()), egui::Pos2::new(-90.0, -60.0));
        let b = g.add(NodeKind::Grade(GradeParams::default()), egui::Pos2::new(90.0, -60.0));
        let m = g.add(NodeKind::blend(), egui::Pos2::new(0.0, 110.0));
        g.connect(a, 0, r);
        g.connect(b, 0, r);
        g.connect(m, 0, a);
        g.connect(m, 1, b);
        g.view = Some(m);
        let text = format!("../grade{a}/gainr * ");
        let mut ed = crate::editor::NodeEditor::default();
        let (mut sel, mut act, mut pin, mut slot, mut rects) =
            (vec![b], Vec::new(), Vec::new(), 2usize, Vec::new());
        let mut v = 1.0f32;
        crate::uishot::render("chanexpr-graph", 900.0, 520.0, 2.0, 6, |ui, f| {
            frame(ui.ctx(), &mut g, 0);
            if f == 0 {
                open(ui.ctx(), b, &[Channel::GainG], &text);
            }
            let full = ui.max_rect();
            let (left, right) = full.split_left_right_at_x(full.left() + 480.0);
            ui.scope_builder(egui::UiBuilder::new().max_rect(left), |ui| {
                ed.show(ui, &mut g, &mut sel, &mut act, &mut pin, &mut slot, &mut rects);
            });
            let driven = driven_for(&g, b);
            let n = g.node_mut(b).unwrap();
            let ekind = n.kind_at(0);
            let mut cx = crate::widgets::RowCtx {
                node: b,
                anim: &mut n.anim,
                ekind: &ekind,
                t: 0,
                hud_live: false,
                driven,
            };
            ui.scope_builder(egui::UiBuilder::new().max_rect(right.shrink(16.0)), |ui| {
                ui.set_width(380.0);
                crate::widgets::slider(ui, &mut cx, "gain g", &mut v, 0.0, 4.0, &[Channel::GainG]);
            });
        });
    }

    #[test]
    fn typing_hands_off_only_for_expressions() {
        assert!(wants_editor("../"));
        assert!(wants_editor("./g"));
        assert!(wants_editor("2*pi"));
        assert!(wants_editor("$F"));
        assert!(!wants_editor("1e3"));
        assert!(!wants_editor("-0.5"));
        assert!(!wants_editor("10/2"));
    }
}
