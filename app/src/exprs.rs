//! Tiny float expression language for the patch bay's Expr node: variables
//! `a`/`b` (the node's two inputs) and `t` (timeline frame), arithmetic with
//! the usual precedence, and a small function set. Evaluation IS the parse —
//! expressions are short and re-parsed per evaluation.
//!
//! Houdini-style CHANNEL REFERENCES read other parameters: `../grade12/gainr`
//! (a node in this graph, then its param), `./gainr` (the node the
//! expression drives), and `ch("../grade12/gainr")` for Houdini hands. `$F`
//! is `t`. What a path names is not this module's business: [`eval_with`]
//! hands the path text to a resolver (`crate::chanexpr`), and plain [`eval`]
//! resolves nothing, so a path there is an error.
//!
//! An expression may run to several LINES. A line `name = …` names a value
//! for the lines after it, and the expression's value is its last line's:
//!
//! ```text
//! g = ../grade12/gainr      # anything after a # is a comment
//! g * 2 + sin($F / 10)
//! ```
//!
//! `;` separates statements like a newline. A line that ends in an operator
//! (or sits inside brackets) continues on the next.

pub fn eval(src: &str, a: f32, b: f32, t: f32) -> Option<f32> {
    eval_with(src, a, b, t, &mut |_| None)
}

/// [`eval`] with channel references: `resolve` gets each path as written
/// (`../grade12/gainr`, `./gainr`, or `ch("…")`'s string) and answers its
/// value, or `None` for a path that names nothing — which fails the whole
/// expression, like any other error.
pub fn eval_with(
    src: &str,
    a: f32,
    b: f32,
    t: f32,
    resolve: &mut dyn FnMut(&str) -> Option<f32>,
) -> Option<f32> {
    eval_report(src, a, b, t, resolve).ok()
}

/// [`eval_with`], saying WHERE it failed: the 1-based line of the statement
/// that could not be read.
pub fn eval_report(
    src: &str,
    a: f32,
    b: f32,
    t: f32,
    resolve: &mut dyn FnMut(&str) -> Option<f32>,
) -> Result<f32, usize> {
    let mut p = P { s: src.as_bytes(), i: 0, a, b, t, resolve, vars: Vec::new(), depth: 0, stmt: 0 };
    p.program().ok_or_else(|| {
        let at = p.stmt.min(p.s.len());
        1 + p.s[..at].iter().filter(|c| **c == b'\n').count()
    })
}

/// [`refs`] with where each sits: `(byte range, path as written)`. For a
/// `ch("…")` the range is the string's inside.
pub fn ref_spans(src: &str) -> Vec<(std::ops::Range<usize>, String)> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let rest = &src[i..];
        if b[i] == b'#' {
            // A comment names nothing.
            i += rest.find('\n').unwrap_or(rest.len());
        } else if let Some((n, path)) = ch_string(rest) {
            let at = i + rest[..n].find(path).unwrap_or(0);
            out.push((at..at + path.len(), path.to_string()));
            i += n;
        } else if let Some(n) = path_len(rest) {
            // A path starts a token: `1../x` is not one.
            let glued = i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_');
            if !glued {
                out.push((i..i + n, rest[..n].to_string()));
            }
            i += n;
        } else {
            i += rest.chars().next().map_or(1, char::len_utf8);
        }
    }
    out
}

/// Every channel reference in `src`, in order, as written. Scans rather
/// than parses, so a half-typed expression still lists what it names.
pub fn refs(src: &str) -> Vec<String> {
    ref_spans(src).into_iter().map(|(_, p)| p).collect()
}

/// Does `src` read anything that can change — a channel, or the frame?
/// Constant math (`2*3`) typed into a field is a number, not a driver.
pub fn is_live(src: &str) -> bool {
    if !refs(src).is_empty() || src.contains("$F") {
        return true;
    }
    // `t`, `a` or `b` as a whole identifier: the frame, or a jack.
    let b = src.as_bytes();
    b.iter().enumerate().any(|(i, c)| {
        let word = |c: &u8| c.is_ascii_alphanumeric() || *c == b'_';
        matches!(c, b't' | b'a' | b'b')
            && !(i > 0 && word(&b[i - 1]))
            && !b.get(i + 1).is_some_and(word)
    })
}

/// A path's grammar: `../name/param` or `./param`, where a name or param is
/// `[A-Za-z0-9_]+`. Exactly two segments after `../`, so `../g/gainr/2` is
/// a path divided by two. Returns the path's byte length.
fn path_len(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    let seg = |from: usize| {
        let n = b[from..]
            .iter()
            .take_while(|c| c.is_ascii_alphanumeric() || **c == b'_')
            .count();
        (n > 0).then_some(from + n)
    };
    if s.starts_with("../") {
        let e = seg(3)?;
        if b.get(e) != Some(&b'/') {
            return None;
        }
        seg(e + 1)
    } else if s.starts_with("./") {
        seg(2)
    } else {
        None
    }
}

/// `ch("path")` / `ch('path')`: (bytes consumed, the path).
fn ch_string(s: &str) -> Option<(usize, &str)> {
    let rest = s.strip_prefix("ch")?;
    let open = rest.len() - rest.trim_start().len();
    let rest = rest.trim_start().strip_prefix('(')?;
    let pad = rest.len() - rest.trim_start().len();
    let rest = rest.trim_start();
    let q = rest.chars().next().filter(|c| *c == '"' || *c == '\'')?;
    let body = &rest[1..];
    let end = body.find(q)?;
    let after = &body[end + 1..];
    let pad2 = after.len() - after.trim_start().len();
    after.trim_start().strip_prefix(')')?;
    Some((2 + open + 1 + pad + 1 + end + 1 + pad2 + 1, &body[..end]))
}

struct P<'a> {
    s: &'a [u8],
    i: usize,
    a: f32,
    b: f32,
    t: f32,
    resolve: &'a mut dyn FnMut(&str) -> Option<f32>,
    /// Values named by earlier lines; a later line of the same name wins.
    vars: Vec<(&'a [u8], f32)>,
    /// Bracket depth: inside brackets a newline is only whitespace.
    depth: u32,
    /// Where the statement being read starts, for the error's line.
    stmt: usize,
}

impl<'a> P<'a> {
    fn ws(&mut self) {
        while self.s.get(self.i).is_some_and(|c| c.is_ascii_whitespace()) {
            self.i += 1;
        }
    }

    /// Spaces and tabs, never a newline.
    fn inline_ws(&mut self) {
        while self.s.get(self.i).is_some_and(|c| matches!(c, b' ' | b'\t' | b'\r')) {
            self.i += 1;
        }
    }

    /// Blank lines, `;` and `#` comments between statements.
    fn skip_blank(&mut self) {
        loop {
            self.ws();
            match self.s.get(self.i) {
                Some(b';') => self.i += 1,
                Some(b'#') => {
                    while self.s.get(self.i).is_some_and(|c| *c != b'\n') {
                        self.i += 1;
                    }
                }
                _ => return,
            }
        }
    }

    /// Statements; the value is the last one's.
    fn program(&mut self) -> Option<f32> {
        let mut last = None;
        loop {
            self.skip_blank();
            if self.i >= self.s.len() {
                return last;
            }
            self.stmt = self.i;
            let name = self.assign_target();
            let v = self.expr()?;
            if let Some(name) = name {
                self.vars.push((name, v));
            }
            last = Some(v);
            // A statement ends at its line's end, a `;` or a comment.
            self.inline_ws();
            match self.s.get(self.i) {
                None | Some(b'\n' | b';' | b'#') => {}
                Some(_) => return None,
            }
        }
    }

    /// `name =` at the start of a statement: the name, with both consumed.
    fn assign_target(&mut self) -> Option<&'a [u8]> {
        let start = self.i;
        let mut j = start;
        if !self.s.get(j).is_some_and(|c| c.is_ascii_alphabetic() || *c == b'_') {
            return None;
        }
        while self.s.get(j).is_some_and(|c| c.is_ascii_alphanumeric() || *c == b'_') {
            j += 1;
        }
        let end = j;
        while self.s.get(j).is_some_and(|c| matches!(c, b' ' | b'\t')) {
            j += 1;
        }
        if self.s.get(j) != Some(&b'=') || self.s.get(j + 1) == Some(&b'=') {
            return None;
        }
        self.i = j + 1;
        Some(&self.s[start..end])
    }

    /// A binary operator. Outside brackets it must be on the line: one at
    /// the START of the next line would make `x = 1` / `-x` read as `1 - x`.
    fn eat_op(&mut self, c: u8) -> bool {
        if self.depth > 0 {
            return self.eat(c);
        }
        self.inline_ws();
        if self.s.get(self.i) == Some(&c) {
            self.i += 1;
            return true;
        }
        false
    }

    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.s.get(self.i).copied()
    }

    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.i += 1;
            return true;
        }
        false
    }

    fn expr(&mut self) -> Option<f32> {
        let mut v = self.term()?;
        loop {
            if self.eat_op(b'+') {
                v += self.term()?;
            } else if self.eat_op(b'-') {
                v -= self.term()?;
            } else {
                return Some(v);
            }
        }
    }

    fn term(&mut self) -> Option<f32> {
        let mut v = self.unary()?;
        loop {
            if self.eat_op(b'*') {
                v *= self.unary()?;
            } else if self.eat_op(b'/') {
                v /= self.unary()?;
            } else if self.eat_op(b'%') {
                v = v.rem_euclid(self.unary()?);
            } else {
                return Some(v);
            }
        }
    }

    fn unary(&mut self) -> Option<f32> {
        if self.eat(b'-') {
            return Some(-self.unary()?);
        }
        let base = self.atom()?;
        if self.eat_op(b'^') {
            return Some(base.powf(self.unary()?));
        }
        Some(base)
    }

    fn atom(&mut self) -> Option<f32> {
        match self.peek()? {
            b'(' => {
                self.i += 1;
                self.depth += 1;
                let v = self.expr()?;
                self.depth -= 1;
                self.eat(b')').then_some(v)
            }
            b'.' if self.s.get(self.i + 1).is_some_and(|c| *c == b'.' || *c == b'/') => {
                self.path()
            }
            b'$' => {
                // Houdini's frame variable.
                if self.s[self.i..].starts_with(b"$F")
                    && !self.s.get(self.i + 2).is_some_and(|c| c.is_ascii_alphanumeric())
                {
                    self.i += 2;
                    return Some(self.t);
                }
                None
            }
            c if c.is_ascii_digit() || c == b'.' => self.number(),
            c if c.is_ascii_alphabetic() || c == b'_' => self.ident(),
            _ => None,
        }
    }

    fn path(&mut self) -> Option<f32> {
        let rest = std::str::from_utf8(&self.s[self.i..]).ok()?;
        let n = path_len(rest)?;
        let path = &rest[..n];
        self.i += n;
        (self.resolve)(path)
    }

    fn number(&mut self) -> Option<f32> {
        let start = self.i;
        while self
            .s
            .get(self.i)
            .is_some_and(|c| c.is_ascii_digit() || *c == b'.')
        {
            self.i += 1;
        }
        std::str::from_utf8(&self.s[start..self.i]).ok()?.parse().ok()
    }

    fn ident(&mut self) -> Option<f32> {
        let start = self.i;
        if let Some((n, path)) =
            std::str::from_utf8(&self.s[self.i..]).ok().and_then(ch_string)
        {
            self.i += n;
            return (self.resolve)(path);
        }
        while self
            .s
            .get(self.i)
            .is_some_and(|c| c.is_ascii_alphanumeric() || *c == b'_')
        {
            self.i += 1;
        }
        let name = std::str::from_utf8(&self.s[start..self.i]).ok()?;
        // A value an earlier line named, the latest of that name.
        if let Some((_, v)) = self.vars.iter().rev().find(|(n, _)| *n == name.as_bytes()) {
            return Some(*v);
        }
        // Variables and constants.
        match name {
            "a" => return Some(self.a),
            "b" => return Some(self.b),
            "t" => return Some(self.t),
            "pi" => return Some(std::f32::consts::PI),
            "tau" => return Some(std::f32::consts::TAU),
            "e" => return Some(std::f32::consts::E),
            _ => {}
        }
        // Functions.
        if !self.eat(b'(') {
            return None;
        }
        let mut args = Vec::new();
        if !self.eat(b')') {
            self.depth += 1;
            loop {
                args.push(self.expr()?);
                if self.eat(b')') {
                    break;
                }
                if !self.eat(b',') {
                    return None;
                }
            }
            self.depth -= 1;
        }
        let f1 = |f: fn(f32) -> f32| (args.len() == 1).then(|| f(args[0]));
        let f2 = |f: fn(f32, f32) -> f32| (args.len() == 2).then(|| f(args[0], args[1]));
        let f3 =
            |f: fn(f32, f32, f32) -> f32| (args.len() == 3).then(|| f(args[0], args[1], args[2]));
        match name {
            "sin" => f1(f32::sin),
            "cos" => f1(f32::cos),
            "tan" => f1(f32::tan),
            "asin" => f1(f32::asin),
            "acos" => f1(f32::acos),
            "atan" => f1(f32::atan),
            "abs" => f1(f32::abs),
            "sqrt" => f1(f32::sqrt),
            "floor" => f1(f32::floor),
            "ceil" => f1(f32::ceil),
            "round" => f1(f32::round),
            "exp" => f1(f32::exp),
            "log" => f1(f32::ln),
            "sign" => f1(f32::signum),
            "fract" => f1(f32::fract),
            "min" => f2(f32::min),
            "max" => f2(f32::max),
            "pow" => f2(f32::powf),
            "atan2" => f2(f32::atan2),
            "step" => f2(|edge, x| if x < edge { 0.0 } else { 1.0 }),
            "clamp" => f3(f32::clamp),
            "mix" => f3(|a, b, x| a + (b - a) * x),
            "smoothstep" => f3(|e0, e1, x| {
                let u = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
                u * u * (3.0 - 2.0 * u)
            }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{eval, eval_report, eval_with, is_live, ref_spans, refs};

    #[test]
    fn arithmetic_and_precedence() {
        assert_eq!(eval("a*2+1", 3.0, 0.0, 0.0), Some(7.0));
        assert_eq!(eval("1+2*3", 0.0, 0.0, 0.0), Some(7.0));
        assert_eq!(eval("(1+2)*3", 0.0, 0.0, 0.0), Some(9.0));
        assert_eq!(eval("2^3", 0.0, 0.0, 0.0), Some(8.0));
        assert_eq!(eval("-a+4", 3.0, 0.0, 0.0), Some(1.0));
        assert_eq!(eval("7 % 3", 0.0, 0.0, 0.0), Some(1.0));
    }

    #[test]
    fn vars_and_functions() {
        assert_eq!(eval("min(a, b)", 5.0, 2.0, 0.0), Some(2.0));
        assert_eq!(eval("clamp(t, 0, 10)", 0.0, 0.0, 25.0), Some(10.0));
        assert_eq!(eval("mix(0, 10, 0.25)", 0.0, 0.0, 0.0), Some(2.5));
        assert_eq!(eval("sin(0) + cos(0)", 0.0, 0.0, 0.0), Some(1.0));
        assert_eq!(eval("abs(-2) * sqrt(4)", 0.0, 0.0, 0.0), Some(4.0));
        assert_eq!(eval("step(0.5, b)", 0.0, 0.7, 0.0), Some(1.0));
        assert_eq!(eval("smoothstep(0, 1, 0.5)", 0.0, 0.0, 0.0), Some(0.5));
    }

    #[test]
    fn paths_resolve_through_the_callback() {
        let mut seen = Vec::new();
        let mut r = |p: &str| {
            seen.push(p.to_string());
            match p {
                "../grade12/gainr" => Some(2.0),
                "./gaing" => Some(3.0),
                _ => None,
            }
        };
        assert_eq!(eval_with("../grade12/gainr * 2 + 1", 0.0, 0.0, 0.0, &mut r), Some(5.0));
        assert_eq!(eval_with("./gaing/2", 0.0, 0.0, 0.0, &mut r), Some(1.5));
        // Two segments, then division: a path never swallows a `/`.
        assert_eq!(eval_with("../grade12/gainr/4", 0.0, 0.0, 0.0, &mut r), Some(0.5));
        assert_eq!(eval_with("ch(\"../grade12/gainr\") * $F", 0.0, 0.0, 10.0, &mut r), Some(20.0));
        assert_eq!(eval_with("ch('./gaing')", 0.0, 0.0, 0.0, &mut r), Some(3.0));
        // An unknown path fails the expression.
        assert_eq!(eval_with("../nope/x + 1", 0.0, 0.0, 0.0, &mut r), None);
        // Numbers that start with a point are still numbers.
        assert_eq!(eval_with(".5 + ./gaing", 0.0, 0.0, 0.0, &mut r), Some(3.5));
        // Plain `eval` resolves nothing.
        assert_eq!(eval("../grade12/gainr", 0.0, 0.0, 0.0), None);
    }

    #[test]
    fn lines_name_values_and_the_last_line_is_the_answer() {
        let mut r = |p: &str| (p == "../grade12/gainr").then_some(2.0);
        let src = "g = ../grade12/gainr   # the key light\n\nk = g * 3\nk + 1";
        assert_eq!(eval_with(src, 0.0, 0.0, 0.0, &mut r), Some(7.0));
        // `;` is a newline; a later line of the same name wins.
        assert_eq!(eval("x = 2; x = x * 5; x + 1", 0.0, 0.0, 0.0), Some(11.0));
        // A name shadows the inputs and constants.
        assert_eq!(eval("a = 10\na * 2", 1.0, 0.0, 0.0), Some(20.0));
        // An assignment alone is its own value.
        assert_eq!(eval("x = 4", 0.0, 0.0, 0.0), Some(4.0));
        // A trailing operator continues the line; so do brackets.
        assert_eq!(eval("1 +\n 2", 0.0, 0.0, 0.0), Some(3.0));
        assert_eq!(eval("max(1,\n  5\n)", 0.0, 0.0, 0.0), Some(5.0));
        assert_eq!(eval("(1\n + 2)", 0.0, 0.0, 0.0), Some(3.0));
        // A new line never continues the one above: `-x` is its own statement.
        assert_eq!(eval("x = 1\n-x", 0.0, 0.0, 0.0), Some(-1.0));
        // Two values on one line are still an error, and so is a name not yet set.
        assert_eq!(eval("1 2", 0.0, 0.0, 0.0), None);
        assert_eq!(eval("y + 1\ny = 2", 0.0, 0.0, 0.0), None);
        // Comments and blank lines alone are nothing.
        assert_eq!(eval("# just a note\n", 0.0, 0.0, 0.0), None);
    }

    #[test]
    fn errors_name_their_line() {
        let mut r = |_: &str| None;
        assert_eq!(eval_report("x = 1\ny = 2 +\nx + y", 0.0, 0.0, 0.0, &mut r), Err(2));
        assert_eq!(eval_report("x = 1\n\nx + nope", 0.0, 0.0, 0.0, &mut r), Err(3));
        assert_eq!(eval_report("x = 1\nx", 0.0, 0.0, 0.0, &mut r), Ok(1.0));
    }

    #[test]
    fn ref_spans_point_at_the_paths() {
        let src = "g = ../grade12/gainr\nch(\"./mix\") * g  # ../not/this";
        let spans = ref_spans(src);
        assert_eq!(spans.len(), 2);
        assert_eq!(&src[spans[0].0.clone()], "../grade12/gainr");
        assert_eq!(&src[spans[1].0.clone()], "./mix");
    }

    #[test]
    fn refs_and_liveness() {
        assert_eq!(
            refs("../grade12/gainr * ch(\"./mix\") + ../blur3/size/2"),
            vec!["../grade12/gainr", "./mix", "../blur3/size"]
        );
        assert!(refs("2 * 3").is_empty());
        assert!(!is_live("2 * 3 + pi"));
        assert!(!is_live("abs(-2)"));
        assert!(is_live("t * 2"));
        assert!(is_live("$F"));
        assert!(is_live("./gainr"));
    }

    #[test]
    fn rejects_garbage() {
        assert_eq!(eval("2 +", 0.0, 0.0, 0.0), None);
        assert_eq!(eval("foo(1)", 0.0, 0.0, 0.0), None);
        assert_eq!(eval("min(1)", 0.0, 0.0, 0.0), None);
        assert_eq!(eval("1 2", 0.0, 0.0, 0.0), None);
        assert_eq!(eval("", 0.0, 0.0, 0.0), None);
        assert_eq!(eval("c*2", 0.0, 0.0, 0.0), None);
    }
}
