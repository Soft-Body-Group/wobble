//! Expression: a pixel written as arithmetic — one expression per output
//! channel, compiled into the fused pass like any built-in. The classic
//! Expression node, in the patch bay's language (the same operators and
//! functions, `name = …` lines, `;`, `#` comments) plus what a test on a
//! pixel needs: comparisons, `&& || !`, `c ? a : b`, `inf`, `nan`, `isinf`,
//! `isnan`.
//!
//! ```text
//! r == 0 ? 1 : 0          # a matte where red is exactly zero
//! isinf(r) ? 0 : r        # infinite fireflies out of a render
//! ```
//!
//! **Variables**: `r g b a` — the input pixel; `x y` — the pixel's centre in
//! pixels from the top left; `width height`; `frame` (also `t` and `$F`, as
//! in the patch bay). `x y width height` are FULL-resolution pixels, so an
//! expression means the same thing with proxy on. **Constants**: `pi tau e
//! inf nan`. A comparison or a logic op is 1 or 0, and anything non-zero is
//! true. `x == inf` and `x == -inf` test exactly that; `x == nan` means
//! `isnan(x)` (IEEE would say "never", which is no use to anybody).
//!
//! An empty field passes its channel through. So does a field that does not
//! parse — the comp keeps rendering while you type, the Shader node's rule
//! — and [`check`] says where it broke, for the panel.
//!
//! **Text is topology.** The expressions ARE the generated code, so a new
//! expression recompiles (the panel commits on Enter, not per keystroke) and
//! the numbers in one are compiled in. `inf`, `nan`, the frame and the proxy
//! ride the uniform block — WGSL refuses an infinite constant — and the
//! `inf`/`nan` tests compare BITS, which no fast-math pass may fold away.
//!
//! [`eval`] is the CPU mirror of the generated code, WGSL's semantics
//! function for function; the tests check the GPU against it.

use crate::codegen::Emit;
use crate::nodes::NodeDef;

const TOPO_BASE: u64 = 0x7E5;

/// The output channels, in field order.
pub const CHANNELS: [char; 4] = ['r', 'g', 'b', 'a'];

/// A [`crate::graph::NodeKind::Expression`]'s params: one expression per
/// output channel. Empty passes the channel through.
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct ExpressionParams {
    pub exprs: [String; 4],
    /// The tick being rendered, resolved per frame by the app. TRANSIENT:
    /// never authored, never saved (Grain's `time`).
    #[cfg_attr(feature = "serde", serde(skip))]
    pub time: f32,
}

// ---------------------------------------------------------------------------
// The language
// ---------------------------------------------------------------------------

/// Where an expression stopped making sense, and why.
#[derive(Clone, Debug, PartialEq)]
pub struct ExprError {
    /// Byte offset into the source.
    pub at: usize,
    pub msg: String,
}

impl ExprError {
    fn new(at: usize, msg: impl Into<String>) -> Self {
        Self { at, msg: msg.into() }
    }

    /// `msg (column n)`, 1-based, for a one-line field.
    pub fn describe(&self, src: &str) -> String {
        let line = 1 + src[..self.at.min(src.len())].matches('\n').count();
        let col = 1 + self.at - src[..self.at.min(src.len())].rfind('\n').map_or(0, |i| i + 1);
        if line > 1 {
            format!("{} (line {line}, column {col})", self.msg)
        } else {
            format!("{} (column {col})", self.msg)
        }
    }
}

/// What an expression can read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Var {
    R,
    G,
    B,
    A,
    X,
    Y,
    Width,
    Height,
    Frame,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Func {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Abs,
    Sqrt,
    Floor,
    Ceil,
    Round,
    Trunc,
    Exp,
    Exp2,
    Log,
    Log2,
    Sign,
    Fract,
    Saturate,
    IsInf,
    IsNan,
    Min,
    Max,
    Pow,
    Atan2,
    Step,
    Clamp,
    Mix,
    Smoothstep,
}

impl Func {
    fn lookup(name: &str) -> Option<(Func, usize)> {
        use Func::*;
        Some(match name {
            "sin" => (Sin, 1),
            "cos" => (Cos, 1),
            "tan" => (Tan, 1),
            "asin" => (Asin, 1),
            "acos" => (Acos, 1),
            "atan" => (Atan, 1),
            "abs" => (Abs, 1),
            "sqrt" => (Sqrt, 1),
            "floor" => (Floor, 1),
            "ceil" => (Ceil, 1),
            "round" => (Round, 1),
            "trunc" => (Trunc, 1),
            "exp" => (Exp, 1),
            "exp2" => (Exp2, 1),
            "log" => (Log, 1),
            "log2" => (Log2, 1),
            "sign" => (Sign, 1),
            "fract" => (Fract, 1),
            "saturate" => (Saturate, 1),
            "isinf" => (IsInf, 1),
            "isnan" => (IsNan, 1),
            "min" => (Min, 2),
            "max" => (Max, 2),
            "pow" => (Pow, 2),
            "atan2" => (Atan2, 2),
            "step" => (Step, 2),
            "clamp" => (Clamp, 3),
            "mix" | "lerp" => (Mix, 3),
            "smoothstep" => (Smoothstep, 3),
            _ => return None,
        })
    }
}

/// Every name an expression can use, for the panel's help.
pub const FUNCTIONS: &str = "sin cos tan asin acos atan atan2 abs sqrt pow exp exp2 log log2 \
    floor ceil round trunc fract sign min max clamp saturate step smoothstep mix lerp isinf isnan";

#[derive(Clone, Debug, PartialEq)]
enum Ex {
    Num(f32),
    Var(Var),
    Inf,
    Nan,
    /// A `name =` binding, by statement index.
    Local(usize),
    Neg(Box<Ex>),
    Not(Box<Ex>),
    Bin(Op, Box<Ex>, Box<Ex>),
    Cond(Box<Ex>, Box<Ex>, Box<Ex>),
    Call(Func, Vec<Ex>),
}

/// A parsed field: its statements, the last one's value being the result.
#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    stmts: Vec<Ex>,
}

#[derive(Clone, Debug, PartialEq)]
enum T {
    Num(f32),
    Id(String),
    Frame,
    Op(&'static str),
    Sep,
}

#[derive(Clone, Debug)]
struct Tok {
    t: T,
    at: usize,
}

/// Two-character operators first, so `<=` is never read as `<` then `=`.
const OPS: [&str; 22] = [
    "==", "!=", "<=", ">=", "&&", "||", "+", "-", "*", "/", "%", "^", "<", ">", "!", "?", ":",
    "(", ")", ",", "=", ";",
];

/// A token after which a line break continues the statement: a binary
/// operator (or `?`, `:`, `(`, `,`, `=`) at a line's end is not finished.
fn continues(t: &T) -> bool {
    matches!(t, T::Op(o) if !matches!(*o, ")" | ";"))
}

fn lex(src: &str) -> Result<Vec<Tok>, ExprError> {
    let b = src.as_bytes();
    let mut out: Vec<Tok> = Vec::new();
    let mut depth = 0i32;
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c == b'#' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if c == b'\n' {
            // A statement ends at a line's end — unless it is inside
            // brackets, or the line ended in an operator.
            let open = depth > 0 || out.last().is_some_and(|t| continues(&t.t));
            if !open && out.last().is_some_and(|t| t.t != T::Sep) {
                out.push(Tok { t: T::Sep, at: i });
            }
            i += 1;
            continue;
        }
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let at = i;
        if c.is_ascii_digit() || (c == b'.' && b.get(i + 1).is_some_and(|d| d.is_ascii_digit())) {
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') {
                i += 1;
            }
            // An exponent: `1e-3`.
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                let mut j = i + 1;
                if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
                    j += 1;
                }
                if j < b.len() && b[j].is_ascii_digit() {
                    i = j;
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            let v: f32 = src[at..i]
                .parse()
                .map_err(|_| ExprError::new(at, format!("`{}` is not a number", &src[at..i])))?;
            if !v.is_finite() {
                return Err(ExprError::new(at, "number too big for a float"));
            }
            out.push(Tok { t: T::Num(v), at });
            continue;
        }
        if c.is_ascii_alphabetic() || c == b'_' {
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            out.push(Tok { t: T::Id(src[at..i].to_string()), at });
            continue;
        }
        if src[i..].starts_with("$F") && !b.get(i + 2).is_some_and(|d| d.is_ascii_alphanumeric()) {
            out.push(Tok { t: T::Frame, at });
            i += 2;
            continue;
        }
        let Some(op) = OPS.iter().find(|o| src[i..].starts_with(**o)) else {
            let ch = src[i..].chars().next().unwrap_or('?');
            return Err(ExprError::new(at, format!("`{ch}` is not part of the language")));
        };
        i += op.len();
        match *op {
            ";" => {
                if out.last().is_some_and(|t| t.t != T::Sep) {
                    out.push(Tok { t: T::Sep, at });
                }
            }
            "(" => {
                depth += 1;
                out.push(Tok { t: T::Op(op), at });
            }
            ")" => {
                depth -= 1;
                out.push(Tok { t: T::Op(op), at });
            }
            _ => out.push(Tok { t: T::Op(op), at }),
        }
    }
    Ok(out)
}

struct Parser<'s> {
    toks: Vec<Tok>,
    i: usize,
    end: usize,
    /// The names bound so far, by statement index.
    locals: Vec<Option<String>>,
    _src: &'s str,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&T> {
        self.toks.get(self.i).map(|t| &t.t)
    }

    fn at(&self) -> usize {
        self.toks.get(self.i).map_or(self.end, |t| t.at)
    }

    fn eat(&mut self, op: &str) -> bool {
        if matches!(self.peek(), Some(T::Op(o)) if *o == op) {
            self.i += 1;
            return true;
        }
        false
    }

    fn expect(&mut self, op: &str) -> Result<(), ExprError> {
        if self.eat(op) {
            Ok(())
        } else {
            Err(self.unexpected(&format!("`{op}`")))
        }
    }

    fn unexpected(&self, want: &str) -> ExprError {
        let found = match self.peek() {
            None => "the end".to_string(),
            Some(T::Sep) => "the end of the statement".to_string(),
            Some(T::Num(v)) => format!("`{v}`"),
            Some(T::Id(s)) => format!("`{s}`"),
            Some(T::Frame) => "`$F`".to_string(),
            Some(T::Op(o)) => format!("`{o}`"),
        };
        ExprError::new(self.at(), format!("expected {want}, found {found}"))
    }

    fn program(&mut self) -> Result<Program, ExprError> {
        let mut stmts = Vec::new();
        while self.i < self.toks.len() {
            if matches!(self.peek(), Some(T::Sep)) {
                self.i += 1;
                continue;
            }
            // `name = …` — a single `=`, never `==` (the lexer reads that
            // as one token).
            let name = match (self.peek(), self.toks.get(self.i + 1).map(|t| &t.t)) {
                (Some(T::Id(n)), Some(T::Op("="))) => {
                    let n = n.clone();
                    self.i += 2;
                    Some(n)
                }
                _ => None,
            };
            let ex = self.cond()?;
            if !matches!(self.peek(), None | Some(T::Sep)) {
                return Err(self.unexpected("an operator or the end of the statement"));
            }
            stmts.push(ex);
            self.locals.push(name);
        }
        if stmts.is_empty() {
            return Err(ExprError::new(0, "empty expression"));
        }
        Ok(Program { stmts })
    }

    fn cond(&mut self) -> Result<Ex, ExprError> {
        let c = self.or()?;
        if self.eat("?") {
            let a = self.cond()?;
            self.expect(":")?;
            let b = self.cond()?;
            return Ok(Ex::Cond(Box::new(c), Box::new(a), Box::new(b)));
        }
        Ok(c)
    }

    fn or(&mut self) -> Result<Ex, ExprError> {
        let mut v = self.and()?;
        while self.eat("||") {
            v = Ex::Bin(Op::Or, Box::new(v), Box::new(self.and()?));
        }
        Ok(v)
    }

    fn and(&mut self) -> Result<Ex, ExprError> {
        let mut v = self.cmp()?;
        while self.eat("&&") {
            v = Ex::Bin(Op::And, Box::new(v), Box::new(self.cmp()?));
        }
        Ok(v)
    }

    fn cmp(&mut self) -> Result<Ex, ExprError> {
        let mut v = self.sum()?;
        loop {
            let op = match self.peek() {
                Some(T::Op("==")) => Op::Eq,
                Some(T::Op("!=")) => Op::Ne,
                Some(T::Op("<")) => Op::Lt,
                Some(T::Op(">")) => Op::Gt,
                Some(T::Op("<=")) => Op::Le,
                Some(T::Op(">=")) => Op::Ge,
                _ => return Ok(v),
            };
            self.i += 1;
            v = Ex::Bin(op, Box::new(v), Box::new(self.sum()?));
        }
    }

    fn sum(&mut self) -> Result<Ex, ExprError> {
        let mut v = self.term()?;
        loop {
            let op = if self.eat("+") {
                Op::Add
            } else if self.eat("-") {
                Op::Sub
            } else {
                return Ok(v);
            };
            v = Ex::Bin(op, Box::new(v), Box::new(self.term()?));
        }
    }

    fn term(&mut self) -> Result<Ex, ExprError> {
        let mut v = self.unary()?;
        loop {
            let op = if self.eat("*") {
                Op::Mul
            } else if self.eat("/") {
                Op::Div
            } else if self.eat("%") {
                Op::Mod
            } else {
                return Ok(v);
            };
            v = Ex::Bin(op, Box::new(v), Box::new(self.unary()?));
        }
    }

    fn unary(&mut self) -> Result<Ex, ExprError> {
        if self.eat("-") {
            return Ok(Ex::Neg(Box::new(self.unary()?)));
        }
        if self.eat("!") {
            return Ok(Ex::Not(Box::new(self.unary()?)));
        }
        let base = self.atom()?;
        // `a ^ -b ^ c` binds right, as in the patch bay.
        if self.eat("^") {
            return Ok(Ex::Bin(Op::Pow, Box::new(base), Box::new(self.unary()?)));
        }
        Ok(base)
    }

    fn atom(&mut self) -> Result<Ex, ExprError> {
        let at = self.at();
        match self.peek().cloned() {
            Some(T::Num(v)) => {
                self.i += 1;
                Ok(Ex::Num(v))
            }
            Some(T::Frame) => {
                self.i += 1;
                Ok(Ex::Var(Var::Frame))
            }
            Some(T::Op("(")) => {
                self.i += 1;
                let v = self.cond()?;
                self.expect(")")?;
                Ok(v)
            }
            Some(T::Id(name)) => {
                self.i += 1;
                if self.eat("(") {
                    let Some((f, arity)) = Func::lookup(&name) else {
                        return Err(ExprError::new(at, format!("no function called `{name}`")));
                    };
                    let mut args = Vec::new();
                    if !self.eat(")") {
                        loop {
                            args.push(self.cond()?);
                            if self.eat(")") {
                                break;
                            }
                            self.expect(",")?;
                        }
                    }
                    if args.len() != arity {
                        let s = if arity == 1 { "" } else { "s" };
                        return Err(ExprError::new(
                            at,
                            format!("`{name}` takes {arity} argument{s}, got {}", args.len()),
                        ));
                    }
                    return Ok(Ex::Call(f, args));
                }
                // A value an earlier line named — the latest of that name —
                // before any built-in name, so a line may shadow one.
                if let Some(k) = self.locals.iter().rposition(|l| l.as_deref() == Some(&name)) {
                    return Ok(Ex::Local(k));
                }
                Ok(match name.as_str() {
                    "r" => Ex::Var(Var::R),
                    "g" => Ex::Var(Var::G),
                    "b" => Ex::Var(Var::B),
                    "a" => Ex::Var(Var::A),
                    "x" => Ex::Var(Var::X),
                    "y" => Ex::Var(Var::Y),
                    "width" => Ex::Var(Var::Width),
                    "height" => Ex::Var(Var::Height),
                    "frame" | "t" => Ex::Var(Var::Frame),
                    "pi" => Ex::Num(std::f32::consts::PI),
                    "tau" => Ex::Num(std::f32::consts::TAU),
                    "e" => Ex::Num(std::f32::consts::E),
                    "inf" => Ex::Inf,
                    "nan" => Ex::Nan,
                    _ if Func::lookup(&name).is_some() => {
                        return Err(ExprError::new(at, format!("`{name}` is a function: `{name}(…)`")));
                    }
                    _ => return Err(ExprError::new(at, format!("no variable called `{name}`"))),
                })
            }
            _ => Err(self.unexpected("a value")),
        }
    }
}

/// Parse one field.
pub fn parse(src: &str) -> Result<Program, ExprError> {
    let toks = lex(src)?;
    let mut p = Parser { toks, i: 0, end: src.len(), locals: Vec::new(), _src: src };
    p.program()
}

/// What the panel shows under a field: `None` for an empty field (it passes
/// through) or one that parses, else the error.
pub fn check(src: &str) -> Option<ExprError> {
    if src.trim().is_empty() {
        return None;
    }
    parse(src).err()
}

// ---------------------------------------------------------------------------
// The CPU mirror
// ---------------------------------------------------------------------------

/// What [`eval`] reads.
#[derive(Clone, Copy, Debug, Default)]
pub struct Env {
    pub rgba: [f32; 4],
    /// Full-resolution pixels, centres at +0.5, y down.
    pub xy: [f32; 2],
    pub size: [f32; 2],
    pub frame: f32,
}

/// Anything but ±0 is true, NaN included — the generated code's rule.
fn truth(v: f32) -> bool {
    v.to_bits() & 0x7fff_ffff != 0
}

fn bit(b: bool) -> f32 {
    if b {
        1.0
    } else {
        0.0
    }
}

fn is_inf(v: f32) -> bool {
    v.to_bits() & 0x7fff_ffff == 0x7f80_0000
}

fn is_nan(v: f32) -> bool {
    v.to_bits() & 0x7fff_ffff > 0x7f80_0000
}

/// `a == b` the way the generated code tests it: exact bits against `inf`
/// and `-inf`, `isnan` against `nan`, IEEE otherwise.
fn equal(a: &Ex, b: &Ex, va: f32, vb: f32) -> bool {
    match (a, b) {
        (_, Ex::Nan) => is_nan(va),
        (Ex::Nan, _) => is_nan(vb),
        _ => va == vb,
    }
}

/// The CPU mirror of the generated code.
pub fn eval(p: &Program, env: &Env) -> f32 {
    let mut vals = Vec::with_capacity(p.stmts.len());
    for s in &p.stmts {
        let v = ev(s, env, &vals);
        vals.push(v);
    }
    *vals.last().unwrap_or(&0.0)
}

fn ev(e: &Ex, env: &Env, locals: &[f32]) -> f32 {
    let r = |x: &Ex| ev(x, env, locals);
    match e {
        Ex::Num(v) => *v,
        Ex::Var(v) => match v {
            Var::R => env.rgba[0],
            Var::G => env.rgba[1],
            Var::B => env.rgba[2],
            Var::A => env.rgba[3],
            Var::X => env.xy[0],
            Var::Y => env.xy[1],
            Var::Width => env.size[0],
            Var::Height => env.size[1],
            Var::Frame => env.frame,
        },
        Ex::Inf => f32::INFINITY,
        Ex::Nan => f32::NAN,
        Ex::Local(k) => locals[*k],
        Ex::Neg(a) => -r(a),
        Ex::Not(a) => bit(!truth(r(a))),
        Ex::Cond(c, a, b) => {
            if truth(r(c)) {
                r(a)
            } else {
                r(b)
            }
        }
        Ex::Bin(op, a, b) => {
            let (x, y) = (r(a), r(b));
            match op {
                Op::Add => x + y,
                Op::Sub => x - y,
                Op::Mul => x * y,
                Op::Div => x / y,
                // The patch bay's `%`: Euclid's, never negative.
                Op::Mod => x - y.abs() * (x / y.abs()).floor(),
                Op::Pow => x.powf(y),
                Op::Eq => bit(equal(a, b, x, y)),
                Op::Ne => bit(!equal(a, b, x, y)),
                Op::Lt => bit(x < y),
                Op::Gt => bit(x > y),
                Op::Le => bit(x <= y),
                Op::Ge => bit(x >= y),
                Op::And => bit(truth(x) && truth(y)),
                Op::Or => bit(truth(x) || truth(y)),
            }
        }
        Ex::Call(f, args) => {
            let v: Vec<f32> = args.iter().map(r).collect();
            match f {
                Func::Sin => v[0].sin(),
                Func::Cos => v[0].cos(),
                Func::Tan => v[0].tan(),
                Func::Asin => v[0].asin(),
                Func::Acos => v[0].acos(),
                Func::Atan => v[0].atan(),
                Func::Abs => v[0].abs(),
                Func::Sqrt => v[0].sqrt(),
                Func::Floor => v[0].floor(),
                Func::Ceil => v[0].ceil(),
                Func::Round => v[0].round_ties_even(),
                Func::Trunc => v[0].trunc(),
                Func::Exp => v[0].exp(),
                Func::Exp2 => v[0].exp2(),
                Func::Log => v[0].ln(),
                Func::Log2 => v[0].log2(),
                Func::Sign => {
                    if v[0] > 0.0 {
                        1.0
                    } else if v[0] < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                }
                Func::Fract => v[0] - v[0].floor(),
                Func::Saturate => v[0].clamp(0.0, 1.0),
                Func::IsInf => bit(is_inf(v[0])),
                Func::IsNan => bit(is_nan(v[0])),
                Func::Min => v[0].min(v[1]),
                Func::Max => v[0].max(v[1]),
                Func::Pow => v[0].powf(v[1]),
                Func::Atan2 => v[0].atan2(v[1]),
                Func::Step => bit(v[1] >= v[0]),
                Func::Clamp => v[0].max(v[1]).min(v[2]),
                Func::Mix => v[0] + (v[1] - v[0]) * v[2],
                // Equal edges are a step at the edge, not 0/0.
                Func::Smoothstep if v[0] == v[1] => bit(v[2] >= v[0]),
                Func::Smoothstep => {
                    let u = ((v[2] - v[0]) / (v[1] - v[0])).clamp(0.0, 1.0);
                    u * u * (3.0 - 2.0 * u)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The generated code
// ---------------------------------------------------------------------------

/// What a compiled field reads, as WGSL expressions.
struct Cx<'a> {
    rgba: &'a str,
    pos: &'a str,
    slot: usize,
}

fn lit(v: f32) -> String {
    // Debug prints `2.0`, `0.5`, `1e-10`: every one a WGSL float literal.
    format!("{v:?}")
}

/// Is `e` the constant `inf` or `-inf`? The bits a test against it is for.
fn inf_bits(e: &Ex) -> Option<u32> {
    match e {
        Ex::Inf => Some(0x7f80_0000),
        Ex::Neg(a) if **a == Ex::Inf => Some(0xff80_0000),
        _ => None,
    }
}

/// Emit `e` as `let`s into `body`; returns the name holding its value.
fn gen(
    e: &Ex,
    cx: &Cx,
    locals: &[String],
    body: &mut Vec<String>,
    fresh: &mut dyn FnMut() -> String,
) -> String {
    // A macro, not a closure: a closure would hold `body` and `fresh` for
    // the whole match, and the arms need them too.
    macro_rules! g {
        ($x:expr) => {
            gen($x, cx, locals, body, fresh)
        };
    }
    let nan = format!("P.v[{}].y", cx.slot);
    let value = match e {
        Ex::Num(v) => lit(*v),
        Ex::Var(v) => {
            let (rgba, pos, s) = (cx.rgba, cx.pos, cx.slot);
            match v {
                Var::R => format!("{rgba}.r"),
                Var::G => format!("{rgba}.g"),
                Var::B => format!("{rgba}.b"),
                Var::A => format!("{rgba}.a"),
                Var::X => format!("({pos}.x / P.v[{s}].w)"),
                Var::Y => format!("({pos}.y / P.v[{s}].w)"),
                Var::Width => format!("(P.v[0].x / P.v[{s}].w)"),
                Var::Height => format!("(P.v[0].y / P.v[{s}].w)"),
                Var::Frame => format!("P.v[{s}].z"),
            }
        }
        Ex::Inf => format!("P.v[{}].x", cx.slot),
        Ex::Nan => nan.clone(),
        Ex::Local(k) => locals[*k].clone(),
        Ex::Neg(a) => {
            let a = g!(a);
            format!("(-{a})")
        }
        Ex::Not(a) => {
            let a = g!(a);
            format!("select(0.0, 1.0, !{})", truthy(&a))
        }
        Ex::Cond(c, a, b) => {
            let (c, a, b) = (g!(c), g!(a), g!(b));
            format!("select({b}, {a}, {})", truthy(&c))
        }
        Ex::Bin(op, a, b) => {
            // Against inf/-inf or nan, `==` is a test of BITS.
            let special = if matches!(op, Op::Eq | Op::Ne) {
                inf_bits(b)
                    .map(|bits| (a.as_ref(), Some(bits)))
                    .or_else(|| inf_bits(a).map(|bits| (b.as_ref(), Some(bits))))
                    .or_else(|| (**b == Ex::Nan).then_some((a.as_ref(), None)))
                    .or_else(|| (**a == Ex::Nan).then_some((b.as_ref(), None)))
            } else {
                None
            };
            if let Some((other, bits)) = special {
                let o = g!(other);
                let test = match bits {
                    Some(bits) => format!("bitcast<u32>({o}) == {bits:#x}u"),
                    None => format!("(bitcast<u32>({o}) & 0x7fffffffu) > 0x7f800000u"),
                };
                let test = if *op == Op::Ne { format!("!({test})") } else { test };
                format!("select(0.0, 1.0, {test})")
            } else {
                let (x, y) = (g!(a), g!(b));
                match op {
                    Op::Add => format!("({x} + {y})"),
                    Op::Sub => format!("({x} - {y})"),
                    Op::Mul => format!("({x} * {y})"),
                    Op::Div => format!("({x} / {y})"),
                    Op::Mod => format!("({x} - abs({y}) * floor({x} / abs({y})))"),
                    Op::Pow => pow(&x, &y, &nan),
                    // IEEE, made to hold: a NaN on either side is false
                    // for every comparison but `!=`. The plain float test
                    // alone does not — Metal folds `x == x` to true.
                    Op::Eq => format!("select(0.0, 1.0, {x} == {y} && {})", neither_nan(&x, &y)),
                    Op::Ne => format!("select(0.0, 1.0, {x} != {y} || !{})", neither_nan(&x, &y)),
                    Op::Lt => format!("select(0.0, 1.0, {x} < {y} && {})", neither_nan(&x, &y)),
                    Op::Gt => format!("select(0.0, 1.0, {x} > {y} && {})", neither_nan(&x, &y)),
                    Op::Le => format!("select(0.0, 1.0, {x} <= {y} && {})", neither_nan(&x, &y)),
                    Op::Ge => format!("select(0.0, 1.0, {x} >= {y} && {})", neither_nan(&x, &y)),
                    Op::And => format!("select(0.0, 1.0, {} && {})", truthy(&x), truthy(&y)),
                    Op::Or => format!("select(0.0, 1.0, {} || {})", truthy(&x), truthy(&y)),
                }
            }
        }
        Ex::Call(f, args) => {
            let mut v = Vec::with_capacity(args.len());
            for a in args {
                v.push(g!(a));
            }
            match f {
                Func::Sin => format!("sin({})", v[0]),
                Func::Cos => format!("cos({})", v[0]),
                Func::Tan => format!("tan({})", v[0]),
                Func::Asin => format!("asin({})", v[0]),
                Func::Acos => format!("acos({})", v[0]),
                Func::Atan => format!("atan({})", v[0]),
                Func::Abs => format!("abs({})", v[0]),
                Func::Sqrt => format!("sqrt({})", v[0]),
                Func::Floor => format!("floor({})", v[0]),
                Func::Ceil => format!("ceil({})", v[0]),
                Func::Round => format!("round({})", v[0]),
                Func::Trunc => format!("trunc({})", v[0]),
                Func::Exp => format!("exp({})", v[0]),
                Func::Exp2 => format!("exp2({})", v[0]),
                Func::Log => format!("log({})", v[0]),
                Func::Log2 => format!("log2({})", v[0]),
                Func::Sign => format!("sign({})", v[0]),
                Func::Fract => format!("({0} - floor({0}))", v[0]),
                Func::Saturate => format!("clamp({}, 0.0, 1.0)", v[0]),
                Func::IsInf => {
                    format!("select(0.0, 1.0, (bitcast<u32>({}) & 0x7fffffffu) == 0x7f800000u)", v[0])
                }
                Func::IsNan => {
                    format!("select(0.0, 1.0, (bitcast<u32>({}) & 0x7fffffffu) > 0x7f800000u)", v[0])
                }
                Func::Min => format!("min({}, {})", v[0], v[1]),
                Func::Max => format!("max({}, {})", v[0], v[1]),
                Func::Pow => pow(&v[0], &v[1], &nan),
                Func::Atan2 => format!("atan2({}, {})", v[0], v[1]),
                Func::Step => format!("step({}, {})", v[0], v[1]),
                Func::Clamp => format!("min(max({}, {}), {})", v[0], v[1], v[2]),
                Func::Mix => format!("mix({}, {}, {})", v[0], v[1], v[2]),
                // WGSL's own smoothstep is undefined for equal or reversed
                // edges; this is the clamped Hermite the CPU mirror
                // computes, and equal edges are a step at the edge.
                Func::Smoothstep => {
                    let u = fresh();
                    body.push(format!(
                        "let {u} = clamp(({2} - {0}) / ({1} - {0}), 0.0, 1.0);",
                        v[0], v[1], v[2]
                    ));
                    format!(
                        "select({u} * {u} * (3.0 - 2.0 * {u}), step({0}, {2}), {0} == {1})",
                        v[0], v[1], v[2]
                    )
                }
            }
        }
    };
    // Every value gets a name: a value used twice is computed once, and
    // the generated code reads like the expression did.
    let name = fresh();
    body.push(format!("let {name} = {value};"));
    name
}

/// "Non-zero", tested on the BITS: anything but ±0 is true, NaN included
/// (C's rule). A float `!= 0.0` is not safe here — under fast math the
/// compiler may assume no NaN and pick the comparison that says false.
fn truthy(v: &str) -> String {
    format!("((bitcast<u32>({v}) & 0x7fffffffu) != 0u)")
}

/// Neither operand is a NaN, tested on the bits.
fn neither_nan(x: &str, y: &str) -> String {
    format!(
        "((bitcast<u32>({x}) & 0x7fffffffu) <= 0x7f800000u && (bitcast<u32>({y}) & 0x7fffffffu) <= 0x7f800000u)"
    )
}

/// `x ^ y` the way `powf` does it: a negative base takes a WHOLE exponent
/// (`(-2)^2` is 4, `(-2)^3` is -8) and gives nan for any other. WGSL's own
/// `pow` is undefined below zero.
fn pow(x: &str, y: &str, nan: &str) -> String {
    format!(
        "select(pow(abs({x}), {y}), select({nan}, select(-1.0, 1.0, fract({y} * 0.5) == 0.0) * pow(abs({x}), {y}), fract({y}) == 0.0), {x} < 0.0)"
    )
}

impl ExpressionParams {
    /// The parsed fields; `None` where a field is empty or does not parse
    /// (it passes its channel through).
    pub fn programs(&self) -> [Option<Program>; 4] {
        std::array::from_fn(|i| {
            let s = &self.exprs[i];
            if s.trim().is_empty() {
                None
            } else {
                parse(s).ok()
            }
        })
    }

    /// The CPU mirror of the whole node for one pixel.
    pub fn apply(&self, env: &Env) -> [f32; 4] {
        let progs = self.programs();
        std::array::from_fn(|i| progs[i].as_ref().map_or(env.rgba[i], |p| eval(p, env)))
    }

    /// The uniform span, with the proxy the comp renders at.
    pub fn pack_scaled(&self, v: &mut [f32], proxy: f32) {
        v[0] = f32::INFINITY;
        v[1] = f32::NAN;
        v[2] = self.time;
        v[3] = if proxy.is_finite() && proxy > 0.0 { proxy } else { 1.0 };
    }
}

impl NodeDef for ExpressionParams {
    const NAME: &'static str = "Expression";
    /// Image, then the mask.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = Some(1);
    /// inf, nan, the frame, the proxy.
    const VEC4_SLOTS: usize = 1;

    /// The text IS the code: every field's source, so a new expression
    /// recompiles and nothing else does.
    fn topo_tag(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.exprs.hash(&mut h);
        TOPO_BASE ^ h.finish().rotate_left(12)
    }

    fn pack(&self, v: &mut [f32]) {
        self.pack_scaled(v, 1.0);
    }

    fn emit(&self, e: &mut Emit) -> String {
        let input = e.input(0);
        let slot = e.slot();
        let pos = e.pos().to_string();
        // The input once, under a name every field reads.
        let src = e.fresh();
        e.line(format_args!("let {src} = {input};"));
        let progs = self.programs();
        let mut chans = Vec::with_capacity(4);
        for (i, prog) in progs.iter().enumerate() {
            let Some(prog) = prog else {
                chans.push(format!("{src}.{}", CHANNELS[i]));
                continue;
            };
            let cx = Cx { rgba: &src, pos: &pos, slot };
            let mut body = Vec::new();
            let mut locals = Vec::new();
            let mut last = String::new();
            for s in &prog.stmts {
                let mut fresh = || e.fresh();
                last = gen(s, &cx, &locals, &mut body, &mut fresh);
                locals.push(last.clone());
            }
            for l in &body {
                e.line(format_args!("{l}"));
            }
            chans.push(last);
        }
        let v = e.fresh();
        e.line(format_args!("let {v} = vec4f({}, {}, {}, {});", chans[0], chans[1], chans[2], chans[3]));
        e.masked(Self::MASK_INPUT, &src, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(src: &str, r: f32) -> f32 {
        let p = parse(src).unwrap_or_else(|e| panic!("{src}: {}", e.describe(src)));
        eval(&p, &Env { rgba: [r, 0.25, 0.5, 1.0], xy: [10.5, 20.5], size: [100.0, 50.0], frame: 7.0 })
    }

    /// The user's two: a matte of exact zeros, and infinities replaced.
    #[test]
    fn the_examples_that_asked_for_it() {
        assert_eq!(ev("r == 0 ? 1:0", 0.0), 1.0);
        assert_eq!(ev("r == 0 ? 1:0", 0.3), 0.0);
        assert_eq!(ev("r == inf ? 1 : r", f32::INFINITY), 1.0);
        assert_eq!(ev("r == inf ? 1 : r", 0.75), 0.75);
        assert_eq!(ev("r == inf ? 0 : r", -f32::INFINITY), -f32::INFINITY);
        assert_eq!(ev("r == -inf ? 0 : r", -f32::INFINITY), 0.0);
        assert_eq!(ev("isinf(r) ? 0 : r", -f32::INFINITY), 0.0);
        assert_eq!(ev("isnan(r) ? 0 : r", f32::NAN), 0.0);
        assert_eq!(ev("r == nan ? 2 : 3", f32::NAN), 2.0);
        assert_eq!(ev("r != r ? 2 : 3", 1.0), 3.0);
    }

    /// The patch bay's precedence, with the new operators slotted in the
    /// way C has them: `?:` lowest, then `||`, `&&`, comparisons, `+ -`,
    /// `* / %`, unary, `^`.
    #[test]
    fn precedence_and_associativity() {
        assert_eq!(ev("1 + 2 * 3", 0.0), 7.0);
        assert_eq!(ev("2 ^ 3 ^ 2", 0.0), 512.0);
        assert_eq!(ev("-2 ^ 2", 0.0), -4.0);
        assert_eq!(ev("1 < 2 == 1", 0.0), 1.0);
        assert_eq!(ev("1 || 0 && 0", 0.0), 1.0);
        assert_eq!(ev("0 ? 1 : 0 ? 2 : 3", 0.0), 3.0);
        assert_eq!(ev("1 ? 0 ? 5 : 6 : 7", 0.0), 6.0);
        assert_eq!(ev("!0 + !5", 0.0), 1.0);
        assert_eq!(ev("-7 % 3", 0.0), 2.0);
        assert_eq!(ev("(-2) ^ 3", 0.0), -8.0);
        assert!(ev("(-2) ^ 0.5", 0.0).is_nan());
    }

    #[test]
    fn variables_lines_and_comments() {
        assert_eq!(ev("x + y", 0.0), 31.0);
        assert_eq!(ev("width * height", 0.0), 5000.0);
        assert_eq!(ev("frame + t + $F", 0.0), 21.0);
        assert_eq!(ev("k = r * 2\nk + 1", 3.0), 7.0);
        assert_eq!(ev("k = 2; k = k * k; k", 0.0), 4.0);
        assert_eq!(ev("r = 5 # shadow the input\nr", 1.0), 5.0);
        // A line ending in an operator goes on; one starting with one is a
        // new statement.
        assert_eq!(ev("1 +\n2", 0.0), 3.0);
        assert_eq!(ev("k = 1\n-k", 0.0), -1.0);
        assert_eq!(ev("clamp(r, 0, 1) + mix(0, 10, 0.5) + lerp(0, 10, 0.1)", 2.0), 7.0);
        assert_eq!(ev("smoothstep(0, 1, 0.5)", 0.0), 0.5);
        assert_eq!(ev("1e-3 * 1000 + .5", 0.0), 1.5);
    }

    /// Errors say what and where, and a broken field never panics.
    #[test]
    fn errors_are_located() {
        let cases = [
            ("r == inf 0: 1 : r", "an operator or the end of the statement"),
            ("r ==", "a value"),
            ("sin(r, g)", "takes 1 argument"),
            ("foo(r)", "no function called `foo`"),
            ("q + 1", "no variable called `q`"),
            ("r ? 1", "expected `:`"),
            ("r @ 2", "not part of the language"),
            ("(r", "expected `)`"),
            ("1e40", "too big"),
        ];
        for (src, want) in cases {
            let err = parse(src).expect_err(src);
            assert!(err.msg.contains(want), "{src}: {:?}", err.msg);
            assert!(err.at <= src.len());
            let _ = err.describe(src);
        }
        assert!(check("").is_none() && check("   ").is_none());
        assert_eq!(check("r == inf 0: 1 : r").map(|e| e.at), Some(9));
    }

    /// Text is the topology; nothing else is.
    #[test]
    fn only_the_text_is_topology() {
        let a = ExpressionParams { exprs: ["r * 2".into(), String::new(), String::new(), String::new()], time: 0.0 };
        let mut b = a.clone();
        b.time = 99.0;
        assert_eq!(a.topo_tag(), b.topo_tag());
        b.exprs[0] = "r * 3".into();
        assert_ne!(a.topo_tag(), b.topo_tag());
        // An empty node is an identity.
        let env = Env { rgba: [0.1, 0.2, 0.3, 0.4], ..Default::default() };
        assert_eq!(ExpressionParams::default().apply(&env), env.rgba);
        // A broken field passes its channel through; the good ones still act.
        let p = ExpressionParams {
            exprs: ["r +".into(), "1".into(), String::new(), String::new()],
            time: 0.0,
        };
        assert_eq!(p.apply(&env), [0.1, 1.0, 0.3, 0.4]);
    }
}
