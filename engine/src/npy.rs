//! Reading numpy/JAX checkpoints — `.npy` files whose payload is a pickled
//! tree of arrays.
//!
//! Why this exists at all: a downloaded model has to be USABLE. The RIFE
//! lane's answer was a python script you ran once yourself, which is fine
//! for the person who wrote it and useless for everyone else — and this
//! machine has neither torch nor numpy installed, which is exactly the
//! situation a new user is in. So the checkpoint is parsed here, in Rust,
//! straight out of the file the Models panel downloaded.
//!
//! This is NOT a general pickle implementation and must never become one.
//! Unpickling arbitrary data is a remote-code-execution format by design;
//! what runs here is a small stack machine over the opcode subset numpy
//! emits, with exactly two callables allowed
//! (`numpy.core.multiarray._reconstruct` and `numpy.dtype`) and every other
//! `GLOBAL` refused by name. Nothing is ever executed, imported or looked
//! up dynamically.
//!
//! The result is a flat `name -> Tensor` map, where names are the nested
//! dict keys joined with `/` — for a Haiku checkpoint that reproduces the
//! module paths the upstream code uses (`params/tapir/~/conv/w`).

/// Dense row-major tensor, PyTorch dim order.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }
}
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::Path;

/// A parsed checkpoint: every float array in the tree, by path.
#[derive(Debug, Default)]
pub struct Checkpoint {
    pub tensors: HashMap<String, Tensor>,
}

impl Checkpoint {
    /// Fetch a tensor, with an error that names what was missing and offers
    /// the nearest keys — a renamed layer is otherwise a very quiet bug.
    pub fn get(&self, name: &str) -> Result<&Tensor> {
        if let Some(t) = self.tensors.get(name) {
            return Ok(t);
        }
        let stem = name.rsplit('/').next().unwrap_or(name);
        let mut near: Vec<&str> = self
            .tensors
            .keys()
            .filter(|k| k.contains(stem))
            .map(String::as_str)
            .take(6)
            .collect();
        near.sort_unstable();
        if near.is_empty() {
            bail!("checkpoint has no tensor {name:?}");
        }
        bail!("checkpoint has no tensor {name:?}; near matches: {near:?}");
    }

    /// Same, asserting the shape — the cheapest guard against a checkpoint
    /// whose layout moved under a port.
    pub fn get_shaped(&self, name: &str, dims: &[usize]) -> Result<&Tensor> {
        let t = self.get(name)?;
        if t.dims != dims {
            bail!("{name}: expected shape {dims:?}, checkpoint has {:?}", t.dims);
        }
        Ok(t)
    }

    /// Every key, sorted — for `models inspect` and for error messages.
    pub fn names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        v.sort_unstable();
        v
    }

    pub fn total_params(&self) -> usize {
        self.tensors.values().map(|t| t.numel()).sum()
    }
}

/// Read a `.npy` holding a pickled tree of arrays.
pub fn read(path: &Path) -> Result<Checkpoint> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    parse(&bytes).with_context(|| format!("parse {}", path.display()))
}

/// Read one `.npy` out of an `.npz` (`np.savez`'s STORED zip) — the same
/// restricted reader on the entry's bytes. A TensorLayer checkpoint keeps
/// its parameter list in `params.npy`, keyed `/0`, `/1`, ... in order.
pub fn read_npz(path: &Path, entry: &str) -> Result<Checkpoint> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let zip = crate::torchpt::Zip::open(&bytes).with_context(|| format!("open {}", path.display()))?;
    parse(zip.get(entry)?).with_context(|| format!("parse {entry} in {}", path.display()))
}

/// Parse `.npy` bytes. Only the object-array form (`descr: '|O'`) is
/// supported — that is what `np.save` of a dict produces, and a plain
/// numeric `.npy` would be a single unnamed array with nothing to key on.
pub fn parse(bytes: &[u8]) -> Result<Checkpoint> {
    if bytes.len() < 12 || &bytes[..6] != b"\x93NUMPY" {
        bail!("not a .npy file (bad magic)");
    }
    let (major, _minor) = (bytes[6], bytes[7]);
    // v1 has a 2-byte header length, v2+ a 4-byte one.
    let (hdr_len, hdr_at): (usize, usize) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        v => bail!("npy version {v} is not supported"),
    };
    let hdr_end = hdr_at
        .checked_add(hdr_len)
        .filter(|e| *e <= bytes.len())
        .context("npy header runs past the end of the file")?;
    let header =
        std::str::from_utf8(&bytes[hdr_at..hdr_end]).context("npy header is not utf-8")?;
    if !header.contains("'|O'") && !header.contains("\"|O\"") {
        bail!(
            "this .npy holds a plain array, not a pickled tree of them (header: {})",
            header.trim()
        );
    }
    let mut vm = Unpickler::new(&bytes[hdr_end..]);
    let root = vm.run()?;
    let mut out = Checkpoint::default();
    flatten(String::new(), &root, &mut out.tensors);
    if out.tensors.is_empty() {
        bail!("no arrays found in the checkpoint");
    }
    Ok(out)
}

/// Walk the dict tree, joining keys with `/`.
fn flatten(prefix: String, v: &Value, out: &mut HashMap<String, Tensor>) {
    match v {
        Value::Dict(entries) => {
            for (k, val) in entries {
                let key = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}/{k}")
                };
                flatten(key, val, out);
            }
        }
        Value::List(items) => {
            for (i, val) in items.iter().enumerate() {
                flatten(format!("{prefix}/{i}"), val, out);
            }
        }
        Value::Array(t) => {
            out.insert(prefix, t.clone());
        }
        // Scalars, None and bookkeeping the checkpoint carries alongside the
        // weights: not tensors, nothing to keep.
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// The pickle subset
// ---------------------------------------------------------------------------

/// Everything the machine can hold. Deliberately small: if a checkpoint
/// needs more than this, that is a decision to make deliberately rather
/// than a gap to fill by widening a pickle interpreter.
#[derive(Debug, Clone)]
enum Value {
    None,
    Bool(bool),
    Int(i64),
    Str(String),
    Bytes(Vec<u8>),
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Vec<(String, Value)>),
    Array(Tensor),
    /// `numpy.dtype(...)` — carried only so `_reconstruct`/BUILD can read
    /// the element type and byte order back off the stack.
    Dtype { kind: String, little: bool },
    /// One of the two allowed callables, waiting for its REDUCE.
    Callable(Call),
    /// A freshly `_reconstruct`ed ndarray before BUILD fills it in.
    RawArray,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Call {
    Reconstruct,
    Dtype,
    /// `_codecs.encode(text, "latin1")`: how Python 3 writes a BYTES
    /// object into a protocol-2 pickle (a str of code points 0..=255).
    /// Pure data in, bytes out — the one non-numpy name the reader takes,
    /// and only with that one encoding.
    Latin1,
}

struct Unpickler<'a> {
    buf: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    /// Positions of MARK objects in `stack`.
    marks: Vec<usize>,
    memo: HashMap<u32, Value>,
}

impl<'a> Unpickler<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            pos: 0,
            stack: Vec::new(),
            marks: Vec::new(),
            memo: HashMap::new(),
        }
    }

    fn byte(&mut self) -> Result<u8> {
        let b = *self.buf.get(self.pos).context("pickle ended mid-opcode")?;
        self.pos += 1;
        Ok(b)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("pickle length overflow")?;
        if end > self.buf.len() {
            bail!("pickle ended mid-value");
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u8v(&mut self) -> Result<usize> {
        Ok(self.byte()? as usize)
    }

    fn u32v(&mut self) -> Result<usize> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
    }

    fn u64v(&mut self) -> Result<usize> {
        let b = self.take(8)?;
        u64::from_le_bytes(b.try_into().unwrap())
            .try_into()
            .context("pickle value too large for this machine")
    }

    fn pop(&mut self) -> Result<Value> {
        self.stack.pop().context("pickle stack underflow")
    }

    /// Everything pushed since the last MARK.
    fn pop_mark(&mut self) -> Result<Vec<Value>> {
        let at = self.marks.pop().context("pickle MARK underflow")?;
        if at > self.stack.len() {
            bail!("pickle MARK past the stack");
        }
        Ok(self.stack.split_off(at))
    }

    fn run(&mut self) -> Result<Value> {
        loop {
            let op = self.byte()?;
            match op {
                b'\x80' => {
                    // PROTO
                    let v = self.byte()?;
                    if v > 5 {
                        bail!("pickle protocol {v} is newer than this reader");
                    }
                }
                b'.' => return self.pop(), // STOP
                b'(' => self.marks.push(self.stack.len()), // MARK
                b'N' => self.stack.push(Value::None),
                b'\x88' => self.stack.push(Value::Bool(true)),
                b'\x89' => self.stack.push(Value::Bool(false)),
                b'J' => {
                    // BININT
                    let b = self.take(4)?;
                    self.stack
                        .push(Value::Int(i32::from_le_bytes(b.try_into().unwrap()) as i64));
                }
                b'K' => {
                    let v = self.byte()?;
                    self.stack.push(Value::Int(v as i64));
                }
                b'M' => {
                    let b = self.take(2)?;
                    self.stack
                        .push(Value::Int(u16::from_le_bytes([b[0], b[1]]) as i64));
                }
                b'\x8a' => {
                    // LONG1 — little-endian signed, arbitrary length. Only
                    // small ones can appear here (shapes, versions).
                    let n = self.u8v()?;
                    let b = self.take(n)?;
                    let mut v: i64 = 0;
                    for (i, byte) in b.iter().enumerate().take(8) {
                        v |= (*byte as i64) << (8 * i);
                    }
                    if n > 0 && n <= 8 && b[n - 1] & 0x80 != 0 && n < 8 {
                        v -= 1i64 << (8 * n);
                    }
                    self.stack.push(Value::Int(v));
                }
                b'X' => {
                    // BINUNICODE
                    let n = self.u32v()?;
                    let s = std::str::from_utf8(self.take(n)?)
                        .context("pickle string is not utf-8")?;
                    self.stack.push(Value::Str(s.to_string()));
                }
                b'\x8c' => {
                    // SHORT_BINUNICODE
                    let n = self.u8v()?;
                    let s = std::str::from_utf8(self.take(n)?)
                        .context("pickle string is not utf-8")?;
                    self.stack.push(Value::Str(s.to_string()));
                }
                b'\x8d' => {
                    // BINUNICODE8
                    let n = self.u64v()?;
                    let s = std::str::from_utf8(self.take(n)?)
                        .context("pickle string is not utf-8")?;
                    self.stack.push(Value::Str(s.to_string()));
                }
                b'C' => {
                    // SHORT_BINBYTES
                    let n = self.u8v()?;
                    let b = self.take(n)?.to_vec();
                    self.stack.push(Value::Bytes(b));
                }
                b'B' => {
                    // BINBYTES
                    let n = self.u32v()?;
                    let b = self.take(n)?.to_vec();
                    self.stack.push(Value::Bytes(b));
                }
                b'\x8e' => {
                    // BINBYTES8
                    let n = self.u64v()?;
                    let b = self.take(n)?.to_vec();
                    self.stack.push(Value::Bytes(b));
                }
                b')' => self.stack.push(Value::Tuple(Vec::new())),
                b'\x85' | b'\x86' | b'\x87' => {
                    // TUPLE1 / TUPLE2 / TUPLE3
                    let n = (op - b'\x84') as usize;
                    let at = self
                        .stack
                        .len()
                        .checked_sub(n)
                        .context("pickle stack underflow building a tuple")?;
                    let items = self.stack.split_off(at);
                    self.stack.push(Value::Tuple(items));
                }
                b't' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::Tuple(items));
                }
                b']' => self.stack.push(Value::List(Vec::new())),
                b'l' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::List(items));
                }
                b'e' => {
                    // APPENDS
                    let items = self.pop_mark()?;
                    match self.stack.last_mut() {
                        Some(Value::List(l)) => l.extend(items),
                        _ => bail!("pickle APPENDS onto a non-list"),
                    }
                }
                b'a' => {
                    // APPEND
                    let item = self.pop()?;
                    match self.stack.last_mut() {
                        Some(Value::List(l)) => l.push(item),
                        _ => bail!("pickle APPEND onto a non-list"),
                    }
                }
                b'}' => self.stack.push(Value::Dict(Vec::new())),
                b'd' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::Dict(pairs(items)?));
                }
                b's' => {
                    // SETITEM
                    let v = self.pop()?;
                    let k = self.pop()?;
                    let k = key_of(&k)?;
                    match self.stack.last_mut() {
                        Some(Value::Dict(d)) => d.push((k, v)),
                        _ => bail!("pickle SETITEM onto a non-dict"),
                    }
                }
                b'u' => {
                    // SETITEMS
                    let items = self.pop_mark()?;
                    let kv = pairs(items)?;
                    match self.stack.last_mut() {
                        Some(Value::Dict(d)) => d.extend(kv),
                        _ => bail!("pickle SETITEMS onto a non-dict"),
                    }
                }
                b'q' => {
                    // BINPUT
                    let i = self.u8v()? as u32;
                    let v = self.stack.last().context("pickle BINPUT on empty stack")?;
                    self.memo.insert(i, v.clone());
                }
                b'r' => {
                    // LONG_BINPUT
                    let i = self.u32v()? as u32;
                    let v = self.stack.last().context("pickle LONG_BINPUT on empty stack")?;
                    self.memo.insert(i, v.clone());
                }
                b'\x94' => {
                    // MEMOIZE
                    let i = self.memo.len() as u32;
                    let v = self.stack.last().context("pickle MEMOIZE on empty stack")?;
                    self.memo.insert(i, v.clone());
                }
                b'h' => {
                    let i = self.u8v()? as u32;
                    let v = self.memo.get(&i).context("pickle BINGET of an unset memo")?;
                    self.stack.push(v.clone());
                }
                b'j' => {
                    let i = self.u32v()? as u32;
                    let v = self
                        .memo
                        .get(&i)
                        .context("pickle LONG_BINGET of an unset memo")?;
                    self.stack.push(v.clone());
                }
                b'c' => {
                    // GLOBAL — the security boundary. Two names, no others,
                    // and neither is ever actually called: REDUCE below
                    // interprets them.
                    let module = self.line()?;
                    let name = self.line()?;
                    let call = match (module.as_str(), name.as_str()) {
                        ("numpy.core.multiarray", "_reconstruct")
                        | ("numpy._core.multiarray", "_reconstruct") => Call::Reconstruct,
                        ("numpy", "dtype") => Call::Dtype,
                        ("_codecs", "encode") => Call::Latin1,
                        // `numpy.ndarray` and `numpy.float32`-style type
                        // objects appear as bare arguments; they carry no
                        // behaviour, so a placeholder is enough.
                        ("numpy", _) => {
                            self.stack.push(Value::Str(format!("numpy.{name}")));
                            continue;
                        }
                        _ => bail!(
                            "checkpoint wants {module}.{name}, which this reader refuses \
                             to interpret — only numpy arrays are allowed here"
                        ),
                    };
                    self.stack.push(Value::Callable(call));
                }
                b'R' => {
                    // REDUCE
                    let args = self.pop()?;
                    let f = self.pop()?;
                    let Value::Callable(call) = f else {
                        bail!("pickle REDUCE on something that is not an allowed callable");
                    };
                    let v = match call {
                        Call::Reconstruct => Value::RawArray,
                        Call::Dtype => dtype_from(&args)?,
                        Call::Latin1 => latin1_bytes(&args)?,
                    };
                    self.stack.push(v);
                }
                b'b' => {
                    // BUILD — where an ndarray actually gets its contents.
                    let state = self.pop()?;
                    let target = self.pop()?;
                    let built = match target {
                        Value::RawArray => array_from_state(&state)?,
                        // dtype's BUILD carries byte order and flags we
                        // already read in `dtype_from`.
                        other => other,
                    };
                    self.stack.push(built);
                }
                b'0' => {
                    self.pop()?;
                }
                other => bail!(
                    "pickle opcode {:?} (0x{other:02x}) at {} is outside the subset numpy emits",
                    other as char,
                    self.pos - 1
                ),
            }
        }
    }

    /// A newline-terminated ASCII token (GLOBAL's module and name).
    fn line(&mut self) -> Result<String> {
        let start = self.pos;
        while self.pos < self.buf.len() && self.buf[self.pos] != b'\n' {
            self.pos += 1;
        }
        if self.pos >= self.buf.len() {
            bail!("pickle ended mid-name");
        }
        let s = std::str::from_utf8(&self.buf[start..self.pos])
            .context("pickle name is not utf-8")?
            .to_string();
        self.pos += 1;
        Ok(s)
    }
}

fn pairs(items: Vec<Value>) -> Result<Vec<(String, Value)>> {
    if items.len() % 2 != 0 {
        bail!("pickle dict has an odd number of items");
    }
    let mut out = Vec::with_capacity(items.len() / 2);
    let mut it = items.into_iter();
    while let (Some(k), Some(v)) = (it.next(), it.next()) {
        out.push((key_of(&k)?, v));
    }
    Ok(out)
}

fn key_of(v: &Value) -> Result<String> {
    match v {
        Value::Str(s) => Ok(s.clone()),
        Value::Int(i) => Ok(i.to_string()),
        Value::Bytes(b) => Ok(String::from_utf8_lossy(b).into_owned()),
        other => bail!("checkpoint dict key is not a string: {other:?}"),
    }
}

/// `numpy.dtype(name, align, copy)` — only the name and the byte order that
/// its BUILD would set matter here.
/// `_codecs.encode(text, "latin1")`, and nothing else it could be asked:
/// the text's code points as bytes. Any other encoding, or a code point
/// past 255, is refused rather than guessed at.
fn latin1_bytes(args: &Value) -> Result<Value> {
    let Value::Tuple(a) = args else { bail!("_codecs.encode without an argument tuple") };
    let (Some(Value::Str(text)), Some(Value::Str(enc))) = (a.first(), a.get(1)) else {
        bail!("_codecs.encode wants (text, encoding)");
    };
    if !matches!(enc.as_str(), "latin1" | "latin-1" | "iso-8859-1") || a.len() != 2 {
        bail!("_codecs.encode with encoding {enc:?}: only latin1 is how a pickle carries bytes");
    }
    let mut out = Vec::with_capacity(text.len());
    for c in text.chars() {
        let v = c as u32;
        if v > 255 {
            bail!("_codecs.encode latin1 of a code point past 255");
        }
        out.push(v as u8);
    }
    Ok(Value::Bytes(out))
}

fn dtype_from(args: &Value) -> Result<Value> {
    let Value::Tuple(items) = args else {
        bail!("numpy.dtype called with something other than a tuple");
    };
    let kind = match items.first() {
        Some(Value::Str(s)) => s.clone(),
        _ => bail!("numpy.dtype without a type string"),
    };
    Ok(Value::Dtype { kind, little: true })
}

/// `ndarray.__setstate__((version, shape, dtype, is_fortran, data))`.
fn array_from_state(state: &Value) -> Result<Value> {
    let Value::Tuple(items) = state else {
        bail!("ndarray state is not a tuple");
    };
    if items.len() < 5 {
        bail!("ndarray state has {} entries, expected 5", items.len());
    }
    let dims: Vec<usize> = match &items[1] {
        Value::Tuple(d) => d
            .iter()
            .map(|v| match v {
                Value::Int(i) if *i >= 0 => Ok(*i as usize),
                other => bail!("array dimension is not a non-negative int: {other:?}"),
            })
            .collect::<Result<_>>()?,
        other => bail!("array shape is not a tuple: {other:?}"),
    };
    let (kind, little) = match &items[2] {
        Value::Dtype { kind, little } => (kind.as_str(), *little),
        other => bail!("array dtype is not a dtype: {other:?}"),
    };
    let fortran = matches!(items[3], Value::Bool(true));
    if fortran {
        bail!("fortran-ordered arrays are not supported");
    }
    let raw = match &items[4] {
        Value::Bytes(b) => b.as_slice(),
        // An object array inside the tree (a list of sub-arrays) is not a
        // tensor; the caller's flatten step drops it.
        Value::List(_) => return Ok(items[4].clone()),
        other => bail!("array data is not bytes: {other:?}"),
    };
    let n: usize = dims.iter().product();
    let data = decode(kind, little, raw, n)?;
    Ok(Value::Array(Tensor { dims, data }))
}

/// Numeric element types a checkpoint actually uses, widened to f32 — the
/// precision everything downstream works in.
fn decode(kind: &str, little: bool, raw: &[u8], n: usize) -> Result<Vec<f32>> {
    // Names arrive either as a dtype string ("f4") or, for a zero-size
    // array, with the byte-order prefix still attached.
    let k = kind.trim_start_matches(['<', '>', '|', '=']);
    let big = kind.starts_with('>') || !little;
    let width = match k {
        "f4" | "i4" | "u4" => 4,
        "f2" => 2,
        "f8" | "i8" | "u8" => 8,
        "i1" | "u1" | "b1" => 1,
        "i2" | "u2" => 2,
        other => bail!("checkpoint holds dtype {other:?}, which this reader does not decode"),
    };
    if raw.len() < n * width {
        bail!(
            "array data is {} bytes, expected {} for {n} × {k}",
            raw.len(),
            n * width
        );
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let b = &raw[i * width..(i + 1) * width];
        let mut le = [0u8; 8];
        if big {
            for (j, v) in b.iter().rev().enumerate() {
                le[j] = *v;
            }
        } else {
            le[..width].copy_from_slice(b);
        }
        let v = match k {
            "f4" => f32::from_le_bytes(le[..4].try_into().unwrap()),
            "f2" => half::f16::from_le_bytes(le[..2].try_into().unwrap()).to_f32(),
            "f8" => f64::from_le_bytes(le) as f32,
            "i4" => i32::from_le_bytes(le[..4].try_into().unwrap()) as f32,
            "u4" => u32::from_le_bytes(le[..4].try_into().unwrap()) as f32,
            "i8" => i64::from_le_bytes(le) as f32,
            "u8" => u64::from_le_bytes(le) as f32,
            "i2" => i16::from_le_bytes(le[..2].try_into().unwrap()) as f32,
            "u2" => u16::from_le_bytes(le[..2].try_into().unwrap()) as f32,
            "i1" => le[0] as i8 as f32,
            "u1" | "b1" => le[0] as f32,
            _ => unreachable!(),
        };
        out.push(v);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the `.npy` wrapper around a pickle body.
    fn npy(body: &[u8]) -> Vec<u8> {
        let header = b"{'descr': '|O', 'fortran_order': False, 'shape': (), }";
        let mut pad = header.to_vec();
        while (10 + pad.len() + 1) % 64 != 0 {
            pad.push(b' ');
        }
        pad.push(b'\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend((pad.len() as u16).to_le_bytes());
        out.extend(pad);
        out.extend(body);
        out
    }

    /// A pickled `{"a": {"w": <f32 array>}}` built by hand, so the reader is
    /// tested against bytes rather than against itself.
    fn sample(dims: &[usize], vals: &[f32]) -> Vec<u8> {
        let mut p = vec![0x80, 0x03]; // PROTO 3
        p.push(b'}'); // EMPTY_DICT   -> outer
        p.push(b'('); // MARK
        p.extend(str_op("a"));
        p.push(b'}'); // inner dict
        p.push(b'(');
        p.extend(str_op("w"));
        // _reconstruct(ndarray, (0,), b'b') -> RawArray
        p.extend(b"cnumpy.core.multiarray\n_reconstruct\n");
        p.extend(b"cnumpy\nndarray\n");
        p.push(b'K');
        p.push(0);
        p.push(0x85); // TUPLE1
        p.push(b'C');
        p.push(1);
        p.push(b'b');
        p.push(0x87); // TUPLE3
        p.push(b'R'); // REDUCE
                      // __setstate__((1, dims, dtype('f4'), False, raw))
        p.push(b'(');
        p.push(b'K');
        p.push(1);
        p.push(b'(');
        for d in dims {
            p.push(b'K');
            p.push(*d as u8);
        }
        p.push(b't'); // shape tuple
        p.extend(b"cnumpy\ndtype\n");
        p.extend(str_op("f4"));
        p.push(0x89); // False
        p.push(0x88); // True
        p.push(0x87); // TUPLE3
        p.push(b'R');
        p.push(b'('); // dtype's own BUILD state
        p.push(b'K');
        p.push(3);
        p.extend(str_op("<"));
        p.push(b'N');
        p.push(b'N');
        p.push(b'N');
        p.push(b'J');
        p.extend((-1i32).to_le_bytes());
        p.push(b'J');
        p.extend((-1i32).to_le_bytes());
        p.push(b'K');
        p.push(0);
        p.push(b't');
        p.push(b'b'); // BUILD on the dtype
        p.push(0x89); // fortran = False
        let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        p.push(b'B');
        p.extend((raw.len() as u32).to_le_bytes());
        p.extend(raw);
        p.push(b't'); // state tuple
        p.push(b'b'); // BUILD on the array
        p.push(b'u'); // SETITEMS -> inner
        p.push(b'u'); // SETITEMS -> outer
        p.push(b'.'); // STOP
        p
    }

    fn str_op(s: &str) -> Vec<u8> {
        let mut v = vec![b'X'];
        v.extend((s.len() as u32).to_le_bytes());
        v.extend(s.as_bytes());
        v
    }

    #[test]
    fn reads_a_nested_dict_of_arrays() {
        let vals = [1.0f32, -2.5, 3.25, 4.0, 0.5, -0.25];
        let ck = parse(&npy(&sample(&[2, 3], &vals))).unwrap();
        assert_eq!(ck.names(), vec!["a/w"]);
        let t = ck.get("a/w").unwrap();
        assert_eq!(t.dims, vec![2, 3]);
        assert_eq!(t.data, vals);
        assert_eq!(ck.total_params(), 6);
    }

    #[test]
    fn shape_mismatch_is_named() {
        let ck = parse(&npy(&sample(&[2, 3], &[0.0; 6]))).unwrap();
        let err = ck.get_shaped("a/w", &[3, 2]).unwrap_err().to_string();
        assert!(err.contains("expected shape [3, 2]"), "{err}");
        assert!(err.contains("[2, 3]"), "{err}");
    }

    #[test]
    fn a_missing_tensor_suggests_near_matches() {
        let ck = parse(&npy(&sample(&[1], &[0.0]))).unwrap();
        let err = ck.get("b/w").unwrap_err().to_string();
        assert!(err.contains("a/w"), "should have offered the near match: {err}");
    }

    /// The security boundary: a pickle naming anything but numpy must be
    /// refused by name, not quietly skipped.
    #[test]
    fn a_pickle_that_calls_out_of_numpy_is_refused() {
        let mut p = vec![0x80, 0x03];
        p.extend(b"cos\nsystem\n");
        p.extend(str_op("rm -rf /"));
        p.push(0x85);
        p.push(b'R');
        p.push(b'.');
        let err = parse(&npy(&p)).unwrap_err().to_string();
        assert!(err.contains("os.system"), "{err}");
        assert!(err.contains("refuses"), "{err}");
    }

    /// Python 3's protocol-2 bytes: `_codecs.encode(text, "latin1")` is
    /// read as those bytes, and only that encoding is accepted.
    #[test]
    fn latin1_bytes_are_read_and_nothing_else_from_codecs() {
        let call = |enc: &str| {
            let mut a = vec![
                Value::Str("\u{0}\u{ff}A".into()),
                Value::Str(enc.into()),
            ];
            latin1_bytes(&Value::Tuple(std::mem::take(&mut a)))
        };
        match call("latin1").unwrap() {
            Value::Bytes(b) => assert_eq!(b, vec![0, 255, b'A']),
            v => panic!("{v:?}"),
        }
        assert!(call("utf-8").is_err());
        assert!(latin1_bytes(&Value::Tuple(vec![Value::Str("\u{100}".into()), Value::Str("latin1".into())])).is_err());
        // Other names in the module stay refused.
        let mut p = vec![0x80, 0x02];
        p.extend(b"c_codecs\ndecode\n");
        p.push(b'.');
        let err = parse(&npy(&p)).unwrap_err().to_string();
        assert!(err.contains("refuses"), "{err}");
    }

    #[test]
    fn an_unknown_opcode_says_so_rather_than_guessing() {
        let mut p = vec![0x80, 0x03];
        p.push(b'\x01'); // POP_MARK, outside the subset
        let err = parse(&npy(&p)).unwrap_err().to_string();
        assert!(err.contains("outside the subset"), "{err}");
    }

    #[test]
    fn a_plain_numeric_npy_is_rejected_clearly() {
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        let hdr = b"{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }          \n";
        out.extend((hdr.len() as u16).to_le_bytes());
        out.extend(hdr);
        out.extend(1.0f32.to_le_bytes());
        out.extend(2.0f32.to_le_bytes());
        let err = parse(&out).unwrap_err().to_string();
        assert!(err.contains("plain array"), "{err}");
    }

    #[test]
    fn truncated_input_does_not_panic() {
        let full = npy(&sample(&[2, 2], &[1.0, 2.0, 3.0, 4.0]));
        for cut in [0, 1, 8, 64, 130, 150, full.len() - 1] {
            let _ = parse(&full[..cut.min(full.len())]);
        }
    }
}
