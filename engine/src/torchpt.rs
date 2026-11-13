//! Reading PyTorch state dicts — the `torch.save` zip a `.pkl` / `.pt`
//! checkpoint is — straight into `name -> Tensor`, with no Python.
//!
//! Same contract as [`crate::npy`], and the same warning: this is NOT a
//! general unpickler and must never become one. Pickle is a
//! remote-code-execution format by design. What runs here is a stack
//! machine over the opcode subset `torch.save` emits for a state dict,
//! with exactly three names allowed — `collections.OrderedDict`,
//! `torch._utils._rebuild_tensor_v2` and the `torch.*Storage` type
//! objects — every other `GLOBAL` refused by name, and none of them ever
//! called: REDUCE interprets them. Tensor bytes come out of the archive's
//! `data/<key>` entries, which are raw little-endian arrays.
//!
//! The archive itself is read here too, and only the shape torch writes:
//! STORED entries (torch never compresses), no zip64. A compressed or
//! zip64 archive is refused with a message, not guessed at.

use crate::npy::Tensor;
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::Path;

/// Read a `torch.save` state dict from disk.
pub fn read(path: &Path) -> Result<HashMap<String, Tensor>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    parse(&bytes).with_context(|| format!("parse {}", path.display()))
}

/// Parse a `torch.save` state dict held in memory.
pub fn parse(bytes: &[u8]) -> Result<HashMap<String, Tensor>> {
    let zip = Zip::open(bytes)?;
    let pkl_name = zip
        .names()
        .find(|n| n.ends_with("/data.pkl") || *n == "data.pkl")
        .context("not a torch checkpoint: no data.pkl in the archive")?
        .to_string();
    let prefix = pkl_name.trim_end_matches("data.pkl").to_string();
    let pkl = zip.get(&pkl_name)?;
    let mut u = Unpickler { buf: pkl, pos: 0, stack: vec![], marks: vec![], memo: HashMap::new() };
    let root = u.run()?;
    let Value::Dict(mut items) = root else {
        bail!("the checkpoint is not a state dict (a dict of tensors at the top)");
    };
    // A training checkpoint wraps the state dict one level down, under
    // `params_ema` (the averaged weights, which are the ones published
    // results come from) or `params` — Real-ESRGAN's releases do.
    if !items.iter().any(|(_, v)| matches!(v, Value::Tensor(_))) {
        for wrap in ["params_ema", "params"] {
            let inner = items.iter().position(|(k, v)| {
                matches!((k, v), (Value::Str(s), Value::Dict(_)) if s == wrap)
            });
            if let Some(i) = inner {
                let Value::Dict(d) = items.swap_remove(i).1 else { unreachable!() };
                items = d;
                break;
            }
        }
    }
    let mut out = HashMap::new();
    for (k, v) in items {
        let Value::Str(name) = k else { continue };
        let Value::Tensor(t) = v else { continue };
        let t = t.materialise(&zip, &prefix).with_context(|| format!("tensor {name}"))?;
        out.insert(name, t);
    }
    if out.is_empty() {
        bail!("the state dict holds no tensors");
    }
    Ok(out)
}

// ------------------------------------------------------------------ the zip

/// A STORED zip (also what `np.savez` writes: `npy::read_npz`).
pub(crate) struct Zip<'a> {
    bytes: &'a [u8],
    /// name -> (data offset, size)
    entries: Vec<(String, usize, usize)>,
}

fn u16_at(b: &[u8], at: usize) -> Result<usize> {
    let s = b.get(at..at + 2).context("archive truncated")?;
    Ok(u16::from_le_bytes([s[0], s[1]]) as usize)
}

fn u32_at(b: &[u8], at: usize) -> Result<usize> {
    let s = b.get(at..at + 4).context("archive truncated")?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize)
}

impl<'a> Zip<'a> {
    pub(crate) fn open(bytes: &'a [u8]) -> Result<Self> {
        // The end-of-central-directory record: the last PK\5\6 within the
        // trailing comment's reach.
        let floor = bytes.len().saturating_sub(22 + 0xffff);
        let eocd = (floor..bytes.len().saturating_sub(21))
            .rev()
            .find(|&i| bytes[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
            .context("not a zip archive (torch checkpoints and npz files are)")?;
        let count = u16_at(bytes, eocd + 10)?;
        let mut at = u32_at(bytes, eocd + 16)?;
        if count == 0xffff || at == 0xffff_ffff {
            bail!("a zip64 checkpoint: too large for this reader");
        }
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            if u32_at(bytes, at)? != 0x0201_4b50 {
                bail!("corrupt zip central directory");
            }
            let method = u16_at(bytes, at + 10)?;
            let csize = u32_at(bytes, at + 20)?;
            let size = u32_at(bytes, at + 24)?;
            let nlen = u16_at(bytes, at + 28)?;
            let xlen = u16_at(bytes, at + 30)?;
            let clen = u16_at(bytes, at + 32)?;
            let local = u32_at(bytes, at + 42)?;
            let name = std::str::from_utf8(bytes.get(at + 46..at + 46 + nlen).context("archive truncated")?)
                .context("zip entry name is not utf-8")?
                .to_string();
            if method != 0 || csize != size {
                bail!("{name} is compressed: torch writes STORED entries, refusing to guess");
            }
            if size == 0xffff_ffff || local == 0xffff_ffff {
                bail!("a zip64 checkpoint: too large for this reader");
            }
            if u32_at(bytes, local)? != 0x0403_4b50 {
                bail!("corrupt zip local header for {name}");
            }
            let data = local + 30 + u16_at(bytes, local + 26)? + u16_at(bytes, local + 28)?;
            if data.checked_add(size).is_none_or(|e| e > bytes.len()) {
                bail!("{name} runs past the end of the archive");
            }
            entries.push((name, data, size));
            at += 46 + nlen + xlen + clen;
        }
        Ok(Self { bytes, entries })
    }

    fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(n, _, _)| n.as_str())
    }

    pub(crate) fn get(&self, name: &str) -> Result<&'a [u8]> {
        let (_, at, n) = self
            .entries
            .iter()
            .find(|(e, _, _)| e == name)
            .with_context(|| format!("{name} is missing from the archive"))?;
        Ok(&self.bytes[*at..*at + *n])
    }
}

// --------------------------------------------------------------- the pickle

#[derive(Clone, Copy, Debug, PartialEq)]
enum Dtype {
    F32,
    F16,
}

impl Dtype {
    fn size(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 => 2,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Call {
    OrderedDict,
    Rebuild,
}

#[derive(Clone, Debug)]
struct TensorRef {
    key: String,
    dtype: Dtype,
    offset: usize,
    size: Vec<usize>,
    stride: Vec<usize>,
}

impl TensorRef {
    /// The tensor's elements, contiguous in its own (row-major) order,
    /// whatever strides it was saved with.
    fn materialise(&self, zip: &Zip, prefix: &str) -> Result<Tensor> {
        let raw = zip.get(&format!("{prefix}data/{}", self.key))?;
        let es = self.dtype.size();
        let n: usize = self.size.iter().product();
        let mut data = Vec::with_capacity(n);
        let mut idx = vec![0usize; self.size.len()];
        for _ in 0..n {
            let e = self.offset + idx.iter().zip(&self.stride).map(|(i, s)| i * s).sum::<usize>();
            let b = raw.get(e * es..e * es + es).context("a tensor reads past its storage")?;
            data.push(match self.dtype {
                Dtype::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                Dtype::F16 => half::f16::from_le_bytes([b[0], b[1]]).to_f32(),
            });
            // Odometer, last axis fastest.
            for d in (0..idx.len()).rev() {
                idx[d] += 1;
                if idx[d] < self.size[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Ok(Tensor { dims: self.size.clone(), data })
    }
}

// Bool and Float are only ever skipped over, but a pickle carries them.
#[allow(dead_code)]
#[derive(Clone, Debug)]
enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Callable(Call),
    StorageType(Dtype),
    Storage(String, Dtype),
    Tensor(TensorRef),
}

impl Value {
    fn usize(&self) -> Result<usize> {
        match self {
            Value::Int(i) if *i >= 0 => Ok(*i as usize),
            other => bail!("expected a size, found {other:?}"),
        }
    }

    fn sizes(&self) -> Result<Vec<usize>> {
        match self {
            Value::Tuple(v) | Value::List(v) => v.iter().map(Value::usize).collect(),
            other => bail!("expected a shape, found {other:?}"),
        }
    }
}

struct Unpickler<'a> {
    buf: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    marks: Vec<usize>,
    memo: HashMap<usize, Value>,
}

impl<'a> Unpickler<'a> {
    fn byte(&mut self) -> Result<u8> {
        let b = *self.buf.get(self.pos).context("pickle ended mid-opcode")?;
        self.pos += 1;
        Ok(b)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("pickle length overflow")?;
        let s = self.buf.get(self.pos..end).context("pickle ended mid-value")?;
        self.pos = end;
        Ok(s)
    }

    fn u32v(&mut self) -> Result<usize> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
    }

    fn pop(&mut self) -> Result<Value> {
        self.stack.pop().context("pickle stack underflow")
    }

    fn pop_mark(&mut self) -> Result<Vec<Value>> {
        let at = self.marks.pop().context("pickle MARK underflow")?;
        if at > self.stack.len() {
            bail!("pickle MARK past the stack");
        }
        Ok(self.stack.split_off(at))
    }

    fn line(&mut self) -> Result<String> {
        let start = self.pos;
        while self.pos < self.buf.len() && self.buf[self.pos] != b'\n' {
            self.pos += 1;
        }
        if self.pos >= self.buf.len() {
            bail!("pickle ended mid-name");
        }
        let s = std::str::from_utf8(&self.buf[start..self.pos]).context("pickle name is not utf-8")?.to_string();
        self.pos += 1;
        Ok(s)
    }

    fn string(&mut self, n: usize) -> Result<Value> {
        let s = self.take(n)?;
        Ok(Value::Str(std::str::from_utf8(s).context("pickle string is not utf-8")?.to_string()))
    }

    fn memo_put(&mut self, i: usize) -> Result<()> {
        let v = self.stack.last().context("pickle memo of an empty stack")?.clone();
        self.memo.insert(i, v);
        Ok(())
    }

    fn memo_get(&mut self, i: usize) -> Result<()> {
        let v = self.memo.get(&i).context("pickle get of an unset memo")?.clone();
        self.stack.push(v);
        Ok(())
    }

    fn run(&mut self) -> Result<Value> {
        loop {
            match self.byte()? {
                b'\x80' => {
                    self.byte()?; // PROTO
                }
                b'\x95' => {
                    self.take(8)?; // FRAME: a length hint
                }
                b'.' => return self.pop(),
                b'(' => self.marks.push(self.stack.len()),
                b'N' => self.stack.push(Value::None),
                b'\x88' => self.stack.push(Value::Bool(true)),
                b'\x89' => self.stack.push(Value::Bool(false)),
                b'K' => {
                    let v = self.byte()? as i64;
                    self.stack.push(Value::Int(v));
                }
                b'M' => {
                    let b = self.take(2)?;
                    self.stack.push(Value::Int(u16::from_le_bytes([b[0], b[1]]) as i64));
                }
                b'J' => {
                    let b = self.take(4)?;
                    self.stack.push(Value::Int(i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64));
                }
                b'\x8a' => {
                    // LONG1: little-endian two's complement, up to 8 bytes.
                    let n = self.byte()? as usize;
                    if n > 8 {
                        bail!("pickle integer wider than 64 bits");
                    }
                    let b = self.take(n)?;
                    let mut v: i64 = 0;
                    for (i, x) in b.iter().enumerate() {
                        v |= (*x as i64) << (8 * i);
                    }
                    if n > 0 && n < 8 && b[n - 1] & 0x80 != 0 {
                        v -= 1i64 << (8 * n);
                    }
                    self.stack.push(Value::Int(v));
                }
                b'G' => {
                    let b = self.take(8)?;
                    self.stack.push(Value::Float(f64::from_be_bytes(b.try_into().unwrap())));
                }
                b'X' => {
                    let n = self.u32v()?;
                    let v = self.string(n)?;
                    self.stack.push(v);
                }
                b'\x8c' => {
                    let n = self.byte()? as usize;
                    let v = self.string(n)?;
                    self.stack.push(v);
                }
                b')' => self.stack.push(Value::Tuple(vec![])),
                b't' => {
                    let v = self.pop_mark()?;
                    self.stack.push(Value::Tuple(v));
                }
                b'\x85' | b'\x86' | b'\x87' => {
                    let n = (self.buf[self.pos - 1] - 0x84) as usize;
                    if self.stack.len() < n {
                        bail!("pickle stack underflow");
                    }
                    let v = self.stack.split_off(self.stack.len() - n);
                    self.stack.push(Value::Tuple(v));
                }
                b']' => self.stack.push(Value::List(vec![])),
                b'a' => {
                    let v = self.pop()?;
                    match self.stack.last_mut() {
                        Some(Value::List(l)) => l.push(v),
                        _ => bail!("pickle APPEND to a non-list"),
                    }
                }
                b'e' => {
                    let items = self.pop_mark()?;
                    match self.stack.last_mut() {
                        Some(Value::List(l)) => l.extend(items),
                        _ => bail!("pickle APPENDS to a non-list"),
                    }
                }
                b'}' => self.stack.push(Value::Dict(vec![])),
                b's' => {
                    let v = self.pop()?;
                    let k = self.pop()?;
                    match self.stack.last_mut() {
                        Some(Value::Dict(d)) => d.push((k, v)),
                        _ => bail!("pickle SETITEM on a non-dict"),
                    }
                }
                b'u' => {
                    let items = self.pop_mark()?;
                    if items.len() % 2 != 0 {
                        bail!("pickle SETITEMS with an odd count");
                    }
                    let mut it = items.into_iter();
                    let mut pairs = vec![];
                    while let (Some(k), Some(v)) = (it.next(), it.next()) {
                        pairs.push((k, v));
                    }
                    match self.stack.last_mut() {
                        Some(Value::Dict(d)) => d.extend(pairs),
                        _ => bail!("pickle SETITEMS on a non-dict"),
                    }
                }
                b'q' => {
                    let i = self.byte()? as usize;
                    self.memo_put(i)?;
                }
                b'r' => {
                    let i = self.u32v()?;
                    self.memo_put(i)?;
                }
                b'\x94' => {
                    let i = self.memo.len();
                    self.memo_put(i)?;
                }
                b'h' => {
                    let i = self.byte()? as usize;
                    self.memo_get(i)?;
                }
                b'j' => {
                    let i = self.u32v()?;
                    self.memo_get(i)?;
                }
                b'c' => {
                    // GLOBAL — the security boundary. Nothing named here is
                    // ever imported or called.
                    let module = self.line()?;
                    let name = self.line()?;
                    let v = match (module.as_str(), name.as_str()) {
                        ("collections", "OrderedDict") => Value::Callable(Call::OrderedDict),
                        ("torch._utils", "_rebuild_tensor_v2") => Value::Callable(Call::Rebuild),
                        ("torch", "FloatStorage") => Value::StorageType(Dtype::F32),
                        ("torch", "HalfStorage") => Value::StorageType(Dtype::F16),
                        _ => bail!(
                            "checkpoint wants {module}.{name}, which this reader refuses to \
                             interpret — only float tensors in a state dict are allowed here"
                        ),
                    };
                    self.stack.push(v);
                }
                b'Q' => {
                    // BINPERSID: ('storage', <type>, key, location, numel).
                    let pid = self.pop()?;
                    let Value::Tuple(t) = pid else { bail!("unexpected persistent id {pid:?}") };
                    match t.as_slice() {
                        [Value::Str(tag), Value::StorageType(d), Value::Str(key), ..] if tag == "storage" => {
                            if key.contains('/') || key.contains("..") {
                                bail!("storage key {key:?} is not a plain name");
                            }
                            self.stack.push(Value::Storage(key.clone(), *d));
                        }
                        _ => bail!("unexpected persistent id {t:?}"),
                    }
                }
                b'R' => {
                    let args = self.pop()?;
                    let f = self.pop()?;
                    let v = match (f, args) {
                        (Value::Callable(Call::OrderedDict), _) => Value::Dict(vec![]),
                        (Value::Callable(Call::Rebuild), Value::Tuple(a)) => {
                            let [Value::Storage(key, dtype), off, size, stride, ..] = a.as_slice() else {
                                bail!("unexpected _rebuild_tensor_v2 arguments");
                            };
                            let size = size.sizes()?;
                            let stride = stride.sizes()?;
                            if size.len() != stride.len() {
                                bail!("a tensor's shape and strides disagree");
                            }
                            Value::Tensor(TensorRef {
                                key: key.clone(),
                                dtype: *dtype,
                                offset: off.usize()?,
                                size,
                                stride,
                            })
                        }
                        _ => bail!("pickle REDUCE on something that is not an allowed callable"),
                    };
                    self.stack.push(v);
                }
                b'b' => {
                    // BUILD: an OrderedDict's `_metadata`. Dropped.
                    self.pop()?;
                }
                b'0' => {
                    self.pop()?;
                }
                other => bail!(
                    "pickle opcode 0x{other:02x} at {} is outside the subset torch.save emits",
                    self.pos - 1
                ),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A STORED zip of the given entries, the way torch.save lays one out.
    fn zip_of(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut out = vec![];
        let mut central = vec![];
        for (name, data) in entries {
            let local = out.len() as u32;
            out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
            out.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(data);
            central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
            central.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            central.extend_from_slice(&(data.len() as u32).to_le_bytes());
            central.extend_from_slice(&(data.len() as u32).to_le_bytes());
            central.extend_from_slice(&(name.len() as u16).to_le_bytes());
            central.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            central.extend_from_slice(&local.to_le_bytes());
            central.extend_from_slice(name.as_bytes());
        }
        let at = out.len() as u32;
        out.extend_from_slice(&central);
        out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        out.extend_from_slice(&[0, 0, 0, 0]);
        out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        out.extend_from_slice(&(central.len() as u32).to_le_bytes());
        out.extend_from_slice(&at.to_le_bytes());
        out.extend_from_slice(&[0, 0]);
        out
    }

    /// The pickle torch writes for {"w": <2x3 tensor>}, with the storage
    /// key and a TRANSPOSED stride so the odometer is exercised.
    fn state_dict_pickle(global: &str) -> Vec<u8> {
        let mut p = vec![0x80, 2];
        p.extend_from_slice(b"ccollections\nOrderedDict\nq\x00)Rq\x01(");
        p.push(b'X');
        p.extend_from_slice(&1u32.to_le_bytes());
        p.extend_from_slice(b"w");
        p.extend_from_slice(format!("c{global}\n").as_bytes());
        p.extend_from_slice(b"((X");
        p.extend_from_slice(&7u32.to_le_bytes());
        p.extend_from_slice(b"storage");
        p.extend_from_slice(b"ctorch\nFloatStorage\n");
        p.push(b'X');
        p.extend_from_slice(&1u32.to_le_bytes());
        p.extend_from_slice(b"0");
        p.push(b'X');
        p.extend_from_slice(&3u32.to_le_bytes());
        p.extend_from_slice(b"cpu");
        p.extend_from_slice(b"K\x06tQK\x00K\x02K\x03\x86K\x01K\x02\x86\x89h\x00)Rtq\x02Ru.");
        p
    }

    #[test]
    fn a_state_dict_reads_with_strides_honoured() {
        let data: Vec<u8> = [0f32, 1., 2., 3., 4., 5.].iter().flat_map(|v| v.to_le_bytes()).collect();
        let pkl = state_dict_pickle("torch._utils\n_rebuild_tensor_v2");
        let z = zip_of(&[("archive/data.pkl", &pkl), ("archive/data/0", &data)]);
        let sd = parse(&z).unwrap();
        let w = &sd["w"];
        assert_eq!(w.dims, vec![2, 3]);
        // stride (1, 2): element [i][j] is storage[i + 2j].
        assert_eq!(w.data, vec![0., 2., 4., 1., 3., 5.]);
    }

    #[test]
    fn a_wrapped_state_dict_reads_its_params_ema() {
        let data: Vec<u8> = [0f32, 1., 2., 3., 4., 5.].iter().flat_map(|v| v.to_le_bytes()).collect();
        let inner = state_dict_pickle("torch._utils\n_rebuild_tensor_v2");
        // {"params_ema": <the state dict>}: EMPTY_DICT, the key, the inner
        // pickle's body (minus PROTO and STOP), SETITEM, STOP.
        let mut p = vec![0x80, 2, b'}', b'X'];
        p.extend_from_slice(&10u32.to_le_bytes());
        p.extend_from_slice(b"params_ema");
        p.extend_from_slice(&inner[2..inner.len() - 1]);
        p.extend_from_slice(b"s.");
        let z = zip_of(&[("archive/data.pkl", &p), ("archive/data/0", &data)]);
        let sd = parse(&z).unwrap();
        assert_eq!(sd["w"].dims, vec![2, 3]);
    }

    #[test]
    fn any_other_global_is_refused_by_name() {
        let data = [0u8; 24];
        let pkl = state_dict_pickle("os\nsystem");
        let z = zip_of(&[("archive/data.pkl", &pkl), ("archive/data/0", &data)]);
        let err = parse(&z).unwrap_err().to_string();
        assert!(err.contains("os.system"), "{err}");
    }
}
