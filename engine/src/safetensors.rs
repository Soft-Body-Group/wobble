//! Reading `.safetensors` checkpoints: an 8-byte header length, a JSON
//! header naming every tensor's dtype, shape and byte range, then the raw
//! little-endian data. No code anywhere in the format, which is why it is
//! the one to prefer (CLAUDE.md "Readers stay safe").
//!
//! The header is parsed by the small JSON reader below rather than serde,
//! so the engine reads checkpoints without its optional serde feature.
//! F32, F16 and BF16 tensors come out as f32; integer tensors (BatchNorm's
//! `num_batches_tracked`) are skipped.

use crate::npy::Tensor;
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::Path;

pub fn read(path: &Path) -> Result<HashMap<String, Tensor>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    parse(&bytes).with_context(|| format!("parse {}", path.display()))
}

pub fn parse(bytes: &[u8]) -> Result<HashMap<String, Tensor>> {
    let n = u64::from_le_bytes(bytes.get(..8).context("not a safetensors file")?.try_into().unwrap()) as usize;
    let header = bytes.get(8..8 + n).context("safetensors header runs past the file")?;
    let data = &bytes[8 + n..];
    let mut j = Json { s: header, i: 0 };
    let Value::Object(items) = j.value()? else { bail!("safetensors header is not an object") };
    let mut out = HashMap::new();
    for (name, v) in items {
        if name == "__metadata__" {
            continue;
        }
        let Value::Object(f) = v else { bail!("{name}: not an object") };
        let get = |k: &str| f.iter().find(|(n, _)| n == k).map(|(_, v)| v);
        let Some(Value::Str(dtype)) = get("dtype") else { bail!("{name}: no dtype") };
        let Some(Value::Array(shape)) = get("shape") else { bail!("{name}: no shape") };
        let Some(Value::Array(off)) = get("data_offsets") else { bail!("{name}: no offsets") };
        let dims: Vec<usize> = shape.iter().map(Value::usize).collect::<Result<_>>()?;
        let (a, b) = match off.as_slice() {
            [a, b] => (a.usize()?, b.usize()?),
            _ => bail!("{name}: bad offsets"),
        };
        let raw = data.get(a..b).with_context(|| format!("{name} runs past the file"))?;
        let count: usize = dims.iter().product();
        let vals: Vec<f32> = match dtype.as_str() {
            "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "F16" => raw.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            "BF16" => raw.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            _ => continue,
        };
        if vals.len() != count {
            bail!("{name}: {} values for shape {dims:?}", vals.len());
        }
        out.insert(name, Tensor { dims, data: vals });
    }
    Ok(out)
}

enum Value {
    Str(String),
    Num(f64),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
    Other,
}

impl Value {
    fn usize(&self) -> Result<usize> {
        match self {
            Value::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Ok(*n as usize),
            _ => bail!("expected a size"),
        }
    }
}

struct Json<'a> {
    s: &'a [u8],
    i: usize,
}

impl Json<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> Result<()> {
        self.ws();
        if self.s.get(self.i) != Some(&c) {
            bail!("safetensors header: expected '{}' at {}", c as char, self.i);
        }
        self.i += 1;
        Ok(())
    }

    fn value(&mut self) -> Result<Value> {
        self.ws();
        match self.s.get(self.i).copied() {
            Some(b'{') => {
                self.i += 1;
                let mut items = vec![];
                self.ws();
                if self.s.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Value::Object(items));
                }
                loop {
                    self.ws();
                    let Value::Str(k) = self.value()? else { bail!("safetensors header: a key that is not a string") };
                    self.eat(b':')?;
                    items.push((k, self.value()?));
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Value::Object(items));
                        }
                        _ => bail!("safetensors header: bad object at {}", self.i),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut items = vec![];
                self.ws();
                if self.s.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Value::Array(items));
                }
                loop {
                    items.push(self.value()?);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Value::Array(items));
                        }
                        _ => bail!("safetensors header: bad array at {}", self.i),
                    }
                }
            }
            Some(b'"') => {
                self.i += 1;
                let mut out = Vec::new();
                loop {
                    match self.s.get(self.i).copied() {
                        None => bail!("safetensors header: unterminated string"),
                        Some(b'"') => {
                            self.i += 1;
                            break;
                        }
                        Some(b'\\') => {
                            let e = *self.s.get(self.i + 1).context("bad escape")?;
                            out.push(match e {
                                b'n' => b'\n',
                                b't' => b'\t',
                                b'u' => bail!("safetensors header: \\u escapes are not supported"),
                                other => other,
                            });
                            self.i += 2;
                        }
                        Some(c) => {
                            out.push(c);
                            self.i += 1;
                        }
                    }
                }
                Ok(Value::Str(String::from_utf8(out).context("safetensors header: not utf-8")?))
            }
            Some(c) if c == b'-' || c.is_ascii_digit() => {
                let start = self.i;
                while self.i < self.s.len() && matches!(self.s[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                    self.i += 1;
                }
                let t = std::str::from_utf8(&self.s[start..self.i]).unwrap();
                Ok(Value::Num(t.parse().with_context(|| format!("safetensors header: bad number {t}"))?))
            }
            Some(_) => {
                // true / false / null: never needed, skipped.
                for w in [&b"true"[..], b"false", b"null"] {
                    if self.s[self.i..].starts_with(w) {
                        self.i += w.len();
                        return Ok(Value::Other);
                    }
                }
                bail!("safetensors header: unexpected byte at {}", self.i)
            }
            None => bail!("safetensors header ended early"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_reads_back_its_tensors() {
        let header = br#"{"__metadata__":{"a":"b"},"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"n":{"dtype":"I64","shape":[],"data_offsets":[16,24]},"h":{"dtype":"F16","shape":[1],"data_offsets":[24,26]}}"#;
        let mut f = (header.len() as u64).to_le_bytes().to_vec();
        f.extend_from_slice(header);
        for v in [1f32, 2., 3., 4.] {
            f.extend_from_slice(&v.to_le_bytes());
        }
        f.extend_from_slice(&7i64.to_le_bytes());
        f.extend_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        let t = parse(&f).unwrap();
        assert_eq!(t["w"].dims, vec![2, 2]);
        assert_eq!(t["w"].data, vec![1., 2., 3., 4.]);
        assert_eq!(t["h"].data, vec![0.5]);
        assert!(!t.contains_key("n"), "integer tensors are skipped");
    }
}
