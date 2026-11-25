//! A minimal layered Photoshop (PSD) writer.
//!
//! The Rust PSD crates read; none writes layers at 16 and 32 bits, so this
//! is the small subset a Render needs: an RGB document of N full-canvas
//! layers with transparency, plus the merged composite every reader that
//! ignores layers shows. 8-bit, 16-bit and 32-bit float, the three depths
//! Photoshop has.
//!
//! Format notes that are not obvious from the spec:
//! - At 16 and 32 bits Photoshop keeps the layer records in an `Lr16` /
//!   `Lr32` tagged block after the global mask info, with the main layer
//!   info section left EMPTY. We do the same: that is what it writes and
//!   what every reader handles.
//! - A NEGATIVE layer count says the composite's first extra channel is its
//!   transparency, which is how a transparent merged image is stored.
//! - The composite's colour is matted against WHITE (Photoshop's own
//!   convention for a transparent document's merged image); layers are
//!   straight (unpremultiplied) colour.
//! - 8-bit channels are PackBits (Photoshop's default); 16 and 32-bit are
//!   raw, which every reader takes.
//! - Classic PSD is capped at 30000 px a side; past that is PSB, refused.

use anyhow::{bail, Result};
use std::io::Write;
use std::path::Path;

/// Bits per channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Depth {
    U8,
    U16,
    /// Float. Photoshop treats a 32-bit document as linear light.
    F32,
}

impl Depth {
    fn bits(self) -> u16 {
        match self {
            Depth::U8 => 8,
            Depth::U16 => 16,
            Depth::F32 => 32,
        }
    }
}

/// One layer: a name and straight (unpremultiplied) RGBA, `w * h * 4`
/// floats, row-major, in whatever encoding the file is meant to hold.
pub struct Layer<'a> {
    pub name: &'a str,
    pub rgba: &'a [f32],
}

/// The largest side a classic PSD can hold.
pub const MAX_SIDE: u32 = 30000;

/// Write a layered PSD. `layers[0]` is the BOTTOM of the stack. The
/// composite is the layers stacked with a normal blend.
pub fn write(path: &Path, w: u32, h: u32, depth: Depth, layers: &[Layer]) -> Result<()> {
    let bytes = encode(w, h, depth, layers)?;
    let tmp = path.with_extension("psd.part");
    {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        f.write_all(&bytes)?;
        f.flush()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// The whole file in memory.
pub fn encode(w: u32, h: u32, depth: Depth, layers: &[Layer]) -> Result<Vec<u8>> {
    if w == 0 || h == 0 {
        bail!("empty image");
    }
    if w > MAX_SIDE || h > MAX_SIDE {
        bail!("PSD is limited to {MAX_SIDE} px a side ({w}x{h}); write an EXR");
    }
    if layers.is_empty() {
        bail!("a PSD needs at least one layer");
    }
    if layers.len() > i16::MAX as usize {
        bail!("too many layers");
    }
    let n = (w * h) as usize;
    for l in layers {
        if l.rgba.len() < n * 4 {
            bail!("layer '{}' has {} values, wants {}", l.name, l.rgba.len(), n * 4);
        }
    }
    let mut out = Vec::with_capacity(n * 4 * (layers.len() + 1) * (depth.bits() as usize / 8) + 4096);

    // Header.
    out.extend_from_slice(b"8BPS");
    be16(&mut out, 1);
    out.extend_from_slice(&[0; 6]);
    be16(&mut out, 4); // composite: RGB + its transparency
    be32(&mut out, h);
    be32(&mut out, w);
    be16(&mut out, depth.bits());
    be16(&mut out, 3); // RGB
    // Colour mode data, image resources: none.
    be32(&mut out, 0);
    be32(&mut out, 0);

    // Layer and mask information.
    let info = layer_info(w, h, depth, layers);
    let mut lmi = Vec::with_capacity(info.len() + 32);
    match depth {
        Depth::U8 => {
            be32(&mut lmi, info.len() as u32);
            lmi.extend_from_slice(&info);
            be32(&mut lmi, 0); // global layer mask info
        }
        Depth::U16 | Depth::F32 => {
            be32(&mut lmi, 0); // main layer info: empty, see the module note
            be32(&mut lmi, 0); // global layer mask info
            lmi.extend_from_slice(b"8BIM");
            lmi.extend_from_slice(if depth == Depth::U16 { b"Lr16" } else { b"Lr32" });
            let mut block = info;
            while block.len() % 4 != 0 {
                block.push(0);
            }
            be32(&mut lmi, block.len() as u32);
            lmi.extend_from_slice(&block);
        }
    }
    be32(&mut out, lmi.len() as u32);
    out.extend_from_slice(&lmi);

    // The merged composite, planar R G B A.
    let comp = composite(layers, n);
    let planes: Vec<Vec<f32>> = (0..4).map(|c| comp.iter().skip(c).step_by(4).copied().collect()).collect();
    match depth {
        Depth::U8 => {
            be16(&mut out, 1);
            let rows: Vec<Vec<u8>> = planes
                .iter()
                .flat_map(|p| p.chunks(w as usize).map(|r| packbits(&quant8(r))))
                .collect();
            for r in &rows {
                be16(&mut out, r.len() as u16);
            }
            for r in &rows {
                out.extend_from_slice(r);
            }
        }
        _ => {
            be16(&mut out, 0);
            for p in &planes {
                out.extend_from_slice(&raw(p, depth));
            }
        }
    }
    Ok(out)
}

/// The layer info structure: count, records, then each layer's channels.
fn layer_info(w: u32, h: u32, depth: Depth, layers: &[Layer]) -> Vec<u8> {
    let n = (w * h) as usize;
    // Encode channel data first: the records carry each channel's length.
    // Channel ids: -1 transparency, 0 1 2 = R G B.
    let ids: [(i16, usize); 4] = [(-1, 3), (0, 0), (1, 1), (2, 2)];
    let data: Vec<Vec<Vec<u8>>> = layers
        .iter()
        .map(|l| {
            ids.iter()
                .map(|&(_, lane)| {
                    let plane: Vec<f32> = l.rgba[..n * 4].iter().skip(lane).step_by(4).copied().collect();
                    channel(&plane, w as usize, depth)
                })
                .collect()
        })
        .collect();
    let mut v = Vec::new();
    // Negative: the composite's first alpha is its transparency.
    be16(&mut v, (-(layers.len() as i16)) as u16);
    for (l, chans) in layers.iter().zip(&data) {
        be32(&mut v, 0); // top
        be32(&mut v, 0); // left
        be32(&mut v, h); // bottom
        be32(&mut v, w); // right
        be16(&mut v, ids.len() as u16);
        for (&(id, _), c) in ids.iter().zip(chans) {
            be16(&mut v, id as u16);
            be32(&mut v, c.len() as u32);
        }
        v.extend_from_slice(b"8BIMnorm");
        v.push(255); // opacity
        v.push(0); // clipping: base
        v.push(0x08); // flags: bit 3 set, bit 4 meaningful (pixel data relevant)
        v.push(0); // filler
        let mut extra = Vec::new();
        be32(&mut extra, 0); // layer mask data
        be32(&mut extra, 0); // blending ranges
        // Pascal name (MacRoman; ASCII here), padded to a multiple of 4
        // INCLUDING the length byte. The real name rides in `luni`.
        let ascii: Vec<u8> =
            l.name.chars().map(|c| if c.is_ascii() && !c.is_ascii_control() { c as u8 } else { b'_' }).take(255).collect();
        extra.push(ascii.len() as u8);
        extra.extend_from_slice(&ascii);
        while (extra.len() - 8) % 4 != 0 {
            extra.push(0);
        }
        // Unicode name.
        let utf16: Vec<u16> = l.name.encode_utf16().collect();
        let mut luni = Vec::new();
        be32(&mut luni, utf16.len() as u32);
        for c in &utf16 {
            be16(&mut luni, *c);
        }
        while luni.len() % 4 != 0 {
            luni.push(0);
        }
        extra.extend_from_slice(b"8BIMluni");
        be32(&mut extra, luni.len() as u32);
        extra.extend_from_slice(&luni);
        be32(&mut v, extra.len() as u32);
        v.extend_from_slice(&extra);
    }
    for chans in &data {
        for c in chans {
            v.extend_from_slice(c);
        }
    }
    if v.len() % 2 != 0 {
        v.push(0);
    }
    v
}

/// One layer channel: compression tag, then the data.
fn channel(plane: &[f32], w: usize, depth: Depth) -> Vec<u8> {
    let mut c = Vec::new();
    match depth {
        Depth::U8 => {
            be16(&mut c, 1);
            let rows: Vec<Vec<u8>> = plane.chunks(w).map(|r| packbits(&quant8(r))).collect();
            for r in &rows {
                be16(&mut c, r.len() as u16);
            }
            for r in &rows {
                c.extend_from_slice(r);
            }
        }
        _ => {
            be16(&mut c, 0);
            c.extend_from_slice(&raw(plane, depth));
        }
    }
    c
}

/// Stack the layers with a normal blend, then matte the colour on white.
fn composite(layers: &[Layer], n: usize) -> Vec<f32> {
    let mut acc = vec![0.0f32; n * 4]; // premultiplied
    for l in layers {
        for (d, s) in acc.chunks_mut(4).zip(l.rgba.chunks(4)) {
            let a = clean(s[3]).clamp(0.0, 1.0);
            for k in 0..3 {
                d[k] = clean(s[k]) * a + d[k] * (1.0 - a);
            }
            d[3] = a + d[3] * (1.0 - a);
        }
    }
    for px in acc.chunks_mut(4) {
        let a = px[3];
        for k in 0..3 {
            px[k] += 1.0 - a;
        }
    }
    acc
}

fn clean(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

fn quant8(r: &[f32]) -> Vec<u8> {
    r.iter().map(|v| (clean(*v).clamp(0.0, 1.0) * 255.0).round() as u8).collect()
}

fn raw(p: &[f32], depth: Depth) -> Vec<u8> {
    match depth {
        Depth::U8 => quant8(p),
        Depth::U16 => p
            .iter()
            .flat_map(|v| ((clean(*v).clamp(0.0, 1.0) * 65535.0).round() as u16).to_be_bytes())
            .collect(),
        Depth::F32 => p.iter().flat_map(|v| clean(*v).to_be_bytes()).collect(),
    }
}

/// PackBits, one row.
fn packbits(row: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(row.len() + row.len() / 128 + 1);
    let n = row.len();
    let mut i = 0;
    while i < n {
        let mut run = 1;
        while i + run < n && run < 128 && row[i + run] == row[i] {
            run += 1;
        }
        if run >= 3 {
            out.push((1 - run as i32) as i8 as u8);
            out.push(row[i]);
            i += run;
        } else {
            let start = i;
            while i < n && i - start < 128 {
                if i + 2 < n && row[i] == row[i + 1] && row[i] == row[i + 2] {
                    break;
                }
                i += 1;
            }
            out.push((i - start - 1) as u8);
            out.extend_from_slice(&row[start..i]);
        }
    }
    out
}

/// Straight RGBA f32 from a Render's premultiplied half-float rows, colour
/// converted AFTER unpremultiplying (a curve applied to premultiplied
/// values darkens every soft edge). `chan`: one channel as grey, opaque.
pub fn from_premult_f16(
    w: usize,
    h: usize,
    stride: usize,
    px: &[half::f16],
    t: Option<&crate::color::Transform>,
    chan: Option<[f32; 4]>,
) -> Vec<f32> {
    use rayon::prelude::*;
    let mut out = vec![0.0f32; w * h * 4];
    out.par_chunks_mut(w * 4).zip(px.par_chunks(stride)).for_each(|(o, row)| {
        for (d, s) in o.chunks_mut(4).zip(row[..w * 4].chunks(4)) {
            let s = [s[0].to_f32(), s[1].to_f32(), s[2].to_f32(), s[3].to_f32()];
            let a = s[3];
            let mut c = if a > 1e-6 { [s[0] / a, s[1] / a, s[2] / a] } else { [0.0; 3] };
            if let Some(t) = t {
                c = t.apply(c);
            }
            if let Some(m) = chan {
                let v = c[0] * m[0] + c[1] * m[1] + c[2] * m[2] + a * m[3];
                d.copy_from_slice(&[v, v, v, 1.0]);
                continue;
            }
            d.copy_from_slice(&[c[0], c[1], c[2], a.clamp(0.0, 1.0)]);
        }
    });
    out
}

fn be16(v: &mut Vec<u8>, x: u16) {
    v.extend_from_slice(&x.to_be_bytes());
}

fn be32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_be_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unpack(mut d: &[u8], want: usize) -> Vec<u8> {
        let mut o = Vec::new();
        while o.len() < want {
            let h = d[0] as i8;
            d = &d[1..];
            if h >= 0 {
                let k = h as usize + 1;
                o.extend_from_slice(&d[..k]);
                d = &d[k..];
            } else if h != -128 {
                o.extend(std::iter::repeat_n(d[0], (1 - h as isize) as usize));
                d = &d[1..];
            }
        }
        o
    }

    #[test]
    fn packbits_round_trips() {
        let mut row: Vec<u8> = (0..300).map(|i| (i % 7) as u8).collect();
        row.extend(std::iter::repeat_n(9, 400));
        row.extend([1, 2, 2, 3, 3, 3, 4]);
        assert_eq!(unpack(&packbits(&row), row.len()), row);
    }

    fn read16(b: &[u8], at: usize) -> u16 {
        u16::from_be_bytes([b[at], b[at + 1]])
    }
    fn read32(b: &[u8], at: usize) -> u32 {
        u32::from_be_bytes(b[at..at + 4].try_into().unwrap())
    }

    /// Walk the file's sections by their lengths and land exactly on the end:
    /// a wrong length anywhere throws every later offset off.
    fn walk(b: &[u8], depth: Depth, w: usize, h: usize) -> (i16, Vec<f32>) {
        assert_eq!(&b[..4], b"8BPS");
        assert_eq!(read16(b, 22), depth.bits());
        let mut at = 26;
        at += 4 + read32(b, at) as usize; // colour mode
        at += 4 + read32(b, at) as usize; // resources
        let lmi = read32(b, at) as usize;
        let lmi_start = at + 4;
        let info = match depth {
            Depth::U8 => lmi_start,
            _ => {
                assert_eq!(read32(b, lmi_start), 0);
                let tag = lmi_start + 8;
                assert_eq!(&b[tag..tag + 4], b"8BIM");
                tag + 12
            }
        };
        let count = read16(b, info + if depth == Depth::U8 { 4 } else { 0 }) as i16;
        at = lmi_start + lmi;
        let comp = read16(b, at);
        at += 2;
        let n = w * h;
        let mut px = vec![0.0; n * 4];
        match depth {
            Depth::U8 => {
                assert_eq!(comp, 1);
                let counts: Vec<usize> = (0..4 * h).map(|i| read16(b, at + i * 2) as usize).collect();
                at += 8 * h;
                for c in 0..4 {
                    for y in 0..h {
                        let len = counts[c * h + y];
                        let row = unpack(&b[at..at + len], w);
                        at += len;
                        for x in 0..w {
                            px[(y * w + x) * 4 + c] = row[x] as f32 / 255.0;
                        }
                    }
                }
            }
            Depth::U16 => {
                for c in 0..4 {
                    for i in 0..n {
                        px[i * 4 + c] = read16(b, at) as f32 / 65535.0;
                        at += 2;
                    }
                }
            }
            Depth::F32 => {
                for c in 0..4 {
                    for i in 0..n {
                        px[i * 4 + c] = f32::from_be_bytes(b[at..at + 4].try_into().unwrap());
                        at += 4;
                    }
                }
            }
        }
        assert_eq!(at, b.len(), "trailing or missing bytes");
        (count, px)
    }

    #[test]
    fn every_depth_parses_to_the_end_and_mattes_on_white() {
        let (w, h) = (5usize, 3usize);
        let mut bottom = vec![0.0f32; w * h * 4];
        for px in bottom.chunks_mut(4) {
            px.copy_from_slice(&[1.0, 0.0, 0.0, 0.5]);
        }
        let top = vec![0.0f32; w * h * 4];
        for depth in [Depth::U8, Depth::U16, Depth::F32] {
            let b = encode(
                w as u32,
                h as u32,
                depth,
                &[Layer { name: "plate", rgba: &bottom }, Layer { name: "töp", rgba: &top }],
            )
            .unwrap();
            let (count, px) = walk(&b, depth, w, h);
            assert_eq!(count, -2);
            // Half-covered red over white: (1, .5, .5), alpha .5.
            let p = &px[..4];
            for (got, want) in p.iter().zip([1.0, 0.5, 0.5, 0.5]) {
                assert!((got - want).abs() < 0.01, "{depth:?}: {p:?}");
            }
        }
    }

    /// Sample files for checking in a real reader:
    /// `WOBBLE_PSD_OUT=dir cargo test -p engine --lib psd -- --ignored`.
    #[test]
    #[ignore]
    fn write_samples() {
        let Some(dir) = std::env::var_os("WOBBLE_PSD_OUT") else { return };
        let dir = std::path::PathBuf::from(dir);
        let (w, h) = (256usize, 128usize);
        // A soft red disc, and a blue bar over it at half coverage.
        let mut disc = vec![0.0f32; w * h * 4];
        let mut bar = vec![0.0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let r = (((x as f32 - 90.0).powi(2) + (y as f32 - 64.0).powi(2)).sqrt() - 50.0) / 8.0;
                disc[i..i + 4].copy_from_slice(&[1.0, 0.2, 0.1, (0.5 - r).clamp(0.0, 1.0)]);
                if (140..220).contains(&x) {
                    bar[i..i + 4].copy_from_slice(&[0.1, 0.3, 1.0, 0.5]);
                }
            }
        }
        for (depth, tag) in [(Depth::U8, 8), (Depth::U16, 16), (Depth::F32, 32)] {
            let layers = [Layer { name: "disc", rgba: &disc }, Layer { name: "bär", rgba: &bar }];
            write(&dir.join(format!("sample{tag}.psd")), w as u32, h as u32, depth, &layers).unwrap();
        }
    }

    #[test]
    fn refuses_what_classic_psd_cannot_hold() {
        let l = [Layer { name: "x", rgba: &[] }];
        assert!(encode(MAX_SIDE + 1, 1, Depth::U8, &l).is_err());
        assert!(encode(1, 1, Depth::U8, &[]).is_err());
    }
}
