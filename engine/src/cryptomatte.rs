//! Cryptomatte: id mattes the way renderers write them (Psyop spec). A
//! cryptomatte layer is pairs of (id, coverage) ranked by coverage —
//! `CryptoObject00.R/G` is rank 0, `.B/A` rank 1, `CryptoObject01` ranks 2
//! and 3, and so on. The id is a float32 whose BITS are a MurmurHash3 of the
//! object name, and the manifest in the EXR header maps names back to ids.
//!
//! The trap, and the reason this module exists: reader textures are f16,
//! and an id survives nothing — not a half conversion, not a filtered
//! sample, not a denormal flush. So a rank is decoded into three 11-bit
//! INTEGER chunks (exact in f16, never denormal, never NaN) plus the
//! coverage, one rank per texture, and the shader loads texels unfiltered
//! and reassembles the u32 before comparing. `encode_id`/`decode_id` are
//! the contract both sides of that share.

use anyhow::{Context, Result};
use half::f16;
use std::collections::HashMap;
use std::path::Path;

/// Ranks the node consumes: three crypto layers of two pairs each, which is
/// what every renderer writes by default (`cryptomatte depth 6`).
pub const RANKS: usize = 6;
/// Picks one node may hold; the uniform budget (two floats per pick).
pub const MAX_PICKS: usize = 32;
/// LIVE reader slots one node owns: the beauty, then one per rank.
pub const SOURCES: usize = 1 + RANKS;
/// The sub-index of the node's MATTE reader, after the live ones.
pub const MATTE_SUB: usize = SOURCES;
/// What marks a virtual layer name as "the finished matte" (see
/// [`CryptomatteParams::matte_layer`] and [`parse_matte_layer`]).
const MATTE_TAG: &str = "@matte";
/// Chunk width. 2^11 = 2048, and f16 represents every integer up to 2048.
pub const CHUNK_BITS: u32 = 11;
const CHUNK_MASK: u32 = (1 << CHUNK_BITS) - 1;
/// What the ids view divides the chunks by: keeps the picture in [0, 1) AND
/// exact, since k/2048 for k < 2048 is an f16 with its 11 significant bits.
pub const CHUNK_SCALE: f32 = 2048.0;

/// MurmurHash3 x86_32, seed 0 — the hash the spec names.
pub fn murmur3_32(data: &[u8], seed: u32) -> u32 {
    const C1: u32 = 0xcc9e_2d51;
    const C2: u32 = 0x1b87_3593;
    let mut h = seed;
    let mut chunks = data.chunks_exact(4);
    for c in &mut chunks {
        let mut k = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h ^= k;
        h = h.rotate_left(13).wrapping_mul(5).wrapping_add(0xe654_6b64);
    }
    let rem = chunks.remainder();
    if !rem.is_empty() {
        let mut k = 0u32;
        for (i, b) in rem.iter().enumerate() {
            k ^= (*b as u32) << (8 * i);
        }
        k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h ^= k;
    }
    h ^= data.len() as u32;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h
}

/// The id bits of an object name: the hash with its float exponent kept
/// out of the denormal/NaN bands, exactly as the spec's `mm3hash_float`.
pub fn id_of_name(name: &str) -> u32 {
    let mut h = murmur3_32(name.as_bytes(), 0);
    let exp = (h >> 23) & 0xff;
    if exp == 0 || exp == 255 {
        h ^= 1 << 23;
    }
    h
}

/// The 7-hex-digit key a layer's header attributes are filed under.
pub fn layer_key(layer_name: &str) -> String {
    format!("{:08x}", murmur3_32(layer_name.as_bytes(), 0))[..7].to_string()
}

/// Split id bits into three chunk integers, each exact in f16.
pub fn encode_id(bits: u32) -> [f16; 3] {
    [
        f16::from_f32((bits & CHUNK_MASK) as f32),
        f16::from_f32(((bits >> CHUNK_BITS) & CHUNK_MASK) as f32),
        f16::from_f32((bits >> (2 * CHUNK_BITS)) as f32),
    ]
}

/// Reassemble id bits from chunk values (as the shader sees them, f32).
pub fn decode_id(c: [f32; 3]) -> u32 {
    let k = |v: f32| (v.round().max(0.0) as u32) & CHUNK_MASK;
    k(c[0]) | (k(c[1]) << CHUNK_BITS) | (k(c[2]) << (2 * CHUNK_BITS))
}

/// Reassemble id bits from the ids VIEW (chunks divided by `CHUNK_SCALE`),
/// which is what a viewer readback hands back.
pub fn decode_scaled(c: [f32; 3]) -> u32 {
    decode_id([c[0] * CHUNK_SCALE, c[1] * CHUNK_SCALE, c[2] * CHUNK_SCALE])
}

/// An 11-bit chunk as a value in [1, 4), exact in f16: the first 1024 at
/// 1/1024 steps over [1, 2), the rest at 1/512 over [2, 4) — the f16 ULP
/// in each octave. The shader's `crypto_lit` is this function.
pub fn encode_lit(k: u32) -> f32 {
    if k < 1024 { 1.0 + k as f32 / 1024.0 } else { 2.0 + (k - 1024) as f32 / 512.0 }
}

fn decode_lit(v: f32) -> u32 {
    let k = if v < 2.0 { (v - 1.0) * 1024.0 } else { 1024.0 + (v - 2.0) * 512.0 };
    (k.round().max(0.0) as u32) & CHUNK_MASK
}

/// Reassemble id bits from an ids-view texel, either encoding: an unpicked
/// object is its chunks over `CHUNK_SCALE` (all below 1), a PICKED one is
/// lit — red and green as [`encode_lit`] (at or above 1), blue at
/// chunk2/4096 (below 0.25, exact) — so it reads as bright yellow on
/// screen, the usual colour for a picked matte, and still decodes, which is
/// what lets a removing click find it.
pub fn decode_view(c: [f32; 3]) -> u32 {
    if c[0] >= 1.0 {
        decode_id([decode_lit(c[0]) as f32, decode_lit(c[1]) as f32, c[2] * 2.0 * CHUNK_SCALE])
    } else {
        decode_scaled(c)
    }
}

/// The virtual layer name a Read uses for one rank of a crypto layer:
/// `CryptoObject` rank 3 -> `CryptoObject01#1`.
pub fn rank_layer_name(layer: &str, rank: usize) -> String {
    format!("{layer}{:02}#{}", rank / 2, rank % 2)
}

/// Inverse of [`rank_layer_name`]: the EXR layer to read and which pair
/// (0 = R/G, 1 = B/A) of it. `None` for any ordinary layer name.
pub fn parse_rank_layer(name: &str) -> Option<(&str, usize)> {
    let (layer, half) = name.rsplit_once('#')?;
    let half: usize = half.parse().ok()?;
    if half > 1 || layer.len() < 2 {
        return None;
    }
    let digits = &layer[layer.len() - 2..];
    if !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((layer, half))
}

/// A parsed matte request: which cryptomatte, which ids, inverted or not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatteSpec {
    pub layer: String,
    pub ids: Vec<u32>,
    pub invert: bool,
}

/// Inverse of [`CryptomatteParams::matte_layer`]. `None` for any other name.
pub fn parse_matte_layer(name: &str) -> Option<MatteSpec> {
    let (layer, rest) = name.split_once(MATTE_TAG)?;
    let mut spec = MatteSpec { layer: layer.to_string(), ids: Vec::new(), invert: false };
    for part in rest.split(';').filter(|p| !p.is_empty()) {
        let (k, v) = part.split_once('=')?;
        match k {
            "inv" => spec.invert = v == "1",
            "ids" => {
                for h in v.split(',').filter(|h| !h.is_empty()) {
                    spec.ids.push(u32::from_str_radix(h, 16).ok()?);
                }
            }
            _ => return None,
        }
    }
    Some(spec)
}

impl MatteSpec {
    /// One pixel's matte from its ranked `(id bits, coverage)` pairs — the
    /// same arithmetic as the fused pass: the coverage of every picked id,
    /// summed, clamped, then inverted.
    pub fn matte(&self, ranks: &[(u32, f32)]) -> f32 {
        let mut m = 0.0f32;
        for (id, cov) in ranks {
            if self.ids.contains(id) {
                m += cov;
            }
        }
        let m = m.clamp(0.0, 1.0);
        if self.invert { 1.0 - m } else { m }
    }
}

/// One cryptomatte in a file: its layer name and the name -> id manifest.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CryptoDef {
    pub name: String,
    pub manifest: HashMap<String, u32>,
}

impl CryptoDef {
    /// The object name for an id, if the manifest knows it.
    pub fn name_of(&self, id: u32) -> Option<&str> {
        self.manifest.iter().find(|(_, v)| **v == id).map(|(k, _)| k.as_str())
    }
}

/// Every cryptomatte a file declares, from its header. A file without the
/// metadata returns an empty list, never an error — that is the common case.
pub fn defs(path: &Path) -> Result<Vec<CryptoDef>> {
    let meta = exr::meta::MetaData::read_from_file(path, false)
        .with_context(|| format!("reading header of {}", path.display()))?;
    let mut attrs: Vec<(String, String)> = Vec::new();
    // Renderers file the metadata per part or per image; take both.
    for h in &meta.headers {
        for (k, v) in h.shared_attributes.other.iter().chain(h.own_attributes.other.iter()) {
            if let exr::meta::attribute::AttributeValue::Text(t) = v {
                attrs.push((k.to_string(), t.to_string()));
            }
        }
    }
    Ok(defs_from_attrs(&attrs))
}

/// The pure half of [`defs`]: header text attributes in, definitions out.
pub fn defs_from_attrs(attrs: &[(String, String)]) -> Vec<CryptoDef> {
    let mut out: Vec<CryptoDef> = Vec::new();
    for (k, v) in attrs {
        let Some(rest) = k.strip_prefix("cryptomatte/") else { continue };
        let Some((key, field)) = rest.split_once('/') else { continue };
        if field != "name" {
            continue;
        }
        let manifest = attrs
            .iter()
            .find(|(k2, _)| k2 == &format!("cryptomatte/{key}/manifest"))
            .map(|(_, m)| parse_manifest(m))
            .unwrap_or_default();
        out.push(CryptoDef { name: v.clone(), manifest });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// The manifest is a flat JSON object of `"name": "hex id"`. A dedicated
/// little scanner rather than a JSON dependency the engine does not
/// otherwise carry: strings with escapes, whitespace, nothing else.
pub fn parse_manifest(s: &str) -> HashMap<String, u32> {
    fn string(chars: &mut std::iter::Peekable<std::str::Chars>) -> Option<String> {
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }
        if chars.next()? != '"' {
            return None;
        }
        let mut out = String::new();
        loop {
            match chars.next()? {
                '"' => return Some(out),
                '\\' => match chars.next()? {
                    'n' => out.push('\n'),
                    't' => out.push('\t'),
                    'u' => {
                        let hex: String = (0..4).filter_map(|_| chars.next()).collect();
                        if let Some(c) =
                            u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32)
                        {
                            out.push(c);
                        }
                    }
                    c => out.push(c),
                },
                c => out.push(c),
            }
        }
    }
    let mut map = HashMap::new();
    let mut chars = s.chars().peekable();
    while chars.peek().is_some_and(|c| c.is_whitespace()) {
        chars.next();
    }
    if chars.next() != Some('{') {
        return map;
    }
    loop {
        let Some(name) = string(&mut chars) else { break };
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }
        if chars.next() != Some(':') {
            break;
        }
        let Some(hex) = string(&mut chars) else { break };
        if let Ok(id) = u32::from_str_radix(hex.trim(), 16) {
            map.insert(name, id);
        }
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }
        match chars.next() {
            Some(',') => continue,
            _ => break,
        }
    }
    map
}

/// Serialise a manifest the way renderers do, for writers and tests.
pub fn manifest_json(names: &[&str]) -> String {
    let body: Vec<String> = names
        .iter()
        .map(|n| format!("\"{}\":\"{:08x}\"", n.replace('"', "\\\""), id_of_name(n)))
        .collect();
    format!("{{{}}}", body.join(","))
}

/// A cryptomatte plate the way a renderer writes one: three orbiting discs
/// with names (`cube`, `sphere`, `cone`), f32 (id, coverage) pairs ranked
/// by coverage in `CryptoObject00..02`, the manifest in the header, and a
/// beauty whose colours are the discs' so a picked matte is checkable by
/// eye. Frame `f` is at phase `f / frames`, so a ONE-frame plate is the
/// frame-0 picture, still — what a suite wants under a running transport.
/// `gen_exr … crypto` is this function.
pub fn write_plate(out_dir: &Path, w: usize, h: usize, frames: usize) -> Result<()> {
    write_plate_with(out_dir, w, h, frames, exr::prelude::Compression::ZIP16)
}

/// [`write_plate`] with a choice of compression (the generator's).
pub fn write_plate_with(
    out_dir: &Path,
    w: usize,
    h: usize,
    frames: usize,
    compression: exr::prelude::Compression,
) -> Result<()> {
    use exr::prelude::*;
    use rayon::prelude::*;

    std::fs::create_dir_all(out_dir)?;
    const NAMES: [&str; 3] = ["cube", "sphere", "cone"];
    const COLS: [[f32; 3]; 3] = [[4.0, 0.6, 0.2], [0.3, 2.5, 0.8], [0.4, 0.7, 3.5]];
    let ids: Vec<u32> = NAMES.iter().map(|n| id_of_name(n)).collect();
    (0..frames).into_par_iter().try_for_each(|f| -> anyhow::Result<()> {
        let path = out_dir.join(format!("frame.{f:04}.exr"));
        let t = f as f32 / frames.max(1) as f32;
        let tau = std::f32::consts::TAU;
        let mut base = [(); 4].map(|_| Vec::with_capacity(w * h));
        // Six ranks: id and coverage per rank.
        let mut rank = [(); 6].map(|_| (Vec::with_capacity(w * h), Vec::with_capacity(w * h)));
        for y in 0..h {
            for x in 0..w {
                let (u, v) = ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
                // Per-object coverage: a hard disc with a ~1.5px soft rim,
                // nearer discs occluding farther ones front to back.
                let mut covs: Vec<(usize, f32)> = Vec::new();
                let mut left = 1.0f32;
                for k in 0..3 {
                    let ph = t * tau + k as f32 * tau / 3.0;
                    let (cx, cy) = (0.5 + 0.3 * ph.cos(), 0.5 + 0.3 * (ph * 1.3).sin());
                    let d = ((u - cx).powi(2) + ((v - cy) * h as f32 / w as f32).powi(2)).sqrt();
                    let px = 1.5 / w as f32;
                    let c = ((0.16 - d) / px + 0.5).clamp(0.0, 1.0) * left;
                    if c > 0.0 {
                        covs.push((k, c));
                        left -= c;
                    }
                }
                covs.sort_by(|a, b| b.1.total_cmp(&a.1));
                let mut rgb = [0.0f32; 3];
                let mut a = 0.0;
                for (k, c) in &covs {
                    for i in 0..3 {
                        rgb[i] += COLS[*k][i] * c;
                    }
                    a += c;
                }
                for i in 0..3 {
                    base[i].push(f16::from_f32(rgb[i]));
                }
                base[3].push(f16::from_f32(a));
                for r in 0..6 {
                    let (id, c) = covs.get(r).map(|(k, c)| (ids[*k], *c)).unwrap_or((0, 0.0));
                    rank[r].0.push(f32::from_bits(id));
                    rank[r].1.push(c);
                }
            }
        }
        let [br, bg, bb, ba] = base;
        let mut channels = vec![
            AnyChannel::new("R", FlatSamples::F16(br)),
            AnyChannel::new("G", FlatSamples::F16(bg)),
            AnyChannel::new("B", FlatSamples::F16(bb)),
            AnyChannel::new("A", FlatSamples::F16(ba)),
        ];
        for (r, (id, cov)) in rank.into_iter().enumerate() {
            let layer = format!("CryptoObject{:02}", r / 2);
            let (idn, covn) = if r % 2 == 0 { ("R", "G") } else { ("B", "A") };
            channels.push(AnyChannel::new(format!("{layer}.{idn}").as_str(), FlatSamples::F32(id)));
            channels.push(AnyChannel::new(format!("{layer}.{covn}").as_str(), FlatSamples::F32(cov)));
        }
        let mut attrs = LayerAttributes::named("rgba");
        let key = layer_key("CryptoObject");
        for (k, v) in [
            ("name", "CryptoObject".to_string()),
            ("hash", "MurmurHash3_32".to_string()),
            ("conversion", "uint32_to_float32".to_string()),
            ("manifest", manifest_json(&NAMES)),
        ] {
            attrs.other.insert(
                Text::from(format!("cryptomatte/{key}/{k}").as_str()),
                AttributeValue::Text(Text::from(v.as_str())),
            );
        }
        let layer = Layer::new(
            (w, h),
            attrs,
            Encoding {
                compression,
                blocks: Blocks::ScanLines,
                line_order: LineOrder::Increasing,
            },
            AnyChannels::sort(channels.into_iter().collect()),
        );
        Image::from_layer(layer).write().non_parallel().to_file(&path)?;
        Ok(())
    })?;
    Ok(())
}


/// One selected object.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CryptoPick {
    pub name: String,
    pub id: u32,
}

/// Slots before the app mints them.
fn unminted() -> Vec<usize> {
    vec![usize::MAX; SOURCES]
}

fn no_slot() -> usize {
    usize::MAX
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CryptomatteParams {
    /// Which cryptomatte of the file feeds the ranks (`CryptoObject`,
    /// `CryptoMaterial`, ...): names the rank layers the sources decode.
    #[cfg_attr(feature = "serde", serde(default))]
    pub layer: String,
    /// This node's OWN reader slots, `SOURCES` of them: the file's rgba
    /// (for the preview) then rank 0..RANKS. The node's one input is the
    /// multi-channel cable — a lazy reference to the file — and these are
    /// how the ranks come off it: each is a reader of the upstream Read's
    /// sequence with its own layer, on the ChannelSelect precedent, so only
    /// the crypto pairs decode and cache. `usize::MAX` = not yet minted;
    /// the app assigns them before playback.
    #[cfg_attr(feature = "serde", serde(default = "unminted"))]
    pub sources: Vec<usize>,
    /// The reader slot of the finished MATTE. Seven f32 id layers a frame
    /// is what makes this node slow, so nobody downstream pays it: they
    /// sample this ONE reader, whose decode is the matte itself — computed
    /// from the id pairs on the decode thread on a miss and kept by the
    /// NVMe tier like any frame, so every frame that is ever requested is
    /// baked, and a pick edit (a different layer string, so a different
    /// cache key) simply starts a fresh set. The live sources above are
    /// instanced only while the node is VIEWED, where picking needs ids.
    #[cfg_attr(feature = "serde", serde(default = "no_slot"))]
    pub matte_source: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    pub picks: Vec<CryptoPick>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub invert: bool,
    /// Transient: set by the app when this node IS the view target, on the
    /// transient graph only (the Keyer precedent). Viewed, the node shows
    /// its IDS picture (every object a flat colour, picked ones lit yellow,
    /// the matte in alpha — press A); everyone downstream gets the matte.
    /// Never authored, never serialized, so the picture can never reach a
    /// render or a consumer.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub previewing: bool,
    /// The comp frame this node's readers' first frame lands on — the
    /// upstream Read's timeline placement, copied in when the app lowers
    /// the graph. TRANSIENT (the Read owns the authored value), like
    /// `previewing`: without it a placed plate's crypto ranks played at
    /// frame one while the plate itself played at 1001.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub start: Option<i64>,
}

impl Default for CryptomatteParams {
    fn default() -> Self {
        Self {
            layer: String::new(),
            sources: unminted(),
            matte_source: usize::MAX,
            picks: Vec::new(),
            invert: false,
            previewing: false,
            start: None,
        }
    }
}

impl CryptomatteParams {
    /// Uniform slots: one header, then two picks per vec4.
    pub const SLOTS: usize = 1 + MAX_PICKS / 2;

    /// The reader slot of source `sub` (0 = rgba, 1.. = ranks), if minted.
    pub fn source(&self, sub: usize) -> Option<usize> {
        self.sources.get(sub).copied().filter(|s| *s != usize::MAX)
    }

    /// Every slot minted — the node can render.
    pub fn minted(&self) -> bool {
        self.sources.len() == SOURCES
            && self.sources.iter().all(|s| *s != usize::MAX)
            && self.matte_source != usize::MAX
    }

    /// Every reader slot in `sub` order: the live sources, then the matte.
    pub fn reader_slots(&self) -> Vec<usize> {
        let mut v = self.sources.clone();
        v.resize(SOURCES, usize::MAX);
        v.push(self.matte_source);
        v
    }

    /// Which readers are in the rig: the live set while the node is viewed
    /// (the ids picture), the matte alone otherwise.
    pub fn reader_sub_active(&self, sub: usize) -> bool {
        (sub == MATTE_SUB) != self.previewing
    }

    /// The layer reader `sub` decodes: `None` is the file's rgba.
    pub fn source_layer(&self, sub: usize) -> Option<String> {
        match sub {
            0 => None,
            MATTE_SUB => Some(self.matte_layer()),
            r => Some(rank_layer_name(&self.layer, r - 1)),
        }
    }

    /// The virtual layer the matte reader asks the decoder for:
    /// `CryptoObject@matte;inv=0;ids=0a0b0c0d,…`. Everything the matte
    /// depends on is IN the string, which is also what keys both cache
    /// tiers — so a pick edit can never replay another selection's frames.
    /// Ids are sorted: the same selection made in another order is the
    /// same matte and the same cache entries.
    pub fn matte_layer(&self) -> String {
        let mut ids: Vec<u32> = self.picks.iter().take(MAX_PICKS).map(|p| p.id).collect();
        ids.sort_unstable();
        ids.dedup();
        let ids: Vec<String> = ids.iter().map(|i| format!("{i:08x}")).collect();
        format!("{}{MATTE_TAG};inv={};ids={}", self.layer, self.invert as u8, ids.join(","))
    }

    /// Make `id` the ONLY pick (a plain click); returns whether anything changed.
    pub fn set_only(&mut self, name: &str, id: u32) -> bool {
        if self.picks.len() == 1 && self.picks[0].id == id {
            return false;
        }
        self.picks.clear();
        self.picks.push(CryptoPick { name: name.to_string(), id });
        true
    }

    /// Add or remove a pick; returns whether anything changed.
    pub fn toggle(&mut self, name: &str, id: u32, add: bool) -> bool {
        match (self.picks.iter().position(|p| p.id == id), add) {
            (Some(_), true) | (None, false) => false,
            (Some(i), false) => {
                self.picks.remove(i);
                true
            }
            (None, true) => {
                if self.picks.len() >= MAX_PICKS {
                    return false;
                }
                self.picks.push(CryptoPick { name: name.to_string(), id });
                true
            }
        }
    }

    /// Write the uniform block: `(ids picture?, n, invert, 0)` then
    /// `(hi, lo)` pairs of the id's 16-bit halves — each exact as an f32,
    /// where a bitcast through a float NaN would not be.
    pub fn pack(&self, v: &mut [f32]) {
        if v.len() < Self::SLOTS * 4 {
            return;
        }
        v[0] = if self.previewing { 1.0 } else { 0.0 };
        let n = self.picks.len().min(MAX_PICKS);
        v[1] = n as f32;
        v[2] = if self.invert { 1.0 } else { 0.0 };
        v[3] = 0.0;
        for (i, p) in self.picks.iter().take(n).enumerate() {
            let o = 4 + i * 2;
            v[o] = (p.id >> 16) as f32;
            v[o + 1] = (p.id & 0xffff) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn murmur_matches_the_reference_vectors() {
        assert_eq!(murmur3_32(b"", 0), 0);
        assert_eq!(murmur3_32(b"hello", 0), 0x248b_fa47);
        assert_eq!(murmur3_32(b"The quick brown fox jumps over the lazy dog", 0), 0x2e4f_f723);
    }

    #[test]
    fn ids_never_land_in_the_denormal_or_nan_bands() {
        for n in 0..5000 {
            let id = id_of_name(&format!("obj{n}"));
            let exp = (id >> 23) & 0xff;
            assert!(exp != 0 && exp != 255, "{n}: {id:08x}");
        }
    }

    #[test]
    fn chunks_round_trip_through_f16_exactly() {
        let mut x = 0x9e37_79b9u32;
        for _ in 0..20_000 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let c = encode_id(x);
            let back = decode_id([c[0].to_f32(), c[1].to_f32(), c[2].to_f32()]);
            assert_eq!(back, x);
            // The ids view scales by 2048 and must stay exact through f16.
            let scaled: [f32; 3] =
                std::array::from_fn(|i| f16::from_f32(c[i].to_f32() / CHUNK_SCALE).to_f32());
            assert_eq!(decode_scaled(scaled), x);
        }
        for edge in [0u32, u32::MAX, 1 << 31, 0x7ff, 0x800] {
            let c = encode_id(edge);
            assert_eq!(decode_id([c[0].to_f32(), c[1].to_f32(), c[2].to_f32()]), edge);
        }
    }

    /// The lit (picked) encoding survives f16 too, and is told apart from
    /// the plain one by green alone.
    #[test]
    fn the_lit_encoding_round_trips_through_f16_and_reads_as_yellow() {
        let mut x = 0x2545_f491u32;
        for _ in 0..20_000 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let c = encode_id(x);
            let lit = [
                encode_lit(c[0].to_f32() as u32),
                encode_lit(c[1].to_f32() as u32),
                c[2].to_f32() / (2.0 * CHUNK_SCALE),
            ];
            let through: [f32; 3] = std::array::from_fn(|i| f16::from_f32(lit[i]).to_f32());
            assert_eq!(decode_view(through), x, "{x:08x}");
            assert!(through[0] >= 1.0 && through[1] >= 1.0 && through[2] < 0.25, "yellow: {through:?}");
            let plain: [f32; 3] = std::array::from_fn(|i| f16::from_f32(c[i].to_f32() / CHUNK_SCALE).to_f32());
            assert_eq!(decode_view(plain), x);
        }
    }

    #[test]
    fn rank_layer_names_round_trip() {
        for r in 0..RANKS {
            let n = rank_layer_name("CryptoObject", r);
            let (layer, half) = parse_rank_layer(&n).unwrap();
            assert_eq!(layer, format!("CryptoObject{:02}", r / 2));
            assert_eq!(half, r % 2);
        }
        assert_eq!(rank_layer_name("CryptoObject", 3), "CryptoObject01#1");
        assert!(parse_rank_layer("diffuse").is_none());
        assert!(parse_rank_layer("CryptoObject00#2").is_none());
        assert!(parse_rank_layer("Crypto#0").is_none());
    }

    #[test]
    fn manifest_parses_and_defs_pair_up() {
        let m = parse_manifest(r#"{ "cube": "3c2b7a11", "sphere\"q": "ffffffff" }"#);
        assert_eq!(m["cube"], 0x3c2b_7a11);
        assert_eq!(m["sphere\"q"], 0xffff_ffff);
        assert!(parse_manifest("not json").is_empty());
        let json = manifest_json(&["cube", "sphere"]);
        let m2 = parse_manifest(&json);
        assert_eq!(m2["cube"], id_of_name("cube"));
        let key = layer_key("CryptoObject");
        let attrs = vec![
            (format!("cryptomatte/{key}/name"), "CryptoObject".to_string()),
            (format!("cryptomatte/{key}/manifest"), json),
            ("cryptomatte/1234567/name".to_string(), "CryptoMaterial".to_string()),
            ("comment".to_string(), "x".to_string()),
        ];
        let d = defs_from_attrs(&attrs);
        assert_eq!(d.len(), 2);
        assert_eq!(d[0].name, "CryptoMaterial");
        assert!(d[0].manifest.is_empty());
        assert_eq!(d[1].name_of(id_of_name("sphere")), Some("sphere"));
    }

    #[test]
    fn the_matte_layer_names_the_selection_and_parses_back() {
        let mut p = CryptomatteParams { layer: "CryptoObject".into(), ..Default::default() };
        p.toggle("b", 0x0000_00bb, true);
        p.toggle("a", 0xaa00_0001, true);
        p.invert = true;
        let name = p.matte_layer();
        assert_eq!(name, "CryptoObject@matte;inv=1;ids=000000bb,aa000001");
        let spec = parse_matte_layer(&name).unwrap();
        assert_eq!(spec.layer, "CryptoObject");
        assert_eq!(spec.ids, [0xbb, 0xaa00_0001]);
        assert!(spec.invert);
        // The same selection made in the other order is the same cache key.
        let mut q = CryptomatteParams { layer: "CryptoObject".into(), invert: true, ..Default::default() };
        q.toggle("a", 0xaa00_0001, true);
        q.toggle("b", 0x0000_00bb, true);
        assert_eq!(q.matte_layer(), name);
        // Not a matte name; and a rank name is not one either.
        assert!(parse_matte_layer("diffuse").is_none());
        assert!(parse_matte_layer("CryptoObject00#1").is_none());
        assert!(parse_rank_layer(&name).is_none());
        // The arithmetic: picked coverage summed, clamped, inverted.
        let s = MatteSpec { layer: String::new(), ids: vec![1, 2], invert: false };
        assert!((s.matte(&[(1, 0.6), (3, 0.3), (2, 0.1)]) - 0.7).abs() < 1e-6);
        assert_eq!(s.matte(&[(1, 0.8), (2, 0.8)]), 1.0);
        // Readers: the matte alone unviewed, the live set viewed.
        assert!(p.reader_sub_active(MATTE_SUB) && !p.reader_sub_active(0));
        p.previewing = true;
        assert!(!p.reader_sub_active(MATTE_SUB) && p.reader_sub_active(3));
        assert_eq!(p.reader_slots().len(), SOURCES + 1);
    }

    #[test]
    fn pack_writes_halves_and_hides_the_view_downstream() {
        let mut p = CryptomatteParams::default();
        assert!(p.toggle("a", 0xdead_beef, true));
        assert!(!p.toggle("a", 0xdead_beef, true), "already picked");
        let mut v = vec![0.0f32; CryptomatteParams::SLOTS * 4];
        p.pack(&mut v);
        assert_eq!(v[0], 0.0, "not the view target: matte for everyone");
        assert_eq!(v[1], 1.0);
        assert_eq!((v[4] as u32) << 16 | v[5] as u32, 0xdead_beef);
        p.previewing = true;
        p.pack(&mut v);
        assert_eq!(v[0], 1.0);
        assert!(p.toggle("a", 0xdead_beef, false));
        assert!(p.picks.is_empty());
        // A plain click replaces whatever was picked.
        p.toggle("a", 1, true);
        p.toggle("b", 2, true);
        assert!(p.set_only("c", 3));
        assert_eq!(p.picks.len(), 1);
        assert_eq!(p.picks[0].id, 3);
        assert!(!p.set_only("c", 3), "already the only pick");
    }
}
