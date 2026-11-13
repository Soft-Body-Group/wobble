//! Text: a design-app text block drawn over its input (or over nothing,
//! unwired) — typography, outline, shadow and glow, TextMeshPro-style
//! per-glyph animators, and an optional path for the baseline to follow.
//!
//! The typesetting is [`crate::textfx`]; the pixels are a per-node layer
//! [`crate::textgpu`] draws from glyph instances into a texture the fused
//! pass samples (`SourceBinding::Text`). So this node's WGSL is one texture
//! read and an over: nothing about the text — words, font, any animator,
//! the path — is topology, and nothing ever recompiles. The layer is
//! re-recorded only when what it would draw changes.
//!
//! `position` is the block's anchor at its alignment point: by default the
//! TOP (`valign`) of the first line at its left / centre / right, canvas
//! pixels, y down. `size` is the em in canvas pixels. Lengths in ems
//! (weight, outline, glow, shadow softness) scale with the text; offsets
//! in pixels do not. Output is premultiplied; colours are authored
//! straight in the working space.

use crate::codegen::Emit;
use crate::geotext::{TextAlign, TextFont};
use crate::nodes::NodeDef;
use crate::textfx::{self, Atlas, Instance, Laid, TextAnimator, TextCase, TextPath, Uniform, VAlign, EM, SPREAD};
use std::sync::Arc;

pub use crate::textfx::{PathPoint, RangeShape, TextUnit};

/// Topology tag; the whole node is one code path.
const TOPO_TAG: u64 = 0x7E48;

/// Channel names (after `text.`) of the node-wide numbers, in panel order.
/// Animators add `a<i>.<name>` for [`textfx::ANIM_PARAMS`].
pub const PARAMS: &[&str] = &[
    "tracking",
    "leading",
    "wrap",
    "rotation",
    "scale.x",
    "scale.y",
    "skew",
    "color.r",
    "color.g",
    "color.b",
    "color.a",
    "weight",
    "softness",
    "outline",
    "outline.r",
    "outline.g",
    "outline.b",
    "outline.a",
    "shadow.x",
    "shadow.y",
    "shadow.soft",
    "shadow.r",
    "shadow.g",
    "shadow.b",
    "shadow.a",
    "glow.size",
    "glow.r",
    "glow.g",
    "glow.b",
    "glow.a",
    "path.offset",
];

/// The port that takes the words off a String node (`PortKind::Text`):
/// wired, they replace `text`. The app resolves it into the engine graph's
/// copy (`to_engine_graph_at_f`); the shader never reads it.
pub const WORDS_INPUT: usize = 1;

/// The prefix every Text channel name carries.
pub const PREFIX: &str = "text.";

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(default))]
pub struct TextParams {
    /// What it says; `\n` starts a paragraph.
    pub text: String,
    pub font: TextFont,
    pub align: TextAlign,
    /// Baseline to baseline, in ems.
    pub leading: f32,
    /// The block's anchor (see `valign`), canvas pixels (y down).
    pub position: [f32; 2],
    /// The em, canvas pixels.
    pub size: f32,
    /// Fill: straight colour in the working space, and alpha.
    pub color: [f32; 4],
    /// Thousandths of an em after every glyph.
    pub tracking: f32,
    pub kerning: bool,
    pub case: TextCase,
    pub valign: VAlign,
    /// Paragraph box width, canvas px; 0 = no wrapping.
    pub wrap: f32,
    /// Wrapped lines (not a paragraph's last) fill the box.
    pub justify: bool,
    /// The block about its position: degrees, factors, a lean in degrees.
    pub rotation: f32,
    pub scale: [f32; 2],
    pub skew: f32,
    /// Ems of stroke weight (a faux bold; negative thins).
    pub weight: f32,
    /// Canvas px of edge softness.
    pub softness: f32,
    /// Ems of outline (a ring round the fill); 0 = none.
    pub outline: f32,
    pub outline_color: [f32; 4],
    pub shadow: bool,
    /// Canvas px.
    pub shadow_offset: [f32; 2],
    /// Ems.
    pub shadow_soft: f32,
    pub shadow_color: [f32; 4],
    pub glow: bool,
    /// Ems.
    pub glow_size: f32,
    pub glow_color: [f32; 4],
    /// Per-glyph range selectors, applied in order.
    pub animators: Vec<TextAnimator>,
    /// The baseline follows this when on.
    pub path: TextPath,
    /// Transient: the frame being rendered (wiggles read it).
    #[cfg_attr(feature = "serde", serde(skip))]
    pub time: f32,
}

impl Default for TextParams {
    fn default() -> Self {
        Self {
            text: "wobble".into(),
            font: TextFont::Sans,
            align: TextAlign::Centre,
            leading: 1.2,
            position: [960.0, 470.0],
            size: 120.0,
            color: [1.0, 1.0, 1.0, 1.0],
            tracking: 0.0,
            kerning: true,
            case: TextCase::AsTyped,
            valign: VAlign::Top,
            wrap: 0.0,
            justify: false,
            rotation: 0.0,
            scale: [1.0, 1.0],
            skew: 0.0,
            weight: 0.0,
            softness: 0.0,
            outline: 0.0,
            outline_color: [0.0, 0.0, 0.0, 1.0],
            shadow: false,
            shadow_offset: [6.0, 6.0],
            shadow_soft: 0.05,
            shadow_color: [0.0, 0.0, 0.0, 0.6],
            glow: false,
            glow_size: 0.1,
            glow_color: [1.0, 0.8, 0.3, 0.8],
            animators: Vec::new(),
            path: TextPath::default(),
            time: 0.0,
        }
    }
}

/// What the layer draws: the atlas, the instances, the pass uniform, each
/// glyph's quad in canvas px, and a hash of all of it.
pub struct Built {
    pub atlas: Arc<Atlas>,
    pub instances: Vec<Instance>,
    pub uniform: Uniform,
    pub quads: Vec<[[f32; 2]; 4]>,
    pub hash: u64,
}

impl TextParams {
    /// The words as drawn (cased).
    pub fn shown(&self) -> String {
        textfx::cased(&self.text, self.case)
    }

    /// The block laid out (ems); `None` when the font will not load.
    pub fn laid(&self) -> Option<Laid> {
        let shown = self.shown();
        let size = self.size.max(1e-3);
        textfx::layout(&textfx::LayoutSpec {
            text: &shown,
            font: &self.font,
            tracking: self.tracking,
            kerning: self.kerning,
            leading: self.leading,
            wrap: self.wrap.max(0.0) / size,
            justify: self.justify,
            valign: self.valign,
        })
    }

    /// The UNANIMATED block's top-left and size in canvas pixels (before
    /// the block's rotation, scale and the path) — what a test or a
    /// viewer box frames.
    pub fn rect(&self) -> Option<([f32; 2], [f32; 2])> {
        let l = self.laid()?;
        let w = l.lines.iter().map(|x| x.0).fold(0.0f32, f32::max) * self.size;
        let n = l.lines.len().max(1) as f32;
        let h = (l.ascent + l.descent + (n - 1.0) * l.leading) * self.size;
        let shift = match self.align {
            TextAlign::Left => 0.0,
            TextAlign::Centre => 0.5,
            TextAlign::Right => 1.0,
        };
        Some(([self.position[0] - shift * w, self.position[1] + l.top * self.size], [w, h]))
    }

    /// Everything the layer needs at `proxy` (canvas px -> layer px) and
    /// `fps` (the wiggles' clock).
    pub fn build(&self, proxy: f32, fps: f32) -> Option<Built> {
        let shown = self.shown();
        let atlas = textfx::atlas(&self.font, &shown)?;
        let laid = self.laid()?;
        let look = textfx::Look {
            size: self.size,
            position: self.position,
            align: self.align,
            rotation: self.rotation,
            scale: self.scale,
            skew: self.skew,
            color: self.color,
            weight: self.weight,
            softness: self.softness,
            outline: self.outline.max(0.0),
            outline_color: self.outline_color,
            anims: &self.animators,
            path: Some(&self.path),
            seconds: self.time / if fps > 0.0 { fps } else { 24.0 },
        };
        let (instances, quads) = textfx::instances(&laid, &atlas, &look, proxy);
        let room = (SPREAD - 2.0 - (self.outline.max(0.0) * EM).min(SPREAD - 2.0)).max(1.0);
        let mut uniform = Uniform::default();
        if self.shadow {
            uniform.shadow = [
                self.shadow_offset[0] * proxy,
                self.shadow_offset[1] * proxy,
                (self.shadow_soft.max(0.0) * EM).min(room),
                1.0,
            ];
            uniform.shadow_col = self.shadow_color;
        }
        if self.glow {
            uniform.glow = [(self.glow_size.max(0.0) * EM).clamp(0.5, room), 1.0, 0.0, 0.0];
            uniform.glow_col = self.glow_color;
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};
        atlas.hash.hash(&mut h);
        bytemuck::cast_slice::<Instance, u8>(&instances).hash(&mut h);
        bytemuck::bytes_of(&uniform).hash(&mut h);
        Some(Built { atlas, instances, uniform, quads, hash: h.finish() })
    }

    /// A number by its channel name after [`PREFIX`].
    pub fn get(&self, name: &str) -> Option<f32> {
        if let Some((i, rest)) = anim_name(name) {
            return self.animators.get(i)?.get(rest);
        }
        let c4 = |c: &[f32; 4], s: &str| -> Option<f32> {
            Some(c[match s {
                "r" => 0,
                "g" => 1,
                "b" => 2,
                "a" => 3,
                _ => return None,
            }])
        };
        Some(match name {
            "tracking" => self.tracking,
            "leading" => self.leading,
            "wrap" => self.wrap,
            "rotation" => self.rotation,
            "scale.x" => self.scale[0],
            "scale.y" => self.scale[1],
            "skew" => self.skew,
            "weight" => self.weight,
            "softness" => self.softness,
            "outline" => self.outline,
            "shadow.x" => self.shadow_offset[0],
            "shadow.y" => self.shadow_offset[1],
            "shadow.soft" => self.shadow_soft,
            "glow.size" => self.glow_size,
            "path.offset" => self.path.offset,
            _ => {
                let (head, lane) = name.rsplit_once('.')?;
                return match head {
                    "color" => c4(&self.color, lane),
                    "outline" => c4(&self.outline_color, lane),
                    "shadow" => c4(&self.shadow_color, lane),
                    "glow" => c4(&self.glow_color, lane),
                    _ => None,
                };
            }
        })
    }

    /// Set a number by its channel name after [`PREFIX`]; false if none.
    pub fn set(&mut self, name: &str, v: f32) -> bool {
        if let Some((i, rest)) = anim_name(name) {
            return self.animators.get_mut(i).is_some_and(|a| a.set(rest, v));
        }
        let f: &mut f32 = match name {
            "tracking" => &mut self.tracking,
            "leading" => &mut self.leading,
            "wrap" => &mut self.wrap,
            "rotation" => &mut self.rotation,
            "scale.x" => &mut self.scale[0],
            "scale.y" => &mut self.scale[1],
            "skew" => &mut self.skew,
            "weight" => &mut self.weight,
            "softness" => &mut self.softness,
            "outline" => &mut self.outline,
            "shadow.x" => &mut self.shadow_offset[0],
            "shadow.y" => &mut self.shadow_offset[1],
            "shadow.soft" => &mut self.shadow_soft,
            "glow.size" => &mut self.glow_size,
            "path.offset" => &mut self.path.offset,
            _ => {
                let Some((head, lane)) = name.rsplit_once('.') else { return false };
                let c = match head {
                    "color" => &mut self.color,
                    "outline" => &mut self.outline_color,
                    "shadow" => &mut self.shadow_color,
                    "glow" => &mut self.glow_color,
                    _ => return false,
                };
                let k = match lane {
                    "r" => 0,
                    "g" => 1,
                    "b" => 2,
                    "a" => 3,
                    _ => return false,
                };
                &mut c[k]
            }
        };
        *f = v;
        true
    }

    /// Every channel name this node has now (after [`PREFIX`]).
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = PARAMS.iter().map(|s| s.to_string()).collect();
        for i in 0..self.animators.len() {
            v.extend(textfx::ANIM_PARAMS.iter().map(|p| format!("a{i}.{p}")));
        }
        v
    }
}

/// `a<i>.<rest>` -> (i, rest).
pub fn anim_name(name: &str) -> Option<(usize, &str)> {
    let (head, rest) = name.split_once('.')?;
    let i = head.strip_prefix('a')?.parse().ok()?;
    Some((i, rest))
}

impl NodeDef for TextParams {
    const NAME: &'static str = "Text";
    /// The picture the text lands on, and the words off a String node
    /// ([`WORDS_INPUT`], the text pipe — read on the CPU, never sampled).
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = None;
    const VEC4_SLOTS: usize = 0;

    /// The text is a layer: one code path.
    fn topo_tag(&self) -> u64 {
        TOPO_TAG
    }

    fn pack(&self, _v: &mut [f32]) {}

    fn emit(&self, e: &mut Emit) -> String {
        let under = e.input(0);
        let pos = e.pos().to_string();
        let t = e.text_table();
        let (ink, out) = (e.fresh(), e.fresh());
        e.line(format_args!("let {ink} = sample_px({t}, {pos});"));
        e.line(format_args!("let {out} = {ink} + {under} * (1.0 - {ink}.a);"));
        out
    }
}

/// Register the shared coverage helper in the shader's function map (the
/// Burnin's distance-field read).
pub(crate) fn text_helper(e: &mut Emit) {
    let (_, fns) = e.color_helpers();
    fns.entry("text_cov".into()).or_insert_with(crate::textraster::text_wgsl);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_about_the_text_is_topology() {
        let a = TextParams::default();
        let mut b = TextParams {
            text: "else".into(),
            position: [1.0, 2.0],
            size: 9.0,
            color: [0.0, 1.0, 0.0, 0.5],
            outline: 0.05,
            ..a.clone()
        };
        b.animators.push(TextAnimator::default());
        assert_eq!(a.topo_tag(), b.topo_tag());
    }

    #[test]
    fn proxy_scales_the_layer_not_the_layout() {
        let p = TextParams::default();
        let full = p.build(1.0, 24.0).unwrap();
        let half = p.build(0.5, 24.0).unwrap();
        assert_eq!(full.quads, half.quads, "canvas px either way");
        assert!((half.instances[0].t[0] - full.instances[0].t[0] * 0.5).abs() < 1e-3);
        assert!((half.instances[0].m[0] - full.instances[0].m[0] * 0.5).abs() < 1e-5);
        assert_ne!(full.hash, half.hash);
    }

    #[test]
    fn a_centred_block_is_centred_on_its_position() {
        let p = TextParams::default();
        let (tl, wh) = p.rect().unwrap();
        assert!((tl[0] + wh[0] * 0.5 - p.position[0]).abs() < 1e-3);
        assert_eq!(tl[1], p.position[1]);
        assert!(wh[0] > p.size, "six letters are wider than an em");
        let mid = TextParams { valign: VAlign::Middle, ..p.clone() };
        let (tl, wh) = mid.rect().unwrap();
        assert!((tl[1] + wh[1] * 0.5 - p.position[1]).abs() < 1e-3);
    }

    #[test]
    fn every_channel_name_reads_back_what_it_set() {
        let mut p = TextParams::default();
        p.animators.push(TextAnimator::default());
        for (i, n) in p.names().iter().enumerate() {
            let v = i as f32 * 0.25 + 0.125;
            assert!(p.set(n, v), "{n}");
            assert_eq!(p.get(n), Some(v), "{n}");
        }
        assert!(!p.set("a3.start", 1.0), "no such animator");
        assert!(!p.set("nonsense", 1.0));
    }
}
