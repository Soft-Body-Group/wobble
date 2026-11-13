//! Burnin: the frame number printed into the picture — `frame 1001` in a
//! corner, on an optional dark box, over its input.
//!
//! It shares the Text node's rendering ([`crate::textraster`], the
//! `text_cov` helper, a per-node atlas behind `SourceBinding::Text`), but
//! the atlas holds the text before and after `{frame}` and one CELL per
//! glyph `0`–`9` and `-`. The frame is a uniform and the shader works out
//! the digits and picks their cells, so a new frame is no re-raster, no
//! upload and no recompile — only an edit to the words is. The corner, the
//! margin, the size, the box and the zero padding are uniforms too.
//!
//! The frame is the tick being rendered (after any time node above it, like
//! a Grain's pattern), plus `offset`; the app resolves it into the
//! transient `time` because the engine does not know what time it is.

use crate::codegen::Emit;
use crate::geotext::TextFont;
use crate::nodes::NodeDef;
use crate::textraster::{self, Raster};
use std::sync::Arc;

const TOPO_TAG: u64 = 0xB081;
/// The token the format string's number replaces.
pub const TOKEN: &str = "{frame}";

/// Where the burn-in sits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BurninCorner {
    TopLeft,
    TopRight,
    #[default]
    BottomLeft,
    BottomRight,
    /// At `position` (the text's top-left, canvas pixels).
    Custom,
}

impl BurninCorner {
    pub const ALL: [BurninCorner; 5] = [
        BurninCorner::TopLeft,
        BurninCorner::TopRight,
        BurninCorner::BottomLeft,
        BurninCorner::BottomRight,
        BurninCorner::Custom,
    ];
    pub fn name(self) -> &'static str {
        match self {
            BurninCorner::TopLeft => "top left",
            BurninCorner::TopRight => "top right",
            BurninCorner::BottomLeft => "bottom left",
            BurninCorner::BottomRight => "bottom right",
            BurninCorner::Custom => "custom",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct BurninParams {
    /// The words, with [`TOKEN`] where the number goes (no token: no
    /// number). One line.
    pub format: String,
    pub font: TextFont,
    /// At least this many digits, zero-filled.
    pub pad: u32,
    /// Added to the tick before it is printed.
    pub offset: i32,
    pub corner: BurninCorner,
    /// From the frame's edge to the box's, canvas pixels.
    pub margin: f32,
    /// The text's top-left for [`BurninCorner::Custom`], canvas pixels.
    pub position: [f32; 2],
    /// The em, canvas pixels.
    pub size: f32,
    /// Straight colour and alpha of the text.
    pub color: [f32; 4],
    /// Draw the box behind it.
    pub boxed: bool,
    /// Straight colour and alpha of the box.
    pub box_color: [f32; 4],
    /// The tick being rendered. TRANSIENT: resolved per frame by the app.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub time: f32,
}

impl Default for BurninParams {
    fn default() -> Self {
        Self {
            format: format!("frame {TOKEN}"),
            font: TextFont::Mono,
            pad: 4,
            offset: 0,
            corner: BurninCorner::BottomLeft,
            margin: 32.0,
            position: [100.0, 100.0],
            size: 40.0,
            color: [1.0, 1.0, 1.0, 1.0],
            boxed: true,
            box_color: [0.0, 0.0, 0.0, 0.6],
            time: 0.0,
        }
    }
}

impl BurninParams {
    /// The words either side of the number, and whether there is one.
    pub fn split(&self) -> (&str, &str, bool) {
        let line = self.format.lines().next().unwrap_or("");
        match line.split_once(TOKEN) {
            Some((a, b)) => (a, b, true),
            None => (line, "", false),
        }
    }

    pub fn raster(&self) -> Option<Arc<Raster>> {
        let (a, b, _) = self.split();
        textraster::burnin(a, b, &self.font)
    }

    /// The number this frame prints.
    pub fn frame(&self) -> i64 {
        self.time.round() as i64 + self.offset as i64
    }

    /// What it reads at frame `f` — the CPU mirror of the shader's digits.
    pub fn text_at(&self, f: i64) -> String {
        let (a, b, num) = self.split();
        if !num {
            return a.to_string();
        }
        let digits = format!("{:0width$}", f.unsigned_abs(), width = self.pad.min(12) as usize);
        format!("{a}{}{digits}{b}", if f < 0 { "-" } else { "" })
    }

    /// The box's padding round the text, canvas pixels.
    pub fn box_pad(&self) -> f32 {
        if self.boxed { self.size * 0.25 } else { 0.0 }
    }

    /// The uniform span, under `proxy`, in a `fmt`-pixel format.
    pub fn pack_fmt(&self, v: &mut [f32], fmt: [f32; 2], proxy: f32) {
        v.iter_mut().for_each(|x| *x = 0.0);
        v[0..4].copy_from_slice(&self.color);
        if self.boxed {
            v[4..8].copy_from_slice(&self.box_color);
        }
        let Some(r) = self.raster() else { return };
        let (_, _, num) = self.split();
        v[8] = self.frame() as f32;
        v[9] = self.pad.min(12) as f32;
        v[10] = self.corner as u32 as f32;
        v[11] = self.margin * proxy;
        v[12] = self.position[0] * proxy;
        v[13] = self.position[1] * proxy;
        v[14] = (self.size * proxy / r.em).max(0.0);
        v[15] = self.box_pad() * proxy;
        v[16] = fmt[0];
        v[17] = fmt[1];
        v[18] = if num { 1.0 } else { 0.0 };
        v[20] = r.prefix[0];
        v[21] = r.prefix[1];
        v[22] = r.suffix[0];
        v[23] = r.suffix[1];
        v[24] = r.cells_x;
        v[25] = r.cell_w;
        v[26] = r.pitch;
        v[27] = r.origin[1];
        v[28] = r.ascent;
        v[29] = r.descent;
    }
}

const BURNIN_WGSL: &str = "// Burnin: the number's digits worked out from the frame uniform, each
// drawn from its cell of the node's atlas, after the prefix and before the
// suffix; laid in a corner over an optional box, over what is under it.
fn burnin_px(t: texture_2d<f32>, p: vec2f, slot: u32, under: vec4f) -> vec4f {
    let col = P.v[slot];
    let bc = P.v[slot + 1u];
    let h2 = P.v[slot + 2u];
    let h3 = P.v[slot + 3u];
    let h4 = P.v[slot + 4u];
    let h5 = P.v[slot + 5u];
    let h6 = P.v[slot + 6u];
    let h7 = P.v[slot + 7u];
    let s = max(h3.z, 1e-6);
    let f = i32(round(h2.x));
    let neg = f < 0;
    let mag = u32(abs(f));
    var nd = 1u;
    var q = mag / 10u;
    for (var i = 0u; i < 12u && q > 0u; i = i + 1u) { nd = nd + 1u; q = q / 10u; }
    var n = max(nd, u32(h2.y));
    if (neg) { n = n + 1u; }
    if (h4.z < 0.5) { n = 0u; }
    let num_w = f32(n) * h6.y;
    let w = h5.y + num_w + h5.w;
    let ext = vec2f(w, h7.x + h7.y) * s;
    let bp = h3.w;
    let m = h2.w;
    let corner = u32(h2.z);
    var o = h3.xy;
    let l0 = m + bp;
    if (corner == 0u) { o = vec2f(l0, l0); }
    if (corner == 1u) { o = vec2f(h4.x - l0 - ext.x, l0); }
    if (corner == 2u) { o = vec2f(l0, h4.y - l0 - ext.y); }
    if (corner == 3u) { o = vec2f(h4.x - l0 - ext.x, h4.y - l0 - ext.y); }
    // The box, its edges one pixel soft.
    let d4 = vec4f(p - (o - bp), (o + ext + bp) - p);
    let box_cov = clamp(min(min(d4.x, d4.y), min(d4.z, d4.w)) + 0.5, 0.0, 1.0);
    let boxl = vec4f(bc.rgb * bc.a, bc.a) * box_cov;
    let c = boxl + under * (1.0 - boxl.a);
    // Which piece of the line this pixel falls in, as an atlas x.
    let l = (p - o) / s;
    var ax = -1e9;
    if (l.x >= 0.0 && l.x < h5.y) {
        ax = h5.x + l.x;
    } else if (l.x >= h5.y && l.x < h5.y + num_w) {
        let k = u32(floor((l.x - h5.y) / h6.y));
        let lx = l.x - h5.y - f32(k) * h6.y;
        var cell = 10u;
        if (!(neg && k == 0u)) {
            var v = mag;
            let e = n - 1u - k;
            for (var i = 0u; i < e; i = i + 1u) { v = v / 10u; }
            cell = v % 10u;
        }
        ax = h6.x + f32(cell) * h6.z + lx;
    } else if (l.x >= h5.y + num_w && l.x < w) {
        ax = h5.z + (l.x - h5.y - num_w);
    }
    let cov = text_cov(t, vec2f(ax, h6.w + l.y), s);
    let ink = vec4f(col.rgb * col.a, col.a) * cov;
    return ink + c * (1.0 - ink.a);
}
";

impl NodeDef for BurninParams {
    const NAME: &'static str = "Burnin";
    const INPUTS: usize = 1;
    const MASK_INPUT: Option<usize> = None;
    /// colour, box colour, (frame, pad, corner, margin), (position, scale,
    /// box pad), (format, has number, -), prefix/suffix, cells, metrics.
    const VEC4_SLOTS: usize = 8;

    fn topo_tag(&self) -> u64 {
        TOPO_TAG
    }

    fn pack(&self, v: &mut [f32]) {
        self.pack_fmt(v, [1920.0, 1080.0], 1.0);
    }

    fn emit(&self, e: &mut Emit) -> String {
        let under = e.input(0);
        let slot = e.slot();
        let pos = e.pos().to_string();
        let t = e.text_table();
        crate::nodes::text::text_helper(e);
        let (_, fns) = e.color_helpers();
        fns.entry("burnin_px".into()).or_insert_with(|| BURNIN_WGSL.to_string());
        let v = e.fresh();
        e.line(format_args!("let {v} = burnin_px({t}, {pos}, {slot}u, {under});"));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_format_reads_like_the_shader_prints() {
        let p = BurninParams::default();
        assert_eq!(p.text_at(1001), "frame 1001");
        assert_eq!(p.text_at(7), "frame 0007");
        assert_eq!(p.text_at(-3), "frame -0003");
        let q = BurninParams { format: "[{frame}] shot".into(), pad: 0, ..p.clone() };
        assert_eq!(q.text_at(12), "[12] shot");
        let r = BurninParams { format: "no number".into(), ..p };
        assert_eq!(r.text_at(5), "no number");
    }

    #[test]
    fn a_new_frame_is_a_uniform_not_topology() {
        let a = BurninParams::default();
        let b = BurninParams { time: 1001.0, corner: BurninCorner::TopRight, ..a.clone() };
        assert_eq!(a.topo_tag(), b.topo_tag());
        let (mut va, mut vb) = ([0.0; 32], [0.0; 32]);
        a.pack(&mut va);
        b.pack(&mut vb);
        assert_eq!(vb[8], 1001.0);
        assert_eq!(va[20..30], vb[20..30], "same words, same atlas");
    }
}
