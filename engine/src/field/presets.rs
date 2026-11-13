//! Stock noise graphs a Noise node can load — a starting point one click
//! away, like a Grain node's film stocks.
//!
//! A preset is only a [`FieldGraph`]: loading one REPLACES the node's graph
//! and leaves its scale, seed, speed and format alone, so a look carries the
//! framing already dialled in. Every preset exposes a slider or two (the
//! numbers worth reaching for without diving in), and lays its nodes out top
//! to bottom so the graph reads when opened.

use super::*;

/// A look the Noise panel can load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoisePreset {
    Clouds,
    Blobs,
    Ridges,
    Liquid,
    Marble,
    Smoke,
    WoodGrain,
    Cells,
    Cracks,
    Caustics,
    Mosaic,
    Dots,
}

impl NoisePreset {
    pub const ALL: [NoisePreset; 12] = [
        NoisePreset::Clouds,
        NoisePreset::Blobs,
        NoisePreset::Ridges,
        NoisePreset::Liquid,
        NoisePreset::Marble,
        NoisePreset::Smoke,
        NoisePreset::WoodGrain,
        NoisePreset::Cells,
        NoisePreset::Cracks,
        NoisePreset::Caustics,
        NoisePreset::Mosaic,
        NoisePreset::Dots,
    ];

    pub fn label(self) -> &'static str {
        match self {
            NoisePreset::Clouds => "clouds",
            NoisePreset::Blobs => "blobs",
            NoisePreset::Ridges => "ridges",
            NoisePreset::Liquid => "liquid",
            NoisePreset::Marble => "marble",
            NoisePreset::Smoke => "smoke",
            NoisePreset::WoodGrain => "wood grain",
            NoisePreset::Cells => "cells",
            NoisePreset::Cracks => "cracks",
            NoisePreset::Caustics => "caustics",
            NoisePreset::Mosaic => "mosaic",
            NoisePreset::Dots => "dots",
        }
    }

    /// One line for the menu's hover.
    pub fn blurb(self) -> &'static str {
        match self {
            NoisePreset::Clouds => "soft fBm — size, roughness",
            NoisePreset::Blobs => "thresholded fBm: islands with a soft edge — threshold, sharpness",
            NoisePreset::Ridges => "ridged fBm, mountain-range creases — size, roughness",
            NoisePreset::Liquid => "fBm pushed around by two more fBms — warp",
            NoisePreset::Marble => "dark veins on white, warped — warp, veins",
            NoisePreset::Smoke => "turbulence stirred by a curl warp — swirl",
            NoisePreset::WoodGrain => "long streaks with a lazy wave through them — waviness",
            NoisePreset::Cells => "voronoi distance to the nearest point — size, jitter",
            NoisePreset::Cracks => "dry-mud plates split by dark cracks — size, width",
            NoisePreset::Caustics => "bright wobbling cell walls; animate with speed — size, sharpness",
            NoisePreset::Mosaic => "a flat grey per cell with grout between — size, grout",
            NoisePreset::Dots => "a grid of soft dots — size, radius, softness, jitter",
        }
    }

    /// The graph itself: ids from 1, an explicit Output at the bottom.
    pub fn graph(self) -> FieldGraph {
        let mut b = Build::default();
        let out = match self {
            NoisePreset::Clouds => {
                let size = b.slider("size", 0.35, 0.0, 0.0);
                let rough = b.slider("roughness", 0.5, 1.0, 0.0);
                let f = b.add(fbm(FieldFbmMode::Fbm, 0.35, 6, 0.0), 0.5, 1.0);
                b.wire(f, &[0, 1], size);
                b.wire(f, &[5], rough);
                f
            }
            NoisePreset::Blobs => {
                // 0.5 + (fbm - threshold) * sharpness * 5, clamped: a
                // threshold that stays soft enough to read as a shape.
                let thr = b.slider("threshold", 0.5, 1.0, 0.0);
                let sharp = b.slider("sharpness", 2.0, 2.0, 0.0);
                let f = b.add(fbm(FieldFbmMode::Fbm, 0.2, 4, 0.0), 0.0, 1.0);
                let sub = b.add(math(MathOp::Sub), 0.5, 2.0);
                b.wire(sub, &[0], f);
                b.wire(sub, &[1], thr);
                let mul = b.add(math(MathOp::Mul), 1.0, 3.0);
                b.wire(mul, &[0], sub);
                b.wire(mul, &[1], sharp);
                let r = b.add(remap(-0.1, 0.1, 0.0, 1.0), 1.0, 4.0);
                b.wire(r, &[0], mul);
                r
            }
            NoisePreset::Ridges => {
                let size = b.slider("size", 0.4, 0.0, 0.0);
                let rough = b.slider("roughness", 0.5, 1.0, 0.0);
                let f = b.add(fbm(FieldFbmMode::Ridged, 0.4, 7, 0.0), 0.5, 1.0);
                b.wire(f, &[0, 1], size);
                b.wire(f, &[5], rough);
                f
            }
            NoisePreset::Liquid => {
                // The warp in Inigo Quilez's sense: the picture is an fBm
                // read at a point that two other fBms have moved.
                let amt = b.slider("warp", 0.3, 2.0, 0.0);
                let base = b.add(fbm(FieldFbmMode::Fbm, 0.3, 5, 0.0), 0.0, 1.0);
                let wx = b.add(fbm(FieldFbmMode::Fbm, 0.3, 4, 11.0), 1.0, 1.0);
                let wy = b.add(fbm(FieldFbmMode::Fbm, 0.3, 4, 23.0), 2.0, 1.0);
                let w = b.add(FieldNode::Warp { mode: WarpMode::Offset, amount: 0.3 }, 1.0, 2.0);
                b.wire(w, &[0], base);
                b.wire(w, &[1], wx);
                b.wire(w, &[2], wy);
                b.wire(w, &[3], amt);
                w
            }
            NoisePreset::Marble => {
                // Ridged noise is bright along the zero crossings of the
                // noise under it — veins. Warp them so they flow, sharpen
                // with a power, and flip so they are dark on white.
                let amt = b.slider("warp", 0.35, 3.0, 0.0);
                let veins = b.slider("veins", 3.0, 3.0, 2.0);
                let ridge = b.add(fbm(FieldFbmMode::Ridged, 0.5, 5, 0.0), 0.0, 1.0);
                let wx = b.add(fbm(FieldFbmMode::Fbm, 0.4, 4, 5.0), 1.0, 1.0);
                let wy = b.add(fbm(FieldFbmMode::Fbm, 0.4, 4, 9.0), 2.0, 1.0);
                let w = b.add(FieldNode::Warp { mode: WarpMode::Offset, amount: 0.35 }, 1.0, 2.0);
                b.wire(w, &[0], ridge);
                b.wire(w, &[1], wx);
                b.wire(w, &[2], wy);
                b.wire(w, &[3], amt);
                let pow = b.add(math(MathOp::Pow), 1.0, 3.0);
                b.wire(pow, &[0], w);
                b.wire(pow, &[1], veins);
                let r = b.add(remap(0.0, 1.0, 0.95, 0.15), 1.0, 4.0);
                b.wire(r, &[0], pow);
                r
            }
            NoisePreset::Smoke => {
                // A curl warp is divergence-free: it swirls the turbulence
                // without pooling or tearing it, which is what smoke does.
                let amt = b.slider("swirl", 0.015, 2.0, 0.0);
                let turb = b.add(fbm(FieldFbmMode::Turbulence, 0.3, 6, 0.0), 0.0, 1.0);
                let pot = b.add(fbm(FieldFbmMode::Fbm, 0.6, 2, 7.0), 1.0, 1.0);
                let w = b.add(FieldNode::Warp { mode: WarpMode::Curl, amount: 0.015 }, 1.0, 2.0);
                b.wire(w, &[0], turb);
                b.wire(w, &[1], pot);
                b.wire(w, &[2], amt);
                w
            }
            NoisePreset::WoodGrain => {
                // Stretched fBm is streaks; a coarse fBm bending y only
                // gives them the lazy wave grain has.
                let amt = b.slider("waviness", 0.06, 2.0, 0.0);
                let mut streak = fbm(FieldFbmMode::Fbm, 1.0, 3, 0.0);
                if let FieldNode::Fbm { scale, .. } = &mut streak {
                    *scale = [1.0, 0.02];
                }
                let s = b.add(streak, 0.0, 1.0);
                let bend = b.add(fbm(FieldFbmMode::Fbm, 0.3, 3, 17.0), 1.0, 1.0);
                let w = b.add(FieldNode::Warp { mode: WarpMode::Offset, amount: 0.06 }, 1.0, 2.0);
                b.wire(w, &[0], s);
                b.wire(w, &[2], bend);
                b.wire(w, &[3], amt);
                w
            }
            NoisePreset::Cells => {
                let size = b.slider("size", 0.08, 0.0, 0.0);
                let jit = b.slider("jitter", 1.0, 1.0, 0.0);
                let v = b.add(voronoi(VoronoiMode::F1, 0.08, 1.0, 0.0), 0.5, 1.0);
                b.wire(v, &[0, 1], size);
                b.wire(v, &[2], jit);
                v
            }
            NoisePreset::Cracks => {
                // F2 - F1 is 0 on a cell wall: divide by the crack width and
                // clamp, and everything wider than a crack is plate.
                let size = b.slider("size", 0.1, 0.0, 0.0);
                let width = b.slider("width", 0.08, 2.0, 2.0);
                let v = b.add(voronoi(VoronoiMode::Ridges, 0.1, 1.0, 0.0), 0.0, 1.0);
                b.wire(v, &[0, 1], size);
                let wx = b.add(fbm(FieldFbmMode::Fbm, 0.04, 3, 3.0), 1.0, 1.0);
                let wy = b.add(fbm(FieldFbmMode::Fbm, 0.04, 3, 8.0), 2.0, 1.0);
                let w = b.add(FieldNode::Warp { mode: WarpMode::Offset, amount: 0.015 }, 1.0, 2.0);
                b.wire(w, &[0], v);
                b.wire(w, &[1], wx);
                b.wire(w, &[2], wy);
                let div = b.add(math(MathOp::Div), 1.0, 3.0);
                b.wire(div, &[0], w);
                b.wire(div, &[1], width);
                let r = b.add(remap(0.0, 1.0, 0.0, 1.0), 1.0, 4.0);
                b.wire(r, &[0], div);
                r
            }
            NoisePreset::Caustics => {
                // Cell walls lit bright and thinned with a power; the
                // Voronoi's points orbit with the node's speed.
                let size = b.slider("size", 0.12, 0.0, 0.0);
                let sharp = b.slider("sharpness", 2.5, 2.0, 2.0);
                let v = b.add(voronoi(VoronoiMode::Ridges, 0.12, 1.0, 0.0), 0.0, 1.0);
                b.wire(v, &[0, 1], size);
                let wx = b.add(fbm(FieldFbmMode::Fbm, 0.25, 3, 4.0), 1.0, 1.0);
                let wy = b.add(fbm(FieldFbmMode::Fbm, 0.25, 3, 6.0), 2.0, 1.0);
                let w = b.add(FieldNode::Warp { mode: WarpMode::Offset, amount: 0.04 }, 1.0, 2.0);
                b.wire(w, &[0], v);
                b.wire(w, &[1], wx);
                b.wire(w, &[2], wy);
                let r = b.add(remap(0.0, 0.12, 1.0, 0.0), 1.0, 3.0);
                b.wire(r, &[0], w);
                let pow = b.add(math(MathOp::Pow), 1.0, 4.0);
                b.wire(pow, &[0], r);
                b.wire(pow, &[1], sharp);
                pow
            }
            NoisePreset::Mosaic => {
                // Two Voronois on the same lattice: one flat grey per cell,
                // one to darken the walls between them.
                let size = b.slider("size", 0.06, 0.0, 0.0);
                let grout = b.slider("grout", 0.04, 2.0, 1.0);
                let cell = b.add(voronoi(VoronoiMode::Cell, 0.06, 0.9, 0.0), 0.5, 1.0);
                let wall = b.add(voronoi(VoronoiMode::Ridges, 0.06, 0.9, 0.0), 1.5, 1.0);
                b.wire(cell, &[0, 1], size);
                b.wire(wall, &[0, 1], size);
                let div = b.add(math(MathOp::Div), 1.5, 2.0);
                b.wire(div, &[0], wall);
                b.wire(div, &[1], grout);
                let r = b.add(remap(0.0, 1.0, 0.0, 1.0), 1.5, 3.0);
                b.wire(r, &[0], div);
                let mul = b.add(math(MathOp::Mul), 1.0, 4.0);
                b.wire(mul, &[0], cell);
                b.wire(mul, &[1], r);
                mul
            }
            NoisePreset::Dots => {
                // (F1 - radius) / softness, clamped and flipped: 1 inside
                // each dot. Jitter 0 is a grid; raise it to scatter them.
                let size = b.slider("size", 0.05, 0.0, 0.0);
                let rad = b.slider("radius", 0.3, 1.0, 0.0);
                let soft = b.slider("softness", 0.05, 2.0, 0.0);
                let jit = b.slider("jitter", 0.0, 3.0, 0.0);
                let v = b.add(voronoi(VoronoiMode::F1, 0.05, 0.0, 0.0), 0.5, 1.0);
                b.wire(v, &[0, 1], size);
                b.wire(v, &[2], jit);
                let sub = b.add(math(MathOp::Sub), 1.0, 2.0);
                b.wire(sub, &[0], v);
                b.wire(sub, &[1], rad);
                let div = b.add(math(MathOp::Div), 1.5, 3.0);
                b.wire(div, &[0], sub);
                b.wire(div, &[1], soft);
                let r = b.add(remap(0.0, 1.0, 1.0, 0.0), 1.5, 4.0);
                b.wire(r, &[0], div);
                r
            }
        };
        b.finish(out)
    }
}

/// What a NEW Noise node starts with: one plain fBm (5 octaves,
/// lacunarity 2, gain 0.5) into the Output. Not `FieldGraph::default()`,
/// which stays empty so a setup saved without a graph loads as it was.
pub fn basic_fbm() -> FieldGraph {
    let mut b = Build::default();
    let f = b.add(fbm(FieldFbmMode::Fbm, 0.3, 5, 0.0), 0.0, 1.0);
    b.finish(f)
}

/// The graph under construction: ids handed out in order, positions on a
/// grid, top to bottom like every graph in the editor.
#[derive(Default)]
struct Build {
    g: FieldGraph,
    row_max: f32,
}

impl Build {
    fn add(&mut self, n: FieldNode, col: f32, row: f32) -> NodeId {
        self.g.next_id += 1;
        let id = self.g.next_id;
        self.g.nodes.insert(id, n);
        self.g.layout.insert(id, [col * 200.0, row * 95.0]);
        self.row_max = self.row_max.max(row);
        id
    }
    fn slider(&mut self, name: &str, value: f32, col: f32, row: f32) -> NodeId {
        self.add(FieldNode::Const { value, name: name.into() }, col, row)
    }
    /// `src` into each of `dst`'s `sockets`.
    fn wire(&mut self, dst: NodeId, sockets: &[usize], src: NodeId) {
        for s in sockets {
            self.g.edges.insert((dst, *s), src);
        }
    }
    fn finish(mut self, result: NodeId) -> FieldGraph {
        let col = self.g.layout.get(&result).map_or(0.0, |p| p[0] / 200.0);
        let out = self.add(FieldNode::Output, col, self.row_max + 1.0);
        self.wire(out, &[0], result);
        self.g.out = Some(out);
        self.g
    }
}

fn fbm(mode: FieldFbmMode, scale: f32, octaves: u32, seed: f32) -> FieldNode {
    FieldNode::Fbm {
        mode,
        scale: [scale, scale],
        offset: [0.0, 0.0],
        lacunarity: 2.0,
        gain: 0.5,
        seed,
        speed: 1.0,
        octaves,
        uniform: 1.0,
    }
}

fn voronoi(mode: VoronoiMode, scale: f32, jitter: f32, seed: f32) -> FieldNode {
    FieldNode::Voronoi { mode, scale: [scale, scale], jitter, seed, speed: 1.0 }
}

fn math(op: MathOp) -> FieldNode {
    FieldNode::Math { op, a: 0.0, b: 0.0 }
}

fn remap(in_lo: f32, in_hi: f32, out_lo: f32, out_hi: f32) -> FieldNode {
    FieldNode::Remap { in_lo, in_hi, out_lo, out_hi, clamp: true }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every preset is a graph the shader compiler accepts, with its Output
    /// explicit and every slider exposed.
    #[test]
    fn every_preset_compiles_and_exposes_its_sliders() {
        for p in NoisePreset::ALL {
            let g = p.graph();
            assert!(
                matches!(g.output().and_then(|o| g.nodes.get(&o)), Some(FieldNode::Output)),
                "{}: no Output",
                p.label()
            );
            let mut fns = Vec::new();
            assert!(compile(1, &g, 1, "0.0", &mut fns).is_some(), "{}: did not compile", p.label());
            let sliders = g.nodes.values().filter(|n| matches!(n, FieldNode::Const { .. })).count();
            assert!(sliders > 0 && g.exposed().len() == sliders, "{}: sliders not exposed", p.label());
            for (id, n) in &g.nodes {
                assert!(g.layout.contains_key(id), "{}: node {id} has no position", p.label());
                // Nothing dangles: every node but the Output feeds something.
                if !matches!(n, FieldNode::Output) {
                    assert!(g.edges.values().any(|s| s == id), "{}: node {id} feeds nothing", p.label());
                }
            }
        }
    }

    /// Each one is a texture, not a flat card: over a grid the CPU evaluator
    /// (the shader's twin) finds real contrast and stays in 0..1.
    #[test]
    fn every_preset_draws_something() {
        for p in NoisePreset::ALL {
            let g = p.graph();
            let mut v = Vec::new();
            for j in 0..48 {
                for i in 0..96 {
                    v.push(eval_cpu(&g, [i as f32 / 96.0, j as f32 / 96.0], 0.0, 0.0, 1.0));
                }
            }
            assert!(v.iter().all(|x| x.is_finite()), "{}: non-finite", p.label());
            let lo = v.iter().cloned().fold(f32::MAX, f32::min);
            let hi = v.iter().cloned().fold(f32::MIN, f32::max);
            assert!(lo >= -1e-3 && hi <= 1.0 + 1e-3, "{}: range {lo}..{hi}", p.label());
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
            assert!(sd > 0.05, "{}: flat (sd {sd})", p.label());
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn presets_survive_a_setup_file() {
        for p in NoisePreset::ALL {
            let g = p.graph();
            let back: FieldGraph = serde_json::from_str(&serde_json::to_string(&g).unwrap()).unwrap();
            assert_eq!(back, g, "{}", p.label());
        }
    }
}

#[cfg(test)]
mod basic_tests {
    use super::*;

    #[test]
    fn a_new_noise_is_a_basic_fbm_and_the_default_stays_empty() {
        let g = basic_fbm();
        let fbms: Vec<_> = g
            .nodes
            .values()
            .filter_map(|n| match n {
                FieldNode::Fbm { mode, lacunarity, gain, octaves, .. } => {
                    Some((*mode, *lacunarity, *gain, *octaves))
                }
                _ => None,
            })
            .collect();
        assert_eq!(fbms, vec![(FieldFbmMode::Fbm, 2.0, 0.5, 5)]);
        assert!(g.out.is_some());
        assert!(FieldGraph::default().nodes.is_empty());
    }
}
