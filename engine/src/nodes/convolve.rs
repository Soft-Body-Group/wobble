//! Convolve: a user kernel of up to 7x7 weights, one gather pass. The
//! kernel is uniforms, so editing a weight — or the size — never recompiles.
//!
//! Wire a picture into the KERNEL port (2) and that picture is the kernel
//! instead, centred on its middle pixel, one kernel pixel per output pixel:
//! a Glint Kernel's star, a Bokeh Kernel's aperture, a photographed PSF.
//! That runs as an FFT convolution (`nodes::fft`) at the grid resolution
//! `res` asks for, so a 500 px star costs what a 5 px one does. The
//! MODULATE port (3) scales the mix per pixel, in either mode, and costs
//! nothing unwired.

use crate::codegen::Emit;
use crate::cost::Speed;
use crate::nodes::fft;
use crate::nodes::stage::{
    Bind, Dispatch, Pass, StageCx, StageDef, StageProgram, TexFormat, TexSpec, Wired,
};
use crate::nodes::NodeDef;

/// The widest kernel, and the side of the grid the weights are stored on.
pub const CONVOLVE_MAX: usize = 7;
/// Weights stored, whatever the size: a smaller kernel is the CENTRE of the
/// grid, so switching size keeps what was typed.
pub const CONVOLVE_CELLS: usize = CONVOLVE_MAX * CONVOLVE_MAX;

/// The kernel-image port.
pub const KERNEL_INPUT: usize = 2;

/// The grid a kernel image is convolved at. Structural (it sizes the
/// transform); the grid steps down further when a side would not fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ConvolveRes {
    Full,
    #[default]
    Half,
    Quarter,
}

impl ConvolveRes {
    pub const ALL: [ConvolveRes; 3] = [ConvolveRes::Full, ConvolveRes::Half, ConvolveRes::Quarter];
    pub fn label(self) -> &'static str {
        match self {
            ConvolveRes::Full => "full",
            ConvolveRes::Half => "half",
            ConvolveRes::Quarter => "quarter",
        }
    }
    pub fn shift(self) -> u32 {
        self as u32
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConvolveParams {
    /// Kernel side: 3, 5 or 7.
    pub size: u32,
    /// 7x7, row-major, top row first; the kernel is its centre size x size.
    pub kernel: Vec<f32>,
    /// Divide by the weights' sum (when it is not ~0), so a blur keeps the
    /// picture's level and the numbers can be typed as integers.
    pub normalize: bool,
    /// Convolve alpha too; off, alpha passes through.
    pub alpha: bool,
    /// 0 = the input, 1 = the convolved picture.
    pub mix: f32,
    /// Which pipe it sits in: wired a MATTE it filters the coverage and
    /// gives a matte (`NodeKind::follows_pipe`). Kept true to its input by
    /// the app (`EditorGraph::connect_out`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub matte: bool,
    /// Kernel-image mode only: the resolution the FFT runs at.
    #[cfg_attr(feature = "serde", serde(default))]
    pub res: ConvolveRes,
}

impl Default for ConvolveParams {
    fn default() -> Self {
        Self::preset(ConvolvePreset::Sharpen)
    }
}

/// Kernels worth having one click away.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvolvePreset {
    Identity,
    Box,
    Gaussian,
    Sharpen,
    Edge,
    Emboss,
}

impl ConvolvePreset {
    pub const ALL: [ConvolvePreset; 6] = [
        ConvolvePreset::Identity,
        ConvolvePreset::Box,
        ConvolvePreset::Gaussian,
        ConvolvePreset::Sharpen,
        ConvolvePreset::Edge,
        ConvolvePreset::Emboss,
    ];
    pub fn name(self) -> &'static str {
        match self {
            ConvolvePreset::Identity => "identity",
            ConvolvePreset::Box => "box",
            ConvolvePreset::Gaussian => "gaussian",
            ConvolvePreset::Sharpen => "sharpen",
            ConvolvePreset::Edge => "edge (laplacian)",
            ConvolvePreset::Emboss => "emboss",
        }
    }
}

impl ConvolveParams {
    /// A 3x3 preset, keeping `mix` and `alpha` at their defaults.
    pub fn preset(p: ConvolvePreset) -> Self {
        let k3: [f32; 9] = match p {
            ConvolvePreset::Identity => [0., 0., 0., 0., 1., 0., 0., 0., 0.],
            ConvolvePreset::Box => [1.; 9],
            ConvolvePreset::Gaussian => [1., 2., 1., 2., 4., 2., 1., 2., 1.],
            ConvolvePreset::Sharpen => [0., -1., 0., -1., 5., -1., 0., -1., 0.],
            ConvolvePreset::Edge => [0., 1., 0., 1., -4., 1., 0., 1., 0.],
            ConvolvePreset::Emboss => [-2., -1., 0., -1., 1., 1., 0., 1., 2.],
        };
        let mut c = Self { size: 3, kernel: vec![0.0; CONVOLVE_CELLS], normalize: true, alpha: false, mix: 1.0, matte: false, res: ConvolveRes::default() };
        for y in 0..3 {
            for x in 0..3 {
                c.kernel[(y + 2) * CONVOLVE_MAX + x + 2] = k3[y * 3 + x];
            }
        }
        c
    }

    /// The side actually used: 3, 5 or 7.
    pub fn side(&self) -> usize {
        match self.size {
            0..=3 => 3,
            4..=5 => 5,
            _ => 7,
        }
    }

    /// Weight `(x, y)` of the full 7x7 grid; 0 for a short kernel vector.
    pub fn weight(&self, x: usize, y: usize) -> f32 {
        self.kernel.get(y * CONVOLVE_MAX + x).copied().unwrap_or(0.0)
    }

    /// Is grid cell `(x, y)` inside the kernel the current size uses?
    pub fn in_use(&self, x: usize, y: usize) -> bool {
        let o = (CONVOLVE_MAX - self.side()) / 2;
        (o..CONVOLVE_MAX - o).contains(&x) && (o..CONVOLVE_MAX - o).contains(&y)
    }

    /// What the weights are divided by: their sum when normalising and it
    /// is not ~0 (an edge kernel sums to 0 and must stay as it is), else 1.
    pub fn scale(&self) -> f32 {
        if !self.normalize {
            return 1.0;
        }
        let mut sum = 0.0;
        for y in 0..CONVOLVE_MAX {
            for x in 0..CONVOLVE_MAX {
                if self.in_use(x, y) {
                    sum += self.weight(x, y);
                }
            }
        }
        if sum.abs() < 1e-6 {
            1.0
        } else {
            1.0 / sum
        }
    }
}

macro_rules! convolve_body {
    () => {
        r#"
fn weight(i: i32) -> f32 {
    return C.k[i / 4][i % 4];
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(dst));
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) { return; }
    let p = vec2i(gid.xy);
    let c = px(textureLoad(src, p, 0));
    let h = i32(C.head.x);
    var acc = PX(0.0);
    for (var dy = -h; dy <= h; dy++) {
        for (var dx = -h; dx <= h; dx++) {
            let w = weight((dy + 3) * 7 + dx + 3);
            if (w == 0.0) { continue; }
            let q = clamp(p + vec2i(dx, dy), vec2i(0), dims - 1);
            acc += w * px(textureLoad(src, q, 0));
        }
    }
    acc *= C.head.y;
    if (C.head.w < 0.5) { acc = keep_alpha(acc, c); }
    textureStore(dst, gid.xy, px_out(mix(c, acc, amount(p))));
}
"#
    };
}

macro_rules! convolve_head {
    ($fmt:literal) => {
        concat!(
            "// Header: (half side, scale, mix, alpha), then the 7x7 grid in 13 vec4s.\n",
            "struct CP { head: vec4f, k: array<vec4f, 13> }\n",
            "@group(0) @binding(0) var<uniform> C: CP;\n",
            "@group(0) @binding(1) var src: texture_2d<f32>;\n",
            "@group(0) @binding(2) var dst: texture_storage_2d<", $fmt, ", write>;\n",
        )
    };
}
macro_rules! convolve_mix {
    () => {
        "fn amount(p: vec2i) -> f32 { return C.head.z; }\n"
    };
}
macro_rules! convolve_mod {
    () => {
        concat!(
            "@group(0) @binding(3) var modt: texture_2d<f32>;\n",
            "fn amount(p: vec2i) -> f32 { return C.head.z * clamp(textureLoad(modt, p, 0).x, 0.0, 1.0); }\n",
        )
    };
}

pub const WGSL: &str = concat!(
    convolve_head!("rgba16float"),
    crate::nodes::stage::px_rgba!(),
    "fn keep_alpha(v: PX, c: PX) -> PX { return vec4f(v.rgb, c.a); }\n",
    convolve_mix!(),
    convolve_body!()
);

/// [`WGSL`] with the modulate matte scaling the mix.
pub const MOD_WGSL: &str = concat!(
    convolve_head!("rgba16float"),
    crate::nodes::stage::px_rgba!(),
    "fn keep_alpha(v: PX, c: PX) -> PX { return vec4f(v.rgb, c.a); }\n",
    convolve_mod!(),
    convolve_body!()
);

/// [`WGSL`] on ONE channel: a matte's convolution (a matte has no alpha
/// of its own to keep).
pub const ONE_WGSL: &str = concat!(
    convolve_head!("r16float"),
    crate::nodes::stage::px_one!(),
    "fn keep_alpha(v: PX, c: PX) -> PX { return v; }\n",
    convolve_mix!(),
    convolve_body!()
);

pub const ONE_MOD_WGSL: &str = concat!(
    convolve_head!("r16float"),
    crate::nodes::stage::px_one!(),
    "fn keep_alpha(v: PX, c: PX) -> PX { return v; }\n",
    convolve_mod!(),
    convolve_body!()
);

/// The FFT passes in order, for a kernel image: what `uniforms` counts.
const FFT_PASSES: usize = 2 + 4 + 1 + 2 + 1;

impl ConvolveParams {
    fn gather(&self, wired: Wired, one: bool) -> StageProgram {
        let modulated = wired.has(Self::MOD_INPUT.unwrap());
        let wgsl = match (one, modulated) {
            (false, false) => WGSL,
            (false, true) => MOD_WGSL,
            (true, false) => ONE_WGSL,
            (true, true) => ONE_MOD_WGSL,
        };
        let mut binds = vec![Bind::Uniform, Bind::Tex("a".into()), Bind::Storage("b".into())];
        if modulated {
            binds.push(Bind::Tex("aux0".into()));
        }
        StageProgram {
            textures: vec![TexSpec::full("b")],
            passes: vec![Pass {
                label: "convolve",
                wgsl,
                binds,
                dispatch: Dispatch::Pixels("a".into()),
                uniform_floats: 4 + 52,
            }],
            result: "b".into(),
        }
    }

    /// The kernel-image chain: load both, transform both, multiply, back.
    /// rgba is one grid texture each (two channels a complex lane, see
    /// `fft`), so each 2D transform is two passes.
    fn fft(&self, canvas: (u32, u32), wired: Wired) -> StageProgram {
        let (_, nx, ny) = fft::grid(canvas, self.res.shift());
        let g = |n: &str| TexSpec::sized(n, nx, ny, TexFormat::Rgba32Float);
        let textures = vec![TexSpec::full("b"), g("f"), g("k"), g("s"), g("t")];
        let st = |n: &str| Bind::Storage(n.into());
        let tx = |n: &str| Bind::Tex(n.into());
        let mut passes = vec![
            Pass {
                label: "convolve-load",
                wgsl: fft::LOAD_IMAGE_WGSL,
                binds: vec![Bind::Uniform, tx("a"), st("f")],
                dispatch: Dispatch::Pixels("f".into()),
                uniform_floats: 4,
            },
            Pass {
                label: "convolve-load-kernel",
                wgsl: fft::LOAD_KERNEL_WGSL,
                binds: vec![Bind::Uniform, tx("aux0"), st("k")],
                dispatch: Dispatch::Pixels("k".into()),
                uniform_floats: 8,
            },
        ];
        // A 2D transform of `x` through `s`: rows x -> s, columns s -> x.
        let groups = fft::groups(nx, ny);
        let transform = |passes: &mut Vec<Pass>, x: &str, s: &str| {
            for (from, to) in [(x, s), (s, x)] {
                passes.push(Pass {
                    label: "convolve-fft",
                    wgsl: fft::FFT_WGSL,
                    binds: vec![Bind::Uniform, tx(from), st(to)],
                    dispatch: Dispatch::Groups(groups, 1),
                    uniform_floats: 4,
                });
            }
        };
        transform(&mut passes, "f", "s");
        transform(&mut passes, "k", "s");
        passes.push(Pass {
            label: "convolve-multiply",
            wgsl: fft::MULTIPLY_WGSL,
            binds: vec![Bind::Uniform, tx("f"), tx("k"), st("s")],
            dispatch: Dispatch::Pixels("s".into()),
            uniform_floats: 4,
        });
        // Back: s through t — the result ends in s.
        transform(&mut passes, "s", "t");
        let modulated = wired.has(Self::MOD_INPUT.unwrap());
        let mut binds = vec![Bind::Uniform, tx("a"), tx("s"), st("b")];
        if modulated {
            binds.push(tx("aux1"));
        }
        passes.push(Pass {
            label: "convolve-out",
            wgsl: if modulated { fft::OUT_MOD_WGSL } else { fft::OUT_WGSL },
            binds,
            dispatch: Dispatch::Pixels("b".into()),
            uniform_floats: 8,
        });
        debug_assert_eq!(passes.len(), FFT_PASSES);
        StageProgram { textures, passes, result: "b".into() }
    }

    fn fft_uniforms(&self, pass: usize, cx: &StageCx, kernel_fmt: (u32, u32)) -> Vec<f32> {
        let (s, nx, ny) = fft::grid(cx.canvas, self.res.shift());
        let block = (1u32 << s) as f32;
        let (w, h) = (cx.canvas.0.div_ceil(1 << s) as f32, cx.canvas.1.div_ceil(1 << s) as f32);
        // Rows of an nx-wide grid are nx long; columns ny.
        let fft_u = |row: bool, sign: f32| {
            let n = if row { nx } else { ny };
            vec![n as f32, n.trailing_zeros() as f32, if row { 0.0 } else { 1.0 }, sign]
        };
        match pass {
            0 => vec![block, w, h, 0.0],
            1 => vec![
                block,
                nx as f32,
                ny as f32,
                0.0,
                (kernel_fmt.0 / 2) as f32,
                (kernel_fmt.1 / 2) as f32,
                0.0,
                0.0,
            ],
            2..=5 => fft_u(pass % 2 == 0, -1.0),
            6 => vec![self.normalize as u32 as f32, nx as f32, ny as f32, 0.0],
            7 | 8 => fft_u(pass % 2 == 1, 1.0),
            _ => vec![
                block,
                w,
                h,
                1.0 / (nx as f32 * ny as f32),
                self.mix.clamp(0.0, 1.0),
                if self.alpha { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ],
        }
    }
}

impl NodeDef for ConvolveParams {
    const NAME: &'static str = "Convolve";
    /// Image, the mask, a kernel picture, the modulate matte.
    const INPUTS: usize = 4;
    const MASK_INPUT: Option<usize> = Some(1);
    const MOD_INPUT: Option<usize> = Some(3);
    /// The kernel rides in the pass's own uniform.
    const VEC4_SLOTS: usize = 0;

    /// Size, weights and switches are all uniforms; only the FFT grid's
    /// resolution (which sizes its textures) is structural.
    fn topo_tag(&self) -> u64 {
        0xC0B7 + ((self.res as u64) << 16)
    }

    fn pack(&self, _: &mut [f32]) {}

    fn emit(&self, e: &mut Emit) -> String {
        let v = e.stage_result();
        e.masked_stage(Self::MASK_INPUT, v)
    }
}

impl StageDef for ConvolveParams {
    /// The kernel picture first (`aux0`), then the modulate matte.
    fn aux_ports(&self, wired: Wired) -> Vec<usize> {
        [KERNEL_INPUT, Self::MOD_INPUT.unwrap()].into_iter().filter(|p| wired.has(*p)).collect()
    }

    fn program(&self, canvas: (u32, u32), wired: Wired) -> StageProgram {
        if wired.has(KERNEL_INPUT) {
            self.fft(canvas, wired)
        } else {
            self.gather(wired, false)
        }
    }

    fn program_one_channel(&self, _canvas: (u32, u32), wired: Wired) -> Option<StageProgram> {
        // A kernel image runs a matte through the four-channel chain.
        (!wired.has(KERNEL_INPUT)).then(|| self.gather(wired, true).one_channel())
    }

    /// Weights are per PIXEL: a kernel does not scale with the proxy.
    fn uniforms_aux(&self, pass: usize, cx: &StageCx, aux_fmt: &[(u32, u32)]) -> Vec<f32> {
        if cx.passes == FFT_PASSES {
            return self.fft_uniforms(pass, cx, aux_fmt.first().copied().unwrap_or(cx.canvas));
        }
        self.uniforms(pass, cx)
    }

    fn uniforms(&self, pass: usize, cx: &StageCx) -> Vec<f32> {
        if cx.passes == FFT_PASSES {
            return self.fft_uniforms(pass, cx, cx.canvas);
        }
        let mut v = vec![
            (self.side() / 2) as f32,
            self.scale(),
            self.mix.clamp(0.0, 1.0),
            if self.alpha { 1.0 } else { 0.0 },
        ];
        for y in 0..CONVOLVE_MAX {
            for x in 0..CONVOLVE_MAX {
                v.push(if self.in_use(x, y) { self.weight(x, y) } else { 0.0 });
            }
        }
        v.resize(4 + 52, 0.0);
        v
    }

    fn speed(&self) -> Speed {
        Speed::Slow
    }
}
