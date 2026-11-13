//! What a node that takes its OWN passes tells the compositor.
//!
//! A pointwise node fuses into the main pass and needs only [`NodeDef`].
//! A node that has to sample its neighbourhood — a blur, a morphology, a
//! pyramid — runs BEFORE the main pass as a stage: its input subtree is
//! rendered into a texture (`a`), a chain of compute passes runs over it,
//! and the fused shader samples the result. `StageDef` is that chain as
//! DATA — the textures it needs, the passes in order, what each one binds,
//! the uniform values per frame — and the compositor allocates, binds and
//! dispatches it. A node never touches wgpu.
//!
//! Every stage is a slow path by construction (manifesto rule 8), so a
//! stage also says what it costs.
//!
//! [`NodeDef`]: crate::nodes::NodeDef

use crate::cost::Speed;

/// A texture the stage owns beside its input `a` and its aux inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TexSpec {
    /// How passes refer to it. `a` and `aux<i>` are taken.
    pub name: String,
    /// The canvas shifted right this many times; 0 is full size.
    pub shift: u32,
    pub format: TexFormat,
    /// A fixed size instead of the shifted canvas: a transform grid, whose
    /// sides are powers of two whatever the canvas.
    pub size: Option<(u32, u32)>,
}

impl TexSpec {
    /// A full-canvas intermediate in the comp's own format.
    pub fn full(name: &str) -> Self {
        Self { name: name.into(), shift: 0, format: TexFormat::Rgba16Float, size: None }
    }

    /// One level of a pyramid: the canvas at `1 / 2^shift`.
    pub fn level(name: &str, shift: u32) -> Self {
        Self { name: name.into(), shift, format: TexFormat::Rgba16Float, size: None }
    }

    /// A texture of exactly `w` x `h`, whatever the canvas.
    pub fn sized(name: &str, w: u32, h: u32, format: TexFormat) -> Self {
        Self { name: name.into(), shift: 0, format, size: Some((w, h)) }
    }

    /// Its size for a canvas.
    pub fn dims(&self, canvas: (u32, u32)) -> (u32, u32) {
        self.size.unwrap_or(((canvas.0 >> self.shift).max(1), (canvas.1 >> self.shift).max(1)))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TexFormat {
    /// The comp's intermediate format.
    Rgba16Float,
    /// Full float, for prefix sums and other accumulators.
    Rgba32Float,
    /// ONE half-float channel: a matte's scratch (`StageDef::program_one_
    /// channel`). A quarter of the bytes of `Rgba16Float` per texel.
    R16Float,
}

/// One binding of a pass, in `@binding` order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Bind {
    /// This pass's uniform buffer; see [`StageDef::uniforms`].
    Uniform,
    /// The compositor's bilinear clamp sampler.
    Sampler,
    /// Read a texture: `a`, `aux<i>`, or one declared in `textures`.
    Tex(String),
    /// Write one, as a storage texture.
    Storage(String),
}

/// How many workgroups a pass dispatches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dispatch {
    /// One thread per texel of the named texture, in 8x8 workgroups: the
    /// shape of every image-sized pass.
    Pixels(String),
    /// One thread per ROW of the named texture, in workgroups of 64.
    Rows(String),
    /// Exactly this many workgroups: a pass whose workgroup is a whole row
    /// (an FFT) and sizes its own grid.
    Groups(u32, u32),
}

/// One compute pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pass {
    pub label: &'static str,
    /// A complete compute module with a `main` entry point, whose group-0
    /// bindings are `binds`, in order. Compiled once per process.
    pub wgsl: &'static str,
    pub binds: Vec<Bind>,
    pub dispatch: Dispatch,
    /// Floats in the uniform this pass reads; 0 when it binds none.
    pub uniform_floats: usize,
}

/// The whole chain, for one canvas size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageProgram {
    pub textures: Vec<TexSpec>,
    pub passes: Vec<Pass>,
    /// The texture the fused shader samples as this node's output.
    pub result: String,
}

/// Which of a node's inputs are wired, by index. Wiring is TOPOLOGY (it is
/// in the hash), so a program may depend on it — and should, wherever an
/// optional port costs something: an unwired port must cost nothing.
#[derive(Clone, Copy, Debug)]
pub struct Wired<'a>(pub &'a [bool]);

impl Wired<'_> {
    pub fn has(&self, input: usize) -> bool {
        self.0.get(input).copied().unwrap_or(false)
    }
}

/// The per-frame facts a stage's uniforms may depend on.
pub struct StageCx {
    pub canvas: (u32, u32),
    /// Pixel-sized params scale by this under proxy rendering.
    pub proxy_scale: f32,
    pub frame: f32,
    pub fps: f32,
    /// Passes in the program being fed. `uniforms` cannot see the wiring,
    /// and a chain whose shape depends on it (an optional port that adds a
    /// pass) tells its passes apart by this.
    pub passes: usize,
}

/// Can this device write single-channel half textures from a compute
/// pass? Set once from the adapter (`GpuContext`); off (the four-channel
/// path, correct but four times the bytes) in a browser and on hardware
/// that cannot. Scale down, never refuse.
static ONE_CHANNEL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_one_channel(on: bool) {
    ONE_CHANNEL.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn one_channel() -> bool {
    ONE_CHANNEL.load(std::sync::atomic::Ordering::Relaxed)
}

/// The pixel type a pass body works in, for kernels written once for both
/// pipes: `PX` is a pixel, `px(texel)` makes one from a texture read and
/// `px_out(v)` turns one back into what `textureStore` takes. A picture is
/// `vec4f` all the way (the aliases compile away); a matte is ONE `f32`,
/// so a matte blur or median does a quarter of the arithmetic and writes
/// a quarter of the bytes. Prepend the matching header to the same body.
macro_rules! px_rgba {
    () => {
        "alias PX = vec4f;\nfn px(v: vec4f) -> PX { return v; }\nfn px_out(v: PX) -> vec4f { return v; }\n"
    };
}
macro_rules! px_one {
    () => {
        "alias PX = f32;\nfn px(v: vec4f) -> PX { return v.x; }\nfn px_out(v: PX) -> vec4f { return vec4f(v); }\n"
    };
}
pub(crate) use {px_one, px_rgba};

impl StageProgram {
    /// Every half-float RGBA scratch texture as ONE channel: for a chain
    /// whose passes were written with the `px_one!` header (their storage
    /// bindings declared `r16float`). Accumulators keep their format.
    pub fn one_channel(mut self) -> Self {
        for t in &mut self.textures {
            if t.format == TexFormat::Rgba16Float {
                t.format = TexFormat::R16Float;
            }
        }
        self
    }
}

pub trait StageDef {
    /// Image ports beyond input 0 that resolve into textures `aux0..`,
    /// each rendered through its own subtree pass at the canvas size.
    fn aux_ports(&self, _wired: Wired) -> Vec<usize> {
        Vec::new()
    }
    /// The chain for a canvas. It may depend on the canvas and on the
    /// STRUCTURAL params — the ones in `topo_tag` — and on nothing else: a
    /// uniform that changed the chain would make a slider drag rebuild
    /// textures and bind groups, which is the recompile this design exists
    /// to avoid. `wired` is the node's wiring, and must agree with what
    /// `aux_ports` was given: `aux<i>` exists only for a port it returned.
    fn program(&self, canvas: (u32, u32), wired: Wired) -> StageProgram;
    /// The same chain run on ONE channel, for a node whose output is a
    /// matte: the same passes and uniforms in the same order (`uniforms`
    /// serves both), single-channel modules and scratch. `None` (the
    /// default) runs a matte through the four-channel chain, which is
    /// correct, just four times the work. Only asked when the device can
    /// write one-channel textures (`one_channel()`).
    fn program_one_channel(&self, _canvas: (u32, u32), _wired: Wired) -> Option<StageProgram> {
        None
    }
    /// The uniform floats for pass `pass`, exactly `uniform_floats` long.
    fn uniforms(&self, pass: usize, cx: &StageCx) -> Vec<f32>;
    /// [`uniforms`](Self::uniforms) for a chain that needs to know the
    /// FORMAT of what feeds its aux ports (`aux_fmt[i]` for `aux<i>`; the
    /// canvas where the graph has none): a kernel picture is centred on
    /// its own middle, which sits wherever its format puts it. One entry
    /// past the aux ports is the node's OWN format. This is what the
    /// compositor calls.
    fn uniforms_aux(&self, pass: usize, cx: &StageCx, _aux_fmt: &[(u32, u32)]) -> Vec<f32> {
        self.uniforms(pass, cx)
    }
    /// What the graph colours it: `Slow` for a pass or a few, `Heavy` for a
    /// chain that dominates the frame.
    fn speed(&self) -> Speed;
}
