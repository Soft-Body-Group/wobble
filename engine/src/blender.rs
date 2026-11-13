//! Blender as the comp's 3D system. A Blender node is a window onto a
//! running Blender: wobble drives its playhead, Blender draws the scene
//! camera and hands the pixels back through shared memory, and the comp's
//! own pictures go the other way as named image textures any material can
//! use. Nothing is encoded and nothing is written to disk on the way.
//!
//! - **Live** ([`live::Session`]): an interactive Blender launched with the
//!   bridge script (or one the user connects from the addon panel). Every
//!   playhead move becomes a `frame` command; Blender's own frame changes
//!   come back as `frame_changed`, so the two timelines follow each other.
//!   Pixels cross in the ring of [`shm`], one memcpy per frame, stamped
//!   with the Blender frame they were drawn at and the scene version they
//!   belong to.
//! - **Viewport or final**: `Quality::Viewport` draws the viewport through
//!   the scene camera (Eevee at interactive rates, refining across draws
//!   until it settles); `Quality::Final` is an F12 render read back off the
//!   compositor's viewer node — seconds a frame in Cycles, and the panel
//!   says so.
//! - **Bake**: the frames land in a per-node GPU layer, never a reader
//!   slot, so a Blender node is always the LIVE picture. Putting a
//!   `GraphBake` below it is what turns a shot into files, and the bake
//!   loop waits for each frame to settle before it writes.
//!
//! Deliberately arm's-length: the process boundary keeps GPLv3 `bpy` out of
//! this binary. The `.blend` is an opaque asset; the format, engine and
//! camera the comp needs are applied by the script from the spec, never
//! trusted to the file. The script ships INSIDE the binary
//! (`include_str!`) and is written next to the cache on every launch, so
//! the Blender side can never drift from the wobble side.
//!
//! VERIFIED on 5.2.1: an offscreen `draw_view3d` with colour management off
//! and an F12 read through the viewer node both hand back scene-linear
//! pixels exactly (an emission of 0.5 reads 0.5), in BLENDER's primaries —
//! Linear Rec.709 unless its OCIO config says otherwise — which is what
//! [`BlenderParams::space`] converts into the comp's working space.

pub mod cycles;
pub mod gpuimport;
pub mod live;
pub mod shm;
pub mod view;

pub use view::ViewState;

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The bridge script, embedded so it always matches this build.
pub const SCRIPT: &str = include_str!("blender/wobble_blender.py");

/// File name the script is written under.
pub const SCRIPT_NAME: &str = "wobble_blender.py";

/// Most image inputs a node offers. Each is a named image in Blender
/// (`wobble_in_N`); past a handful you want a Layer stack first.
pub const MAX_INPUTS: usize = 8;

/// The image name an input port lands under in Blender.
pub fn input_image_name(port: usize) -> String {
    format!("wobble_in_{port}")
}

/// Render engine forced on the scene. `File` leaves whatever the `.blend` has.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BlenderEngine {
    #[default]
    File,
    Eevee,
    Cycles,
}

impl BlenderEngine {
    pub const ALL: [BlenderEngine; 3] =
        [BlenderEngine::File, BlenderEngine::Eevee, BlenderEngine::Cycles];

    pub fn name(self) -> &'static str {
        match self {
            BlenderEngine::File => "from file",
            BlenderEngine::Eevee => "eevee",
            BlenderEngine::Cycles => "cycles",
        }
    }

    /// The spec's spelling, which the script switches on.
    pub fn spec_name(self) -> &'static str {
        match self {
            BlenderEngine::File => "file",
            BlenderEngine::Eevee => "eevee",
            BlenderEngine::Cycles => "cycles",
        }
    }
}

/// What the node shows: the viewport drawn through the camera, or a real
/// render of each frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Quality {
    /// `draw_view3d` through the scene camera: interactive, refined
    /// across draws until it settles.
    #[default]
    Viewport,
    /// An F12 render per frame, read off the compositor's viewer node.
    Final,
}

impl Quality {
    pub const ALL: [Quality; 2] = [Quality::Viewport, Quality::Final];
    pub fn name(self) -> &'static str {
        match self {
            Quality::Viewport => "viewport",
            Quality::Final => "final render",
        }
    }
    pub fn spec_name(self) -> &'static str {
        match self {
            Quality::Viewport => "viewport",
            Quality::Final => "final",
        }
    }
}

/// The viewport shading the live draw uses. `AsIs` takes the user's own
/// viewport; the others are set on it (and stay, so what Blender shows is
/// what wobble gets).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Shading {
    AsIs,
    Material,
    #[default]
    Rendered,
}

impl Shading {
    pub const ALL: [Shading; 3] = [Shading::AsIs, Shading::Material, Shading::Rendered];
    pub fn name(self) -> &'static str {
        match self {
            Shading::AsIs => "as in blender",
            Shading::Material => "material preview",
            Shading::Rendered => "rendered",
        }
    }
    pub fn spec_name(self) -> &'static str {
        match self {
            Shading::AsIs => "",
            Shading::Material => "MATERIAL",
            Shading::Rendered => "RENDERED",
        }
    }
}

/// A render pass the viewport can draw on its own (Blender's
/// `View3DShading.render_pass`), each an extra output of the node and an
/// extra ~4ms draw a frame. The id is the ring stamp's `pass`; 0 is the
/// combined picture on output 0. Fixed: an id in a setup must mean the
/// same pass forever.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Pass {
    /// Depth along the camera axis, in scene units: Blender's mist pass
    /// with the world's mist set linear over the clip range and scaled
    /// back (the viewport has no Z pass of its own).
    Z,
    Normal,
    Position,
    Emission,
    Environment,
    Ao,
    Shadow,
    DiffuseLight,
    DiffuseColor,
    SpecularLight,
    SpecularColor,
    VolumeLight,
    Transparent,
}

impl Pass {
    pub const ALL: [Pass; 13] = [
        Pass::Z,
        Pass::Normal,
        Pass::Position,
        Pass::Emission,
        Pass::Environment,
        Pass::Ao,
        Pass::Shadow,
        Pass::DiffuseLight,
        Pass::DiffuseColor,
        Pass::SpecularLight,
        Pass::SpecularColor,
        Pass::VolumeLight,
        Pass::Transparent,
    ];

    /// The ring stamp's `pass` (0 = combined).
    pub fn id(self) -> u8 {
        match self {
            Pass::Z => 1,
            Pass::Normal => 2,
            Pass::Position => 3,
            Pass::Emission => 4,
            Pass::Environment => 5,
            Pass::Ao => 6,
            Pass::Shadow => 7,
            Pass::DiffuseLight => 8,
            Pass::DiffuseColor => 9,
            Pass::SpecularLight => 10,
            Pass::SpecularColor => 11,
            Pass::VolumeLight => 12,
            Pass::Transparent => 13,
        }
    }

    /// Blender's identifier for `View3DShading.render_pass`.
    pub fn blender_name(self) -> &'static str {
        match self {
            Pass::Z => "MIST",
            Pass::Normal => "NORMAL",
            Pass::Position => "POSITION",
            Pass::Emission => "EMISSION",
            Pass::Environment => "ENVIRONMENT",
            Pass::Ao => "AO",
            Pass::Shadow => "SHADOW",
            Pass::DiffuseLight => "DIFFUSE_LIGHT",
            Pass::DiffuseColor => "DIFFUSE_COLOR",
            Pass::SpecularLight => "SPECULAR_LIGHT",
            Pass::SpecularColor => "SPECULAR_COLOR",
            Pass::VolumeLight => "VOLUME_LIGHT",
            Pass::Transparent => "TRANSPARENT",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Pass::Z => "Z (depth)",
            Pass::Normal => "normal",
            Pass::Position => "position",
            Pass::Emission => "emission",
            Pass::Environment => "environment",
            Pass::Ao => "AO",
            Pass::Shadow => "shadow",
            Pass::DiffuseLight => "diffuse light",
            Pass::DiffuseColor => "diffuse colour",
            Pass::SpecularLight => "specular light",
            Pass::SpecularColor => "specular colour",
            Pass::VolumeLight => "volume light",
            Pass::Transparent => "transparent",
        }
    }

    /// One number per pixel: rides the DATA pipe (a Defocus Z port takes
    /// it as is).
    pub fn scalar(self) -> bool {
        matches!(self, Pass::Z | Pass::Ao | Pass::Shadow)
    }

    /// The layer name a ChannelSelect / Breakout picks off the fat cable.
    pub fn key(self) -> &'static str {
        match self {
            Pass::Z => "Z",
            Pass::Normal => "normal",
            Pass::Position => "position",
            Pass::Emission => "emission",
            Pass::Environment => "environment",
            Pass::Ao => "AO",
            Pass::Shadow => "shadow",
            Pass::DiffuseLight => "diffuse_light",
            Pass::DiffuseColor => "diffuse_color",
            Pass::SpecularLight => "specular_light",
            Pass::SpecularColor => "specular_color",
            Pass::VolumeLight => "volume_light",
            Pass::Transparent => "transparent",
        }
    }

    pub fn by_key(key: &str) -> Option<Pass> {
        Pass::ALL.iter().copied().find(|p| p.key() == key)
    }
}

/// Whose eye the picture is: the scene camera's, or the 3D view's own
/// (free look, not saved with the file — what you see in Blender's
/// viewport, orbitable from either side).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum View {
    #[default]
    Camera,
    Free,
}

impl View {
    pub const ALL: [View; 2] = [View::Camera, View::Free];
    pub fn name(self) -> &'static str {
        match self {
            View::Camera => "camera",
            View::Free => "free (3D view)",
        }
    }
    pub fn spec_name(self) -> &'static str {
        match self {
            View::Camera => "camera",
            View::Free => "free",
        }
    }
}

/// Which playhead follows which.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Sync {
    /// Either timeline moves the other.
    #[default]
    Both,
    /// Blender follows wobble; Blender's own scrubbing stays in Blender.
    Wobble,
    /// wobble follows Blender; wobble's playhead is not sent.
    Blender,
}

impl Sync {
    pub const ALL: [Sync; 3] = [Sync::Both, Sync::Wobble, Sync::Blender];
    pub fn name(self) -> &'static str {
        match self {
            Sync::Both => "both ways",
            Sync::Wobble => "blender follows",
            Sync::Blender => "wobble follows",
        }
    }
    /// wobble's playhead is sent to Blender.
    pub fn drives_blender(self) -> bool {
        matches!(self, Sync::Both | Sync::Wobble)
    }
    /// Blender's playhead moves wobble's.
    pub fn follows_blender(self) -> bool {
        matches!(self, Sync::Both | Sync::Blender)
    }
}

/// The node's authored state. Everything here is a knob that undoes and
/// saves; the session itself is app state.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BlenderParams {
    /// Path to the `.blend`. Empty means a fresh scene: `open` starts
    /// Blender with its default file and the node learns the path when
    /// the user saves.
    pub blend: String,
    /// Scene name; empty = the file's active scene.
    pub scene: String,
    /// Object name of the camera; empty = the scene camera.
    pub camera: String,
    pub engine: BlenderEngine,
    pub quality: Quality,
    pub shading: Shading,
    /// Viewport samples to settle at (0 = the scene's own viewport
    /// samples); for a final render, the samples override (0 = the file's).
    pub samples: u32,
    /// Comp tick = Blender frame + offset. A scene animated 1..48 with an
    /// offset of 1000 plays at 1001..1048.
    pub offset: i64,
    /// Film → Transparent (straight into the comp's premultiplied `over`).
    pub transparent: bool,
    /// Image input ports. Each arrives in Blender as the named image
    /// `wobble_in_N`, in Blender's scene-linear space, for any material.
    #[cfg_attr(feature = "serde", serde(default))]
    pub inputs: usize,
    /// The space the pixels come back in. Blender's scene-linear is
    /// Linear Rec.709 unless its OCIO config says otherwise.
    #[cfg_attr(feature = "serde", serde(default = "default_space"))]
    pub space: crate::color::InputSpace,
    #[cfg_attr(feature = "serde", serde(default))]
    pub sync: Sync,
    /// Whose eye: the camera's, or the 3D view's free look.
    #[cfg_attr(feature = "serde", serde(default))]
    pub view: View,
    /// Blender's hands over the picture in wobble's viewer (MMB orbit,
    /// shift+MMB pan, ctrl+MMB / wheel dolly) move that eye — the 3D view
    /// in free mode, the camera itself in camera mode.
    #[cfg_attr(feature = "serde", serde(default = "yes"))]
    pub orbit: bool,
    /// Mount every input on a card in front of the camera: a plane
    /// carrying `wobble_in_N` as unlit emission through its own alpha,
    /// fitted to the frame, one unit further back per port.
    #[cfg_attr(feature = "serde", serde(default))]
    pub cards: bool,
    /// Extra passes, one output each after the combined picture, in this
    /// order: output k is `passes[k - 1]`. Viewport mode draws each one
    /// (an extra draw a frame); a final render carries only the combined.
    #[cfg_attr(feature = "serde", serde(default))]
    pub passes: Vec<Pass>,
    /// Output 0 is the FAT cable — the combined picture and every pass by
    /// name — for a Breakout or a ChannelSelect to pick from, the way a
    /// multi-layer Read's is. The per-pass outputs stay.
    #[cfg_attr(feature = "serde", serde(default))]
    pub fat: bool,
    /// TRANSIENT (the engine graph only): this node IS one pass of `host`,
    /// lowered off its output by [`expand`]. Never authored, never saved.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub pass: Option<Pass>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub host: Option<crate::graph::NodeId>,
}

#[cfg(feature = "serde")]
fn yes() -> bool {
    true
}

fn default_space() -> crate::color::InputSpace {
    crate::color::InputSpace::Explicit(crate::color::Colorspace::LINEAR_REC709)
}

impl Default for BlenderParams {
    fn default() -> Self {
        Self {
            blend: String::new(),
            scene: String::new(),
            camera: String::new(),
            engine: BlenderEngine::File,
            quality: Quality::Viewport,
            shading: Shading::Rendered,
            samples: 0,
            offset: 0,
            transparent: true,
            inputs: 0,
            space: default_space(),
            sync: Sync::Both,
            view: View::Camera,
            orbit: true,
            cards: false,
            passes: Vec::new(),
            fat: false,
            pass: None,
            host: None,
        }
    }
}

impl BlenderParams {
    pub fn input_count(&self) -> usize {
        self.inputs.min(MAX_INPUTS)
    }

    /// The layer this node samples: (the node whose session draws, the
    /// pass id) — itself and the combined picture unless it is a lowered
    /// pass of a host.
    pub fn layer_key(&self, me: crate::graph::NodeId) -> (crate::graph::NodeId, u8) {
        (self.host.unwrap_or(me), self.pass.map(Pass::id).unwrap_or(0))
    }

    /// `"MIST=1,NORMAL=2"`: what the spec names, with the stamp id each
    /// draw carries.
    fn passes_spec(&self) -> String {
        self.passes
            .iter()
            .map(|p| format!("{}={}", p.blender_name(), p.id()))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// The Blender frame a comp tick asks for.
    pub fn frame_of(&self, tick: u64) -> i64 {
        tick as i64 - self.offset
    }

    /// The comp tick a Blender frame plays at (clamped at 0).
    pub fn tick_of(&self, frame: i64) -> u64 {
        (frame + self.offset).max(0) as u64
    }

    /// The `.blend` as typed, `~` expanded; `None` when there is none.
    pub fn blend_path(&self) -> Option<PathBuf> {
        if self.blend.trim().is_empty() {
            return None;
        }
        Some(crate::paths::expand_home(self.blend.trim()))
    }

    /// The space the pixels are taken to be in. `Default` is the
    /// per-format table for FILES; a Blender picture has no file, so it
    /// means Blender's own scene-linear here.
    pub fn effective_space(&self) -> crate::color::InputSpace {
        match self.space {
            crate::color::InputSpace::Default => default_space(),
            s => s,
        }
    }

    /// The part of the params that picks the generated code: only the
    /// space the pixels arrive in (an input transform is WGSL). Everything
    /// else is a message to Blender, never a recompile.
    pub fn topo_tag(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.effective_space().hash(&mut h);
        h.finish()
    }

    /// The one line Blender applies (`{"cmd":"spec",...}`): the whole
    /// scene setup, idempotent, sent whenever any of it changes. `w`/`h`
    /// is the comp format the draw and the render match.
    pub fn spec_line(&self, w: u32, h: u32) -> String {
        format!(
            "{{\"cmd\":\"spec\",\"w\":{w},\"h\":{h},\"scene\":{},\"camera\":{},\"engine\":\"{}\",\"quality\":\"{}\",\"shading\":\"{}\",\"samples\":{},\"transparent\":{},\"inputs\":{},\"view\":\"{}\",\"cards\":{},\"passes\":\"{}\"}}\n",
            live::json_str(&self.scene),
            live::json_str(&self.camera),
            self.engine.spec_name(),
            self.quality.spec_name(),
            self.shading.spec_name(),
            self.samples,
            self.transparent,
            self.input_count(),
            self.view.spec_name(),
            self.cards,
            self.passes_spec(),
        )
    }
}

/// Lower every Blender node's pass outputs onto nodes of their own, the
/// way `lumabands::expand` lowers a LumaKey's bands: the engine only ever
/// sees single-output nodes. A consumer of output k (or the view target
/// on it) gets a node that IS pass `passes[k - 1]` of the host — same
/// inputs, same side tables — and samples the host's layer for that
/// pass. Inert while no Blender node declares a pass.
pub fn expand(g: &mut crate::graph::Graph) {
    use crate::graph::{NodeId, NodeKind};
    let mut multi: Vec<NodeId> = g
        .nodes
        .iter()
        .filter(|(_, k)| matches!(k, NodeKind::Blender(p) if !p.passes.is_empty()))
        .map(|(id, _)| *id)
        .collect();
    if multi.is_empty() {
        return;
    }
    multi.sort_unstable();
    let mut next = g.nodes.keys().copied().max().unwrap_or(0) + 1;
    for src in multi {
        let mut consumers: std::collections::HashMap<u32, Vec<(NodeId, usize)>> =
            Default::default();
        for ((dst, input), s) in g.edges.iter() {
            if *s != src {
                continue;
            }
            let out = g.edge_out.get(&(*dst, *input)).copied().unwrap_or(0);
            consumers.entry(out).or_default().push((*dst, *input));
        }
        let viewed = (g.view == Some(src)).then_some(g.view_out);
        if let Some(v) = viewed {
            consumers.entry(v).or_default();
        }
        let Some(NodeKind::Blender(params)) = g.nodes.get(&src).cloned() else { continue };
        let mut wanted: Vec<u32> = consumers.keys().copied().collect();
        wanted.sort_unstable();
        for out in wanted {
            if out == 0 {
                continue;
            }
            // An output past the list reads as the last pass: an edge left
            // behind by a param edit gives a picture, not black.
            let Some(pass) = params.passes.get((out as usize - 1).min(params.passes.len().saturating_sub(1))).copied() else { continue };
            let id = next;
            next += 1;
            let kid = BlenderParams { pass: Some(pass), host: Some(src), passes: Vec::new(), fat: false, inputs: 0, ..params.clone() };
            g.nodes.insert(id, NodeKind::Blender(kid));
            for i in 0..params.input_count() {
                if let Some(up) = g.input_of(src, i) {
                    let up_out = g.edge_out.get(&(src, i)).copied().unwrap_or(0);
                    g.connect_out(id, i, up, up_out);
                }
            }
            if g.disabled.contains(&src) {
                g.disabled.insert(id);
            }
            if let Some(f) = g.formats.get(&src).copied() {
                g.formats.insert(id, f);
            }
            for (dst, input) in consumers.get(&out).into_iter().flatten() {
                g.connect_out(*dst, *input, id, 0);
            }
            if viewed == Some(out) {
                g.view = Some(id);
                g.view_out = 0;
            }
        }
        if viewed == Some(0) {
            g.view_out = 0;
        }
        // The host keeps its passes for the panel's sake in the editor
        // graph; in the engine it is the combined picture alone.
        if let Some(NodeKind::Blender(p)) = g.nodes.get_mut(&src) {
            p.passes.clear();
        }
    }
    let stale: Vec<(NodeId, usize)> = g
        .edge_out
        .keys()
        .filter(|k| g.edges.get(k).map(|s| g.outputs_of(*s)).unwrap_or(1) <= 1)
        .copied()
        .collect();
    for k in stale {
        g.edge_out.remove(&k);
    }
}

/// The blender executable: `WOBBLE_BLENDER` wins, then the macOS bundle, then
/// the newest `Blender X.Y` under Program Files on Windows (the installer
/// never puts it on PATH), then whatever `blender` resolves to on PATH.
pub fn executable() -> PathBuf {
    if let Ok(p) = std::env::var("WOBBLE_BLENDER") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    let mac = Path::new("/Applications/Blender.app/Contents/MacOS/Blender");
    if mac.is_file() {
        return mac.to_path_buf();
    }
    if cfg!(windows) {
        let pf = std::env::var_os("ProgramFiles")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"C:\Program Files"));
        if let Some(p) = newest_installed(&pf.join("Blender Foundation")) {
            return p;
        }
    }
    PathBuf::from("blender")
}

/// The `blender.exe` in the highest-versioned `Blender X.Y` folder of the
/// Windows installer's root (`Blender 4.10` beats `Blender 4.9`).
fn newest_installed(root: &Path) -> Option<PathBuf> {
    let version = |name: &str| -> Option<Vec<u32>> {
        let v = name.strip_prefix("Blender")?.trim();
        v.split('.').map(|n| n.parse().ok()).collect()
    };
    std::fs::read_dir(root)
        .ok()?
        .flatten()
        .filter_map(|e| {
            let v = version(&e.file_name().to_string_lossy())?;
            let exe = e.path().join("blender.exe");
            exe.is_file().then_some((v, exe))
        })
        .max_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, exe)| exe)
}

/// `blender --version`, for the UI to prove the executable is reachable.
pub fn probe_version() -> Result<String> {
    let exe = executable();
    let out = Command::new(&exe)
        .arg("--version")
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("running {}", exe.display()))?;
    let text = String::from_utf8_lossy(&out.stdout);
    Ok(text.lines().next().unwrap_or("blender").trim().to_string())
}

/// The native module's file name as Blender's Python imports it
/// (`import wobble_native`; the abi3 tag is CPython's stable-ABI
/// convention).
pub const NATIVE_NAME: &str = "wobble_native.abi3.so";

/// Where the built native module is: `WOBBLE_NATIVE`, else next to this
/// executable, else the workspace's `target/release` relative to a test
/// binary. `None` = not built here; the bridge runs without it.
pub fn native_module_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("WOBBLE_NATIVE") {
        if !p.is_empty() && Path::new(&p).is_file() {
            return Some(PathBuf::from(p));
        }
    }
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let lib = if cfg!(target_os = "macos") { "libwobble_native.dylib" } else { "libwobble_native.so" };
    for base in [dir.to_path_buf(), dir.join(".."), dir.join("../../release"), dir.join("../release")] {
        let p = base.join(lib);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// Write the embedded bridge script into `dir` (created if needed) and
/// hand back its path — and the native module beside it when this build
/// has one, so the script's `import wobble_native` finds it. Each is
/// rewritten only when the bytes differ, so a running Blender that
/// already imported them is not disturbed by a no-op launch.
pub fn install_script(dir: &Path) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(SCRIPT_NAME);
    if !std::fs::read_to_string(&path).map(|s| s == SCRIPT).unwrap_or(false) {
        std::fs::write(&path, SCRIPT)?;
    }
    if let Some(native) = native_module_path() {
        let dst = dir.join(NATIVE_NAME);
        let bytes = std::fs::read(&native)?;
        if std::fs::read(&dst).map(|b| b != bytes).unwrap_or(true) {
            std::fs::write(&dst, &bytes)?;
        }
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_offset_maps_ticks_and_frames_both_ways() {
        let p = BlenderParams { offset: 1000, ..Default::default() };
        assert_eq!(p.frame_of(1001), 1);
        assert_eq!(p.tick_of(48), 1048);
        assert_eq!(p.tick_of(-5000), 0, "a tick is never negative");
        let z = BlenderParams::default();
        assert_eq!(z.frame_of(12), 12);
    }

    #[test]
    fn only_the_colour_space_is_topology() {
        let a = BlenderParams::default();
        let mut b = a.clone();
        b.samples = 64;
        b.quality = Quality::Final;
        b.offset = 7;
        b.inputs = 3;
        assert_eq!(a.topo_tag(), b.topo_tag(), "knobs are messages, not recompiles");
        b.space = crate::color::InputSpace::Working;
        assert_ne!(a.topo_tag(), b.topo_tag(), "the input transform is WGSL");
    }

    #[test]
    fn pass_outputs_lower_onto_nodes_of_their_own() {
        use crate::graph::{Graph, NodeKind};
        let mut g = Graph::default();
        let host = BlenderParams { passes: vec![Pass::Z, Pass::Normal], ..Default::default() };
        g.nodes.insert(1, NodeKind::Blender(host));
        g.nodes.insert(2, NodeKind::Grade(Default::default()));
        g.nodes.insert(3, NodeKind::Grade(Default::default()));
        g.connect_out(2, 0, 1, 0);
        g.connect_out(3, 0, 1, 2); // the normal pass
        g.view = Some(1);
        g.view_out = 1; // viewing the mist
        expand(&mut g);
        let kid_of = |dst: u64| g.input_of(dst, 0).unwrap();
        assert_eq!(kid_of(2), 1, "the combined stays on the host");
        let normal = kid_of(3);
        assert_ne!(normal, 1);
        match g.nodes.get(&normal) {
            Some(NodeKind::Blender(p)) => {
                assert_eq!((p.pass, p.host), (Some(Pass::Normal), Some(1)));
                assert_eq!(p.layer_key(normal), (1, Pass::Normal.id()));
                assert!(p.passes.is_empty());
            }
            other => panic!("{other:?}"),
        }
        let v = g.view.unwrap();
        assert!(v != 1 && v != normal, "the viewed mist is its own node");
        assert!(matches!(g.nodes.get(&v), Some(NodeKind::Blender(p)) if p.pass == Some(Pass::Z)));
        assert_eq!(g.view_out, 0);
        assert!(g.edge_out.is_empty(), "every edge points at a single-output node now");
        assert!(matches!(g.nodes.get(&1), Some(NodeKind::Blender(p)) if p.passes.is_empty() && p.pass.is_none()));
    }

    #[test]
    fn the_spec_is_one_flat_line() {
        let p = BlenderParams { camera: "Cam \"A\"".into(), inputs: 2, ..Default::default() };
        let line = p.spec_line(2048, 1080);
        assert!(line.ends_with('\n') && line.matches('\n').count() == 1);
        assert!(line.contains("\"camera\":\"Cam \\\"A\\\"\""));
        assert!(line.contains("\"w\":2048,\"h\":1080"));
        assert!(line.contains("\"shading\":\"RENDERED\""));
        assert!(line.contains("\"inputs\":2"));
        assert!(line.contains("\"view\":\"camera\",\"cards\":false"));
        // The script's parser is json.loads: the line must be JSON.
        assert!(live::parse_flat(line.trim()).is_some());
    }

    #[test]
    fn the_newest_installed_blender_wins_by_version_not_by_name() {
        let root = std::env::temp_dir().join(format!("wobble-bf-{}", std::process::id()));
        for (dir, exe) in [("Blender 4.9", true), ("Blender 4.10", true), ("Blender 5.0", false), ("Other", true)] {
            let d = root.join(dir);
            std::fs::create_dir_all(&d).unwrap();
            if exe {
                std::fs::write(d.join("blender.exe"), b"").unwrap();
            }
        }
        let got = newest_installed(&root).unwrap();
        assert_eq!(got, root.join("Blender 4.10").join("blender.exe"), "5.0 has no exe; 4.10 > 4.9");
        assert!(newest_installed(&root.join("missing")).is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_script_is_embedded_and_installs_once() {
        assert!(SCRIPT.contains("WOBBLE_PORT"));
        let dir = std::env::temp_dir().join(format!("wobble-script-{}", std::process::id()));
        let p = install_script(&dir).unwrap();
        let m1 = std::fs::metadata(&p).unwrap().modified().unwrap();
        std::thread::sleep(web_time::Duration::from_millis(20));
        install_script(&dir).unwrap();
        assert_eq!(std::fs::metadata(&p).unwrap().modified().unwrap(), m1);
    }
}
