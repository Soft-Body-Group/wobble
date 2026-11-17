//! Sammie: click-to-roto with SAM 2, as a node whose matte plays back like
//! a bake.
//!
//! The tool (Zarxrax's *Sammie-Roto 2*, github.com/Zarxrax/Sammie-Roto-2)
//! bundles Meta's SAM 2.1 video segmenter (and EfficientTAM, a lighter
//! SAM 2) with MatAnyone 2, a video MATTING network that turns a hard
//! segment into a soft, hair-aware alpha. Click a few points on the subject
//! in one frame — include and exclude — and SAM 2 tracks the object
//! through the shot. All of it is PyTorch, so the engine never runs it: the
//! app renders the plate to disk, runs the tool's models as a process on an
//! explicit click (the CorridorKey precedent), and the node then samples the
//! matte through its reader slot — a Read's cost, and a loaded matte CLEARS
//! the slow taint above it like a GraphBake.
//!
//! What it renders:
//!
//! - **A matte loaded** (`loaded`, transient, set by the app like a
//!   GraphBake's `baked`): `Cutout` is the plate × the matte (premultiplied,
//!   the matte in alpha — what a Blend's foreground wants); `Matte` is the
//!   matte itself on the MASK pipe, in every channel, like a Spline.
//! - **Nothing loaded**: `Cutout` passes the plate through (so the node can
//!   be viewed while its points are placed); `Matte` is empty.
//!
//! Points live on the node in NORMALISED plate coordinates (0..1, y down,
//! the viewer's output space), so a proxy toggle does not move them, and
//! each belongs to the comp frame it was clicked on. Points on several
//! frames are several prompts for the same object: SAM 2 takes each as a
//! correction where the track drifted.
//!
//! Licence (manifesto rule 10): Sammie itself is GPL-3.0 and is never
//! bundled — the user installs it, and wobble runs it as a separate
//! process through `tools/wobble_sammie.py`, which imports only the SAM 2
//! (Apache-2.0) and MatAnyone packages it vendors. SAM 2 and EfficientTAM
//! are Apache-2.0: a plain segment is free to deliver. **MatAnyone is the
//! NTU S-Lab License 1.0, non-commercial**, so a node with matting on is
//! tagged, and so is a node whose loaded matte was MADE with it
//! (`nc_take`) — switching matting off afterwards does not change what the
//! pixels on disk are.

/// Which segmenter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SamModel {
    /// SAM 2.1 base+: the tool's own default.
    Base,
    /// SAM 2.1 large: the best tracker there is with open weights —
    /// steadier on long shots, occlusion and look-alikes. Ours by default.
    #[default]
    Large,
    /// EfficientTAM 512: fastest, least detail.
    Efficient,
}

impl SamModel {
    pub const ALL: [SamModel; 3] = [SamModel::Large, SamModel::Base, SamModel::Efficient];

    pub fn label(self) -> &'static str {
        match self {
            SamModel::Base => "SAM 2.1 base+",
            SamModel::Large => "SAM 2.1 large (best)",
            SamModel::Efficient => "EfficientTAM (fast)",
        }
    }

    /// The driver's word for it, which is also the tool's model key.
    pub fn arg(self) -> &'static str {
        match self {
            SamModel::Base => "Base",
            SamModel::Large => "Large",
            SamModel::Efficient => "Efficient",
        }
    }
}

/// What turns the segment into a matte.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Matting {
    /// The segment as it is: a hard edge, free to deliver.
    #[default]
    Off,
    /// MatAnyone 2 on top of the segment: soft edges, hair, motion blur.
    /// NON-COMMERCIAL weights.
    MatAnyone2,
}

impl Matting {
    pub const ALL: [Matting; 2] = [Matting::Off, Matting::MatAnyone2];

    pub fn label(self) -> &'static str {
        match self {
            Matting::Off => "off (hard segment)",
            Matting::MatAnyone2 => "MatAnyone 2 (non-commercial)",
        }
    }

    pub fn arg(self) -> &'static str {
        match self {
            Matting::Off => "off",
            Matting::MatAnyone2 => "matanyone2",
        }
    }

    pub fn commercial(self) -> bool {
        self == Matting::Off
    }
}

/// What goes downstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SammieOut {
    /// Plate × matte, matte in alpha, on the image pipe.
    #[default]
    Cutout,
    /// The matte, on the mask pipe.
    Matte,
}

impl SammieOut {
    pub const ALL: [SammieOut; 2] = [SammieOut::Cutout, SammieOut::Matte];

    pub fn label(self) -> &'static str {
        match self {
            SammieOut::Cutout => "cutout (plate × matte)",
            SammieOut::Matte => "matte",
        }
    }
}

/// One click: where, on which comp frame, and which side of the object.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SamPoint {
    pub frame: i64,
    /// Normalised plate coordinates, 0..1, y down.
    pub x: f32,
    pub y: f32,
    /// Include (on the object) or exclude (not it).
    pub positive: bool,
}

/// The node's authored state, plus two transient fields.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct SammieParams {
    /// Reader slot the matte plays through.
    pub source: usize,
    pub model: SamModel,
    pub matting: Matting,
    /// Matting's internal resolution: the plate's SHORTER side is cut to
    /// this before MatAnyone sees it (0 = the plate's own size), and the
    /// matte is scaled back up. Sammie's own "internal resolution": it
    /// applies to matting only — SAM 2 always segments at 1024x1024
    /// whatever the plate, so there is nothing to set for that stage.
    pub matting_res: u32,
    pub points: Vec<SamPoint>,
    pub output: SammieOut,
    /// The matte LOADED in the reader was made with non-commercial
    /// weights. Authored (it saves): it is a fact about pixels on disk,
    /// set by the app when it adopts a run, and only a run without them
    /// clears it.
    pub nc_take: bool,
    /// Auto preview: a click on the viewer segments the frame on screen at
    /// once (the app keeps the model loaded between clicks). UI state, not
    /// pixels: not in the recipe.
    pub auto_preview: bool,
    /// TRANSIENT: a matte is loaded in the reader slot. Set by the app on
    /// the engine-graph copy only; flipping it recompiles.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub loaded: bool,
    /// TRANSIENT: the comp frame the first matte file holds.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub start: Option<i64>,
}

impl Default for SammieParams {
    fn default() -> Self {
        Self {
            source: 0,
            model: SamModel::Large,
            matting: Matting::Off,
            matting_res: 0,
            points: Vec::new(),
            output: SammieOut::Cutout,
            nc_take: false,
            auto_preview: true,
            loaded: false,
            start: None,
        }
    }
}

impl SammieParams {
    pub const NAME: &'static str = "Sammie";
    /// The plate.
    pub const INPUTS: usize = 1;
    pub const VEC4_SLOTS: usize = 0;

    pub fn new(source: usize) -> Self {
        Self { source, ..Default::default() }
    }

    /// Topology: whether a matte is loaded, and which pipe the output is
    /// on (a cutout fuses the plate in, a matte does not).
    pub fn topo_tag(&self) -> u64 {
        self.loaded as u64 | (self.output as u64) << 1
    }

    /// Is anything this node shows, or would make, non-commercial?
    pub fn commercial(&self) -> bool {
        self.matting.commercial() && !self.nc_take
    }

    /// The plate is read by the shader only for a cutout, or while nothing
    /// is loaded (the pass-through).
    pub fn samples_plate(&self) -> bool {
        !self.loaded || self.output == SammieOut::Cutout
    }

    /// The settings that shape pixels, for the app's result key: change one
    /// and the matte has to be made again, into a new directory. Points
    /// are compared by their bits. The matting resolution rides in the
    /// matting field's high bits, and only while matting is on (it shapes
    /// nothing otherwise): at 0 the key is the one it always was, so mattes
    /// made before the setting existed still load.
    pub fn matting_key(&self) -> u32 {
        let res = if self.matting == Matting::Off { 0 } else { self.matting_res };
        self.matting as u32 | res << 8
    }

    pub fn recipe(&self) -> (u8, u32, Vec<(i64, u32, u32, bool)>) {
        let mut pts: Vec<(i64, u32, u32, bool)> =
            self.points.iter().map(|p| (p.frame, p.x.to_bits(), p.y.to_bits(), p.positive)).collect();
        // The order clicks were made in is not a prompt.
        pts.sort();
        (self.model as u8, self.matting_key(), pts)
    }

    /// The comp frames that carry points, in order.
    pub fn keyframes(&self) -> Vec<i64> {
        let mut f: Vec<i64> = self.points.iter().map(|p| p.frame).collect();
        f.sort();
        f.dedup();
        f
    }

    pub fn input_name(_input: usize) -> &'static str {
        "image (the plate)"
    }

    /// The WGSL for a loaded matte. The tool writes the matte at the
    /// plate's own size in every channel; `plate` is the fused input 0,
    /// present for a cutout.
    pub fn emit_loaded(
        &self,
        src: &str,
        pos: &str,
        plate: Option<&str>,
        body: &mut String,
        fresh: &mut dyn FnMut() -> String,
    ) -> String {
        use std::fmt::Write;
        let m = fresh();
        writeln!(body, "    let {m} = clamp(sample_px({src}, {pos}).r, 0.0, 1.0);").unwrap();
        let o = fresh();
        match (self.output, plate) {
            (SammieOut::Cutout, Some(p)) => writeln!(body, "    let {o} = {p} * {m};").unwrap(),
            _ => writeln!(body, "    let {o} = vec4f({m});").unwrap(),
        }
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(frame: i64, x: f32) -> SamPoint {
        SamPoint { frame, x, y: 0.5, positive: true }
    }

    #[test]
    fn only_loading_and_the_pipe_change_the_topology() {
        let mut a = SammieParams::new(3);
        let b = a.clone();
        a.points.push(pt(1, 0.2));
        a.model = SamModel::Large;
        a.matting = Matting::MatAnyone2;
        assert_eq!(a.topo_tag(), b.topo_tag(), "settings are the tool's, not code");
        a.loaded = true;
        assert_ne!(a.topo_tag(), b.topo_tag());
        let c = a.clone();
        a.output = SammieOut::Matte;
        assert_ne!(a.topo_tag(), c.topo_tag());
    }

    #[test]
    fn the_recipe_is_the_prompts_and_the_models_and_nothing_else() {
        let mut a = SammieParams::default();
        a.points = vec![pt(1, 0.2), pt(4, 0.7)];
        let mut b = a.clone();
        b.output = SammieOut::Matte;
        b.source = 9;
        b.loaded = true;
        b.nc_take = true;
        b.auto_preview = false;
        b.points.reverse();
        assert_eq!(a.recipe(), b.recipe(), "output, slot and click order are not a re-run");
        for f in [
            &|p: &mut SammieParams| p.model = SamModel::Efficient,
            &|p: &mut SammieParams| p.matting = Matting::MatAnyone2,
            &|p: &mut SammieParams| p.points[0].x = 0.21,
            &|p: &mut SammieParams| p.points[0].positive = false,
            &|p: &mut SammieParams| p.points[1].frame = 5,
            &|p: &mut SammieParams| p.points.pop().map(|_| ()).unwrap(),
        ] as [&dyn Fn(&mut SammieParams); 6]
        {
            let mut c = a.clone();
            f(&mut c);
            assert_ne!(a.recipe(), c.recipe());
        }
        assert_eq!(a.keyframes(), vec![1, 4]);
    }

    #[test]
    fn matting_res_is_a_rerun_only_while_matting() {
        let mut a = SammieParams::default();
        a.points = vec![pt(1, 0.2)];
        let mut b = a.clone();
        b.matting_res = 720;
        assert_eq!(a.recipe(), b.recipe(), "no matting, nothing to resize");
        a.matting = Matting::MatAnyone2;
        b.matting = Matting::MatAnyone2;
        assert_ne!(a.recipe(), b.recipe());
        // At 0 the key is the old u8's number (same JSON), so old runs load.
        assert_eq!(a.matting_key(), Matting::MatAnyone2 as u32);
    }

    #[test]
    fn matting_and_a_matted_take_are_non_commercial() {
        let mut p = SammieParams::default();
        assert!(p.commercial());
        p.matting = Matting::MatAnyone2;
        assert!(!p.commercial());
        // Switching it off does not launder a matte already made with it.
        p.matting = Matting::Off;
        p.nc_take = true;
        assert!(!p.commercial());
    }

    #[test]
    fn the_loaded_shader_samples_and_picks_the_pipe() {
        let mut n = 0;
        let mut fresh = || {
            n += 1;
            format!("t{n}")
        };
        let p = SammieParams { loaded: true, ..Default::default() };
        let mut body = String::new();
        let out = p.emit_loaded("src0", "pos", Some("plate"), &mut body, &mut fresh);
        assert!(body.contains("sample_px(src0, pos).r"), "{body}");
        assert!(body.contains("plate * t1"), "{body}");
        assert_eq!(out, "t2");
        let p = SammieParams { loaded: true, output: SammieOut::Matte, ..Default::default() };
        let mut body = String::new();
        p.emit_loaded("src0", "pos", None, &mut body, &mut || "m".into());
        assert!(body.contains("vec4f(m)"), "{body}");
        assert!(!p.samples_plate());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn transients_are_never_saved() {
        let p = SammieParams {
            loaded: true,
            start: Some(7),
            points: vec![pt(3, 0.25)],
            nc_take: true,
            ..SammieParams::new(2)
        };
        let j = serde_json::to_string(&p).unwrap();
        assert!(!j.contains("loaded") && !j.contains("start"), "{j}");
        let back: SammieParams = serde_json::from_str(&j).unwrap();
        assert!(!back.loaded && back.start.is_none());
        assert_eq!(back.points, p.points);
        assert!(back.nc_take, "what the pixels were made with saves");
        let old: SammieParams = serde_json::from_str(r#"{"source":1}"#).unwrap();
        assert_eq!(old.model, SamModel::Large);
    }
}
