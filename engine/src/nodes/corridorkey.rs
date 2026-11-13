//! CorridorKey: Corridor Digital's neural green / blue screen keyer, as a
//! node whose result plays back like a bake.
//!
//! The network (Niko Pueringer's *CorridorKey*, github.com/nikopueringer/
//! CorridorKey) solves the UNMIXING problem a colour-difference key cannot:
//! it reconstructs the straight foreground colour and a linear alpha
//! through motion blur, defocus and hair, from the plate plus a COARSE
//! alpha hint. It is a 300 MB PyTorch / MLX model, so the engine never runs
//! it. The app renders the node's inputs to disk, runs the tool as a
//! process on an explicit click (the Blender / planar / GenAI precedent),
//! and the node then samples the result through its reader slot — a Read's
//! cost, cyan in the graph, and a loaded key CLEARS the slow taint above it
//! like a GraphBake (the pixels come off disk).
//!
//! What it renders:
//!
//! - **A key loaded** (`loaded`, transient, set by the app like a
//!   GraphBake's `baked`): the tool's `Processed` frames — despilled
//!   foreground PREMULTIPLIED by the cleaned matte, linear Rec.709, which the
//!   reader leaf carries into the working space (a matrix, safe on
//!   premultiplied pixels). `output` picks premultiplied or the matte alone.
//! - **Nothing loaded**: input 0 passes through untouched, so dropping the
//!   node into a chain changes nothing until a key is run.
//!
//! Licence (manifesto rule 10): the tool is CC BY-NC-SA 4.0 with an
//! explicit grant — "you may use this tool for any lawful purpose,
//! including for processing images as part of a commercial project". The
//! PIXELS are free to deliver, so the node is not tagged non-commercial;
//! what the licence forbids is repackaging or selling the tool and paid
//! inference services, neither of which a node in an open-source comp does
//! (the user installs the tool themselves, into their own home). The panel
//! shows the licence before anything is installed.

/// Which model to run: the tool has separate weights for green and blue,
/// and can pick from the plate's own screen pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Screen {
    #[default]
    Auto,
    Green,
    Blue,
}

impl Screen {
    pub const ALL: [Screen; 3] = [Screen::Auto, Screen::Green, Screen::Blue];

    pub fn label(self) -> &'static str {
        match self {
            Screen::Auto => "auto (from the plate)",
            Screen::Green => "green",
            Screen::Blue => "blue",
        }
    }

    /// The tool's own word for it.
    pub fn arg(self) -> &'static str {
        match self {
            Screen::Auto => "auto",
            Screen::Green => "green",
            Screen::Blue => "blue",
        }
    }
}

/// What the node hands downstream once a key is loaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CorridorOut {
    /// Despilled foreground × matte, matte in alpha: what a Blend wants.
    #[default]
    Premult,
    /// The matte in rgb, alpha 1: readable by any mask port.
    Matte,
}

impl CorridorOut {
    pub const ALL: [CorridorOut; 2] = [CorridorOut::Premult, CorridorOut::Matte];

    pub fn label(self) -> &'static str {
        match self {
            CorridorOut::Premult => "premultiplied",
            CorridorOut::Matte => "matte",
        }
    }

    fn code(self) -> f32 {
        match self {
            CorridorOut::Premult => 0.0,
            CorridorOut::Matte => 1.0,
        }
    }
}

/// The node's authored state, plus two transient fields.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct CorridorKeyParams {
    /// Reader slot the keyed frames play through.
    pub source: usize,
    pub screen: Screen,
    /// The tool's traditional despill on the reconstructed foreground,
    /// 0..1. Its default.
    pub despill: f32,
    /// Multiplier on the refiner's deltas: how hard the detail pass pushes.
    pub refiner: f32,
    /// Drop matte islands smaller than `despeckle_px` pixels (tracking
    /// markers, noise).
    pub despeckle: bool,
    pub despeckle_px: u32,
    /// What goes downstream while loaded.
    pub output: CorridorOut,
    /// TRANSIENT: a key is loaded in the reader slot. Set by the app on the
    /// engine-graph copy only; flipping it recompiles (sample vs pass
    /// through), exactly like a GraphBake's `baked`.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub loaded: bool,
    /// TRANSIENT: the comp frame the first keyed file holds, so the key
    /// plays back where the comp it was made from lives.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub start: Option<i64>,
}

impl Default for CorridorKeyParams {
    fn default() -> Self {
        Self {
            source: 0,
            screen: Screen::Auto,
            despill: 0.5,
            refiner: 1.0,
            despeckle: true,
            despeckle_px: 400,
            output: CorridorOut::Premult,
            loaded: false,
            start: None,
        }
    }
}

impl CorridorKeyParams {
    pub const NAME: &'static str = "CorridorKey";
    /// The plate, and a coarse alpha hint (white = the subject, roughly).
    pub const INPUTS: usize = 2;
    /// One slot: the output mode.
    pub const VEC4_SLOTS: usize = 1;

    pub fn new(source: usize) -> Self {
        Self { source, ..Default::default() }
    }

    /// Topology: only whether a key is loaded changes the shader. The
    /// settings shape what the TOOL renders, not what the shader does.
    pub fn topo_tag(&self) -> u64 {
        self.loaded as u64
    }

    /// The settings that shape pixels, for the app's result key: change
    /// one and the key has to be run again into a new directory.
    pub fn recipe(&self) -> (u8, u32, u32, bool, u32) {
        (
            match self.screen {
                Screen::Auto => 0,
                Screen::Green => 1,
                Screen::Blue => 2,
            },
            self.despill.to_bits(),
            self.refiner.to_bits(),
            self.despeckle,
            self.despeckle_px,
        )
    }

    pub fn input_name(input: usize) -> &'static str {
        match input {
            0 => "image (the plate)",
            _ => "hint (coarse matte, optional)",
        }
    }

    pub fn pack(&self, out: &mut [f32]) {
        out[0] = self.output.code();
        out[1] = 0.0;
        out[2] = 0.0;
        out[3] = 0.0;
    }

    /// The WGSL for a loaded key: the file sampled at the output pixel
    /// (the tool writes frames at the plate's own size), the input
    /// transform on the premultiplied colour (linear to linear: a matrix
    /// commutes with the multiply), then the output pick.
    pub fn emit_loaded(
        src: &str,
        pos: &str,
        slot: usize,
        idt: Option<&str>,
        body: &mut String,
        fresh: &mut dyn FnMut() -> String,
    ) -> String {
        use std::fmt::Write;
        let v = fresh();
        writeln!(body, "    let {v} = sample_px({src}, {pos});").unwrap();
        let c = match idt {
            Some(f) => {
                let c = fresh();
                writeln!(body, "    let {c} = vec4f({f}({v}.rgb), {v}.a);").unwrap();
                c
            }
            None => v,
        };
        let o = fresh();
        writeln!(
            body,
            "    let {o} = select({c}, vec4f(vec3f({c}.a), 1.0), P.v[{slot}].x > 0.5);"
        )
        .unwrap();
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_loading_changes_the_topology() {
        let mut a = CorridorKeyParams::new(3);
        let b = a.clone();
        a.despill = 0.9;
        a.screen = Screen::Blue;
        a.output = CorridorOut::Matte;
        assert_eq!(a.topo_tag(), b.topo_tag(), "settings are the tool's, not code");
        a.loaded = true;
        assert_ne!(a.topo_tag(), b.topo_tag());
    }

    #[test]
    fn the_recipe_is_every_pixel_shaping_setting_and_nothing_else() {
        let a = CorridorKeyParams::default();
        let mut b = a.clone();
        b.output = CorridorOut::Matte;
        b.source = 9;
        b.loaded = true;
        assert_eq!(a.recipe(), b.recipe(), "the output pick is a uniform, not a re-run");
        for f in [
            &|p: &mut CorridorKeyParams| p.screen = Screen::Blue,
            &|p: &mut CorridorKeyParams| p.despill = 0.7,
            &|p: &mut CorridorKeyParams| p.refiner = 1.5,
            &|p: &mut CorridorKeyParams| p.despeckle = false,
            &|p: &mut CorridorKeyParams| p.despeckle_px = 10,
        ] as [&dyn Fn(&mut CorridorKeyParams); 5]
        {
            let mut c = a.clone();
            f(&mut c);
            assert_ne!(a.recipe(), c.recipe());
        }
    }

    #[test]
    fn the_loaded_shader_samples_transforms_and_picks() {
        let mut body = String::new();
        let mut n = 0;
        let out = CorridorKeyParams::emit_loaded("src0", "pos", 4, Some("cs_idt_x"), &mut body, &mut || {
            n += 1;
            format!("t{n}")
        });
        assert!(body.contains("sample_px(src0, pos)"));
        assert!(body.contains("cs_idt_x(t1.rgb)"), "{body}");
        assert!(body.contains("P.v[4].x > 0.5"));
        assert_eq!(out, "t3");
        let mut body = String::new();
        CorridorKeyParams::emit_loaded("src0", "pos", 0, None, &mut body, &mut || "q".into());
        assert!(!body.contains("cs_idt"));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn transients_are_never_saved() {
        let p = CorridorKeyParams { loaded: true, start: Some(7), ..CorridorKeyParams::new(2) };
        let j = serde_json::to_string(&p).unwrap();
        assert!(!j.contains("loaded") && !j.contains("start"), "{j}");
        let back: CorridorKeyParams = serde_json::from_str(&j).unwrap();
        assert!(!back.loaded && back.start.is_none());
        assert_eq!(back.source, 2);
        // An old setup with no such fields loads at the defaults.
        let old: CorridorKeyParams = serde_json::from_str(r#"{"source":1}"#).unwrap();
        assert_eq!(old.despill, 0.5);
    }
}
