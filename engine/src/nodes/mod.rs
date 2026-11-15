//! One module per built-in node: its params, its WGSL and every fact the
//! graph, the codegen and the uniform packer need about it.
//!
//! `NodeKind` still names every node and every match on it still exists —
//! that is what makes the compiler list the arms a new node forgot — but
//! each arm is one line pointing here. A new node is one file in this
//! directory, an `impl NodeDef` in it, and those one-liners.

pub mod blank;
pub mod starfield;
pub mod planet;
pub mod plasma;
pub mod particles;
pub mod bokehkernel;
pub mod glintkernel;
pub mod blur;
pub mod colorcompress;
pub mod colorslice;
pub mod contrast;
pub mod convolve;
pub mod fft;
pub mod corridorkey;
pub mod sammie;
pub mod seedvr;
pub mod depth;
pub mod ltxhdr;
pub mod hdrcnn;
pub mod despill;
pub mod dilate;
pub mod edgedetect;
pub mod expression;
pub mod flipflop;
pub mod mirror;
pub mod flow;
pub mod rifetime;
pub mod freeze;
pub mod framefix;
pub mod gamma;
pub mod tonemap;
pub mod huerotate;
pub mod lut;
pub mod text;
pub mod burnin;
pub mod lightwrap;
pub mod glint;
pub mod diffuse;
pub mod glow;
pub mod godrays;
pub mod lensflare;
pub mod edgeextend;
pub mod gradient;
pub mod gridwarp;
pub mod lensdistort;
pub mod warp;
pub mod grain;
pub mod median;
pub mod stabilize;
pub mod camtrack;
pub mod highlightrebuild;
pub mod motionblur;
pub mod genai;
pub mod prompt;
pub mod render3d;
pub mod saturation;
pub mod screenkey;
pub mod softcrop;
pub mod stage;
pub mod upscale;
pub mod uvwarp;
pub mod env;
pub mod refract;
pub mod vectorblur;
pub mod vectorwarp;
pub mod todata;
pub mod torgba;

use crate::codegen::Emit;

/// The engine-side contract of a built-in node that fuses into the main
/// pass. Every item is required on purpose: a node that forgets to say how
/// many inputs it has should fail to compile, not default to something.
pub trait NodeDef {
    /// What the graph, the menus and the panel call it.
    const NAME: &'static str;
    /// Input ports, the mask included.
    const INPUTS: usize;
    /// Which port carries the mask, if any: coverage that MIXES the effect
    /// back against the input.
    const MASK_INPUT: Option<usize>;
    /// Which port carries the modulate matte, if any: coverage that SCALES
    /// the effect itself, per pixel (half coverage = half the blur, not a
    /// half-and-half mix). A matte on the wire either way; the socket is
    /// what differs. A node that pays for it must have a path that does not
    /// when it is unwired (`stage::Wired`).
    const MOD_INPUT: Option<usize> = None;
    /// vec4 slots in the shared uniform block; 0 for a node with nothing
    /// to pack. The span never moves with a param VALUE — that would make
    /// a slider drag a recompile.
    const VEC4_SLOTS: usize;

    /// The part of the params that picks the generated code, folded into
    /// the topology hash. Everything else is a uniform and must stay out.
    fn topo_tag(&self) -> u64;
    /// Write the uniform values into this node's span, `VEC4_SLOTS * 4`
    /// floats long, in the order the WGSL reads them.
    fn pack(&self, span: &mut [f32]);
    /// Emit the node into the fused pass and return the WGSL expression
    /// for its colour. See [`Emit`] for what is on offer.
    fn emit(&self, e: &mut Emit) -> String;
}

/// Every pass module a built-in stage can dispatch, so the compositor can
/// compile them all up front rather than on a node's first use. A module
/// missing here still works — it compiles the first time it is needed,
/// which costs one frame.
/// The single-channel modules matte chains run (`StageDef::program_one_
/// channel`), compiled at startup only where the device can write a
/// one-channel texture (`stage::one_channel`).
pub fn builtin_one_channel_wgsl() -> Vec<&'static str> {
    vec![
        blur::ONE_WGSL,
        blur::ONE_MOD_WGSL,
        dilate::ONE_WGSL,
        dilate::ONE_MOD_WGSL,
        median::ONE_H_WGSL,
        median::ONE_V_WGSL,
        convolve::ONE_WGSL,
        convolve::ONE_MOD_WGSL,
        vectorblur::ONE_WGSL,
        vectorblur::ONE_MOD_WGSL,
    ]
}

pub fn builtin_pass_wgsl() -> Vec<&'static str> {
    vec![
        blur::WGSL,
        blur::MOD_WGSL,
        dilate::WGSL,
        dilate::MOD_WGSL,
        edgedetect::WGSL,
        median::H_WGSL,
        median::V_WGSL,
        highlightrebuild::REF_WGSL,
        highlightrebuild::H_WGSL,
        highlightrebuild::V_WGSL,
        highlightrebuild::REBUILD_WGSL,
        convolve::WGSL,
        convolve::MOD_WGSL,
        fft::FFT_WGSL,
        fft::LOAD_IMAGE_WGSL,
        fft::LOAD_KERNEL_WGSL,
        fft::MULTIPLY_WGSL,
        fft::OUT_WGSL,
        fft::OUT_MOD_WGSL,
        glow::DOWN_WGSL,
        glow::UP_WGSL,
        glow::COMP_WGSL,
        contrast::DOWN_WGSL,
        contrast::FINAL_WGSL,
        godrays::EXTRACT_WGSL,
        godrays::EXTRACT_MATTE_WGSL,
        godrays::RADIAL_WGSL,
        edgeextend::SEED_WGSL,
        edgeextend::SEED_MATTE_WGSL,
        edgeextend::JFA_WGSL,
        edgeextend::PREP_WGSL,
        edgeextend::PREP_MATTE_WGSL,
        edgeextend::DOWN_WGSL,
        edgeextend::UP_WGSL,
        edgeextend::COMP_WGSL,
        edgeextend::COMP_MATTE_WGSL,
        glint::EXTRACT_WGSL,
        glint::STREAK_WGSL,
        glint::MORPH_WGSL,
        diffuse::PREP_WGSL,
        diffuse::PREP_MOD_WGSL,
        diffuse::PREP_BOTH_WGSL,
        diffuse::DOWN_WGSL,
        diffuse::UP_WGSL,
        diffuse::COMP_WGSL,
        lightwrap::PACK_H_WGSL,
        lightwrap::V_WGSL,
        lightwrap::COMP_WGSL,
        vectorblur::WGSL,
        vectorblur::MOD_WGSL,
    ]
}
