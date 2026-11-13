//! The app half of each built-in node: its properties panel, its graph
//! glyph and its channel bindings. The engine half — params, WGSL, the
//! facts the graph needs — is `engine::nodes`, one file with the same name.
//!
//! Same rule as there: `NodeKind` dispatches, and the arms in `lib.rs`,
//! `editor.rs`, `anim.rs`, `curves.rs` and `widgets.rs` are one-liners
//! into here. The `Channel` variants themselves stay in `anim.rs` — one
//! flat enum, because a channel's NAME is what a patch cable and a setup
//! file store.

pub mod blank;
pub mod starfield;
pub mod planet;
pub mod plasma;
pub mod particles;
pub mod bokehkernel;
pub mod glintkernel;
pub mod prompt;
pub mod scene3d;
pub mod blur;
pub mod colorcompress;
pub mod colorslice;
pub mod convolve;
pub mod corridorkey;
pub mod sammie;
pub mod seedvr;
pub mod depth;
pub mod ltxhdr;
pub mod hdrcnn;
pub mod expression;
pub mod flow;
pub mod rifetime;
pub mod freeze;
pub mod framefix;
pub mod despill;
pub mod huerotate;
pub mod lut;
pub mod text;
pub mod burnin;
pub mod mirror;
pub mod lensdistort;
pub mod flipflop;
pub mod gamma;
pub mod tonemap;
pub mod lightwrap;
pub mod dilate;
pub mod edgedetect;
pub mod glow;
pub mod godrays;
pub mod lensflare;
pub mod contrast;
pub mod edgeextend;
pub mod glint;
pub mod diffuse;
pub mod gradient;
pub mod gridwarp;
pub mod warp;
pub mod grain;
pub mod median;
pub mod highlightrebuild;
pub mod stabilize;
pub mod camtrack;
pub mod motionblur;
pub mod saturation;
pub mod screenkey;
pub mod softcrop;
pub mod upscale;
pub mod uvwarp;
pub mod env;
pub mod vectorblur;
pub mod vectorwarp;
pub mod todata;
pub mod torgba;
pub mod switch;
pub mod loopout;
pub mod custom;
pub mod breakout;
pub mod recombine;
pub mod channelselect;
pub mod read;
pub mod shader;
pub mod tomatte;
pub mod mult;
pub mod grid;

/// Every channel a node module declares, in module order. `anim::ALL` holds
/// the rest; the two together are the channel universe a setup can name.
pub fn channels() -> impl Iterator<Item = crate::anim::Channel> {
    // ToData is not here on purpose: its only control is a channel PICK,
    // so it declares no channels at all.
    saturation::CHANNELS
        .iter()
        .chain(blur::CHANNELS)
        .chain(colorcompress::CHANNELS)
        .chain(colorslice::CHANNELS)
        .chain(huerotate::CHANNELS)
        .chain(lut::CHANNELS)
        .chain(text::CHANNELS)
        .chain(burnin::CHANNELS)
        .chain(mirror::CHANNELS)
        .chain(lensdistort::CHANNELS)
        .chain(gamma::CHANNELS)
        .chain(tonemap::CHANNELS)
        .chain(despill::CHANNELS)
        .chain(lightwrap::CHANNELS)
        .chain(screenkey::CHANNELS)
        .chain(dilate::CHANNELS)
        .chain(edgedetect::CHANNELS)
        .chain(median::CHANNELS)
        .chain(highlightrebuild::CHANNELS)
        .chain(ltxhdr::CHANNELS)
        .chain(stabilize::CHANNELS)
        .chain(camtrack::CHANNELS)
        .chain(convolve::CHANNELS)
        .chain(glow::CHANNELS)
        .chain(godrays::CHANNELS)
        .chain(edgeextend::CHANNELS)
        .chain(glint::CHANNELS)
        .chain(glintkernel::CHANNELS)
        .chain(bokehkernel::CHANNELS)
        .chain(diffuse::CHANNELS)
        .chain(softcrop::CHANNELS)
        .chain(motionblur::CHANNELS)
        .chain(gradient::CHANNELS)
        .chain(gridwarp::CHANNELS)
        .chain(warp::CHANNELS)
        .chain(blank::CHANNELS)
        .chain(prompt::CHANNELS)
        .chain(grain::CHANNELS)
        .chain(vectorblur::CHANNELS)
        .chain(scene3d::CHANNELS)
        .chain(vectorwarp::CHANNELS)
        .chain(rifetime::CHANNELS)
        .chain(freeze::CHANNELS)
        .chain(framefix::CHANNELS)
        .copied()
}

/// The curve editor's label for a channel a node module owns.
pub fn label(ch: crate::anim::Channel) -> Option<&'static str> {
    saturation::label(ch)
        .or_else(|| blur::label(ch))
        .or_else(|| colorcompress::label(ch))
        .or_else(|| colorslice::label(ch))
        .or_else(|| huerotate::label(ch))
        .or_else(|| lut::label(ch))
        .or_else(|| text::label(ch))
        .or_else(|| burnin::label(ch))
        .or_else(|| mirror::label(ch))
        .or_else(|| lensdistort::label(ch))
        .or_else(|| gamma::label(ch))
        .or_else(|| tonemap::label(ch))
        .or_else(|| despill::label(ch))
        .or_else(|| lightwrap::label(ch))
        .or_else(|| screenkey::label(ch))
        .or_else(|| dilate::label(ch))
        .or_else(|| edgedetect::label(ch))
        .or_else(|| median::label(ch))
        .or_else(|| highlightrebuild::label(ch))
        .or_else(|| ltxhdr::label(ch))
        .or_else(|| stabilize::label(ch))
        .or_else(|| convolve::label(ch))
        .or_else(|| glow::label(ch))
        .or_else(|| godrays::label(ch))
        .or_else(|| edgeextend::label(ch))
        .or_else(|| glint::label(ch))
        .or_else(|| glintkernel::label(ch))
        .or_else(|| bokehkernel::label(ch))
        .or_else(|| diffuse::label(ch))
        .or_else(|| softcrop::label(ch))
        .or_else(|| motionblur::label(ch))
        .or_else(|| gradient::label(ch))
        .or_else(|| gridwarp::label(ch))
        .or_else(|| warp::label(ch))
        .or_else(|| blank::label(ch))
        .or_else(|| prompt::label(ch))
        .or_else(|| grain::label(ch))
        .or_else(|| vectorblur::label(ch))
        .or_else(|| scene3d::label(ch))
        .or_else(|| vectorwarp::label(ch))
        .or_else(|| rifetime::label(ch))
        .or_else(|| freeze::label(ch))
}
