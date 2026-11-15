//! The long-running node jobs, one slot per kind of node that runs one: the ML nodes, the trackers and Blender.
//! These were loose fields on `Gfx`; grouped, not changed.

#[allow(unused_imports)] // a group of plain types needs nothing from the crate
use crate::*;

/// The long-running node jobs, one slot per kind of node that runs one: the ML nodes, the trackers and Blender.
pub(crate) struct NodeJobs {
    /// GenAI runs: capture, quote, the one job going, the account.
    pub(crate) genai: genaiui::Runs,
    pub(crate) corridor: corridorkey::Jobs,
    pub(crate) sammie: sammie::Jobs,
    pub(crate) flow: flowui::Jobs,
    /// A Stabilize node's analysis (`stabilizeui`).
    pub(crate) stab: stabilizeui::Jobs,
    /// A CameraTrack node's solve (`camtrackui`).
    pub(crate) camtrack: camtrackui::Jobs,
    pub(crate) upscale: upscaleui::UpscaleJobs,
    pub(crate) hdrcnn: hdrcnnui::HdrCnnJobs,
    pub(crate) seedvr: seedvr::Jobs,
    pub(crate) depth: depth::Jobs,
    pub(crate) ltxhdr: ltxhdr::Jobs,
    pub(crate) difflight: difflight::Jobs,
}
