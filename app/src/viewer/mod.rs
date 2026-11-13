//! The viewer's overlays and gizmos, one file per mode: each an
//! `impl Gfx` block that `viewer_ui` (lib.rs) calls in order.
mod huds;
mod keyerhud;
mod sammie;
mod textpath;
mod camtrack;
mod planar;
mod tracker;
mod geoviz;
mod geohud;
mod geoonion;
mod geobrush;
mod geoedit;
mod geobones;
mod geotools;
mod geogradient;
mod geoxform;
mod paintcursor;
mod spline;
mod cornerpin;
mod pathedit;
mod transform;
mod hudchips;
mod hudsliders;
mod cryptopick;
mod eyedroppick;
