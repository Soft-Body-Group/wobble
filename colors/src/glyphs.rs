//! Wobble's node icons, for the designer: the graph's node bodies copied
//! shape for shape — `editor::paint_bar` + `editor::paint_body`, and the
//! `glyph` fns in `app/src/nodes/` they call. Wobble's `glyph_parity` test
//! includes this file, paints every node both ways and fails on any
//! difference, naming the node: change a glyph there, copy it here.
//!
//! Nodes are drawn in their DEFAULT state (a Switch thrown to its first
//! input, an Extract with two pairs), which is what the test compares.

// Wobble's test build only reaches the painting.
#![allow(dead_code)]

use crate::theme::{self, Slot};
use egui::{Color32, Painter, Pos2, Rect, Stroke, Vec2};

/// A node's radius in graph units (the editor's `R`).
pub const R: f32 = 24.0;

// The defaults the settings-dependent icons are drawn at.
const SWITCH_INPUTS: usize = 2;
const SWITCH_LIVE: usize = 0;
const LAYER_INPUTS: usize = 2;
const LAYER_MAX: usize = 16;
const EXTRACT_PAIRS: usize = 2;
const EXTRACT_MAX: u32 = 32;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Icon {
    Read,
    Grid,
    Gradient,
    Blank,
    Starfield,
    Planet,
    Plasma,
    Particles,
    GlintKernel,
    BokehKernel,
    Prompt,
    Grain,
    Noise,
    Geo,
    Sdf,
    Blender,
    Paint,
    Grade,
    Mult,
    Curve,
    Saturation,
    Contrast,
    ColorCompress,
    HueRotate,
    Lut,
    Text,
    Burnin,
    Mirror,
    LensDistort,
    FlipFlop,
    Gamma,
    ToneMap,
    Expression,
    Despill,
    Lightwrap,
    Upscale,
    HdrCnn,
    SeedVr,
    Depth,
    LtxHdr,
    Tinter,
    ScreenKey,
    CorridorKey,
    Sammie,
    Blur,
    MotionBlur,
    Flow,
    Rifetime,
    Freeze,
    FrameFix,
    VectorBlur,
    Camera3d,
    Card3d,
    Scene3d,
    Render3d,
    VectorWarp,
    Defocus,
    Glow,
    Godrays,
    LensFlare,
    EdgeExtend,
    Glint,
    EdgeDetect,
    Median,
    HighlightRebuild,
    Stabilize,
    CameraTrack,
    Convolve,
    Diffuse,
    Dilate,
    Transform,
    Reformat,
    CornerPin,
    SoftCrop,
    UvWarp,
    Env,
    GridWarp,
    Refract,
    Warp,
    Spline,
    LumaKey,
    Keyer,
    Cryptomatte,
    MatteCombine,
    Mask,
    Extract,
    MatteIn,
    ToMatte,
    ToRgba,
    ToData,
    Premult,
    Unpremult,
    Breakout,
    Recombine,
    ChannelSelect,
    Shuffle,
    Blend,
    Mix,
    Layer,
    Switch,
    Slip,
    Speed,
    TimeRemap,
    RangeRemap,
    TimeRange,
    LoopIn,
    LoopOut,
    Bake,
    Tracker,
    PlanarTrack,
    Write,
    Shader,
    ShaderBroken,
    Plugin,
    SubwobInput,
    SubwobControl,
}

/// One node as the designer shows it: its icon, its name, the theme colour
/// it wears (`editor::type_color`; `glyph_parity` checks the pairing).
pub struct Node {
    pub icon: Icon,
    pub name: &'static str,
    pub slot: Slot,
}

const fn n(icon: Icon, name: &'static str, slot: Slot) -> Node {
    Node { icon, name, slot }
}

/// Every node the graph draws in a theme colour. Not here: Dots (they wear
/// their wire's colour) and installed plugins (their manifest's) — data,
/// not theme.
pub const NODES: &[Node] = &[
    n(Icon::Read, "Load", Slot::node_read),
    n(Icon::Grid, "Grid", Slot::node_grid),
    n(Icon::Gradient, "Gradient", Slot::node_gradient),
    n(Icon::Blank, "Blank", Slot::node_blank),
    n(Icon::Starfield, "Starfield", Slot::node_starfield),
    n(Icon::Planet, "Planet", Slot::node_planet),
    n(Icon::Plasma, "Plasma", Slot::node_plasma),
    n(Icon::Particles, "Particles", Slot::node_particles),
    n(Icon::GlintKernel, "Glint Kernel", Slot::node_glint_kernel),
    n(Icon::BokehKernel, "Bokeh Kernel", Slot::node_bokeh_kernel),
    n(Icon::Prompt, "Prompt", Slot::node_prompt),
    n(Icon::Grain, "Grain", Slot::node_grain),
    n(Icon::Noise, "Noise", Slot::node_noise),
    n(Icon::Geo, "2dini", Slot::node_geo),
    n(Icon::Sdf, "SDF", Slot::node_sdf),
    n(Icon::Blender, "Blender", Slot::node_blender),
    n(Icon::Paint, "Paint", Slot::node_paint),
    n(Icon::Grade, "Grade", Slot::node_grade),
    n(Icon::Mult, "Mult", Slot::node_mult),
    n(Icon::Curve, "Curve", Slot::node_curve),
    n(Icon::Saturation, "Saturation", Slot::node_saturation),
    n(Icon::Contrast, "Contrast", Slot::node_contrast),
    n(Icon::ColorCompress, "ColorCompress", Slot::node_color_compress),
    n(Icon::HueRotate, "HueRotate", Slot::node_hue_rotate),
    n(Icon::Lut, "LUT", Slot::node_lut),
    n(Icon::Text, "Text", Slot::node_text),
    n(Icon::Burnin, "Burnin", Slot::node_burnin),
    n(Icon::Mirror, "Mirror", Slot::node_mirror),
    n(Icon::LensDistort, "Lens Distortion", Slot::node_lens_distort),
    n(Icon::FlipFlop, "FlipFlop", Slot::node_flip_flop),
    n(Icon::Gamma, "Gamma", Slot::node_gamma),
    n(Icon::ToneMap, "Tone Map", Slot::node_tonemap),
    n(Icon::Expression, "Expression", Slot::node_expression),
    n(Icon::Despill, "Despill", Slot::node_despill),
    n(Icon::Lightwrap, "Lightwrap", Slot::node_lightwrap),
    n(Icon::Upscale, "Upscale", Slot::node_upscale),
    n(Icon::HdrCnn, "HDRCNN", Slot::node_hdrcnn),
    n(Icon::SeedVr, "SeedVR2", Slot::node_seedvr),
    n(Icon::Depth, "Depth", Slot::node_depth),
    n(Icon::LtxHdr, "LTX HDR", Slot::node_ltx_hdr),
    n(Icon::Tinter, "Tinter", Slot::node_tinter),
    n(Icon::Blur, "Blur", Slot::node_blur),
    n(Icon::MotionBlur, "MotionBlur", Slot::node_motion_blur),
    n(Icon::Flow, "Flow", Slot::node_flow),
    n(Icon::Rifetime, "Rifetime", Slot::node_rifetime),
    n(Icon::Freeze, "Freeze", Slot::node_freeze),
    n(Icon::FrameFix, "Frame Fix", Slot::node_framefix),
    n(Icon::VectorBlur, "VectorBlur", Slot::node_vector_blur),
    n(Icon::Camera3d, "Camera", Slot::node_camera3d),
    n(Icon::Card3d, "Card", Slot::node_card3d),
    n(Icon::Scene3d, "Blend 3D", Slot::node_scene3d),
    n(Icon::Render3d, "Render3D", Slot::node_render3d),
    n(Icon::Env, "Env", Slot::node_env),
    n(Icon::VectorWarp, "VectorWarp", Slot::node_vector_warp),
    n(Icon::Defocus, "Defocus", Slot::node_defocus),
    n(Icon::Glow, "Glow", Slot::node_glow),
    n(Icon::Godrays, "Godrays", Slot::node_godrays),
    n(Icon::LensFlare, "Lens Flare", Slot::node_lens_flare),
    n(Icon::Glint, "Glint", Slot::node_glint),
    n(Icon::EdgeDetect, "EdgeDetect", Slot::node_edge_detect),
    n(Icon::Median, "Median", Slot::node_median),
    n(Icon::HighlightRebuild, "HighlightRebuild", Slot::node_highlight_rebuild),
    n(Icon::Stabilize, "Stabilize", Slot::node_stabilize),
    n(Icon::CameraTrack, "CameraTrack", Slot::node_camera_track),
    n(Icon::Convolve, "Convolve", Slot::node_convolve),
    n(Icon::Diffuse, "Diffuse", Slot::node_diffuse),
    n(Icon::Dilate, "Dilate", Slot::node_dilate),
    n(Icon::EdgeExtend, "EdgeExtend", Slot::node_edge_extend),
    n(Icon::Transform, "Transform", Slot::node_transform),
    n(Icon::Reformat, "Reformat", Slot::node_reformat),
    n(Icon::CornerPin, "CornerPin", Slot::node_corner_pin),
    n(Icon::SoftCrop, "SoftCrop", Slot::node_soft_crop),
    n(Icon::UvWarp, "UV Warp", Slot::node_uvwarp),
    n(Icon::GridWarp, "GridWarp", Slot::node_grid_warp),
    n(Icon::Warp, "Warp", Slot::node_warp),
    n(Icon::Refract, "Refract", Slot::node_refract),
    n(Icon::Spline, "Spline", Slot::node_spline),
    n(Icon::LumaKey, "LumaKey", Slot::node_luma_key),
    n(Icon::Keyer, "Keyer", Slot::node_keyer),
    n(Icon::ScreenKey, "ScreenKey", Slot::node_screen_key),
    n(Icon::CorridorKey, "CorridorKey", Slot::node_corridor_key),
    n(Icon::Sammie, "Sammie", Slot::node_sammie),
    n(Icon::Cryptomatte, "Cryptomatte", Slot::node_cryptomatte),
    n(Icon::MatteCombine, "MatteCombine", Slot::node_matte_combine),
    n(Icon::Mask, "Mask", Slot::node_mask),
    n(Icon::Extract, "Extract", Slot::node_extract),
    n(Icon::MatteIn, "MatteIn", Slot::node_matte_in),
    n(Icon::ToMatte, "ToMatte", Slot::node_to_matte),
    n(Icon::ToRgba, "ToRgba", Slot::node_to_rgba),
    n(Icon::Premult, "Premult", Slot::node_premult),
    n(Icon::Unpremult, "Unpremult", Slot::node_premult),
    n(Icon::Breakout, "Breakout", Slot::node_breakout),
    n(Icon::Recombine, "Recombine", Slot::node_recombine),
    n(Icon::ChannelSelect, "ChannelSelect", Slot::node_channel_select),
    n(Icon::Shuffle, "Reorder", Slot::node_shuffle),
    n(Icon::Blend, "Blend", Slot::node_blend),
    n(Icon::Mix, "Mix", Slot::node_mix),
    n(Icon::Layer, "Layer", Slot::node_layer),
    n(Icon::Switch, "Switch", Slot::node_switch),
    n(Icon::Slip, "Slip", Slot::node_time),
    n(Icon::Speed, "Speed", Slot::node_time),
    n(Icon::TimeRemap, "Time Curve", Slot::node_time),
    n(Icon::RangeRemap, "Time Remap", Slot::node_time),
    n(Icon::TimeRange, "Time Range", Slot::node_time),
    n(Icon::LoopIn, "LoopIn", Slot::node_loop),
    n(Icon::LoopOut, "LoopOut", Slot::node_loop),
    n(Icon::Bake, "Bake", Slot::node_bake),
    n(Icon::Tracker, "Tracker", Slot::node_tracker),
    n(Icon::PlanarTrack, "PlanarTrack", Slot::node_planar),
    n(Icon::Write, "Render", Slot::node_write),
    n(Icon::Shader, "Shader", Slot::node_shader),
    n(Icon::ShaderBroken, "Shader, not compiling", Slot::node_shader_broken),
    n(Icon::Plugin, "plugin, missing", Slot::node_plugin_missing),
    n(Icon::SubwobInput, "subwob input", Slot::node_tool_input),
    n(Icon::SubwobControl, "subwob control", Slot::node_tool_control),
    n(Icon::ToData, "ToData", Slot::node_to_data),
];

/// Paint `icon` centred on `sp` at `scale` screen points per graph unit (a
/// node is `R` graph units in radius). Returns the body's extent, for hits
/// and for where the name goes.
pub fn paint(painter: &Painter, icon: Icon, sp: Pos2, scale: f32, fill: Color32, stroke: Stroke) -> Rect {
    let mut r = R * scale;
    // `editor::paint_bar`: a merge bar's slab, the glyph shrunk into it.
    if let Some((h, dx)) = bar(icon) {
        let half = h * scale;
        let c = sp + Vec2::new(dx * scale, 0.0);
        painter.rect(Rect::from_center_size(c, half * 2.0), half.y, fill, stroke, egui::StrokeKind::Middle);
        r = half.y * 0.8;
    }
    body(painter, icon, sp, r, scale, fill, stroke);
    extent(icon, sp, scale)
}

/// Where `paint` would draw: the slab of a wide node, a node's square
/// otherwise.
pub fn extent(icon: Icon, sp: Pos2, scale: f32) -> Rect {
    let wide = match icon {
        Icon::Breakout | Icon::Recombine => Some(harp(icon)),
        _ => bar(icon),
    };
    match wide {
        Some((h, dx)) => Rect::from_center_size(sp + Vec2::new(dx * scale, 0.0), h * 2.0 * scale),
        None => Rect::from_center_size(sp, Vec2::splat(2.0 * R * scale)),
    }
}

/// `breakout::wide_rect` of a default merge bar: (half extent, centre offset),
/// graph units.
fn bar(icon: Icon) -> Option<(Vec2, f32)> {
    match icon {
        Icon::Layer => Some((Vec2::new(40.0, 13.0), 0.0)),
        Icon::Extract => Some((Vec2::new(57.0, 13.0), -17.0)),
        _ => None,
    }
}

/// `breakout::wide_rect` of a default harp half.
fn harp(icon: Icon) -> (Vec2, f32) {
    match icon {
        Icon::Recombine => (Vec2::new(99.0, 13.0), 0.0),
        _ => (Vec2::new(37.0, 13.0), -17.0),
    }
}

/// `editor::paint_body`, arm for arm.
fn body(painter: &Painter, icon: Icon, sp: Pos2, r: f32, scale: f32, fill: Color32, stroke: Stroke) {
    match icon {
        // The harp pair: wide slabs, their width the strings' span —
        // the node IS its comb, so how much it covers reads at a
        // glance. The receiver is the same slab a shade darker,
        // hanging under it.
        Icon::Breakout | Icon::Recombine => {
            let (h, dx) = harp(icon);
            let half = h * scale;
            let c = sp + Vec2::new(dx * scale, 0.0);
            let body = Rect::from_center_size(c, half * 2.0);
            painter.rect(body, half.y, fill, stroke, egui::StrokeKind::Middle);
            // The breakout's + : add an AOV straight off the bar —
            // it opens the upstream layer list without a trip to
            // the properties rack.
            if icon == Icon::Breakout && half.y > 5.0 {
                let pc = Pos2::new(body.max.x - half.y * 1.1, c.y);
                let pr = (half.y * 0.5).clamp(4.0, 9.0);
                let st = Stroke::new(
                    (pr * 0.3).max(1.0),
                    Color32::from_gray(0x14).gamma_multiply(0.9),
                );
                painter.line_segment(
                    [pc - Vec2::new(pr, 0.0), pc + Vec2::new(pr, 0.0)],
                    st,
                );
                painter.line_segment(
                    [pc - Vec2::new(0.0, pr), pc + Vec2::new(0.0, pr)],
                    st,
                );
            }
        }
        Icon::Switch => {
            // A points-lever: the body plus a stub aimed at the live
            // input, so which way it is thrown reads at a glance.
            painter.circle(sp, r * 0.8, fill, stroke);
            let n_in = SWITCH_INPUTS.max(1) as f32;
            // Fan the inputs across the top, same order they splay.
            let t = if n_in > 1.0 {
                SWITCH_LIVE as f32 / (n_in - 1.0)
            } else {
                0.5
            };
            // Inputs splay across the TOP, so the lever sweeps the
            // upper half: 3/4 pi (up-left, input 0) to 1/4 pi.
            let a = std::f32::consts::PI * (0.75 - 0.5 * t);
            painter.line_segment(
                [sp, sp + Vec2::new(a.cos(), -a.sin()) * r * 0.95],
                Stroke::new((r * 0.22).max(1.2), theme::node_ink_light()),
            );
            // A stud per input on the rim: the positions the lever can be
            // thrown to.
            let n = SWITCH_INPUTS;
            for i in 0..n {
                let t = if n > 1 { i as f32 / (n - 1) as f32 } else { 0.5 };
                let a = std::f32::consts::PI * (0.75 - 0.5 * t);
                let ink = theme::node_ink_light();
                let ink = if i == SWITCH_LIVE { ink } else { ink.gamma_multiply(0.35) };
                painter.circle_filled(
                    sp + Vec2::new(a.cos(), -a.sin()) * r * 0.8,
                    (r * 0.11).max(1.0),
                    ink,
                );
            }
        }
        Icon::Read => {
            // Triangle pointing down: a source funneling into the comp.
            let pts = vec![
                Pos2::new(sp.x - r, sp.y - r * 0.7),
                Pos2::new(sp.x + r, sp.y - r * 0.7),
                Pos2::new(sp.x, sp.y + r),
            ];
            painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
        }
        Icon::Bake => {
            // Disk platter: a cached point in the graph.
            let rr = r * 0.85;
            painter.rect(
                Rect::from_center_size(sp, Vec2::new(rr * 2.0, rr * 1.6)),
                rr * 0.35,
                fill,
                stroke,
                egui::StrokeKind::Outside,
            );
            let st = Stroke::new((r * 0.16).max(1.0), theme::node_ink());
            painter.line_segment(
                [
                    Pos2::new(sp.x - rr * 0.6, sp.y),
                    Pos2::new(sp.x + rr * 0.6, sp.y),
                ],
                st,
            );
        }
        Icon::Shader | Icon::ShaderBroken => {
            // Angle brackets: source code, and distinct from the
            // plugin hexagon because the source lives IN the setup.
            let w = r * 0.75;
            for s in [-1.0f32, 1.0] {
                painter.add(egui::Shape::line(
                    vec![
                        Pos2::new(sp.x + s * w, sp.y - r * 0.8),
                        Pos2::new(sp.x + s * r * 0.25, sp.y),
                        Pos2::new(sp.x + s * w, sp.y + r * 0.8),
                    ],
                    stroke,
                ));
            }
        }
        Icon::Plugin | Icon::SubwobInput | Icon::SubwobControl => {
            // Hexagon: the plugin shape.
            let pts = (0..6)
                .map(|k| {
                    let a = k as f32 / 6.0 * std::f32::consts::TAU
                        + std::f32::consts::FRAC_PI_6;
                    Pos2::new(sp.x + a.cos() * r, sp.y + a.sin() * r)
                })
                .collect();
            painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
        }
        Icon::Blend => {
            // Triangle pointing up: two streams funnel to one.
            let pts = vec![
                Pos2::new(sp.x, sp.y - r),
                Pos2::new(sp.x + r, sp.y + r * 0.7),
                Pos2::new(sp.x - r, sp.y + r * 0.7),
            ];
            painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
        }
        Icon::Geo => {
            // A filled body, then a pentagon outline: geometry, not
            // a picture and not a field.
            painter.rect(
                Rect::from_center_size(sp, Vec2::splat(r * 1.7)),
                r * 0.3,
                fill,
                stroke,
                egui::StrokeKind::Middle,
            );
            let pts: Vec<Pos2> = (0..5)
                .map(|i| {
                    let a = -std::f32::consts::FRAC_PI_2
                        + i as f32 * std::f32::consts::TAU / 5.0;
                    Pos2::new(sp.x + a.cos() * r * 0.6, sp.y + a.sin() * r * 0.6)
                })
                .collect();
            painter.add(egui::Shape::closed_line(pts, stroke));
        }
        Icon::Noise => {
            // A filled body FIRST: the waves below are drawn in the
            // outline stroke, which is near-black until the node is
            // selected, so on their own they are invisible.
            painter.rect(
                Rect::from_center_size(sp, Vec2::splat(r * 1.7)),
                r * 0.3,
                fill,
                stroke,
                egui::StrokeKind::Middle,
            );
            // Three stacked waves: a field, not a picture.
            for k in 0..3i32 {
                let y = sp.y + (k - 1) as f32 * r * 0.55;
                let mut pts = Vec::new();
                for i in 0..=8 {
                    let t = i as f32 / 8.0;
                    pts.push(Pos2::new(
                        sp.x - r + t * r * 2.0,
                        y + (t * std::f32::consts::TAU + k as f32).sin() * r * 0.18,
                    ));
                }
                painter.add(egui::Shape::line(pts, stroke));
            }
        }
        Icon::MatteCombine => {
            // Two overlapping discs: the Venn diagram every one of
            // its ops is a region of.
            painter.circle(Pos2::new(sp.x - r * 0.36, sp.y), r * 0.72, fill, stroke);
            painter.circle_stroke(Pos2::new(sp.x + r * 0.36, sp.y), r * 0.72, stroke);
        }
        Icon::Dilate => dilate(painter, sp, r, fill, stroke),
        Icon::EdgeExtend => edgeextend(painter, sp, r, fill, stroke),
        Icon::Sdf => sdf(painter, sp, r, fill, stroke),
        Icon::Layer => {
            // Three offset sheets: a stack seen slightly from above.
            for k in [1.0f32, 0.0, -1.0] {
                let c = Pos2::new(sp.x + k * r * 0.28, sp.y + k * r * 0.34);
                painter.rect(
                    Rect::from_center_size(c, Vec2::new(r * 1.35, r * 0.78)),
                    1.0,
                    fill,
                    stroke,
                    egui::StrokeKind::Outside,
                );
            }
            // No +: a noodle dropped on the bar is a new layer.
        }
        Icon::Mix => {
            // Two triangles meeting nose to nose: the crossfade.
            for s in [-1.0f32, 1.0] {
                painter.add(egui::Shape::convex_polygon(
                    vec![
                        Pos2::new(sp.x, sp.y),
                        Pos2::new(sp.x + r * s, sp.y - r * 0.8),
                        Pos2::new(sp.x + r * s, sp.y + r * 0.8),
                    ],
                    fill,
                    stroke,
                ));
            }
        }
        Icon::Transform => {
            let half = r * 0.82;
            painter.rect(
                Rect::from_center_size(sp, Vec2::splat(half * 2.0)),
                2.0,
                fill,
                stroke,
                egui::StrokeKind::Outside,
            );
        }
        Icon::Blur => blur(painter, sp, r, fill, stroke),
        Icon::Grade => {
            painter.circle(sp, r * 0.9, fill, stroke);
        }
        Icon::Defocus => {
            // Hexagon: a bokeh iris.
            let pts: Vec<Pos2> = (0..6)
                .map(|k| {
                    let a = k as f32 * std::f32::consts::TAU / 6.0
                        - std::f32::consts::TAU / 12.0;
                    Pos2::new(sp.x + a.cos() * r, sp.y + a.sin() * r)
                })
                .collect();
            painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
        }
        Icon::Glow => glow(painter, sp, r, fill, stroke),
        Icon::Godrays => godrays(painter, sp, r, fill, stroke),
        Icon::LensFlare => lensflare(painter, sp, r, fill, stroke),
        Icon::Glint => glint(painter, sp, r, fill, stroke),
        Icon::EdgeDetect => edgedetect(painter, sp, r, fill, stroke),
        Icon::Median => median(painter, sp, r, fill, stroke),
        Icon::HighlightRebuild => highlightrebuild(painter, sp, r, fill, stroke),
        Icon::Stabilize => stabilize(painter, sp, r, fill, stroke),
        Icon::CameraTrack => camera_track(painter, sp, r, fill, stroke),
        Icon::Convolve => convolve(painter, sp, r, fill, stroke),
        Icon::Diffuse => diffuse(painter, sp, r, fill, stroke),
        Icon::CornerPin => {
            // Trapezoid: a plane in perspective.
            let pts = vec![
                Pos2::new(sp.x - r * 0.55, sp.y - r * 0.8),
                Pos2::new(sp.x + r * 0.55, sp.y - r * 0.8),
                Pos2::new(sp.x + r, sp.y + r * 0.8),
                Pos2::new(sp.x - r, sp.y + r * 0.8),
            ];
            painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
        }
        Icon::Spline => {
            // Ring: a matte shape.
            painter.circle(sp, r * 0.85, Color32::TRANSPARENT, stroke);
            painter.circle_stroke(
                sp,
                r * 0.55,
                Stroke::new((r * 0.3).max(1.5), fill),
            );
        }
        Icon::Paint => {
            // Brush tip: kite pointing down.
            let pts = vec![
                Pos2::new(sp.x, sp.y + r),
                Pos2::new(sp.x - r * 0.7, sp.y + r * 0.1),
                Pos2::new(sp.x, sp.y - r),
                Pos2::new(sp.x + r * 0.7, sp.y + r * 0.1),
            ];
            painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
        }
        Icon::Curve => {
            // Rounded square with an S-curve through it.
            painter.rect(
                Rect::from_center_size(sp, Vec2::splat(r * 1.6)),
                r * 0.3,
                fill,
                stroke,
                egui::StrokeKind::Outside,
            );
            painter.add(egui::epaint::CubicBezierShape::from_points_stroke(
                [
                    Pos2::new(sp.x - r * 0.6, sp.y + r * 0.5),
                    Pos2::new(sp.x + r * 0.5, sp.y + r * 0.5),
                    Pos2::new(sp.x - r * 0.5, sp.y - r * 0.5),
                    Pos2::new(sp.x + r * 0.6, sp.y - r * 0.5),
                ],
                false,
                Color32::TRANSPARENT,
                Stroke::new((r * 0.18).max(1.0), theme::node_ink()),
            ));
        }
        Icon::Extract => {
            // A Mask glyph sliced into its outputs: one wedge of
            // the lower half per pair, gaps between them.
            painter.circle_stroke(sp, r * 0.85, Stroke::new(2.0, fill));
            let k = EXTRACT_PAIRS;
            let gap = 0.12 / k as f32;
            for j in 0..k {
                let a0 = std::f32::consts::PI * (j as f32 / k as f32 + gap);
                let a1 = std::f32::consts::PI * ((j + 1) as f32 / k as f32 - gap);
                let mut pts = vec![sp];
                pts.extend((0..=6).map(|s| {
                    let a = a0 + (a1 - a0) * s as f32 / 6.0;
                    Pos2::new(sp.x - a.cos() * r * 0.85, sp.y + a.sin() * r * 0.85)
                }));
                painter.add(egui::Shape::convex_polygon(pts, fill, Stroke::NONE));
            }
            // The +, off the right flank: one more pair, no trip
            // to the properties rack (the Breakout's gesture).
            if r > 5.0 && (EXTRACT_PAIRS as u32) < EXTRACT_MAX {
                // On the bar, at its right end.
                let pc = match bar(icon) {
                    Some((h, dx)) => sp + Vec2::new((dx + h.x) * scale - r * 1.1, 0.0),
                    None => sp + Vec2::new(r * 1.45, 0.0),
                };
                let pr = (r * 0.28).clamp(3.5, 8.0);
                let st = Stroke::new((pr * 0.3).max(1.0), theme::node_ink_light());
                painter.line_segment([pc - Vec2::new(pr, 0.0), pc + Vec2::new(pr, 0.0)], st);
                painter.line_segment([pc - Vec2::new(0.0, pr), pc + Vec2::new(0.0, pr)], st);
            }
        }
        Icon::Mask => {
            // Circle, lower half solid: image cut by coverage.
            painter.circle_stroke(sp, r * 0.85, Stroke::new(2.0, fill));
            let pts: Vec<Pos2> = (0..=12)
                .map(|k| {
                    let a = std::f32::consts::PI * k as f32 / 12.0;
                    Pos2::new(sp.x - a.cos() * r * 0.85, sp.y + a.sin() * r * 0.85)
                })
                .collect();
            painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
        }
        Icon::LumaKey => {
            // Crescent: bright circle bitten by the background.
            painter.circle_filled(sp, r * 0.85, fill);
            painter.circle_filled(
                Pos2::new(sp.x + r * 0.45, sp.y - r * 0.25),
                r * 0.6,
                theme::graph_bg(),
            );
        }
        Icon::Keyer => {
            // The node itself, drawn: a selection blob sitting off
            // centre inside the colour circle.
            painter.circle_stroke(sp, r * 0.85, Stroke::new((r * 0.16).max(1.0), fill));
            let c = Pos2::new(sp.x + r * 0.2, sp.y - r * 0.16);
            let pts: Vec<Pos2> = (0..14)
                .map(|k| {
                    let a = std::f32::consts::TAU * k as f32 / 14.0 + 0.6;
                    Pos2::new(c.x + a.cos() * r * 0.46, c.y + a.sin() * r * 0.26)
                })
                .collect();
            painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
        }
        Icon::Cryptomatte => {
            // An id map: a row of tiles, one of them picked (filled).
            let s = r * 0.5;
            for (k, dx) in [-1.1f32, 0.0, 1.1].into_iter().enumerate() {
                let c = Pos2::new(sp.x + dx * s, sp.y);
                let rr = Rect::from_center_size(c, Vec2::splat(s * 0.95));
                if k == 1 {
                    painter.rect_filled(rr, 1.0, fill);
                } else {
                    painter.rect_stroke(rr, 1.0, stroke, egui::StrokeKind::Outside);
                }
            }
        }
        Icon::MatteIn => {
            // ToMatte's mirror: the mask colour enters from the left
            // and the square comes out an image.
            let h = r * 0.8;
            painter.rect(
                Rect::from_center_size(sp, Vec2::splat(h * 2.0)),
                1.0,
                fill,
                stroke,
                egui::StrokeKind::Outside,
            );
            let pts = vec![
                Pos2::new(sp.x - h, sp.y - h),
                Pos2::new(sp.x, sp.y - h),
                Pos2::new(sp.x, sp.y + h),
                Pos2::new(sp.x - h, sp.y + h),
            ];
            painter.add(egui::Shape::convex_polygon(
                pts,
                theme::wire_mask().gamma_multiply(0.9),
                Stroke::NONE,
            ));
        }
        Icon::UvWarp => uvwarp(painter, sp, r, fill, stroke),
        Icon::Env => env(painter, sp, r, fill, stroke),
        Icon::GridWarp => gridwarp(painter, sp, r, fill, stroke),
        Icon::Warp => warp(painter, sp, r, fill, stroke),
        Icon::Refract => refract(painter, sp, r, fill, stroke),
        Icon::SoftCrop => {
            softcrop(painter, sp, r, fill, stroke)
        }
        Icon::MotionBlur => {
            motionblur(painter, sp, r, fill, stroke)
        }
        Icon::Flow => flow(painter, sp, r, fill, stroke),
        Icon::Rifetime => retime(painter, sp, r, fill, stroke),
        Icon::Freeze => freeze(painter, sp, r, fill, stroke),
        Icon::FrameFix => framefix(painter, sp, r, fill, stroke),
        Icon::VectorBlur => vectorblur(painter, sp, r, fill, stroke),
        Icon::Camera3d => camera3d(painter, sp, r, fill, stroke),
        Icon::Card3d => card3d(painter, sp, r, fill, stroke),
        Icon::Scene3d => scene3d(painter, sp, r, fill, stroke),
        Icon::Render3d => render3d(painter, sp, r, fill, stroke),
        Icon::VectorWarp => vectorwarp(painter, sp, r, fill, stroke),
        Icon::ColorCompress => {
            colorcompress(painter, sp, r, fill, stroke)
        }
        Icon::HueRotate => {
            huerotate(painter, sp, r, fill, stroke)
        }
        Icon::Lut => lut(painter, sp, r, fill, stroke),
        Icon::Text => text(painter, sp, r, fill, stroke),
        Icon::Burnin => burnin(painter, sp, r, fill, stroke),
        Icon::Mirror => mirror(painter, sp, r, fill, stroke),
        Icon::LensDistort => lensdistort(painter, sp, r, fill, stroke),
        Icon::FlipFlop => flipflop(painter, sp, r, fill, stroke),
        Icon::Gamma => gamma(painter, sp, r, fill, stroke),
        Icon::ToneMap => tonemap(painter, sp, r, fill, stroke),
        Icon::Expression => {
            expression(painter, sp, r, fill, stroke)
        }
        Icon::Despill => despill(painter, sp, r, fill, stroke),
        Icon::Lightwrap => lightwrap(painter, sp, r, fill, stroke),
        Icon::Upscale => upscale(painter, sp, r, fill, stroke),
        Icon::HdrCnn => hdrcnn(painter, sp, r, fill, stroke),
        Icon::SeedVr => seedvr(painter, sp, r, fill, stroke),
        Icon::Depth => depth(painter, sp, r, fill, stroke),
        Icon::LtxHdr => ltxhdr(painter, sp, r, fill, stroke),
        Icon::ScreenKey => {
            screenkey(painter, sp, r, fill, stroke)
        }
        Icon::CorridorKey => {
            corridorkey(painter, sp, r, fill, stroke)
        }
        Icon::Sammie => sammie(painter, sp, r, fill, stroke),
        Icon::Tinter => {
            colorslice(painter, sp, r, fill, stroke)
        }
        Icon::ToData => {
            todata(painter, sp, r, fill, stroke)
        }
        Icon::ToRgba => {
            torgba(painter, sp, r, fill, stroke)
        }
        Icon::ToMatte => {
            // Half image, half matte: a square whose right side is
            // cut away into the mask colour it outputs.
            let h = r * 0.8;
            painter.rect(
                Rect::from_center_size(sp, Vec2::splat(h * 2.0)),
                1.0,
                fill,
                stroke,
                egui::StrokeKind::Outside,
            );
            let pts = vec![
                Pos2::new(sp.x, sp.y - h),
                Pos2::new(sp.x + h, sp.y - h),
                Pos2::new(sp.x + h, sp.y + h),
                Pos2::new(sp.x, sp.y + h),
            ];
            painter.add(egui::Shape::convex_polygon(
                pts,
                theme::wire_mask().gamma_multiply(0.9),
                Stroke::NONE,
            ));
        }
        Icon::Mult => {
            // A circle with a multiplication cross through it: the
            // Grade's disc, pared back to one operation.
            painter.circle(sp, r * 0.8, fill, stroke);
            let st = Stroke::new((r * 0.16).max(1.0), theme::node_ink());
            let d = r * 0.38;
            painter.line_segment(
                [Pos2::new(sp.x - d, sp.y - d), Pos2::new(sp.x + d, sp.y + d)],
                st,
            );
            painter.line_segment(
                [Pos2::new(sp.x - d, sp.y + d), Pos2::new(sp.x + d, sp.y - d)],
                st,
            );
        }
        Icon::Gradient => gradient(painter, sp, r, fill, stroke),
        Icon::Blank => blank(painter, sp, r, fill, stroke),
        Icon::Starfield => starfield(painter, sp, r, fill, stroke),
        Icon::Planet => planet(painter, sp, r, fill, stroke),
        Icon::Plasma => plasma(painter, sp, r, fill, stroke),
        Icon::Particles => particles(painter, sp, r, fill, stroke),
        Icon::GlintKernel => glint_kernel(painter, sp, r, fill, stroke),
        Icon::BokehKernel => bokeh_kernel(painter, sp, r, fill, stroke),
        Icon::Prompt => prompt(painter, sp, r, fill, stroke),
        Icon::Grain => grain(painter, sp, r, fill, stroke),
        Icon::Blender => blender(painter, sp, r, fill, stroke),
        Icon::Grid => {
            // Source triangle ruled into cells: a generated plate.
            let pts = vec![
                Pos2::new(sp.x - r, sp.y - r * 0.7),
                Pos2::new(sp.x + r, sp.y - r * 0.7),
                Pos2::new(sp.x, sp.y + r),
            ];
            painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
            let st = Stroke::new((r * 0.12).max(0.8), theme::node_ink());
            painter.line_segment(
                [
                    Pos2::new(sp.x - r * 0.45, sp.y - r * 0.2),
                    Pos2::new(sp.x + r * 0.45, sp.y - r * 0.2),
                ],
                st,
            );
            painter.line_segment(
                [
                    Pos2::new(sp.x, sp.y - r * 0.6),
                    Pos2::new(sp.x, sp.y + r * 0.5),
                ],
                st,
            );
        }
        Icon::Reformat => {
            // Small frame nested in a big one: a change of format.
            painter.rect(
                Rect::from_center_size(sp, egui::vec2(r * 1.9, r * 1.3)),
                1.0,
                fill,
                stroke,
                egui::StrokeKind::Outside,
            );
            painter.rect_stroke(
                Rect::from_center_size(sp, egui::vec2(r * 1.0, r * 0.7)),
                0.0,
                Stroke::new((r * 0.14).max(1.0), theme::node_ink()),
                egui::StrokeKind::Inside,
            );
        }
        Icon::LoopIn => {
            // A tap: the ring something loops back INTO.
            painter.circle_stroke(sp, r * 0.8, stroke);
            painter.circle(sp, r * 0.28, fill, stroke);
        }
        Icon::LoopOut => {
            // A cycle: three arc dashes chasing each other.
            for k in 0..3 {
                let a0 = k as f32 * std::f32::consts::TAU / 3.0;
                let pts: Vec<Pos2> = (0..8)
                    .map(|j| {
                        let a = a0 + j as f32 * 0.09;
                        Pos2::new(
                            sp.x + a.cos() * r * 0.8,
                            sp.y + a.sin() * r * 0.8,
                        )
                    })
                    .collect();
                painter.add(egui::Shape::line(pts, stroke));
            }
            painter.circle(sp, r * 0.28, fill, stroke);
        }
        Icon::ChannelSelect => {
            // Many strands in, one out: the fat cable narrowing.
            for k in [-1.0f32, 0.0, 1.0] {
                painter.line_segment(
                    [
                        Pos2::new(sp.x + k * r * 0.55, sp.y - r * 0.9),
                        Pos2::new(sp.x, sp.y + r * 0.1),
                    ],
                    stroke,
                );
            }
            painter.line_segment(
                [Pos2::new(sp.x, sp.y + r * 0.1), Pos2::new(sp.x, sp.y + r * 0.9)],
                stroke,
            );
            painter.circle(sp, r * 0.28, fill, stroke);
        }
        Icon::PlanarTrack => {
            // A quad in perspective: the plane being followed.
            let st = Stroke::new((r * 0.18).max(1.4), fill);
            let q = [
                Pos2::new(sp.x - r * 0.9, sp.y - r * 0.5),
                Pos2::new(sp.x + r * 0.9, sp.y - r * 0.8),
                Pos2::new(sp.x + r * 0.6, sp.y + r * 0.8),
                Pos2::new(sp.x - r * 0.6, sp.y + r * 0.5),
            ];
            for k in 0..4 {
                painter.line_segment([q[k], q[(k + 1) % 4]], st);
            }
            painter.line_segment([q[0], q[2]], Stroke::new((r * 0.1).max(1.0), fill));
        }
        Icon::Tracker => {
            // Crosshair in a box: the pattern being followed.
            painter.rect_stroke(
                Rect::from_center_size(sp, Vec2::splat(r * 1.6)),
                0.0,
                Stroke::new((r * 0.2).max(1.5), fill),
                egui::StrokeKind::Inside,
            );
            let st = Stroke::new((r * 0.14).max(1.0), fill);
            painter.line_segment(
                [Pos2::new(sp.x - r, sp.y), Pos2::new(sp.x + r, sp.y)],
                st,
            );
            painter.line_segment(
                [Pos2::new(sp.x, sp.y - r), Pos2::new(sp.x, sp.y + r)],
                st,
            );
        }
        Icon::Saturation => saturation(painter, sp, r, fill, stroke),
        Icon::Contrast => contrast(painter, sp, r, fill, stroke),
        Icon::Slip | Icon::Speed | Icon::TimeRemap | Icon::RangeRemap | Icon::TimeRange => {
            // Clock face; a retime's hand is swept off the hour.
            painter.circle(sp, r * 0.85, fill, stroke);
            let st = Stroke::new((r * 0.16).max(1.0), theme::node_ink());
            painter.line_segment([sp, Pos2::new(sp.x, sp.y - r * 0.6)], st);
            let arm = if icon != Icon::Slip {
                Pos2::new(sp.x + r * 0.55, sp.y + r * 0.3)
            } else {
                Pos2::new(sp.x + r * 0.5, sp.y)
            };
            painter.line_segment([sp, arm], st);
        }
        Icon::Write => {
            // A down-triangle like a Read, inverted into a funnel
            // with a bar under it: the comp going to disk.
            let pts = vec![
                Pos2::new(sp.x - r, sp.y - r * 0.9),
                Pos2::new(sp.x + r, sp.y - r * 0.9),
                Pos2::new(sp.x, sp.y + r * 0.35),
            ];
            painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
            painter.rect_filled(
                Rect::from_center_size(
                    Pos2::new(sp.x, sp.y + r * 0.75),
                    egui::vec2(r * 1.6, r * 0.36),
                ),
                1.0,
                fill,
            );
        }
        Icon::Shuffle => {
            // Four stacked bars with two of them crossing: channels
            // going in one order and coming out another.
            painter.rect(
                Rect::from_center_size(sp, Vec2::splat(r * 1.6)),
                1.0,
                fill,
                stroke,
                egui::StrokeKind::Outside,
            );
            let st = Stroke::new((r * 0.13).max(0.8), theme::node_ink());
            let (x0, x1) = (sp.x - r * 0.55, sp.x + r * 0.55);
            for (k, dy) in [-0.45f32, 0.45].iter().enumerate() {
                let y = sp.y + dy * r;
                painter.line_segment(
                    [Pos2::new(x0, y), Pos2::new(x1, y)],
                    st,
                );
                let _ = k;
            }
            // The crossing pair.
            painter.line_segment(
                [Pos2::new(x0, sp.y - r * 0.15), Pos2::new(x1, sp.y + r * 0.15)],
                st,
            );
            painter.line_segment(
                [Pos2::new(x0, sp.y + r * 0.15), Pos2::new(x1, sp.y - r * 0.15)],
                st,
            );
        }
        Icon::Premult | Icon::Unpremult => {
            // Circle over a square: colour against its alpha. The
            // premult direction fills the disc, unpremult hollows it.
            painter.rect(
                Rect::from_center_size(sp, Vec2::splat(r * 1.5)),
                1.0,
                fill,
                stroke,
                egui::StrokeKind::Outside,
            );
            let ink = theme::node_ink();
            if icon == Icon::Premult {
                painter.circle_filled(sp, r * 0.45, ink);
            } else {
                painter.circle_stroke(
                    sp,
                    r * 0.45,
                    Stroke::new((r * 0.16).max(1.0), ink),
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The glyphs that live in the nodes' own modules.

/// `app/src/nodes/dilate.rs` `glyph`.
/// `app/src/sdfui.rs` `glyph`.
fn sdf(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.rect(
        Rect::from_center_size(c, Vec2::splat(r * 1.7)),
        r * 0.3,
        fill,
        stroke,
        egui::StrokeKind::Middle,
    );
    painter.circle(c, r * 0.2, stroke.color, Stroke::NONE);
    painter.circle_stroke(c, r * 0.42, stroke);
    painter.circle_stroke(c, r * 0.64, stroke);
}

fn dilate(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.5, fill, stroke);
    painter.circle_stroke(c, r * 0.95, stroke);
}

/// `app/src/nodes/edgeextend.rs` `glyph`.
fn edgeextend(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.45, fill, stroke);
    painter.circle_stroke(c, r * 0.7, Stroke::new(stroke.width, fill.gamma_multiply(0.7)));
    painter.circle_stroke(c, r * 0.95, Stroke::new(stroke.width, fill.gamma_multiply(0.35)));
}

/// `app/src/nodes/convolve.rs` `glyph`.
fn convolve(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let s = r * 0.55;
    for y in -1..=1 {
        for x in -1..=1 {
            let q = Pos2::new(c.x + x as f32 * s, c.y + y as f32 * s);
            if x == 0 && y == 0 {
                painter.circle(q, r * 0.22, fill, stroke);
            } else {
                painter.circle_stroke(q, r * 0.14, stroke);
            }
        }
    }
}

/// `app/src/nodes/median.rs` `glyph`.
fn median(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let outer = Rect::from_center_size(c, Vec2::splat(r * 1.6));
    painter.rect(outer, 0.0, fill, stroke, egui::StrokeKind::Middle);
    let s = r * 0.42;
    for (k, h) in [(-1.0f32, 0.45f32), (0.0, 0.8), (1.0, 0.45)] {
        let x = c.x + k * s;
        painter.line_segment([Pos2::new(x, c.y + r * 0.5), Pos2::new(x, c.y + r * 0.5 - h * r)], stroke);
    }
}

/// `app/src/nodes/highlightrebuild.rs` `glyph`.
fn highlightrebuild(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let rr = r * 0.62;
    painter.circle(c + Vec2::new(0.0, r * 0.18), rr, fill, stroke);
    let ink = theme::node_ink();
    let cut = c.y - r * 0.12;
    let half = r * 0.75;
    painter.line_segment([Pos2::new(c.x - half, cut), Pos2::new(c.x + half, cut)], Stroke::new((r * 0.12).max(1.0), ink));
    for k in [-1.0f32, 0.0, 1.0] {
        let x = c.x + k * r * 0.42;
        painter.line_segment([Pos2::new(x, cut - r * 0.22), Pos2::new(x + k * r * 0.12, cut - r * 0.58)], stroke);
    }
}

/// `app/src/nodes/camtrack.rs` `glyph`.
fn camera_track(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let body = egui::Rect::from_center_size(c + Vec2::new(-r * 0.35, r * 0.25), Vec2::new(r * 0.8, r * 0.6));
    painter.rect(body, r * 0.1, fill, stroke, egui::StrokeKind::Middle);
    let lens = vec![
        Pos2::new(body.right(), body.center().y - r * 0.14),
        Pos2::new(body.right() + r * 0.35, body.center().y - r * 0.32),
        Pos2::new(body.right() + r * 0.35, body.center().y + r * 0.32),
        Pos2::new(body.right(), body.center().y + r * 0.14),
    ];
    painter.add(egui::Shape::convex_polygon(lens, fill, stroke));
    let ink = theme::node_ink();
    for (dx, dy) in [(0.15, -0.75), (0.55, -0.55), (0.8, -0.2), (0.35, -0.35), (0.75, -0.8), (0.1, -0.45)] {
        painter.circle_filled(c + Vec2::new(r * dx, r * dy), (r * 0.08).max(1.0), ink);
    }
}

/// `app/src/nodes/stabilize.rs` `glyph`.
fn stabilize(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let (s, co) = (0.25f32).sin_cos();
    let rot = |v: Vec2| Vec2::new(v.x * co - v.y * s, v.x * s + v.y * co);
    let outer: Vec<Pos2> = [Vec2::new(-1.0, -0.7), Vec2::new(1.0, -0.7), Vec2::new(1.0, 0.7), Vec2::new(-1.0, 0.7)]
        .iter()
        .map(|v| c + rot(*v * r * 0.85))
        .collect();
    painter.add(egui::Shape::convex_polygon(outer, fill, stroke));
    let inner = egui::Rect::from_center_size(c, Vec2::new(r * 1.0, r * 0.62));
    painter.rect_stroke(inner, 0.0, Stroke::new((r * 0.12).max(1.0), theme::node_ink()), egui::StrokeKind::Middle);
}

/// `app/src/nodes/edgedetect.rs` `glyph`.
fn edgedetect(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let outer = Rect::from_center_size(c, Vec2::splat(r * 1.6));
    painter.rect_filled(outer, 0.0, fill);
    painter.rect_stroke(outer, 0.0, stroke, egui::StrokeKind::Middle);
    painter.rect_stroke(outer.shrink(r * 0.32), 0.0, stroke, egui::StrokeKind::Middle);
}

/// `app/src/nodes/blur.rs` `glyph`.
fn blur(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let pts = vec![
        Pos2::new(c.x, c.y - r),
        Pos2::new(c.x + r, c.y),
        Pos2::new(c.x, c.y + r),
        Pos2::new(c.x - r, c.y),
    ];
    painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
}

/// `app/src/nodes/glow.rs` `glyph`.
fn glow(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.45, fill, stroke);
    for k in 0..8 {
        let a = k as f32 * std::f32::consts::TAU / 8.0;
        let d = egui::vec2(a.cos(), a.sin());
        painter.line_segment([c + d * r * 0.65, c + d * r * 1.15], stroke);
    }
}

/// `app/src/nodes/lensflare.rs` `glyph`.
fn lensflare(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let sun = Pos2::new(c.x - r * 0.4, c.y - r * 0.4);
    painter.circle(sun, r * 0.26, fill, stroke);
    painter.line_segment([Pos2::new(sun.x - r * 0.75, sun.y), Pos2::new(sun.x + r * 0.95, sun.y)], stroke);
    painter.circle_stroke(Pos2::new(c.x + r * 0.2, c.y + r * 0.2), r * 0.16, stroke);
    painter.circle_stroke(Pos2::new(c.x + r * 0.6, c.y + r * 0.6), r * 0.28, stroke);
}

/// `app/src/nodes/godrays.rs` `glyph`.
fn godrays(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let sun = c + egui::vec2(0.0, -r * 0.45);
    painter.circle(sun, r * 0.32, fill, stroke);
    for k in 0..5 {
        let a = std::f32::consts::FRAC_PI_2 + (k as f32 - 2.0) * 0.38;
        let d = egui::vec2(a.cos(), a.sin());
        painter.line_segment([sun + d * r * 0.5, sun + d * r * 1.45], stroke);
    }
}

/// `app/src/nodes/glint.rs` `glyph`.
fn glint(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.32, fill, stroke);
    for k in 0..8 {
        let a = k as f32 * std::f32::consts::TAU / 8.0;
        let d = egui::vec2(a.cos(), a.sin());
        // Four long rays on the axes, four short ones between.
        let reach = if k % 2 == 0 { 1.25 } else { 0.75 };
        painter.line_segment([c + d * r * 0.42, c + d * r * reach], stroke);
    }
}

/// `app/src/nodes/diffuse.rs` `glyph`.
fn diffuse(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c + egui::vec2(0.0, -r * 0.2), r * 0.5, fill, stroke);
    for (dy, w) in [(0.25, 1.1), (0.65, 0.8)] {
        let y = c.y + r * dy;
        painter.line_segment([Pos2::new(c.x - r * w, y), Pos2::new(c.x + r * w, y)], stroke);
    }
}

/// `app/src/nodes/softcrop.rs` `glyph`.
fn softcrop(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.rect(
        egui::Rect::from_center_size(c, egui::vec2(r * 1.7, r * 1.7)),
        r * 0.3,
        fill,
        stroke,
        egui::StrokeKind::Middle,
    );
    let (h, k) = (r * 0.55, r * 0.3);
    let ink = Stroke::new((r * 0.14).max(1.0), crate::theme::node_ink());
    for (sx, sy) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        let p = c + egui::vec2(sx * h, sy * h);
        painter.line_segment([p, p - egui::vec2(sx * k, 0.0)], ink);
        painter.line_segment([p, p - egui::vec2(0.0, sy * k)], ink);
    }
}

/// `app/src/nodes/motionblur.rs` `glyph`.
fn motionblur(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let head = Pos2::new(c.x + r * 0.35, c.y);
    for (dy, len) in [(-0.45, 0.9), (0.0, 1.2), (0.45, 0.9)] {
        let y = c.y + dy * r;
        painter.line_segment(
            [Pos2::new(head.x - r * len, y), Pos2::new(head.x - r * 0.3, y)],
            Stroke::new(stroke.width.max(1.0), fill.gamma_multiply(0.7)),
        );
    }
    painter.circle(head, r * 0.55, fill, stroke);
}

/// `app/src/nodes/colorcompress.rs` `glyph`.
fn colorcompress(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.rect(
        egui::Rect::from_center_size(c, egui::vec2(r * 1.7, r * 1.7)),
        r * 0.3,
        fill,
        stroke,
        egui::StrokeKind::Middle,
    );
    let ink = Stroke::new((r * 0.14).max(1.0), crate::theme::node_ink());
    let h = r * 0.62;
    // y is down, so the line rises to the RIGHT.
    painter.line_segment([c + Vec2::new(-h, h * 0.55), c + Vec2::new(h, -h * 0.55)], ink);
    // The two points it is pinned to, on the input axis.
    let tick = (r * 0.22).max(1.5);
    for x in [-h * 0.45, h * 0.45] {
        let p = c + Vec2::new(x, h);
        painter.line_segment([p, p - Vec2::new(0.0, tick)], ink);
    }
}

/// `app/src/nodes/huerotate.rs` `glyph`.
fn huerotate(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.14).max(1.0), crate::theme::node_ink());
    let rr = r * 0.52;
    let (a0, a1) = (-0.35 * std::f32::consts::PI, 1.25 * std::f32::consts::PI);
    let n = 20;
    let pts: Vec<Pos2> = (0..=n)
        .map(|k| {
            let a = a0 + (a1 - a0) * k as f32 / n as f32;
            c + Vec2::new(a.cos(), a.sin()) * rr
        })
        .collect();
    let tip = *pts.last().unwrap();
    painter.add(egui::Shape::line(pts, ink));
    // The head points along the turn at its end.
    let t = Vec2::new(-a1.sin(), a1.cos());
    let side = Vec2::new(t.y, -t.x);
    let s = r * 0.26;
    painter.add(egui::Shape::convex_polygon(
        vec![tip + t * s, tip - t * s * 0.2 + side * s * 0.8, tip - t * s * 0.2 - side * s * 0.8],
        ink.color,
        Stroke::NONE,
    ));
}

/// `app/src/nodes/lut.rs` `glyph`.
fn lut(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.11).max(1.0), crate::theme::node_ink());
    let s = r * 0.36;
    let d = Vec2::new(r * 0.2, -r * 0.2);
    let f = [Vec2::new(-s, -s), Vec2::new(s, -s), Vec2::new(s, s), Vec2::new(-s, s)]
        .map(|v| c + v - d * 0.5);
    let b = f.map(|p| p + d);
    for i in 0..4 {
        painter.line_segment([f[i], f[(i + 1) % 4]], ink);
        painter.line_segment([f[i], b[i]], ink);
    }
    painter.line_segment([b[0], b[1]], ink);
    painter.line_segment([b[1], b[2]], ink);
}

/// `app/src/nodes/text.rs` `glyph`.
fn text(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.16).max(1.0), crate::theme::node_ink());
    let s = r * 0.42;
    painter.line_segment([c + Vec2::new(-s, -s), c + Vec2::new(s, -s)], ink);
    painter.line_segment([c + Vec2::new(0.0, -s), c + Vec2::new(0.0, s)], ink);
}

/// `app/src/nodes/burnin.rs` `glyph`.
fn burnin(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.11).max(1.0), crate::theme::node_ink());
    let (w, h) = (r * 0.5, r * 0.3);
    let rect = egui::Rect::from_center_size(c, Vec2::new(2.0 * w, 2.0 * h));
    painter.rect_stroke(rect, 1.0, ink, egui::StrokeKind::Middle);
    for k in 0..4 {
        let x = c.x - w + (k as f32 + 0.5) * (2.0 * w / 4.0);
        painter.line_segment([Pos2::new(x, c.y - h * 0.5), Pos2::new(x, c.y + h * 0.5)], ink);
    }
}

/// `app/src/nodes/lensdistort.rs` `glyph`.
fn lensdistort(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.1).max(1.0), crate::theme::node_ink());
    let h = r * 0.55;
    let n = 10;
    let bend = |q: Vec2| -> Vec2 {
        let r2 = (q / h).length_sq();
        q * (1.0 - 0.22 * r2)
    };
    for k in [-1.0f32, 0.0, 1.0] {
        let across: Vec<Pos2> =
            (0..=n).map(|i| c + bend(Vec2::new((i as f32 / n as f32 * 2.0 - 1.0) * h, k * h))).collect();
        let down: Vec<Pos2> =
            (0..=n).map(|i| c + bend(Vec2::new(k * h, (i as f32 / n as f32 * 2.0 - 1.0) * h))).collect();
        painter.add(egui::Shape::line(across, ink));
        painter.add(egui::Shape::line(down, ink));
    }
}

/// `app/src/nodes/mirror.rs` `glyph`.
fn mirror(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.12).max(1.0), crate::theme::node_ink());
    let h = r * 0.62;
    let w = r * 0.45;
    for (a, b) in [(-1.0, -0.6), (-0.3, 0.1), (0.4, 0.8)] {
        painter.line_segment([c + Vec2::new(0.0, h * a), c + Vec2::new(0.0, h * b)], ink);
    }
    let wedge = |sx: f32, filled: bool| {
        let pts = vec![
            c + Vec2::new(sx * r * 0.14, -h * 0.55),
            c + Vec2::new(sx * (r * 0.14 + w), 0.0),
            c + Vec2::new(sx * r * 0.14, h * 0.55),
        ];
        if filled {
            painter.add(egui::Shape::convex_polygon(pts, ink.color, Stroke::NONE));
        } else {
            painter.add(egui::Shape::closed_line(pts, ink));
        }
    };
    wedge(-1.0, true);
    wedge(1.0, false);
}

/// `app/src/nodes/flipflop.rs` `glyph`.
fn flipflop(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.12).max(1.0), crate::theme::node_ink());
    let l = r * 0.55;
    let k = r * 0.2;
    for d in [Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)] {
        let n = Vec2::new(-d.y, d.x);
        let (a, b) = (c - d * l, c + d * l);
        painter.line_segment([a, b], ink);
        for (tip, s) in [(a, 1.0), (b, -1.0)] {
            painter.line_segment([tip, tip + (d * s + n) * k], ink);
            painter.line_segment([tip, tip + (d * s - n) * k], ink);
        }
    }
}

/// `app/src/nodes/gamma.rs` `glyph`.
fn tonemap(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.rect(
        egui::Rect::from_center_size(c, egui::vec2(r * 1.7, r * 1.7)),
        r * 0.3,
        fill,
        stroke,
        egui::StrokeKind::Middle,
    );
    let ink = Stroke::new((r * 0.14).max(1.0), crate::theme::node_ink());
    let h = r * 0.6;
    // y is down. The ceiling, then the identity into a shoulder under it.
    let top = c.y - h * 0.8;
    painter.line_segment([Pos2::new(c.x - h, top), Pos2::new(c.x + h, top)], Stroke::new(ink.width * 0.5, ink.color));
    let n = 16;
    let pts: Vec<Pos2> = (0..=n)
        .map(|k| {
            let x = k as f32 / n as f32 * 2.0;
            let y = if x < 0.6 { x } else { 0.6 + 0.4 * (1.0 - (-(x - 0.6) / 0.4).exp()) };
            c + Vec2::new(-h + h * x, h - 1.8 * h * y)
        })
        .collect();
    painter.add(egui::Shape::line(pts, ink));
}

fn gamma(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.rect(
        egui::Rect::from_center_size(c, egui::vec2(r * 1.7, r * 1.7)),
        r * 0.3,
        fill,
        stroke,
        egui::StrokeKind::Middle,
    );
    let ink = Stroke::new((r * 0.14).max(1.0), crate::theme::node_ink());
    let h = r * 0.6;
    // y is down: x^(1/2.2) rising to the right, bowed up.
    let n = 16;
    let pts: Vec<Pos2> = (0..=n)
        .map(|k| {
            let t = k as f32 / n as f32;
            c + Vec2::new(-h + 2.0 * h * t, h - 2.0 * h * t.powf(1.0 / 2.2))
        })
        .collect();
    painter.add(egui::Shape::line(pts, ink));
}

/// `app/src/nodes/expression.rs` `glyph`.
fn expression(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.14).max(1.0), crate::theme::node_ink());
    let (w, gap) = (r * 0.42, r * 0.17);
    for dy in [-gap, gap] {
        painter.line_segment([Pos2::new(c.x - w, c.y + dy), Pos2::new(c.x + w, c.y + dy)], ink);
    }
}

/// `app/src/nodes/colorslice.rs` `glyph`.
fn colorslice(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let rr = r * 0.9;
    painter.circle(c, rr, fill, stroke);
    let lifted = fill.lerp_to_gamma(Color32::WHITE, 0.45);
    let mut wedge = vec![c];
    for k in 0..=6 {
        let a = std::f32::consts::TAU * (-0.25 + k as f32 / 36.0);
        wedge.push(c + Vec2::new(a.cos(), a.sin()) * rr);
    }
    painter.add(egui::Shape::convex_polygon(wedge, lifted, Stroke::NONE));
    let ink = Stroke::new((r * 0.1).max(1.0), crate::theme::node_ink());
    for k in 0..6 {
        let a = std::f32::consts::TAU * (-0.25 + k as f32 / 6.0);
        painter.line_segment([c, c + Vec2::new(a.cos(), a.sin()) * rr], ink);
    }
}

/// `app/src/nodes/warp.rs` `glyph`.
fn warp(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.1).max(1.0), crate::theme::node_ink());
    let faint = Stroke::new((r * 0.08).max(0.8), crate::theme::node_ink().gamma_multiply(0.5));
    const N: usize = 12;
    let curve = |bend: f32| -> Vec<Pos2> {
        (0..=N)
            .map(|i| {
                let t = i as f32 / N as f32 * 2.0 - 1.0;
                c + Vec2::new(t * r * 0.6, r * 0.15 - (1.0 - t * t) * r * bend)
            })
            .collect()
    };
    painter.add(egui::Shape::line(curve(0.0), faint));
    painter.add(egui::Shape::line(curve(0.5), ink));
    painter.line_segment([c + Vec2::new(0.0, r * 0.15), c + Vec2::new(0.0, r * 0.15 - r * 0.5)], faint);
}

/// `app/src/nodes/gridwarp.rs` `glyph`.
fn gridwarp(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let h = r * 0.8;
    painter.rect(
        Rect::from_center_size(c, Vec2::splat(h * 2.0)),
        2.0,
        fill,
        stroke,
        egui::StrokeKind::Middle,
    );
    let ink = Stroke::new((r * 0.1).max(1.0), crate::theme::node_ink());
    // A 2x2 lattice, its centre pulled up and right; each line bows
    // toward it.
    let pull = Vec2::new(h * 0.28, -h * 0.28);
    let at = |x: f32, y: f32| {
        let fall = (1.0 - x.abs()) * (1.0 - y.abs());
        c + Vec2::new(x * h * 0.8, y * h * 0.8) + pull * fall
    };
    const N: usize = 10;
    for k in [-1.0f32, 0.0, 1.0] {
        let across: Vec<Pos2> = (0..=N).map(|i| at(i as f32 / N as f32 * 2.0 - 1.0, k)).collect();
        let down: Vec<Pos2> = (0..=N).map(|i| at(k, i as f32 / N as f32 * 2.0 - 1.0)).collect();
        painter.add(egui::Shape::line(across, ink));
        painter.add(egui::Shape::line(down, ink));
    }
    painter.circle_filled(at(0.0, 0.0), (r * 0.14).max(1.0), crate::theme::node_ink());
}

/// `app/src/nodes/uvwarp.rs` `glyph`.
/// A globe with its meridians, under a frame: the sky through a camera.
fn env(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let rr = r * 0.82;
    painter.circle(c, rr, fill, stroke);
    let ink = Stroke::new((r * 0.1).max(1.0), crate::theme::node_ink());
    let n = 16;
    // The equator and two parallels.
    for k in [-0.5f32, 0.0, 0.5] {
        let y = c.y + k * rr;
        let hw = rr * (1.0 - k * k).sqrt();
        painter.line_segment([Pos2::new(c.x - hw, y), Pos2::new(c.x + hw, y)], ink);
    }
    // Two meridians, as ellipses' halves.
    for w in [0.45f32, 0.0] {
        let pts: Vec<Pos2> = (0..=n)
            .map(|i| {
                let t = std::f32::consts::PI * (i as f32 / n as f32 - 0.5);
                Pos2::new(c.x + w * rr * t.cos(), c.y + rr * t.sin())
            })
            .collect();
        painter.add(egui::Shape::line(pts, ink));
        if w > 0.0 {
            let pts: Vec<Pos2> = (0..=n)
                .map(|i| {
                    let t = std::f32::consts::PI * (i as f32 / n as f32 - 0.5);
                    Pos2::new(c.x - w * rr * t.cos(), c.y + rr * t.sin())
                })
                .collect();
            painter.add(egui::Shape::line(pts, ink));
        }
    }
}

fn uvwarp(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let h = r * 0.8;
    painter.rect(
/// `app/src/nodes/refract.rs` `glyph`.
/// A pane of glass with a ray kinking through it.
fn refract(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let h = r * 0.8;
    painter.rect(
        egui::Rect::from_center_size(c, egui::vec2(h * 2.0, h * 1.1)),
        2.0,
        fill,
        stroke,
        egui::StrokeKind::Middle,
    );
    let ink = Stroke::new((r * 0.13).max(1.0), crate::theme::node_ink());
    // In from above, bent towards the normal inside, out parallel below.
    let pts = [
        c + Vec2::new(-h * 0.85, -h * 1.05),
        c + Vec2::new(-h * 0.3, -h * 0.55),
        c + Vec2::new(h * 0.05, h * 0.55),
        c + Vec2::new(h * 0.6, h * 1.05),
    ];
    painter.add(egui::Shape::line(pts.to_vec(), ink));
}

        egui::Rect::from_center_size(c, egui::vec2(h * 2.0, h * 2.0)),
        2.0,
        fill,
        stroke,
        egui::StrokeKind::Middle,
    );
    let ink = Stroke::new((r * 0.11).max(1.0), crate::theme::node_ink());
    let n = 12;
    for k in [-1.0f32, 0.0, 1.0] {
        let off = k * h * 0.5;
        // One bowed line each way, bending further from the middle.
        let across: Vec<Pos2> = (0..=n)
            .map(|i| {
                let t = i as f32 / n as f32 * 2.0 - 1.0;
                c + Vec2::new(t * h * 0.8, off + (1.0 - t * t) * h * 0.22 * (1.0 + k.abs()))
            })
            .collect();
        let down: Vec<Pos2> = (0..=n)
            .map(|i| {
                let t = i as f32 / n as f32 * 2.0 - 1.0;
                c + Vec2::new(off + (1.0 - t * t) * h * 0.22 * (1.0 + k.abs()), t * h * 0.8)
            })
            .collect();
        painter.add(egui::Shape::line(across, ink));
        painter.add(egui::Shape::line(down, ink));
    }
}

/// `app/src/nodes/todata.rs` `glyph`.
fn todata(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let w = r * 0.42;
    painter.rect(
        egui::Rect::from_center_size(c, egui::vec2(w * 2.0, r * 1.5)),
        1.0,
        fill,
        stroke,
        egui::StrokeKind::Middle,
    );
    // The lane's value keeps going past the box, both ways.
    for s in [-1.0f32, 1.0] {
        painter.line_segment(
            [c + egui::vec2(0.0, s * r * 0.75), c + egui::vec2(0.0, s * r * 1.15)],
            stroke,
        );
    }
}

/// `app/src/nodes/torgba.rs` `glyph`.
fn torgba(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let h = r * 0.8;
    painter.rect(
        Rect::from_center_size(c, Vec2::splat(h * 2.0)),
        1.0,
        fill,
        stroke,
        egui::StrokeKind::Outside,
    );
    painter.add(egui::Shape::convex_polygon(
        vec![
            Pos2::new(c.x - h, c.y - h),
            Pos2::new(c.x, c.y - h),
            Pos2::new(c.x, c.y + h),
            Pos2::new(c.x - h, c.y + h),
        ],
        crate::theme::wire_mask().gamma_multiply(0.9),
        Stroke::NONE,
    ));
}

/// `app/src/nodes/gradient.rs` `glyph`.
fn gradient(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, Color32::from_gray(20), stroke);
    const N: usize = 5;
    let rr = r * 0.9;
    for k in 0..N {
        let x0 = -rr + 2.0 * rr * k as f32 / N as f32;
        let x1 = -rr + 2.0 * rr * (k + 1) as f32 / N as f32;
        let y0 = (rr * rr - x0 * x0).max(0.0).sqrt();
        let y1 = (rr * rr - x1 * x1).max(0.0).sqrt();
        let a = (k + 1) as f32 / N as f32;
        let col = Color32::from_rgb(
            (fill.r() as f32 * a) as u8,
            (fill.g() as f32 * a) as u8,
            (fill.b() as f32 * a) as u8,
        );
        let pts = vec![
            Pos2::new(c.x + x0, c.y - y0),
            Pos2::new(c.x + x1, c.y - y1),
            Pos2::new(c.x + x1, c.y + y1),
            Pos2::new(c.x + x0, c.y + y0),
        ];
        painter.add(egui::Shape::convex_polygon(pts, col, Stroke::NONE));
    }
}

/// `app/src/nodes/glintkernel.rs` `glyph`.
fn glint_kernel(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let frame = egui::Rect::from_center_size(c, egui::vec2(r * 1.8, r * 1.8));
    painter.rect(frame, r * 0.15, fill, stroke, egui::StrokeKind::Middle);
    for k in 0..4 {
        let a = k as f32 * std::f32::consts::TAU / 4.0;
        let d = egui::vec2(a.cos(), a.sin());
        painter.line_segment([c - d * r * 0.75, c + d * r * 0.75], stroke);
    }
}

/// `app/src/nodes/bokehkernel.rs` `glyph`.
fn bokeh_kernel(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let frame = egui::Rect::from_center_size(c, egui::vec2(r * 1.8, r * 1.8));
    painter.rect(frame, r * 0.15, Color32::TRANSPARENT, stroke, egui::StrokeKind::Middle);
    let pts: Vec<Pos2> = (0..6)
        .map(|k| {
            let a = k as f32 * std::f32::consts::TAU / 6.0;
            Pos2::new(c.x + a.cos() * r * 0.7, c.y + a.sin() * r * 0.7)
        })
        .collect();
    painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
}

/// `app/src/nodes/blank.rs` `glyph`.
fn blank(painter: &Painter, c: Pos2, r: f32, fill: egui::Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
}

/// `app/src/nodes/planet.rs` `glyph`.
fn planet(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let rr = r * 0.8;
    painter.circle(c, rr, fill, stroke);
    for k in [-0.4f32, 0.05, 0.45] {
        let y = c.y + k * rr;
        let w = (1.0 - k * k).max(0.0).sqrt() * rr * 0.92;
        painter.line_segment([Pos2::new(c.x - w, y), Pos2::new(c.x + w, y)], stroke);
    }
    painter.circle_stroke(Pos2::new(c.x + rr * 0.35, c.y - rr * 0.35), rr * 0.18, stroke);
}

/// `app/src/nodes/starfield.rs` `glyph`.
fn starfield(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, Color32::from_gray(16), stroke);
    // Four spikes, each a triangle (a convex shape egui fills correctly).
    let star = |p: Pos2, s: f32| {
        let k = s * 0.22;
        for (dx, dy) in [(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)] {
            let tip = Pos2::new(p.x + dx * s, p.y + dy * s);
            let a = Pos2::new(p.x - dy * k, p.y + dx * k);
            let b = Pos2::new(p.x + dy * k, p.y - dx * k);
            painter.add(egui::Shape::convex_polygon(vec![tip, a, b], fill, Stroke::NONE));
        }
    };
    star(Pos2::new(c.x - r * 0.1, c.y + r * 0.05), r * 0.55);
    painter.circle_filled(Pos2::new(c.x + r * 0.45, c.y - r * 0.4), r * 0.09, fill);
    painter.circle_filled(Pos2::new(c.x + r * 0.35, c.y + r * 0.5), r * 0.07, fill);
    painter.circle_filled(Pos2::new(c.x - r * 0.5, c.y - r * 0.45), r * 0.06, fill);
}

/// `app/src/nodes/plasma.rs` `glyph`.
fn plasma(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let centre = Pos2::new(c.x, c.y + r * 0.45);
    let rad = r * 0.62;
    painter.circle(centre, rad, fill, stroke);
    let line = Stroke::new((r * 0.11).max(1.0), fill);
    // An arch between two limb angles (radians, 0 = up), `h` radii tall.
    let arch = |a0: f32, a1: f32, h: f32| {
        let pts: Vec<Pos2> = (0..=10)
            .map(|i| {
                let s = i as f32 / 10.0;
                let a = a0 + (a1 - a0) * s;
                let k = rad * (1.0 + h * (std::f32::consts::PI * s).sin());
                Pos2::new(centre.x + a.sin() * k, centre.y - a.cos() * k)
            })
            .collect();
        painter.add(egui::Shape::line(pts, line));
    };
    arch(-0.75, 0.05, 0.75);
    arch(0.25, 0.9, 0.45);
}

/// `app/src/nodes/particles.rs` `glyph`.
fn particles(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let src = Pos2::new(c.x - r * 0.45, c.y + r * 0.45);
    painter.circle(src, r * 0.22, fill, stroke);
    let dots = [(0.1, -0.15, 0.13), (0.45, -0.05, 0.1), (0.25, -0.5, 0.11), (0.6, -0.45, 0.08), (-0.1, -0.6, 0.09)];
    for (dx, dy, s) in dots {
        painter.circle_filled(Pos2::new(c.x + dx * r, c.y + dy * r), r * s, fill);
    }
}

/// `app/src/nodes/prompt.rs` `glyph`.
fn prompt(painter: &Painter, c: Pos2, r: f32, fill: egui::Color32, stroke: Stroke) {
    let page = egui::Rect::from_center_size(c, egui::vec2(r * 1.3, r * 1.5));
    painter.rect(page, r * 0.15, fill, stroke, egui::StrokeKind::Middle);
    let ink = Stroke::new(stroke.width.max(1.0), stroke.color);
    for (k, w) in [(0.0f32, 0.8f32), (1.0, 0.8), (2.0, 0.5)] {
        let y = page.top() + r * (0.4 + 0.35 * k);
        let x0 = page.left() + r * 0.22;
        painter.line_segment([egui::pos2(x0, y), egui::pos2(x0 + r * w, y)], ink);
    }
}

/// `app/src/nodes/grain.rs` `glyph`.
fn grain(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = crate::theme::node_ink();
    // A fixed scatter, so the glyph is the same every frame.
    const DOTS: [(f32, f32, f32); 9] = [
        (-0.42, -0.30, 0.13),
        (0.10, -0.50, 0.09),
        (0.45, -0.18, 0.12),
        (-0.15, -0.05, 0.10),
        (0.25, 0.20, 0.08),
        (-0.50, 0.25, 0.09),
        (-0.05, 0.45, 0.12),
        (0.50, 0.40, 0.07),
        (0.05, 0.10, 0.06),
    ];
    for (x, y, s) in DOTS {
        painter.circle_filled(c + egui::vec2(x, y) * r, (s * r).max(0.8), ink);
    }
}

/// `app/src/nodes/saturation.rs` `glyph`.
fn saturation(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let grey =
        Color32::from_gray(((fill.r() as u32 + fill.g() as u32 + fill.b() as u32) / 3) as u8);
    let mut half = vec![Pos2::new(c.x, c.y - r * 0.9)];
    for k in 0..=8 {
        let a = std::f32::consts::PI * (k as f32 / 8.0 - 0.5);
        half.push(c + Vec2::new(a.cos(), a.sin()) * r * 0.9);
    }
    painter.add(egui::Shape::convex_polygon(half, grey, Stroke::NONE));
}

/// `app/src/nodes/contrast.rs` `glyph`.
fn contrast(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.14).max(1.0), theme::node_ink());
    let h = r * 0.55;
    let n = 16;
    let pts: Vec<Pos2> = (0..=n)
        .map(|k| {
            let t = k as f32 / n as f32 * 2.0 - 1.0;
            c + Vec2::new(h * t, -h * (t * 2.2).tanh() / 2.2f32.tanh())
        })
        .collect();
    painter.add(egui::Shape::line(pts, ink));
}

/// `app/src/blenderui.rs` `glyph`.
fn blender(painter: &egui::Painter, c: egui::Pos2, r: f32, fill: egui::Color32, stroke: egui::Stroke) {
    let body = egui::Rect::from_center_size(egui::pos2(c.x - r * 0.15, c.y), egui::vec2(r * 1.5, r * 1.1));
    painter.rect(body, r * 0.2, fill, stroke, egui::StrokeKind::Middle);
    let lens = [
        egui::pos2(c.x + r * 0.55, c.y - r * 0.3),
        egui::pos2(c.x + r * 1.05, c.y - r * 0.6),
        egui::pos2(c.x + r * 1.05, c.y + r * 0.6),
        egui::pos2(c.x + r * 0.55, c.y + r * 0.3),
    ];
    painter.add(egui::Shape::convex_polygon(lens.to_vec(), fill, stroke));
    painter.circle_stroke(egui::pos2(c.x - r * 0.15, c.y), r * 0.3, stroke);
}

/// `app/src/nodes/corridorkey.rs` `glyph`.
fn corridorkey(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = crate::theme::node_ink();
    let screen = Rect::from_center_size(c, Vec2::new(r * 1.1, r * 0.9));
    painter.rect_filled(screen, r * 0.12, ink);
    painter.circle_filled(c + Vec2::new(0.0, -r * 0.12), r * 0.17, fill);
    let base = Pos2::new(c.x, screen.bottom());
    let pts: Vec<Pos2> = (0..=10)
        .map(|k| {
            let a = std::f32::consts::PI * (1.0 + k as f32 / 10.0);
            base + Vec2::new(a.cos(), a.sin()) * r * 0.36
        })
        .collect();
    painter.add(egui::Shape::convex_polygon(pts, fill, Stroke::NONE));
    // The spark, top right of the screen.
    let s = c + Vec2::new(r * 0.42, -r * 0.34);
    let k = r * 0.16;
    painter.line_segment([s - Vec2::new(k, 0.0), s + Vec2::new(k, 0.0)], Stroke::new(stroke.width, fill));
    painter.line_segment([s - Vec2::new(0.0, k), s + Vec2::new(0.0, k)], Stroke::new(stroke.width, fill));
}

/// `app/src/nodes/sammie.rs` `glyph`.
fn sammie(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = crate::theme::node_ink();
    // Head and shoulders.
    painter.circle_filled(c + Vec2::new(-r * 0.12, -r * 0.22), r * 0.24, ink);
    let base = Pos2::new(c.x - r * 0.12, c.y + r * 0.5);
    let pts: Vec<Pos2> = (0..=10)
        .map(|k| {
            let a = std::f32::consts::PI * (1.0 + k as f32 / 10.0);
            base + Vec2::new(a.cos() * 0.46, a.sin() * 0.42) * r
        })
        .collect();
    painter.add(egui::Shape::convex_polygon(pts, ink, Stroke::NONE));
    // The include click, on the figure.
    painter.circle_filled(c + Vec2::new(-r * 0.12, r * 0.22), r * 0.09, fill);
    // The exclude click, off it.
    let x = c + Vec2::new(r * 0.46, -r * 0.3);
    let k = r * 0.12;
    let s = Stroke::new(stroke.width, ink);
    painter.line_segment([x - Vec2::new(k, k), x + Vec2::new(k, k)], s);
    painter.line_segment([x - Vec2::new(k, -k), x + Vec2::new(k, -k)], s);
}

/// `app/src/nodes/screenkey.rs` `glyph`.
fn screenkey(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = crate::theme::node_ink();
    let screen = Rect::from_center_size(c, Vec2::new(r * 1.1, r * 0.9));
    painter.rect_filled(screen, r * 0.12, ink);
    // Head, then shoulders rising from the screen's bottom edge.
    painter.circle_filled(c + Vec2::new(0.0, -r * 0.12), r * 0.17, fill);
    let base = Pos2::new(c.x, screen.bottom());
    let pts: Vec<Pos2> = (0..=10)
        .map(|k| {
            let a = std::f32::consts::PI * (1.0 + k as f32 / 10.0);
            base + Vec2::new(a.cos(), a.sin()) * r * 0.36
        })
        .collect();
    painter.add(egui::Shape::convex_polygon(pts, fill, Stroke::NONE));
}

/// `app/src/nodes/despill.rs` `glyph`.
fn despill(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.14).max(1.0), theme::node_ink());
    // The drop: a point on top, a round belly below.
    let n = 14;
    let mut pts = vec![c + Vec2::new(0.0, -r * 0.6)];
    for k in 0..=n {
        let a = std::f32::consts::PI * (-0.15 + 1.3 * k as f32 / n as f32);
        pts.push(c + Vec2::new(a.cos(), a.sin()) * r * 0.32 + Vec2::new(0.0, r * 0.15));
    }
    pts.push(c + Vec2::new(0.0, -r * 0.6));
    painter.add(egui::Shape::line(pts, ink));
    painter.line_segment([c + Vec2::new(-r * 0.5, r * 0.5), c + Vec2::new(r * 0.5, -r * 0.5)], ink);
}

/// `app/src/nodes/lightwrap.rs` `glyph`.
fn upscale(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let big = egui::Rect::from_center_size(c, Vec2::splat(r * 1.7));
    painter.rect(big, r * 0.15, fill, stroke, egui::StrokeKind::Middle);
    let s = r * 0.62;
    let small = egui::Rect::from_min_size(Pos2::new(big.min.x + r * 0.22, big.max.y - r * 0.22 - s), Vec2::splat(s));
    painter.rect_filled(small, r * 0.08, theme::node_ink());
    let ink = Stroke::new((r * 0.14).max(1.0), theme::node_ink());
    let a = Pos2::new(small.max.x + r * 0.1, small.min.y - r * 0.1);
    let b = Pos2::new(big.max.x - r * 0.25, big.min.y + r * 0.25);
    painter.line_segment([a, b], ink);
    painter.line_segment([b, b + Vec2::new(-r * 0.32, 0.0)], ink);
    painter.line_segment([b, b + Vec2::new(0.0, r * 0.32)], ink);
}

fn seedvr(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let sq = egui::Rect::from_center_size(c, Vec2::splat(r * 1.7));
    painter.rect(sq, r * 0.35, fill, stroke, egui::StrokeKind::Middle);
    let ink = theme::node_ink();
    let seed = c + Vec2::new(0.0, r * 0.32);
    painter.add(egui::Shape::convex_polygon(
        vec![
            seed + Vec2::new(0.0, -r * 0.28),
            seed + Vec2::new(r * 0.2, 0.0),
            seed + Vec2::new(0.0, r * 0.28),
            seed + Vec2::new(-r * 0.2, 0.0),
        ],
        ink,
        Stroke::NONE,
    ));
    let s = Stroke::new((r * 0.12).max(1.0), ink);
    let top = seed + Vec2::new(0.0, -r * 0.62);
    painter.line_segment([seed + Vec2::new(0.0, -r * 0.28), top], s);
    painter.line_segment([top, top + Vec2::new(-r * 0.3, r * 0.12)], s);
    painter.line_segment([top, top + Vec2::new(r * 0.3, r * 0.12)], s);
}

fn depth(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let sq = egui::Rect::from_center_size(c, Vec2::splat(r * 1.7));
    painter.rect(sq, r * 0.35, fill, stroke, egui::StrokeKind::Middle);
    let ink = theme::node_ink();
    for (i, (w, dy)) in [(0.95, 0.42), (0.68, 0.02), (0.42, -0.34)].into_iter().enumerate() {
        let a = 1.0 - i as f32 * 0.3;
        let y = c.y + r * dy;
        let s = Stroke::new((r * 0.14 * a).max(1.0), ink.gamma_multiply(a));
        painter.line_segment([Pos2::new(c.x - r * w * 0.5, y), Pos2::new(c.x + r * w * 0.5, y)], s);
    }
}

/// `app/src/nodes/hdrcnn.rs` `glyph`.
fn hdrcnn(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let sq = egui::Rect::from_center_size(c, Vec2::splat(r * 1.7));
    painter.rect(sq, r * 0.15, fill, stroke, egui::StrokeKind::Middle);
    let ink = theme::node_ink();
    let cut = c.y - r * 0.05;
    painter.line_segment([Pos2::new(c.x - r * 0.62, cut), Pos2::new(c.x + r * 0.62, cut)], Stroke::new((r * 0.12).max(1.0), ink));
    for (i, j) in [(-1, 0), (0, 0), (1, 0), (0, 1)] {
        let cell = egui::Rect::from_center_size(
            Pos2::new(c.x + i as f32 * r * 0.3, cut - r * 0.2 - j as f32 * r * 0.3),
            Vec2::splat(r * 0.22),
        );
        painter.rect_filled(cell, 0.0, ink);
    }
    painter.circle_stroke(c + Vec2::new(0.0, r * 0.32), r * 0.3, Stroke::new((r * 0.1).max(1.0), ink));
}

/// `app/src/nodes/ltxhdr.rs` `glyph`.
fn ltxhdr(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let sq = egui::Rect::from_center_size(c, Vec2::splat(r * 1.7));
    painter.rect(sq, r * 0.35, fill, stroke, egui::StrokeKind::Middle);
    let ink = theme::node_ink();
    let sun = c + Vec2::new(0.0, r * 0.18);
    painter.circle_filled(sun, r * 0.3, ink);
    for k in 0..5 {
        let a = std::f32::consts::PI * (1.0 + k as f32 / 4.0);
        let d = Vec2::new(a.cos(), a.sin());
        painter.line_segment([sun + d * r * 0.42, sun + d * r * 0.72], Stroke::new((r * 0.1).max(1.0), ink));
    }
    let y = c.y + r * 0.5;
    painter.line_segment([Pos2::new(c.x - r * 0.7, y), Pos2::new(c.x + r * 0.7, y)], Stroke::new((r * 0.1).max(1.0), ink));
}

fn lightwrap(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    painter.circle_filled(c, r * 0.42, theme::node_ink());
    let ink = Stroke::new((r * 0.14).max(1.0), theme::node_ink());
    let n = 16;
    let (a0, a1) = (-0.95 * std::f32::consts::PI, -0.05 * std::f32::consts::PI);
    let pts: Vec<Pos2> = (0..=n)
        .map(|k| {
            let a = a0 + (a1 - a0) * k as f32 / n as f32;
            c + Vec2::new(a.cos(), a.sin()) * r * 0.62
        })
        .collect();
    painter.add(egui::Shape::line(pts, ink));
}

/// `app/src/nodes/flow.rs` `glyph`.
fn flow(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.12).max(1.0), crate::theme::node_ink());
    for (dy, len, tilt) in [(-0.4f32, 0.5f32, -0.15f32), (0.0, 0.75, 0.0), (0.4, 0.55, 0.15)] {
        let a = c + Vec2::new(-r * 0.45, dy * r);
        let b = a + Vec2::new(len * r, tilt * r);
        painter.line_segment([a, b], ink);
        let back = (a - b).normalized() * r * 0.18;
        let side = Vec2::new(-back.y, back.x) * 0.7;
        painter.line_segment([b, b + back + side], ink);
        painter.line_segment([b, b + back - side], ink);
    }
}

/// `app/src/nodes/vectorblur.rs` `glyph`.
fn vectorblur(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let head = c + Vec2::new(r * 0.38, -r * 0.1);
    for k in 1..=3 {
        let t = k as f32 * 0.22;
        painter.circle_filled(head - Vec2::new(r * t * 1.6, -r * t * 0.4), r * 0.5, fill.gamma_multiply(0.22 * (4 - k) as f32));
    }
    painter.circle(head, r * 0.5, fill, stroke);
    let ink = Stroke::new((r * 0.12).max(1.0), crate::theme::node_ink());
    let (a, b) = (head - Vec2::new(r * 0.3, -r * 0.08), head + Vec2::new(r * 0.3, -r * 0.08));
    painter.line_segment([a, b], ink);
    painter.line_segment([b, b - Vec2::new(r * 0.16, r * 0.1)], ink);
    painter.line_segment([b, b - Vec2::new(r * 0.12, -r * 0.16)], ink);
}

/// `app/src/nodes/vectorwarp.rs` `glyph`.
fn vectorwarp(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let h = r * 0.7;
    let pts = vec![
        c + Vec2::new(-h, -h),
        c + Vec2::new(h * 0.6, -h),
        c + Vec2::new(h * 1.15, 0.0),
        c + Vec2::new(h * 0.6, h),
        c + Vec2::new(-h, h),
    ];
    painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
    let ink = Stroke::new((r * 0.12).max(1.0), crate::theme::node_ink());
    let (a, b) = (c + Vec2::new(-h * 0.6, 0.0), c + Vec2::new(h * 0.55, 0.0));
    painter.line_segment([a, b], ink);
    painter.line_segment([b, b + Vec2::new(-r * 0.2, -r * 0.18)], ink);
    painter.line_segment([b, b + Vec2::new(-r * 0.2, r * 0.18)], ink);
}

/// `app/src/nodes/retime.rs` `glyph`.
fn retime(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.12).max(1.0), crate::theme::node_ink());
    for k in 0..3 {
        let x = c.x - r * 0.45 + k as f32 * r * 0.3;
        let rect = egui::Rect::from_center_size(Pos2::new(x, c.y + r * 0.1), Vec2::new(r * 0.36, r * 0.5));
        painter.rect_stroke(rect, 1.0, ink, egui::StrokeKind::Middle);
    }
    painter.line_segment([c + Vec2::new(-r * 0.5, -r * 0.42), c + Vec2::new(r * 0.5, -r * 0.42)], ink);
}

/// `app/src/nodes/freeze.rs` `glyph`.
fn freeze(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = crate::theme::node_ink();
    for dx in [-0.22, 0.22] {
        let bar = egui::Rect::from_center_size(c + Vec2::new(r * dx, 0.0), Vec2::new(r * 0.2, r * 0.8));
        painter.rect_filled(bar, r * 0.04, ink);
    }
}

/// `app/src/nodes/framefix.rs` `glyph`.
fn framefix(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.12).max(1.0), crate::theme::node_ink());
    for k in [-1.0f32, 1.0] {
        let rect = egui::Rect::from_center_size(c + Vec2::new(k * r * 0.42, 0.0), Vec2::new(r * 0.3, r * 0.5));
        painter.rect_stroke(rect, 1.0, ink, egui::StrokeKind::Middle);
    }
    let mid = egui::Rect::from_center_size(c, Vec2::new(r * 0.3, r * 0.5));
    painter.rect_filled(mid, 1.0, crate::theme::node_ink());
    painter.line_segment([c + Vec2::new(-r * 0.27, 0.0), c + Vec2::new(r * 0.27, 0.0)], ink);
}

/// `app/src/nodes/scene3d.rs` `camera_glyph`.
fn camera3d(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let body = egui::Rect::from_center_size(c - Vec2::new(r * 0.2, 0.0), Vec2::new(r * 1.0, r * 0.8));
    painter.rect(body, r * 0.12, fill, stroke, egui::StrokeKind::Middle);
    let lens = vec![
        Pos2::new(body.right(), c.y - r * 0.18),
        Pos2::new(body.right() + r * 0.5, c.y - r * 0.4),
        Pos2::new(body.right() + r * 0.5, c.y + r * 0.4),
        Pos2::new(body.right(), c.y + r * 0.18),
    ];
    painter.add(egui::Shape::convex_polygon(lens, fill, stroke));
    let ink = Stroke::new((r * 0.1).max(1.0), crate::theme::node_ink());
    painter.circle_stroke(body.center(), r * 0.22, ink);
}

/// `app/src/nodes/scene3d.rs` `card_glyph`.
fn card3d(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let q = vec![
        c + Vec2::new(-r * 0.55, -r * 0.45),
        c + Vec2::new(r * 0.45, -r * 0.7),
        c + Vec2::new(r * 0.55, r * 0.45),
        c + Vec2::new(-r * 0.45, r * 0.7),
    ];
    painter.add(egui::Shape::convex_polygon(q, fill, stroke));
    let ink = Stroke::new((r * 0.1).max(1.0), crate::theme::node_ink());
    painter.line_segment([c + Vec2::new(-r * 0.3, r * 0.25), c + Vec2::new(r * 0.05, -r * 0.15)], ink);
    painter.line_segment([c + Vec2::new(r * 0.05, -r * 0.15), c + Vec2::new(r * 0.3, r * 0.1)], ink);
}

/// `app/src/nodes/scene3d.rs` `scene_glyph`.
fn scene3d(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    for k in (0..3).rev() {
        let o = Vec2::new(k as f32 * r * 0.22, -(k as f32) * r * 0.22);
        let rect = egui::Rect::from_center_size(c + o - Vec2::new(r * 0.22, -r * 0.22), Vec2::new(r * 0.8, r * 0.8));
        painter.rect(rect, r * 0.08, fill.gamma_multiply(1.0 - 0.25 * k as f32), stroke, egui::StrokeKind::Middle);
    }
}

/// `app/src/nodes/scene3d.rs` `render_glyph`.
fn render3d(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let eye = c - Vec2::new(r * 0.6, 0.0);
    let frame = egui::Rect::from_center_size(c + Vec2::new(r * 0.3, 0.0), Vec2::new(r * 0.6, r * 1.1));
    painter.add(egui::Shape::convex_polygon(
        vec![eye, frame.left_top(), frame.left_bottom()],
        fill.gamma_multiply(0.5),
        stroke,
    ));
    painter.rect(frame, r * 0.06, fill, stroke, egui::StrokeKind::Middle);
    painter.circle_filled(eye, r * 0.12, crate::theme::node_ink());
}
