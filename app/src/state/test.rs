//! The --test-* suites' state: one slot (or a few) per suite, None unless that suite is running.
//! These were loose fields on `Gfx`; grouped, not changed.

use crate::*;

/// The --test-* suites' state: one slot (or a few) per suite, None unless that suite is running.
pub(crate) struct SuiteState {
    /// --test-inout: the in point a mark drag started from.
    pub(crate) inout_mark: Option<u64>,
    /// --test-spline: where the transformed-shape drag pressed. The cursor
    /// path has to be anchored there, not re-read from the point it is
    /// dragging — now that the point tracks the cursor exactly, moving
    /// relative to the point compounds each hop.
    pub(crate) spline_press: Option<egui::Pos2>,
    /// --test-spline: screen positions a pivot / distort stage holds on to
    /// across its gesture (the stash is only two floats).
    pub(crate) spline_aux: Vec<egui::Pos2>,
    /// --test-click state machine: frame counter for inject/verify stages.
    pub(crate) click: Option<u32>,
    /// The node the test clicked, pinned at injection time (draw order — and
    /// therefore node_rects order — changes when a click raises a node).
    pub(crate) target: Option<(u64, egui::Pos2)>,
    /// --test-seek state machine: (stage, frames elapsed, displayed marker).
    pub(crate) seek: Option<(u32, u32, u64)>,
    /// --test-play state machine: (stage, displayed marker, stage start).
    pub(crate) play: Option<(u32, u64, Instant)>,
    /// --test-browser state machine: (frames elapsed, row selected yet).
    pub(crate) browser: Option<(u32, bool)>,
    /// --test-setup: (stage, frames, expected nodes, expected loads).
    pub(crate) setup: Option<(u32, u32, usize, usize)>,
    /// --test-slice: (stage, frames, spliced-node id).
    pub(crate) slice: Option<(u32, u32, u64)>,
    /// --test-box: (stage, frames).
    pub(crate) box_: Option<(u32, u32)>,
    /// --test-curves: (stage, frames).
    pub(crate) curves: Option<(u32, u32)>,
    /// --test-curves: where the group-scale drag grabbed the box edge.
    pub(crate) scale_from: Option<egui::Pos2>,
    /// --test-inout: (stage, frames).
    pub(crate) inout: Option<(u32, u32)>,
    /// (stage, frames in stage) for `--test-frames`.
    pub(crate) frames: Option<(u32, u32)>,
    /// `--test-gizmokey`: the key its first phase landed, so the second can
    /// prove a LOCKED channel still holds it, and the key its second landed.
    pub(crate) gizmokey_key6: f32,
    pub(crate) gizmokey_y8: f32,
    /// `--test-gizmokey`: the element's position before the path-key drag.
    pub(crate) gizmokey_before: [f32; 2],
    /// (stage, frames in stage) for `--test-bake`.
    pub(crate) bake: Option<(u32, u32)>,
    /// (stage, frames in stage) for `--test-range`, its clock, the last
    /// probe (a reading is believed once two agree) and the frames probed.
    pub(crate) range: Option<(u32, u32)>,
    pub(crate) range_t0: Option<Instant>,
    pub(crate) range_last: Option<f32>,
    pub(crate) range_seen: Vec<(u64, f32)>,
    /// `--test-bake`: what the LIVE subtree showed at each probed frame.
    pub(crate) bake_live: Vec<(u64, f32)>,
    /// `--test-bake`: what the BAKED node showed, frame by frame.
    pub(crate) bake_seen: Vec<(u64, f32)>,
    /// `--test-bake`: the unplaced pass's live readings, by OFFSET into the
    /// shot. Placing a plate changes what a frame is CALLED, never what it
    /// shows, so the placed pass must read the same values.
    pub(crate) bake_ref: Vec<(u64, f32)>,
    /// `--test-bake`: which pass — 0 = a plain shot, 1 = the same comp with
    /// its plate PLACED at 1011, which is where a bake and the comp can
    /// disagree about what frame a frame is.
    pub(crate) bake_pass: u32,
    /// `--test-bake`: the previous probe, so a value is only believed once
    /// it has stopped changing (a seek lands, then the comp renders, then
    /// the texture shows it — reading earlier reads the frame before).
    pub(crate) bake_last: Option<f32>,
    pub(crate) bake_t0: Option<Instant>,
    /// Frames `--test-frames` actually saw during playback.
    pub(crate) frames_seen: Vec<u64>,
    /// When `--test-frames` started. It waits on real decodes, so it times
    /// out on a CLOCK — a redraw budget just measures how fast this machine
    /// spins its UI loop (see CLAUDE.md, "Rules for suites").
    pub(crate) frames_t0: Option<Instant>,
    /// --test-shot: (stage, frames, mark).
    pub(crate) shot: Option<(u32, u32, u64)>,
    /// --test-paint: (stage, frames, paint node id).
    pub(crate) paint: Option<(u32, u32, u64)>,
    /// --test-paint: the Bake its last stages inserted.
    pub(crate) paint_bake: Option<u64>,
    /// --test-tab: (stage, frames).
    pub(crate) tab: Option<(u32, u32)>,
    /// --test-quickadd: (stage, start).
    pub(crate) quickadd: Option<(u32, std::time::Instant)>,
    /// --test-quickadd: the node count a stage compares against.
    pub(crate) quickadd_n: usize,
    /// --test-zones: (stage, frames in it, the Grade), the press point, and
    /// the wall-clock start the budget runs on.
    pub(crate) zones: Option<(u32, u32, u64)>,
    pub(crate) zones_at: egui::Pos2,
    pub(crate) zones_t0: Option<Instant>,
    pub(crate) zones_since: Instant,
    pub(crate) zones_px: Option<[f32; 3]>,
    pub(crate) picker: Option<(u32, u32)>,
    /// Input-injection suites: the viewer rect last frame + how long it has
    /// held still, and the anchor a gesture was started from. Panels animate
    /// their height, so a press computed from a moving rect can miss its
    /// handle by a few pixels.
    pub(crate) prev_rect: egui::Rect,
    pub(crate) stable: u32,
    pub(crate) pos: Option<egui::Pos2>,
    /// `--test-click` proxy phase: (full-res pivot-drag translate x the
    /// on-screen image width, full-res canvas width). The product is what
    /// must survive proxy: the same screen drag has to move the image the
    /// same distance on screen, whatever size the canvas is.
    pub(crate) proxy_drag: (f32, u32, Option<web_time::Instant>),
    /// The viewer rect a multi-click gesture was anchored to.
    pub(crate) rect: egui::Rect,
    /// --test-anim: (stage, frames, last sampled tick, good samples, interp seen).
    pub(crate) anim: Option<(u32, u32, u64, u32, bool)>,
    /// --test-patchlive: (stage, frames).
    pub(crate) patchlive: Option<(u32, u32)>,
    /// --test-gizmokey: (stage, frames).
    pub(crate) gizmokey: Option<(u32, u32)>,
    /// --test-magnet: (stage, frames, new node id).
    pub(crate) magnet: Option<(u32, u32, u64)>,
    /// --test-nodeux: (stage, frames, scratch blend id, scratch grade id).
    pub(crate) nodeux: Option<(u32, u32, u64, u64)>,
    /// --test-blendnav: (stage, frames, packed pan baseline, packed zoom
    /// baseline) — f32 bits, because the gestures are asserted RELATIVE to
    /// where the view was when each one began.
    pub(crate) blendnav: Option<(u32, u32, u64, u64)>,
    /// --test-tv: (stage, frames, frames spent in the current wait).
    pub(crate) tv: Option<(u32, u32, u32)>,
    /// --test-mswipe: (stage, frames, Read B's start pos, release point).
    pub(crate) mswipe: Option<(u32, u32, egui::Pos2, egui::Pos2)>,
    /// --test-spline: (stage, frames, spline node id).
    /// (stage, frames, node, stash) — the stash carries the pre-drag
    /// bounds across the group-scale stages.
    pub(crate) spline: Option<(u32, u32, u64, [f32; 2])>,
    /// --test-cornerpin: (stage, frames, node, press point, pins at the press).
    pub(crate) cornerpin: Option<(u32, u32, u64, egui::Pos2, [[f32; 2]; 4])>,
    /// --test-gridwarp: (stage, node, press point, deadline).
    pub(crate) gridwarp: Option<(u32, u64, egui::Pos2, Instant)>,
    /// --test-warp: (stage, node, press point, deadline).
    pub(crate) warp: Option<(u32, u64, egui::Pos2, Instant)>,
    /// --test-gridwarp: the destination grid at a box gesture's press, and
    /// the pivot it turned about.
    pub(crate) gridwarp_snap: Vec<[f32; 2]>,
    pub(crate) gridwarp_pivot: egui::Pos2,
    /// --test-cornerpin: where the box's rotate knob was at its press.
    pub(crate) cornerpin_knob: egui::Pos2,
    /// --test-keyer: (stage, frames, node id).
    pub(crate) keyer: Option<(u32, u32, u64)>,
    /// --test-colourkey: (stage, frames, keyer node id).
    pub(crate) colourkey: Option<(u32, u32, u64)>,
    /// The matte the colourkey suite measured before inverting.
    pub(crate) colourkey_m: f32,
    /// --test-rgb: (stage, frames, Mult node id).
    pub(crate) rgb: Option<(u32, u32, u64)>,
    /// The triple the rgb suite banked before the gesture under test.
    pub(crate) rgb_before: [f32; 3],
    /// Where its current drag was pressed, and how wide the track was.
    /// Captured ONCE per gesture: the knob moves as the value changes, so a
    /// position recomputed each frame would chase itself and the drag would
    /// accelerate.
    pub(crate) rgb_anchor: (egui::Pos2, f32),
    /// --test-cpick: (stage, frames, Grid node id).
    pub(crate) cpick: Option<(u32, u32, u64)>,
    /// (stage, frames in stage, the Grid) and the row's settled rect.
    pub(crate) propspie: Option<(u32, u32, u64)>,
    pub(crate) propspie_at: (egui::Rect, u32),
    /// --test-viewpie: (stage, frames in it, started).
    pub(crate) viewpie: Option<(u32, u32, web_time::Instant)>,
    /// The Grid's line colour before the picker touched it.
    pub(crate) cpick_before: [f32; 4],
    /// The rect the current gesture is anchored to, and how many frames it
    /// has held still — captured once, like `test_rgb_anchor`.
    pub(crate) cpick_at: (egui::Rect, u32),
    /// Wall clock for the stage that waits on real decodes.
    pub(crate) cpick_t0: web_time::Instant,
    /// The picker's scale before the resize under test.
    pub(crate) cpick_scale: egui::Vec2,
    /// Viewer and graph zoom when the pinch went in, which it must not move.
    pub(crate) cpick_zoom: (f32, f32),
    /// --test-nodepack: (stage, frames, node under test).
    pub(crate) nodepack: Option<(u32, u32, u64)>,
    /// --test-planar: (stage, frames, planar node), its clock, and where
    /// the gizmo press landed.
    pub(crate) planar: Option<(u32, u32, u64)>,
    pub(crate) planar_t0: web_time::Instant,
    /// --test-planar: the stabilise CornerPin's worst corner residual.
    pub(crate) planar_stab: f64,
    pub(crate) planar_press: egui::Pos2,
    /// The gizmo stage's half of the PASS line, kept for the matte stages.
    pub(crate) planar_note: String,
    /// --test-track: (stage, frames, tracker node).
    pub(crate) track: Option<(u32, u32, u64)>,
    /// --test-channels: (stage, frames, shuffle node).
    pub(crate) channels: Option<(u32, u32, u64)>,
    pub(crate) crypto: Option<(u32, u32, u64)>,
    /// When the crypto suite entered its current stage: its waits are on
    /// the WALL CLOCK (a rig rebuild and a decode take real time; a frame
    /// count is not a duration under test).
    pub(crate) crypto_since: Option<web_time::Instant>,
    /// When the crypto suite started: the overall cap, which a per-stage
    /// retry must not keep resetting.
    pub(crate) crypto_t0: Option<web_time::Instant>,
    /// --test-curveedit: (stage, frames, curve node id).
    pub(crate) curveedit: Option<(u32, u32, u64)>,
    /// --test-noise: (stage, frames, the Noise node).
    pub(crate) noise: Option<(u32, u32, u64)>,
    /// --test-geo: (stage, frames, the 2dini node).
    pub(crate) geo: Option<(u32, u32, u64)>,
    /// --test-geotools: (stage, frames in it, when the run began).
    pub(crate) geotools: Option<(u32, u32, web_time::Instant)>,
    /// --test-geotools: the nodes it is tracking, by role.
    pub(crate) geotools_ids: std::collections::HashMap<&'static str, u64>,
    /// --test-geotools: when a wall-clock wait began.
    pub(crate) geotools_mark: Option<web_time::Instant>,
    /// --test-sdf: (stage, frames, the SDF node), its last pixel probe (and
    /// when), and the reading the stage is waiting to move away from.
    pub(crate) sdf: Option<(u32, u32, u64)>,
    pub(crate) sdf_last: Option<(Instant, [f32; 3])>,
    pub(crate) sdf_base: Option<[f32; 3]>,
    /// --test-mblur: (stage, frames, the unblurred mean in bits).
    pub(crate) mblur: Option<(u32, u32, u64)>,
    /// --test-write: (stage, frames, write node).
    pub(crate) write: Option<(u32, u32, u64)>,
    /// Frames the running track had keyed last time the suite looked —
    /// the suite waits on PROGRESS, not on a frame count, so a machine
    /// under GPU load can't time it out mid-track.
    pub(crate) track_progress: u64,
    /// When the track suite last saw a stage change or bake progress: its
    /// timeout is WALL CLOCK (a pattern frame is real work, and the
    /// unpaced test loop spins thousands of iterations a second).
    pub(crate) track_clock: web_time::Instant,
    /// --test-track's single- and multi-point worst errors, for the report.
    pub(crate) track_worst: f64,
    pub(crate) track_points_worst: f64,
    /// --test-track: the inner-graph drag in progress (press, release) and
    /// where the dragged node started.
    pub(crate) track_drag: (egui::Pos2, egui::Pos2),
    pub(crate) track_move_from: [f32; 2],
    /// --test-track: the frame-0 reference track, so the second (mid-
    /// playhead) pass can be checked against it.
    pub(crate) track_ref: Vec<(u64, f32, f32)>,
    /// --test-track: where the second pass seeds its pattern (the point the
    /// reference track reached at the frame it restarts from).
    pub(crate) track_seed: [f32; 2],
    /// --test-zoom: (stage, frames, panel width measured at zoom 1).
    pub(crate) zoom: Option<(u32, u32, f32)>,
    /// The tail of --test-zoom: per-panel width through real input.
    pub(crate) panelw: Option<PanelWTest>,
    /// --test-cachemeter: (stage, frames).
    pub(crate) cachemeter: Option<(u32, u32)>,
    /// --test-precache: (stage, when the stage began).
    pub(crate) precache: Option<(u32, Instant)>,
    /// --test-bin: (stage, frames, nodes before).
    pub(crate) bin: Option<(u32, u32, usize)>,
    /// --test-thumbs: (stage, frames).
    pub(crate) thumbs: Option<(u32, u32)>,
    /// --test-dpx: (stage, frames, node).
    pub(crate) dpx: Option<(u32, u32, u64)>,
    pub(crate) still: Option<(u32, u32, u64)>,
    pub(crate) scrub: Option<(u32, u32, usize)>,
    pub(crate) scrubperf: Option<(u32, u64, u64)>,
    /// --test-scrubfid: scrub FIDELITY — the hand's target against the tick
    /// on screen, per displayed frame (`drive_test_scrubfid`).
    pub(crate) scrubfid: Option<Box<ScrubFid>>,
    /// --test-proxy: a proxy toggle re-renders everything at the new size,
    /// effects included, with no further input (`drive_test_proxy`).
    pub(crate) proxy: Option<Box<ProxyTest>>,
    /// The crop-into-Layer workflow under load (`stress.rs`).
    pub(crate) stress: Option<Box<stress::StressTest>>,
    pub(crate) grow: Option<(u32, u64, u64)>,
    /// Wall clock for --test-scrub. Under test the event loop spins far
    /// faster than real time, so a frame COUNT is not a duration — and
    /// this suite is waiting on a 24fps transport and real decodes.
    pub(crate) scrub_t0: Option<web_time::Instant>,
    /// Wall clock for --test-seek: stage 5 waits on 24 frames at 24fps,
    /// which is a SECOND, and under test the loop spins far faster than
    /// real time — a frame budget there times out in ~0.2s.
    pub(crate) seek_t0: Option<web_time::Instant>,
    pub(crate) rc_t0: Option<web_time::Instant>,
    pub(crate) grow_t0: Option<web_time::Instant>,
    /// --test-plugin: (stage, frames, node).
    pub(crate) plugin: Option<(u32, u32, u64)>,
    /// --test-shader: (stage, frames, Shader node id).
    pub(crate) shader: Option<(u32, u32, u64)>,
    /// --test-cost: (stage, frames, slow node id).
    pub(crate) cost: Option<(u32, u32, u64)>,
    /// --test-blender: stage, wall clocks, the node, the fake bridge.
    pub(crate) blender: Option<TestBlender>,
    /// --test-genai: its stages, clocks and stand-in fal.
    pub(crate) genai: Option<genaiui::TestGenai>,
    pub(crate) corridor: Option<corridorkey::TestCorridor>,
    pub(crate) seedvr: Option<seedvr::TestSeedVr>,
    pub(crate) depth: Option<depth::TestDepth>,
    pub(crate) ltxhdr: Option<ltxhdr::TestLtxHdr>,
    pub(crate) upscale: Option<upscaleui::TestUpscale>,
    pub(crate) hdrcnn: Option<hdrcnnui::TestHdrCnn>,
    pub(crate) flow: Option<flowui::TestFlow>,
    pub(crate) stab: Option<stabilizeui::TestStab>,
    /// --test-stabilize: the lock-off residual it measured.
    pub(crate) stab_lock: f64,
    pub(crate) camtrack: Option<camtrackui::TestCamTrack>,
    pub(crate) sammie: Option<sammie::TestSammie>,
    /// --test-focus: (stage, frames, Defocus node id).
    pub(crate) focus: Option<(u32, u32, u64)>,
    /// --test-focus: wall clock for the waits on a view switch.
    pub(crate) focus_t: Option<web_time::Instant>,
    /// --test-viewswitch: (stage, frames).
    pub(crate) viewswitch: Option<(u32, u32)>,
    /// --test-keymap: (stage, frames).
    pub(crate) keymap: Option<(u32, u32)>,
    /// --test-region: (stage, frames).
    pub(crate) region: Option<(u32, u32)>,
    /// --test-harp: (stage, stage start, scratch: rig builds, delivered,
    /// displayed at the window mark; rig builds + displayed at the last view
    /// flip; probe values as f32 bits).
    pub(crate) harp: Option<(u32, Instant, [u64; 8])>,
    /// --test-layer: (stage, frames, node).
    pub(crate) layer: Option<(u32, u32, u64)>,
    /// --test-layerpick: stage, frames in stage, and the two Transforms
    /// (foreground, background) a ctrl-click should land on.
    pub(crate) layerpick: Option<(u32, u32, [u64; 2])>,
    /// --test-roto: (stage, frames in stage, the Spline).
    pub(crate) roto: Option<(u32, u32, u64)>,
    /// --test-roto: a gesture's press and release points.
    pub(crate) roto_drag: (egui::Pos2, egui::Pos2),
    /// `--test-roto` has been through the dive stage.
    pub(crate) roto_dived: bool,
    pub(crate) roto_clock: Option<web_time::Instant>,
    /// --test-eyedrop: (stage, frames in stage, the Blank node).
    pub(crate) eyedrop: Option<(u32, u32, u64)>,
    pub(crate) text: Option<textui::TestText>,
    /// --test-viewlock: (stage, frames in stage, [subwob instance, tail], the
    /// readout before the inner edit).
    pub(crate) viewlock: Option<(u32, u32, [u64; 2], [f32; 4])>,
    /// --test-dual: (stage, frames in stage, the Grade the right pane views).
    pub(crate) dual: Option<(u32, u32, u64)>,
    /// `--test-record`: stage and when it began.
    pub(crate) record: Option<(u32, web_time::Instant)>,
    /// --test-compose: (stage, frames in stage), the press point, the
    /// pointer last frame, and the dragged control's rect at the press.
    pub(crate) compose: Option<(u32, u32)>,
    pub(crate) compose_at: (egui::Pos2, egui::Pos2, egui::Rect),
    /// The fixture's four rgb controls, at [0,0] [5,0] [0,3] [5,3].
    pub(crate) compose_ids: [u64; 4],
    /// --test-subwob: (stage, frames in stage); then the wall clock of the
    /// last edit, the picture's mean before the collapse, and the ids
    /// (Read, Grade, Blur, the subwob node).
    pub(crate) subwob: Option<(u32, u32, web_time::Instant)>,
    pub(crate) subwob_at: (web_time::Instant, [f32; 3], [u64; 4]),
    /// --test-click: the transform's forward affine before the ctrl-drag
    /// pivot phase, which must leave it unchanged.
    pub(crate) pivot_aff: [f32; 6],
    /// --test-rendercache: (stage, frames, displayed marker).
    pub(crate) rc: Option<(u32, u32, u64)>,
}
