//! EXPERIMENT (branch exp/scrub-fold): wobble as a GUEST inside another
//! app's window — Scrub's (scrub/app/src/fold.rs).
//!
//! The host owns the event loop, the window and the GPU device. Wobble's
//! whole frame (its egui pass and the viewer blit) is drawn into a texture
//! of the host's device instead of a swapchain, and its input is the egui
//! events the host hands over, already moved into the guest's rectangle.
//! Nothing else about the app changes: one `Gfx`, one document, the same
//! `redraw`. A window is still made (egui-winit wants one) but it is never
//! shown and never presented to.
//!
//! The comp's output texture is on the same device, so the host can sample
//! it directly: a clip that IS a wobble comp, with no copy.

use std::sync::Arc;

use engine::GpuContext;
use winit::application::ApplicationHandler as _;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

/// The comp's output for the host: the node with this name. With none in
/// the comp, the host gets whatever the viewer shows.
pub const OUT_NODE: &str = "SCRUB OUT";

/// The output node of `g`, if it has one.
pub(crate) fn out_node(g: &crate::editor::EditorGraph) -> Option<u64> {
    g.nodes.iter().find(|(_, n)| n.ui_label == OUT_NODE).map(|(id, _)| *id)
}

/// What a hosted `Gfx` draws into and is fed from (`Gfx::hosted`).
pub(crate) struct State {
    /// The stand-in for the swapchain: (texture, view), in `config.format`.
    pub target: Option<(wgpu::Texture, wgpu::TextureView)>,
    events: Vec<egui::Event>,
    modifiers: egui::Modifiers,
    focused: bool,
    /// The guest's screen, in points (origin at its own top left).
    screen: egui::Vec2,
    ppp: f32,
    /// The cursor the last frame asked for.
    pub cursor: egui::CursorIcon,
    /// The comp has an output node (`encode_host_pane` keeps both).
    pub has_out: bool,
    /// ...and the viewer is on it, so the main output IS the output.
    pub out_is_view: bool,
}

impl State {
    /// The host's input in place of the hidden window's.
    pub fn feed(&mut self, raw: &mut egui::RawInput) {
        raw.screen_rect = Some(egui::Rect::from_min_size(egui::Pos2::ZERO, self.screen));
        if let Some(vp) = raw.viewports.get_mut(&egui::ViewportId::ROOT) {
            vp.native_pixels_per_point = Some(self.ppp);
            vp.focused = Some(self.focused);
        }
        raw.focused = self.focused;
        raw.events.clear();
        // Modifiers are sticky state in egui: said every frame.
        raw.events.push(egui::Event::ModifiersChanged(self.modifiers));
        raw.events.append(&mut self.events);
    }
}

/// The GPU device both apps share, made the way wobble needs it (its
/// features and limits). The host builds its own handles from `handles`.
pub struct Seed(GpuContext);

impl Seed {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Seed(GpuContext::new()?))
    }

    pub fn handles(&self) -> (wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue) {
        (self.0.instance.clone(), self.0.adapter.clone(), (*self.0.device).clone(), (*self.0.queue).clone())
    }
}

/// Wobble, hosted.
pub struct Hosted {
    app: crate::App,
}

impl Hosted {
    pub fn new(el: &ActiveEventLoop, seed: Seed) -> anyhow::Result<Self> {
        let mut opts = crate::base_options();
        // A worker's quiet start: no session, no welcome or consent windows,
        // no scripting server, no uploads. Prefs (keys, theme) still load.
        opts.worker = true;
        let attrs = Window::default_attributes()
            .with_visible(false)
            .with_active(false)
            .with_title("wobble (hosted)")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 800.0));
        let window = Arc::new(el.create_window(attrs)?);
        let mut g = crate::build_gfx_with(window, seed.0, &opts)?;
        g.hosted = Some(State {
            target: None,
            events: Vec::new(),
            modifiers: egui::Modifiers::NONE,
            focused: false,
            screen: egui::vec2(1280.0, 800.0),
            ppp: 1.0,
            cursor: egui::CursorIcon::Default,
            has_out: false,
            out_is_view: false,
        });
        Ok(Self { app: crate::App { opts, gfx: Some(g) } })
    }

    fn g(&mut self) -> &mut crate::Gfx {
        self.app.gfx.as_mut().expect("built in new")
    }

    fn gr(&self) -> &crate::Gfx {
        self.app.gfx.as_ref().expect("built in new")
    }

    /// The hidden window's own events are dropped; a window wobble made
    /// since (a torn-off tab) gets its event.
    pub fn window_event(&mut self, el: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if id == self.gr().window.id() {
            return;
        }
        self.app.window_event(el, id, event);
    }

    /// Between frames, where windows can be made: the dock's asks.
    pub fn between(&mut self, el: &ActiveEventLoop) {
        let g = self.g();
        if let Some(l) = g.dock_pending.take() {
            g.dock_switch(l, el);
        }
        if let Some(docks) = g.docks.as_mut() {
            docks.apply(Some(el), &g.ui_gpu);
            for h in &mut docks.hosts {
                if !h.win.is_external() {
                    h.win.request_repaint();
                }
            }
            docks.poll();
        }
    }

    /// The guest's size in pixels and the host's pixels per point.
    pub fn resize(&mut self, w: u32, h: u32, ppp: f32) {
        let (w, h) = (w.max(16), h.max(16));
        let g = self.g();
        let fmt = g.config.format;
        let st = g.hosted.as_mut().expect("hosted");
        st.ppp = ppp;
        st.screen = egui::vec2(w as f32 / ppp, h as f32 / ppp);
        if st.target.is_some() && (g.config.width, g.config.height) == (w, h) {
            return;
        }
        let tex = g.ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hosted wobble"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        st.target = Some((tex, view));
        g.config.width = w;
        g.config.height = h;
    }

    /// This frame's input: egui events in the guest's own points.
    pub fn input(&mut self, events: Vec<egui::Event>, modifiers: egui::Modifiers, focused: bool) {
        let st = self.g().hosted.as_mut().expect("hosted");
        st.events.extend(events);
        st.modifiers = modifiers;
        st.focused = focused;
    }

    /// One wobble frame: its UI pass and its render, into the texture.
    pub fn frame(&mut self) {
        if let Err(e) = self.g().redraw() {
            log::error!("hosted wobble frame failed: {e:#}");
        }
    }

    /// The whole wobble window as a texture (gamma-encoded, `format`).
    pub fn ui_view(&self) -> Option<wgpu::TextureView> {
        self.gr().hosted.as_ref().and_then(|h| h.target.as_ref().map(|t| t.1.clone()))
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.gr().config.format
    }

    pub fn cursor(&self) -> egui::CursorIcon {
        self.gr().hosted.as_ref().map_or(egui::CursorIcon::Default, |h| h.cursor)
    }

    /// The comp's output: its OUTPUT NODE (`OUT_NODE`), scene-linear, in
    /// the comp's working space, and its size in pixels. A comp with no
    /// output node gives what its viewer shows.
    pub fn output(&self) -> Option<(wgpu::TextureView, (u32, u32))> {
        let g = self.gr();
        let st = g.hosted.as_ref()?;
        if st.has_out && !st.out_is_view {
            // Its own render; until the first one lands, nothing (the host
            // keeps its picture) rather than the viewer's node.
            let o = g.host_pane.as_ref().filter(|o| o.shown)?;
            return Some((o.comp.output_view()?.clone(), o.comp.output_size()?));
        }
        let c = &g.comp;
        Some((c.output_view()?.clone(), c.output_size()?))
    }

    /// Whether the comp has an output node.
    pub fn has_out(&self) -> bool {
        self.gr().hosted.as_ref().is_some_and(|h| h.has_out)
    }

    /// The comp's own panel — the control nodes at its top level, arranged
    /// with the panel composer exactly as a subwob's are — drawn into the
    /// HOST's ui with wobble's panel widgets. Edits land in the comp.
    /// False: the comp has no controls (nothing was drawn).
    pub fn controls_ui(&mut self, ui: &mut egui::Ui) -> bool {
        use crate::tools;
        use engine::graph::NodeKind;
        let g = self.g();
        if !g.dive.is_empty() || tools::params_of(&g.egraph).is_empty() {
            return false;
        }
        let body = g.egraph.clone();
        let (tw, th) = (g.panel_tiles.0.max(1), g.panel_tiles.1.max(1));
        let (cols, rows) = (tools::cols(tw), tools::rows(th));
        // Every knob's value by its parameter name, as a subwob instance
        // holds them.
        let knobs = |b: &crate::editor::EditorGraph| {
            let mut out = std::collections::BTreeMap::new();
            for (id, n) in tools::ordered_controls(b) {
                let (Some(k), NodeKind::Custom { params, .. }) = (tools::control_of(&n.kind), &n.kind) else { continue };
                for (i, lane) in k.lanes().iter().enumerate() {
                    out.insert(tools::lane_param(id, k, i), (id, lane.to_string(), params.get(*lane).copied().unwrap_or(0.0)));
                }
            }
            out
        };
        let at = knobs(&body);
        let mut vals: std::collections::BTreeMap<String, f32> = at.iter().map(|(k, v)| (k.clone(), v.2)).collect();
        let mut ungang = std::collections::BTreeSet::new();
        let mut anim = std::collections::BTreeMap::new();
        let dummy = NodeKind::Custom { type_id: String::new(), params: Default::default() };
        let mut cx = crate::widgets::RowCtx { node: 0, anim: &mut anim, ekind: &dummy, t: 0, hud_live: false, driven: Vec::new() };
        crate::tool_panel_ui(ui, &mut cx, &body, cols, rows, &mut vals, &mut ungang, body.working, 1.0, None);
        for (name, v) in vals {
            let Some((id, lane, was)) = at.get(&name) else { continue };
            if *was == v {
                continue;
            }
            if let Some(NodeKind::Custom { params, .. }) = g.egraph.node_mut(*id).map(|n| &mut n.kind) {
                params.insert(lane.clone(), v);
            }
        }
        true
    }

    /// The comp's working space, as its JSON (Scrub's colour types are
    /// wobble's, copied: the same JSON).
    pub fn working(&self) -> serde_json::Value {
        serde_json::to_value(self.gr().egraph.working).unwrap_or(serde_json::Value::Null)
    }

    /// The whole properties panels the comp publishes (`host_panels`),
    /// drawn into the HOST's ui by wobble's own properties rack: every node
    /// in its usual form. False: none published (nothing was drawn).
    pub fn node_panels_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let g = self.g();
        if !g.dive.is_empty() {
            return false;
        }
        let live: Vec<u64> = g.egraph.host_panels.iter().copied().filter(|id| g.egraph.nodes.contains_key(id)).collect();
        if live.is_empty() {
            return false;
        }
        // The rack shows `prop_order`: lend it the published list.
        let keep = std::mem::replace(&mut g.prop.order, live.clone());
        g.properties_ui(ui, false);
        let now = std::mem::replace(&mut g.prop.order, keep);
        // A reorder in the rack is kept; a panel can only be unpublished
        // in wobble's composer.
        let mut order: Vec<u64> = now.into_iter().filter(|id| live.contains(id)).collect();
        for id in live {
            if !order.contains(&id) {
                order.push(id);
            }
        }
        g.egraph.host_panels = order;
        true
    }

    /// Wobble's own transport is running.
    pub fn playing(&self) -> bool {
        self.gr().playing
    }

    /// The comp frame on screen.
    pub fn tick(&self) -> u64 {
        self.gr().shown_tick
    }

    pub fn seek(&mut self, tick: u64) {
        self.g().seek_to(tick);
    }

    /// A scripting-API call (api/src/lib.rs), answered in process.
    pub fn call(&mut self, method: &str, params: serde_json::Value) -> Result<serde_json::Value, String> {
        self.g().api_call(method, &params).map_err(|e| e.message)
    }
}
