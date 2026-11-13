//! The browser build's stand-in for `imfram::dock`, which is desktop-only
//! (its tabs tear off into windows of their own). In a page the panes sit
//! in one fixed layout: the graph left of the viewer, the timeline under
//! both, the properties along the bottom. Same names and shapes as the
//! parts of `imfram::dock` wobble uses, so `dockui` and the call sites in
//! `main` compile unchanged; the window-making parts are no-ops.

use std::marker::PhantomData;

use egui::Ui;
use serde::{Deserialize, Serialize};

pub trait Role: Copy + Eq + std::fmt::Debug + Serialize + serde::de::DeserializeOwned + 'static {
    fn title(self) -> &'static str;
    fn size(self) -> (u32, u32);
    fn detached() -> Self;
}

pub trait Tab: Copy + Eq + std::hash::Hash + std::fmt::Debug + Serialize + serde::de::DeserializeOwned + 'static {
    type Role: Role;
    fn all() -> &'static [Self];
    fn title(self) -> &'static str;
    fn help(self) -> &'static str {
        ""
    }
    fn home(self) -> Self::Role;
    fn strip(self) -> Option<f32> {
        None
    }
}

pub trait Panels<T: Tab> {
    fn raw_input(&mut self, _ctx: &egui::Context, _raw: &mut egui::RawInput, _tabs: &[T]) {}
    fn begin(&mut self, _ui: &mut Ui) {}
    fn bar_shown(&self) -> bool {
        true
    }
    fn tab(&mut self, ui: &mut Ui, tab: T);
    fn see_through(&self, _tab: &T) -> bool {
        false
    }
    fn scroll_bars(&self, _tab: &T) -> [bool; 2] {
        [true, true]
    }
    fn tab_bars(&self) -> bool {
        true
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    Left,
    Right,
    Top,
    Bottom,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Zone {
    Center,
    Side(Side),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Drop {
    Leaf(usize, Zone),
    Root(Side),
}

/// Which panes are open; where they sit is fixed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tree<T> {
    tabs: Vec<T>,
}

impl<T: Copy + PartialEq> Tree<T> {
    pub fn new(tabs: Vec<T>) -> Self {
        Tree { tabs }
    }
    pub fn split(&mut self, _side: Side, _frac: f32, tabs: Vec<T>) {
        self.tabs.extend(tabs);
    }
    pub fn find(&self, tab: T) -> Option<usize> {
        self.tabs.iter().position(|t| *t == tab)
    }
    pub fn insert(&mut self, tab: T, _drop: Drop) {
        if self.find(tab).is_none() {
            self.tabs.push(tab);
        }
    }
    pub fn set_active(&mut self, _tab: T) {}
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Ask<T> {
    Detach(T),
    Open(T),
    Close(T),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Saved<T, R> {
    pub role: R,
    pub tree: Tree<T>,
    pub at: Option<[f64; 2]>,
    pub size: Option<[u32; 2]>,
    pub visible: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Layout<T, R> {
    pub windows: Vec<Saved<T, R>>,
}

/// The GPU handle the desktop dock draws detached windows with; nothing to
/// hold in a page.
#[derive(Clone)]
pub struct Gpu;

impl Gpu {
    pub fn new(_: wgpu::Instance, _: wgpu::Adapter, _: wgpu::Device, _: wgpu::Queue) -> Self {
        Gpu
    }
}

pub struct HostWin;

impl HostWin {
    pub fn is_external(&self) -> bool {
        true
    }
    pub fn request_repaint(&self) {}
}

pub struct Host<T: Tab> {
    pub role: T::Role,
    pub win: HostWin,
    pub tree: Tree<T>,
}

pub struct Docks<T: Tab> {
    pub hosts: Vec<Host<T>>,
    pub asks: Vec<Ask<T>>,
    solo: Option<T>,
    _t: PhantomData<T>,
}

impl<T: Tab> Docks<T> {
    pub fn new(_name: &str, _testing: bool) -> Self {
        Docks { hosts: Vec::new(), asks: Vec::new(), solo: None, _t: PhantomData }
    }
    pub fn add_external(
        &mut self,
        role: T::Role,
        _window: std::sync::Arc<winit::window::Window>,
        _ctx: egui::Context,
        tree: Tree<T>,
    ) -> usize {
        self.hosts.push(Host { role, win: HostWin, tree });
        self.hosts.len() - 1
    }
    pub fn is_external(&self, _i: usize) -> bool {
        true
    }
    pub fn set_clear(&mut self, _color: wgpu::Color) {}
    pub fn solo(&mut self, tab: T) {
        self.solo = Some(tab);
    }
    pub fn soloed(&self, _i: usize) -> Option<T> {
        self.solo
    }
    pub fn unsolo(&mut self, _i: usize) {
        self.solo = None;
    }
    pub fn host_of(&self, tab: T) -> Option<usize> {
        self.hosts.iter().position(|h| h.tree.find(tab).is_some())
    }
    pub fn by_window(&self, _id: winit::window::WindowId) -> Option<usize> {
        None
    }
    pub fn poll(&mut self) {}
    /// Opens and closes asked for this frame; there is nowhere to detach to.
    pub fn apply(&mut self, _el: Option<&winit::event_loop::ActiveEventLoop>, _gpu: &Gpu) {
        for ask in std::mem::take(&mut self.asks) {
            let Some(h) = self.hosts.first_mut() else { break };
            match ask {
                Ask::Open(t) => h.tree.insert(t, Drop::Root(Side::Bottom)),
                Ask::Close(t) => h.tree.tabs.retain(|x| *x != t),
                Ask::Detach(_) => {}
            }
        }
    }
    pub fn frame<P: Panels<T>>(&mut self, _i: usize, _gpu: &Gpu, _panels: &mut P, _offscreen: Option<Vec<egui::Event>>) -> Option<wgpu::Texture> {
        None
    }
    pub fn event(&mut self, _gpu: &Gpu, _i: usize, _event: &winit::event::WindowEvent) -> bool {
        false
    }
    pub fn save(&self) -> Layout<T, T::Role> {
        Layout { windows: Vec::new() }
    }
    pub fn restore(&mut self, _el: &winit::event_loop::ActiveEventLoop, _gpu: &Gpu, _layout: Layout<T, T::Role>) {}
    pub fn close_requested(&mut self, i: usize) -> bool {
        if i > 0 && i < self.hosts.len() {
            self.hosts.remove(i);
        }
        true
    }

    /// The fixed layout. `T::all()` is read in `dockui`'s order (graph,
    /// properties, animation, viewer, timeline); the animation shares the
    /// properties' place and shows when the properties are closed.
    pub fn draw_external<P: Panels<T>>(&mut self, i: usize, ui: &mut Ui, panels: &mut P) {
        panels.begin(ui);
        let Some(host) = self.hosts.get(i) else { return };
        let open = |t: T| host.tree.find(t).is_some();
        let all = T::all();
        let (nodes, props, curves, viewer, timeline) = (all[0], all[1], all[2], all[3], all[4]);
        let gap = 4.0;
        let frame = egui::Frame::NONE;
        if let Some(t) = self.solo {
            egui::CentralPanel::default().frame(frame).show(ui, |ui| panels.tab(ui, t));
            return;
        }
        let bottom_tab = if open(props) { Some(props) } else if open(curves) { Some(curves) } else { None };
        if let Some(t) = bottom_tab {
            let h = (ui.available_height() * 0.3).max(160.0);
            egui::Panel::bottom("webdock-bottom")
                .frame(frame)
                .resizable(false)
                .exact_size(h)
                .show(ui, |ui| panels.tab(ui, t));
        }
        if open(timeline) {
            egui::Panel::bottom("webdock-timeline")
                .frame(frame)
                .resizable(false)
                .exact_size(timeline.strip().unwrap_or(48.0))
                .show(ui, |ui| panels.tab(ui, timeline));
        }
        if open(nodes) {
            let w = ui.available_width() * 0.4;
            egui::Panel::left("webdock-nodes")
                .frame(frame)
                .resizable(false)
                .exact_size(w - gap)
                .show(ui, |ui| panels.tab(ui, nodes));
        }
        egui::CentralPanel::default().frame(frame).show(ui, |ui| panels.tab(ui, viewer));
    }
}

/// The grip that drags a tab on the desktop; nothing to drag here.
pub fn grip(_ui: &mut Ui) {}
