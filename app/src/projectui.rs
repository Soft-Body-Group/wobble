//! The project browser: projects, their sequences and shots, and in each
//! shot only what wobble works with — its comps (to open, or a place to
//! save one) or the footage a Load can read. No folders, no paths: where
//! a shot keeps its comps is the project's anatomy's business
//! (`projects::Project::comp_dir`), and where the project is mounted is
//! this machine's (`projects::managed_root`).

use std::collections::HashMap;
use std::path::PathBuf;

use crate::assets::{self, AssetRef};
use crate::projects::{self, Project};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Footage for this Load.
    Media(u64),
    Open,
    Save,
    /// A plain Load's footage, copied into the shot picked.
    Ingest(u64),
}

pub enum Act {
    None,
    Close,
    Open(PathBuf),
    Save(PathBuf),
    Media(u64, AssetRef),
    /// Copy this Load's footage into this plates folder.
    Ingest(u64, PathBuf),
}

/// (project index, level id; None = the project itself)
type Level = (usize, Option<i64>);

pub struct Browser {
    pub mode: Mode,
    projects: Vec<Project>,
    root: Option<PathBuf>,
    root_edit: String,
    filter: String,
    /// What each opened level holds: comps (Open/Save) or footage (Media).
    listed: HashMap<Level, Vec<(String, String)>>,
    /// Where to open the tree: the project and level names of the open comp
    /// or of the Load's footage.
    here: Option<(String, Vec<String>)>,
    /// Save: the level picked, the task word and the file name.
    chosen: Option<Level>,
    task: String,
    name: String,
    pub error: Option<String>,
}

impl Browser {
    pub fn new(mode: Mode, here: Option<&AssetRef>) -> Self {
        let root = projects::managed_root();
        let mut b = Browser {
            mode,
            projects: projects::projects(false),
            root_edit: root.as_ref().map(|r| r.to_string_lossy().into_owned()).unwrap_or_default(),
            root,
            filter: String::new(),
            listed: HashMap::new(),
            here: here.map(|r| (r.project.clone(), r.shot.clone())),
            chosen: None,
            task: "comp".into(),
            name: String::new(),
            error: None,
        };
        // Saving: start in the open comp's shot, named as its next version
        // would be there.
        if mode == Mode::Save {
            if let Some((pn, names)) = b.here.clone() {
                if let Some(pi) = b.projects.iter().position(|p| p.name == pn) {
                    if let Some(inst) = b.projects[pi].by_names(&names) {
                        b.choose((pi, inst.map(|i| i.id)));
                    }
                }
            }
        }
        b
    }

    fn refresh(&mut self) {
        self.root = projects::managed_root();
        self.projects = projects::projects(true);
        self.listed.clear();
        self.chosen = None;
    }

    fn choose(&mut self, l: Level) {
        self.chosen = Some(l);
        let p = &self.projects[l.0];
        let inst = l.1.and_then(|id| p.instance(id));
        self.name = projects::new_comp_name(p, inst, &self.task, &p.comp_dir(inst));
    }

    /// What a level holds, as (label, value): comp names and paths, or
    /// footage labels and places under the level.
    fn list(&mut self, l: Level) -> Vec<(String, String)> {
        let mode = self.mode;
        let projects = &self.projects;
        self.listed
            .entry(l)
            .or_insert_with(|| {
                let p = &projects[l.0];
                let inst = l.1.and_then(|id| p.instance(id));
                match mode {
                    Mode::Media(_) => {
                        projects::media(p, inst).into_iter().map(|rel| (assets::asset_label(&rel), rel)).collect()
                    }
                    _ => projects::comps(&p.comp_dir(inst))
                        .into_iter()
                        .map(|c| {
                            let n = c.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                            (n, c.to_string_lossy().into_owned())
                        })
                        .collect(),
                }
            })
            .clone()
    }

    pub fn ui(&mut self, ctx: &egui::Context) -> Act {
        let mut act = Act::None;
        let mut open = true;
        let title = match self.mode {
            Mode::Media(_) => "load from project",
            Mode::Open => "open from project",
            Mode::Save => "save to project",
            Mode::Ingest(_) => "copy into project",
        };
        let mut refresh = false;
        egui::Window::new(title)
            .id(egui::Id::new("project-browser"))
            .open(&mut open)
            .default_size([400.0, 480.0])
            .resizable(true)
            .collapsible(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("filter");
                    ui.add(egui::TextEdit::singleline(&mut self.filter).desired_width(180.0));
                    if ui.button("refresh").on_hover_text("look for new projects, shots and files").clicked() {
                        refresh = true;
                    }
                });
                // Where projects live is set once per machine, not browsed.
                egui::CollapsingHeader::new("this machine")
                    .id_salt("project-root")
                    .default_open(self.root.is_none())
                    .show(ui, |ui| {
                        ui.weak("where the managed projects folder is on this computer (each OS has its own)");
                        ui.horizontal(|ui| {
                            ui.add(egui::TextEdit::singleline(&mut self.root_edit).desired_width(260.0));
                            if ui.button("set").clicked() {
                                match projects::set_managed_root(&self.root_edit) {
                                    Ok(()) => refresh = true,
                                    Err(e) => self.error = Some(e),
                                }
                            }
                        });
                    });
                if let Some(e) = &self.error {
                    ui.colored_label(crate::theme::error(), e);
                }
                if self.mode == Mode::Save {
                    self.save_bar(ui, &mut act);
                }
                if let Mode::Ingest(node) = self.mode {
                    ui.weak("the footage is COPIED into the shot's plates (the original stays where it is); the Load then reads the copy by project and shot");
                    let target = self.chosen.map(|l| {
                        let p = &self.projects[l.0];
                        let inst = l.1.and_then(|id| p.instance(id));
                        let names = p.names(inst);
                        (p.plates_dir(inst), if names.is_empty() { p.name.clone() } else { names.join(" › ") })
                    });
                    let label = match &target {
                        Some((_, w)) => format!("copy into {w}"),
                        None => "pick a shot below".into(),
                    };
                    if ui.add_enabled(target.is_some(), egui::Button::new(label)).clicked() {
                        if let Some((plates, _)) = target {
                            act = Act::Ingest(node, plates);
                        }
                    }
                }
                ui.separator();
                if self.projects.is_empty() {
                    ui.weak(if self.root.is_none() {
                        "no managed projects folder found — set it above"
                    } else {
                        "no projects in the managed projects folder"
                    });
                    return;
                }
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    for pi in 0..self.projects.len() {
                        let name = self.projects[pi].name.clone();
                        let here = self.here.as_ref().is_some_and(|(p, _)| *p == name);
                        egui::CollapsingHeader::new(egui::RichText::new(&name).strong())
                            .id_salt(("project", pi, &name))
                            .default_open(here || self.projects.len() == 1)
                            .show(ui, |ui| self.level_ui(ui, pi, None, &[], &mut act));
                    }
                });
            });
        if refresh {
            self.refresh();
        }
        if !open {
            return Act::Close;
        }
        act
    }

    fn save_bar(&mut self, ui: &mut egui::Ui, act: &mut Act) {
        ui.horizontal(|ui| {
            ui.label("task");
            if ui.add(egui::TextEdit::singleline(&mut self.task).desired_width(80.0)).changed() {
                if let Some(l) = self.chosen {
                    self.choose(l);
                }
            }
            ui.label("name");
            ui.add(egui::TextEdit::singleline(&mut self.name).desired_width(200.0));
        });
        let target = self.chosen.and_then(|l| {
            let p = &self.projects[l.0];
            let inst = l.1.and_then(|id| p.instance(id));
            let mut name = self.name.trim().to_string();
            if name.is_empty() {
                return None;
            }
            if !name.to_lowercase().ends_with(".wbl") {
                name.push_str(".wbl");
            }
            let names = p.names(inst);
            let where_ = if names.is_empty() { p.name.clone() } else { names.join(" › ") };
            Some((p.comp_dir(inst).join(&name), name, where_))
        });
        let label = match &target {
            Some((_, n, w)) => format!("save {n} in {w}"),
            None => "pick a shot below".into(),
        };
        if ui.add_enabled(target.is_some(), egui::Button::new(label)).clicked() {
            if let Some((path, ..)) = target {
                *act = Act::Save(path);
            }
        }
    }

    /// One level: what it holds, then its levels.
    fn level_ui(&mut self, ui: &mut egui::Ui, pi: usize, inst: Option<i64>, chain: &[String], act: &mut Act) {
        let filter = self.filter.to_lowercase();
        if let Mode::Ingest(_) = self.mode {
            let here = self.chosen == Some((pi, inst));
            let label = if chain.is_empty() { format!("into {} itself", self.projects[pi].name) } else { format!("into {}", chain.join(" › ")) };
            if ui.selectable_label(here, label).clicked() {
                self.chosen = Some((pi, inst));
            }
        } else if self.mode == Mode::Save {
            let here = self.chosen == Some((pi, inst));
            let n = self.list((pi, inst)).len();
            let label = match n {
                0 => "save here".to_string(),
                1 => "save here (1 comp)".to_string(),
                n => format!("save here ({n} comps)"),
            };
            if ui.selectable_label(here, label).clicked() {
                self.choose((pi, inst));
            }
        } else {
            let items = self.list((pi, inst));
            for (label, value) in items.iter().filter(|(l, _)| filter.is_empty() || l.to_lowercase().contains(&filter)) {
                if ui.selectable_label(false, format!("▸ {label}")).clicked() {
                    *act = match self.mode {
                        Mode::Media(node) => {
                            let p = &self.projects[pi];
                            Act::Media(node, projects::reference(p, inst.and_then(|id| p.instance(id)), value))
                        }
                        _ => Act::Open(PathBuf::from(value)),
                    };
                }
            }
        }
        let kids: Vec<(i64, String)> =
            self.projects[pi].children(inst).iter().map(|k| (k.id, k.value.clone())).collect();
        for (id, value) in kids {
            let mut sub = chain.to_vec();
            sub.push(value.clone());
            let here = self.here.as_ref().is_some_and(|(p, names)| {
                *p == self.projects[pi].name && names.starts_with(&sub)
            });
            egui::CollapsingHeader::new(&value)
                .id_salt(("project-level", pi, id))
                .default_open(here)
                .show(ui, |ui| self.level_ui(ui, pi, Some(id), &sub, act));
        }
    }
}
