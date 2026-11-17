//! `--test-setup`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-setup driver: save -> wipe -> load -> verify graph + sequences
    /// come back.
    pub(crate) fn drive_test_setup(&mut self) {
        let Some((stage, mut frames, nodes, loads)) = self.test.setup else { return };
        frames += 1;
        if frames > 1800 {
            println!("SETUP TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let path = std::env::temp_dir().join("wobble_test_setup.wbl");
        match stage {
            0 if frames >= 3 => {
                let nodes = self.egraph.nodes.len();
                let loads = self.loaded.len();
                if let Err(e) = self.save_setup(&path) {
                    println!("SETUP TEST FAIL: save: {e:#}");
                    std::process::exit(1);
                }
                // Wipe everything the setup should restore.
                self.egraph = EditorGraph::default();
                self.loaded.clear();
                self.playback_dirty = true;
                self.fps = 99.0;
                self.test.setup = Some((1, frames, nodes, loads));
                return;
            }
            1 => {
                if let Err(e) = self.load_setup(&path) {
                    println!("SETUP TEST FAIL: load: {e:#}");
                    std::process::exit(1);
                }
                let n = self.egraph.nodes.len();
                if n != nodes || (self.fps - 24.0).abs() > 0.01 {
                    println!("SETUP TEST FAIL: roundtrip nodes {n}/{nodes} fps {}", self.fps);
                    std::process::exit(1);
                }
                self.test.setup = Some((2, frames, nodes, loads));
                return;
            }
            2 => {
                if self.loaded.len() == loads && (loads == 0 || self.playback.is_some()) {
                    // Now the other half of the contract: a history jump
                    // (undo/redo) restores nodes whose loads went with them,
                    // so it has to re-adopt the footage as well.
                    let g = self.egraph.clone();
                    self.loaded.clear();
                    self.load_errors.clear();
                    self.apply_history(g);
                    self.test.setup = Some((3, frames, nodes, loads));
                    return;
                }
            }
            3 => {
                let back = self.loaded.len();
                if back != loads {
                    println!(
                        "SETUP TEST FAIL: history jump left {back} of {loads} sequences loaded"
                    );
                    std::process::exit(1);
                }
                // "new": over an untouched comp it starts over at once; over
                // an edit it asks first, and discarding leaves a fresh,
                // file-less comp that counts as saved.
                if self.comp_unsaved() {
                    println!("SETUP TEST FAIL: a just-opened comp reads as unsaved");
                    std::process::exit(1);
                }
                if let Some(n) = self.egraph.nodes.values_mut().next() {
                    n.pos.x += 40.0;
                }
                // Frames for the project stage, from the plate this suite opened.
                let plate: Vec<std::path::PathBuf> =
                    self.loaded.values().next().map(|l| l.seq.frames.iter().take(3).cloned().collect()).unwrap_or_default();
                self.new_setup_action();
                if !self.new_setup_ask || self.current_setup.is_none() {
                    println!("SETUP TEST FAIL: new over an edit did not ask first");
                    std::process::exit(1);
                }
                self.new_setup();
                let fresh = EditorGraph::default_graph("", "", None).nodes.len();
                if self.new_setup_ask
                    || self.current_setup.is_some()
                    || self.egraph.nodes.len() != fresh
                    || self.comp_unsaved()
                {
                    println!(
                        "SETUP TEST FAIL: new left {} nodes (want {fresh}), file {:?}",
                        self.egraph.nodes.len(),
                        self.current_setup
                    );
                    std::process::exit(1);
                }
                // Projects: a managed root of one project, its shot holding
                // the plate. A Project Load names it, a comp saved into the
                // shot keeps the names, and opening the comp finds it again.
                let root = std::env::temp_dir().join(format!("wobble_test_projects_{}", std::process::id()));
                let _ = std::fs::remove_dir_all(&root);
                let proj = root.join("live").join("spooky_beach");
                let shot = proj.join("sequences").join("dev").join("spb_010");
                let made = std::fs::create_dir_all(proj.join(projects::MANIFEST_DIR))
                    .and_then(|_| std::fs::create_dir_all(shot.join("scripts")))
                    .and_then(|_| std::fs::create_dir_all(shot.join("plates").join("bg")))
                    .and_then(|_| std::fs::write(proj.join(projects::MANIFEST_DIR).join(projects::MANIFEST_FILE), PROJECT))
                    .and_then(|_| {
                        plate.iter().enumerate().try_for_each(|(i, f)| {
                            std::fs::copy(f, shot.join("plates").join("bg").join(format!("bg.{i:04}.exr"))).map(|_| ())
                        })
                    });
                if let Err(e) = made.as_ref().map_err(|e| e.to_string()).and_then(|_| {
                    if plate.len() == 3 { Ok(()) } else { Err(format!("plate gave {} frames", plate.len())) }
                }) {
                    println!("SETUP TEST FAIL: project fixture: {e}");
                    std::process::exit(1);
                }
                // SAFETY: the suite is single-threaded here; nothing else reads it.
                unsafe { std::env::set_var("WOBBLE_PROJECTS_ROOT", &root) };
                projects::forget();
                let n = editor::ENode::new(NodeKind::read(self.next_read_slot, assets::PROJECT_LOAD), egui::pos2(0.0, 0.0));
                self.next_read_slot += 1;
                if !n.project_only || !matches!(&n.kind, NodeKind::Read { path, .. } if path.is_empty()) {
                    println!("SETUP TEST FAIL: a Project Load is not marked, or kept its marker path");
                    std::process::exit(1);
                }
                let id = self.egraph.next_id;
                self.egraph.next_id += 1;
                self.egraph.nodes.insert(id, n);
                // Every mode of the browser draws (Media is picked below).
                self.open_project_browser(projectui::Mode::Media(id));
                let ps = projects::projects(false);
                let Some(p) = ps.first() else {
                    println!("SETUP TEST FAIL: the fixture project was not found under {}", root.display());
                    std::process::exit(1);
                };
                let inst = p.by_names(&["dev".into(), "spb_010".into()]).flatten();
                let listed = projects::media(p, inst);
                if listed != ["plates/bg/bg.####.exr"] {
                    println!("SETUP TEST FAIL: the shot lists {listed:?}");
                    std::process::exit(1);
                }
                if let Some(n) = self.egraph.node_mut(id) {
                    n.asset = Some(projects::reference(p, inst, &listed[0]));
                }
                self.load_asset_fresh(id);
                self.test.setup = Some((4, frames, id as usize, loads));
                return;
            }
            // The Project Load has its frames: save into the shot, open it again.
            4 => {
                let id = nodes as u64;
                if let Some(e) = self.load_errors.get(&id) {
                    println!("SETUP TEST FAIL: the Project Load did not load: {e}");
                    std::process::exit(1);
                }
                if !self.loaded.get(&id).is_some_and(|l| l.seq.frames.len() == 3) {
                    self.test.setup = Some((4, frames, nodes, loads));
                    return;
                }
                self.project_browser = None;
                self.open_project_browser(projectui::Mode::Save);
                let ps = projects::projects(false);
                let p = &ps[0];
                let inst = p.by_names(&["dev".into(), "spb_010".into()]).flatten();
                let dir = p.comp_dir(inst);
                let path = dir.join(projects::new_comp_name(p, inst, "comp", &dir));
                if let Err(e) = self.save_to_project(&path) {
                    println!("SETUP TEST FAIL: save to project: {e}");
                    std::process::exit(1);
                }
                let text = std::fs::read_to_string(&path).unwrap_or_default();
                if !path.ends_with("scripts/spb_010_comp_v001.wbl")
                    || !text.contains("\"project_only\": true")
                    || !text.contains("\"spb_010\"")
                {
                    println!("SETUP TEST FAIL: the saved comp {} lacks the Project Load's names", path.display());
                    std::process::exit(1);
                }
                self.new_setup();
                self.open_project_browser(projectui::Mode::Open);
                if let Err(e) = self.load_setup(&path) {
                    println!("SETUP TEST FAIL: reopen: {e:#}");
                    std::process::exit(1);
                }
                let back = self.egraph.nodes.iter().find(|(_, n)| n.project_only).map(|(k, _)| *k);
                let Some(back) = back else {
                    println!("SETUP TEST FAIL: the reopened comp has no Project Load");
                    std::process::exit(1);
                };
                self.test.setup = Some((5, frames, back as usize, loads));
                return;
            }
            5 => {
                let id = nodes as u64;
                if let Some(e) = self.load_errors.get(&id) {
                    println!("SETUP TEST FAIL: the reopened Project Load did not load: {e}");
                    std::process::exit(1);
                }
                if !self.loaded.get(&id).is_some_and(|l| l.seq.frames.len() == 3) {
                    self.test.setup = Some((5, frames, nodes, loads));
                    return;
                }
                if let Some(d) = std::env::var_os("WOBBLE_PROJECTS_ROOT") {
                    let _ = std::fs::remove_dir_all(d);
                }
                println!(
                    "SETUP TEST PASS: save/load round trip, sequences reloaded, fps restored, history jump re-adopted them, new asks over edits, a Project Load saved into a shot and opened again finds its footage by name"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test.setup = Some((stage, frames, nodes, loads));
    }
}

/// One project as organisicator publishes it: a sequence holding a shot
/// whose scripts folder is where comps go.
const PROJECT: &str = r#"{
    "format": 1, "app": "organisicator", "name": "spooky_beach", "code": "spb",
    "folders": [
        {"path": "sequences/{sequence}", "type": "sequence"},
        {"path": "sequences/{sequence}/{shot}", "type": "shot"},
        {"path": "sequences/{sequence}/{shot}/scripts", "type": "software", "exts": "nk blend wbl"},
        {"path": "sequences/{sequence}/{shot}/plates", "type": "source", "exts": "exr"}
    ],
    "instances": [
        {"id": 1, "parent": null, "decl": "sequences/{sequence}", "value": "dev", "rel": "sequences/dev"},
        {"id": 2, "parent": 1, "decl": "sequences/{sequence}/{shot}", "value": "spb_010", "rel": "sequences/dev/spb_010"}
    ]
}"#;
