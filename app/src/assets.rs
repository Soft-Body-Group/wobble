//! Project assets: a Load that names WHAT it reads — an asset in
//! organisicator's database — instead of WHERE it is on disk.
//!
//! organisicator (a companion project manager) serves a read-only API on 127.0.0.1
//! while it runs (`organisicator/src-tauri/src/api.rs`; port and token in
//! `~/.organisicator/api.json`). An [`AssetRef`] stores the asset's database
//! ID, plus the names it had when last seen (only for showing while
//! organisicator is not running). Every load asks organisicator where that
//! id is now, so a project moved, or a shot or file renamed through
//! organisicator, keeps loading: the database rewrites the paths of
//! everything it moved in place and keeps the rows.
//!
//! The path an asset resolves to is written into the Load's ordinary `path`
//! (so cache keys, bakes and the reader see a plain Load), but it is
//! DERIVED: the id is the truth, the panel never shows the path, and every
//! load asks again. organisicator not running is an error that says so.

use std::collections::HashMap;
use std::path::PathBuf;

use serde_json::Value;

/// The path the "Project Load" menu entry gives its Load, so `ENode::new`
/// can tell it from a plain one (see `ENode::project_only`).
pub const PROJECT_LOAD: &str = "\u{0}project-load";

/// A Load's asset, by id. Stored on the node (`ENode::asset`) and in the
/// setup; never a path.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct AssetRef {
    /// organisicator's id for the indexed file or sequence. The truth.
    pub id: i64,
    /// The names it had when last resolved — project, sequence / shot
    /// names outermost first, its place under the shot — refreshed on every
    /// load. Shown in the panel; never used to find it.
    #[serde(default)]
    pub project: String,
    #[serde(default)]
    pub shot: Vec<String>,
    #[serde(default)]
    pub rel: String,
}

impl AssetRef {
    pub fn asset_label(&self) -> String {
        if self.rel.is_empty() { format!("asset {}", self.id) } else { asset_label(&self.rel) }
    }

    pub fn project_label(&self) -> &str {
        if self.project.is_empty() { "?" } else { &self.project }
    }

    pub fn shot_label(&self) -> String {
        if self.shot.is_empty() { "(project)".into() } else { self.shot.join(" › ") }
    }
}

/// `plates/bg_v001/bg_v001.####.exr` → `plates › bg_v001`;
/// `plates/clip.mov` → `plates › clip.mov`.
pub fn asset_label(rel: &str) -> String {
    let s: Vec<&str> = rel.split(['/', '\\']).filter(|s| !s.is_empty()).collect();
    let Some((last, dirs)) = s.split_last() else { return String::new() };
    let stem = last.split('.').next().unwrap_or(last);
    let mut parts: Vec<&str> = dirs.to_vec();
    if dirs.last() != Some(&stem) {
        parts.push(last);
    }
    parts.join(" › ")
}

// ---- talking to organisicator ------------------------------------------------

const NOT_RUNNING: &str = "organisicator isn't running — start it to load project assets";

/// Where organisicator's API is: the discovery file it writes on start.
#[derive(Clone, Debug)]
pub struct Client {
    pub base: String,
    pub token: String,
}

impl Client {
    /// From `~/.organisicator/api.json` (`$ORGANISICATOR_API_FILE` overrides
    /// the location — suites point it at a stand-in).
    pub fn discover() -> Result<Client, String> {
        let file = match std::env::var_os("ORGANISICATOR_API_FILE") {
            Some(f) => PathBuf::from(f),
            None => {
                let home = std::env::var_os("HOME")
                    .or_else(|| std::env::var_os("USERPROFILE"))
                    .ok_or(NOT_RUNNING)?;
                PathBuf::from(home).join(".organisicator").join("api.json")
            }
        };
        let text = std::fs::read_to_string(file).map_err(|_| NOT_RUNNING.to_string())?;
        let v: Value = serde_json::from_str(&text).map_err(|_| NOT_RUNNING.to_string())?;
        let port = v.get("port").and_then(|p| p.as_u64()).ok_or(NOT_RUNNING)?;
        let token = v.get("token").and_then(|t| t.as_str()).ok_or(NOT_RUNNING)?.to_string();
        Ok(Client { base: format!("http://127.0.0.1:{port}"), token })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn get(&self, path: &str) -> Result<Value, String> {
        use std::time::Duration;
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(5)))
            .http_status_as_error(false)
            .build()
            .into();
        // A stale discovery file (organisicator quit) refuses the connection.
        let mut resp = agent
            .get(format!("{}{path}", self.base))
            .header("x-organisicator-token", &self.token)
            .call()
            .map_err(|_| NOT_RUNNING.to_string())?;
        let status = resp.status().as_u16();
        let text = resp.body_mut().with_config().limit(32 << 20).read_to_string().map_err(|e| e.to_string())?;
        let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        if status == 200 {
            Ok(v)
        } else {
            Err(v
                .get("error")
                .and_then(|e| e.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| format!("organisicator answered {status}")))
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn get(&self, _path: &str) -> Result<Value, String> {
        Err("project assets need the desktop app".into())
    }

    /// POST a JSON body. Long timeout: an ingest copies a whole plate.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn post(&self, path: &str, body: &Value) -> Result<Value, String> {
        use std::time::Duration;
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(60 * 60)))
            .http_status_as_error(false)
            .build()
            .into();
        let mut resp = agent
            .post(format!("{}{path}", self.base))
            .header("x-organisicator-token", &self.token)
            .header("content-type", "application/json")
            .send(body.to_string())
            .map_err(|_| NOT_RUNNING.to_string())?;
        let status = resp.status().as_u16();
        let text = resp.body_mut().with_config().limit(1 << 20).read_to_string().map_err(|e| e.to_string())?;
        let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        if status == 200 {
            Ok(v)
        } else {
            Err(v
                .get("error")
                .and_then(|e| e.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| format!("organisicator answered {status}")))
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn post(&self, _path: &str, _body: &Value) -> Result<Value, String> {
        Err("project assets need the desktop app".into())
    }
}

/// Where an asset is now, and what it is called now.
#[derive(Clone, Debug, PartialEq)]
pub struct Resolved {
    pub path: String,
    /// The reference with its names refreshed (a renamed shot shows its
    /// new name).
    pub named: AssetRef,
}

/// Asks organisicator where `r` is now.
pub fn resolve(client: &Client, r: &AssetRef) -> Result<Resolved, String> {
    let v = client.get(&format!("/v1/assets/{}", r.id))?;
    let path = v.get("path").and_then(|p| p.as_str()).unwrap_or_default().to_string();
    let named = AssetRef {
        id: r.id,
        project: v["project"]["name"].as_str().map(str::to_string).unwrap_or_else(|| r.project.clone()),
        shot: v["shot"]
            .as_array()
            .map(|a| a.iter().filter_map(|s| s["value"].as_str().map(str::to_string)).collect())
            .unwrap_or_else(|| r.shot.clone()),
        rel: v.get("rel").and_then(|x| x.as_str()).map(str::to_string).unwrap_or_else(|| r.rel.clone()),
    };
    let gone = v.get("missing").and_then(|m| m.as_bool()).unwrap_or(false)
        || !v.get("exists").and_then(|m| m.as_bool()).unwrap_or(true);
    if path.is_empty() || gone {
        return Err(format!(
            "{} is gone from {} / {} (deleted, or its drive is offline)",
            named.asset_label(),
            named.project_label(),
            named.shot_label()
        ));
    }
    Ok(Resolved { path, named })
}

/// Where `r` is now. An id asks organisicator (and only it: no fallback).
/// A reference with NO id is a by-name one from the project browser
/// (`crate::projectui`): its names are looked up in the projects under this
/// machine's managed root (`crate::projects`), which is how a comp made on
/// one OS finds its footage on another, organisicator running or not.
pub fn resolve_now(r: &AssetRef) -> Result<Resolved, String> {
    let by_names = || crate::projects::resolve(r).map(|path| Resolved { path, named: r.clone() });
    if r.id == 0 {
        return by_names();
    }
    // organisicator first: it follows renames and moves. When it cannot
    // answer here (not running, or its index holds the path another machine
    // indexed), the names it gave last time still find the file under this
    // machine's managed root — never a stored path.
    choose(r, Client::discover().and_then(|c| resolve(&c, r)), by_names)
}

/// Whether organisicator's answer for `saved.id` is plausibly the asset the
/// Load saved. Ids are each machine's own database rows, so a comp saved
/// on the Mac asking Windows for "asset 42" may get some other file: when
/// the project name AND the place under the shot both differ, it isn't the
/// same asset (one rename through organisicator changes one or the other,
/// not both). Nothing saved to compare with: trust the id.
pub fn same_asset(saved: &AssetRef, now: &AssetRef) -> bool {
    saved.project.is_empty() || saved.rel.is_empty() || saved.project == now.project || saved.rel == now.rel
}

/// organisicator's answer `by_id`, or the by-names lookup when it has none
/// or its id names a different asset (see [`same_asset`]).
fn choose(
    r: &AssetRef,
    by_id: Result<Resolved, String>,
    by_names: impl FnOnce() -> Result<Resolved, String>,
) -> Result<Resolved, String> {
    match by_id {
        Ok(x) if same_asset(r, &x.named) => Ok(x),
        Ok(x) => by_names().map_err(|e| {
            format!(
                "organisicator's asset {} here is {} / {}, not {} / {} (the comp was saved on another computer?) — and by name: {e}",
                r.id,
                x.named.project_label(),
                x.named.asset_label(),
                r.project_label(),
                r.asset_label()
            )
        }),
        Err(e) if r.rel.is_empty() || r.project.is_empty() => Err(e),
        Err(e) => by_names().map_err(|_| e),
    }
}

/// organisicator's asset id for a file or `####` sequence written in place
/// inside one of its projects (it indexes the folder where it is: POST
/// `/v1/register`). Errors when organisicator isn't running or the path is
/// outside every project it knows.
pub fn register_id(path: &str) -> Result<i64, String> {
    let v = Client::discover()?.post("/v1/register", &serde_json::json!({ "path": path }))?;
    v.get("id").and_then(|i| i.as_i64()).ok_or_else(|| format!("organisicator has no asset for {path}"))
}

/// organisicator's id for `project`'s project (matched by where it is on
/// this machine) and for the level at `inst_rel` in it.
pub fn api_ids(client: &Client, root: &std::path::Path, inst_rel: Option<&str>) -> Option<(i64, Option<i64>)> {
    let canon = |p: &std::path::Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    let want = canon(root);
    let pid = client
        .get("/v1/projects")
        .ok()?
        .as_array()?
        .iter()
        .find(|p| p["root"].as_str().is_some_and(|r| canon(std::path::Path::new(r)) == want))?["id"]
        .as_i64()?;
    let Some(rel) = inst_rel else { return Some((pid, None)) };
    let norm = |s: &str| s.replace('\\', "/").trim_matches('/').to_string();
    let v = client.get(&format!("/v1/projects/{pid}")).ok()?;
    let iid = v["instances"].as_array()?.iter().find(|i| i["rel"].as_str().map(norm) == Some(norm(rel)))?["id"].as_i64()?;
    Some((pid, Some(iid)))
}

/// The id organisicator's index holds for `rel` under a level, if indexed.
pub fn api_asset_id(client: &Client, pid: i64, inst: Option<i64>, rel: &str) -> Option<i64> {
    let q = inst.map(|i| format!("?instance={i}")).unwrap_or_default();
    let v = client.get(&format!("/v1/projects/{pid}/assets{q}")).ok()?;
    v["assets"].as_array()?.iter().find(|a| a["rel"].as_str() == Some(rel))?["id"].as_i64()
}

// ---- ingest: a plain Load's footage into a shot ---------------------------------

/// Where an ingest puts a plate: a project, a shot in it (None = the
/// project's own folders) and a folder under that.
#[derive(Clone, Debug, PartialEq)]
pub struct Dest {
    pub project: i64,
    pub project_name: String,
    pub instance: Option<i64>,
    pub shot: Vec<String>,
    pub folder: String,
}

/// Asks organisicator to COPY `src` (a Load's path: a sequence pattern, a
/// movie, a still or a folder) into `dest` and index it. Blocking — run it
/// off the UI thread. Returns the reference to the copy.
pub fn ingest(client: &Client, src: &str, dest: &Dest) -> Result<AssetRef, String> {
    let v = client.post(
        "/v1/ingest",
        &serde_json::json!({
            "project": dest.project,
            "instance": dest.instance,
            "folder": dest.folder,
            "src": src,
        }),
    )?;
    let id = v.get("id").and_then(|i| i.as_i64()).ok_or("organisicator gave no asset id")?;
    Ok(AssetRef { id, project: dest.project_name.clone(), shot: dest.shot.clone(), rel: String::new() })
}

/// The folder an ingest offers first, from what the footage is.
pub const DEFAULT_FOLDER: &str = "plates";

// ---- where a comp is collected ----------------------------------------------------------------

#[derive(Clone, Debug)]
struct Proj {
    id: i64,
    name: String,
}

#[derive(Clone, Debug)]
struct Inst {
    id: i64,
    parent: Option<i64>,
    value: String,
}

/// Where a whole comp is COLLECTED (`collect.rs`): a project and a shot in
/// it from organisicator's tree, and the folder under the shot its footage
/// goes in. (Picking FOOTAGE for a Load is not here: that is an addon's
/// picker, `addons::PickerDef` — the organisicator addon's media browser.)
pub struct Picker {
    client: Result<Client, String>,
    projects: Result<Vec<Proj>, String>,
    trees: HashMap<i64, Result<Vec<Inst>, String>>,
    pub filter: String,
    /// The folder under the shot, and the shot picked so far.
    folder: String,
    chosen: Option<Dest>,
}

impl Picker {
    /// The shot a comp is collected into; the folder is where its footage goes.
    pub fn for_collect() -> Self {
        let mut p = Picker {
            client: Err(String::new()),
            projects: Ok(Vec::new()),
            trees: HashMap::new(),
            filter: String::new(),
            folder: DEFAULT_FOLDER.to_string(),
            chosen: None,
        };
        p.refresh();
        p
    }

    pub fn refresh(&mut self) {
        self.client = Client::discover();
        self.trees.clear();
        self.projects = self.client.clone().and_then(|c| {
            let v = c.get("/v1/projects")?;
            Ok(v.as_array()
                .map(|a| {
                    a.iter()
                        .filter(|p| p["online"].as_bool().unwrap_or(true))
                        .filter_map(|p| Some(Proj { id: p["id"].as_i64()?, name: p["name"].as_str()?.to_string() }))
                        .collect()
                })
                .unwrap_or_default())
        });
    }

    fn tree(&mut self, pid: i64) -> Result<Vec<Inst>, String> {
        let client = self.client.clone();
        self.trees
            .entry(pid)
            .or_insert_with(|| {
                let v = client?.get(&format!("/v1/projects/{pid}"))?;
                Ok(v["instances"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|i| {
                                Some(Inst {
                                    id: i["id"].as_i64()?,
                                    parent: i["parent"].as_i64(),
                                    value: i["value"].as_str()?.to_string(),
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default())
            })
            .clone()
    }

    /// Draws the chooser; returns the destination once it is confirmed.
    /// `open` goes false when the window is closed.
    pub fn ui(&mut self, ctx: &egui::Context, open: &mut bool) -> Option<Dest> {
        let mut refresh = false;
        let mut go = false;
        egui::Window::new("collect into…")
            .id(egui::Id::new("collect-picker"))
            .open(open)
            .default_size([380.0, 460.0])
            .resizable(true)
            .collapsible(false)
            .show(ctx, |ui| {
                ui.weak("pick the shot the comp is collected into; its footage goes in this folder of the shot");
                ui.horizontal(|ui| {
                    ui.label("folder");
                    ui.add(egui::TextEdit::singleline(&mut self.folder).desired_width(140.0));
                    let label = match &self.chosen {
                        Some(d) => format!("use {} / {}", d.project_name, if d.shot.is_empty() { "(project)".into() } else { d.shot.join(" › ") }),
                        None => "pick a shot below".into(),
                    };
                    if ui.add_enabled(self.chosen.is_some() && !self.folder.trim().is_empty(), egui::Button::new(label)).clicked() {
                        go = true;
                    }
                });
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("filter");
                    ui.add(egui::TextEdit::singleline(&mut self.filter).desired_width(180.0));
                    if ui.button("refresh").on_hover_text("ask organisicator again").clicked() {
                        refresh = true;
                    }
                });
                let projects = match &self.projects {
                    Ok(p) => p.clone(),
                    Err(e) => {
                        ui.colored_label(crate::theme::error(), e);
                        return;
                    }
                };
                if projects.is_empty() {
                    ui.weak("organisicator has no projects online");
                    return;
                }
                ui.separator();
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    for p in &projects {
                        egui::CollapsingHeader::new(egui::RichText::new(&p.name).strong())
                            .id_salt(("collect-proj", p.id))
                            .show(ui, |ui| match self.tree(p.id) {
                                Err(e) => {
                                    ui.colored_label(crate::theme::error(), e);
                                }
                                Ok(tree) => self.level_ui(ui, p, &tree, None, &[]),
                            });
                    }
                });
            });
        if refresh {
            self.refresh();
        }
        if go {
            if let Some(d) = &self.chosen {
                let mut d = d.clone();
                d.folder = self.folder.trim().trim_matches('/').to_string();
                return Some(d);
            }
        }
        None
    }

    /// One level of the tree: "put it here", then its children.
    fn level_ui(&mut self, ui: &mut egui::Ui, p: &Proj, tree: &[Inst], inst: Option<i64>, chain: &[String]) {
        let filter = self.filter.to_lowercase();
        let here = self.chosen.as_ref().is_some_and(|d| d.project == p.id && d.instance == inst);
        let label = if chain.is_empty() { format!("put it in {} itself", p.name) } else { format!("put it in {}", chain.join(" › ")) };
        if (filter.is_empty() || label.to_lowercase().contains(&filter)) && ui.selectable_label(here, label).clicked() {
            self.chosen = Some(Dest {
                project: p.id,
                project_name: p.name.clone(),
                instance: inst,
                shot: chain.to_vec(),
                folder: String::new(),
            });
        }
        let mut kids: Vec<&Inst> = tree.iter().filter(|i| i.parent == inst).collect();
        kids.sort_by_key(|i| i.value.to_lowercase());
        for k in kids {
            let mut sub = chain.to_vec();
            sub.push(k.value.clone());
            egui::CollapsingHeader::new(&k.value)
                .id_salt(("collect-inst", p.id, k.id))
                .show(ui, |ui| self.level_ui(ui, p, tree, Some(k.id), &sub));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    /// A stand-in organisicator: answers each request with `answer(path)`,
    /// checking the token like the real one.
    fn fake(answer: fn(&str) -> (u16, Value)) -> Client {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for s in l.incoming().flatten() {
                let mut r = BufReader::new(s.try_clone().unwrap());
                let mut line = String::new();
                r.read_line(&mut line).unwrap();
                let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
                let (mut authed, mut len) = (false, 0usize);
                loop {
                    let mut h = String::new();
                    if r.read_line(&mut h).unwrap_or(0) == 0 || h.trim().is_empty() {
                        break;
                    }
                    let lower = h.to_ascii_lowercase();
                    authed |= lower.starts_with("x-organisicator-token: tok");
                    if let Some(v) = lower.strip_prefix("content-length:") {
                        len = v.trim().parse().unwrap_or(0);
                    }
                }
                // Read the body: closing with it unread resets the
                // connection on Windows, and the client sees no answer.
                let _ = std::io::Read::read_exact(&mut r, &mut vec![0; len]);
                let (code, body) = if authed { answer(&path) } else { (401, json!({"error": "token"})) };
                let text = body.to_string();
                let mut s = s;
                let _ = write!(s, "HTTP/1.1 {code} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}", text.len());
            }
        });
        Client { base: format!("http://127.0.0.1:{port}"), token: "tok".into() }
    }

    fn renamed(path: &str) -> (u16, Value) {
        match path {
            "/v1/assets/42" => (200, json!({
                "id": 42, "kind": "sequence", "exists": true, "missing": false,
                "path": "/m/live/spooky/sequences/coffee/cf_015/plates/bg/bg.####.exr",
                "rel": "plates/bg/bg.####.exr",
                "project": {"id": 5, "name": "spooky", "code": "spb"},
                "shot": [{"id": 7, "value": "coffee"}, {"id": 9, "value": "cf_015"}],
            })),
            "/v1/assets/43" => (200, json!({"id": 43, "kind": "file", "exists": false, "missing": true,
                "path": "/m/x.exr", "rel": "x.exr", "project": {"name": "spooky"}, "shot": []})),
            _ => (404, json!({"error": "no asset 99 in organisicator's index"})),
        }
    }

    #[test]
    fn an_asset_resolves_by_id_and_takes_its_new_names() {
        let c = fake(renamed);
        let old = AssetRef { id: 42, project: "spooky".into(), shot: vec!["coffee".into(), "cf_010".into()], rel: "plates/bg/bg.####.exr".into() };
        let r = resolve(&c, &old).unwrap();
        assert_eq!(r.path, "/m/live/spooky/sequences/coffee/cf_015/plates/bg/bg.####.exr");
        assert_eq!(r.named.shot, ["coffee", "cf_015"]);
        assert_eq!(r.named.id, 42);
    }

    #[test]
    fn what_is_wrong_is_said_in_words() {
        let c = fake(renamed);
        let e = resolve(&c, &AssetRef { id: 43, ..Default::default() }).unwrap_err();
        assert!(e.contains("x.exr") && e.contains("gone"), "{e}");
        let e = resolve(&c, &AssetRef { id: 99, ..Default::default() }).unwrap_err();
        assert!(e.contains("no asset 99"), "{e}");
        let bad = Client { token: "wrong".into(), ..c };
        assert!(resolve(&bad, &AssetRef { id: 42, ..Default::default() }).is_err());
        // Nothing listening: organisicator is not running, and we say so.
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        drop(l);
        let dead = Client { base: format!("http://127.0.0.1:{port}"), token: "tok".into() };
        assert_eq!(resolve(&dead, &AssetRef { id: 42, ..Default::default() }).unwrap_err(), NOT_RUNNING);
    }

    fn ingests(path: &str) -> (u16, Value) {
        match path {
            "/v1/ingest" => (200, json!({"id": 77, "path": "/m/p/shots/a/plates/bg.####.exr", "copied": 1})),
            _ => (404, json!({"error": "no"})),
        }
    }

    #[test]
    fn an_ingest_hands_back_a_reference_by_id() {
        let c = fake(ingests);
        let d = Dest { project: 5, project_name: "spooky".into(), instance: Some(3), shot: vec!["a".into()], folder: "plates".into() };
        let r = ingest(&c, "/dl/bg.####.exr", &d).unwrap();
        assert_eq!((r.id, r.project.as_str(), r.shot.clone()), (77, "spooky", vec!["a".to_string()]));
    }

    /// Against a REAL organisicator API (its `serve_a_fixture_until_killed`
    /// test, or the app): ingest `$LIVE_SRC` into project 5 / shot 3,
    /// then resolve the id it answers. Ignored: needs the server running.
    #[test]
    #[ignore]
    fn live_ingest_then_resolve() {
        let c = Client::discover().expect("organisicator API (ORGANISICATOR_API_FILE)");
        let src = std::env::var("LIVE_SRC").expect("LIVE_SRC");
        let d = Dest { project: 5, project_name: "spooky".into(), instance: Some(3), shot: vec!["a_010".into()], folder: "plates".into() };
        let r = ingest(&c, &src, &d).unwrap();
        let res = resolve(&c, &r).unwrap();
        println!("asset {} -> {} ({:?})", r.id, res.path, res.named);
        assert!(res.path.contains("shots/a_010/plates"), "{}", res.path);
        assert_eq!(res.named.shot, ["a_010"]);
    }

    /// Another machine's id that names some other file here is not
    /// trusted: the saved names find the right one, or it's an error that
    /// says why — never the wrong footage.
    #[test]
    fn another_machines_id_falls_back_to_names() {
        let saved = AssetRef { id: 42, project: "spooky".into(), shot: vec!["cf_010".into()], rel: "renders/light/light_####.png".into() };
        let at = |project: &str, rel: &str, path: &str| Resolved {
            path: path.into(),
            named: AssetRef { id: 42, project: project.into(), shot: vec![], rel: rel.into() },
        };
        let names = || Ok(at("spooky", "renders/light/light_####.png", "/d/spooky/cf_010/renders/light/light_####.png"));
        let no_names = || Err("not on this machine".to_string());
        // The same asset, renamed (shot or project): the id wins.
        let renamed = at("spooky", "renders/light/light_####.png", "/m/spooky/cf_015/renders/light/light_####.png");
        assert_eq!(choose(&saved, Ok(renamed.clone()), no_names).unwrap(), renamed);
        let moved = at("spooky_2026", "renders/light/light_####.png", "/m/spooky_2026/x");
        assert_eq!(choose(&saved, Ok(moved.clone()), no_names).unwrap(), moved);
        // Some other file under the same id: the names find it...
        let other = at("beach", "plates/bg.####.exr", "/w/beach/plates/bg.####.exr");
        assert_eq!(choose(&saved, Ok(other.clone()), names).unwrap().path, "/d/spooky/cf_010/renders/light/light_####.png");
        // ...or it's an error naming both, not the wrong footage.
        let e = choose(&saved, Ok(other), no_names).unwrap_err();
        assert!(e.contains("beach") && e.contains("another computer"), "{e}");
        // A reference with no names trusts the id.
        let bare = AssetRef { id: 42, ..Default::default() };
        assert!(same_asset(&bare, &at("beach", "x.exr", "/x").named));
    }

    #[test]
    fn labels_drop_a_pattern_that_repeats_its_folder() {
        assert_eq!(asset_label("plates/bg_v001/bg_v001.####.exr"), "plates › bg_v001");
        assert_eq!(asset_label("plates/bg.####.exr"), "plates › bg.####.exr");
    }

    #[test]
    fn a_reference_round_trips_through_json() {
        let r = AssetRef { id: 42, project: "p".into(), shot: vec!["a".into()], rel: "x.exr".into() };
        let s = serde_json::to_string(&r).unwrap();
        assert_eq!(serde_json::from_str::<AssetRef>(&s).unwrap(), r);
    }
}
