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

pub fn resolve_now(r: &AssetRef) -> Result<Resolved, String> {
    resolve(&Client::discover()?, r)
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

// ---- the picker ----------------------------------------------------------------

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

/// One pickable asset in a shot.
#[derive(Clone, Debug, PartialEq)]
pub struct Asset {
    pub id: i64,
    pub rel: String,
    pub label: String,
    pub frames: Option<i64>,
    pub sequence: bool,
}

/// What a shot holds, as organisicator lists it (only what wobble reads).
pub fn parse_assets(v: &Value) -> (bool, Vec<Asset>) {
    let indexed = v.get("indexed").and_then(|i| i.as_bool()).unwrap_or(false);
    let list = v["assets"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| {
                    let rel = x["rel"].as_str()?.to_string();
                    // Only what a Load can open: the decoder's own list.
                    let probe = rel.replace('#', "0");
                    if !engine::decode::is_supported(std::path::Path::new(&probe)) {
                        return None;
                    }
                    Some(Asset {
                        id: x["id"].as_i64()?,
                        label: asset_label(&rel),
                        rel,
                        frames: x["frames"].as_i64(),
                        sequence: x["kind"].as_str() == Some("sequence"),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    (indexed, list)
}

/// The asset picker's state: which Load it fills, organisicator's projects,
/// and the trees / shot contents fetched so far (each on first expand).
pub struct Picker {
    pub node: u64,
    client: Result<Client, String>,
    projects: Result<Vec<Proj>, String>,
    trees: HashMap<i64, Result<Vec<Inst>, String>>,
    contents: HashMap<(i64, Option<i64>), Result<(bool, Vec<Asset>), String>>,
    pub filter: String,
    /// The reference the Load holds now, to open the tree onto it.
    current: Option<AssetRef>,
    /// Some = choosing where to INGEST the Load's footage, not an asset:
    /// the folder under the shot, and the shot picked so far.
    ingest: Option<(String, Option<Dest>)>,
    /// Choosing where a whole comp is COLLECTED (`collect.rs`), not one
    /// Load: only the words differ.
    collect: bool,
}

/// What the picker was used for this frame.
pub enum Picked {
    Asset(AssetRef),
    IngestTo(Dest),
}

impl Picker {
    pub fn new(node: u64, current: Option<AssetRef>) -> Self {
        let mut p = Picker {
            node,
            client: Err(String::new()),
            projects: Ok(Vec::new()),
            trees: HashMap::new(),
            contents: HashMap::new(),
            filter: String::new(),
            current,
            ingest: None,
            collect: false,
        };
        p.refresh();
        p
    }

    /// The picker as a destination chooser for an ingest.
    pub fn for_ingest(node: u64) -> Self {
        let mut p = Picker::new(node, None);
        p.ingest = Some((DEFAULT_FOLDER.to_string(), None));
        p
    }

    /// The shot a comp is collected into; the folder is where its footage goes.
    pub fn for_collect() -> Self {
        let mut p = Picker::for_ingest(u64::MAX);
        p.collect = true;
        p
    }

    pub fn refresh(&mut self) {
        self.client = Client::discover();
        self.trees.clear();
        self.contents.clear();
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

    fn assets(&mut self, pid: i64, inst: Option<i64>) -> Result<(bool, Vec<Asset>), String> {
        let client = self.client.clone();
        self.contents
            .entry((pid, inst))
            .or_insert_with(|| {
                let q = inst.map(|i| format!("?instance={i}")).unwrap_or_default();
                Ok(parse_assets(&client?.get(&format!("/v1/projects/{pid}/assets{q}"))?))
            })
            .clone()
    }

    /// Draws the picker; returns the reference picked this frame, if any.
    /// `open` goes false when the window is closed.
    pub fn ui(&mut self, ctx: &egui::Context, open: &mut bool) -> Option<Picked> {
        let mut picked = None;
        let mut refresh = false;
        let mut go = false;
        let title = match (self.collect, self.ingest.is_some()) {
            (true, _) => "collect into…",
            (_, true) => "ingest into…",
            _ => "pick asset",
        };
        let collect = self.collect;
        egui::Window::new(title)
            .id(egui::Id::new(("asset-picker", self.node)))
            .open(open)
            .default_size([380.0, 460.0])
            .resizable(true)
            .collapsible(false)
            .show(ctx, |ui| {
                if let Some((folder, chosen)) = &mut self.ingest {
                    ui.weak(if collect {
                        "pick the shot the comp is collected into; its footage goes in this folder of the shot"
                    } else {
                        "a COPY of this Load's footage goes into the shot you pick; the Load then reads the copy by id"
                    });
                    ui.horizontal(|ui| {
                        ui.label("folder");
                        ui.add(egui::TextEdit::singleline(folder).desired_width(140.0));
                        let label = match chosen {
                            Some(d) => format!("{} {} / {}", if collect { "use" } else { "ingest into" }, d.project_name, if d.shot.is_empty() { "(project)".into() } else { d.shot.join(" › ") }),
                            None => "pick a shot below".into(),
                        };
                        if ui.add_enabled(chosen.is_some() && !folder.trim().is_empty(), egui::Button::new(label)).clicked() {
                            go = true;
                        }
                    });
                    ui.separator();
                }
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
                        let here = self.current.as_ref().is_some_and(|c| c.project == p.name);
                        egui::CollapsingHeader::new(egui::RichText::new(&p.name).strong())
                            .id_salt(("asset-proj", self.node, p.id))
                            .default_open(here)
                            .show(ui, |ui| match self.tree(p.id) {
                                Err(e) => {
                                    ui.colored_label(crate::theme::error(), e);
                                }
                                Ok(tree) => {
                                    if let Some(r) = self.level_ui(ui, p, &tree, None, &[]) {
                                        picked = Some(r);
                                    }
                                }
                            });
                    }
                });
            });
        if refresh {
            self.refresh();
        }
        if go {
            if let Some((folder, Some(d))) = &self.ingest {
                let mut d = d.clone();
                d.folder = folder.trim().trim_matches('/').to_string();
                return Some(Picked::IngestTo(d));
            }
        }
        picked.map(Picked::Asset)
    }

    /// One level of the tree: the assets of `inst`, then its children.
    fn level_ui(
        &mut self,
        ui: &mut egui::Ui,
        p: &Proj,
        tree: &[Inst],
        inst: Option<i64>,
        chain: &[String],
    ) -> Option<AssetRef> {
        let mut picked = None;
        let filter = self.filter.to_lowercase();
        if let Some((_, chosen)) = &mut self.ingest {
            let here = chosen.as_ref().is_some_and(|d| d.project == p.id && d.instance == inst);
            let label = if chain.is_empty() { format!("put it in {} itself", p.name) } else { format!("put it in {}", chain.join(" › ")) };
            if ui.selectable_label(here, label).clicked() {
                *chosen = Some(Dest {
                    project: p.id,
                    project_name: p.name.clone(),
                    instance: inst,
                    shot: chain.to_vec(),
                    folder: String::new(),
                });
            }
        } else {
        match self.assets(p.id, inst) {
            Err(e) => {
                ui.colored_label(crate::theme::error(), e);
            }
            Ok((indexed, assets)) => {
                if !indexed {
                    ui.weak("not indexed yet — open this project's Work view in organisicator");
                }
                for a in assets.iter().filter(|a| filter.is_empty() || a.label.to_lowercase().contains(&filter)) {
                    let sel = self.current.as_ref().is_some_and(|c| c.id == a.id);
                    let span = match (a.sequence, a.frames) {
                        (true, Some(n)) => format!("{n} frames"),
                        (true, None) => "sequence".into(),
                        _ => String::new(),
                    };
                    if ui.selectable_label(sel, format!("▸ {}    {}", a.label, span)).clicked() {
                        picked = Some(AssetRef {
                            id: a.id,
                            project: p.name.clone(),
                            shot: chain.to_vec(),
                            rel: a.rel.clone(),
                        });
                    }
                }
            }
        }
        }
        let mut kids: Vec<&Inst> = tree.iter().filter(|i| i.parent == inst).collect();
        kids.sort_by_key(|i| i.value.to_lowercase());
        for k in kids {
            let mut sub = chain.to_vec();
            sub.push(k.value.clone());
            let here = self.current.as_ref().is_some_and(|c| c.shot.starts_with(&sub));
            egui::CollapsingHeader::new(&k.value)
                .id_salt(("asset-inst", self.node, p.id, k.id))
                .default_open(here)
                .show(ui, |ui| {
                    if let Some(r) = self.level_ui(ui, p, tree, Some(k.id), &sub) {
                        picked = Some(r);
                    }
                });
        }
        picked
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

    #[test]
    fn the_picker_lists_only_what_a_load_can_open() {
        let (indexed, a) = parse_assets(&json!({"indexed": true, "assets": [
            {"id": 1, "kind": "sequence", "rel": "plates/bg/bg.####.exr", "frames": 96},
            {"id": 2, "kind": "file", "rel": "scripts/comp.nk"},
            {"id": 3, "kind": "file", "rel": "ref/look.jpg"},
        ]}));
        assert!(indexed);
        assert_eq!(a.iter().map(|a| a.id).collect::<Vec<_>>(), [1, 3]);
        assert_eq!(a[0].label, "plates › bg");
        assert_eq!(a[0].frames, Some(96));
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
