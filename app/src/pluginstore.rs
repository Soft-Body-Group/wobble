//! The plugin store: install plugins from the registry the website serves.
//! User-facing, plugins are WOBBINS (the site, this window, the dialogs);
//! on disk and in code they stay plugins — `plugin.json`,
//! `~/.wobble/plugins` — so nothing already installed moves.
//!
//! The website (`wobble.ing/wobbins`) is for BROWSING; its
//! "install" button opens `wobble://install/<id>`, which lands here through
//! `urlhook`. A link carries an ID and nothing else — never a URL — so the
//! only place bytes can come from is the registry below, the same rule a
//! comp naming model ids follows (`engine::models`).
//!
//! Nothing installs without a click IN THE APP: a link opens a confirm
//! dialog showing what the plugin is, and an addon that runs a process says
//! so in red. An installed addon stays disabled until the addons window
//! enables it (`Prefs::addons_enabled` is the opt-in, untouched here).
//!
//! A bundle is JSON (`{id, version, files: [{path, text}]}`) rather than a
//! zip: plugins are text, and a format with no archive semantics has no
//! zip-slip or symlink to get wrong. Its size and sha256 come from the
//! index, and it lands in a staging folder that is renamed into place only
//! once every file is written.
//!
//! PAID wobbins have no public bundle (`file: null` in the index): they
//! come from `/api/wobbins/<id>/download` with the account's token
//! (`account.rs`), and only for someone who owns one. Buying happens on the
//! website — the app opens the wobbin's page — never in here.
//!
//! All of the paid half is the `commerce` feature (on by default, OFF in a
//! distributed build): without it there is no account, no sign-in, no
//! price and no tip button, and paid wobbins are not listed at all.

#[cfg(feature = "commerce")]
use crate::account;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};

/// The registry. `WOBBLE_PLUGIN_REGISTRY` overrides it (a local copy of the
/// site while developing it); bundles resolve RELATIVE to it.
pub const REGISTRY_URL: &str = "https://wobble.ing/wobbins/index.json";

const MAX_BUNDLE: u64 = 8 * 1024 * 1024;
const MAX_FILES: usize = 256;

pub fn registry_url() -> String {
    std::env::var("WOBBLE_PLUGIN_REGISTRY").unwrap_or_else(|_| REGISTRY_URL.into())
}

/// The site the registry lives on (scheme + host), for its pages.
pub fn site_base() -> String {
    let reg = registry_url();
    match reg.find("://").and_then(|i| reg[i + 3..].find('/').map(|j| i + 3 + j)) {
        Some(end) => reg[..end].to_string(),
        None => reg,
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Entry {
    pub id: String,
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub author: String,
    /// "node" | "tool" | "addon" (a folder may hold more than one kind;
    /// this is the headline).
    pub kind: String,
    #[serde(default)]
    pub description: String,
    /// An addon with a `run` line: installing it puts a program on disk that
    /// runs once enabled.
    #[serde(default)]
    pub runs_code: bool,
    #[serde(default)]
    pub license: String,
    /// Relative to the registry URL. Never absolute. None for a paid
    /// wobbin, which downloads through the account API.
    #[serde(default)]
    pub file: Option<String>,
    pub bytes: u64,
    pub sha256: String,
    /// Taking it costs something: a fixed price, or a pay-what-you-want
    /// floor above zero. Only an owner can download it.
    #[serde(default)]
    pub paid: bool,
    #[serde(default)]
    #[cfg_attr(not(feature = "commerce"), allow(dead_code))]
    pub price_cents: u64,
    #[serde(default)]
    pub pwyw: bool,
    #[serde(default)]
    #[cfg_attr(not(feature = "commerce"), allow(dead_code))]
    pub min_cents: u64,
}

#[cfg(feature = "commerce")]
fn dollars(c: u64) -> String {
    if c % 100 == 0 { format!("${}", c / 100) } else { format!("${}.{:02}", c / 100, c % 100) }
}

impl Entry {
    /// "free", "$4", "$2+", "pay what you want".
    #[cfg(feature = "commerce")]
    pub fn price_label(&self) -> String {
        match (self.pwyw, self.min_cents, self.price_cents) {
            (true, 0, _) => "pay what you want".into(),
            (true, m, _) => format!("{}+", dollars(m)),
            (false, _, 0) => "free".into(),
            (false, _, p) => dollars(p),
        }
    }
}

#[derive(Deserialize)]
struct Index {
    plugins: Vec<Entry>,
}

#[derive(Deserialize)]
struct Bundle {
    id: String,
    version: String,
    files: Vec<BundleFile>,
}

#[derive(Deserialize)]
struct BundleFile {
    path: String,
    text: String,
}

/// `wobble://install/<id>` (or `wobble://install?id=<id>`) -> the id.
pub fn parse_link(url: &str) -> Option<String> {
    let rest = url.strip_prefix("wobble://")?.trim_end_matches('/');
    let id = rest
        .strip_prefix("install/")
        .or_else(|| rest.strip_prefix("install?id="))?;
    valid_id(id).then(|| id.to_string())
}

/// Ids are folder names: lowercase, digits, `_` and `-`, nothing else.
pub fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

/// A bundle path must stay inside its folder: relative, plain components.
fn safe_rel(p: &str) -> Option<PathBuf> {
    let path = Path::new(p);
    if p.is_empty() || p.contains('\\') || path.is_absolute() {
        return None;
    }
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::Normal(s) if !s.to_string_lossy().starts_with('.') => out.push(s),
            _ => return None,
        }
    }
    Some(out)
}

pub fn install_root() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|h| Path::new(&h).join(".wobble/plugins"))
        .unwrap_or_else(|| PathBuf::from("plugins"))
}

/// Written beside an installed plugin so the store knows its version (the
/// manifests do not all carry one). A dotfile, so no bundle can ship it.
const RECEIPT: &str = ".store.json";

/// The version installed under `root/<id>`: the store's receipt, else "?"
/// for a folder put there by hand.
pub fn installed_version(root: &Path, id: &str) -> Option<String> {
    let dir = root.join(id);
    if let Ok(t) = std::fs::read_to_string(dir.join(RECEIPT)) {
        if let Some(v) = serde_json::from_str::<serde_json::Value>(&t)
            .ok()
            .and_then(|v| v.get("version")?.as_str().map(String::from))
        {
            return Some(v);
        }
    }
    dir.is_dir().then(|| "?".into())
}

/// The web build has no HTTP client (no ureq on wasm32): no store.
#[cfg(target_arch = "wasm32")]
fn get(url: &str, _cap: u64) -> anyhow::Result<Vec<u8>> {
    anyhow::bail!("GET {url}: the plugin store is not available in the web build")
}

#[cfg(not(target_arch = "wasm32"))]
fn get(url: &str, cap: u64) -> anyhow::Result<Vec<u8>> {
    use std::io::Read;
    let resp = ureq::get(url).call().map_err(|e| anyhow::anyhow!("GET {url}: {e}"))?;
    let mut buf = Vec::new();
    resp.into_body().into_reader().take(cap + 1).read_to_end(&mut buf)?;
    anyhow::ensure!((buf.len() as u64) <= cap, "{url} is larger than {cap} bytes");
    Ok(buf)
}

fn fetch_index() -> anyhow::Result<Vec<Entry>> {
    let bytes = get(&registry_url(), 4 * 1024 * 1024)?;
    let idx: Index = serde_json::from_slice(&bytes)?;
    Ok(idx
        .plugins
        .into_iter()
        .filter(|e| valid_id(&e.id) && (cfg!(feature = "commerce") || !e.paid))
        .collect())
}

/// Resolve a bundle's relative `file` against the registry URL.
fn bundle_url(registry: &str, file: &str) -> anyhow::Result<String> {
    anyhow::ensure!(
        !file.contains("://") && !file.starts_with('/') && !file.contains(".."),
        "registry file '{file}' is not a relative path"
    );
    let base = registry.rsplit_once('/').map_or(registry, |(b, _)| b);
    Ok(format!("{base}/{file}"))
}

/// Check a downloaded bundle against its index entry and write it to
/// `root/<id>`, replacing what was there only once the new copy is whole.
pub fn install_bytes(root: &Path, e: &Entry, bytes: &[u8]) -> anyhow::Result<()> {
    anyhow::ensure!(valid_id(&e.id), "bad id '{}'", e.id);
    anyhow::ensure!(bytes.len() as u64 == e.bytes, "size {} != {} in the registry", bytes.len(), e.bytes);
    let hash = format!("{:x}", Sha256::digest(bytes));
    anyhow::ensure!(hash.eq_ignore_ascii_case(&e.sha256), "sha256 mismatch — refusing to install");
    let b: Bundle = serde_json::from_slice(bytes)?;
    anyhow::ensure!(b.id == e.id && b.version == e.version, "bundle is {}@{}, expected {}@{}", b.id, b.version, e.id, e.version);
    anyhow::ensure!(!b.files.is_empty() && b.files.len() <= MAX_FILES, "bundle has {} files", b.files.len());
    let mut rels = Vec::new();
    for f in &b.files {
        rels.push(safe_rel(&f.path).ok_or_else(|| anyhow::anyhow!("unsafe path '{}' in bundle", f.path))?);
    }
    anyhow::ensure!(
        rels.iter().any(|r| ["plugin.json", "tool.json", "addon.json"].iter().any(|m| r == Path::new(m))),
        "bundle has no plugin.json, tool.json or addon.json"
    );
    std::fs::create_dir_all(root)?;
    let stage = root.join(format!(".staging-{}", e.id));
    let old = root.join(format!(".old-{}", e.id));
    let dest = root.join(&e.id);
    let _ = std::fs::remove_dir_all(&stage);
    let _ = std::fs::remove_dir_all(&old);
    for (rel, f) in rels.iter().zip(&b.files) {
        let p = stage.join(rel);
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(&p, &f.text)?;
    }
    std::fs::write(
        stage.join(RECEIPT),
        serde_json::json!({ "version": e.version, "sha256": e.sha256 }).to_string(),
    )?;
    if dest.exists() {
        std::fs::rename(&dest, &old)?;
    }
    if let Err(err) = std::fs::rename(&stage, &dest) {
        let _ = std::fs::rename(&old, &dest);
        return Err(err.into());
    }
    let _ = std::fs::remove_dir_all(&old);
    Ok(())
}

enum Msg {
    Index(anyhow::Result<Vec<Entry>>),
    Installed(String, anyhow::Result<()>),
    #[cfg(feature = "commerce")]
    Library(anyhow::Result<HashSet<String>>),
}

/// Fetch one bundle: the public copy for a free wobbin, the account API for
/// a paid one (with the token; the site checks ownership).
fn fetch_bundle(e: &Entry, token: Option<&str>) -> anyhow::Result<Vec<u8>> {
    let cap = e.bytes.min(MAX_BUNDLE);
    if !e.paid {
        let file = e.file.as_deref().ok_or_else(|| anyhow::anyhow!("the registry lists no file for {}", e.id))?;
        return get(&bundle_url(&registry_url(), file)?, cap);
    }
    #[cfg(not(feature = "commerce"))]
    {
        let _ = token;
        anyhow::bail!("{} is not available in this build", e.name)
    }
    #[cfg(feature = "commerce")]
    {
    let token = token.ok_or_else(|| anyhow::anyhow!("sign in to install a wobbin you bought"))?;
    let url = format!("{}/api/wobbins/{}/download", account::site_base(), e.id);
    let (status, bytes) = account::get(&url, Some(token), cap)?;
    match status {
        200 => Ok(bytes),
        401 => anyhow::bail!("your sign-in has expired — sign in again"),
        402 => anyhow::bail!("this account does not own {} — buy it on the website", e.name),
        s => anyhow::bail!("download failed (HTTP {s})"),
    }
    }
}

#[cfg(feature = "commerce")]
fn fetch_library(token: &str) -> anyhow::Result<HashSet<String>> {
    let (status, bytes) = account::get(&format!("{}/api/library", account::site_base()), Some(token), 1024 * 1024)?;
    anyhow::ensure!(status == 200, "library: HTTP {status}");
    let v: serde_json::Value = serde_json::from_slice(&bytes)?;
    Ok(v["owned"]
        .as_array()
        .map(|a| a.iter().filter_map(|o| o["wobbin_id"].as_str().map(String::from)).collect())
        .unwrap_or_default())
}

pub struct Store {
    pub window_open: bool,
    index: Option<Vec<Entry>>,
    index_err: Option<String>,
    loading: bool,
    /// Ids a link asked for, waiting on the index to confirm them.
    wanted: Vec<String>,
    /// The confirm dialog: an entry a link asked for.
    confirm: Option<Entry>,
    busy: Option<String>,
    status: Option<(bool, String)>,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    /// The signed-in account, read from disk once (`None` inside = signed out).
    #[cfg(feature = "commerce")]
    account: Option<Option<account::Saved>>,
    #[cfg(feature = "commerce")]
    signin: Option<account::SignIn>,
    /// Wobbin ids the account owns (paid or tipped).
    owned: HashSet<String>,
}

impl Default for Store {
    fn default() -> Self {
        let (tx, rx) = channel();
        Store {
            window_open: false,
            index: None,
            index_err: None,
            loading: false,
            wanted: Vec::new(),
            confirm: None,
            busy: None,
            status: None,
            tx,
            rx,
            #[cfg(feature = "commerce")]
            account: None,
            #[cfg(feature = "commerce")]
            signin: None,
            owned: HashSet::new(),
        }
    }
}

impl Store {
    #[cfg(feature = "commerce")]
    fn account(&mut self) -> Option<&account::Saved> {
        self.account.get_or_insert_with(account::load).as_ref()
    }

    #[cfg(feature = "commerce")]
    fn token(&mut self) -> Option<String> {
        self.account().map(|a| a.token.clone())
    }

    #[cfg(not(feature = "commerce"))]
    fn token(&mut self) -> Option<String> {
        None
    }

    /// A `wobble://` link arrived.
    pub fn open_link(&mut self, url: &str) {
        match parse_link(url) {
            Some(id) => {
                self.wanted.push(id);
                self.refresh();
            }
            None => self.status = Some((false, format!("not an install link: {url}"))),
        }
    }

    fn refresh(&mut self) {
        if self.loading {
            return;
        }
        self.loading = true;
        let tx = self.tx.clone();
        crate::bg::spawn(move || {
            let _ = tx.send(Msg::Index(fetch_index()));
        });
        self.refresh_library();
    }

    #[cfg(not(feature = "commerce"))]
    fn refresh_library(&mut self) {}

    #[cfg(feature = "commerce")]
    fn refresh_library(&mut self) {
        let Some(token) = self.token() else {
            self.owned.clear();
            return;
        };
        let tx = self.tx.clone();
        crate::bg::spawn(move || {
            let _ = tx.send(Msg::Library(fetch_library(&token)));
        });
    }

    fn install(&mut self, e: Entry) {
        let tx = self.tx.clone();
        let token = self.token();
        self.busy = Some(e.id.clone());
        crate::bg::spawn(move || {
            let r = fetch_bundle(&e, token.as_deref()).and_then(|b| install_bytes(&install_root(), &e, &b));
            let _ = tx.send(Msg::Installed(e.id.clone(), r));
        });
    }

    /// The wobbin's page on the website, where buying happens.
    fn open_page(e: &Entry) {
        crate::open_url(&format!("{}/wobbins/#{}", site_base(), e.id));
    }

    /// Drain the workers. Returns true when a plugin was installed, so the
    /// app rescans its plugin folders.
    fn poll(&mut self) -> bool {
        let mut installed = false;
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::Index(r) => {
                    self.loading = false;
                    match r {
                        Ok(list) => {
                            self.index_err = None;
                            for id in std::mem::take(&mut self.wanted) {
                                match list.iter().find(|e| e.id == id) {
                                    Some(e) => self.confirm = Some(e.clone()),
                                    None => self.status = Some((false, format!("'{id}' is not in the registry"))),
                                }
                            }
                            self.index = Some(list);
                        }
                        Err(e) => {
                            self.index_err = Some(format!("{e:#}"));
                            if !self.wanted.is_empty() {
                                self.wanted.clear();
                                self.status = Some((false, format!("could not reach the wobbin registry: {e:#}")));
                            }
                        }
                    }
                }
                #[cfg(feature = "commerce")]
                Msg::Library(r) => match r {
                    Ok(owned) => self.owned = owned,
                    Err(e) => log::warn!("wobbin library: {e:#}"),
                },
                Msg::Installed(id, r) => {
                    self.busy = None;
                    match r {
                        Ok(()) => {
                            self.status = Some((true, format!("installed {id}")));
                            installed = true;
                        }
                        Err(e) => self.status = Some((false, format!("{id}: {e:#}"))),
                    }
                }
            }
        }
        #[cfg(feature = "commerce")]
        if let Some(si) = &mut self.signin {
            if let Some(r) = si.poll() {
                self.signin = None;
                match r {
                    Ok(saved) => {
                        self.status = Some((true, format!("signed in as {}", saved.email.as_deref().unwrap_or("your account"))));
                        self.account = Some(Some(saved));
                        self.refresh_library();
                    }
                    Err(e) => self.status = Some((false, format!("{e:#}"))),
                }
            }
        }
        installed
    }

    fn entry_lines(ui: &mut egui::Ui, e: &Entry, owned: bool) {
        ui.horizontal(|ui| {
            ui.strong(format!("{}  {}", e.name, e.version));
            #[cfg(feature = "commerce")]
            if owned && e.paid {
                ui.colored_label(crate::theme::accent(), "owned");
            } else {
                ui.label(e.price_label());
            }
            #[cfg(not(feature = "commerce"))]
            let _ = owned;
        });
        ui.weak(format!(
            "{} · {}{}",
            e.kind,
            if e.author.is_empty() { "unknown author" } else { &e.author },
            if e.license.is_empty() { String::new() } else { format!(" · {}", e.license) }
        ));
        if !e.description.is_empty() {
            ui.label(&e.description);
        }
        if e.runs_code {
            ui.colored_label(
                crate::theme::error(),
                "runs a program on this machine once you enable it in addons…",
            );
        }
    }

    /// Install / update / reinstall, or — for a paid wobbin this account
    /// does not own — a button to its page on the website. Returns true
    /// when it acted (the confirm dialog closes).
    fn action_button(&mut self, ui: &mut egui::Ui, e: &Entry, root: &Path) -> bool {
        #[cfg(feature = "commerce")]
        if e.paid && !self.owned.contains(&e.id) {
            let tip = if self.account.as_ref().is_some_and(|a| a.is_some()) {
                "buy it on the website, then install it here"
            } else {
                "buy it on the website, then sign in here to install it"
            };
            let clicked = ui.button(format!("buy {} ↗", e.price_label())).on_hover_text(tip).clicked();
            if clicked {
                Self::open_page(e);
            }
            return clicked;
        }
        let have = installed_version(root, &e.id);
        let label = match &have {
            Some(v) if *v == e.version => "reinstall",
            Some(_) => "update",
            None => "install",
        };
        if let Some(v) = have.filter(|v| *v != e.version) {
            ui.weak(format!("installed {v}"));
        }
        let clicked = ui.add_enabled(self.busy.is_none(), egui::Button::new(label)).clicked();
        if clicked {
            self.install(e.clone());
        }
        if cfg!(feature = "commerce") && e.pwyw && !e.paid && ui.small_button("tip ↗").on_hover_text("pay what you want, on the website").clicked() {
            Self::open_page(e);
        }
        clicked
    }

    #[cfg(not(feature = "commerce"))]
    fn account_row(&mut self, _ui: &mut egui::Ui) {}

    #[cfg(feature = "commerce")]
    fn account_row(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if let Some(si) = &self.signin {
                ui.spinner();
                match &si.code {
                    Some(code) => {
                        ui.label("in the browser, approve");
                        ui.strong(egui::RichText::new(code).monospace().size(16.0));
                        if let Some(url) = si.url.clone() {
                            if ui.small_button("open again ↗").clicked() {
                                crate::open_url(&url);
                            }
                        }
                    }
                    None => {
                        ui.label("starting sign-in…");
                    }
                }
                if ui.button("cancel").clicked() {
                    si.cancel();
                }
                return;
            }
            let who = self.account().map(|a| a.email.clone().unwrap_or_else(|| "your account".into()));
            match who {
                Some(email) => {
                    ui.weak("signed in as");
                    ui.label(email);
                    if ui.small_button("account ↗").clicked() {
                        crate::open_url(&format!("{}/account/", account::site_base()));
                    }
                    if ui.small_button("sign out").clicked() {
                        if let Some(Some(a)) = self.account.take() {
                            account::sign_out(&a);
                        }
                        self.account = Some(None);
                        self.owned.clear();
                    }
                }
                None => {
                    if ui
                        .button("sign in")
                        .on_hover_text("to install wobbins you bought — opens the browser")
                        .clicked()
                    {
                        self.signin = Some(account::SignIn::start());
                    }
                }
            }
        });
    }

    /// Once per UI pass. Returns true when a plugin was installed.
    pub fn frame(&mut self, ctx: &egui::Context) -> bool {
        for url in crate::urlhook::drain() {
            self.open_link(&url);
        }
        let installed = self.poll();
        #[cfg(feature = "commerce")]
        let signing_in = self.signin.is_some();
        #[cfg(not(feature = "commerce"))]
        let signing_in = false;
        if self.loading || self.busy.is_some() || signing_in {
            ctx.request_repaint();
        }
        let root = install_root();

        if let Some(e) = self.confirm.clone() {
            let mut keep = true;
            let owned = self.owned.contains(&e.id);
            egui::Window::new("install wobbin?")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.set_max_width(360.0);
                    Self::entry_lines(ui, &e, owned);
                    if !e.paid || owned {
                        ui.weak(format!("into {}", root.join(&e.id).display()));
                    } else {
                        #[cfg(feature = "commerce")]
                        if self.account().is_none() {
                            ui.weak("bought it already? sign in, in help ➡ get wobbins…");
                        }
                    }
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if self.action_button(ui, &e, &root) {
                            keep = false;
                        }
                        if ui.button("cancel").clicked() {
                            keep = false;
                        }
                    });
                });
            if !keep {
                self.confirm = None;
            }
        }

        if self.window_open {
            if self.index.is_none() && self.index_err.is_none() {
                self.refresh();
            }
            let mut open = true;
            egui::Window::new("wobbins")
                .open(&mut open)
                .default_width(460.0)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        if ui.button("browse the website ↗").clicked() {
                            crate::open_url(&format!("{}/wobbins/", site_base()));
                        }
                        if ui.add_enabled(!self.loading, egui::Button::new("refresh")).clicked() {
                            self.refresh();
                        }
                        if self.loading {
                            ui.spinner();
                        }
                    });
                    self.account_row(ui);
                    if let Some(err) = &self.index_err {
                        ui.colored_label(crate::theme::error(), err);
                    }
                    ui.separator();
                    let list = self.index.clone().unwrap_or_default();
                    egui::ScrollArea::vertical().max_height(420.0).show(ui, |ui| {
                        for e in &list {
                            let owned = self.owned.contains(&e.id);
                            ui.horizontal(|ui| {
                                ui.vertical(|ui| {
                                    ui.set_max_width(300.0);
                                    Self::entry_lines(ui, e, owned);
                                });
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    self.action_button(ui, e, &root);
                                });
                            });
                            ui.separator();
                        }
                    });
                });
            self.window_open = open;
        }

        if let Some((ok, msg)) = &self.status {
            let (ok, msg) = (*ok, msg.clone());
            let mut close = false;
            egui::Window::new("wobbin store")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-16.0, -72.0))
                .show(ctx, |ui| {
                    if ok {
                        ui.label(&msg);
                    } else {
                        ui.colored_label(crate::theme::error(), &msg);
                    }
                    close = ui.button("ok").clicked();
                });
            if close {
                self.status = None;
            }
        }
        if let Some(b) = &self.busy {
            egui::Window::new("installing")
                .collapsible(false)
                .resizable(false)
                .title_bar(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(format!("installing {b}…"));
                    });
                });
        }
        installed
    }
}

impl crate::Gfx {
    pub(crate) fn plugin_store_frame(&mut self, ctx: &egui::Context) {
        if self.store.frame(ctx) {
            crate::load_plugins();
            self.addons_reload();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(bytes: &[u8]) -> Entry {
        Entry {
            id: "demo".into(),
            name: "Demo".into(),
            version: "1.0".into(),
            author: String::new(),
            kind: "node".into(),
            description: String::new(),
            runs_code: false,
            license: String::new(),
            file: Some("dl/demo-1.0.json".into()),
            bytes: bytes.len() as u64,
            sha256: format!("{:x}", Sha256::digest(bytes)),
            paid: false,
            price_cents: 0,
            pwyw: false,
            min_cents: 0,
        }
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("wobble-store-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn links_carry_an_id_and_nothing_else() {
        assert_eq!(parse_link("wobble://install/invert").as_deref(), Some("invert"));
        assert_eq!(parse_link("wobble://install?id=soft_glow").as_deref(), Some("soft_glow"));
        assert_eq!(parse_link("wobble://install/invert/").as_deref(), Some("invert"));
        assert!(parse_link("wobble://install/../etc").is_none());
        assert!(parse_link("wobble://install/https://evil").is_none());
        assert!(parse_link("wobble://install/A").is_none());
        assert!(parse_link("http://install/x").is_none());
    }

    #[test]
    fn bundle_urls_stay_on_the_registry() {
        let r = "https://h/plugins/index.json";
        assert_eq!(bundle_url(r, "dl/x.json").unwrap(), "https://h/plugins/dl/x.json");
        assert!(bundle_url(r, "https://evil/x").is_err());
        assert!(bundle_url(r, "/x").is_err());
        assert!(bundle_url(r, "../x").is_err());
    }

    #[test]
    fn a_verified_bundle_installs_and_replaces() {
        let root = tmp("ok");
        let b = br#"{"id":"demo","version":"1.0","files":[{"path":"plugin.json","text":"{}"},{"path":"sub/a.wgsl","text":"x"}]}"#;
        install_bytes(&root, &entry(b), b).unwrap();
        assert_eq!(std::fs::read_to_string(root.join("demo/sub/a.wgsl")).unwrap(), "x");
        assert_eq!(installed_version(&root, "demo").as_deref(), Some("1.0"));
        std::fs::write(root.join("demo/stale.txt"), "old").unwrap();
        install_bytes(&root, &entry(b), b).unwrap();
        assert!(!root.join("demo/stale.txt").exists(), "a reinstall replaces the folder");
        assert!(!root.join(".old-demo").exists() && !root.join(".staging-demo").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_tampered_or_escaping_bundle_is_refused() {
        let root = tmp("bad");
        let b = br#"{"id":"demo","version":"1.0","files":[{"path":"plugin.json","text":"{}"}]}"#;
        let mut e = entry(b);
        e.sha256 = "00".repeat(32);
        assert!(install_bytes(&root, &e, b).is_err());
        for evil in ["../x.json", "/etc/x", ".hidden/plugin.json", "a\\\\b"] {
            let b = format!(r#"{{"id":"demo","version":"1.0","files":[{{"path":"plugin.json","text":"{{}}"}},{{"path":"{evil}","text":"x"}}]}}"#);
            assert!(install_bytes(&root, &entry(b.as_bytes()), b.as_bytes()).is_err(), "{evil}");
        }
        let b = br#"{"id":"other","version":"1.0","files":[{"path":"plugin.json","text":"{}"}]}"#;
        assert!(install_bytes(&root, &entry(b), b).is_err(), "id must match the entry");
        assert!(!root.join("demo").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    #[cfg(feature = "commerce")]
    fn prices_read_like_prices() {
        let mut e = entry(b"x");
        assert_eq!(e.price_label(), "free");
        e.price_cents = 400;
        assert_eq!(e.price_label(), "$4");
        e.price_cents = 450;
        assert_eq!(e.price_label(), "$4.50");
        e.pwyw = true;
        assert_eq!(e.price_label(), "pay what you want");
        e.min_cents = 200;
        assert_eq!(e.price_label(), "$2+");
    }

    #[test]
    fn a_paid_wobbin_parses_without_a_file_and_old_indexes_still_parse() {
        let paid: Entry = serde_json::from_str(r#"{"id":"x","name":"X","version":"1","kind":"node","file":null,"bytes":1,"sha256":"a","paid":true,"price_cents":500}"#).unwrap();
        assert!(paid.paid && paid.file.is_none());
        let old: Entry = serde_json::from_str(r#"{"id":"x","name":"X","version":"1","kind":"node","file":"dl/x.json","bytes":1,"sha256":"a"}"#).unwrap();
        assert!(!old.paid && old.file.as_deref() == Some("dl/x.json"));
        // A paid wobbin with nobody signed in never reaches the network.
        let err = fetch_bundle(&paid, None).unwrap_err().to_string();
        let want = if cfg!(feature = "commerce") { "sign in" } else { "not available in this build" };
        assert!(err.contains(want), "{err}");
    }

    /// Against a served registry: `python3 -m http.server -d public` in the
    /// website, then `WOBBLE_PLUGIN_REGISTRY=http://127.0.0.1:8000/plugins/index.json
    /// cargo test -p app live_registry -- --ignored`.
    #[test]
    #[ignore]
    fn live_registry_installs_everything() {
        let root = tmp("live");
        let list = fetch_index().unwrap();
        assert!(!list.is_empty());
        for e in &list {
            if e.paid {
                continue;
            }
            let b = fetch_bundle(e, None).unwrap();
            install_bytes(&root, e, &b).unwrap();
            assert_eq!(installed_version(&root, &e.id).as_deref(), Some(e.version.as_str()));
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The whole paid path through the app's own code, against a local site
    /// (`npm run dev` in website_wobble, with a wobbin priced in its
    /// registry and `STRIPE_WEBHOOK_SECRET=whsec_local_dev` in .dev.vars):
    /// device-code sign-in (approved over HTTP, as a browser would), the
    /// token saved, a paid download refused, a signed webhook buying it, the
    /// library and the download then succeeding and installing.
    ///   WOBBLE_E2E_PAID=softglow cargo test -p app paid_path -- --ignored
    #[test]
    #[ignore]
    #[cfg(feature = "commerce")]
    fn paid_path_end_to_end() {
        let site = "http://localhost:8787";
        let paid_id = std::env::var("WOBBLE_E2E_PAID").unwrap_or_else(|_| "softglow".into());
        let home = tmp("e2e-home");
        std::fs::create_dir_all(&home).unwrap();
        std::env::set_var("HOME", &home);
        std::env::set_var("WOBBLE_PLUGIN_REGISTRY", format!("{site}/wobbins/index.json"));
        std::env::set_var("WOBBLE_NO_BROWSER", "1");

        let list = fetch_index().unwrap();
        let e = list.iter().find(|e| e.id == paid_id).expect("the paid wobbin is in the local registry").clone();
        assert!(e.paid && e.file.is_none());
        assert!(fetch_bundle(&e, None).unwrap_err().to_string().contains("sign in"));

        // A browser session, by the email link (DEV_MODE shows the link).
        let (_, start) = account::post_json(&format!("{site}/api/auth/email"), &serde_json::json!({ "email": "e2e@example.com" }), None).unwrap();
        let link = start["dev_link"].as_str().unwrap().to_string();
        let token = link.split("token=").nth(1).unwrap().to_string();
        let agent = ureq::Agent::new_with_config(ureq::Agent::config_builder().http_status_as_error(false).max_redirects(0).build());
        let resp = agent
            .post(&format!("{site}/auth/email"))
            .header("origin", site)
            .header("content-type", "application/x-www-form-urlencoded")
            .send(format!("token={token}"))
            .unwrap();
        let cookie = resp.headers().get("set-cookie").unwrap().to_str().unwrap().split(';').next().unwrap().to_string();

        // The app's sign-in, approved from that session.
        let mut si = account::SignIn::start();
        let code = loop {
            assert!(si.poll().is_none(), "sign-in ended early");
            if let Some(c) = si.code.clone() {
                break c;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        };
        let r = agent
            .post(&format!("{site}/api/device/approve"))
            .header("origin", site)
            .header("cookie", &cookie)
            .header("content-type", "application/json")
            .send(serde_json::json!({ "code": code, "approve": true }).to_string())
            .unwrap();
        assert_eq!(r.status().as_u16(), 200);
        let saved = loop {
            if let Some(r) = si.poll() {
                break r.unwrap();
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
        };
        assert!(saved.token.starts_with("wob_"));
        assert_eq!(account::load().unwrap().token, saved.token, "the token was saved for this site");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(home.join(".wobble/account.json")).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "the token file is the user's alone");
        }

        // Signed in but not an owner: refused, with a reason a person can act on.
        assert!(fetch_bundle(&e, Some(&saved.token)).unwrap_err().to_string().contains("buy it"));

        // Stripe says it was paid.
        let (_, who) = account::get(&format!("{site}/api/me"), Some(&saved.token), 65536).unwrap();
        let uid = serde_json::from_slice::<serde_json::Value>(&who).unwrap()["user"]["id"].as_str().unwrap().to_string();
        let body = serde_json::json!({ "id": "evt_e2e", "type": "checkout.session.completed", "data": { "object": {
            "id": format!("cs_e2e_{}", std::process::id()), "payment_status": "paid", "amount_total": e.price_cents, "currency": "usd",
            "payment_intent": format!("pi_e2e_{}", std::process::id()), "metadata": { "user_id": uid, "wobbin_id": e.id } } } }).to_string();
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        let sig = {
            use sha2::Sha256;
            // HMAC-SHA256 by hand: sha2 is here, hmac is not.
            let key = b"whsec_local_dev";
            let mut k = [0u8; 64];
            k[..key.len()].copy_from_slice(key);
            let (mut ipad, mut opad) = ([0x36u8; 64], [0x5cu8; 64]);
            for i in 0..64 {
                ipad[i] ^= k[i];
                opad[i] ^= k[i];
            }
            let inner = Sha256::new().chain_update(ipad).chain_update(format!("{t}.{body}")).finalize();
            format!("{:x}", Sha256::new().chain_update(opad).chain_update(inner).finalize())
        };
        let r = agent
            .post(&format!("{site}/api/stripe/webhook"))
            .header("stripe-signature", &format!("t={t},v1={sig}"))
            .send(body)
            .unwrap();
        assert_eq!(r.status().as_u16(), 200);

        assert!(fetch_library(&saved.token).unwrap().contains(&e.id));
        let bytes = fetch_bundle(&e, Some(&saved.token)).unwrap();
        let root = home.join(".wobble/plugins");
        install_bytes(&root, &e, &bytes).unwrap();
        assert_eq!(installed_version(&root, &e.id).as_deref(), Some(e.version.as_str()));

        account::sign_out(&saved);
        assert!(account::load().is_none());
        let _ = std::fs::remove_dir_all(&home);
    }
}
