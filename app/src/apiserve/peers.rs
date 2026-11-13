//! Other wobbles on the local network: announce this one, find the others,
//! and let a paired one in.
//!
//! - **Off until the user turns it on** (`network…`, in preferences): then
//!   this wobble announces itself over mDNS (`_wobble._tcp`, `mdns-sd`,
//!   the same on macOS, Windows and Linux) and opens the scripting API to
//!   the network on a port of its own (`api::server::serve_network`).
//! - **Nothing gets in without the pairing code.** The API can open and
//!   save files, so a network connection must `auth` with this wobble's
//!   code before any call; the window shows the code to read out to the
//!   other machine, and a new one can be made at any time.
//! - **Finding is free**: while the window is open (or sharing is on) this
//!   wobble lists the others it hears, and "check" connects to one with the
//!   code you typed, asks `app_info`, and remembers the code for next time.
//!
//! The setting and the codes live in `~/.wobble/network.json` (0600), never
//! in prefs or a setup. Suites never load it, share or browse.

use std::collections::{BTreeMap, HashMap};
use std::hash::{BuildHasher, Hasher};
use std::io::{BufRead, BufReader, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};
use serde::{Deserialize, Serialize};

pub const SERVICE: &str = "_wobble._tcp.local.";

#[derive(Default, Serialize, Deserialize)]
struct Config {
    #[serde(default)]
    share: bool,
    #[serde(default)]
    code: String,
    #[serde(default)]
    id: String,
    /// Codes that worked, by the other wobble's id.
    #[serde(default)]
    known: BTreeMap<String, String>,
    /// Paired wobbles may render shares of their Render nodes here
    /// (`farm`). Off unless the user ticks it, on top of sharing.
    #[serde(default)]
    render: bool,
    /// Paired wobbles may send finished renders here (`deliver`).
    #[serde(default)]
    accept: bool,
    /// Where they land; empty = `default_inbox()`.
    #[serde(default)]
    inbox: String,
}

#[derive(Clone, Debug)]
pub struct Peer {
    pub id: String,
    pub name: String,
    pub version: String,
    pub addrs: Vec<Ipv4Addr>,
    pub port: u16,
}

enum Check {
    Running,
    Ok(String),
    Failed(String),
}

#[derive(Default)]
struct Net {
    open: bool,
    loaded: bool,
    cfg: Config,
    daemon: Option<ServiceDaemon>,
    browse: Option<mdns_sd::Receiver<ServiceEvent>>,
    advertised: Option<String>,
    found: BTreeMap<String, Peer>,
    typed: HashMap<String, String>,
    /// Which paired peers help render (ticked in the window).
    helpers: HashMap<String, bool>,
    checks: HashMap<String, Check>,
    results: Option<(Sender<(String, Result<String, String>)>, Receiver<(String, Result<String, String>)>)>,
    listening: Option<u16>,
    /// The code changed while sharing: the listener restarts with it.
    restart: bool,
    error: Option<String>,
}

fn net() -> std::sync::MutexGuard<'static, Net> {
    static NET: OnceLock<Mutex<Net>> = OnceLock::new();
    NET.get_or_init(|| Mutex::new(Net::default())).lock().unwrap_or_else(|e| e.into_inner())
}

fn config_path() -> Option<PathBuf> {
    crate::home_dir().map(|h| h.join(".wobble").join("network.json"))
}

fn save(cfg: &Config) {
    let Some(p) = config_path() else { return };
    if let Some(d) = p.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    if let Ok(s) = serde_json::to_string_pretty(cfg) {
        if std::fs::write(&p, s).is_ok() {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600));
            }
        }
    }
}

fn random_u64() -> u64 {
    // RandomState's keys come from the OS's randomness, once per value.
    std::collections::hash_map::RandomState::new().build_hasher().finish()
}

/// A pairing code: eight characters a person can read out without
/// confusing 0/O or 1/I, as XXXX-XXXX.
pub fn new_code() -> String {
    const A: &[u8] = b"ABCDEFGHJKMNPQRSTUVWXYZ23456789";
    let (mut r, mut s) = (random_u64(), String::new());
    for i in 0..8 {
        if i == 4 {
            s.push('-');
        }
        s.push(A[(r % A.len() as u64) as usize] as char);
        r /= A.len() as u64;
        if r == 0 {
            r = random_u64();
        }
    }
    s
}

fn test_run() -> bool {
    crate::TEST_RUN.load(std::sync::atomic::Ordering::Relaxed)
}

fn ensure_loaded(n: &mut Net) {
    if n.loaded || test_run() {
        return;
    }
    n.loaded = true;
    if let Some(p) = config_path() {
        if let Ok(s) = std::fs::read_to_string(p) {
            n.cfg = serde_json::from_str(&s).unwrap_or_default();
        }
    }
    let mut dirty = false;
    if n.cfg.code.is_empty() {
        n.cfg.code = new_code();
        dirty = true;
    }
    if n.cfg.id.is_empty() {
        n.cfg.id = format!("{:016x}", random_u64());
        dirty = true;
    }
    if dirty {
        save(&n.cfg);
    }
}

/// Sharing is on (and this is not a suite).
pub fn wants_share() -> bool {
    let mut n = net();
    ensure_loaded(&mut n);
    // A render worker is this wobble's helper, never a wobble of its own.
    n.cfg.share && !test_run() && std::env::var_os("WOBBLE_WORKER").is_none()
}

pub fn code() -> String {
    let mut n = net();
    ensure_loaded(&mut n);
    n.cfg.code.clone()
}

/// The listener must be reopened (a new code).
pub fn take_restart() -> bool {
    std::mem::take(&mut net().restart)
}

/// Open the window (a render arrived: say so).
pub fn open_window() {
    net().open = true;
}

pub fn toggle_window() {
    let mut n = net();
    n.open = !n.open;
}

fn machine_name() -> String {
    static NAME: OnceLock<String> = OnceLock::new();
    NAME.get_or_init(|| {
        std::process::Command::new("hostname")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            // The machine's own name, without the network's domain.
            .map(|s| s.trim().split('.').next().unwrap_or("").to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "wobble".into())
    })
    .clone()
}

fn daemon(n: &mut Net) -> Option<&ServiceDaemon> {
    if n.daemon.is_none() {
        match ServiceDaemon::new() {
            Ok(d) => n.daemon = Some(d),
            Err(e) => {
                n.error = Some(format!("mDNS: {e}"));
                return None;
            }
        }
    }
    n.daemon.as_ref()
}

/// The network listener is up on `port`: announce it.
pub fn listening(port: u16) {
    let mut n = net();
    n.listening = Some(port);
    n.error = None;
    let id = n.cfg.id.clone();
    let host = machine_name();
    let Some(d) = daemon(&mut n) else { return };
    // One instance name per process: two wobbles on one machine both show.
    let instance = format!("wobble on {host} ({})", &id[..6.min(id.len())]);
    let props = [("id", id.as_str()), ("name", host.as_str()), ("ver", env!("CARGO_PKG_VERSION"))];
    let hostname = format!("{}.local.", host.replace(' ', "-"));
    match ServiceInfo::new(SERVICE, &instance, &hostname, "", port, &props[..]) {
        Ok(info) => {
            let info = info.enable_addr_auto();
            let full = info.get_fullname().to_string();
            match d.register(info) {
                Ok(()) => n.advertised = Some(full),
                Err(e) => n.error = Some(format!("announce: {e}")),
            }
        }
        Err(e) => n.error = Some(format!("announce: {e}")),
    }
}

/// Sharing went off: stop announcing.
pub fn stopped() {
    let mut n = net();
    n.listening = None;
    if let (Some(full), Some(d)) = (n.advertised.take(), n.daemon.as_ref()) {
        let _ = d.unregister(&full);
    }
}

pub fn set_error(e: String) {
    net().error = Some(e);
}

/// Once a frame: browse while it is wanted, take what was heard, take
/// the checks that finished.
pub fn poll() {
    let mut n = net();
    if test_run() {
        return;
    }
    let wanted = n.open || n.cfg.share;
    if wanted && n.browse.is_none() {
        if let Some(d) = daemon(&mut n) {
            match d.browse(SERVICE) {
                Ok(rx) => n.browse = Some(rx),
                Err(e) => n.error = Some(format!("browse: {e}")),
            }
        }
    } else if !wanted && n.browse.is_some() {
        if let Some(d) = n.daemon.as_ref() {
            let _ = d.stop_browse(SERVICE);
        }
        n.browse = None;
        n.found.clear();
    }
    let me = n.cfg.id.clone();
    let mut events = Vec::new();
    if let Some(rx) = &n.browse {
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
    }
    for e in events {
        match e {
            ServiceEvent::ServiceResolved(info) => {
                let id = info.get_property_val_str("id").unwrap_or_default().to_string();
                if id.is_empty() || id == me {
                    continue;
                }
                // Loopback would mean THIS machine to whoever dials it.
                let mut addrs: Vec<Ipv4Addr> =
                    info.get_addresses_v4().into_iter().filter(|a| !a.is_loopback()).collect();
                addrs.sort();
                n.found.insert(
                    info.get_fullname().to_string(),
                    Peer {
                        id,
                        name: info.get_property_val_str("name").unwrap_or("?").to_string(),
                        version: info.get_property_val_str("ver").unwrap_or("?").to_string(),
                        addrs,
                        port: info.get_port(),
                    },
                );
            }
            ServiceEvent::ServiceRemoved(_, full) => {
                n.found.remove(&full);
            }
            _ => {}
        }
    }
    let mut done = Vec::new();
    if let Some((_, rx)) = &n.results {
        while let Ok(r) = rx.try_recv() {
            done.push(r);
        }
    }
    for (id, r) in done {
        match r {
            Ok(s) => {
                if let Some(code) = n.typed.get(&id).cloned() {
                    n.cfg.known.insert(id.clone(), code);
                    save(&n.cfg);
                }
                n.checks.insert(id, Check::Ok(s));
            }
            Err(e) => {
                n.checks.insert(id, Check::Failed(e));
            }
        }
    }
}

/// What `network_peers` answers.
pub fn snapshot() -> serde_json::Value {
    let mut n = net();
    ensure_loaded(&mut n);
    let peers: Vec<_> = n
        .found
        .values()
        .map(|p| {
            serde_json::json!({
                "id": p.id, "name": p.name, "version": p.version, "port": p.port,
                "addresses": p.addrs.iter().map(|a| a.to_string()).collect::<Vec<_>>(),
                "paired": n.cfg.known.contains_key(&p.id),
            })
        })
        .collect();
    serde_json::json!({
        "sharing": n.cfg.share && !test_run(),
        "port": n.listening,
        "name": machine_name(),
        "id": n.cfg.id,
        "peers": peers,
    })
}

/// A connection to a wobble's API: another machine's network port
/// (unlocked with its code) or this one's own loopback port (no code).
pub struct Conn {
    read: BufReader<TcpStream>,
    out: TcpStream,
    id: u64,
}

impl Conn {
    pub fn open(addrs: &[Ipv4Addr], port: u16, code: Option<&str>) -> Result<Conn, String> {
        let mut last = "no address".to_string();
        for a in addrs {
            let sa = SocketAddr::from((*a, port));
            match TcpStream::connect_timeout(&sa, Duration::from_secs(3)) {
                Ok(s) => {
                    let _ = s.set_nodelay(true);
                    let _ = s.set_read_timeout(Some(Duration::from_secs(120)));
                    let read = BufReader::new(s.try_clone().map_err(|e| e.to_string())?);
                    let mut c = Conn { read, out: s, id: 0 };
                    if let Some(code) = code {
                        c.call("auth", serde_json::json!({ "key": code }))?;
                    }
                    return Ok(c);
                }
                Err(e) => last = format!("{sa}: {e}"),
            }
        }
        Err(last)
    }

    pub fn call(&mut self, method: &str, params: serde_json::Value) -> Result<serde_json::Value, String> {
        self.id += 1;
        let mut l = serde_json::json!({ "jsonrpc": "2.0", "id": self.id, "method": method, "params": params }).to_string();
        l.push('\n');
        self.out.write_all(l.as_bytes()).map_err(|e| e.to_string())?;
        let mut resp = String::new();
        if self.read.read_line(&mut resp).map_err(|e| e.to_string())? == 0 {
            return Err("the other wobble closed the connection".into());
        }
        let v: serde_json::Value = serde_json::from_str(&resp).map_err(|e| e.to_string())?;
        match v.get("error") {
            Some(e) => Err(e["message"].as_str().unwrap_or("refused").to_string()),
            None => Ok(v["result"].clone()),
        }
    }
}

/// Connect to a peer with a code, unlock it and ask what it is showing.
/// Blocking; runs on a thread.
pub fn check_peer(addrs: &[Ipv4Addr], port: u16, code: &str) -> Result<String, String> {
    let mut c = Conn::open(addrs, port, Some(code))?;
    let info = c.call("app_info", serde_json::json!({}))?;
    let setup = info["setup"]
        .as_str()
        .and_then(|p| std::path::Path::new(p).file_name().map(|f| f.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "no setup saved".into());
    Ok(format!("connected · {setup} · {} nodes", info["nodes"]))
}

/// A paired peer — one heard now whose code worked before — by id.
pub fn peer(id: &str) -> Option<(Peer, String)> {
    let n = net();
    let p = n.found.values().find(|p| p.id == id)?.clone();
    let code = n.cfg.known.get(id)?.clone();
    Some((p, code))
}

pub fn name() -> String {
    machine_name()
}

/// Paired wobbles may render here (`farm`): the user's second switch.
pub fn render_allowed() -> bool {
    let mut n = net();
    ensure_loaded(&mut n);
    n.cfg.share && n.cfg.render && !test_run()
}

/// Where received renders land, if this wobble accepts them.
pub fn inbox() -> Option<PathBuf> {
    let mut n = net();
    ensure_loaded(&mut n);
    if !(n.cfg.share && n.cfg.accept) || test_run() {
        return None;
    }
    Some(inbox_of(&n.cfg))
}

fn inbox_of(cfg: &Config) -> PathBuf {
    let t = cfg.inbox.trim();
    if t.is_empty() { default_inbox() } else { engine::paths::expand_home(t) }
}

fn default_inbox() -> PathBuf {
    crate::home_dir().unwrap_or_else(std::env::temp_dir).join("Wobble Received")
}

/// Wobbles heard now whose code worked before.
pub fn paired() -> Vec<Peer> {
    let mut n = net();
    ensure_loaded(&mut n);
    n.found.values().filter(|p| n.cfg.known.contains_key(&p.id)).cloned().collect()
}

/// A line under a peer in the window (a send's outcome).
pub fn note_peer(id: &str, s: String) {
    let mut n = net();
    let r = if s.starts_with("sent") { Check::Ok(s) } else { Check::Failed(s) };
    n.checks.insert(id.to_string(), r);
}

/// What the window asks the app to do (it has no hold on the comp).
pub enum Ask {
    /// Send the selected nodes to this peer.
    Send(String),
    /// Render the selected Render node with these peers (ids) and this one.
    Render(Vec<String>),
}

/// What the app tells the window about the comp.
pub struct Here {
    /// How many nodes are selected.
    pub selected: usize,
    /// The selected Render node's name, if one is selected.
    pub render: Option<String>,
}

/// The window (preferences → network…). Returns what the user asked for.
pub fn window(ctx: &egui::Context, here: &Here) -> Vec<Ask> {
    let mut asks = Vec::new();
    let mut n = net();
    if !n.open {
        return asks;
    }
    ensure_loaded(&mut n);
    let mut open = true;
    egui::Window::new("network").open(&mut open).default_width(420.0).show(ctx, |ui| {
        let mut share = n.cfg.share;
        if ui
            .checkbox(&mut share, "share this wobble on the network")
            .on_hover_text(
                "announce this wobble on the local network and let a paired wobble\n\
                 drive it through the scripting API (it can open and save files)",
            )
            .changed()
        {
            n.cfg.share = share;
            save(&n.cfg);
        }
        if n.cfg.share {
            ui.horizontal(|ui| {
                ui.label("pairing code");
                ui.label(egui::RichText::new(&n.cfg.code).monospace().strong());
                if ui.small_button("new code").on_hover_text("anyone paired with the old code is locked out").clicked() {
                    n.cfg.code = new_code();
                    save(&n.cfg);
                    n.restart = true;
                }
            });
            match n.listening {
                Some(p) => ui.weak(format!("{} · listening on port {p}", machine_name())),
                None => ui.weak("starting…"),
            };
            let mut render = n.cfg.render;
            if ui
                .checkbox(&mut render, "let paired wobbles render here")
                .on_hover_text(
                    "a paired wobble may ask this machine to render frames of its Render node:\n\
                     only from a saved setup in a folder both machines can see, never one with\n\
                     Blender, process-backed or GenAI nodes, in a hidden worker that leaves\n\
                     this comp alone",
                )
                .changed()
            {
                n.cfg.render = render;
                save(&n.cfg);
            }
            let mut accept = n.cfg.accept;
            if ui
                .checkbox(&mut accept, "accept renders from paired wobbles")
                .on_hover_text(
                    "a paired wobble may send the frames its Render node wrote (\"send to…\");\n\
                     only picture and movie files, saved under the folder below,\n\
                     in <sender>/<render name>",
                )
                .changed()
            {
                n.cfg.accept = accept;
                save(&n.cfg);
            }
            if n.cfg.accept {
                ui.horizontal(|ui| {
                    ui.label("into");
                    let hint = default_inbox().display().to_string();
                    if ui
                        .add(egui::TextEdit::singleline(&mut n.cfg.inbox).hint_text(hint).desired_width(240.0))
                        .lost_focus()
                    {
                        save(&n.cfg);
                    }
                    if ui.small_button("reveal").clicked() {
                        super::deliver::reveal(&inbox_of(&n.cfg));
                    }
                });
            }
        } else {
            ui.weak("not shared: other wobbles cannot see or reach this one");
        }
        if let Some(e) = &n.error {
            ui.colored_label(crate::theme::error(), e);
        }
        ui.separator();
        ui.label(egui::RichText::new("wobbles nearby").weak());
        if n.found.is_empty() {
            ui.weak("none heard yet (they must have sharing on)");
        }
        let peers: Vec<Peer> = n.found.values().cloned().collect();
        for p in peers {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(&p.name).strong());
                let addr = p.addrs.first().map(|a| a.to_string()).unwrap_or_else(|| "?".into());
                ui.weak(format!("{addr}:{} · {}", p.port, p.version));
            });
            ui.horizontal(|ui| {
                let known = n.cfg.known.get(&p.id).cloned().unwrap_or_default();
                let typed = n.typed.entry(p.id.clone()).or_insert(known);
                ui.add(egui::TextEdit::singleline(typed).hint_text("their code").desired_width(110.0).font(egui::TextStyle::Monospace));
                let code = typed.trim().to_uppercase();
                let running = matches!(n.checks.get(&p.id), Some(Check::Running));
                if ui.add_enabled(!running && !code.is_empty(), egui::Button::new("check")).clicked() {
                    let tx = n.results.get_or_insert_with(mpsc::channel).0.clone();
                    n.checks.insert(p.id.clone(), Check::Running);
                    let (id, addrs, port) = (p.id.clone(), p.addrs.clone(), p.port);
                    crate::bg::spawn(move || {
                        let _ = tx.send((id, check_peer(&addrs, port, &code)));
                    });
                }
                let paired = n.cfg.known.contains_key(&p.id);
                if paired
                    && here.selected > 0
                    && ui
                        .button(format!("send {} node{}", here.selected, if here.selected == 1 { "" } else { "s" }))
                        .on_hover_text("copy the selected nodes into that wobble's graph")
                        .clicked()
                {
                    asks.push(Ask::Send(p.id.clone()));
                }
                match n.checks.get(&p.id) {
                    Some(Check::Running) => {
                        ui.weak("checking…");
                    }
                    Some(Check::Ok(s)) => {
                        ui.label(s);
                    }
                    Some(Check::Failed(e)) => {
                        ui.colored_label(crate::theme::error(), e);
                    }
                    None => {}
                }
            });
            ui.add_space(4.0);
        }
        ui.separator();
        ui.label(egui::RichText::new("help render").weak());
        let helpers: Vec<Peer> = n.found.values().filter(|p| n.cfg.known.contains_key(&p.id)).cloned().collect();
        match &here.render {
            None => {
                ui.weak("select a Render node to share its frames with paired wobbles");
            }
            Some(name) => {
                let mine = env!("CARGO_PKG_VERSION");
                let mut ids = Vec::new();
                for p in &helpers {
                    let ok = p.version == mine;
                    let on = n.helpers.entry(p.id.clone()).or_insert(ok);
                    ui.add_enabled_ui(ok, |ui| {
                        ui.checkbox(on, p.name.as_str()).on_disabled_hover_text(format!(
                            "{} runs {}, this one {mine}: a helper must run the same version",
                            p.name, p.version
                        ));
                    });
                    if ok && *on {
                        ids.push(p.id.clone());
                    }
                }
                if helpers.is_empty() {
                    ui.weak("no paired wobbles heard: check one above first");
                }
                let label = format!("render {name} with {} helper{}", ids.len(), if ids.len() == 1 { "" } else { "s" });
                if ui
                    .add_enabled(!super::farm::busy(), egui::Button::new(label))
                    .on_hover_text(
                        "split the frames between this wobble and the helpers: the setup must be saved\n\
                         in a folder they can see, each renders in the background and sends its frames here",
                    )
                    .clicked()
                {
                    asks.push(Ask::Render(ids));
                }
            }
        }
        super::farm::progress_ui(ui);
        super::deliver::received_ui(ui);
    });
    if !open {
        n.open = false;
    }
    if n.open || n.cfg.share {
        ctx.request_repaint_after(Duration::from_millis(500));
    }
    asks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_readable_and_differ() {
        let a = new_code();
        assert_eq!(a.len(), 9);
        assert_eq!(&a[4..5], "-");
        assert!(a.chars().all(|c| c == '-' || "ABCDEFGHJKMNPQRSTUVWXYZ23456789".contains(c)), "{a}");
        assert_ne!(a, new_code());
    }

    /// The real listener and the real check, on loopback: the right code
    /// connects and reports, the wrong one is refused in words.
    #[test]
    fn check_unlocks_a_listener_with_the_code() {
        let (host, handle) = api::server::serve_network(0, "ABCD-2345".into()).unwrap();
        let rx = host.rx;
        std::thread::spawn(move || {
            for c in rx {
                c.answer(Ok(serde_json::json!({ "nodes": 3, "setup": "/x/shot.wbl" })));
            }
        });
        let ok = check_peer(&[Ipv4Addr::LOCALHOST], host.port, "ABCD-2345").unwrap();
        assert_eq!(ok, "connected · shot.wbl · 3 nodes");
        let bad = check_peer(&[Ipv4Addr::LOCALHOST], host.port, "WRONG-0000").unwrap_err();
        assert!(bad.contains("pairing code"), "{bad}");
        handle.close();
    }
}
