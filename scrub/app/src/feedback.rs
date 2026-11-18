//! "request a feature", red at the far right of the menubar, the way
//! Wobble has it: a line in front of the form first, then a title and a
//! body sent to the same feedback Worker Wobble's reports go to
//! (tools/feedback-worker, `POST /v1/report`, kind "feature"). Scrub's
//! land in the same list, told apart by a "[scrub] " title prefix and a
//! "scrub …" version.
//!
//! The ingest key is baked in at compile time from `WOBBLE_INGEST_KEY`
//! (the release scripts set it, as for Wobble); a build without it says
//! so instead of sending. `WOBBLE_REPORT_URL` / `WOBBLE_REPORT_KEY`
//! override both for development against `wrangler dev`.

use egui::Color32;
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::time::{Duration, SystemTime};

const ENDPOINT: &str = "https://feedback.wobble.ing";
const INGEST_KEY: &str = match option_env!("WOBBLE_INGEST_KEY") {
    Some(k) => k,
    None => "",
};
const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Default)]
pub struct Feedback {
    /// The line in front of the form is up.
    pub gate: bool,
    /// The form is up.
    pub open: bool,
    title: String,
    body: String,
    email: String,
    inflight: Option<Receiver<Result<String, String>>>,
    last: Option<Result<String, String>>,
}

impl Feedback {
    pub fn request(&mut self) {
        self.open = false;
        self.gate = true;
    }

    /// The form or its gate has the keys (Scrub's stand down).
    pub fn up(&self) -> bool {
        self.gate || self.open
    }

    /// The red button, for the menubar's right end.
    pub fn button(&mut self, ui: &mut egui::Ui) {
        let red = egui::RichText::new("request a feature").color(imfram::theme::error());
        if ui.small_button(red).on_hover_text("something Scrub should do").clicked() {
            self.request();
        }
    }

    pub fn windows(&mut self, ctx: &egui::Context) {
        self.gate_window(ctx);
        self.form(ctx);
    }

    fn gate_window(&mut self, ctx: &egui::Context) {
        if !self.gate {
            return;
        }
        egui::Window::new("request a feature")
            .id(egui::Id::new("scrub-feature-gate"))
            .collapsible(false)
            .resizable(false)
            .title_bar(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.add_space(6.0);
                ui.label(egui::RichText::new("I am not going to make it more like that other editing app").size(16.0));
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    if ui.button("oh").clicked() {
                        self.gate = false;
                    }
                    if ui.button("that isn't what i was going to say").clicked() {
                        self.gate = false;
                        self.open = true;
                    }
                });
            });
    }

    fn form(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.inflight {
            match rx.try_recv() {
                Ok(res) => {
                    if res.is_ok() {
                        self.title.clear();
                        self.body.clear();
                    }
                    self.last = Some(res);
                    self.inflight = None;
                }
                Err(TryRecvError::Empty) => ctx.request_repaint_after(Duration::from_millis(150)),
                Err(TryRecvError::Disconnected) => {
                    self.last = Some(Err("send thread died".into()));
                    self.inflight = None;
                }
            }
        }
        if !self.open {
            return;
        }
        let mut open = true;
        let mut send = false;
        egui::Window::new("request a feature")
            .id(egui::Id::new("scrub-feature-form"))
            .open(&mut open)
            .collapsible(false)
            .default_width(440.0)
            .show(ctx, |ui| {
                ui.add(egui::TextEdit::singleline(&mut self.title).hint_text("one-line summary").desired_width(f32::INFINITY));
                ui.add(
                    egui::TextEdit::multiline(&mut self.body)
                        .hint_text("what you want to do, and what gets in the way today")
                        .desired_rows(8)
                        .desired_width(f32::INFINITY),
                );
                ui.horizontal(|ui| {
                    ui.label("email");
                    ui.add(egui::TextEdit::singleline(&mut self.email).hint_text("optional, so we can reply").desired_width(240.0));
                });
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui.button("cancel").clicked() {
                        self.open = false;
                        self.last = None;
                    }
                    let can = self.inflight.is_none() && !self.title.trim().is_empty();
                    if ui.add_enabled(can, egui::Button::new("send")).clicked() {
                        send = true;
                    }
                    if self.inflight.is_some() {
                        ui.weak("sending…");
                    }
                    match &self.last {
                        Some(Ok(_)) => {
                            ui.weak("sent ✔ thank you");
                        }
                        Some(Err(e)) => {
                            ui.colored_label(Color32::from_rgb(230, 80, 70), e);
                        }
                        None => {}
                    }
                });
            });
        if !open {
            self.open = false;
            self.last = None;
        }
        if send {
            self.send();
        }
    }

    fn send(&mut self) {
        let body = payload(self.title.trim(), self.body.trim(), self.email.trim());
        let (tx, rx) = channel();
        self.inflight = Some(rx);
        self.last = None;
        std::thread::Builder::new()
            .name("scrub-feedback".into())
            .spawn(move || {
                let _ = tx.send(post(&body));
            })
            .ok();
    }
}

/// The Worker's report shape (Wobble's `report::Report`, schema 2); the
/// fields Scrub has no value for go empty.
fn payload(title: &str, body: &str, email: &str) -> String {
    let ts = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    serde_json::json!({
        "schema": 2,
        "kind": "feature",
        "ts": ts,
        "version": format!("scrub {VERSION}"),
        "git": "",
        "os": format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
        "gpu": "",
        "tester": email,
        "title": format!("[scrub] {title}"),
        "body": body,
        "logs": "",
        "session": "",
        "install": "",
        "code": "",
        "sig": "",
    })
    .to_string()
}

fn post(body: &str) -> Result<String, String> {
    let url = std::env::var("WOBBLE_REPORT_URL").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| ENDPOINT.into());
    let key = std::env::var("WOBBLE_REPORT_KEY").unwrap_or_else(|_| INGEST_KEY.into());
    if key.is_empty() {
        return Err("this build can't send requests (it has no feedback key): nothing was sent".into());
    }
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(20))).http_status_as_error(false).build().into();
    let mut resp = agent
        .post(format!("{}/v1/report", url.trim_end_matches('/')))
        .header("content-type", "application/json")
        .header("x-wobble-key", &key)
        .send(body)
        .map_err(|e| format!("{e}"))?;
    let status = resp.status().as_u16();
    let text = resp.body_mut().read_to_string().unwrap_or_default();
    if (200..300).contains(&status) {
        Ok(serde_json::from_str::<serde_json::Value>(&text).ok().and_then(|v| v.get("id").and_then(|i| i.as_str()).map(String::from)).unwrap_or_else(|| "?".into()))
    } else {
        Err(format!("HTTP {status}: {}", text.chars().take(200).collect::<String>()))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn requests_are_marked_as_scrubs() {
        let v: serde_json::Value = serde_json::from_str(&super::payload("a thing", "why", "")).unwrap();
        assert_eq!(v["kind"], "feature");
        assert_eq!(v["title"], "[scrub] a thing");
        assert!(v["version"].as_str().unwrap().starts_with("scrub "));
    }

    #[test]
    fn the_button_opens_the_gate_first() {
        let mut f = super::Feedback::default();
        f.request();
        assert!(f.gate && !f.open && f.up());
    }
}
