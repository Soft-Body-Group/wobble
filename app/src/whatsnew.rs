//! What's new: the first launch of a newer version lists what changed since
//! the version that ran before it. The list is `WHATSNEW.md`, compiled in, so
//! it needs no network and says exactly what this build holds.
//!
//! The file is `## <version>` headings, newest first, each over `- ` bullets;
//! a bullet may open with a `**bold**` lead, drawn strong.

use std::sync::Mutex;

const TEXT: &str = include_str!("../WHATSNEW.md");

#[derive(Clone, Debug, PartialEq)]
pub struct Section {
    pub version: String,
    pub bullets: Vec<String>,
}

/// Every section in the file, in file order. A line that is neither a
/// heading nor a bullet continues the bullet above it.
pub fn parse(text: &str) -> Vec<Section> {
    let mut out: Vec<Section> = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if let Some(v) = t.strip_prefix("## ") {
            out.push(Section { version: v.trim().trim_start_matches('v').to_string(), bullets: Vec::new() });
        } else if let Some(s) = out.last_mut() {
            if let Some(b) = t.strip_prefix("- ") {
                s.bullets.push(b.trim().to_string());
            } else if !t.is_empty() && !t.starts_with('#') {
                if let Some(b) = s.bullets.last_mut() {
                    b.push(' ');
                    b.push_str(t);
                }
            }
        }
    }
    out.retain(|s| !s.bullets.is_empty());
    out
}

/// What to show on the first run of `now` after `last` ran: every section
/// newer than `last`, up to and including `now`. With no `last` (a version
/// from before this window existed) just `now`'s own section.
pub fn since(sections: &[Section], last: Option<&str>, now: &str) -> Vec<Section> {
    use std::cmp::Ordering::*;
    let cmp = crate::report::semver_cmp;
    sections
        .iter()
        .filter(|s| matches!(cmp(&s.version, now), Some(Less | Equal)))
        .filter(|s| match last {
            Some(l) => cmp(&s.version, l) == Some(Greater),
            None => cmp(&s.version, now) == Some(Equal),
        })
        .cloned()
        .collect()
}

static PENDING: Mutex<Vec<Section>> = Mutex::new(Vec::new());

/// At launch: queue what changed since `last` (the version that ran before).
/// Nothing when this IS `last`.
pub fn arm(last: Option<&str>) {
    if last == Some(crate::report::VERSION) {
        return;
    }
    let news = since(&parse(TEXT), last, crate::report::VERSION);
    if let Ok(mut p) = PENDING.lock() {
        *p = news;
    }
}

/// The window, while anything is queued. Closing it empties the queue.
pub fn show(ctx: &egui::Context) {
    let news = match PENDING.lock() {
        Ok(p) if !p.is_empty() => p.clone(),
        _ => return,
    };
    if window(ctx, &news) {
        if let Ok(mut p) = PENDING.lock() {
            p.clear();
        }
    }
}

/// Draw the list; true once it is dismissed.
pub fn window(ctx: &egui::Context, news: &[Section]) -> bool {
    let mut done = false;
    let title = match news.first() {
        Some(s) => format!("what's new in {}", s.version),
        None => "what's new".to_string(),
    };
    egui::Window::new(title)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .default_width(420.0)
        .show(ctx, |ui| {
            ui.set_max_width(420.0);
            egui::ScrollArea::vertical().max_height(420.0).show(ui, |ui| {
                for (i, s) in news.iter().enumerate() {
                    if news.len() > 1 {
                        if i > 0 {
                            ui.add_space(6.0);
                        }
                        ui.weak(&s.version);
                    }
                    for b in &s.bullets {
                        bullet(ui, b);
                    }
                }
            });
            ui.add_space(6.0);
            if ui.button("ok").clicked() {
                done = true;
            }
        });
    done
}

/// One bullet: a `**lead**` drawn strong, the rest plain, wrapped.
fn bullet(ui: &mut egui::Ui, b: &str) {
    let (lead, rest) = split_lead(b);
    ui.horizontal_top(|ui| {
        ui.label("•");
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            if let Some(l) = lead {
                ui.label(egui::RichText::new(l).strong());
            }
            ui.label(rest);
        });
    });
}

/// `**Lead**: rest` -> (Some("Lead"), ": rest"); anything else is all rest.
fn split_lead(b: &str) -> (Option<&str>, &str) {
    b.strip_prefix("**")
        .and_then(|r| r.split_once("**"))
        .map_or((None, b), |(lead, rest)| (Some(lead), rest))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "# What's new\n\n## 0.1.0-alpha.7\n- **Seven**: one\n  continued\n\n\
                          ## 0.1.0-alpha.6\n- six\n\n## 0.1.0-alpha.5\n- five\n";

    #[test]
    fn sections_and_continuations_parse() {
        let s = parse(SAMPLE);
        assert_eq!(s.len(), 3);
        assert_eq!(s[0].version, "0.1.0-alpha.7");
        assert_eq!(s[0].bullets, vec!["**Seven**: one continued"]);
    }

    #[test]
    fn an_upgrade_lists_every_version_it_skipped_and_nothing_newer() {
        let s = parse(SAMPLE);
        let v = |n: &[Section]| n.iter().map(|s| s.version.clone()).collect::<Vec<_>>();
        assert_eq!(v(&since(&s, Some("0.1.0-alpha.5"), "0.1.0-alpha.7")), ["0.1.0-alpha.7", "0.1.0-alpha.6"]);
        assert_eq!(v(&since(&s, Some("0.1.0-alpha.5"), "0.1.0-alpha.6")), ["0.1.0-alpha.6"]);
        assert!(since(&s, Some("0.1.0-alpha.7"), "0.1.0-alpha.7").is_empty());
        // No record of the last version: this one's own notes only.
        assert_eq!(v(&since(&s, None, "0.1.0-alpha.6")), ["0.1.0-alpha.6"]);
    }

    #[test]
    fn a_bold_lead_splits_off() {
        assert_eq!(split_lead("**Graph**: tidy"), (Some("Graph"), ": tidy"));
        assert_eq!(split_lead("plain"), (None, "plain"));
    }

    #[test]
    fn the_shipped_file_parses() {
        // Every heading must be a version the update check can compare.
        for s in parse(TEXT) {
            assert!(crate::report::semver_cmp(&s.version, "0.0.0").is_some(), "{}", s.version);
        }
    }

    /// The window over the shipped file's newest section, drawn to
    /// `target/whatsnew-gpu.png`.
    #[test]
    #[ignore]
    fn whatsnew_window_render() {
        let news: Vec<Section> = parse(TEXT).into_iter().take(1).collect();
        crate::uishot::render("whatsnew-gpu", 520.0, 560.0, 2.0, 4, |ui, _| {
            window(ui.ctx(), &news);
        });
    }
}
