//! After a save: hand the clip on. Nothing here uploads anything or signs
//! in anywhere — the OS share sheet, the file manager, the clipboard, or a
//! site's own compose page opened in the browser (the user attaches the
//! file there). No accounts, no keys.

use std::path::{Path, PathBuf};

use super::Saved;

/// Compose pages that take text in the URL. The user attaches the clip.
pub const SITES: &[(&str, &str)] = &[
    ("Bluesky", "https://bsky.app/intent/compose"),
];

/// Where clips are saved: `WOBBLE_RECORD_DIR`, else the folder picked in
/// the recorder's options, else `default_dir`.
pub fn save_dir() -> PathBuf {
    super::picked_dir(false).unwrap_or_else(default_dir)
}

/// The platform's videos folder, under Wobble.
pub fn default_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let videos = if cfg!(target_os = "macos") { "Movies" } else { "Videos" };
    home.join(videos).join("Wobble")
}

/// `wobble-2026-10-15-231504`, local time where we can get it cheaply
/// (UTC otherwise; it only has to sort and not collide).
pub fn stamp_name() -> String {
    let secs = web_time::SystemTime::now()
        .duration_since(web_time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    // Civil from days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("wobble-{y:04}-{m:02}-{d:02}-{:02}{:02}{:02}", rem / 3600, rem % 3600 / 60, rem % 60)
}

/// Show the file in the platform's file manager.
pub fn reveal(path: &Path) {
    #[cfg(target_os = "macos")]
    let r = std::process::Command::new("open").arg("-R").arg(path).spawn();
    #[cfg(target_os = "windows")]
    let r = {
        let mut arg = std::ffi::OsString::from("/select,");
        arg.push(path);
        std::process::Command::new("explorer").arg(arg).spawn()
    };
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let r = std::process::Command::new("xdg-open").arg(path.parent().unwrap_or(path)).spawn();
    if let Err(e) = r {
        log::warn!("revealing {}: {e}", path.display());
    }
}

/// Put the FILE on the clipboard (paste it into a post, a chat, Finder).
#[cfg(target_arch = "wasm32")]
pub fn copy_file(_path: &Path) -> Result<(), String> {
    Err("no clipboard for files in a browser".into())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn copy_file(path: &Path) -> Result<(), String> {
    let mut cb = arboard::Clipboard::new().map_err(|e| e.to_string())?;
    cb.set().file_list(&[path]).map_err(|e| e.to_string())
}

/// egui temp key the share panel's "blog…" button leaves the clip under,
/// for the app to open its submit dialog on (`blogpost.rs`).
pub const BLOG_REQ: &str = "rec-blog";

/// The share window's lower half, once there is a file.
pub fn panel(
    ui: &mut egui::Ui,
    saved: &Saved,
    anchor: &mut Option<egui::Rect>,
    window: Option<&winit::window::Window>,
) {
    let file = saved.mp4.as_ref().or(saved.gif.as_ref());
    for p in [&saved.mp4, &saved.gif].into_iter().flatten() {
        let size = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        ui.horizontal(|ui| {
            ui.label(p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
            ui.weak(format!("{:.1} MB", size as f64 / 1e6));
        });
    }
    if let Some(n) = &saved.note {
        ui.weak(n);
    }
    let Some(file) = file else { return };
    ui.horizontal(|ui| {
        #[cfg(target_os = "macos")]
        {
            let r = ui.button("share…").on_hover_text("the system share sheet: Messages, Mail, AirDrop, …");
            if r.clicked() {
                *anchor = Some(r.rect);
                if let Some(window) = window {
                    share_sheet(file, r.rect, ui.ctx().pixels_per_point(), window);
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        let _ = (&anchor, window);
        if ui
            .button("blog…")
            .on_hover_text("submit it to the wobble blog at wobble.ing/wobbles/; it appears once approved")
            .clicked()
        {
            ui.ctx().data_mut(|d| d.insert_temp(egui::Id::new(BLOG_REQ), file.clone()));
        }
        if ui.button("reveal").on_hover_text("show it in the file manager").clicked() {
            reveal(file);
        }
        let copied = egui::Id::new("rec-copied");
        let r = ui.button("copy").on_hover_text("put the file on the clipboard, to paste into a post or a chat");
        if r.clicked() {
            let ok = copy_file(file);
            ui.ctx().data_mut(|d| d.insert_temp(copied, ok.clone().err()));
            if let Err(e) = ok {
                log::warn!("copying {}: {e}", file.display());
            }
        }
        if let Some(res) = ui.ctx().data(|d| d.get_temp::<Option<String>>(copied)) {
            match res {
                None => ui.weak("copied"),
                Some(e) => ui.colored_label(crate::theme::error(), e),
            };
        }
    });
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        for (name, base) in SITES {
            if ui
                .button(*name)
                .on_hover_text(format!("open {name}'s compose page — attach the clip there"))
                .clicked()
            {
                crate::open_url(base);
                reveal(file);
            }
        }
    });
    ui.weak("opens in your browser with the file shown beside it to drop in; wobble uploads nothing except to its own blog, when you send it");
}

/// NSSharingServicePicker anchored on the button. The picker is kept
/// alive (one at a time) while AppKit shows it.
#[cfg(target_os = "macos")]
fn share_sheet(path: &Path, rect: egui::Rect, ppp: f32, window: &winit::window::Window) {
    use objc2::encode::{Encode, Encoding};
    use objc2::runtime::{AnyClass, AnyObject, Bool};
    use objc2::msg_send;
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Rect {
        x: f64,
        y: f64,
        w: f64,
        h: f64,
    }
    unsafe impl Encode for Rect {
        const ENCODING: Encoding = Encoding::Struct(
            "CGRect",
            &[
                Encoding::Struct("CGPoint", &[f64::ENCODING, f64::ENCODING]),
                Encoding::Struct("CGSize", &[f64::ENCODING, f64::ENCODING]),
            ],
        );
    }
    static PICKER: std::sync::Mutex<usize> = std::sync::Mutex::new(0);

    let Ok(handle) = window.window_handle() else { return };
    let RawWindowHandle::AppKit(h) = handle.as_raw() else { return };
    let view = h.ns_view.as_ptr() as *mut AnyObject;
    let Ok(cpath) = std::ffi::CString::new(path.to_string_lossy().as_bytes()) else { return };
    let _ = ppp;
    unsafe {
        let (Some(nsstring), Some(nsurl), Some(nsarray), Some(picker_cls)) = (
            AnyClass::get(c"NSString"),
            AnyClass::get(c"NSURL"),
            AnyClass::get(c"NSArray"),
            AnyClass::get(c"NSSharingServicePicker"),
        ) else {
            return;
        };
        let s: *mut AnyObject = msg_send![nsstring, stringWithUTF8String: cpath.as_ptr()];
        let url: *mut AnyObject = msg_send![nsurl, fileURLWithPath: s];
        let items: *mut AnyObject = msg_send![nsarray, arrayWithObject: url];
        let alloc: *mut AnyObject = msg_send![picker_cls, alloc];
        let picker: *mut AnyObject = msg_send![alloc, initWithItems: items];
        if picker.is_null() {
            return;
        }
        // egui points are the view's points; AppKit's y runs up unless the
        // view is flipped.
        let flipped: Bool = msg_send![view, isFlipped];
        let bounds: Rect = msg_send![view, bounds];
        let y = if flipped.as_bool() { rect.min.y as f64 } else { bounds.h - rect.max.y as f64 };
        let r = Rect { x: rect.min.x as f64, y, w: rect.width() as f64, h: rect.height() as f64 };
        // NSMinYEdge = 1 (below the button in a flipped view, above it
        // otherwise — either reads fine).
        let _: () = msg_send![picker, showRelativeToRect: r, ofView: view, preferredEdge: 1usize];
        let mut old = PICKER.lock().unwrap();
        if *old != 0 {
            let _: () = msg_send![*old as *mut AnyObject, release];
        }
        *old = picker as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamps_sort_and_look_like_dates() {
        let s = stamp_name();
        assert!(s.starts_with("wobble-20"), "{s}");
        assert_eq!(s.len(), "wobble-2026-10-15-231504".len());
    }
}
