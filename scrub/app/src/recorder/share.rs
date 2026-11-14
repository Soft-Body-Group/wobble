//! Where clips go, and how they are named. (wobble's share module, without
//! its publishing: no share sheet, no compose pages, no blog — a clip is
//! saved and can be shown in the file manager, nothing more.)

use std::path::{Path, PathBuf};

/// Where clips are saved: `EDIFY_RECORD_DIR`, else the folder picked in
/// the recorder's options, else `default_dir`.
pub fn save_dir() -> PathBuf {
    super::picked_dir(false).unwrap_or_else(default_dir)
}

/// The platform's videos folder, under Scrub (wobble's is under Wobble).
pub fn default_dir() -> PathBuf {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    let videos = if cfg!(target_os = "macos") { "Movies" } else { "Videos" };
    home.join(videos).join("Scrub")
}

/// `scrub-2026-10-15-231504` (UTC; it only has to sort and not collide).
pub fn stamp_name() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
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
    format!("scrub-{y:04}-{m:02}-{d:02}-{:02}{:02}{:02}", rem / 3600, rem % 3600 / 60, rem % 60)
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
