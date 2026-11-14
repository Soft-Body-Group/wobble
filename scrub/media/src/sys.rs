//! What the machine has: RAM and cores. No dependency for two numbers.

pub fn cores() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

/// Physical RAM in bytes, where we know how to ask.
pub fn ram_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl").args(["-n", "hw.memsize"]).output().ok()?;
        return String::from_utf8_lossy(&out.stdout).trim().parse().ok();
    }
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kb: u64 = s.lines().find(|l| l.starts_with("MemTotal:"))?.split_whitespace().nth(1)?.parse().ok()?;
        return Some(kb * 1024);
    }
    #[allow(unreachable_code)]
    None
}

/// The app's name (it was Edify).
pub const APP: &str = "Scrub";

/// An environment setting: `SCRUB_<key>`, else the old `EDIFY_<key>`.
pub fn var(key: &str) -> Result<String, std::env::VarError> {
    std::env::var(format!("SCRUB_{key}")).or_else(|_| std::env::var(format!("EDIFY_{key}")))
}

/// `var`, as an `OsString`.
pub fn var_os(key: &str) -> Option<std::ffi::OsString> {
    std::env::var_os(format!("SCRUB_{key}")).or_else(|| std::env::var_os(format!("EDIFY_{key}")))
}

fn home() -> std::path::PathBuf {
    std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(std::path::PathBuf::from).unwrap_or_else(std::env::temp_dir)
}

/// `~/Library/Application Support/Scrub` (prefs, recent files, autosave,
/// Wobble comps). The first time, what Edify kept in its own folder is
/// COPIED here (the old folder is left as it was).
pub fn support_dir() -> std::path::PathBuf {
    static DIR: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        let base = home().join("Library/Application Support");
        let new = base.join(APP);
        let old = base.join("Edify");
        if !new.exists() && old.is_dir() {
            let _ = copy_tree(&old, &new);
        }
        new
    })
    .clone()
}

/// `~/Library/Caches/Scrub` (`~/.cache/scrub` elsewhere). The first time,
/// Edify's cache folder is MOVED here (a cache, so moving loses nothing).
pub fn cache_dir() -> std::path::PathBuf {
    static DIR: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        let (new, old) = if cfg!(target_os = "macos") {
            (home().join("Library/Caches").join(APP), home().join("Library/Caches/Edify"))
        } else {
            (home().join(".cache/scrub"), home().join(".cache/edify"))
        };
        if !new.exists() && old.is_dir() {
            let _ = std::fs::rename(&old, &new);
        }
        new
    })
    .clone()
}

fn copy_tree(from: &std::path::Path, to: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for e in std::fs::read_dir(from)? {
        let e = e?;
        let p = e.path();
        if e.file_type()?.is_dir() {
            copy_tree(&p, &to.join(e.file_name()))?;
        } else {
            std::fs::copy(&p, to.join(e.file_name()))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn scrub_settings_win_and_the_old_edify_ones_still_count() {
        std::env::set_var("EDIFY_SYS_TEST_A", "old");
        assert_eq!(super::var("SYS_TEST_A").as_deref(), Ok("old"));
        std::env::set_var("SCRUB_SYS_TEST_A", "new");
        assert_eq!(super::var("SYS_TEST_A").as_deref(), Ok("new"));
        assert!(super::var_os("SYS_TEST_NONE").is_none());
    }
}
