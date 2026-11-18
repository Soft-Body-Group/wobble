//! Makes the `scrub` binary an app that starts on its own:
//!
//! - Windows: scrub.exe carries its icon (`icons/scrub.ico`).
//! - Linux: a dev build against FFmpeg in a private prefix (pop-os's
//!   `~/.local/ffmpeg`, because the distro's 4.4 is too old) gets that
//!   prefix's lib dir as RUNPATH, so `./target/release/scrub` starts with no
//!   `LD_LIBRARY_PATH`. A release build already passes
//!   `-rpath,$ORIGIN/lib` (tools/release_public_linux.sh) and is left alone,
//!   so no build machine path ever ships; so is FFmpeg in a system dir.

fn main() {
    // Host-gated like its build-dependency: a Windows build is built on
    // Windows (CI, release_public_win.ps1).
    #[cfg(windows)]
    {
        println!("cargo:rerun-if-changed=icons/scrub.ico");
        println!("cargo:rerun-if-changed=icons/scrub.rc");
        let _ = embed_resource::compile("icons/scrub.rc", embed_resource::NONE).manifest_optional();
    }
    #[cfg(target_os = "linux")]
    linux_rpath();
}

#[cfg(target_os = "linux")]
fn linux_rpath() {
    println!("cargo:rerun-if-env-changed=FFMPEG_DIR");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    let flags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    if flags.contains("$ORIGIN") {
        return;
    }
    // Where ffmpeg-sys-next found the libraries: FFMPEG_DIR, else pkg-config.
    let dir = match std::env::var("FFMPEG_DIR") {
        Ok(d) if !d.is_empty() => Some(format!("{d}/lib")),
        _ => std::process::Command::new("pkg-config")
            .args(["--variable=libdir", "libavcodec"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()),
    };
    let Some(dir) = dir.filter(|d| !d.is_empty()) else { return };
    if ["/usr/lib", "/usr/lib64", "/lib", "/lib64", "/usr/local/lib"]
        .iter()
        .any(|sys| dir == *sys || dir.starts_with(&format!("{sys}/")))
    {
        return;
    }
    if std::path::Path::new(&dir).is_dir() {
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{dir}");
    }
}
