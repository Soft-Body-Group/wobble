//! Bakes the git revision (env WOBBLE_GIT) and a build stamp (env
//! WOBBLE_STAMP, unix seconds) into the binary, so crash and feedback reports
//! say which build a tester was actually running, and the new-build check can
//! order builds. The stamp is the release time when tools/release_alpha.sh
//! sets WOBBLE_BUILD_TS (two releases from one dirty commit must still
//! order), else the commit time.
//! Absent git (a tarball build) degrades to "unknown" / 0 rather than failing.

fn git(args: &[&str]) -> Option<String> {
    let out = std::process::Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    let mut rev = git(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_default();
    if rev.is_empty() {
        rev = "unknown".into();
    } else if git(&["status", "--porcelain"]).is_some_and(|s| !s.is_empty()) {
        rev.push_str("-dirty");
    }
    let stamp = std::env::var("WOBBLE_BUILD_TS")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| git(&["log", "-1", "--format=%ct"]))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "0".into());
    println!("cargo:rustc-env=WOBBLE_GIT={rev}");
    println!("cargo:rustc-env=WOBBLE_STAMP={stamp}");
    println!("cargo:rerun-if-env-changed=WOBBLE_BUILD_TS");
    // The exe's icon. Host-gated like its build-dependency: a Windows build
    // is built on Windows (CI, release_public_win.ps1).
    #[cfg(windows)]
    {
        println!("cargo:rerun-if-changed=icons/wobble.ico");
        let _ = embed_resource::compile("icons/wobble.rc", embed_resource::NONE).manifest_optional();
    }
    // Re-run when the checked-out commit moves: HEAD for a branch switch,
    // the branch's own ref for a commit on it. A stale -dirty marker between
    // edits is acceptable; a stale commit id is not.
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=../.git/packed-refs");
    if let Ok(head) = std::fs::read_to_string("../.git/HEAD") {
        if let Some(r) = head.trim().strip_prefix("ref: ") {
            println!("cargo:rerun-if-changed=../.git/{r}");
        }
    }
}
