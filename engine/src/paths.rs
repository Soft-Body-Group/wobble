//! Path text as a person types it.
//!
//! Anywhere Wobble takes a path from a text field rather than from a file
//! dialog, `~` has to mean what it means in every shell and every other
//! Mac app. Typing it and getting "no such file or directory" is the kind
//! of small wrongness that makes an app feel unfinished.

use std::path::PathBuf;

/// The user's home directory. `USERPROFILE` is the Windows spelling.
pub fn home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

/// Expand a LEADING `~` against the home directory.
///
/// Only leading, and only as a whole path segment: `~/plates` and a bare
/// `~` expand, `~sam/plates` and `/tmp/~/x` do not. Another user's home
/// needs a password-database lookup we do not do, and guessing it by
/// gluing a name onto our own home's parent is how you end up silently
/// reading the wrong directory — so it is left alone to fail honestly.
///
/// Whitespace around the path goes too: a path pasted from a terminal or
/// a chat window usually arrives with some.
pub fn expand_home(input: &str) -> PathBuf {
    let s = input.trim();
    let rest = match s.strip_prefix('~') {
        Some(r) => r,
        None => return PathBuf::from(s),
    };
    // `~` alone, or `~/…`. Anything else after the tilde is a user name.
    let tail = match rest {
        "" => "",
        r if r.starts_with('/') || r.starts_with('\\') => &r[1..],
        _ => return PathBuf::from(s),
    };
    match home() {
        Some(h) if tail.is_empty() => h,
        Some(h) => h.join(tail),
        None => PathBuf::from(s),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run `f` with HOME set to `home`, restoring it afterwards. The env is
    /// process-wide, so these cases live in ONE test rather than racing.
    fn with_home(home: &str, f: impl FnOnce()) {
        let old = std::env::var_os("HOME");
        let old_win = std::env::var_os("USERPROFILE");
        // SAFETY: single-threaded test body; both vars restored below.
        unsafe {
            std::env::set_var("HOME", home);
            std::env::remove_var("USERPROFILE");
        }
        f();
        unsafe {
            match old {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
            if let Some(v) = old_win {
                std::env::set_var("USERPROFILE", v);
            }
        }
    }

    #[test]
    fn tilde_expands_only_where_it_means_home() {
        with_home("/Users/someone", || {
            // Compared as PATHS, not strings: on Windows `join` adds a `\`,
            // and `/Users/someone\plates/shot01` is the same path there as
            // the forward-slash spelling (both are separators), so a string
            // compare would fail on Windows for a path that is correct.
            let e = |s: &str| expand_home(s);
            let same = |got: PathBuf, want: &str| assert_eq!(got, PathBuf::from(want), "{want}");
            // The cases that must work.
            same(e("~"), "/Users/someone");
            same(e("~/"), "/Users/someone");
            same(e("~/plates/shot01"), "/Users/someone/plates/shot01");
            // Pasted paths carry whitespace.
            same(e("  ~/plates  "), "/Users/someone/plates");

            // ANOTHER user's home is not ours to guess: left alone so it
            // fails visibly rather than reading the wrong directory.
            same(e("~bob/plates"), "~bob/plates");
            // A tilde that is not the first thing is just a character —
            // it is legal in a filename.
            same(e("/Volumes/borg/~/x"), "/Volumes/borg/~/x");
            same(e("/Volumes/borg/a~b"), "/Volumes/borg/a~b");
            // Ordinary paths are untouched.
            same(e("/Volumes/borg"), "/Volumes/borg");
            same(e("relative/dir"), "relative/dir");
            same(e(""), "");
        });
    }
}

/// The scratch directory: `std::env::temp_dir()` natively. That PANICS in a
/// browser (no filesystem), so a page gets a path that exists nowhere —
/// anything written there fails like any other file operation in a page.
pub fn temp_dir() -> PathBuf {
    #[cfg(not(target_arch = "wasm32"))]
    return std::env::temp_dir();
    #[cfg(target_arch = "wasm32")]
    return PathBuf::from("/tmp");
}
