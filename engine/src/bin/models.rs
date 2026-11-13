//! Model cache CLI — the same registry and fetcher the app's Models panel
//! drives, without a window.
//!
//! usage:
//!   models list                 what exists, and what is already cached
//!   models where                print the cache directory
//!   models get <id> [<id>...]   download (resumes a part file), verify, cache
//!   models hash <id|path>       SHA-256 of a cached model or any file —
//!                               what you paste into CATALOG's `sha256`
//!   models inspect <id> [sub]   parse a checkpoint and list its tensors,
//!                               optionally only those whose name holds `sub`
//!   models rm <id>              drop a cached model (and any part file)
//!
//! `models get` is the headless half of the feature: CI, a Dockerfile or a
//! setup script can warm the cache without launching the compositor.

use engine::models::{self, ModelSpec, State};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("list");
    let rest = &args[args.len().min(1)..];
    let code = match cmd {
        "list" => list(),
        "where" => {
            println!("{}", models::models_dir().display());
            0
        }
        "get" => get(rest),
        "hash" => hash(rest),
        "inspect" => inspect(rest),
        "rm" => rm(rest),
        "-h" | "--help" | "help" => {
            usage();
            0
        }
        other => {
            eprintln!("unknown command {other:?}\n");
            usage();
            2
        }
    };
    std::process::exit(code);
}

fn usage() {
    eprintln!(
        "usage: models [list|where|get <id>...|hash <id|path>|inspect <id>|rm <id>...]\n\
         \n\
         cache dir: {} (override with $WOBBLE_MODELS_DIR)",
        models::models_dir().display()
    );
}

fn list() -> i32 {
    println!("cache: {}\n", models::models_dir().display());
    for m in models::CATALOG {
        let status = match models::state(m) {
            State::Ready => "cached".to_string(),
            State::Missing => "not downloaded".to_string(),
            State::Partial { have } => format!(
                "partial {} of {}",
                models::human_bytes(have),
                m.size_label()
            ),
            State::Bad(why) => format!("BAD: {why}"),
        };
        println!("{:<20} {:>9}  {}", m.id, m.size_label(), status);
        println!("  {} — {}", m.label, m.about);
        println!("  {} · {}", m.license, m.source);
        if !m.verifiable() {
            println!("  (no sha256 pinned — run `models hash {}` after fetching)", m.id);
        }
        println!();
    }
    0
}

fn resolve(id: &str) -> Option<&'static ModelSpec> {
    match models::spec(id) {
        Some(m) => Some(m),
        None => {
            eprintln!("no model {id:?}; `models list` shows what there is");
            None
        }
    }
}

fn get(ids: &[String]) -> i32 {
    if ids.is_empty() {
        eprintln!("get what? `models list` shows the ids");
        return 2;
    }
    let mut bad = 0;
    for id in ids {
        let Some(spec) = resolve(id) else {
            bad = 1;
            continue;
        };
        if models::state(spec) == State::Ready {
            println!("{id}: already cached at {}", spec.path().display());
            continue;
        }
        println!(
            "{id}: {} from {}\n  {} · {}",
            spec.size_label(),
            spec.url,
            spec.license,
            spec.source
        );
        let mut dl = models::fetch(spec);
        // A CLI wants a line that moves, not a repaint loop: print on each
        // whole percent so a slow link still shows life without spamming.
        let mut shown = u64::MAX;
        while dl.poll() {
            let pct = dl.fraction().map(|f| (f * 100.0) as u64).unwrap_or(0);
            if pct != shown {
                shown = pct;
                print!(
                    "\r  {pct:>3}%  {} / {}",
                    models::human_bytes(dl.have),
                    models::human_bytes(dl.total)
                );
                use std::io::Write as _;
                let _ = std::io::stdout().flush();
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        println!();
        match &dl.error {
            Some(e) => {
                eprintln!("  failed: {e}");
                bad = 1;
            }
            None => {
                println!("  cached at {}", spec.path().display());
                if !spec.verifiable() {
                    match models::hash_file(&spec.path()) {
                        Ok(h) => println!("  sha256 {h}  <- paste into CATALOG"),
                        Err(e) => eprintln!("  hash failed: {e}"),
                    }
                }
            }
        }
    }
    bad
}

fn hash(args: &[String]) -> i32 {
    let Some(what) = args.first() else {
        eprintln!("hash what? an id, or a path to any file");
        return 2;
    };
    let path = match models::spec(what) {
        Some(m) => m.path(),
        None => Path::new(what).to_path_buf(),
    };
    match models::hash_file(&path) {
        Ok(h) => {
            println!("{h}  {}", path.display());
            0
        }
        Err(e) => {
            eprintln!("{e:#}");
            1
        }
    }
}

/// Parse a cached checkpoint and print what is in it — the first thing you
/// want when porting a net, and the check that a download is actually
/// READABLE rather than merely the right size.
fn inspect(args: &[String]) -> i32 {
    let Some(what) = args.first() else {
        eprintln!("inspect what? an id, or a path to a .npy checkpoint");
        return 2;
    };
    let path = match models::spec(what) {
        Some(m) => m.path(),
        None => Path::new(what).to_path_buf(),
    };
    let t0 = std::time::Instant::now();
    let ck = match engine::npy::read(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e:#}");
            return 1;
        }
    };
    let names = ck.names();
    println!(
        "{}\n{} tensors, {} parameters, parsed in {:.2}s",
        path.display(),
        names.len(),
        ck.total_params(),
        t0.elapsed().as_secs_f32()
    );
    let filter = args.get(1).map(String::as_str);
    for n in names {
        if filter.is_some_and(|f| !n.contains(f)) {
            continue;
        }
        println!("  {:<70} {:?}", n, ck.tensors[n].dims);
    }
    0
}

fn rm(ids: &[String]) -> i32 {
    let mut bad = 0;
    for id in ids {
        let Some(spec) = resolve(id) else {
            bad = 1;
            continue;
        };
        for p in [spec.path(), spec.part_path()] {
            match std::fs::remove_file(&p) {
                Ok(()) => println!("removed {}", p.display()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    eprintln!("{}: {e}", p.display());
                    bad = 1;
                }
            }
        }
    }
    bad
}
