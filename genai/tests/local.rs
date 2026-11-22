//! A local model (`Service::Local`) end to end, with a stand-in for the
//! tool (`WOBBLE_LTX_CMD`): no GPU, no weights. What is checked is the part
//! that is easy to get quietly wrong — the step is neither billed nor sent
//! anywhere, the start frame goes as a PATH, an empty image port is
//! text-to-video, the clip lands as the take, and a re-run is a cache hit
//! that starts nothing.
#![cfg(unix)]

use genai::comfy::Comfy;
use genai::pipeline::{Op, Pipeline};
use genai::run::{self, Event, Inputs, Png, Services};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::time::Duration;

const MODEL: &str = "local/ltx-2.5-distilled";

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("wobble-genai-local-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The stand-in: appends each job it is given to `jobs.log` and writes a
/// few bytes as the clip. One for the whole test binary (the variable is
/// process-wide), logging beside the job, so each test reads its own.
fn stand_in() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let d = std::env::temp_dir().join(format!("wobble-genai-local-tool-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let sh = d.join("fake_ltx.sh");
        std::fs::write(
            &sh,
            r#"while [ $# -gt 0 ]; do case "$1" in --job) job="$2"; shift;; --out) out="$2"; shift;; esac; shift; done
log="$(dirname "$(dirname "$job")")/jobs.log"
cat "$job" >> "$log"; echo >> "$log"
printf 'not really an mp4' > "$out"
echo "WOBBLE done"
"#,
        )
        .unwrap();
        std::env::set_var("WOBBLE_LTX_CMD", format!("sh {}", sh.display()));
    });
}

fn drain(job: &run::Job) -> Result<genai::Take, String> {
    loop {
        match job.rx.recv_timeout(Duration::from_secs(20)).expect("run stalled") {
            Event::Done(t) => return Ok(t),
            Event::Failed(e) => return Err(e),
            Event::Step(..) | Event::Take(_) | Event::Text(..) => {}
        }
    }
}

fn services() -> Services {
    // No fal key, no relay: a local step needs neither.
    Services { remote: None, comfy: Comfy::new("http://127.0.0.1:9") }
}

fn jobs(dir: &Path) -> Vec<Value> {
    std::fs::read_to_string(dir.join("jobs.log"))
        .unwrap_or_default()
        .split("\n\n")
        .filter(|s| !s.trim().is_empty())
        .map(|s| serde_json::from_str(s.trim()).unwrap())
        .collect()
}

fn prompt(p: &mut Pipeline, text: &str) {
    for s in &mut p.steps {
        if let Op::Prompt { text: t } = &mut s.op {
            *t = text.into();
        }
    }
}

#[test]
fn image_to_video_sends_the_start_frame_as_a_path_and_a_rerun_starts_nothing() {
    stand_in();
    let dir = tmp("i2v");
    let mut p = Pipeline::for_model(MODEL).unwrap();
    prompt(&mut p, "a slow push in");
    let mut ins = Inputs { comp: (1920, 1080), tick: 1001, ..Default::default() };
    ins.pics.insert(0, Png { bytes: genai::image::png_srgb_rgb(&[0.18; 4 * 4 * 4], 4, 4).unwrap(), w: 4, h: 4 });

    let plan = run::plan(&p, &ins, &dir).unwrap();
    assert!(plan.billed(&p).is_empty(), "a local step is not billed");
    let take = drain(&run::start(p.clone(), plan, ins.clone(), services(), dir.clone(), vec![])).unwrap();
    assert_eq!(take.file, "result.mp4");
    assert_eq!(take.usd, None, "nothing charged");
    assert!(!take.commercial, "LTX is tagged");
    assert_eq!(std::fs::read(dir.join(&take.key).join("result.mp4")).unwrap(), b"not really an mp4");

    let sent = jobs(&dir);
    assert_eq!(sent.len(), 1);
    let j = &sent[0];
    assert_eq!(j["prompt"], "a slow push in");
    let image = j["image"].as_str().expect("the start frame, as a path");
    assert!(Path::new(image).is_file() && !image.starts_with("data:"), "{image}");
    assert_eq!((j["width"].as_u64(), j["height"].as_u64()), (Some(1280), Some(720)));

    // Nothing changed: cached, and the tool is not started again.
    let plan = run::plan(&p, &ins, &dir).unwrap();
    assert!(plan.todo.is_empty());
    let again = drain(&run::start(p, plan, ins, services(), dir.clone(), vec![])).unwrap();
    assert_eq!(again.key, take.key);
    assert_eq!(jobs(&dir).len(), 1);
    // The work directory is gone; only the take is left.
    assert!(!std::fs::read_dir(&dir).unwrap().flatten().any(|e| e.file_name().to_string_lossy().starts_with(".local-")));
}

#[test]
fn an_empty_image_port_is_text_to_video() {
    stand_in();
    let dir = tmp("t2v");
    let mut p = Pipeline::for_model(MODEL).unwrap();
    prompt(&mut p, "fog rolling over a harbour at dawn");
    let gen = p.steps.iter().position(|s| matches!(s.op, Op::Generate { .. })).unwrap();
    p.steps[gen].inputs[1] = None;
    let ins = Inputs { comp: (1080, 1920), tick: 1, ..Default::default() };
    let plan = run::plan(&p, &ins, &dir).unwrap();
    drain(&run::start(p, plan, ins, services(), dir.clone(), vec![])).unwrap();
    let j = &jobs(&dir)[0];
    assert_eq!(j["image"], Value::Null);
    assert_eq!((j["width"].as_u64(), j["height"].as_u64()), (Some(720), Some(1280)), "the comp's aspect");
}
