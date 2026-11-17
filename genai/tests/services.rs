//! The runner end to end, against throwaway services on localhost. No
//! network, no keys, no money: what is checked is the part that is easy to
//! get quietly wrong — the queue dance (submit, poll, fetch), the request a
//! step builds, a re-run being a cache hit that sends NOTHING, the relay's
//! approved-price cap, a cancel reaching the provider, and a ComfyUI
//! workflow getting its uploads and placeholders.

use genai::catalog;
use genai::comfy::Comfy;
use genai::fal::Fal;
use genai::http::Http;
use genai::pipeline::{Op, Opts, Pipeline};
use genai::relay::Relay;
use genai::run::{self, Event, Inputs, Png, Services};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// One request as the server saw it.
#[derive(Debug, Clone)]
struct Req {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Req {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }
    fn header(&self, k: &str) -> Option<&str> {
        self.headers.iter().find(|(n, _)| n.eq_ignore_ascii_case(k)).map(|(_, v)| v.as_str())
    }
}

type Handler = Arc<dyn Fn(&Req, u16) -> (u16, &'static str, Vec<u8>) + Send + Sync>;

/// A server that records every request and answers through `handler`.
fn serve(handler: Handler) -> (u16, Arc<Mutex<Vec<Req>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    let log = Arc::new(Mutex::new(vec![]));
    let log2 = log.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(conn) = conn else { continue };
            let h = handler.clone();
            let log = log2.clone();
            std::thread::spawn(move || handle(conn, h, log, port));
        }
    });
    (port, log)
}

fn handle(mut conn: TcpStream, h: Handler, log: Arc<Mutex<Vec<Req>>>, port: u16) {
    let mut reader = BufReader::new(conn.try_clone().unwrap());
    let mut first = String::new();
    if reader.read_line(&mut first).unwrap_or(0) == 0 {
        return;
    }
    let mut parts = first.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let mut headers = vec![];
    let mut len = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = line.trim_end().split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                len = v.trim().parse().unwrap_or(0);
            }
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }
    let mut body = vec![0u8; len];
    let _ = reader.read_exact(&mut body);
    let req = Req { method, path, headers, body };
    log.lock().unwrap().push(req.clone());
    let (status, ct, out) = h(&req, port);
    let head = format!(
        "HTTP/1.1 {status} X\r\nContent-Type: {ct}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        out.len()
    );
    let _ = conn.write_all(head.as_bytes());
    let _ = conn.write_all(&out);
    let _ = conn.flush();
}

fn j(v: Value) -> (u16, &'static str, Vec<u8>) {
    (200, "application/json", serde_json::to_vec(&v).unwrap())
}

/// A tiny real PNG, so the downloaded result is a picture.
fn png_bytes() -> Vec<u8> {
    genai::image::png_srgb_rgb(&[0.18; 4 * 4 * 4], 4, 4).unwrap()
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("wobble-genai-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
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

/// Every take a batch lands, in order (the last one is the `Done`).
fn drain_all(job: &run::Job) -> Result<Vec<genai::Take>, String> {
    let mut takes = vec![];
    loop {
        match job.rx.recv_timeout(Duration::from_secs(20)).expect("run stalled") {
            Event::Take(t) => takes.push(t),
            Event::Done(t) => {
                takes.push(t);
                return Ok(takes);
            }
            Event::Failed(e) => return Err(e),
            Event::Step(..) | Event::Text(..) => {}
        }
    }
}

/// A fal lookalike, shaped after fal's queue, pricing and error docs:
/// submit (request id + the three URLs + queue position), status polls that
/// answer 202 while queued and running (each carrying `request_id` and
/// `response_url`, logs and metrics the way fal sends them), COMPLETED,
/// then the output JSON pointing at an image on the same server.
fn fake_fal() -> (u16, Arc<Mutex<Vec<Req>>>) {
    let polls = Arc::new(Mutex::new(0u32));
    serve(Arc::new(move |r: &Req, port: u16| {
        let base = format!("http://127.0.0.1:{port}");
        let response_url = format!("{base}/fal-ai/flux-pro/requests/abc123");
        match (r.method.as_str(), r.path.as_str()) {
            // A language model: its own request id, and words for a result.
            ("POST", p) if p.contains("any-llm") && !p.contains("/requests/") => j(json!({
                "request_id": "txt1",
                "status_url": format!("{base}/fal-ai/any-llm/requests/txt1/status"),
                "response_url": format!("{base}/fal-ai/any-llm/requests/txt1"),
                "cancel_url": format!("{base}/fal-ai/any-llm/requests/txt1/cancel"),
            })),
            ("GET", "/fal-ai/any-llm/requests/txt1") => j(json!({
                "output": "a moth on a lamp, night, rain on the glass, 85mm",
                "partial": false,
                "error": null
            })),
            ("POST", p) if !p.contains("/requests/") => j(json!({
                "request_id": "abc123",
                // The APP id, not the full endpoint — fal's own shape for
                // an endpoint with a sub-path.
                "status_url": format!("{base}/fal-ai/flux-pro/requests/abc123/status"),
                "response_url": response_url,
                "cancel_url": format!("{base}/fal-ai/flux-pro/requests/abc123/cancel"),
                "queue_position": 2
            })),
            ("GET", p) if p.ends_with("/status") => {
                let mut n = polls.lock().unwrap();
                *n += 1;
                match *n {
                    1 => accepted(json!({"status": "IN_QUEUE", "request_id": "abc123",
                        "response_url": response_url, "queue_position": 1})),
                    2 => accepted(json!({"status": "IN_PROGRESS", "request_id": "abc123",
                        "response_url": response_url,
                        "logs": [{"message": "step 1/28", "timestamp": "2026-10-14T12:00:00Z"}]})),
                    _ => j(json!({"status": "COMPLETED", "request_id": "abc123",
                        "response_url": response_url, "logs": [],
                        "metrics": {"inference_time": 3.2}})),
                }
            }
            ("GET", "/fal-ai/flux-pro/requests/abc123") => j(json!({
                "images": [{"url": format!("{base}/cdn/out.png?sig=s3cret"), "content_type": "image/png",
                            "width": 4, "height": 4}],
                "timings": {"inference": 3.1},
                "has_nsfw_concepts": [false],
                "prompt": "make it night",
                "seed": 7
            })),
            ("GET", p) if p.starts_with("/cdn/") => (200, "image/png", png_bytes()),
            ("GET", p) if p.starts_with("/v1/models/pricing") => j(json!({
                "prices": [
                    {"endpoint_id": "fal-ai/flux-pro/kontext", "unit_price": 0.04, "unit": "image", "currency": "USD"}
                ],
                "next_cursor": null,
                "has_more": false
            })),
            ("PUT", p) if p.ends_with("/cancel") => {
                (202, "application/json", br#"{"status":"CANCELLATION_REQUESTED"}"#.to_vec())
            }
            _ => (404, "application/json", br#"{"detail":"Not Found"}"#.to_vec()),
        }
    }))
}

/// fal answers a status poll that is not finished with 202.
fn accepted(v: Value) -> (u16, &'static str, Vec<u8>) {
    (202, "application/json", serde_json::to_vec(&v).unwrap())
}

/// Reference -> Kontext edit (with a prompt) -> Output.
fn edit_pipeline() -> Pipeline {
    let mut p = Pipeline::default();
    // Re-point the default text-to-image step at the editor.
    p.set_model(1, "fal/flux-kontext-pro");
    let r = p.push(Op::Reference, [0.4, -0.6]);
    assert!(p.connect(0, 1, 0));
    assert!(p.connect(r, 1, 1));
    p
}

fn inputs() -> Inputs {
    Inputs {
        reference: Some(Png { bytes: png_bytes(), w: 4, h: 4 }),
        matte: None,
        prompt: None,
        comp: (2048, 1080),
        tick: 1001,
        ..Default::default()
    }
}

#[test]
fn fal_runs_the_queue_downloads_the_picture_and_a_rerun_sends_nothing() {
    let (port, log) = fake_fal();
    let base = format!("http://127.0.0.1:{port}");
    let dir = tmp("fal");
    let mut p = edit_pipeline();
    if let Op::Prompt { text } = &mut p.steps[0].op {
        *text = "make it night".into();
    }

    let plan = run::plan(&p, &inputs(), &dir).unwrap();
    assert_eq!(plan.billed(&p), vec![1], "one billed call: the editor");

    // The quote comes from the pricing API, with the same key.
    let fal = Fal::with_bases("test-key", &base, &base);
    let q = run::quote(&p, &plan, &inputs(), &fal, &Http::new()).unwrap();
    assert_eq!(q.total(), Some(0.04));
    assert!(q.exact());

    let svcs = Services { remote: Some(Box::new(Fal::with_bases("test-key", &base, &base))), comfy: Comfy::new(&base) };
    let job = run::start(p.clone(), plan, inputs(), svcs, dir.clone(), vec![(1, 0.04)]);
    let take = drain(&job).unwrap();
    assert!(take.path(&dir).exists());
    assert_eq!(std::fs::read(take.path(&dir)).unwrap(), png_bytes());
    assert_eq!(take.tick, 1001);
    assert!(take.commercial);
    assert!(take.label.contains("kontext") && take.label.contains("make it night"), "{}", take.label);

    {
        let reqs = log.lock().unwrap();
        let submit = reqs
            .iter()
            .find(|r| r.method == "POST" && r.path == "/fal-ai/flux-pro/kontext")
            .expect("submitted to the endpoint");
        assert_eq!(submit.header("authorization"), Some("Key test-key"));
        let body = submit.json();
        assert_eq!(body["prompt"], "make it night");
        assert!(body["image_url"].as_str().unwrap().starts_with("data:image/png;base64,"));
        assert_eq!(body["output_format"], "png");
        // The status poll followed the URL fal gave, not one we built.
        assert!(reqs.iter().any(|r| r.path == "/fal-ai/flux-pro/requests/abc123/status"));
    }

    // Same pipeline, same inputs: planned as cached, runs without a single
    // request.
    let before = log.lock().unwrap().len();
    let plan = run::plan(&p, &inputs(), &dir).unwrap();
    assert!(plan.todo.is_empty(), "the edit is cached");
    let svcs = Services { remote: None, comfy: Comfy::new(&base) };
    let again = drain(&run::start(p.clone(), plan, inputs(), svcs, dir.clone(), vec![])).unwrap();
    assert_eq!(again.key, take.key);
    assert_eq!(log.lock().unwrap().len(), before, "a cache hit sends nothing");

    // Change the seed and it is a new request again.
    if let Op::Generate { seed, .. } = &mut p.steps[1].op {
        *seed += 1;
    }
    assert_eq!(run::plan(&p, &inputs(), &dir).unwrap().todo, vec![1]);
    // Move the step in the inner graph and it is not.
    if let Op::Generate { seed, .. } = &mut p.steps[1].op {
        *seed -= 1;
    }
    p.steps[1].at = [0.9, 0.9];
    assert!(run::plan(&p, &inputs(), &dir).unwrap().todo.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_validation_error_reads_as_the_field_fal_wanted() {
    let (port, _log) = serve(Arc::new(|_r: &Req, _| {
        (422, "application/json", br#"{"detail":[{"loc":["body","image_url"],"msg":"field required"}]}"#.to_vec())
    }));
    let base = format!("http://127.0.0.1:{port}");
    let dir = tmp("fal422");
    let p = edit_pipeline();
    let plan = run::plan(&p, &inputs(), &dir).unwrap();
    let svcs = Services { remote: Some(Box::new(Fal::with_bases("k", &base, &base))), comfy: Comfy::new(&base) };
    let err = drain(&run::start(p, plan, inputs(), svcs, dir.clone(), vec![])).unwrap_err();
    assert!(err.contains("HTTP 422") && err.contains("body.image_url: field required"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_failed_request_reads_as_the_reason_in_its_result() {
    // fal's failure path: the status says COMPLETED with `error` and
    // `error_type`, and the result URL answers 422 with the model's
    // `detail` — which is the part that says what to change.
    let (port, log) = serve(Arc::new(|r: &Req, port: u16| {
        let base = format!("http://127.0.0.1:{port}");
        match (r.method.as_str(), r.path.as_str()) {
            ("POST", _) => j(json!({
                "request_id": "f1",
                "status_url": format!("{base}/fal-ai/flux-pro/requests/f1/status"),
                "response_url": format!("{base}/fal-ai/flux-pro/requests/f1"),
                "cancel_url": format!("{base}/fal-ai/flux-pro/requests/f1/cancel"),
                "queue_position": 0
            })),
            ("GET", p) if p.ends_with("/status") => j(json!({
                "status": "COMPLETED", "request_id": "f1",
                "response_url": format!("{base}/fal-ai/flux-pro/requests/f1"),
                "logs": [], "error": "Unprocessable Entity", "error_type": "content_policy_violation"
            })),
            ("GET", "/fal-ai/flux-pro/requests/f1") => (422, "application/json", serde_json::to_vec(&json!({
                "detail": [{
                    "loc": ["body", "prompt"],
                    "msg": "The content could not be processed because it contained material flagged by a content checker.",
                    "type": "content_policy_violation",
                    "url": "https://docs.fal.ai/errors#content_policy_violation"
                }]
            })).unwrap()),
            _ => (404, "application/json", br#"{"detail":"Not Found"}"#.to_vec()),
        }
    }));
    let base = format!("http://127.0.0.1:{port}");
    let dir = tmp("falfail");
    let p = edit_pipeline();
    let plan = run::plan(&p, &inputs(), &dir).unwrap();
    let svcs = Services { remote: Some(Box::new(Fal::with_bases("k", &base, &base))), comfy: Comfy::new(&base) };
    let err = drain(&run::start(p, plan, inputs(), svcs, dir.clone(), vec![])).unwrap_err();
    assert!(err.contains("HTTP 422") && err.contains("body.prompt: The content could not be processed"), "{err}");
    // A failure lands nothing on disk to be served as a cached result.
    assert!(!log.lock().unwrap().iter().any(|r| r.path.starts_with("/cdn/")));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_failed_request_with_no_result_body_still_names_the_error_type() {
    let (port, _log) = serve(Arc::new(|r: &Req, port: u16| {
        let base = format!("http://127.0.0.1:{port}");
        match (r.method.as_str(), r.path.as_str()) {
            ("POST", _) => j(json!({
                "request_id": "f2", "status_url": format!("{base}/s"),
                "response_url": format!("{base}/r"), "cancel_url": format!("{base}/c")
            })),
            ("GET", "/s") => j(json!({
                "status": "COMPLETED", "request_id": "f2", "response_url": format!("{base}/r"),
                "error": "Request timed out", "error_type": "request_timeout"
            })),
            // An unreadable result: the status is all there is to go on.
            _ => (504, "text/plain", b"".to_vec()),
        }
    }));
    let base = format!("http://127.0.0.1:{port}");
    let dir = tmp("falfail2");
    let p = Pipeline::default();
    let plan = run::plan(&p, &inputs(), &dir).unwrap();
    let svcs = Services { remote: Some(Box::new(Fal::with_bases("k", &base, &base))), comfy: Comfy::new(&base) };
    let err = drain(&run::start(p, plan, inputs(), svcs, dir.clone(), vec![])).unwrap_err();
    assert!(err.contains("Request timed out (request_timeout)"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_pricing_error_reads_as_fal_s_message() {
    // The platform API's error shape is not the queue's.
    let (port, _log) = serve(Arc::new(|_r: &Req, _| {
        (401, "application/json", serde_json::to_vec(&json!({
            "error": {"type": "authorization_error", "message": "Invalid API key"}
        })).unwrap())
    }));
    let base = format!("http://127.0.0.1:{port}");
    let dir = tmp("pricing401");
    let p = edit_pipeline();
    let plan = run::plan(&p, &inputs(), &dir).unwrap();
    let fal = Fal::with_bases("bad", &base, &base);
    let err = run::quote(&p, &plan, &inputs(), &fal, &Http::new()).unwrap_err().to_string();
    assert!(err.contains("HTTP 401: Invalid API key"), "{err}");
}

#[test]
fn a_cancel_reaches_the_provider() {
    // Never completes.
    let (port, log) = serve(Arc::new(|r: &Req, port: u16| {
        let base = format!("http://127.0.0.1:{port}");
        match r.method.as_str() {
            "POST" => j(json!({
                "request_id": "q", "status_url": format!("{base}/s"),
                "response_url": format!("{base}/r"), "cancel_url": format!("{base}/c")
            })),
            "PUT" => (202, "application/json", br#"{"status":"CANCELLATION_REQUESTED"}"#.to_vec()),
            _ => accepted(json!({"status": "IN_QUEUE", "request_id": "q",
                "response_url": format!("{base}/r"), "queue_position": 9})),
        }
    }));
    let base = format!("http://127.0.0.1:{port}");
    let dir = tmp("cancel");
    let p = Pipeline::default();
    let plan = run::plan(&p, &inputs(), &dir).unwrap();
    let svcs = Services { remote: Some(Box::new(Fal::with_bases("k", &base, &base))), comfy: Comfy::new(&base) };
    let job = run::start(p, plan, inputs(), svcs, dir.clone(), vec![]);
    std::thread::sleep(Duration::from_millis(600));
    job.cancel();
    let err = drain(&job).unwrap_err();
    assert!(err.contains("cancelled"), "{err}");
    std::thread::sleep(Duration::from_millis(100));
    assert!(log.lock().unwrap().iter().any(|r| r.method == "PUT" && r.path == "/c"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_relay_is_capped_at_the_approved_price_and_reports_the_charge() {
    let (port, log) = serve(Arc::new(|r: &Req, port: u16| {
        let base = format!("http://127.0.0.1:{port}");
        match (r.method.as_str(), r.path.as_str()) {
            ("POST", "/v1/quote") => j(json!({
                "items": [{"usd": 0.004, "exact": true, "note": "1 MP"}],
                "balance_usd": 12.5
            })),
            ("POST", "/v1/jobs") => {
                let b = r.json();
                if b["max_usd"].as_f64().unwrap_or(0.0) < 0.004 {
                    (402, "application/json", br#"{"error":"the price moved above what you approved"}"#.to_vec())
                } else {
                    (202, "application/json", serde_json::to_vec(&json!({"job": "job_1", "usd": 0.004})).unwrap())
                }
            }
            ("GET", "/v1/jobs/job_1") => j(json!({
                "status": "done",
                "charged_usd": 0.004,
                "output": {"images": [{"url": format!("{base}/cdn/x.png")}]}
            })),
            ("GET", "/cdn/x.png") => (200, "image/png", png_bytes()),
            ("GET", "/v1/me") => j(json!({"account": "tester", "balance_usd": 12.5})),
            _ => (404, "application/json", br#"{"error":"no route"}"#.to_vec()),
        }
    }));
    let base = format!("http://127.0.0.1:{port}");
    let dir = tmp("relay");
    let mut p = Pipeline::default();
    if let Op::Prompt { text } = &mut p.steps[0].op {
        *text = "a moth on a lamp".into();
    }
    let plan = run::plan(&p, &inputs(), &dir).unwrap();
    let relay = Relay::new(&base, "wob_token");
    let http = Http::new();
    assert_eq!(relay.me(&http).unwrap().balance_usd, 12.5);
    let q = run::quote(&p, &plan, &inputs(), &relay, &http).unwrap();
    assert_eq!((q.total(), q.balance_usd), (Some(0.004), Some(12.5)));

    // Approved below the price: refused, with the relay's reason, before
    // anything ran.
    let svcs = Services { remote: Some(Box::new(Relay::new(&base, "wob_token"))), comfy: Comfy::new(&base) };
    let err = drain(&run::start(p.clone(), plan.clone(), inputs(), svcs, dir.clone(), vec![(1, 0.001)]))
        .unwrap_err();
    assert!(err.contains("402") && err.contains("price moved"), "{err}");

    // Approved at the quote: runs, and the take records the real charge.
    let svcs = Services { remote: Some(Box::new(Relay::new(&base, "wob_token"))), comfy: Comfy::new(&base) };
    let take = drain(&run::start(p.clone(), plan, inputs(), svcs, dir.clone(), vec![(1, 0.004)])).unwrap();
    assert_eq!(take.usd, Some(0.004));
    let reqs = log.lock().unwrap();
    let job = reqs.iter().rev().find(|r| r.path == "/v1/jobs").unwrap();
    assert_eq!(job.header("authorization"), Some("Bearer wob_token"));
    assert_eq!(job.json()["endpoint"], catalog::get(catalog::DEFAULT_MODEL).unwrap().endpoint);
    assert_eq!(job.json()["input"]["prompt"], "a moth on a lamp");
    // The job is priced on the area the quote assumed.
    assert!(job.json()["megapixels"].as_f64().is_some_and(|mp| (mp - 1.0).abs() < 0.05));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn comfy_uploads_the_reference_fills_the_placeholders_and_fetches_the_save() {
    let (port, log) = serve(Arc::new(|r: &Req, _port: u16| match (r.method.as_str(), r.path.as_str()) {
        ("POST", "/upload/image") => j(json!({"name": "wobble_5_image.png", "subfolder": "", "type": "input"})),
        ("POST", "/prompt") => j(json!({"prompt_id": "p-1", "number": 0, "node_errors": {}})),
        ("GET", "/history/p-1") => j(json!({"p-1": {
            "status": {"status_str": "success", "completed": true},
            "outputs": {"9": {"images": [{"filename": "ComfyUI_0001.png", "subfolder": "", "type": "output"}]}}
        }})),
        ("GET", p) if p.starts_with("/view?") => (200, "image/png", png_bytes()),
        _ => (404, "application/json", b"{}".to_vec()),
    }));
    let base = format!("http://127.0.0.1:{port}");
    let dir = tmp("comfy");
    let wf_path = dir.with_extension("wf.json");
    std::fs::write(
        &wf_path,
        json!({
            "3": {"class_type": "KSampler", "inputs": {"seed": "{{seed}}"}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{prompt}}, film still"}},
            "10": {"class_type": "LoadImage", "inputs": {"image": "{{image}}"}},
            "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}}
        })
        .to_string(),
    )
    .unwrap();

    let mut p = Pipeline::default();
    if let Op::Prompt { text } = &mut p.steps[0].op {
        *text = "neon rain".into();
    }
    let c = p.push(
        Op::Comfy { workflow: wf_path.to_string_lossy().into(), seed: 5, commercial: false },
        [0.5, 0.0],
    );
    let r = p.push(Op::Reference, [0.5, -0.6]);
    assert!(p.connect(0, c, 0));
    assert!(p.connect(r, c, 1));
    let out = p.output().unwrap();
    assert!(p.connect(c, out, 0));

    let plan = run::plan(&p, &inputs(), &dir).unwrap();
    assert!(plan.billed(&p).is_empty(), "comfy is local: nothing billed");
    assert!(!plan.commercial, "an unasserted comfy workflow is tagged");
    let svcs = Services { remote: None, comfy: Comfy::new(&base) };
    let take = drain(&run::start(p, plan, inputs(), svcs, dir.clone(), vec![])).unwrap();
    assert!(!take.commercial);
    assert!(take.path(&dir).exists());

    let reqs = log.lock().unwrap();
    let upload = reqs.iter().find(|r| r.path == "/upload/image").unwrap();
    assert!(upload.header("content-type").unwrap().starts_with("multipart/form-data"));
    let prompt = reqs.iter().find(|r| r.path == "/prompt").unwrap().json();
    assert_eq!(prompt["prompt"]["3"]["inputs"]["seed"], 5);
    assert_eq!(prompt["prompt"]["6"]["inputs"]["text"], "neon rain, film still");
    assert_eq!(prompt["prompt"]["10"]["inputs"]["image"], "wobble_5_image.png");
    assert!(reqs.iter().any(|r| r.path.starts_with("/view?filename=ComfyUI_0001.png")));
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&wf_path);
}

#[test]
fn nothing_is_sent_when_the_plan_is_incomplete() {
    let mut p = edit_pipeline();
    let dir = tmp("incomplete");
    // No reference picture: refused at plan time.
    let mut ins = inputs();
    ins.reference = None;
    let err = run::plan(&p, &ins, &dir).unwrap_err();
    assert!(err.contains("nothing is wired"), "{err}");
    // An unwired prompt: named.
    p.disconnect(1, 0);
    assert!(run::plan(&p, &inputs(), &dir).unwrap_err().contains("prompt"));
    let _ = Opts::default();
}

/// A fal that makes a video: Seedance's queue, its pricing row (per
/// second), and an mp4 on its CDN.
#[test]
fn a_video_step_is_quoted_per_second_and_lands_an_mp4_take() {
    const MP4: &[u8] = b"\x00\x00\x00\x18ftypmp42not-really-a-movie";
    let (port, log) = serve(Arc::new(|r: &Req, port: u16| {
        let base = format!("http://127.0.0.1:{port}");
        let rid = format!("{base}/bytedance/seedance-2.0/requests/v1");
        match (r.method.as_str(), r.path.as_str()) {
            ("POST", "/bytedance/seedance-2.0/fast/image-to-video") => j(json!({
                "request_id": "v1", "status_url": format!("{rid}/status"),
                "response_url": rid, "cancel_url": format!("{rid}/cancel"), "queue_position": 0
            })),
            ("GET", p) if p.ends_with("/status") => j(json!({"status": "COMPLETED", "request_id": "v1",
                "response_url": rid})),
            ("GET", "/bytedance/seedance-2.0/requests/v1") => j(json!({
                "video": {"url": format!("{base}/cdn/clip.mp4"), "content_type": "video/mp4",
                          "file_name": "clip.mp4", "file_size": MP4.len()},
                "seed": 42
            })),
            ("GET", "/cdn/clip.mp4") => (200, "video/mp4", MP4.to_vec()),
            ("GET", p) if p.starts_with("/v1/models/pricing") => j(json!({
                "prices": [{"endpoint_id": "bytedance/seedance-2.0/fast/image-to-video",
                            "unit_price": 0.2419, "unit": "seconds", "currency": "USD"}],
                "next_cursor": null, "has_more": false
            })),
            _ => (404, "application/json", br#"{"detail":"Not Found"}"#.to_vec()),
        }
    }));
    let base = format!("http://127.0.0.1:{port}");
    let dir = tmp("video");
    let mut p = edit_pipeline();
    p.set_model(1, "fal/seedance-2-fast-i2v");
    if let Op::Prompt { text } = &mut p.steps[0].op {
        *text = "the camera pushes in".into();
    }
    if let Op::Generate { opts, .. } = &mut p.steps[1].op {
        opts.seconds = 6;
    }
    let plan = run::plan(&p, &inputs(), &dir).unwrap();
    assert_eq!(plan.billed(&p), vec![1]);

    let fal = Fal::with_bases("k", &base, &base);
    let q = run::quote(&p, &plan, &inputs(), &fal, &Http::new()).unwrap();
    assert!((q.total().unwrap() - 0.2419 * 6.0).abs() < 1e-9, "{q:?}");
    assert!(q.exact(), "720p is the per-second price fal lists");

    let svcs = Services { remote: Some(Box::new(Fal::with_bases("k", &base, &base))), comfy: Comfy::new(&base) };
    let take = drain(&run::start(p, plan, inputs(), svcs, dir.clone(), vec![(1, 1.46)])).unwrap();
    assert_eq!(take.file, "result.mp4");
    assert!(take.is_video());
    assert_eq!(take.tick, 1001, "a clip plays from the tick it was made at");
    assert_eq!(std::fs::read(take.path(&dir)).unwrap(), MP4);

    let reqs = log.lock().unwrap();
    let body = reqs.iter().find(|r| r.method == "POST").unwrap().json();
    assert_eq!(body["duration"], "6");
    assert_eq!(body["resolution"], "720p");
    assert_eq!(body["generate_audio"], false);
    assert_eq!(body["prompt"], "the camera pushes in");
    assert!(body["image_url"].as_str().unwrap().starts_with("data:image/png;base64,"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_wired_prompt_reaches_the_request() {
    let (port, log) = fake_fal();
    let base = format!("http://127.0.0.1:{port}");
    let dir = tmp("promptin");
    let mut p = edit_pipeline();
    p.steps[0].op = Op::PromptIn;
    let ins = Inputs { prompt: Some("make it dusk".into()), ..inputs() };
    let plan = run::plan(&p, &ins, &dir).unwrap();
    let svcs = Services { remote: Some(Box::new(Fal::with_bases("k", &base, &base))), comfy: Comfy::new(&base) };
    let take = drain(&run::start(p, plan, ins, svcs, dir.clone(), vec![])).unwrap();
    assert!(take.label.contains("make it dusk"), "{}", take.label);
    let reqs = log.lock().unwrap();
    let body = reqs.iter().find(|r| r.method == "POST" && r.path == "/fal-ai/flux-pro/kontext").unwrap().json();
    assert_eq!(body["prompt"], "make it dusk");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Fan-out: one prompt and one reference through three edit models. Three
/// requests to three endpoints, three takes with their own keys, labels
/// and recipes, one summed quote, and a second run that sends nothing.
#[test]
fn a_fan_out_runs_each_model_once_and_lands_a_take_for_each() {
    let (port, log) = fake_fal();
    let base = format!("http://127.0.0.1:{port}");
    let dir = tmp("fanout");
    let mut p = edit_pipeline();
    if let Op::Prompt { text } = &mut p.steps[0].op {
        *text = "make it night".into();
    }
    let vary = genai::Vary {
        models: vec!["fal/seedream-5-pro-edit".into(), "fal/nano-banana-2-edit".into()],
        ..Default::default()
    };
    let variants = vary.expand(&p);
    assert_eq!(variants.len(), 3);
    let planned: Vec<(Pipeline, run::Plan)> =
        variants.iter().map(|v| (v.clone(), run::plan(v, &inputs(), &dir).unwrap())).collect();

    // One quote for the lot: the fake prices Kontext only, so two of the
    // three calls come back unpriced rather than dropped.
    let fal = Fal::with_bases("test-key", &base, &base);
    let q = run::quote_many(&planned, &inputs(), &fal, &Http::new()).unwrap();
    assert_eq!(q.items.len(), 3);
    assert_eq!(q.items.iter().filter(|i| i.is_some()).count(), 1);

    let start = |planned: &[(Pipeline, run::Plan)], remote: bool| {
        let svcs = Services {
            remote: remote.then(|| Box::new(Fal::with_bases("test-key", &base, &base)) as Box<dyn genai::service::Service>),
            comfy: Comfy::new(&base),
        };
        let vs = planned
            .iter()
            .map(|(p, plan)| run::Variant { pipeline: p.clone(), plan: plan.clone(), approved: vec![], inputs: None })
            .collect();
        run::start_many(vs, inputs(), svcs, dir.clone())
    };
    let takes = drain_all(&start(&planned, true)).unwrap();
    assert_eq!(takes.len(), 3);
    let mut keys: Vec<_> = takes.iter().map(|t| t.key.clone()).collect();
    keys.sort();
    keys.dedup();
    assert_eq!(keys.len(), 3, "each model's result is its own");
    for (t, short) in takes.iter().zip(["kontext", "seedream edit", "banana 2 edit"]) {
        assert!(t.label.contains(short), "{} / {short}", t.label);
        assert!(t.path(&dir).exists());
        // The recipe beside each take is the variant that made it.
        let recipe = genai::history::load_recipe(&dir, &t.key).expect("a recipe per take");
        let model = recipe.iter().find_map(|s| match &s.op {
            Op::Generate { model, .. } => genai::catalog::get(model).map(|m| m.short),
            _ => None,
        });
        assert_eq!(model, Some(short));
    }
    {
        let reqs = log.lock().unwrap();
        let posts: Vec<_> = reqs.iter().filter(|r| r.method == "POST").map(|r| r.path.clone()).collect();
        assert_eq!(
            posts,
            ["/fal-ai/flux-pro/kontext", "/bytedance/seedream/v5/pro/edit", "/fal-ai/nano-banana-2/edit"]
        );
        // The same prompt reached all three.
        assert!(reqs.iter().filter(|r| r.method == "POST").all(|r| r.json()["prompt"] == "make it night"));
    }

    // Again, unchanged: three cache hits, no service needed, nothing sent.
    let before = log.lock().unwrap().len();
    let planned: Vec<(Pipeline, run::Plan)> =
        variants.iter().map(|v| (v.clone(), run::plan(v, &inputs(), &dir).unwrap())).collect();
    assert!(planned.iter().all(|(_, plan)| plan.todo.is_empty()));
    let again = drain_all(&start(&planned, false)).unwrap();
    assert_eq!(again.iter().map(|t| &t.key).collect::<Vec<_>>(), takes.iter().map(|t| &t.key).collect::<Vec<_>>());
    assert_eq!(log.lock().unwrap().len(), before);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A prompt through the enhancer and on into a model: the model is sent
/// the ENHANCED words, the panel is told what they were, and the text step
/// caches like any other.
#[test]
fn an_enhanced_prompt_is_what_the_next_step_is_sent() {
    let (port, log) = fake_fal();
    let base = format!("http://127.0.0.1:{port}");
    let dir = tmp("enhance");
    let mut p = Pipeline { steps: vec![], takes: vec![], take: 0 };
    let t = p.push(Op::Prompt { text: "a moth".into() }, [0.0, 0.0]);
    let e = p.push(Op::Generate { model: "fal/prompt-enhance".into(), seed: 1, opts: Default::default() }, [0.0, 0.0]);
    let g = p.push(Op::Generate { model: "fal/flux-schnell".into(), seed: 1, opts: Default::default() }, [0.0, 0.0]);
    let o = p.push(Op::Output, [0.0, 0.0]);
    assert!(p.connect(t, e, 0) && p.connect(e, g, 0) && p.connect(g, o, 0));
    let plan = run::plan(&p, &inputs(), &dir).unwrap();
    assert_eq!(plan.billed(&p), vec![e, g]);

    let svcs = Services { remote: Some(Box::new(Fal::with_bases("test-key", &base, &base))), comfy: Comfy::new(&base) };
    let job = run::start(p.clone(), plan, inputs(), svcs, dir.clone(), vec![]);
    let mut words = None;
    let take = loop {
        match job.rx.recv_timeout(Duration::from_secs(20)).expect("run stalled") {
            Event::Text(i, w) => {
                assert_eq!(i, e);
                words = Some(w);
            }
            Event::Done(t) => break t,
            Event::Failed(why) => panic!("{why}"),
            _ => {}
        }
    };
    let enhanced = "a moth on a lamp, night, rain on the glass, 85mm";
    assert_eq!(words.as_deref(), Some(enhanced));
    assert!(take.path(&dir).exists());
    {
        let reqs = log.lock().unwrap();
        let llm = reqs.iter().find(|r| r.method == "POST" && r.path == "/fal-ai/any-llm").expect("asked the llm");
        assert_eq!(llm.json()["prompt"], "a moth");
        assert!(llm.json()["system_prompt"].as_str().unwrap().contains("rewrite prompts"));
        let gen = reqs.iter().find(|r| r.method == "POST" && r.path == "/fal-ai/flux/schnell").expect("generated");
        assert_eq!(gen.json()["prompt"], enhanced);
    }
    // Unchanged: both steps cached, nothing sent, and the words are told again.
    let before = log.lock().unwrap().len();
    let plan = run::plan(&p, &inputs(), &dir).unwrap();
    assert!(plan.todo.is_empty());
    let job = run::start(p, plan, inputs(), Services { remote: None, comfy: Comfy::new(&base) }, dir.clone(), vec![]);
    let mut again = None;
    loop {
        match job.rx.recv_timeout(Duration::from_secs(20)).expect("run stalled") {
            Event::Text(_, w) => again = Some(w),
            Event::Done(_) => break,
            Event::Failed(why) => panic!("{why}"),
            _ => {}
        }
    }
    assert_eq!(again.as_deref(), Some(enhanced));
    assert_eq!(log.lock().unwrap().len(), before);
    let _ = std::fs::remove_dir_all(&dir);
}
