//! `--test-write`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    pub(crate) fn drive_test_write(&mut self) {
        let Some((stage, mut frames, wid)) = self.test.write else { return };
        frames += 1;
        if frames > 2500 {
            println!("WRITE TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let mut next = stage;
        let dir = std::env::temp_dir().join("wobble-write-test");
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                let _ = std::fs::remove_dir_all(&dir);
                // A Write over the default comp's grade, with the grade
                // pushed somewhere unmistakable so the files can be told
                // apart from the source plate.
                if let Some(editor::ENode { kind: NodeKind::Grade(g), .. }) =
                    self.egraph.node_mut(3)
                {
                    g.gain = [4.0, 0.25, 0.25];
                }
                let w = self.egraph.add(
                    // The new form: a location and a name, enclosed in a
                    // subfolder of that name (the default).
                    NodeKind::Write(engine::graph::WriteParams {
                        location: dir.to_string_lossy().into_owned(),
                        name: "out".into(),
                        start: 1001,
                        range: Some((0, 2)),
                        range_mode: Some(engine::graph::WriteRange::Range),
                        output: None,
                        output_default: false,
                        ..Default::default()
                    }),
                    egui::pos2(300.0, 0.0),
                );
                // Over the grade (node 3 in the default graph), and viewed,
                // so the passthrough is exercised too.
                self.egraph.connect(w, 0, 3);
                self.egraph.view = Some(w);
                self.test.write = Some((1, 0, w));
                return;
            }
            1 if frames > 30 => {
                let Some(NodeKind::Write(wp)) =
                    self.egraph.node(wid).map(|n| n.kind.clone())
                else {
                    return;
                };
                self.start_write(wid, &wp);
                if self.graph_bake.is_none() {
                    println!(
                        "WRITE TEST FAIL: render did not start ({})",
                        self.load_errors.get(&wid).map(|s| s.as_str()).unwrap_or("no reason")
                    );
                    std::process::exit(1);
                }
                next = 2;
            }
            // Wait on the render itself, not on a frame count.
            2 if self.graph_bake.is_some() => {
                frames = 0;
            }
            2 => {
                if let Some(e) = self.load_errors.get(&wid) {
                    println!("WRITE TEST FAIL: {e}");
                    std::process::exit(1);
                }
                let files: Vec<std::path::PathBuf> = (1001..=1003)
                    .map(|n| dir.join("out").join(format!("out.{n}.exr")))
                    .collect();
                if let Some(missing) = files.iter().find(|f| !f.exists()) {
                    println!("WRITE TEST FAIL: {} was not written", missing.display());
                    std::process::exit(1);
                }
                // Read them back through the ordinary decoder.
                let (mut means, mut sizes, mut probes) =
                    (Vec::new(), Vec::new(), Vec::<f32>::new());
                for f in &files {
                    let Ok(info) = self.decoder.read_header(f) else {
                        println!("WRITE TEST FAIL: {} has no readable header", f.display());
                        std::process::exit(1);
                    };
                    let stride = info.row_stride_f16();
                    let mut buf = vec![half::f16::ZERO; stride * info.height as usize];
                    if let Err(e) = self.decoder.decode_into(f, info, stride, &mut buf) {
                        println!("WRITE TEST FAIL: decoding {}: {e:#}", f.display());
                        std::process::exit(1);
                    }
                    sizes.push((info.width, info.height));
                    // Mean red and green over the frame: the grade pushed red
                    // up and green down, so the ratio is the fingerprint.
                    let (mut r, mut g, mut n) = (0.0f64, 0.0f64, 0u64);
                    for y in 0..info.height as usize {
                        for x in 0..info.width as usize {
                            let i = y * stride + x * 4;
                            r += buf[i].to_f32() as f64;
                            g += buf[i + 1].to_f32() as f64;
                            n += 1;
                        }
                    }
                    // Frame-to-frame motion needs a POINT, not an average:
                    // this plate scrolls a sine across x, whose whole-frame
                    // mean is the same on every frame.
                    let probe = {
                        let (x, y) = (info.width as usize / 4, info.height as usize / 2);
                        buf[y * stride + x * 4].to_f32()
                    };
                    probes.push(probe);
                    means.push((r / n as f64, g / n as f64));
                }
                let want = self.comp.output_size();
                if sizes.iter().any(|s| Some(*s) != want) {
                    println!("WRITE TEST FAIL: sizes {sizes:?}, comp is {want:?}");
                    std::process::exit(1);
                }
                // The grade is IN the files (red well above green)...
                let graded = means.iter().all(|(r, g)| *r > *g * 2.0);
                // ...and the frames are not all the same picture.
                let moved = probes.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-3);
                // "make Load": a Load beside the Render, reading the
                // sequence it wrote over the render's frames.
                let made = self
                    .egraph
                    .node(wid)
                    .and_then(|n| match &n.kind {
                        NodeKind::Write(wp) => Some(wp.clone()),
                        _ => None,
                    })
                    .and_then(|wp| self.load_from_write(wid, &wp));
                let load_ok = made.is_some_and(|id| {
                    let range_ok = matches!(
                        self.egraph.node(id).map(|n| &n.kind),
                        Some(NodeKind::Read { path, first: Some(0), last: Some(2), start: Some(0), .. })
                            if *path == dir.join("out").join("out.####.exr").to_string_lossy()
                    );
                    let loaded = self.loaded.get(&id).map(|l| l.seq.frames.len());
                    if !range_ok || loaded != Some(3) {
                        println!(
                            "WRITE TEST FAIL: make Load gave {:?}, {loaded:?} frames loaded ({})",
                            self.egraph.node(id).map(|n| &n.kind),
                            self.load_errors.get(&id).map(|s| s.as_str()).unwrap_or("no error")
                        );
                        std::process::exit(1);
                    }
                    true
                });
                if !load_ok {
                    println!("WRITE TEST FAIL: make Load made no node");
                    std::process::exit(1);
                }
                if graded && moved {
                    eprintln!(
                        "write test: wrote and read back 3 EXRs at {:?}, graded \
                         (mean r/g {:.3}/{:.3}) and moving frame to frame \
                         (probe {:.4} -> {:.4})",
                        sizes[0], means[0].0, means[0].1, probes[0], probes[2]
                    );
                    let _ = std::fs::remove_dir_all(&dir);
                    // Then a comp with no footage at all: a Noise into a
                    // Write. There is no playback rig to drive the frames,
                    // and the render must not need one.
                    let ids: Vec<u64> = self.egraph.nodes.keys().copied().collect();
                    ids.into_iter().for_each(|id| self.egraph.remove(id));
                    let noise = self.egraph.add(NodeKind::Noise(Default::default()), egui::pos2(0.0, 0.0));
                    let w = self.egraph.add(
                        // An old single-path Render still writes where it says.
                        NodeKind::Write(engine::graph::WriteParams {
                            path: dir.join("noise.####.exr").to_string_lossy().into_owned(),
                            start: 1001,
                            range: Some((0, 2)),
                            range_mode: Some(engine::graph::WriteRange::Range),
                            output: None,
                            output_default: false,
                            ..Default::default()
                        }),
                        egui::pos2(0.0, 120.0),
                    );
                    self.egraph.connect(w, 0, noise);
                    self.egraph.view = Some(w);
                    self.playback_dirty = true;
                    self.test.write = Some((3, 0, w));
                    return;
                }
                println!(
                    "WRITE TEST FAIL: graded={graded} moved={moved} means={means:?} \
                     probes={probes:?}"
                );
                std::process::exit(1);
            }
            3 if self.playback.is_none() && frames > 30 => {
                let Some(NodeKind::Write(wp)) = self.egraph.node(wid).map(|n| n.kind.clone()) else {
                    return;
                };
                self.start_write(wid, &wp);
                if self.graph_bake.is_none() {
                    println!(
                        "WRITE TEST FAIL: a render with no footage did not start ({})",
                        self.load_errors.get(&wid).map(|s| s.as_str()).unwrap_or("no reason")
                    );
                    std::process::exit(1);
                }
                next = 4;
            }
            4 if self.graph_bake.is_some() => {
                frames = 0;
            }
            4 => {
                if let Some(e) = self.load_errors.get(&wid) {
                    println!("WRITE TEST FAIL: no-footage render: {e}");
                    std::process::exit(1);
                }
                let files: Vec<std::path::PathBuf> =
                    (1001..=1003).map(|n| dir.join(format!("noise.{n}.exr"))).collect();
                if let Some(missing) = files.iter().find(|f| !f.exists()) {
                    println!("WRITE TEST FAIL: no-footage render did not write {}", missing.display());
                    std::process::exit(1);
                }
                // Then the same Render as a 16-bit layered PSD sequence.
                if let Some(NodeKind::Write(wp)) = self.egraph.node_mut(wid).map(|n| &mut n.kind) {
                    wp.path.clear();
                    wp.location = dir.to_string_lossy().into_owned();
                    wp.name = "noise".into();
                    wp.subfolder = false;
                    wp.output_default = true;
                    wp.format = engine::graph::WriteFormat::Psd16;
                }
                next = 7;
            }
            7 if frames > 10 => {
                let Some(NodeKind::Write(wp)) = self.egraph.node(wid).map(|n| n.kind.clone()) else {
                    return;
                };
                self.start_write(wid, &wp);
                if self.graph_bake.is_none() {
                    println!(
                        "WRITE TEST FAIL: a PSD render did not start ({})",
                        self.load_errors.get(&wid).map(|s| s.as_str()).unwrap_or("no reason")
                    );
                    std::process::exit(1);
                }
                next = 8;
            }
            8 if self.graph_bake.is_some() => {
                frames = 0;
            }
            8 => {
                if let Some(e) = self.load_errors.get(&wid) {
                    println!("WRITE TEST FAIL: PSD render: {e}");
                    std::process::exit(1);
                }
                let size = self.comp.output_size().unwrap_or((0, 0));
                for n in 1001..=1003 {
                    let f = dir.join(format!("noise.{n}.psd"));
                    let head = std::fs::read(&f).unwrap_or_default();
                    let be = |at: usize| head.get(at..at + 4).map(|b| u32::from_be_bytes(b.try_into().unwrap()));
                    let ok = head.starts_with(b"8BPS")
                        && be(14) == Some(size.1)
                        && be(18) == Some(size.0)
                        && head.get(22..24) == Some(&[0, 16]);
                    if !ok {
                        println!("WRITE TEST FAIL: {} is not a {}x{} 16-bit PSD", f.display(), size.0, size.1);
                        std::process::exit(1);
                    }
                }
                // Then the same Render as a movie under proxy (a movie
                // build with a ProRes encoder; a stock build stops here).
                let fmt = engine::graph::WriteFormat::ProRes422Hq;
                if movieout::MovieOut::can_encode(fmt).is_ok() {
                    if let Some(NodeKind::Write(wp)) = self.egraph.node_mut(wid).map(|n| &mut n.kind) {
                        wp.path.clear();
                        wp.location = dir.to_string_lossy().into_owned();
                        wp.name = "noise".into();
                        wp.subfolder = false;
                        wp.format = fmt;
                    }
                    self.proxy = 0.5;
                    self.playback_dirty = true;
                    self.last_engine_graph = None;
                    next = 5;
                } else {
                    println!(
                        "WRITE TEST PASS: wrote and read back 3 graded, moving EXRs under \
                         location/name/, made a Load of them over the render's frames, then \
                         rendered a Noise with no footage in the comp through an old-style path, \
                         and as 16-bit PSDs (no movie encoder in this build: movie stage skipped)"
                    );
                    let _ = std::fs::remove_dir_all(&dir);
                    std::process::exit(0);
                }
            }
            5 if frames > 30 => {
                let Some(NodeKind::Write(wp)) = self.egraph.node(wid).map(|n| n.kind.clone()) else {
                    return;
                };
                self.start_write(wid, &wp);
                if self.graph_bake.is_none() {
                    println!(
                        "WRITE TEST FAIL: a proxy movie render did not start ({})",
                        self.load_errors.get(&wid).map(|s| s.as_str()).unwrap_or("no reason")
                    );
                    std::process::exit(1);
                }
                next = 6;
            }
            6 if self.graph_bake.is_some() => {
                frames = 0;
            }
            6 => {
                if let Some(e) = self.load_errors.get(&wid) {
                    println!("WRITE TEST FAIL: proxy movie render: {e}");
                    std::process::exit(1);
                }
                #[cfg_attr(not(feature = "movie"), allow(unused_variables))]
                let mov = dir.join("noise_prx.mov");
                #[cfg(feature = "movie")]
                let wp_size = |w: u32, h: u32| match self.egraph.node(wid).map(|n| &n.kind) {
                    Some(NodeKind::Write(wp)) => wp.movie_size(w, h),
                    _ => (w, h),
                };
                #[cfg(feature = "movie")]
                {
                    let dec = engine::movie::MovieDecoder::default();
                    let n = dec.frame_count(&mov).ok();
                    let size = dec.read_header(&mov).ok().map(|i| (i.width, i.height));
                    let want = self.comp.output_size().map(|(w, h)| wp_size(w, h));
                    if n != Some(3) || size.is_none() || size != want {
                        println!(
                            "WRITE TEST FAIL: {} has {n:?} frames at {size:?} (proxy canvas {want:?})",
                            mov.display()
                        );
                        std::process::exit(1);
                    }
                }
                println!(
                    "WRITE TEST PASS: wrote and read back 3 graded, moving EXRs under \
                     location/name/, made a Load of them over the render's frames, \
                     rendered a Noise with no footage through an old-style path and as \
                     16-bit PSDs, then the same as a 3-frame ProRes .mov under proxy, named _prx"
                );
                let _ = std::fs::remove_dir_all(&dir);
                std::process::exit(0);
            }
            _ => {}
        }
        let frames = if next == stage { frames } else { 0 };
        self.test.write = Some((next, frames, wid));
    }
}
