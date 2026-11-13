//! A 2dini animation as a static web page: every frame's geometry packed
//! into one self-contained HTML file with a tiny Canvas2D player. No
//! WebGPU, no wasm, no other files — open it, or drop it on any host.
//!
//! The page draws what the layer draws: closed shapes filled under their
//! own fill rule, strokes centred on the outline over the fill, bare
//! points as discs of their `pscale`. Colours are converted from the
//! comp's linear values to sRGB (no view transform: a drawing's colours
//! are flat picks, and a filmic curve would shift them). Frames that are
//! exactly alike share their data, so a still drawing costs one frame.
//!
//! Data: one little-endian Int16 stream, base64 in the page. Coordinates
//! are output pixels × [`QUANT`]. Per frame: prim count, then per prim
//! `style, kind` and either `x, y, r` (a point) or `rings, len…, x, y…`.

use crate::geo::{Geometry, PrimKind, Style};

/// Sub-pixel steps per output pixel: ±8191 px of reach in an i16.
pub const QUANT: f32 = 4.0;

fn srgb(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let v = if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
    (v * 255.0).round() as u8
}

fn css(c: [f32; 4]) -> Option<String> {
    (c[3] > 0.0).then(|| {
        format!("rgba({},{},{},{:.3})", srgb(c[0]), srgb(c[1]), srgb(c[2]), c[3].clamp(0.0, 1.0))
    })
}

/// The palette key: styles compared bit for bit.
fn style_key(s: &Style) -> [u32; 10] {
    let b = |v: f32| v.to_bits();
    [
        b(s.fill[0]),
        b(s.fill[1]),
        b(s.fill[2]),
        b(s.fill[3]),
        b(s.stroke[0]),
        b(s.stroke[1]),
        b(s.stroke[2]),
        b(s.stroke[3]),
        b(s.width),
        s.rule as u32,
    ]
}

/// The styles a page draws with, deduplicated bit for bit; a prim names
/// its style by index.
#[derive(Default)]
pub struct Palette {
    styles: Vec<Style>,
    keys: Vec<[u32; 10]>,
}

impl Palette {
    fn index(&mut self, s: &Style) -> usize {
        let k = style_key(s);
        self.keys.iter().position(|x| *x == k).unwrap_or_else(|| {
            self.keys.push(k);
            self.styles.push(*s);
            self.keys.len() - 1
        })
    }

    /// `[fill css|null, stroke css|null, width, rule]` per style.
    pub fn json(&self) -> String {
        let all: Vec<String> = self
            .styles
            .iter()
            .map(|s| {
                let f = css(s.fill).map_or("null".into(), |c| format!("\"{c}\""));
                let k = if s.width > 0.0 { css(s.stroke) } else { None }
                    .map_or("null".into(), |c| format!("\"{c}\""));
                let rule = if s.rule == crate::geo::FillRule::EvenOdd { "evenodd" } else { "nonzero" };
                format!("[{f},{k},{},\"{rule}\"]", s.width.max(0.0))
            })
            .collect();
        format!("[{}]", all.join(","))
    }

    pub fn clear(&mut self) {
        self.styles.clear();
        self.keys.clear();
    }
}

/// Append one frame of `g` to `out` (format in the module docs), its
/// styles into `pal`. `ppu` = output pixels per geo unit.
pub fn pack_frame(g: &Geometry, ppu: f32, pal: &mut Palette, out: &mut Vec<i16>) {
    let q = |v: f32| (v * ppu * QUANT).round().clamp(-32767.0, 32767.0) as i16;
    let count_at = out.len();
    out.push(0);
    let mut n = 0i16;
    for (pi, pr) in g.prims.iter().enumerate() {
        if n == i16::MAX {
            break;
        }
        match pr.kind {
            PrimKind::Point => {
                let Some(p) = pr.verts.first().and_then(|&v| g.points.get(v as usize)) else {
                    continue;
                };
                let si = pal.index(&pr.style);
                out.extend([si as i16, 2, q(p.p[0]), q(p.p[1])]);
                out.push((p.pscale.max(0.0) * QUANT).round().min(32767.0) as i16);
            }
            PrimKind::Poly { closed } => {
                let rings: Vec<Vec<[f32; 2]>> =
                    g.rings(pi).into_iter().filter(|r| r.len() >= 2).collect();
                if rings.is_empty() || rings.len() > 32767 {
                    continue;
                }
                let si = pal.index(&pr.style);
                out.extend([si as i16, closed as i16, rings.len() as i16]);
                for r in &rings {
                    out.push(r.len().min(32767) as i16);
                }
                for r in &rings {
                    for p in r.iter().take(32767) {
                        out.extend([q(p[0]), q(p[1])]);
                    }
                }
            }
        }
        n += 1;
    }
    out[count_at] = n;
}

fn b64_i16(data: &[i16]) -> String {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    crate::geophys::b64_encode(&bytes)
}

fn page(title: &str, meta: &str, driver: &str) -> String {
    let title = title.replace(['<', '>', '&'], "");
    PAGE.replace("{{TITLE}}", &title).replace("{{META}}", meta).replace("{{DRIVER}}", driver)
}

/// Pack `frames` (geometry per tick, in play order) into a page that loops
/// them. `w × h` is the output format: geo x 0..1 spans `w` pixels.
pub fn export_html(title: &str, frames: &[Geometry], w: u32, h: u32, fps: f32) -> String {
    let mut pal = Palette::default();
    let mut data: Vec<i16> = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    for (fi, g) in frames.iter().enumerate() {
        // A frame just like an earlier one points at that one's data.
        if let Some(j) = frames[..fi].iter().rposition(|p| p == g) {
            offsets.push(offsets[j]);
            continue;
        }
        offsets.push(data.len());
        pack_frame(g, w as f32, &mut pal, &mut data);
    }
    let offs: Vec<String> = offsets.iter().map(|o| o.to_string()).collect();
    let meta = format!(
        "{{\"w\":{w},\"h\":{h},\"fps\":{fps},\"q\":{QUANT},\"styles\":{},\"frames\":[{}]}}",
        pal.json(),
        offs.join(",")
    );
    let driver = BAKED.replace("{{DATA}}", &b64_i16(&data));
    page(title, &meta, &driver)
}

/// A page that EVALUATES the drawing as it plays, forever: `wasm` is the
/// geoplayer module (see `geoplayer/`), `setup` the JSON it loads (the
/// 2dini's params, the Noise on its field port, the format). Time runs
/// from tick `start` at `fps`, continuously — moving noise never repeats;
/// a baked simulation plays once and holds its last frame.
pub fn export_live_html(title: &str, setup: &str, wasm: &[u8], w: u32, h: u32, fps: f32, start: f32) -> String {
    let meta = format!("{{\"w\":{w},\"h\":{h},\"fps\":{fps},\"q\":{QUANT},\"start\":{start}}}");
    let driver = LIVE
        .replace("{{WASM}}", &crate::geophys::b64_encode(wasm))
        .replace("{{SETUP}}", &crate::geophys::b64_encode(setup.as_bytes()));
    page(title, &meta, &driver)
}

const PAGE: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<style>
  :root { --bg: #111; }
  html, body { margin: 0; height: 100%; background: var(--bg); overflow: hidden; }
  canvas { display: block; width: 100vw; height: 100vh; }
</style>
</head>
<body>
<canvas id="c"></canvas>
<script>
// Made in wobble (2dini), drawn with Canvas2D.
const M = {{META}};
const cv = document.getElementById("c"), cx = cv.getContext("2d");
let dirty = true;
function size() {
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(innerWidth * dpr); cv.height = Math.round(innerHeight * dpr);
  dirty = true;
}
addEventListener("resize", size); size();
const unb64 = s => Uint8Array.from(atob(s), ch => ch.charCodeAt(0));
// One frame: D an Int16Array, i where the frame starts, S its styles.
function draw(D, i, S) {
  const s = Math.min(cv.width / M.w, cv.height / M.h);
  const ox = (cv.width - M.w * s) / 2, oy = (cv.height - M.h * s) / 2;
  cx.setTransform(1, 0, 0, 1, 0, 0);
  cx.clearRect(0, 0, cv.width, cv.height);
  cx.setTransform(s / M.q, 0, 0, s / M.q, ox, oy);
  cx.lineJoin = "round"; cx.lineCap = "round";
  const n = D[i++];
  for (let k = 0; k < n; k++) {
    const st = S[D[i++]], kind = D[i++];
    const p = new Path2D();
    if (kind === 2) {
      const x = D[i++], y = D[i++], r = D[i++];
      p.arc(x, y, r, 0, Math.PI * 2);
    } else {
      const nr = D[i++], lens = [];
      for (let r = 0; r < nr; r++) lens.push(D[i++]);
      for (const len of lens) {
        for (let v = 0; v < len; v++) {
          const x = D[i++], y = D[i++];
          v ? p.lineTo(x, y) : p.moveTo(x, y);
        }
        if (kind === 1) p.closePath();
      }
    }
    if (st[0] && kind !== 0) { cx.fillStyle = st[0]; cx.fill(p, st[3]); }
    if (st[1]) { cx.strokeStyle = st[1]; cx.lineWidth = st[2] * M.q; cx.stroke(p); }
  }
}
{{DRIVER}}
</script>
</body>
</html>
"#;

/// Loops the packed frames.
const BAKED: &str = r#"const D = new Int16Array(unb64("{{DATA}}").buffer);
const t0 = performance.now();
let last = -1;
function tick(now) {
  const f = Math.floor((now - t0) / 1000 * M.fps) % M.frames.length;
  if (f !== last || dirty) { draw(D, M.frames[f], M.styles); last = f; dirty = false; }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);"#;

/// Evaluates every displayed frame in the embedded geoplayer module.
const LIVE: &str = r#"(async () => {
  // The module's few imports are web-binding leftovers the evaluator never
  // calls: stub each one.
  const mod = await WebAssembly.compile(unb64("{{WASM}}"));
  const imports = {};
  for (const im of WebAssembly.Module.imports(mod)) {
    (imports[im.module] ||= {})[im.name] = () => { throw new Error("wobble: " + im.name); };
  }
  const instance = await WebAssembly.instantiate(mod, imports);
  const X = instance.exports;
  const setup = unb64("{{SETUP}}");
  const at = X.alloc(setup.length);
  new Uint8Array(X.memory.buffer, at, setup.length).set(setup);
  if (!X.load(at, setup.length)) { console.error("wobble: setup did not load"); return; }
  const dec = new TextDecoder();
  const t0 = performance.now();
  function tick(now) {
    const n = X.frame(M.start + (now - t0) / 1000 * M.fps);
    const D = new Int16Array(X.memory.buffer, X.data_ptr(), n);
    const S = JSON.parse(dec.decode(new Uint8Array(X.memory.buffer, X.styles_ptr(), X.styles_len())));
    draw(D, 0, S);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
})();"#;


#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::{Point, Prim};

    fn square(fill: [f32; 4]) -> Geometry {
        let mut g = Geometry::default();
        for p in [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]] {
            g.points.push(Point { p, ..Default::default() });
        }
        g.prims.push(Prim {
            layer: Default::default(),
            kind: PrimKind::Poly { closed: true },
            verts: vec![0, 1, 2, 3],
            style: Style { fill, ..Default::default() },
            phys: None,
            src: 0,
            rings: Vec::new(),
        });
        g
    }

    fn decode(page: &str) -> Vec<i16> {
        let b = page.split("unb64(\"").nth(1).unwrap().split('"').next().unwrap();
        let bytes = crate::geophys::b64_decode(b).unwrap();
        bytes.chunks(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
    }

    #[test]
    fn a_frame_packs_to_pixels_and_repeats_share_data() {
        let a = square([1.0, 0.0, 0.0, 1.0]);
        let b = square([0.0, 0.0, 1.0, 1.0]);
        let page = export_html("t", &[a.clone(), b, a], 1000, 500, 24.0);
        assert!(page.contains("\"frames\":[0,"), "{}", &page[..400.min(page.len())]);
        let meta = page.split("const M = ").nth(1).unwrap().split(";\n").next().unwrap();
        let frames: Vec<usize> = meta
            .split("\"frames\":[")
            .nth(1)
            .unwrap()
            .split(']')
            .next()
            .unwrap()
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0], frames[2], "a repeated frame reuses its data");
        assert!(meta.contains("rgba(255,0,0,1.000)") && meta.contains("rgba(0,0,255,1.000)"));
        let d = decode(&page);
        // count, style, closed, rings, len, then x0 y0: 0.1 of 1000 px × 4.
        assert_eq!(&d[..7], &[1, 0, 1, 1, 4, 400, 400]);
    }

    #[test]
    fn linear_mid_grey_reads_as_srgb() {
        assert_eq!(srgb(0.2158), 128);
        assert_eq!(srgb(1.0), 255);
        assert_eq!(css([0.0; 4]), None);
    }
}
