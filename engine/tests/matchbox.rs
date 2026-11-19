//! The GLSL dialect of the Shader node (Flame Matchbox shaders).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{Graph, NodeKind, ShaderParams};
use engine::timing::GpuTimer;
use engine::{matchbox, usershader};
use half::f16;
use std::collections::BTreeMap;
use std::path::Path;

const W: u32 = 64;
const H: u32 = 32;

fn ctx() -> Option<engine::GpuContext> {
    engine::GpuContext::new().map_err(|e| eprintln!("SKIP: no adapter ({e})")).ok()
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, px: &[[f16; 4]]) {
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &comp.reader_texture(0).unwrap().texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(px),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 8),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
}

fn render(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<[f32; 4]> {
    assert!(comp.set_graph(graph), "the plan failed to compile");
    let timer = GpuTimer::new(ctx);
    let mut warm = ctx.device.create_command_encoder(&Default::default());
    comp.encode_comp(&mut warm, &timer);
    ctx.queue.submit([warm.finish()]);
    comp.write_params(graph);
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    comp.encode_comp(&mut encoder, &timer);
    let padded = (W * 8).div_ceil(256) * 256;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (padded * H) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: comp.output_texture().unwrap(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().expect("a mapped buffer");
    let mut out = Vec::new();
    for y in 0..H as usize {
        let halves: &[f16] = bytemuck::cast_slice(&data[y * padded as usize..][..(W * 8) as usize]);
        out.extend(halves.chunks(4).map(|p| [p[0].to_f32(), p[1].to_f32(), p[2].to_f32(), p[3].to_f32()]));
    }
    drop(data);
    staging.unmap();
    out
}

/// Red ramps left to right, green ramps top to bottom, alpha 0.5.
fn ramp() -> Vec<[f16; 4]> {
    let mut v = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let c = [(x as f32 + 0.5) / W as f32, (y as f32 + 0.5) / H as f32, 0.0, 0.5];
            v.push(c.map(f16::from_f32));
        }
    }
    v
}

/// Something to look at: the ramp with a bright disc and a dark bar in it.
fn plate() -> Vec<[f16; 4]> {
    let mut v = ramp();
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            let (dx, dy) = (x - 22, y - 16);
            let p = &mut v[(y as u32 * W + x as u32) as usize];
            if dx * dx + dy * dy < 64 {
                *p = [1.0, 0.9, 0.7, 1.0].map(f16::from_f32);
            } else if (44..48).contains(&x) {
                *p = [0.05, 0.05, 0.1, 1.0].map(f16::from_f32);
            }
        }
    }
    v
}

fn graph_of(src: &str) -> Graph {
    let d = usershader::compile(src);
    assert!(d.ok(), "{:?}", d.errors);
    let mut p = ShaderParams::default();
    p.set_source(src.to_string());
    let mut g = Graph::default();
    g.nodes.insert(1, NodeKind::read(0, ""));
    g.nodes.insert(2, NodeKind::Shader(p));
    g.edges.insert((2, 0), 1);
    g.view = Some(2);
    g
}

fn run(src: &str) -> Vec<[f32; 4]> {
    let Some(ctx) = ctx() else { return Vec::new() };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, &ramp());
    render(&ctx, &mut comp, &graph_of(src))
}

fn at(px: &[[f32; 4]], x: u32, y: u32) -> [f32; 4] {
    px[(y * W + x) as usize]
}

fn near(a: f32, b: f32) -> bool {
    (a - b).abs() < 0.01
}

const HEAD: &str = "#version 120\nuniform sampler2D front;\n\
uniform float adsk_result_w, adsk_result_h;\n\
vec2 res = vec2(adsk_result_w, adsk_result_h);\n";

#[test]
fn reading_the_input_is_upright_and_keeps_its_alpha() {
    let src = format!("{HEAD}void main() {{ gl_FragColor = vec4(texture2D(front, gl_FragCoord.xy / res).rgb, 0.0); }}");
    let px = run(&src);
    if px.is_empty() { return; }
    let c = at(&px, 10, 4);
    assert!(near(c[0], 10.5 / W as f32) && near(c[1], 4.5 / H as f32), "{c:?}");
    assert!(near(c[3], 0.5), "alpha comes from the input without a matte: {c:?}");
}

#[test]
fn frag_coord_starts_at_the_bottom_left() {
    let src = format!("{HEAD}void main() {{ gl_FragColor = vec4(gl_FragCoord.xy / res, 0.0, 1.0); }}");
    let px = run(&src);
    if px.is_empty() { return; }
    let top = at(&px, 0, 0);
    assert!(near(top[1], 1.0 - 0.5 / H as f32), "top row is y = 1 in GLSL: {top:?}");
}

#[test]
fn the_xml_supplies_defaults_and_labels() {
    let src = format!(
        "{HEAD}uniform float gain;\nuniform vec3 tint;\nuniform bool off;\nuniform int mode;\n\
void main() {{ vec3 c = vec3(gain) * tint; if (off) c = vec3(0.0); if (mode == 2) c *= 2.0; gl_FragColor = vec4(c, 1.0); }}\n\
<ShaderNodePreset Name=\"Thing\" MatteProvider=\"False\"><Shader Index=\"1\">\
<Uniform Name=\"front\" Type=\"sampler2D\" Index=\"0\" InputType=\"Front\"></Uniform>\
<Uniform Name=\"gain\" Type=\"float\" DisplayName=\"Gain\" Default=\"0.5\" Min=\"0\" Max=\"4\" Page=\"0\" Col=\"0\" Row=\"0\"></Uniform>\
<Uniform Name=\"tint\" Type=\"vec3\" ValueType=\"Colour\" DisplayName=\"Tint\" Page=\"0\" Col=\"0\" Row=\"1\">\
<SubUniform Default=\"1.0\" Min=\"-1000000.0\" Max=\"1000000.0\"></SubUniform>\
<SubUniform Default=\"0.5\" Min=\"-1000000.0\" Max=\"1000000.0\"></SubUniform>\
<SubUniform Default=\"0.0\" Min=\"-1000000.0\" Max=\"1000000.0\"></SubUniform></Uniform>\
<Uniform Name=\"off\" Type=\"bool\" Default=\"False\"></Uniform>\
<Uniform Name=\"mode\" Type=\"int\" Default=\"2\" Min=\"0\" Max=\"9\"><PopupEntry Title=\"One\" Value=\"1\"></PopupEntry>\
<PopupEntry Title=\"Two\" Value=\"2\"></PopupEntry></Uniform>\
</Shader><Page Name=\"Main\" Page=\"0\"><Col Name=\"Look\" Col=\"0\" Page=\"0\"></Col></Page></ShaderNodePreset>"
    );
    let d = usershader::compile(&src);
    assert!(d.ok(), "{:?}", d.errors);
    assert_eq!(d.label, "Thing");
    let g = d.param("gain").unwrap();
    assert_eq!((g.label.as_str(), g.default[0], g.max), ("Gain", 0.5, 4.0));
    assert_eq!(g.group.as_deref(), Some("Look"));
    assert_eq!(d.param("tint").unwrap().default[..3], [1.0, 0.5, 0.0]);
    assert_eq!(d.param("tint").unwrap().max, 0.0, "a huge range means no slider");
    assert_eq!(d.param("mode").unwrap().choices.len(), 2);
    let px = run(&src);
    if px.is_empty() { return; }
    let c = at(&px, 5, 5);
    assert!(near(c[0], 1.0) && near(c[1], 0.5) && near(c[2], 0.0), "{c:?}");
}

#[test]
fn passes_read_each_other_and_repeats_accumulate() {
    let one = format!(
        "{HEAD}uniform sampler2D adsk_results_pass1;\nuniform int adsk_result_execution;\n\
void main() {{ float v = texture2D(adsk_results_pass1, gl_FragCoord.xy / res).r;\n\
 gl_FragColor = vec4(adsk_result_execution == 0 ? 1.0 : v + 1.0); }}"
    );
    let two = format!(
        "{HEAD}uniform sampler2D adsk_results_pass1;\n\
void main() {{ gl_FragColor = texture2D(adsk_results_pass1, gl_FragCoord.xy / res) * 2.0; }}"
    );
    let xml = "<ShaderNodePreset Name=\"R\"><Shader Index=\"1\" NbExecutions=\"5\"></Shader></ShaderNodePreset>";
    let src = matchbox::bundle(&[one, two], Some(xml));
    let d = usershader::compile(&src);
    assert!(d.ok(), "{:?}", d.errors);
    assert_eq!(d.passes, 6);
    assert!(d.result_textures() <= 3, "results are recycled: {}", d.result_textures());
    let px = run(&src);
    if px.is_empty() { return; }
    assert!(near(at(&px, 9, 9)[0], 10.0), "{:?}", at(&px, 9, 9));
}

#[test]
fn an_unwired_input_can_read_white() {
    let src = format!(
        "{HEAD}uniform sampler2D strength, other;\n\
void main() {{ gl_FragColor = vec4(texture2D(strength, vec2(0.5)).r, texture2D(other, vec2(0.5)).r, 0.0, 1.0); }}\n\
<ShaderNodePreset Name=\"W\"><Shader Index=\"1\">\
<Uniform Name=\"strength\" Type=\"sampler2D\" NoInput=\"White\"></Uniform>\
<Uniform Name=\"other\" Type=\"sampler2D\" NoInput=\"Black\"></Uniform>\
</Shader></ShaderNodePreset>"
    );
    let px = run(&src);
    if px.is_empty() { return; }
    let c = at(&px, 3, 3);
    assert!(near(c[0], 1.0) && near(c[1], 0.0), "{c:?}");
}

#[test]
fn derivatives_are_real_differences() {
    let src = format!(
        "{HEAD}float f(float x) {{ return x * x; }}\n\
void main() {{ gl_FragColor = vec4(dFdx(gl_FragCoord.x), dFdy(gl_FragCoord.y), fwidth(gl_FragCoord.x), 1.0); }}"
    );
    let px = run(&src);
    if px.is_empty() { return; }
    let c = at(&px, 20, 9);
    assert!(near(c[0], 1.0) && near(c[1], 1.0) && near(c[2], 1.0), "{c:?}");
}

#[test]
fn glsl_120_habits_compile() {
    let src = "#version 120\n#extension GL_ARB_shader_texture_lod : enable\n\
uniform sampler2D front, uv;\nuniform float size, adsk_result_w, adsk_result_h;\n\
const float table[] = float[](1.0, 2.0, 3.0);\n\
float later(float x);\n\
vec4 blur(sampler2D s, vec2 p) { return texture2D(s, p) * later(size); }\n\
float helper(float x) { return x * 2.0; }\n\
float later(float x) { return helper(x) + table[1]; }\n\
void main(void) { gl_FragData[0] = blur(front, gl_TexCoord[0].st) + texture2D(uv, vec2(0.5)); }\n";
    let d = usershader::compile(src);
    assert!(d.ok(), "{:?}", d.errors);
    assert_eq!(d.inputs, vec!["front", "uv"]);
}

#[test]
fn a_non_commercial_preset_taints_the_node() {
    let src = "void main() { gl_FragColor = vec4(1.0); }\n<ShaderNodePreset Name=\"N\" CommercialUsePermitted=\"False\"></ShaderNodePreset>";
    let mut p = ShaderParams::default();
    p.set_source(src.to_string());
    let kind = NodeKind::Shader(p);
    assert_eq!(engine::cost::licence_of(&kind), engine::cost::Licence::NonCommercial);
}

#[test]
fn pasting_over_the_starter_takes_the_presets_default() {
    let mut p = ShaderParams::glsl();
    p.set_source(
        "uniform float amount;\nvoid main() { gl_FragColor = vec4(amount); }\n\
<ShaderNodePreset Name=\"N\"><Shader Index=\"1\"><Uniform Name=\"amount\" Type=\"float\" Default=\"0.75\"></Uniform></Shader></ShaderNodePreset>"
            .to_string(),
    );
    assert_eq!(p.values["amount"][0], 0.75);
}

#[test]
fn errors_point_at_the_users_line() {
    let d = usershader::compile("uniform float a;\n\nvoid main() {\n  gl_FragColor = nope(a);\n}\n");
    assert!(!d.ok());
    assert_eq!(d.errors[0].line, Some(4), "{:?}", d.errors);
}

/// Compile every shader in a folder of Matchbox files and report the failures:
/// `WOBBLE_MATCHBOX_DIR=<dir> cargo test -p engine --test matchbox corpus -- --ignored --nocapture`
#[test]
#[ignore]
fn corpus() {
    let dir = std::env::var("WOBBLE_MATCHBOX_DIR").expect("WOBBLE_MATCHBOX_DIR");
    let mut sets: BTreeMap<String, BTreeMap<u32, String>> = BTreeMap::new();
    for e in std::fs::read_dir(&dir).unwrap().flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let Some(stem) = name.strip_suffix(".glsl") else { continue };
        let (base, pass) = match stem.rsplit_once('.') {
            Some((b, n)) if n.parse::<u32>().is_ok() => (b.to_string(), n.parse().unwrap()),
            _ => (stem.to_string(), 1),
        };
        if let Ok(t) = std::fs::read(e.path()) {
            sets.entry(base).or_default().insert(pass, String::from_utf8_lossy(&t).into_owned());
        }
    }
    let (mut good, mut bad) = (0, 0);
    let mut built = Vec::new();
    let mut why: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (base, passes) in &sets {
        let xml = std::fs::read(Path::new(&dir).join(format!("{base}.xml")))
            .ok()
            .map(|b| String::from_utf8_lossy(&b).into_owned());
        let list: Vec<String> = passes.values().cloned().collect();
        let src = matchbox::bundle(&list, xml.as_deref());
        let d = usershader::compile(&src);
        if d.ok() {
            good += 1;
            built.push((base.clone(), src));
        } else {
            bad += 1;
            let e = &d.errors[0];
            let at = e.line.and_then(|l| src.lines().nth(l - 1)).unwrap_or("").trim();
            why.entry(e.msg.chars().take(90).collect())
                .or_default()
                .push(format!("{base}:{:?} `{}`", e.line, at.chars().take(60).collect::<String>()));
        }
    }
    let mut v: Vec<_> = why.into_iter().collect();
    v.sort_by_key(|(_, b)| std::cmp::Reverse(b.len()));
    for (m, b) in v {
        println!("{:3}  {m}\n       {}", b.len(), b.iter().take(4).cloned().collect::<Vec<_>>().join(" "));
    }
    println!("CORPUS: {good} ok, {bad} failed of {}", good + bad);

    // What compiled must also build real pipelines, dispatch, and make a
    // picture. WOBBLE_MATCHBOX_SHEET=<png> writes them all as a contact sheet.
    let Some(ctx) = ctx() else { return };
    let mut rendered = 0;
    let (cols, scale) = (5usize, 3usize);
    let (tw, th) = (W as usize * scale, H as usize * scale);
    let rows = built.len().div_ceil(cols);
    let mut sheet = vec![30u8; (tw + 4) * cols * (th + 4) * rows * 3];
    for (i, (name, src)) in built.iter().enumerate() {
        let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
        comp.ensure_output(W, H);
        upload(&ctx, &mut comp, &plate());
        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let px = render(&ctx, &mut comp, &graph_of(src));
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "{name}: {err:?}");
        let nan = px.iter().filter(|p| p.iter().any(|v| !v.is_finite())).count();
        let lit = px.iter().filter(|p| p[..3].iter().any(|v| v.abs() > 1e-3)).count();
        let same = px.iter().zip(plate()).filter(|(a, b)| {
            (0..3).all(|k| (a[k] - b[k].to_f32()).abs() < 1e-3)
        }).count();
        println!("{name:18} nan {nan:4}  lit {lit:4}  same-as-input {same:4} of {}", px.len());
        let (ox, oy) = ((i % cols) * (tw + 4), (i / cols) * (th + 4));
        for y in 0..th {
            for x in 0..tw {
                let p = px[(y / scale) * W as usize + x / scale];
                let o = ((oy + y) * (tw + 4) * cols + ox + x) * 3;
                for k in 0..3 {
                    sheet[o + k] = (p[k].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
                }
            }
        }
        rendered += 1;
    }
    if let Ok(out) = std::env::var("WOBBLE_MATCHBOX_SHEET") {
        let f = std::io::BufWriter::new(std::fs::File::create(out).unwrap());
        let mut e = png::Encoder::new(f, ((tw + 4) * cols) as u32, ((th + 4) * rows) as u32);
        e.set_color(png::ColorType::Rgb);
        e.set_depth(png::BitDepth::Eight);
        e.write_header().unwrap().write_image_data(&sheet).unwrap();
    }
    println!("RENDERED: {rendered}");
}
