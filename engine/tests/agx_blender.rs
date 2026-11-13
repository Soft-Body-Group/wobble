//! Wobble's AgX against Blender's own: scene-linear Rec.709 patches pushed
//! through Blender 5.2.1 LTS (view AgX, look None, display sRGB, saved as a
//! 16-bit PNG, dither off) and through `viewtf` with the same cube. The
//! table comes from `tools/agx_reference.py`; rerun it after a Blender
//! upgrade that changes AgX. Skips when no Blender is installed.
//!
//! The looks' table (`LOOKS`) is OCIO 2.6 running Blender's own
//! `config.ocio` and cubes on the same patches; `tools/agx_reference.py
//! --looks` makes the same table from Blender itself.

use engine::viewtf::{agx, Eotf, Gamut, Look, ViewSettings, ViewTransform};

/// (scene-linear Rec.709, Blender's sRGB-display code values).
const BLENDER: &[([f32; 3], [f32; 3])] = &[
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ([0.001, 0.001, 0.001], [0.0033, 0.0033, 0.0033]),
    ([0.01, 0.01, 0.01], [0.0725, 0.0725, 0.07251]),
    ([0.05, 0.05, 0.05], [0.22721, 0.22721, 0.22724]),
    ([0.18, 0.18, 0.18], [0.46133, 0.46133, 0.46137]),
    ([0.5, 0.5, 0.5], [0.66517, 0.66517, 0.6652]),
    ([1.0, 1.0, 1.0], [0.77098, 0.77098, 0.77099]),
    ([2.0, 2.0, 2.0], [0.8518, 0.8518, 0.85182]),
    ([4.0, 4.0, 4.0], [0.91369, 0.91369, 0.91371]),
    ([16.0, 16.0, 16.0], [0.99857, 0.99857, 0.99857]),
    ([100.0, 100.0, 100.0], [1.0, 1.0, 1.0]),
    ([1.0, 0.0, 0.0], [0.85801, 0.22409, 0.1292]),
    ([0.0, 1.0, 0.0], [0.43247, 0.77264, 0.31843]),
    ([0.0, 0.0, 1.0], [0.07909, 0.33761, 0.8215]),
    ([0.18, 0.02, 0.01], [0.44611, 0.12834, 0.04468]),
    ([0.02, 0.1, 0.3], [0.1039, 0.36918, 0.57729]),
    ([0.9, 0.6, 0.1], [0.77198, 0.67553, 0.4548]),
    ([4.0, 0.5, 0.1], [0.97725, 0.69749, 0.61225]),
    ([0.3, 2.0, 0.4], [0.69062, 0.85003, 0.67665]),
    ([0.05, 0.2, 5.0], [0.6034, 0.69889, 0.97879]),
    ([0.7, 0.35, 0.9], [0.71307, 0.63461, 0.76295]),
    ([0.123, 0.456, 0.789], [0.48188, 0.65214, 0.74917]),
    ([8.0, 8.0, 0.2], [0.96797, 0.9555, 0.8253]),
];

/// Each look's sRGB-display code values for `BLENDER`'s patches, in order.
const LOOKS: &[(Look, [[f32; 3]; 23])] = &[
    (Look::Punchy, [
        [0.00136, 0.00136, 0.00136],
        [0.00172, 0.00172, 0.00172],
        [0.01003, 0.01003, 0.01003],
        [0.13608, 0.13608, 0.1361],
        [0.34629, 0.34629, 0.34632],
        [0.56317, 0.56317, 0.56321],
        [0.69234, 0.69234, 0.69236],
        [0.79383, 0.79383, 0.79385],
        [0.87192, 0.87192, 0.87194],
        [0.97702, 0.97702, 0.97703],
        [1.00002, 1.00002, 1.00002],
        [0.73067, 0.11539, 0.0],
        [0.27557, 0.68671, 0.11543],
        [0.01817, 0.26117, 0.70786],
        [0.31566, 0.0547, 0.0],
        [0.00854, 0.2713, 0.44671],
        [0.68842, 0.57754, 0.30355],
        [0.9427, 0.60038, 0.48799],
        [0.58807, 0.79116, 0.5732],
        [0.48188, 0.60839, 0.94542],
        [0.6209, 0.52545, 0.68051],
        [0.3313, 0.54909, 0.66193],
        [0.93894, 0.92358, 0.75928],
    ]),
    (Look::Greyscale, [
        [0.0, 0.0, 0.0],
        [0.0033, 0.0033, 0.0033],
        [0.0725, 0.0725, 0.07251],
        [0.22721, 0.22721, 0.22723],
        [0.46132, 0.46132, 0.46135],
        [0.66515, 0.66515, 0.66518],
        [0.77096, 0.77096, 0.77098],
        [0.85178, 0.85178, 0.8518],
        [0.91367, 0.91367, 0.91368],
        [0.99855, 0.99855, 0.99855],
        [1.00002, 1.00002, 1.00002],
        [0.52238, 0.52238, 0.52242],
        [0.69034, 0.69034, 0.69036],
        [0.46787, 0.46787, 0.4679],
        [0.24428, 0.24428, 0.2443],
        [0.37012, 0.37012, 0.37016],
        [0.69046, 0.69045, 0.69048],
        [0.80024, 0.80024, 0.80026],
        [0.80443, 0.80443, 0.80445],
        [0.7763, 0.7763, 0.77633],
        [0.67638, 0.67638, 0.67641],
        [0.64148, 0.64147, 0.6415],
        [0.94956, 0.94956, 0.94957],
    ]),
    (Look::VeryHighContrast, [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.00955, 0.00955, 0.00955],
        [0.14197, 0.14197, 0.14198],
        [0.46131, 0.46131, 0.46135],
        [0.75541, 0.75541, 0.75543],
        [0.87932, 0.87932, 0.87933],
        [0.96071, 0.96071, 0.96072],
        [1.00002, 1.00002, 1.00002],
        [1.00002, 1.00002, 1.00002],
        [1.00002, 1.00002, 1.00002],
        [0.9421, 0.219, 0.18751],
        [0.4496, 0.8811, 0.3351],
        [0.24808, 0.38438, 0.87742],
        [0.34968, 0.044, 0.0],
        [0.00162, 0.32527, 0.57902],
        [0.88211, 0.76125, 0.50814],
        [1.00002, 0.8168, 0.75831],
        [0.80176, 0.95794, 0.79332],
        [0.78173, 0.83514, 1.00002],
        [0.80596, 0.72202, 0.86301],
        [0.53261, 0.7302, 0.84895],
        [1.00002, 1.00002, 0.95753],
    ]),
    (Look::HighContrast, [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.01904, 0.01904, 0.01905],
        [0.16458, 0.16458, 0.16459],
        [0.46131, 0.46131, 0.46135],
        [0.73104, 0.73104, 0.73107],
        [0.85108, 0.85108, 0.85109],
        [0.93383, 0.93383, 0.93384],
        [0.99155, 0.99155, 0.99156],
        [1.00002, 1.00002, 1.00002],
        [1.00002, 1.00002, 1.00002],
        [0.9285, 0.20424, 0.1572],
        [0.41572, 0.85577, 0.28602],
        [0.21581, 0.37232, 0.87099],
        [0.38249, 0.05976, 0.0],
        [0.01153, 0.338, 0.58668],
        [0.85576, 0.73631, 0.47939],
        [1.00002, 0.77643, 0.7039],
        [0.76675, 0.93246, 0.75709],
        [0.72543, 0.79361, 1.00002],
        [0.78271, 0.69615, 0.83977],
        [0.51279, 0.70905, 0.82508],
        [1.00002, 1.00002, 0.9214],
    ]),
    (Look::MediumHighContrast, [
        [0.0, 0.0, 0.0],
        [0.00024, 0.00024, 0.00025],
        [0.04009, 0.04009, 0.0401],
        [0.19403, 0.19403, 0.19405],
        [0.46132, 0.46131, 0.46135],
        [0.699, 0.699, 0.69902],
        [0.81378, 0.81378, 0.8138],
        [0.8965, 0.8965, 0.89651],
        [0.95706, 0.95706, 0.95707],
        [1.00002, 1.00002, 1.00002],
        [1.00002, 1.00002, 1.00002],
        [0.90494, 0.18883, 0.0921],
        [0.40836, 0.81883, 0.2822],
        [0.14363, 0.35075, 0.85813],
        [0.4226, 0.08315, 0.0],
        [0.03832, 0.35283, 0.59075],
        [0.8188, 0.70639, 0.45594],
        [1.00002, 0.72979, 0.64341],
        [0.72512, 0.89661, 0.71373],
        [0.6562, 0.74349, 1.00002],
        [0.7507, 0.66611, 0.80614],
        [0.49037, 0.6829, 0.79206],
        [0.9999, 0.99458, 0.87186],
    ]),
    (Look::BaseContrast, [
        [0.0, 0.0, 0.0],
        [0.0033, 0.0033, 0.0033],
        [0.0725, 0.0725, 0.07251],
        [0.22721, 0.22721, 0.22723],
        [0.46132, 0.46132, 0.46135],
        [0.66515, 0.66515, 0.66518],
        [0.77096, 0.77096, 0.77098],
        [0.85178, 0.85178, 0.8518],
        [0.91367, 0.91367, 0.91368],
        [0.99855, 0.99855, 0.99855],
        [1.00002, 1.00002, 1.00002],
        [0.8576, 0.22603, 0.13354],
        [0.43246, 0.77263, 0.31843],
        [0.13117, 0.3499, 0.81655],
        [0.44611, 0.12834, 0.04467],
        [0.10391, 0.36916, 0.57728],
        [0.77196, 0.67552, 0.45479],
        [0.97723, 0.69747, 0.61224],
        [0.6906, 0.85001, 0.67663],
        [0.61059, 0.70262, 0.97705],
        [0.71305, 0.63459, 0.76293],
        [0.48188, 0.65212, 0.74916],
        [0.96795, 0.95548, 0.82528],
    ]),
    (Look::MediumLowContrast, [
        [0.00024, 0.00024, 0.00024],
        [0.00779, 0.00779, 0.00779],
        [0.09214, 0.09214, 0.09215],
        [0.24555, 0.24555, 0.24557],
        [0.46132, 0.46131, 0.46135],
        [0.64722, 0.64722, 0.64725],
        [0.7469, 0.7469, 0.74692],
        [0.82587, 0.82587, 0.82589],
        [0.88792, 0.88792, 0.88793],
        [0.97577, 0.97577, 0.97578],
        [1.00002, 1.00002, 1.00002],
        [0.84083, 0.24828, 0.16143],
        [0.42354, 0.75102, 0.31208],
        [0.17933, 0.35797, 0.80483],
        [0.46377, 0.14974, 0.07097],
        [0.12913, 0.37882, 0.57801],
        [0.75036, 0.65694, 0.44235],
        [0.95508, 0.67454, 0.58606],
        [0.66382, 0.82611, 0.65059],
        [0.5893, 0.68299, 0.95779],
        [0.69528, 0.61777, 0.74329],
        [0.47183, 0.63676, 0.73052],
        [0.94427, 0.93109, 0.79096],
    ]),
    (Look::LowContrast, [
        [0.00165, 0.00165, 0.00165],
        [0.01637, 0.01637, 0.01638],
        [0.11482, 0.11482, 0.11483],
        [0.26449, 0.26449, 0.26451],
        [0.46132, 0.46132, 0.46135],
        [0.62929, 0.62928, 0.62931],
        [0.72195, 0.72195, 0.72198],
        [0.79762, 0.79762, 0.79764],
        [0.85893, 0.85893, 0.85895],
        [0.94894, 0.94894, 0.94895],
        [1.00002, 1.00002, 1.00002],
        [0.81993, 0.27202, 0.19557],
        [0.41767, 0.72753, 0.31048],
        [0.21357, 0.36679, 0.79094],
        [0.48056, 0.17388, 0.10017],
        [0.16636, 0.38873, 0.57611],
        [0.72761, 0.63852, 0.43115],
        [0.92786, 0.65202, 0.56085],
        [0.63768, 0.79915, 0.62334],
        [0.56672, 0.66074, 0.93464],
        [0.67485, 0.59989, 0.72165],
        [0.45973, 0.62108, 0.71082],
        [0.91645, 0.90279, 0.75447],
    ]),
    (Look::VeryLowContrast, [
        [0.00589, 0.00589, 0.00589],
        [0.03256, 0.03256, 0.03256],
        [0.14086, 0.14086, 0.14087],
        [0.28566, 0.28565, 0.28568],
        [0.46132, 0.46132, 0.46135],
        [0.60973, 0.60973, 0.60976],
        [0.69496, 0.69496, 0.69499],
        [0.76685, 0.76685, 0.76688],
        [0.82627, 0.82627, 0.82629],
        [0.91678, 0.91678, 0.91679],
        [0.99551, 0.99551, 0.99551],
        [0.7959, 0.29414, 0.2268],
        [0.41713, 0.70234, 0.31662],
        [0.23429, 0.37791, 0.77116],
        [0.49419, 0.20175, 0.13606],
        [0.20852, 0.39841, 0.57224],
        [0.70126, 0.61926, 0.42412],
        [0.89632, 0.62764, 0.53969],
        [0.61259, 0.76899, 0.59837],
        [0.54288, 0.63966, 0.90651],
        [0.65392, 0.58201, 0.69857],
        [0.45469, 0.60395, 0.68763],
        [0.88418, 0.87024, 0.71599],
    ]),
];

fn srgb_agx(look: Look) -> ViewSettings {
    let mut v = ViewSettings {
        transform: ViewTransform::AgX,
        display: Gamut::Rec709,
        eotf: Eotf::Srgb,
        look,
        ..Default::default()
    };
    v.mirror_working(engine::color::Colorspace::LINEAR_REC709);
    v
}

/// Worst channel error of `got` against `want`, and the patch it was on.
fn worst(got: &[[f32; 3]], want: &[[f32; 3]]) -> (f32, [f32; 3]) {
    let mut w = (0.0f32, [0.0f32; 3]);
    for ((g, t), (scene, _)) in got.iter().zip(want).zip(BLENDER) {
        for k in 0..3 {
            let e = (g[k] - t[k]).abs();
            if e > w.0 {
                w = (e, *scene);
            }
        }
    }
    w
}

#[test]
fn agx_looks_match_blender() {
    if agx::lumcomp().is_none() {
        eprintln!("no Blender (or no AgX looks in it): skipping");
        return;
    }
    assert_eq!(LOOKS.len(), Look::AGX.len() - 1, "every AgX look has a row");
    for (look, want) in LOOKS {
        let v = srgb_agx(*look);
        let got: Vec<[f32; 3]> =
            BLENDER.iter().map(|(s, _)| v.display_rgb(*s).expect("AgX readout")).collect();
        let (e, at) = worst(&got, want);
        eprintln!("{look:?} vs Blender: worst {e:.5} at {at:?}");
        assert!(e < 1.0 / 512.0, "{look:?}: worst {e:.5} at {at:?}");
    }
}

#[test]
fn agx_matches_blender() {
    if !agx::available() {
        eprintln!("no Blender install found: skipping");
        return;
    }
    let mut v = ViewSettings {
        transform: ViewTransform::AgX,
        display: Gamut::Rec709,
        eotf: Eotf::Srgb,
        ..Default::default()
    };
    v.mirror_working(engine::color::Colorspace::LINEAR_REC709);
    let mut worst = (0.0f32, [0.0f32; 3]);
    for (scene, want) in BLENDER {
        let got = v.display_rgb(*scene).expect("AgX readout");
        for k in 0..3 {
            let e = (got[k] - want[k]).abs();
            if e > worst.0 {
                worst = (e, *scene);
            }
        }
    }
    eprintln!("AgX vs Blender: worst {:.5} at {:?}", worst.0, worst.1);
    // 16-bit PNG quantisation is 1.5e-5; hold it well under one 8-bit step.
    assert!(worst.0 < 1.0 / 512.0, "worst {:.5} at {:?}", worst.0, worst.1);
}

/// The SHADER, not just its CPU mirror: the same patches through a real
/// Blit (tetrahedral `textureLoad` on the uploaded cube) into a float
/// target, read back and held to Blender's numbers — the base view and
/// every look.
#[test]
fn agx_shader_matches_blender() {
    if !agx::available() {
        eprintln!("no Blender install found: skipping");
        return;
    }
    let Ok(ctx) = engine::GpuContext::new() else {
        eprintln!("no adapter; skipping");
        return;
    };
    let base: Vec<[f32; 3]> = BLENDER.iter().map(|(_, w)| *w).collect();
    let mut cases = vec![(Look::None, base)];
    if agx::lumcomp().is_some() {
        cases.extend(LOOKS.iter().map(|(l, w)| (*l, w.to_vec())));
    }
    for (look, want) in cases {
        let got = blit_patches(&ctx, &srgb_agx(look));
        let (e, at) = worst(&got, &want);
        eprintln!("AgX shader {look:?} vs Blender: worst {e:.5} at {at:?}");
        assert!(e < 1.0 / 512.0, "{look:?}: worst {e:.5} at {at:?}");
    }
}

/// `BLENDER`'s patches through a real Blit with `v`, read back.
fn blit_patches(ctx: &engine::GpuContext, v: &ViewSettings) -> Vec<[f32; 3]> {
    const FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
    let w = BLENDER.len() as u32;
    let tex = |usage, format| {
        ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: w, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    // The comp's own output format: half float, filterable.
    let src = tex(
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        wgpu::TextureFormat::Rgba16Float,
    );
    let dst = tex(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC, FMT);
    let px: Vec<half::f16> = BLENDER
        .iter()
        .flat_map(|(s, _)| [s[0], s[1], s[2], 1.0])
        .map(half::f16::from_f32)
        .collect();
    ctx.queue.write_texture(
        src.as_image_copy(),
        bytemuck::cast_slice(&px),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: None },
        src.size(),
    );
    let mut blit = engine::blit::Blit::new(&ctx.device, FMT);
    blit.set_view(&ctx.device, v);
    let bpr = (w * 16).div_ceil(256) * 256;
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bpr as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    blit.encode(
        &ctx.device,
        &ctx.queue,
        &mut enc,
        &src.create_view(&Default::default()),
        &dst.create_view(&Default::default()),
        None,
        [0.0, 0.0, w as f32, 1.0],
        false,
        v,
    );
    enc.copy_texture_to_buffer(
        dst.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: None },
        },
        dst.size(),
    );
    ctx.queue.submit([enc.finish()]);
    buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let data = buf.slice(..).get_mapped_range().expect("a mapped buffer");
    let got: &[[f32; 4]] = bytemuck::cast_slice(&data[..(w * 16) as usize]);
    got.iter().map(|g| [g[0], g[1], g[2]]).collect()
}
