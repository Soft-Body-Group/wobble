//! What every bake shares, whichever node asked for it: the marker that says
//! a directory's frames are complete, and the EXR writer a subtree render
//! lands its frames through.

use crate::graph::ExrCompression;
use exr::prelude::{Encoding, Image, Layer, LayerAttributes, SpecificChannels, Vec2, WritableImage};
use half::f16;
use std::path::Path;

/// Written into a bake directory once every frame is on disk. A directory
/// without it is a bake that stopped — adopt nothing from it.
const DONE: &str = ".wobble-bake-done";

pub fn is_baked(dir: &Path) -> bool {
    dir.join(DONE).is_file()
}

pub fn mark_baked(dir: &Path) -> std::io::Result<()> {
    std::fs::write(dir.join(DONE), b"")
}

/// The file layout for a compression pick. RLE keeps the tiled layout every
/// render had before the pick existed (so its bytes are unchanged); the rest
/// are scanline files in increasing order, which every reader takes and the
/// zip methods are tuned for.
pub fn encoding(c: ExrCompression) -> Encoding {
    use exr::prelude::{Blocks, Compression, LineOrder};
    let compression = match c {
        ExrCompression::Rle => return Encoding::FAST_LOSSLESS,
        ExrCompression::None => Compression::Uncompressed,
        ExrCompression::Zip1 => Compression::ZIP1,
        ExrCompression::Zip16 => Compression::ZIP16,
        ExrCompression::Piz => Compression::PIZ,
        ExrCompression::Pxr24 => Compression::PXR24,
        ExrCompression::B44 => Compression::B44,
        ExrCompression::B44a => Compression::B44A,
    };
    Encoding { compression, blocks: Blocks::ScanLines, line_order: LineOrder::Increasing }
}

fn tag<L>(image: &mut Image<L>, chroma: Option<[(f32, f32); 4]>) {
    if let Some([r, g, b, wp]) = chroma {
        image.attributes.chromaticities = Some(exr::meta::attribute::Chromaticities {
            red: Vec2(r.0, r.1),
            green: Vec2(g.0, g.1),
            blue: Vec2(b.0, b.1),
            white: Vec2(wp.0, wp.1),
        });
    }
}

/// One rgba f16 EXR, optionally tagged with the chromaticities the pixels
/// are in, so a reader outside Wobble knows what it is looking at.
pub fn write_rgba_f16_exr_tagged(
    path: &Path,
    w: usize,
    h: usize,
    row_stride_f16: usize,
    data: &[f16],
    chroma: Option<[(f32, f32); 4]>,
) -> anyhow::Result<()> {
    write_rgba_f16_exr_with(path, w, h, row_stride_f16, data, chroma, ExrCompression::Rle)
}

/// `write_rgba_f16_exr_tagged` with the compression picked.
pub fn write_rgba_f16_exr_with(
    path: &Path,
    w: usize,
    h: usize,
    row_stride_f16: usize,
    data: &[f16],
    chroma: Option<[(f32, f32); 4]>,
    compression: ExrCompression,
) -> anyhow::Result<()> {
    let layer = Layer::new(
        (w, h),
        LayerAttributes::named("rgba"),
        encoding(compression),
        SpecificChannels::rgba(|p: Vec2<usize>| {
            let px = &data[p.y() * row_stride_f16 + p.x() * 4..];
            (px[0], px[1], px[2], px[3])
        }),
    );
    let mut image = Image::from_layer(layer);
    tag(&mut image, chroma);
    image.write().to_file(path)?;
    Ok(())
}

/// What a single-channel frame takes from each rgba pixel.
#[derive(Clone, Copy, Debug)]
pub enum OneChannel {
    /// One lane exactly (0 = r .. 3 = a).
    Lane(usize),
    /// A weighted sum of rgb (luminance, weights from the primaries).
    Weights([f32; 3]),
}

/// One half-float channel called `name`, taken from an rgba f16 readback:
/// what a single-channel Render writes. Tagged like the picture (pass
/// `None` for alpha, which is coverage, not colour). Reads back as grey
/// through our own decoder, like any lone channel.
#[allow(clippy::too_many_arguments)]
pub fn write_one_f16_exr(
    path: &Path,
    w: usize,
    h: usize,
    row_stride_f16: usize,
    data: &[f16],
    pick: OneChannel,
    name: &str,
    chroma: Option<[(f32, f32); 4]>,
    compression: ExrCompression,
) -> anyhow::Result<()> {
    let layer = Layer::new(
        (w, h),
        LayerAttributes::named(name),
        encoding(compression),
        SpecificChannels::build().with_channel(name).with_pixel_fn(|p: Vec2<usize>| {
            let px = &data[p.y() * row_stride_f16 + p.x() * 4..];
            (match pick {
                OneChannel::Lane(i) => px[i.min(3)],
                OneChannel::Weights(k) => f16::from_f32(
                    k[0] * px[0].to_f32() + k[1] * px[1].to_f32() + k[2] * px[2].to_f32(),
                ),
            },)
        }),
    );
    let mut image = Image::from_layer(layer);
    tag(&mut image, chroma);
    image.write().to_file(path)?;
    Ok(())
}

/// One DATA frame: the red lane of an rgba f16 readback as a single float32
/// channel named `Z` (the depth convention, and what `decode` reads back as
/// grey), with NO colour attributes — a measurement, not a picture. What a
/// Render writes when it is fed the data pipe. The readback is half float,
/// so float32 holds every value the comp produced exactly.
pub fn write_z_f32_exr(path: &Path, w: usize, h: usize, row_stride_f16: usize, data: &[f16]) -> anyhow::Result<()> {
    write_z_f32_exr_with(path, w, h, row_stride_f16, data, ExrCompression::Rle)
}

/// `write_z_f32_exr` with the compression picked.
pub fn write_z_f32_exr_with(
    path: &Path,
    w: usize,
    h: usize,
    row_stride_f16: usize,
    data: &[f16],
    compression: ExrCompression,
) -> anyhow::Result<()> {
    let layer = Layer::new(
        (w, h),
        LayerAttributes::named("data"),
        encoding(compression),
        SpecificChannels::build()
            .with_channel("Z")
            .with_pixel_fn(|p: Vec2<usize>| (data[p.y() * row_stride_f16 + p.x() * 4].to_f32(),)),
    );
    Image::from_layer(layer).write().to_file(path)?;
    Ok(())
}

/// One MATTE frame: the red lane of an rgba f16 readback as a single half
/// channel `Y`, RLE in 64-pixel tiles — a matte is runs of 0 and 1, which
/// RLE stores in next to nothing and decodes nearly for free. Reads back as
/// grey through our own decoder (a lone channel feeds all three).
pub fn write_matte_f16_exr(path: &Path, w: usize, h: usize, row_stride_f16: usize, data: &[f16]) -> anyhow::Result<()> {
    let layer = Layer::new(
        (w, h),
        LayerAttributes::named("matte"),
        Encoding::FAST_LOSSLESS,
        SpecificChannels::build()
            .with_channel("Y")
            .with_pixel_fn(|p: Vec2<usize>| (data[p.y() * row_stride_f16 + p.x() * 4],)),
    );
    Image::from_layer(layer).write().to_file(path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: usize, h: usize) -> Vec<f16> {
        (0..w * h)
            .flat_map(|i| [i as f32 * 0.25, -1.5, 1e3 + i as f32, 0.5].map(f16::from_f32))
            .collect()
    }

    /// An IMAGE render's bytes are pinned: the data path was added beside
    /// this writer and must never have touched it.
    #[test]
    fn the_image_writer_is_byte_identical() {
        use sha2::{Digest, Sha256};
        let dir = std::env::temp_dir().join(format!("wobble-bakes-rgba-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("f.exr");
        let chroma = crate::color::Colorspace::LINEAR_REC709.primaries.chromaticities();
        write_rgba_f16_exr_tagged(&p, 7, 5, 7 * 4, &frame(7, 5), Some(chroma)).unwrap();
        let hex: String = Sha256::digest(std::fs::read(&p).unwrap()).iter().map(|b| format!("{b:02x}")).collect();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(hex, "120dbb1e6b1f59dbaf51ae8c30ea457561babbc4a0651b3002016c81a5ec3978");
    }

    /// A DATA render is one float32 channel `Z`, the red lane exactly, with
    /// no chromaticities, and reads back through our own decoder as grey.
    #[test]
    fn a_matte_frame_is_one_half_channel_that_reads_back_grey() {
        let dir = std::env::temp_dir().join(format!("wobble-bakes-m-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("m.exr");
        let (w, h) = (7usize, 5usize);
        let px = frame(w, h);
        write_matte_f16_exr(&p, w, h, w * 4, &px).unwrap();
        let img = exr::prelude::read_all_flat_layers_from_file(&p).unwrap();
        let layer = &img.layer_data[0];
        let chans: Vec<String> = layer.channel_data.list.iter().map(|c| c.name.to_string()).collect();
        assert_eq!(chans, ["Y"]);
        match &layer.channel_data.list[0].sample_data {
            exr::prelude::FlatSamples::F16(v) => {
                for (i, s) in v.iter().enumerate() {
                    assert_eq!(*s, px[i * 4], "sample {i}");
                }
            }
            other => panic!("not half: {other:?}"),
        }
        let dec = crate::decode::AnyDecoder::new(false);
        let info = crate::decode::ExrDecoder::read_header(&dec, &p).unwrap();
        let stride = info.row_stride_f16();
        let mut out = vec![f16::ZERO; stride * info.height as usize];
        crate::decode::ExrDecoder::decode_into(&dec, &p, info, stride, &mut out).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        for i in 0..w * h {
            let (y, x) = (i / w, i % w);
            let o = &out[y * stride + x * 4..];
            assert_eq!((o[0], o[1], o[2]), (px[i * 4], px[i * 4], px[i * 4]), "pixel {i}");
        }
    }

    /// Every compression pick writes a file that reads back: the lossless
    /// ones exactly, the lossy ones close; RLE is the old bytes.
    #[test]
    fn every_compression_reads_back() {
        let dir = std::env::temp_dir().join(format!("wobble-bakes-comp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (w, h) = (37usize, 21usize);
        let px: Vec<f16> = (0..w * h)
            .flat_map(|i| [(i % 7) as f32 * 0.25, 0.5, (i / w) as f32 * 0.1, 1.0].map(f16::from_f32))
            .collect();
        for c in ExrCompression::ALL {
            let p = dir.join(format!("{c:?}.exr"));
            write_rgba_f16_exr_with(&p, w, h, w * 4, &px, None, c).unwrap();
            let img = exr::prelude::read_all_flat_layers_from_file(&p).unwrap();
            assert_eq!(img.layer_data[0].encoding.compression, encoding(c).compression, "{c:?}");
            let dec = crate::decode::AnyDecoder::new(false);
            let info = crate::decode::ExrDecoder::read_header(&dec, &p).unwrap();
            let stride = info.row_stride_f16();
            let mut out = vec![f16::ZERO; stride * h];
            crate::decode::ExrDecoder::decode_into(&dec, &p, info, stride, &mut out).unwrap();
            // Lossless: every sample exact. Lossy (B44 on a picture with an
            // edge every pixel): the frame on average close.
            let mut err = 0.0f32;
            for y in 0..h {
                for x in 0..w {
                    for l in 0..4 {
                        let (a, b) = (px[(y * w + x) * 4 + l].to_f32(), out[y * stride + x * 4 + l].to_f32());
                        if c.lossy(false) {
                            err += (a - b).abs();
                        } else {
                            assert_eq!(a, b, "{c:?} {x},{y}.{l}");
                        }
                    }
                }
            }
            let mean = err / (w * h * 4) as f32;
            assert!(mean < 0.1, "{c:?} mean error {mean}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A single-channel frame is ONE half channel with the given name: a
    /// lane exactly, or the weighted luma.
    #[test]
    fn a_single_channel_frame_is_one_named_channel() {
        let dir = std::env::temp_dir().join(format!("wobble-bakes-one-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (w, h) = (7usize, 5usize);
        let px = frame(w, h);
        let read = |p: &Path| {
            let img = exr::prelude::read_all_flat_layers_from_file(p).unwrap();
            let layer = &img.layer_data[0];
            let names: Vec<String> = layer.channel_data.list.iter().map(|c| c.name.to_string()).collect();
            let v = match &layer.channel_data.list[0].sample_data {
                exr::prelude::FlatSamples::F16(v) => v.clone(),
                other => panic!("not half: {other:?}"),
            };
            (names, v, img.attributes.chromaticities.is_some())
        };
        let p = dir.join("a.exr");
        write_one_f16_exr(&p, w, h, w * 4, &px, OneChannel::Lane(3), "A", None, ExrCompression::Zip16).unwrap();
        let (names, v, tagged) = read(&p);
        assert_eq!(names, ["A"]);
        assert!(!tagged);
        assert!(v.iter().enumerate().all(|(i, s)| *s == px[i * 4 + 3]));
        let p = dir.join("y.exr");
        let k = [0.25, 0.5, 0.25];
        let chroma = crate::color::Colorspace::LINEAR_REC709.primaries.chromaticities();
        write_one_f16_exr(&p, w, h, w * 4, &px, OneChannel::Weights(k), "matte", Some(chroma), ExrCompression::Rle)
            .unwrap();
        let (names, v, tagged) = read(&p);
        assert_eq!(names, ["matte"]);
        assert!(tagged);
        for (i, s) in v.iter().enumerate() {
            let q = &px[i * 4..];
            let want = f16::from_f32(k[0] * q[0].to_f32() + k[1] * q[1].to_f32() + k[2] * q[2].to_f32());
            assert_eq!(*s, want, "pixel {i}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_data_frame_is_one_exact_float_channel() {
        let dir = std::env::temp_dir().join(format!("wobble-bakes-z-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("z.exr");
        let (w, h) = (7usize, 5usize);
        let px = frame(w, h);
        write_z_f32_exr(&p, w, h, w * 4, &px).unwrap();
        let img = exr::prelude::read_all_flat_layers_from_file(&p).unwrap();
        let layer = &img.layer_data[0];
        let chans: Vec<String> = layer.channel_data.list.iter().map(|c| c.name.to_string()).collect();
        assert_eq!(chans, ["Z"]);
        assert!(img.attributes.chromaticities.is_none(), "data carries no colour space");
        match &layer.channel_data.list[0].sample_data {
            exr::prelude::FlatSamples::F32(v) => {
                for (i, s) in v.iter().enumerate() {
                    assert_eq!(*s, px[i * 4].to_f32(), "sample {i}");
                }
            }
            other => panic!("not float32: {other:?}"),
        }
        // Through the app's decoder: grey, alpha 1.
        let dec = crate::decode::AnyDecoder::new(false);
        let info = crate::decode::ExrDecoder::read_header(&dec, &p).unwrap();
        let stride = info.row_stride_f16();
        let mut out = vec![f16::ZERO; stride * info.height as usize];
        crate::decode::ExrDecoder::decode_into(&dec, &p, info, stride, &mut out).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let q = &out[3 * stride + 2 * 4..][..4];
        let want = px[(3 * w + 2) * 4];
        assert_eq!([q[0], q[1], q[2], q[3]], [want, want, want, f16::ONE]);
    }
}
