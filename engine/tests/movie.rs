//! Movie decode end to end: encode a clip with the same FFmpeg we link,
//! then read frames back out of it in and out of order.
//!
//! The clip is a ramp of flat greys — frame N is a different, brighter
//! constant than frame N-1 — because that turns the two failures that
//! actually happen into assertions:
//!
//! - an **off-by-one seek** shows up as a decoded value belonging to a
//!   neighbouring frame, which a ramp makes visible and a moving picture
//!   would hide;
//! - a **broken random access** shows up as scrambled reads disagreeing
//!   with sequential ones.
//!
//! Exact code values are deliberately NOT asserted: ProRes is 10-bit
//! limited-range 4:2:2 and H.264 is lossy, so the value is approximate by
//! construction. The ORDER and the AGREEMENT are what must be exact.

#![cfg(feature = "movie")]

use engine::decode::ExrDecoder;
use engine::movie::MovieDecoder;
use ffmpeg_next as ff;
use half::f16;
use std::path::{Path, PathBuf};

const W: u32 = 64;
const H: u32 = 64;
const FRAMES: u64 = 24;

/// Luma code value for frame `n`, as an 8-bit level. Spread across most of
/// the legal range so 10-bit quantisation and lossy coding cannot make two
/// neighbouring frames collide.
fn level_of(n: u64) -> u8 {
    (24 + n * 8) as u8
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("wobble-movie-test");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// Encode `FRAMES` flat-grey frames with `codec_name` into `path`.
/// Returns false when this build of FFmpeg has no such encoder.
fn encode(path: &Path, codec_name: &str, pixel: ff::format::Pixel) -> bool {
    ff::init().unwrap();
    let Some(codec) = ff::encoder::find_by_name(codec_name) else {
        return false;
    };
    let mut octx = ff::format::output(&path).unwrap();
    let global_header = octx
        .format()
        .flags()
        .contains(ff::format::Flags::GLOBAL_HEADER);

    let tb = ff::Rational::new(1, 24);
    let mut enc = ff::codec::context::Context::new_with_codec(codec)
        .encoder()
        .video()
        .unwrap();
    enc.set_width(W);
    enc.set_height(H);
    enc.set_format(pixel);
    enc.set_time_base(tb);
    // A short GOP so the test exercises seeking to a keyframe AND decoding
    // forward from it; one long GOP would only ever test the latter.
    enc.set_gop(8);
    if global_header {
        enc.set_flags(ff::codec::Flags::GLOBAL_HEADER);
    }
    let mut enc = enc.open_as(codec).unwrap();

    let mut stream = octx.add_stream(codec).unwrap();
    stream.set_parameters(&enc);
    stream.set_time_base(tb);
    let stream_index = stream.index();
    octx.write_header().unwrap();
    // AFTER write_header: the muxer picks its own timescale (QuickTime and
    // MP4 both do), and rescaling to the value we asked for instead of the
    // one it chose puts every packet at the wrong timestamp.
    let stream_tb = octx.stream(stream_index).unwrap().time_base();

    // 10-bit planes hold little-endian u16 samples; 8-bit ones hold bytes.
    let ten_bit = pixel == ff::format::Pixel::YUV422P10LE;
    for n in 0..FRAMES {
        // A FRESH frame each time: the encoder is multithreaded and keeps a
        // reference to what it was sent, so filling one buffer over and
        // over rewrites pictures that have not been encoded yet.
        let mut frame = ff::frame::Video::new(pixel, W, H);
        let y = level_of(n);
        // Y is the ramp; the chroma planes sit at neutral so every frame is
        // a flat grey rather than a colour.
        for (plane, neutral) in [(0usize, false), (1, true), (2, true)] {
            let stride = frame.stride(plane);
            let planes_h = if plane == 0 { H as usize } else { chroma_h(pixel) };
            let planes_w = if plane == 0 { W as usize } else { chroma_w(pixel) };
            let data = frame.data_mut(plane);
            let value: u16 = match (ten_bit, neutral) {
                (true, true) => 512,
                (true, false) => (y as u16) << 2,
                (false, true) => 128,
                (false, false) => y as u16,
            };
            for row in 0..planes_h {
                let base = row * stride;
                for x in 0..planes_w {
                    if ten_bit {
                        data[base + x * 2..base + x * 2 + 2]
                            .copy_from_slice(&value.to_le_bytes());
                    } else {
                        data[base + x] = value as u8;
                    }
                }
            }
        }
        frame.set_pts(Some(n as i64));
        enc.send_frame(&frame).unwrap();
        drain(&mut enc, &mut octx, stream_index, tb, stream_tb);
    }
    enc.send_eof().unwrap();
    drain(&mut enc, &mut octx, stream_index, tb, stream_tb);
    octx.write_trailer().unwrap();
    true
}

fn chroma_w(pixel: ff::format::Pixel) -> usize {
    match pixel {
        ff::format::Pixel::YUV420P | ff::format::Pixel::YUV422P10LE => W as usize / 2,
        _ => W as usize,
    }
}

fn chroma_h(pixel: ff::format::Pixel) -> usize {
    match pixel {
        ff::format::Pixel::YUV420P => H as usize / 2,
        _ => H as usize,
    }
}

fn drain(
    enc: &mut ff::codec::encoder::Video,
    octx: &mut ff::format::context::Output,
    stream_index: usize,
    enc_tb: ff::Rational,
    stream_tb: ff::Rational,
) {
    let mut packet = ff::Packet::empty();
    while enc.receive_packet(&mut packet).is_ok() {
        packet.set_stream(stream_index);
        packet.rescale_ts(enc_tb, stream_tb);
        packet.write_interleaved(octx).unwrap();
    }
}

/// Decode one frame and return its average red value (the clip is grey, so
/// any channel is the picture).
fn grey_at(dec: &MovieDecoder, path: &Path, frame: u64) -> f32 {
    let info = dec.read_header(path).unwrap();
    let stride = info.row_stride_f16();
    let mut buf = vec![f16::ZERO; stride * info.height as usize];
    dec.decode_frame_into(path, info, stride, None, frame, &mut buf)
        .unwrap_or_else(|e| panic!("frame {frame}: {e:#}"));
    // Average the middle rows: 4:2:0 chroma and block edges make the
    // outermost pixels the least representative, and we are measuring a
    // constant.
    let mut sum = 0f32;
    let mut n = 0f32;
    for y in H as usize / 4..H as usize * 3 / 4 {
        for x in W as usize / 4..W as usize * 3 / 4 {
            sum += buf[y * stride + x * 4].to_f32();
            n += 1.0;
        }
    }
    sum / n
}

/// The whole contract for one codec, on one encoded clip.
fn round_trip(path: &Path) {
    let dec = MovieDecoder::default();

    // A movie's length comes from the container, not from a directory.
    assert_eq!(
        dec.frame_count(path).unwrap(),
        FRAMES,
        "{}: wrong frame count",
        path.display()
    );
    let info = dec.read_header(path).unwrap();
    assert_eq!((info.width, info.height), (W, H));

    // Sequential: this is the warm path the player takes, and it is what
    // every other read is checked against.
    let seq: Vec<f32> = (0..FRAMES).map(|n| grey_at(&dec, path, n)).collect();

    // The ramp must be strictly increasing. A seek that lands one frame off
    // shows up here as a flat step or an inversion.
    for n in 1..FRAMES as usize {
        assert!(
            seq[n] > seq[n - 1],
            "{}: frame {n} ({}) is not brighter than {} ({}) — frames are \
             mis-indexed",
            path.display(),
            seq[n],
            n - 1,
            seq[n - 1]
        );
    }

    // Scrambled access must agree with sequential access exactly: same
    // frame, same pixels, whatever order it was asked for. Backwards and
    // long jumps both force a real seek rather than a catch-up.
    for n in [FRAMES - 1, 0, 13, 1, 22, 7, FRAMES - 2, 3] {
        let got = grey_at(&dec, path, n);
        assert!(
            (got - seq[n as usize]).abs() < 1e-4,
            "{}: frame {n} read {got} out of order but {} in order",
            path.display(),
            seq[n as usize]
        );
    }

    // Decoded greys are neutral and opaque, and land near the code value
    // that was encoded — loosely, because the codecs are lossy and the
    // transfer curve is deliberately NOT applied here.
    let stride = info.row_stride_f16();
    let mut buf = vec![f16::ZERO; stride * H as usize];
    dec.decode_frame_into(path, info, stride, None, 12, &mut buf).unwrap();
    let mid = (H as usize / 2) * stride + (W as usize / 2) * 4;
    let (r, g, b, a) = (
        buf[mid].to_f32(),
        buf[mid + 1].to_f32(),
        buf[mid + 2].to_f32(),
        buf[mid + 3].to_f32(),
    );
    assert!((r - g).abs() < 0.02 && (g - b).abs() < 0.02, "not neutral: {r} {g} {b}");
    assert_eq!(a, 1.0, "a clip without alpha must decode opaque");
    // Limited-range luma L maps to roughly (L - 16) / 219 once swscale has
    // expanded it; 0.15 of slack covers the matrix, the chroma and the
    // codec.
    let want = (level_of(12) as f32 - 16.0) / 219.0;
    assert!((r - want).abs() < 0.15, "frame 12 read {r}, expected about {want}");
}

#[test]
fn prores_round_trips_frame_accurately() {
    let path = tmp("ramp_prores.mov");
    if !encode(&path, "prores_ks", ff::format::Pixel::YUV422P10LE) {
        eprintln!("SKIP: this FFmpeg has no prores_ks encoder");
        return;
    }
    round_trip(&path);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn h264_round_trips_frame_accurately() {
    let path = tmp("ramp_h264.mp4");
    let encoded = ["libx264", "h264_videotoolbox", "h264"]
        .iter()
        .any(|c| encode(&path, c, ff::format::Pixel::YUV420P));
    if !encoded {
        eprintln!("SKIP: this FFmpeg has no H.264 encoder");
        return;
    }
    round_trip(&path);
    let _ = std::fs::remove_file(&path);
}

/// The pool is shared across the player's rayon threads, so several
/// decodes of one file have to be able to run at once without handing two
/// jobs the same session.
#[test]
fn concurrent_decodes_of_one_file_agree() {
    let path = tmp("ramp_concurrent.mov");
    if !encode(&path, "prores_ks", ff::format::Pixel::YUV422P10LE) {
        eprintln!("SKIP: this FFmpeg has no prores_ks encoder");
        return;
    }
    let dec = std::sync::Arc::new(MovieDecoder::default());
    let want: Vec<f32> = (0..FRAMES).map(|n| grey_at(&dec, &path, n)).collect();

    let mut handles = Vec::new();
    for t in 0..4u64 {
        let dec = dec.clone();
        let path = path.clone();
        handles.push(std::thread::spawn(move || {
            // Each thread walks the clip from a different offset, so the
            // pool is asked for overlapping-but-different frames.
            (0..FRAMES)
                .map(|i| {
                    let n = (i + t * 5) % FRAMES;
                    (n, grey_at(&dec, &path, n))
                })
                .collect::<Vec<_>>()
        }));
    }
    for h in handles {
        for (n, got) in h.join().unwrap() {
            assert!(
                (got - want[n as usize]).abs() < 1e-4,
                "frame {n} read {got} concurrently but {} alone",
                want[n as usize]
            );
        }
    }
    let _ = std::fs::remove_file(&path);
}

/// `engine::movie::encode_mp4` — what a GenAI clip step uploads — writes a
/// clip this decoder reads back, frame for frame, in order, at its rate.
#[test]
fn encode_mp4_round_trips() {
    if !engine::movie::can_encode_mp4() {
        eprintln!("skipped: this FFmpeg has no video encoder");
        return;
    }
    let path = tmp("genai_clip.mp4");
    let frames: Vec<Vec<u8>> =
        (0..FRAMES).map(|n| vec![level_of(n); (W * H * 3) as usize]).collect();
    engine::movie::encode_mp4(&path, &frames, W, H, 24.0).unwrap();
    let dec = MovieDecoder::default();
    assert_eq!(dec.frame_count(&path).unwrap(), FRAMES);
    let info = engine::movie::probe(&path).unwrap();
    assert_eq!((info.fps_num, info.fps_den), (24, 1));
    let reads: Vec<f32> = (0..FRAMES).map(|n| grey_at(&dec, &path, n)).collect();
    for w in reads.windows(2) {
        assert!(w[1] > w[0], "frames out of order: {reads:?}");
    }
    assert!(engine::movie::encode_mp4(&path, &frames, W + 1, H, 24.0).is_err(), "odd sizes refused");
}
