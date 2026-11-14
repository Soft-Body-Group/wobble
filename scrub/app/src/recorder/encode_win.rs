//! H.264 .mp4 through Media Foundation's sink writer: the H.264 encoder
//! Windows ships (hardware where the driver offers one), nothing GPL and
//! no FFmpeg. Frames go in as RGB32 at their own times.
//!
//! Not run on this project's Mac: compile-checked for
//! `x86_64-pc-windows-msvc`; the first Windows run is its real test. An
//! "N" edition of Windows without the Media Feature Pack has no H.264
//! encoder, and the error says so.

use std::mem::ManuallyDrop;
use std::path::Path;

use windows::core::{HSTRING, PCWSTR};
use windows::Win32::Media::MediaFoundation::*;
use windows::Win32::System::Com::{CoInitializeEx, CoUninitialize, COINIT_MULTITHREADED};

use super::{VideoSink, FPS};

pub struct WinSink {
    /// Released in `drop` BEFORE `MFShutdown`: a field drops after the
    /// struct's own `Drop`, and releasing the writer once Media Foundation
    /// is shut down is an access violation.
    writer: ManuallyDrop<IMFSinkWriter>,
    stream: u32,
    w: u32,
    h: u32,
    last: Option<(Vec<u8>, i64)>,
}

/// 100 ns units.
fn hns(t: f64) -> i64 {
    (t * 1e7).round() as i64
}

fn e(what: &str, err: windows::core::Error) -> String {
    format!("{what}: {err} — Windows may be missing its H.264 encoder (the Media Feature Pack)")
}

pub fn open(path: &Path, w: u32, h: u32, bitrate: u64) -> Result<Box<dyn VideoSink>, String> {
    let _ = std::fs::remove_file(path);
    unsafe {
        // The save thread is ours alone; S_FALSE (already initialised) is fine.
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        MFStartup(MF_VERSION, MFSTARTUP_FULL).map_err(|x| e("starting Media Foundation", x))?;
        let url = HSTRING::from(path.as_os_str());
        let writer = MFCreateSinkWriterFromURL(PCWSTR(url.as_ptr()), None, None)
            .map_err(|x| e("creating the movie", x))?;

        let out = MFCreateMediaType().map_err(|x| e("media type", x))?;
        out.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video).map_err(|x| e("media type", x))?;
        out.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_H264).map_err(|x| e("media type", x))?;
        out.SetUINT32(&MF_MT_AVG_BITRATE, bitrate.min(u32::MAX as u64) as u32).map_err(|x| e("media type", x))?;
        out.SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32).map_err(|x| e("media type", x))?;
        out.SetUINT64(&MF_MT_FRAME_SIZE, ((w as u64) << 32) | h as u64).map_err(|x| e("media type", x))?;
        out.SetUINT64(&MF_MT_FRAME_RATE, ((FPS as u64) << 32) | 1).map_err(|x| e("media type", x))?;
        out.SetUINT64(&MF_MT_PIXEL_ASPECT_RATIO, (1u64 << 32) | 1).map_err(|x| e("media type", x))?;
        let stream = writer.AddStream(&out).map_err(|x| e("adding the video stream", x))?;

        let input = MFCreateMediaType().map_err(|x| e("media type", x))?;
        input.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video).map_err(|x| e("media type", x))?;
        input.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_RGB32).map_err(|x| e("media type", x))?;
        input.SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32).map_err(|x| e("media type", x))?;
        input.SetUINT64(&MF_MT_FRAME_SIZE, ((w as u64) << 32) | h as u64).map_err(|x| e("media type", x))?;
        input.SetUINT64(&MF_MT_FRAME_RATE, ((FPS as u64) << 32) | 1).map_err(|x| e("media type", x))?;
        input.SetUINT64(&MF_MT_PIXEL_ASPECT_RATIO, (1u64 << 32) | 1).map_err(|x| e("media type", x))?;
        // Top-down rows (RGB32's default stride is bottom-up).
        input.SetUINT32(&MF_MT_DEFAULT_STRIDE, w * 4).map_err(|x| e("media type", x))?;
        writer
            .SetInputMediaType(stream, &input, None)
            .map_err(|x| e("the H.264 encoder refused the input", x))?;
        writer.BeginWriting().map_err(|x| e("starting the movie", x))?;
        Ok(Box::new(WinSink { writer: ManuallyDrop::new(writer), stream, w, h, last: None }))
    }
}

impl WinSink {
    fn write(&mut self, bgra: &[u8], t: i64, dur: i64) -> Result<(), String> {
        unsafe {
            let len = bgra.len() as u32;
            let buf = MFCreateMemoryBuffer(len).map_err(|x| e("frame buffer", x))?;
            let mut ptr: *mut u8 = std::ptr::null_mut();
            buf.Lock(&mut ptr, None, None).map_err(|x| e("frame buffer", x))?;
            std::ptr::copy_nonoverlapping(bgra.as_ptr(), ptr, bgra.len());
            buf.Unlock().map_err(|x| e("frame buffer", x))?;
            buf.SetCurrentLength(len).map_err(|x| e("frame buffer", x))?;
            let sample = MFCreateSample().map_err(|x| e("frame", x))?;
            sample.AddBuffer(&buf).map_err(|x| e("frame", x))?;
            sample.SetSampleTime(t).map_err(|x| e("frame", x))?;
            sample.SetSampleDuration(dur.max(1)).map_err(|x| e("frame", x))?;
            self.writer.WriteSample(self.stream, &sample).map_err(|x| e("encoding a frame", x))
        }
    }
}

impl VideoSink for WinSink {
    fn push(&mut self, rgba: &[u8], t: f64) -> Result<(), String> {
        let mut bgra = Vec::with_capacity((self.w * self.h * 4) as usize);
        for p in rgba.chunks_exact(4) {
            bgra.extend_from_slice(&[p[2], p[1], p[0], 255]);
        }
        // A frame's duration is the gap to the next, so each is written
        // one late.
        let t = hns(t);
        if let Some((px, t0)) = self.last.take() {
            self.write(&px, t0, t - t0)?;
        }
        self.last = Some((bgra, t));
        Ok(())
    }

    fn finish(mut self: Box<Self>, end: f64) -> Result<(), String> {
        if let Some((px, t0)) = self.last.take() {
            self.write(&px, t0, hns(end) - t0)?;
        }
        unsafe { self.writer.Finalize().map_err(|x| e("finishing the movie", x)) }
    }
}

impl Drop for WinSink {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.writer);
            let _ = MFShutdown();
            CoUninitialize();
        }
    }
}
