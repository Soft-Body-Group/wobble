//! H.264 .mp4 through AVFoundation (`AVAssetWriter`): Apple's encoder,
//! hardware where there is one, nothing GPL shipped. Frames are BGRA
//! `CVPixelBuffer`s at their own presentation times, tagged Rec.709.

use std::ffi::{c_void, CString};
use std::path::Path;

use objc2::encode::{Encode, Encoding, RefEncode};
use objc2::rc::autoreleasepool;
use objc2::runtime::{AnyClass, AnyObject, Bool};
use objc2::msg_send;

use super::VideoSink;

#[repr(C)]
#[derive(Clone, Copy)]
struct CMTime {
    value: i64,
    timescale: i32,
    flags: u32,
    epoch: i64,
}

unsafe impl Encode for CMTime {
    const ENCODING: Encoding =
        Encoding::Struct("?", &[i64::ENCODING, i32::ENCODING, u32::ENCODING, i64::ENCODING]);
}

/// Microseconds; flags = valid.
fn cm_time(t: f64) -> CMTime {
    CMTime { value: (t * 1e6).round() as i64, timescale: 1_000_000, flags: 1, epoch: 0 }
}

#[repr(C)]
struct CVBuffer {
    _p: [u8; 0],
}
unsafe impl RefEncode for CVBuffer {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Encoding::Struct("__CVBuffer", &[]));
}

#[link(name = "AVFoundation", kind = "framework")]
unsafe extern "C" {
    static AVFileTypeMPEG4: *mut AnyObject;
    static AVMediaTypeVideo: *mut AnyObject;
    static AVVideoCodecKey: *mut AnyObject;
    static AVVideoCodecTypeH264: *mut AnyObject;
    static AVVideoWidthKey: *mut AnyObject;
    static AVVideoHeightKey: *mut AnyObject;
    static AVVideoCompressionPropertiesKey: *mut AnyObject;
    static AVVideoAverageBitRateKey: *mut AnyObject;
    static AVVideoMaxKeyFrameIntervalKey: *mut AnyObject;
    static AVVideoProfileLevelKey: *mut AnyObject;
    static AVVideoProfileLevelH264HighAutoLevel: *mut AnyObject;
    static AVVideoColorPropertiesKey: *mut AnyObject;
    static AVVideoColorPrimariesKey: *mut AnyObject;
    static AVVideoColorPrimaries_ITU_R_709_2: *mut AnyObject;
    static AVVideoTransferFunctionKey: *mut AnyObject;
    static AVVideoTransferFunction_ITU_R_709_2: *mut AnyObject;
    static AVVideoYCbCrMatrixKey: *mut AnyObject;
    static AVVideoYCbCrMatrix_ITU_R_709_2: *mut AnyObject;
}

#[link(name = "CoreVideo", kind = "framework")]
unsafe extern "C" {
    fn CVPixelBufferCreate(
        alloc: *const c_void,
        w: usize,
        h: usize,
        fmt: u32,
        attrs: *const c_void,
        out: *mut *mut CVBuffer,
    ) -> i32;
    fn CVPixelBufferLockBaseAddress(pb: *mut CVBuffer, flags: u64) -> i32;
    fn CVPixelBufferUnlockBaseAddress(pb: *mut CVBuffer, flags: u64) -> i32;
    fn CVPixelBufferGetBaseAddress(pb: *mut CVBuffer) -> *mut c_void;
    fn CVPixelBufferGetBytesPerRow(pb: *mut CVBuffer) -> usize;
}

#[link(name = "CoreMedia", kind = "framework")]
unsafe extern "C" {}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFRelease(p: *const c_void);
}

const BGRA: u32 = u32::from_be_bytes(*b"BGRA");
const STATUS_WRITING: isize = 1;
const STATUS_COMPLETED: isize = 2;

pub struct MacSink {
    writer: *mut AnyObject,
    input: *mut AnyObject,
    adaptor: *mut AnyObject,
    w: u32,
    h: u32,
    last_t: f64,
}

// The objects are only touched from the one save thread that made them.
unsafe impl Send for MacSink {}

fn class(name: &std::ffi::CStr) -> Result<&'static AnyClass, String> {
    AnyClass::get(name).ok_or_else(|| format!("{} missing", name.to_string_lossy()))
}

unsafe fn number(v: i64) -> *mut AnyObject {
    let cls = AnyClass::get(c"NSNumber").expect("NSNumber");
    unsafe { msg_send![cls, numberWithLongLong: v] }
}

unsafe fn dict(pairs: &[(*mut AnyObject, *mut AnyObject)]) -> *mut AnyObject {
    let keys: Vec<*const AnyObject> = pairs.iter().map(|p| p.0 as *const AnyObject).collect();
    let vals: Vec<*const AnyObject> = pairs.iter().map(|p| p.1 as *const AnyObject).collect();
    let cls = AnyClass::get(c"NSDictionary").expect("NSDictionary");
    unsafe { msg_send![cls, dictionaryWithObjects: vals.as_ptr(), forKeys: keys.as_ptr(), count: pairs.len()] }
}

unsafe fn describe(err: *mut AnyObject) -> String {
    if err.is_null() {
        return "unknown error".into();
    }
    unsafe {
        let s: *mut AnyObject = msg_send![err, localizedDescription];
        if s.is_null() {
            return "unknown error".into();
        }
        let p: *const std::ffi::c_char = msg_send![s, UTF8String];
        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

pub fn open(path: &Path, w: u32, h: u32, bitrate: u64) -> Result<Box<dyn VideoSink>, String> {
    // AVAssetWriter refuses to overwrite.
    let _ = std::fs::remove_file(path);
    let cpath = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
    autoreleasepool(|_| unsafe {
        let s: *mut AnyObject = msg_send![class(c"NSString")?, stringWithUTF8String: cpath.as_ptr()];
        let url: *mut AnyObject = msg_send![class(c"NSURL")?, fileURLWithPath: s];
        let mut err: *mut AnyObject = std::ptr::null_mut();
        let writer: *mut AnyObject =
            msg_send![class(c"AVAssetWriter")?, assetWriterWithURL: url, fileType: AVFileTypeMPEG4, error: &mut err];
        if writer.is_null() {
            return Err(format!("AVAssetWriter: {}", describe(err)));
        }
        let compression = dict(&[
            (AVVideoAverageBitRateKey, number(bitrate as i64)),
            (AVVideoMaxKeyFrameIntervalKey, number(60)),
            (AVVideoProfileLevelKey, AVVideoProfileLevelH264HighAutoLevel),
        ]);
        let colour = dict(&[
            (AVVideoColorPrimariesKey, AVVideoColorPrimaries_ITU_R_709_2),
            (AVVideoTransferFunctionKey, AVVideoTransferFunction_ITU_R_709_2),
            (AVVideoYCbCrMatrixKey, AVVideoYCbCrMatrix_ITU_R_709_2),
        ]);
        let settings = dict(&[
            (AVVideoCodecKey, AVVideoCodecTypeH264),
            (AVVideoWidthKey, number(w as i64)),
            (AVVideoHeightKey, number(h as i64)),
            (AVVideoCompressionPropertiesKey, compression),
            (AVVideoColorPropertiesKey, colour),
        ]);
        let input: *mut AnyObject = msg_send![
            class(c"AVAssetWriterInput")?,
            assetWriterInputWithMediaType: AVMediaTypeVideo,
            outputSettings: settings
        ];
        if input.is_null() {
            return Err("AVAssetWriterInput refused the settings".into());
        }
        let _: () = msg_send![input, setExpectsMediaDataInRealTime: Bool::NO];
        let adaptor: *mut AnyObject = msg_send![
            class(c"AVAssetWriterInputPixelBufferAdaptor")?,
            assetWriterInputPixelBufferAdaptorWithAssetWriterInput: input,
            sourcePixelBufferAttributes: std::ptr::null_mut::<AnyObject>()
        ];
        let ok: Bool = msg_send![writer, canAddInput: input];
        if !ok.as_bool() || adaptor.is_null() {
            return Err("AVAssetWriter cannot take an H.264 input".into());
        }
        let _: () = msg_send![writer, addInput: input];
        let ok: Bool = msg_send![writer, startWriting];
        if !ok.as_bool() {
            let e: *mut AnyObject = msg_send![writer, error];
            return Err(format!("AVAssetWriter: {}", describe(e)));
        }
        let _: () = msg_send![writer, startSessionAtSourceTime: cm_time(0.0)];
        // Outlive the pool.
        let writer: *mut AnyObject = msg_send![writer, retain];
        let input: *mut AnyObject = msg_send![input, retain];
        let adaptor: *mut AnyObject = msg_send![adaptor, retain];
        Ok(Box::new(MacSink { writer, input, adaptor, w, h, last_t: -1.0 }) as Box<dyn VideoSink>)
    })
}

impl MacSink {
    fn status(&self) -> isize {
        unsafe { msg_send![self.writer, status] }
    }

    fn fail(&self, what: &str) -> String {
        let e: *mut AnyObject = unsafe { msg_send![self.writer, error] };
        format!("{what}: {}", unsafe { describe(e) })
    }
}

impl VideoSink for MacSink {
    fn push(&mut self, rgba: &[u8], t: f64) -> Result<(), String> {
        // Times must rise strictly.
        let t = if t <= self.last_t { self.last_t + 1e-3 } else { t };
        self.last_t = t;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let ready: Bool = unsafe { msg_send![self.input, isReadyForMoreMediaData] };
            if ready.as_bool() {
                break;
            }
            if self.status() != STATUS_WRITING {
                return Err(self.fail("the encoder stopped"));
            }
            if std::time::Instant::now() > deadline {
                return Err("the encoder stopped taking frames".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        autoreleasepool(|_| unsafe {
            let mut pb: *mut CVBuffer = std::ptr::null_mut();
            let r = CVPixelBufferCreate(std::ptr::null(), self.w as usize, self.h as usize, BGRA, std::ptr::null(), &mut pb);
            if r != 0 || pb.is_null() {
                return Err(format!("CVPixelBufferCreate failed ({r})"));
            }
            CVPixelBufferLockBaseAddress(pb, 0);
            let base = CVPixelBufferGetBaseAddress(pb) as *mut u8;
            let stride = CVPixelBufferGetBytesPerRow(pb);
            let row = self.w as usize * 4;
            for y in 0..self.h as usize {
                let src = &rgba[y * row..(y + 1) * row];
                let dst = std::slice::from_raw_parts_mut(base.add(y * stride), row);
                for (d, s) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
                    d[0] = s[2];
                    d[1] = s[1];
                    d[2] = s[0];
                    d[3] = 255;
                }
            }
            CVPixelBufferUnlockBaseAddress(pb, 0);
            let ok: Bool = msg_send![self.adaptor, appendPixelBuffer: pb, withPresentationTime: cm_time(t)];
            CFRelease(pb as *const c_void);
            if ok.as_bool() { Ok(()) } else { Err(self.fail("appending a frame")) }
        })
    }

    fn finish(self: Box<Self>, end: f64) -> Result<(), String> {
        let end = end.max(self.last_t + 1e-3);
        unsafe {
            let _: () = msg_send![self.input, markAsFinished];
            let _: () = msg_send![self.writer, endSessionAtSourceTime: cm_time(end)];
            let block = block2::RcBlock::new(|| {});
            let _: () = msg_send![self.writer, finishWritingWithCompletionHandler: &*block];
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while self.status() == STATUS_WRITING && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let st = self.status();
        if st == STATUS_COMPLETED { Ok(()) } else { Err(self.fail("finishing the movie")) }
    }
}

impl Drop for MacSink {
    fn drop(&mut self) {
        unsafe {
            if self.status() == STATUS_WRITING {
                let _: () = msg_send![self.writer, cancelWriting];
            }
            let _: () = msg_send![self.adaptor, release];
            let _: () = msg_send![self.input, release];
            let _: () = msg_send![self.writer, release];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real encoder, end to end: a moving bar, read back by size and
    /// by the container's own magic.
    #[test]
    fn encodes_an_mp4() {
        let (w, h) = (320u32, 180u32);
        let path = std::env::temp_dir().join(format!("wobble-rec-mac-{}.mp4", std::process::id()));
        let mut sink = open(&path, w, h, 2_000_000).expect("open");
        for i in 0..30 {
            let mut px = vec![30u8; (w * h * 4) as usize];
            for y in 0..h {
                for x in (i * 8)..(i * 8 + 20).min(w) {
                    let o = ((y * w + x) * 4) as usize;
                    px[o..o + 4].copy_from_slice(&[250, 200, 40, 255]);
                }
            }
            sink.push(&px, i as f64 / 30.0).expect("push");
        }
        sink.finish(1.0).expect("finish");
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(bytes.len() > 2000, "{} bytes", bytes.len());
        assert_eq!(&bytes[4..8], b"ftyp");
    }
}
