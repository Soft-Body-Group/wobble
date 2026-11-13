//! The pixel ring between wobble and Blender: a shared mapping both
//! processes open, so a frame crosses as one memcpy — or, where the
//! mapping can be handed to the GPU (`gpuimport`), as no CPU copy at all —
//! and never touches an encoder or a disk. On Unix the mapping is POSIX
//! shared memory (`shm_open`): pages with no file behind them, so the
//! kernel can never write them back to the SSD. Windows keeps a temp file
//! mapping ([`Source::File`]); its pages live in the page cache and are
//! written back only under memory pressure.
//!
//! Layout — little-endian, fixed, mirrored by `wobble_blender.py`:
//!
//! ```text
//! 0      magic  b"WOBSHM02"
//! 8      u32    version (2)
//! 12     u32    slot count
//! 16     u64    bytes per slot (a multiple of PAGE, so every slot starts
//!               on a page boundary — what a GPU import of the mapping needs)
//! 64     slot descriptors, DESC_BYTES each:
//!          +0  u64 seq     seqlock: odd while the writer is inside the slot
//!          +8  i64 frame   the BLENDER frame the pixels were drawn at
//!          +16 u32 w, +20 u32 h
//!          +24 u32 fmt     FMT_RGBA16F | FMT_RGBA32F
//!          +28 u32 flags   FLAG_FINAL | FLAG_SETTLED
//!          +32 u64 scene   the writer's scene version when it drew
//!          +40 f32 ms      what the draw + readback cost Blender
//!          +44 u32 samples viewport samples the draw used
//!          +48 u32 stride  bytes per row (>= w * bpp, a multiple of 256:
//!                          a buffer-to-texture copy's row alignment)
//!          +52 u32 pass    which picture: 0 = combined, else a `Pass` id
//! HEADER_BYTES  slot 0 pixels, then slot 1, ...  (rows BOTTOM-UP as the
//!               GPU reads them back; the shader flips)
//! ```
//!
//! The writer picks a free slot, writes under the seqlock and ANNOUNCES
//! the frame on the socket; the reader either copies the slot out at once
//! (CPU path) or points the GPU at it (import path) and ACKS the slot when
//! it is done with it. A slot lapped mid-copy is detected by its sequence
//! and dropped rather than shown torn.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{fence, Ordering};

/// Where a ring's pages live: named POSIX shared memory (the name without
/// its leading slash, as Python's `SharedMemory` spells it; wobble adds
/// the slash) with the mapping's length, or a plain file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    Posix { name: String, bytes: usize },
    File(PathBuf),
}

impl Source {
    /// What the socket names it: the parser's inverse.
    pub fn json_fields(&self) -> String {
        match self {
            Source::Posix { name, bytes } => {
                format!("\"name\":{},\"bytes\":{bytes}", super::live::json_str(name))
            }
            Source::File(p) => format!("\"path\":{}", super::live::json_str(&p.to_string_lossy())),
        }
    }

    /// Open (or, with `create` = the length, create) the mapping
    /// read-write. Every reader maps read-write too: nothing here writes
    /// through it, but a GPU import of the pages may refuse a read-only
    /// range.
    pub fn map(&self, create: Option<usize>) -> Result<memmap2::MmapMut> {
        match self {
            Source::File(path) => {
                let mut o = std::fs::OpenOptions::new();
                o.read(true).write(true);
                if create.is_some() {
                    o.create(true).truncate(true);
                }
                let f = o.open(path).with_context(|| format!("pixel ring {}", path.display()))?;
                if let Some(n) = create {
                    f.set_len(n as u64)?;
                }
                // SAFETY: shared with one writer that only ever writes inside
                // slots under the seqlock; every read is validated by `seq`.
                unsafe { memmap2::MmapMut::map_mut(&f) }.context("mapping pixel ring")
            }
            #[cfg(unix)]
            Source::Posix { name, bytes } => {
                use std::os::fd::FromRawFd;
                let cname = std::ffi::CString::new(format!("/{name}"))
                    .context("pixel ring name")?;
                let fd = unsafe {
                    if let Some(n) = create {
                        // A stale object of another size makes ftruncate
                        // fail on macOS: start clean.
                        libc::shm_unlink(cname.as_ptr());
                        let fd = libc::shm_open(cname.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o600);
                        if fd >= 0 && libc::ftruncate(fd, n as libc::off_t) != 0 {
                            let e = std::io::Error::last_os_error();
                            libc::close(fd);
                            libc::shm_unlink(cname.as_ptr());
                            return Err(e).context("sizing shared memory");
                        }
                        fd
                    } else {
                        libc::shm_open(cname.as_ptr(), libc::O_RDWR, 0)
                    }
                };
                if fd < 0 {
                    return Err(std::io::Error::last_os_error())
                        .with_context(|| format!("shared memory /{name}"));
                }
                let f = unsafe { std::fs::File::from_raw_fd(fd) };
                let len = create.unwrap_or(*bytes);
                // SAFETY: as above. The length is given: fstat on a shm
                // object does not report it on every platform.
                unsafe { memmap2::MmapOptions::new().len(len).map_mut(&f) }
                    .context("mapping shared memory")
            }
            #[cfg(not(unix))]
            Source::Posix { name, .. } => bail!("POSIX shared memory (/{name}) is not available here"),
        }
    }

    /// Remove the object (a writer's, when it is done with it).
    pub fn unlink(&self) {
        match self {
            Source::File(p) => {
                let _ = std::fs::remove_file(p);
            }
            #[cfg(unix)]
            Source::Posix { name, .. } => {
                if let Ok(c) = std::ffi::CString::new(format!("/{name}")) {
                    unsafe { libc::shm_unlink(c.as_ptr()) };
                }
            }
            #[cfg(not(unix))]
            Source::Posix { .. } => {}
        }
    }

    /// The source a writer on this platform makes: shared memory on Unix,
    /// a file in `dir` elsewhere.
    pub fn for_writer(name: &str, dir: &Path, bytes: usize) -> Source {
        if cfg!(unix) {
            Source::Posix { name: name.to_string(), bytes }
        } else {
            Source::File(dir.join(format!("{name}.shm")))
        }
    }
}

pub const MAGIC: &[u8; 8] = b"WOBSHM02";
pub const VERSION: u32 = 2;
/// Apple silicon's page; a multiple of every other platform's.
pub const PAGE: usize = 16384;
pub const HEADER_BYTES: usize = PAGE;
pub const DESC_OFFSET: usize = 64;
pub const DESC_BYTES: usize = 64;
pub const MAX_SLOTS: usize = (HEADER_BYTES - DESC_OFFSET) / DESC_BYTES;
/// Row alignment a buffer-to-texture copy wants.
pub const ROW_ALIGN: usize = 256;

/// Four half floats per pixel.
pub const FMT_RGBA16F: u32 = 1;
/// Four floats per pixel: what Blender's readback hands over as is.
pub const FMT_RGBA32F: u32 = 2;
/// A final (F12) render rather than a viewport draw.
pub const FLAG_FINAL: u32 = 1;
/// The picture is done: nothing left to refine.
pub const FLAG_SETTLED: u32 = 2;

/// Bytes per pixel of a format, `None` for one this reader does not know.
pub fn bpp(fmt: u32) -> Option<usize> {
    match fmt {
        FMT_RGBA16F => Some(8),
        FMT_RGBA32F => Some(16),
        _ => None,
    }
}

/// The smallest 256-byte multiple holding a row.
pub fn stride_for(w: u32, fmt: u32) -> Option<usize> {
    let row = w as usize * bpp(fmt)?;
    Some(row.div_ceil(ROW_ALIGN) * ROW_ALIGN)
}

/// What a writer stamps on a slot besides the pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stamp {
    pub frame: i64,
    pub w: u32,
    pub h: u32,
    pub fmt: u32,
    pub flags: u32,
    pub scene: u64,
    pub ms: f32,
    pub samples: u32,
    /// 0 = the combined picture, else the `blender::Pass` id.
    pub pass: u8,
}

impl Default for Stamp {
    fn default() -> Self {
        Stamp { frame: 0, w: 0, h: 0, fmt: FMT_RGBA16F, flags: 0, scene: 0, ms: 0.0, samples: 0, pass: 0 }
    }
}

/// A slot's descriptor as read: the stamp plus where the pixels are.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Desc {
    pub seq: u64,
    pub stamp: Stamp,
    /// Bytes per row in the slot.
    pub stride: u32,
    /// Byte offset of the pixels from the start of the mapping.
    pub offset: usize,
}

impl Desc {
    pub fn is_final(&self) -> bool {
        self.stamp.flags & FLAG_FINAL != 0
    }
    pub fn settled(&self) -> bool {
        self.stamp.flags & FLAG_SETTLED != 0
    }
}

/// One frame out of the ring, copied.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub seq: u64,
    pub frame: i64,
    pub w: u32,
    pub h: u32,
    pub fmt: u32,
    pub flags: u32,
    pub scene: u64,
    pub ms: f32,
    pub samples: u32,
    pub pass: u8,
    /// `w * h * bpp` bytes, rows un-padded, bottom-up — what a texture
    /// upload takes as is (the shader flips).
    pub pixels: Vec<u8>,
}

impl Frame {
    pub fn is_final(&self) -> bool {
        self.flags & FLAG_FINAL != 0
    }
    pub fn settled(&self) -> bool {
        self.flags & FLAG_SETTLED != 0
    }
}

fn rd_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}
fn rd_u64(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

/// Read `seq` with a volatile load: the other process writes it.
fn load_seq(base: *const u8, slot: usize) -> u64 {
    let p = unsafe { base.add(DESC_OFFSET + slot * DESC_BYTES) } as *const u64;
    u64::from_le(unsafe { std::ptr::read_volatile(p) })
}

/// The reading end (wobble, for frames coming back from Blender).
pub struct Reader {
    map: memmap2::MmapMut,
    slots: usize,
    slot_bytes: usize,
}

impl Reader {
    pub fn open(src: &Source) -> Result<Reader> {
        let map = src.map(None)?;
        let (slots, slot_bytes) = check_header(&map, src)?;
        Ok(Reader { map, slots, slot_bytes })
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// The whole mapping's length.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// A slot's descriptor, if it holds a finished frame this reader can
    /// use. No pixels are touched.
    pub fn desc(&self, slot: usize) -> Option<Desc> {
        if slot >= self.slots {
            return None;
        }
        let seq = load_seq(self.map.as_ptr(), slot);
        if seq & 1 == 1 || seq == 0 {
            return None;
        }
        fence(Ordering::Acquire);
        let d = DESC_OFFSET + slot * DESC_BYTES;
        let desc: [u8; DESC_BYTES] = self.map[d..d + DESC_BYTES].try_into().unwrap();
        let (w, h, fmt) = (rd_u32(&desc, 16), rd_u32(&desc, 20), rd_u32(&desc, 24));
        let stride = rd_u32(&desc, 48);
        let bpp = bpp(fmt)?;
        let bytes = stride as usize * h as usize;
        if w == 0 || h == 0 || (stride as usize) < w as usize * bpp || bytes > self.slot_bytes {
            return None;
        }
        Some(Desc {
            seq,
            stamp: Stamp {
                frame: rd_u64(&desc, 8) as i64,
                w,
                h,
                fmt,
                flags: rd_u32(&desc, 28),
                scene: rd_u64(&desc, 32),
                ms: f32::from_le_bytes(desc[40..44].try_into().unwrap()),
                samples: rd_u32(&desc, 44),
                pass: rd_u32(&desc, 52).min(255) as u8,
            },
            stride,
            offset: HEADER_BYTES + slot * self.slot_bytes,
        })
    }

    /// The slot's sequence is still `seq`: nothing has written it since.
    pub fn still(&self, slot: usize, seq: u64) -> bool {
        fence(Ordering::Acquire);
        slot < self.slots && load_seq(self.map.as_ptr(), slot) == seq
    }

    /// Copy slot `slot` out, rows un-padded. `None` when it is being
    /// written, was lapped during the copy, or holds something this
    /// reader cannot use.
    pub fn read(&self, slot: usize) -> Option<Frame> {
        let d = self.desc(slot)?;
        let bpp = bpp(d.stamp.fmt)?;
        let row = d.stamp.w as usize * bpp;
        let mut pixels = Vec::with_capacity(row * d.stamp.h as usize);
        for y in 0..d.stamp.h as usize {
            let at = d.offset + y * d.stride as usize;
            pixels.extend_from_slice(&self.map[at..at + row]);
        }
        if !self.still(slot, d.seq) {
            return None; // lapped mid-copy
        }
        Some(Frame {
            seq: d.seq,
            frame: d.stamp.frame,
            w: d.stamp.w,
            h: d.stamp.h,
            fmt: d.stamp.fmt,
            flags: d.stamp.flags,
            scene: d.stamp.scene,
            ms: d.stamp.ms,
            samples: d.stamp.samples,
            pass: d.stamp.pass,
            pixels,
        })
    }
}

fn check_header(map: &[u8], src: &Source) -> Result<(usize, usize)> {
    if map.len() < HEADER_BYTES || &map[..8] != MAGIC {
        bail!("{src:?} is not a wobble pixel ring (v{VERSION})");
    }
    let version = rd_u32(map, 8);
    if version != VERSION {
        bail!("pixel ring version {version}, this wobble speaks {VERSION}");
    }
    let slots = rd_u32(map, 12) as usize;
    let slot_bytes = rd_u64(map, 16) as usize;
    if slots == 0
        || slots > MAX_SLOTS
        || slot_bytes % PAGE != 0
        || map.len() < HEADER_BYTES + slots * slot_bytes
    {
        bail!("pixel ring header does not match its size");
    }
    Ok((slots, slot_bytes))
}

/// The writing end. Blender writes the return ring from Python; this is the
/// Rust twin, used for the ring that carries wobble's textures TO Blender
/// and by the tests.
pub struct Writer {
    map: memmap2::MmapMut,
    slots: usize,
    slot_bytes: usize,
    next: usize,
    seq: u64,
    /// Where it is, for the announcement and the unlink on drop.
    pub source: Source,
}

impl Writer {
    /// `slot_bytes` is rounded up to a whole number of pages. A
    /// `Source::Posix`'s `bytes` is set here (the caller names it; the
    /// size follows from the slots).
    pub fn create(src: &Source, slots: usize, slot_bytes: usize) -> Result<Writer> {
        if slots == 0 || slots > MAX_SLOTS {
            bail!("{slots} slots (1..={MAX_SLOTS})");
        }
        let slot_bytes = slot_bytes.max(1).div_ceil(PAGE) * PAGE;
        let total = HEADER_BYTES + slots * slot_bytes;
        let source = match src {
            Source::Posix { name, .. } => Source::Posix { name: name.clone(), bytes: total },
            f => f.clone(),
        };
        let mut map = source.map(Some(total))?;
        map[..8].copy_from_slice(MAGIC);
        map[8..12].copy_from_slice(&VERSION.to_le_bytes());
        map[12..16].copy_from_slice(&(slots as u32).to_le_bytes());
        map[16..24].copy_from_slice(&(slot_bytes as u64).to_le_bytes());
        Ok(Writer { map, slots, slot_bytes, next: 0, seq: 0, source })
    }

    /// Bytes one slot holds.
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// Bytes a `w` x `h` frame of `fmt` takes in a slot, rows padded.
    pub fn bytes_for(w: u32, h: u32, fmt: u32) -> Option<usize> {
        Some(stride_for(w, fmt)? * h as usize)
    }

    /// Write one frame (rows un-padded, `w * h * bpp` bytes) into the next
    /// slot, padding its rows; returns `(slot, seq)` — what the
    /// announcement names.
    pub fn write(&mut self, stamp: Stamp, pixels: &[u8]) -> Result<(usize, u64)> {
        let bpp = bpp(stamp.fmt).context("unknown pixel format")?;
        let row = stamp.w as usize * bpp;
        if pixels.len() != row * stamp.h as usize {
            bail!("{} bytes for {}x{} fmt {}", pixels.len(), stamp.w, stamp.h, stamp.fmt);
        }
        let stride = stride_for(stamp.w, stamp.fmt).unwrap();
        if stride * stamp.h as usize > self.slot_bytes {
            bail!("frame of {} bytes, slots hold {}", stride * stamp.h as usize, self.slot_bytes);
        }
        let slot = self.next;
        self.next = (self.next + 1) % self.slots;
        self.seq += 2;
        let seq = self.seq;
        let d = DESC_OFFSET + slot * DESC_BYTES;
        let seq_p = unsafe { self.map.as_mut_ptr().add(d) } as *mut u64;
        unsafe { std::ptr::write_volatile(seq_p, (seq - 1).to_le()) };
        fence(Ordering::Release);
        let desc = &mut self.map[d + 8..d + DESC_BYTES];
        desc[0..8].copy_from_slice(&stamp.frame.to_le_bytes());
        desc[8..12].copy_from_slice(&stamp.w.to_le_bytes());
        desc[12..16].copy_from_slice(&stamp.h.to_le_bytes());
        desc[16..20].copy_from_slice(&stamp.fmt.to_le_bytes());
        desc[20..24].copy_from_slice(&stamp.flags.to_le_bytes());
        desc[24..32].copy_from_slice(&stamp.scene.to_le_bytes());
        desc[32..36].copy_from_slice(&stamp.ms.to_le_bytes());
        desc[36..40].copy_from_slice(&stamp.samples.to_le_bytes());
        desc[40..44].copy_from_slice(&(stride as u32).to_le_bytes());
        desc[44..48].copy_from_slice(&(stamp.pass as u32).to_le_bytes());
        let at = HEADER_BYTES + slot * self.slot_bytes;
        for y in 0..stamp.h as usize {
            self.map[at + y * stride..at + y * stride + row]
                .copy_from_slice(&pixels[y * row..(y + 1) * row]);
        }
        fence(Ordering::Release);
        unsafe { std::ptr::write_volatile(seq_p, seq.to_le()) };
        Ok((slot, seq))
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        // The mapping outlives the name for anyone who still has it open;
        // the name itself goes with the writer.
        self.source.unlink();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> Source {
        let d = std::env::temp_dir().join(format!("wobble-shm-test-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        Source::File(d.join(name))
    }

    /// The Unix ring: POSIX shared memory, no file anywhere.
    #[cfg(unix)]
    #[test]
    fn posix_shared_memory_round_trips_and_unlinks() {
        let name = format!("wobt{}", std::process::id() % 100000);
        let src = Source::Posix { name: name.clone(), bytes: 0 };
        let mut w = Writer::create(&src, 2, 8).unwrap();
        let (slot, _) = w.write(Stamp { frame: 9, w: 1, h: 1, ..Default::default() }, &[5; 8]).unwrap();
        // A reader opens by name + length, as the announcement carries them.
        let r = Reader::open(&w.source).unwrap();
        assert_eq!(r.read(slot).unwrap().frame, 9);
        assert_eq!(w.source, Source::Posix { name: name.clone(), bytes: HEADER_BYTES + 2 * PAGE });
        drop(w);
        assert!(Reader::open(&Source::Posix { name, bytes: HEADER_BYTES + 2 * PAGE }).is_err(), "unlinked with the writer");
    }

    #[test]
    fn a_frame_round_trips_with_its_stamp_and_padded_rows() {
        let p = tmp("rt.shm");
        // 5 px * 8 bytes = 40-byte rows, padded to 256 in the slot.
        let mut w = Writer::create(&p, 3, Writer::bytes_for(5, 2, FMT_RGBA16F).unwrap()).unwrap();
        let px: Vec<u8> = (0..5 * 2 * 8).map(|i| i as u8).collect();
        let stamp = Stamp {
            frame: 1001,
            w: 5,
            h: 2,
            fmt: FMT_RGBA16F,
            flags: FLAG_SETTLED,
            scene: 7,
            ms: 5.5,
            samples: 16,
            pass: 3,
        };
        let (slot, seq) = w.write(stamp, &px).unwrap();
        let r = Reader::open(&p).unwrap();
        let d = r.desc(slot).unwrap();
        assert_eq!(d.stride, 256);
        assert_eq!(d.offset, HEADER_BYTES + slot * r.slot_bytes());
        assert_eq!(r.slot_bytes() % PAGE, 0, "slots start on page boundaries");
        let f = r.read(slot).unwrap();
        assert_eq!(f.seq, seq);
        assert_eq!((f.frame, f.w, f.h, f.scene, f.samples, f.fmt), (1001, 5, 2, 7, 16, FMT_RGBA16F));
        assert_eq!(f.pass, 3);
        assert!(f.settled() && !f.is_final());
        assert_eq!(f.ms, 5.5);
        assert_eq!(f.pixels, px, "rows come back un-padded");
    }

    #[test]
    fn f32_frames_are_a_format_of_their_own() {
        let p = tmp("f32.shm");
        let mut w = Writer::create(&p, 2, Writer::bytes_for(3, 1, FMT_RGBA32F).unwrap()).unwrap();
        let px = vec![7u8; 3 * 16];
        let stamp = Stamp { w: 3, h: 1, fmt: FMT_RGBA32F, ..Default::default() };
        let (slot, _) = w.write(stamp, &px).unwrap();
        let f = Reader::open(&p).unwrap().read(slot).unwrap();
        assert_eq!(f.fmt, FMT_RGBA32F);
        assert_eq!(f.pixels, px);
        assert!(w.write(Stamp { w: 3, h: 1, fmt: 9, ..Default::default() }, &px).is_err());
    }

    #[test]
    fn slots_rotate_and_an_unwritten_slot_reads_as_nothing() {
        let p = tmp("rot.shm");
        let mut w = Writer::create(&p, 2, 8).unwrap();
        let r = Reader::open(&p).unwrap();
        assert!(r.read(0).is_none() && r.read(1).is_none());
        let s = |frame| Stamp { frame, w: 1, h: 1, ..Default::default() };
        assert_eq!(w.write(s(1), &[1; 8]).unwrap().0, 0);
        assert_eq!(w.write(s(2), &[2; 8]).unwrap().0, 1);
        assert_eq!(w.write(s(3), &[3; 8]).unwrap().0, 0);
        assert_eq!(r.read(0).unwrap().frame, 3);
        assert_eq!(r.read(1).unwrap().frame, 2);
        assert!(r.read(2).is_none());
        assert!(r.still(1, r.desc(1).unwrap().seq));
    }

    #[test]
    fn a_slot_mid_write_is_not_read() {
        let p = tmp("torn.shm");
        let mut w = Writer::create(&p, 1, 8).unwrap();
        w.write(Stamp { frame: 1, w: 1, h: 1, ..Default::default() }, &[0; 8]).unwrap();
        // Leave the seqlock odd, as a writer inside the slot would.
        let seq_p = unsafe { w.map.as_mut_ptr().add(DESC_OFFSET) } as *mut u64;
        unsafe { std::ptr::write_volatile(seq_p, 3u64.to_le()) };
        let r = Reader::open(&p).unwrap();
        assert!(r.read(0).is_none() && r.desc(0).is_none());
    }

    #[test]
    fn a_foreign_file_is_refused() {
        let p = tmp("junk.shm");
        let Source::File(path) = &p else { unreachable!() };
        std::fs::write(path, vec![0u8; HEADER_BYTES + PAGE]).unwrap();
        assert!(Reader::open(&p).is_err());
    }
}
