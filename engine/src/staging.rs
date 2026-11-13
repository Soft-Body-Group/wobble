//! Persistent staging-buffer ring: decode threads write EXR pixels straight
//! into mapped wgpu buffers; the render thread encodes buffer->texture copies
//! and the Player re-maps a slot when it evicts the frame. A decoded frame
//! STAYS in its slot after it is shown, so the ring is the scrub cache: a
//! tick the hand comes back to is one GPU copy. The ring is a BYTE budget, not
//! an allocation: slots are created as decodes need them, each the size its
//! frame needs, and a free slot of the wrong size is retired for the asker.
//!
//! This is the discrete-GPU upload path. On UMA (Apple Silicon) the copy is
//! still cheap (GPU-side blit in shared memory); a future `Uploader` impl can
//! skip the texture copy entirely. The trait seam is `Uploader` below.

use crossbeam_channel::{Receiver, Sender, TryRecvError};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;

/// A slot's place in the map → write → unmap → copy → remap cycle. Kept by
/// the ring so the two things wgpu will abort the process over — copying
/// from a mapped buffer, and mapping a buffer twice — are refused here with
/// a log line instead of reaching `Queue::submit`.
const MAPPED: u8 = 0; // in the free list, or handed to a decode worker
const UNMAPPED: u8 = 1; // worker done: pixels ready, copy allowed
const MAPPING: u8 = 2; // copy submitted, map_async in flight

pub struct StagingRing {
    device: wgpu::Device,
    /// A buffer per slot index, created on first acquire and RETIRED (the
    /// index handed back) when a different size needs its bytes: a ring
    /// sized by a byte budget would otherwise allocate gigabytes at rig
    /// build, and one fixed slot size made every reader pay for the
    /// largest (a 2K element beside an 8K plate took a 265 MB slot a frame).
    slots: Vec<std::sync::Mutex<Option<Arc<wgpu::Buffer>>>>,
    /// Bytes of the buffer behind each index (0: none).
    sizes: Vec<AtomicU64>,
    state: Vec<Arc<AtomicU8>>,
    /// The NOMINAL slot size the ring was built with: what the size-less
    /// calls (`try_acquire`, `available`, `slot_count`) count in.
    pub slot_bytes: u64,
    free_rx: Receiver<usize>,
    free_tx: Sender<usize>,
    inner: std::sync::Mutex<Inner>,
    /// An allocation was refused (reported once; the UI shows it).
    oom: AtomicBool,
    /// Bumped every time a slot is handed out for new pixels: (slot, fill)
    /// names one decoded frame, so an upload can tell a texture already
    /// holds it (`fill`).
    fills: Vec<AtomicUsize>,
}

struct Inner {
    /// Free (mapped) slots by buffer size.
    free: std::collections::HashMap<u64, Vec<usize>>,
    /// Indices with no buffer behind them.
    unused: Vec<usize>,
    /// Bytes of every buffer that exists, free or not.
    created_bytes: u64,
    created: usize,
    /// The ceiling on `created_bytes`: what the ring was given, or what it
    /// had when the driver refused an allocation. A pinned (host-visible)
    /// buffer is what runs out first on a discrete GPU, and wgpu treats an
    /// uncaptured OOM as fatal — so a refused slot shrinks the ring instead.
    budget: u64,
}

impl Inner {
    fn free_buffer_bytes(&self) -> u64 {
        self.free.iter().map(|(size, v)| size * v.len() as u64).sum()
    }
}

impl StagingRing {
    /// A ring of up to `count` slots of `slot_bytes` each — or, asked for
    /// other sizes (`try_acquire_sized`), of whatever fits in the same
    /// bytes. Nothing is allocated yet: slots come into being as acquires
    /// need them.
    pub fn new(device: &wgpu::Device, count: usize, slot_bytes: u64) -> Self {
        Self::with_budget(device, count as u64 * slot_bytes, slot_bytes, count)
    }

    /// A ring holding at most `budget` bytes of buffers in at most
    /// `max_slots` of them, `slot_bytes` being the nominal size.
    pub fn with_budget(device: &wgpu::Device, budget: u64, slot_bytes: u64, max_slots: usize) -> Self {
        let (free_tx, free_rx) = crossbeam_channel::unbounded();
        Self {
            device: device.clone(),
            slots: (0..max_slots).map(|_| std::sync::Mutex::new(None)).collect(),
            sizes: (0..max_slots).map(|_| AtomicU64::new(0)).collect(),
            state: (0..max_slots).map(|_| Arc::new(AtomicU8::new(MAPPED))).collect(),
            slot_bytes,
            free_rx,
            free_tx,
            inner: std::sync::Mutex::new(Inner {
                free: Default::default(),
                // Popped from the back: index 0 first.
                unused: (0..max_slots).rev().collect(),
                created_bytes: 0,
                created: 0,
                budget,
            }),
            oom: AtomicBool::new(false),
            fills: (0..max_slots).map(|_| AtomicUsize::new(0)).collect(),
        }
    }

    /// The lock, with every slot whose remap has completed filed under its
    /// size.
    fn inner(&self) -> std::sync::MutexGuard<'_, Inner> {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            match self.free_rx.try_recv() {
                Ok(i) => {
                    let size = self.sizes[i].load(Ordering::Acquire);
                    g.free.entry(size).or_default().push(i);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => panic!("staging ring closed"),
            }
        }
        g
    }

    /// A decode worker has written its pixels and unmapped the slot: it may
    /// now be copied from. Call right after `buffer.unmap()`.
    pub fn note_unmapped(&self, slot: usize) {
        self.state[slot].store(UNMAPPED, Ordering::Release);
    }

    /// May this slot be the source of a GPU copy right now?
    pub fn copyable(&self, slot: usize) -> bool {
        self.state[slot].load(Ordering::Acquire) == UNMAPPED
    }

    /// The ring's capacity in NOMINAL slots: what it was given, or fewer
    /// once the driver refused one (`out_of_memory`).
    pub fn slot_count(&self) -> usize {
        ((self.budget_bytes() / self.slot_bytes.max(1)) as usize).min(self.slots.len())
    }

    /// The bytes of buffers the ring may hold.
    pub fn budget_bytes(&self) -> u64 {
        self.inner().budget
    }

    /// The driver refused a slot and the ring stopped growing there.
    pub fn out_of_memory(&self) -> bool {
        self.oom.load(Ordering::Acquire)
    }

    /// A free (mapped) slot of the nominal size; see `try_acquire_sized`.
    pub fn try_acquire(&self) -> Option<usize> {
        self.try_acquire_sized(self.slot_bytes)
    }

    /// A free (mapped) slot of exactly `size` bytes: one waiting in the
    /// free list, else a new buffer while the budget allows — retiring free
    /// buffers of OTHER sizes to make the room. None when the bytes are all
    /// behind frames (the caller evicts and comes back).
    pub fn try_acquire_sized(&self, size: u64) -> Option<usize> {
        let got = self.acquire_sized(size);
        if let Some(i) = got {
            self.fills[i].fetch_add(1, Ordering::AcqRel);
        }
        got
    }

    /// Which filling of `slot` its pixels are: changes every time the slot
    /// is handed out again.
    pub fn fill(&self, slot: usize) -> usize {
        self.fills[slot].load(Ordering::Acquire)
    }

    fn acquire_sized(&self, size: u64) -> Option<usize> {
        let size = size.max(1);
        let mut g = self.inner();
        if let Some(i) = g.free.get_mut(&size).and_then(|v| v.pop()) {
            return Some(i);
        }
        // Room for a new one? Free buffers of another size are bytes too.
        if g.created_bytes + size > g.budget + g.free_buffer_bytes().min(g.created_bytes) {
            return None;
        }
        while g.created_bytes + size > g.budget {
            // Largest first: fewest buffers destroyed for the room.
            let Some(other) = g.free.iter().filter(|(_, v)| !v.is_empty()).map(|(s, _)| *s).max() else {
                return None;
            };
            let i = g.free.get_mut(&other).and_then(|v| v.pop())?;
            *self.slots[i].lock().unwrap_or_else(|e| e.into_inner()) = None;
            self.sizes[i].store(0, Ordering::Release);
            g.created_bytes -= other;
            g.created -= 1;
            g.unused.push(i);
        }
        let i = g.unused.pop()?;
        // Scoped, so a refused allocation comes back here instead of
        // reaching the uncaptured-error handler (which aborts the app).
        // The scope is per-thread in wgpu 29; acquires run on the render
        // thread. A browser cannot block on the pop, and has no pinned
        // memory to run out of in this sense.
        #[cfg(not(target_arch = "wasm32"))]
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("staging-{i}")),
            size,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(e) = pollster::block_on(scope.pop()) {
            g.unused.push(i);
            g.budget = g.created_bytes;
            if !self.oom.swap(true, Ordering::AcqRel) {
                log::warn!(
                    "staging ring: a {:.0} MB slot was refused ({e}); the ring stays at {} slots ({:.1} GB)",
                    size as f64 / 1e6,
                    g.created,
                    g.created_bytes as f64 / 1e9
                );
            }
            return None;
        }
        g.created_bytes += size;
        g.created += 1;
        self.sizes[i].store(size, Ordering::Release);
        self.state[i].store(MAPPED, Ordering::Release);
        *self.slots[i].lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(buffer));
        Some(i)
    }

    pub fn buffer(&self, slot: usize) -> Arc<wgpu::Buffer> {
        self.slots[slot]
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .expect("staging slot used before it was acquired")
    }

    /// Bytes of the buffer behind `slot`.
    pub fn slot_size(&self, slot: usize) -> u64 {
        self.sizes[slot].load(Ordering::Acquire)
    }

    /// Re-map a slot after the GPU copy was submitted; returns it to the free
    /// list when the map completes (driven by device.poll on the render side).
    pub fn release_after_copy(&self, slot: usize) {
        // Only an UNMAPPED slot can be re-mapped. A second release of the
        // same slot (a frame set discarded AND released, or an evicted frame
        // whose set was already taken) used to map a mapped buffer — that is
        // the `remap failed: BufferAsyncError` line, and the free-list
        // double entry behind `Buffer … is still mapped` at submit.
        if self.state[slot]
            .compare_exchange(UNMAPPED, MAPPING, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            log::warn!(
                "staging slot {slot} released while {} — ignored",
                match self.state[slot].load(Ordering::Acquire) {
                    MAPPED => "already mapped",
                    MAPPING => "already remapping",
                    _ => "in an unknown state",
                }
            );
            return;
        }
        let tx = self.free_tx.clone();
        // The slot keeps its buffer until it is back in the free list, so
        // the callback only needs to hand the index back. A failed map
        // leaves the slot out of the free list rather than lying about its
        // state.
        let flag = self.state[slot].clone();
        self.buffer(slot).slice(..).map_async(wgpu::MapMode::Write, move |r| match r {
            Ok(()) => {
                flag.store(MAPPED, Ordering::Release);
                let _ = tx.send(slot);
            }
            Err(e) => log::error!("staging slot {slot} remap failed: {e:?}"),
        });
    }

    /// Free buffers, of any size.
    pub fn free_count(&self) -> usize {
        self.inner().free.values().map(Vec::len).sum()
    }

    /// NOMINAL slots a decode could take right now: free, or not yet created.
    pub fn available(&self) -> usize {
        (self.free_bytes() / self.slot_bytes.max(1)) as usize
    }

    /// Bytes a decode could take right now: free buffers (of any size — a
    /// wrong-sized one is retired for the asker) and budget not yet spent.
    pub fn free_bytes(&self) -> u64 {
        let g = self.inner();
        g.budget.saturating_sub(g.created_bytes) + g.free_buffer_bytes()
    }

    /// Slots on their way back to the free list (copy submitted, map in
    /// flight): the next poll returns them.
    pub fn remapping(&self) -> usize {
        self.state.iter().filter(|s| s.load(Ordering::Acquire) == MAPPING).count()
    }

    /// The bytes behind the slots on their way back.
    pub fn remapping_bytes(&self) -> u64 {
        self.state
            .iter()
            .zip(&self.sizes)
            .filter(|(s, _)| s.load(Ordering::Acquire) == MAPPING)
            .map(|(_, b)| b.load(Ordering::Acquire))
            .sum()
    }

    /// Slots with a buffer behind them.
    pub fn created(&self) -> usize {
        self.inner().created
    }

    /// The ring's footprint: bytes of every buffer that exists.
    pub fn created_bytes(&self) -> u64 {
        self.inner().created_bytes
    }
}

/// Upload-path seam (brief: staging ring assumes discrete VRAM; UMA path can
/// read decode output directly). Only the ring impl exists in the POC.
pub trait Uploader: Send + Sync {
    /// Encode whatever is needed to make `slot`'s pixels visible to sampling
    /// as `texture` this frame.
    fn encode_upload(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ring: &StagingRing,
        slot: usize,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
        bytes_per_row: u32,
    );
}

pub struct CopyUploader;

impl Uploader for CopyUploader {
    fn encode_upload(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ring: &StagingRing,
        slot: usize,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
        bytes_per_row: u32,
    ) {
        // A mapped buffer in a submitted copy aborts the process (wgpu
        // validation). Dropping one frame's upload is the lesser harm.
        if !ring.copyable(slot) {
            log::warn!("staging slot {slot} is not copyable this frame — upload skipped");
            return;
        }
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &ring.buffer(slot),
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two operations wgpu aborts the process over — copying from a
    /// mapped slot, and mapping a slot twice — are refused by the ring's
    /// state machine instead. A slot released twice ends up in the free list
    /// ONCE, and a slot is only copyable between its unmap and its release.
    #[test]
    fn a_slot_is_released_once_and_copyable_only_when_unmapped() {
        let Ok(ctx) = crate::GpuContext::new() else {
            eprintln!("no GPU; skipping");
            return;
        };
        let ring = StagingRing::new(&ctx.device, 2, 256);
        assert_eq!(ring.available(), 2);
        let slot = ring.try_acquire().expect("a free slot");
        assert!(!ring.copyable(slot), "a slot handed to a worker is still mapped");

        // The worker writes and unmaps.
        ring.buffer(slot).unmap();
        ring.note_unmapped(slot);
        assert!(ring.copyable(slot), "unmapped: the copy may be encoded");

        // Released after the copy — and then, by mistake, again.
        ring.release_after_copy(slot);
        assert!(!ring.copyable(slot), "a remapping slot must not be copied from");
        ring.release_after_copy(slot); // the double release: ignored, not a second map_async

        // Let the map complete.
        let _ = ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        std::thread::sleep(web_time::Duration::from_millis(50));
        let _ = ctx.device.poll(wgpu::PollType::Poll);
        assert_eq!(ring.available(), 2, "back in the free list exactly once");
        // Both entries are distinct slots: no double entry for `slot`.
        let a = ring.try_acquire().unwrap();
        let b = ring.try_acquire().unwrap();
        assert_ne!(a, b, "a double release would have put the same slot in twice");
        assert!(ring.try_acquire().is_none());
    }

    /// One byte budget, slots of the size each frame needs: small frames do
    /// not pay for the largest, and a free slot of the wrong size gives its
    /// bytes to the asker.
    #[test]
    fn slots_come_in_the_size_asked_for_under_one_budget() {
        let Ok(ctx) = crate::GpuContext::new() else {
            eprintln!("no GPU; skipping");
            return;
        };
        let settle = |ring: &StagingRing, slot: usize| {
            ring.buffer(slot).unmap();
            ring.note_unmapped(slot);
            ring.release_after_copy(slot);
            let _ = ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            std::thread::sleep(web_time::Duration::from_millis(50));
            let _ = ctx.device.poll(wgpu::PollType::Poll);
        };
        // Room for two 1024-byte slots, or eight of 256.
        let ring = StagingRing::with_budget(&ctx.device, 2048, 1024, 16);
        let big = ring.try_acquire_sized(1024).expect("a big slot");
        let small: Vec<usize> = (0..4).map(|_| ring.try_acquire_sized(256).expect("a small slot")).collect();
        assert_eq!((ring.slot_size(big), ring.slot_size(small[0])), (1024, 256));
        assert_eq!(ring.free_bytes(), 0);
        assert!(ring.try_acquire_sized(256).is_none(), "the bytes are all behind frames");
        // The big one comes back: its bytes serve four small ones.
        settle(&ring, big);
        assert_eq!(ring.free_bytes(), 1024);
        let more: Vec<usize> = (0..4).map(|_| ring.try_acquire_sized(256).expect("room from the retired slot")).collect();
        assert_eq!(ring.created_bytes(), 2048);
        assert!(ring.try_acquire_sized(256).is_none());
        // And the other way: four small ones back make room for a big one,
        // while a same-sized ask takes one from the free list as it is.
        for s in &more {
            settle(&ring, *s);
        }
        let again = ring.try_acquire_sized(256).expect("a free slot of that size");
        assert!(more.contains(&again));
        assert!(ring.try_acquire_sized(1024).is_none(), "768 free bytes are not 1024");
        settle(&ring, again);
        let big2 = ring.try_acquire_sized(1024).expect("four retired small slots are one big one");
        assert_eq!(ring.slot_size(big2), 1024);
        assert_eq!(ring.created_bytes(), 2048);
        assert_eq!(small.len(), 4);
    }
}
