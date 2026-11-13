//! GPU timestamp instrumentation + CPU-side frame stats.
//!
//! Slot layout per frame (5 timestamps):
//!   0 = before the upload copies
//!   1 = fused pass begin, 2 = fused pass end
//!   3 = blit pass begin,  4 = blit pass end
//! upload = t1 - t0 (the copies + encoder gap), fused = t2 - t1, blit = t4 - t3.
//!
//! Slot 0 is written two ways:
//!   - TIMESTAMP_QUERY_INSIDE_ENCODERS available: `encoder.write_timestamp`
//!     right before the copies (precise).
//!   - Otherwise (Metal/Apple): an empty "sentinel" compute pass whose
//!     end-of-pass timestamp lands just before the copies. Pass-boundary
//!     timestamps are all Metal gives us; encoders run back-to-back so the
//!     gap is a fair upload measure.
//!
//! Readback is a small ring so mapping never stalls the frame loop.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub const TS_UPLOAD_BEGIN: u32 = 0;
pub const TS_FUSED_BEGIN: u32 = 1;
pub const TS_FUSED_END: u32 = 2;
pub const TS_BLIT_BEGIN: u32 = 3;
pub const TS_BLIT_END: u32 = 4;
/// Blur/intermediate passes that run before the main fused pass.
pub const TS_PREPASS_BEGIN: u32 = 5;
pub const TS_PREPASS_END: u32 = 6;
const TS_COUNT: u32 = 7;
const RING: usize = 4;

#[derive(Clone, Copy, Debug, Default)]
pub struct GpuTimings {
    pub upload_ms: f32,
    pub prepass_ms: f32,
    pub fused_ms: f32,
    pub blit_ms: f32,
}

struct RingSlot {
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    mapped: Arc<AtomicBool>,
    in_flight: bool,
    /// How many timestamp slots were resolved this round (3 = no blit pass).
    resolved: u32,
    /// Whether upload copies were actually encoded this round (frames with no
    /// new plate would otherwise drag the upload average toward zero).
    had_upload: bool,
    /// Whether blur/intermediate passes ran this round.
    had_prepass: bool,
    /// Whether the blit pass ran this round.
    had_blit: bool,
}

pub struct GpuTimer {
    query_set: Option<wgpu::QuerySet>,
    slots: Vec<RingSlot>,
    next: usize,
    /// Slot resolved this frame, awaiting its map_async kick after submit.
    pending_map: Option<usize>,
    period_ns: f32,
    pub inside_encoders: bool,
    /// 1x1 no-op dispatch: an empty pass gets elided by the backend and its
    /// timestamps never land, so the sentinel needs one real command.
    noop: wgpu::ComputePipeline,
    pub latest: GpuTimings,
    avg: Avg3,
}

#[derive(Default)]
struct Avg3 {
    upload: RollingAvg,
    prepass: RollingAvg,
    fused: RollingAvg,
    blit: RollingAvg,
}

pub struct RollingAvg {
    window: Vec<f32>,
    idx: usize,
}

impl Default for RollingAvg {
    fn default() -> Self {
        Self { window: Vec::with_capacity(60), idx: 0 }
    }
}

impl RollingAvg {
    pub fn push(&mut self, v: f32) {
        if self.window.len() < 60 {
            self.window.push(v);
        } else {
            self.window[self.idx] = v;
            self.idx = (self.idx + 1) % 60;
        }
    }
    pub fn avg(&self) -> f32 {
        if self.window.is_empty() {
            0.0
        } else {
            self.window.iter().sum::<f32>() / self.window.len() as f32
        }
    }
}

impl GpuTimer {
    pub fn new(ctx: &crate::device::GpuContext) -> Self {
        Self::for_device(
            &ctx.device,
            &ctx.queue,
            ctx.timestamps,
            ctx.timestamps_inside_encoders && ctx.backend != wgpu::Backend::Metal,
        )
    }

    /// The same, for a device this crate did NOT create — an app that owns
    /// its own wgpu device (an egui/eframe one, say) still needs a timer to
    /// pass to `encode_comp`. Callers decide the two capability questions
    /// [`Self::new`] answers from a [`crate::device::GpuContext`]; passing
    /// `false` for both gives a timer that measures nothing and costs
    /// nothing, which is what a host that only wants the pixels should do.
    pub fn for_device(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        timestamps: bool,
        inside_encoders: bool,
    ) -> Self {
        let query_set = timestamps.then(|| {
            device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("frame-timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: TS_COUNT,
            })
        });
        let slots = (0..RING)
            .map(|i| RingSlot {
                resolve: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("ts-resolve-{i}")),
                    size: TS_COUNT as u64 * 8,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                readback: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("ts-readback-{i}")),
                    size: TS_COUNT as u64 * 8,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                mapped: Arc::new(AtomicBool::new(false)),
                in_flight: false,
                resolved: 0,
                had_upload: false,
                had_prepass: false,
                had_blit: false,
            })
            .collect();
        // Metal advertises INSIDE_ENCODERS but Apple GPUs only sample
        // timestamps at stage boundaries, so encoder-level write_timestamp
        // resolves to garbage there — which is why the Metal exclusion lives
        // in `new`, where the backend is known.
        let noop_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ts-noop"),
            source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
        });
        let noop = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ts-noop"),
            layout: None,
            module: &noop_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            noop,
            query_set,
            slots,
            next: 0,
            pending_map: None,
            period_ns: queue.get_timestamp_period(),
            inside_encoders,
            latest: GpuTimings::default(),
            avg: Avg3::default(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.query_set.is_some()
    }

    pub fn query_set(&self) -> Option<&wgpu::QuerySet> {
        self.query_set.as_ref()
    }

    /// Mark the upload-begin slot. Call right before encoding the copies.
    pub fn stamp_upload_begin(&self, encoder: &mut wgpu::CommandEncoder) {
        let Some(qs) = &self.query_set else { return };
        if self.inside_encoders {
            encoder.write_timestamp(qs, TS_UPLOAD_BEGIN);
        } else {
            // Sentinel: minimal pass whose end-of-pass timestamp lands just
            // before the copies.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ts-sentinel"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: qs,
                    beginning_of_pass_write_index: None,
                    end_of_pass_write_index: Some(TS_UPLOAD_BEGIN),
                }),
            });
            pass.set_pipeline(&self.noop);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    pub fn fused_pass_writes(&self) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        self.query_set.as_ref().map(|qs| wgpu::ComputePassTimestampWrites {
            query_set: qs,
            beginning_of_pass_write_index: Some(TS_FUSED_BEGIN),
            end_of_pass_write_index: Some(TS_FUSED_END),
        })
    }

    /// Attach to the FIRST blur/intermediate pass of a frame.
    pub fn prepass_begin_writes(&self) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        self.query_set.as_ref().map(|qs| wgpu::ComputePassTimestampWrites {
            query_set: qs,
            beginning_of_pass_write_index: Some(TS_PREPASS_BEGIN),
            end_of_pass_write_index: None,
        })
    }

    /// Attach to the LAST blur/intermediate pass of a frame.
    pub fn prepass_end_writes(&self) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        self.query_set.as_ref().map(|qs| wgpu::ComputePassTimestampWrites {
            query_set: qs,
            beginning_of_pass_write_index: None,
            end_of_pass_write_index: Some(TS_PREPASS_END),
        })
    }

    pub fn blit_pass_writes(&self) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        self.query_set.as_ref().map(|qs| wgpu::RenderPassTimestampWrites {
            query_set: qs,
            beginning_of_pass_write_index: Some(TS_BLIT_BEGIN),
            end_of_pass_write_index: Some(TS_BLIT_END),
        })
    }

    /// Resolve this frame's queries into a ring slot. Call after all passes.
    /// Flags say which optional passes actually ran this frame, so their
    /// slots aren't read back as stale garbage.
    pub fn resolve(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        uploaded: bool,
        prepass: bool,
        include_blit: bool,
    ) {
        let Some(qs) = &self.query_set else { return };
        let idx = self.next;
        let slot = &mut self.slots[idx];
        if slot.in_flight {
            // Ring exhausted (readback lagging); skip this frame's timing.
            return;
        }
        // Always resolve the full set (prepass slots sit above blit's);
        // the flags gate which diffs collect() trusts.
        let count = if prepass {
            TS_COUNT
        } else if include_blit {
            TS_PREPASS_BEGIN
        } else {
            TS_BLIT_BEGIN
        };
        encoder.resolve_query_set(qs, 0..count, &slot.resolve, 0);
        encoder.copy_buffer_to_buffer(&slot.resolve, 0, &slot.readback, 0, count as u64 * 8);
        slot.in_flight = true;
        slot.resolved = count;
        slot.had_upload = uploaded;
        slot.had_prepass = prepass;
        slot.had_blit = include_blit;
        self.pending_map = Some(idx);
        self.next = (idx + 1) % RING;
    }

    /// Kick off the async map for the slot just resolved. Call after submit.
    pub fn after_submit(&mut self) {
        let Some(idx) = self.pending_map.take() else { return };
        let slot = &self.slots[idx];
        let mapped = slot.mapped.clone();
        slot.readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            if r.is_ok() {
                mapped.store(true, Ordering::Release);
            }
        });
    }

    /// Harvest any completed readbacks (call once per frame, after poll).
    pub fn collect(&mut self) {
        for slot in &mut self.slots {
            if slot.in_flight && slot.mapped.load(Ordering::Acquire) {
                {
                    let view = slot.readback.slice(..).get_mapped_range().expect("a mapped readback");
                    let ts: &[u64] = bytemuck::cast_slice(&view);
                    let ms = |a: u32, b: u32| -> f32 {
                        let (a, b) = (ts[a as usize], ts[b as usize]);
                        if b > a {
                            (b - a) as f32 * self.period_ns * 1e-6
                        } else {
                            0.0
                        }
                    };
                    // Upload window ends where GPU comp work begins: the
                    // first prepass when blurs ran, else the fused pass.
                    let upload_end =
                        if slot.had_prepass { TS_PREPASS_BEGIN } else { TS_FUSED_BEGIN };
                    self.latest = GpuTimings {
                        upload_ms: if slot.had_upload { ms(TS_UPLOAD_BEGIN, upload_end) } else { 0.0 },
                        prepass_ms: if slot.had_prepass {
                            ms(TS_PREPASS_BEGIN, TS_PREPASS_END)
                        } else {
                            0.0
                        },
                        fused_ms: ms(TS_FUSED_BEGIN, TS_FUSED_END),
                        blit_ms: if slot.had_blit { ms(TS_BLIT_BEGIN, TS_BLIT_END) } else { 0.0 },
                    };
                    if slot.had_upload {
                        self.avg.upload.push(self.latest.upload_ms);
                    }
                    if slot.had_prepass {
                        self.avg.prepass.push(self.latest.prepass_ms);
                    }
                    self.avg.fused.push(self.latest.fused_ms);
                    if slot.had_blit {
                        self.avg.blit.push(self.latest.blit_ms);
                    }
                }
                slot.readback.unmap();
                slot.mapped.store(false, Ordering::Release);
                slot.in_flight = false;
            }
        }
    }

    pub fn averages(&self) -> GpuTimings {
        GpuTimings {
            upload_ms: self.avg.upload.avg(),
            prepass_ms: self.avg.prepass.avg(),
            fused_ms: self.avg.fused.avg(),
            blit_ms: self.avg.blit.avg(),
        }
    }
}
