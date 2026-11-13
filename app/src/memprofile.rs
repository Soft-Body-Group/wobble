//! Memory profile: how the frame caches divide a machine's memory.
//!
//! There are two kinds of machine, and the caches want opposite shapes:
//!
//! - **Unified** (Apple Silicon, and the unified-memory APUs arriving on
//!   other platforms): CPU and GPU share one pool. A mapped staging buffer
//!   is ordinary memory, so the staging ring IS the scrub cache — a frame
//!   decoded once is one GPU copy away from the screen. The budget splits
//!   half to each player's ring, half to the CPU RAM tier. This is the
//!   shape wobble was built on, and it stays the default there.
//! - **Discrete** (a PCIe card with its own VRAM — the 4090 workstation):
//!   a mapped staging buffer is PINNED host memory, page-locked and
//!   allocated through the driver from a heap well under physical RAM, and
//!   an allocation past it is a fatal out-of-memory. The unified shape put
//!   half the RAM budget there per player (two players: the whole budget
//!   again), which is how a 92 GB budget on a 128 GB machine crashed a 6K
//!   comp. Here the ring shrinks to an UPLOAD PIPE under its own small
//!   pinned budget, the RAM budget goes whole to the pageable RAM tier, and
//!   every Read caches there (on unified it is opt-in per Read, because the
//!   ring already holds the frames).
//!
//! `Auto` reads the adapter: wgpu reports a unified-memory GPU as
//! `IntegratedGpu` (on Metal that IS `has_unified_memory`), a PCIe card as
//! `DiscreteGpu`. Anything it cannot place is treated as discrete — the
//! shape that cannot exhaust pinned memory.

/// What the user picked (prefs). `Auto` follows the adapter.
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Pick {
    #[default]
    Auto,
    Unified,
    Discrete,
}

impl Pick {
    pub const ALL: [Pick; 3] = [Pick::Auto, Pick::Unified, Pick::Discrete];
}

/// The profile in force.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Unified,
    Discrete,
}

impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Kind::Unified => "unified memory",
            Kind::Discrete => "discrete GPU",
        }
    }
}

pub fn detect(device_type: wgpu::DeviceType) -> Kind {
    match device_type {
        wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Cpu => Kind::Unified,
        wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::VirtualGpu | wgpu::DeviceType::Other => {
            Kind::Discrete
        }
    }
}

pub fn resolve(pick: Pick, device_type: wgpu::DeviceType) -> Kind {
    match pick {
        Pick::Auto => detect(device_type),
        Pick::Unified => Kind::Unified,
        Pick::Discrete => Kind::Discrete,
    }
}

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Installed physical memory in bytes, asked of the OS once. `None` where
/// the OS will not say (a browser).
pub fn physical_ram() -> Option<u64> {
    static RAM: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *RAM.get_or_init(query_physical_ram)
}

#[cfg(target_os = "windows")]
fn query_physical_ram() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: a zeroed MEMORYSTATUSEX with dwLength set is the documented
    // input; the call only writes into it.
    let mut st: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    st.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    (unsafe { GlobalMemoryStatusEx(&mut st) } != 0).then_some(st.ullTotalPhys)
}

#[cfg(target_os = "macos")]
fn query_physical_ram() -> Option<u64> {
    std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
}

#[cfg(all(unix, not(target_os = "macos")))]
fn query_physical_ram() -> Option<u64> {
    let info = std::fs::read_to_string("/proc/meminfo").ok()?;
    parse_meminfo(&info)
}

#[cfg(not(any(unix, target_os = "windows")))]
fn query_physical_ram() -> Option<u64> {
    None
}

/// `MemTotal:  131072000 kB` -> bytes.
#[cfg_attr(not(all(unix, not(target_os = "macos"))), allow(dead_code))]
fn parse_meminfo(info: &str) -> Option<u64> {
    let line = info.lines().find(|l| l.starts_with("MemTotal:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// The RAM budget when the user has not set one, in GiB.
/// Unified: a quarter of RAM, 4–24 (the rule wobble always had — it is
/// shared with the GPU there, and half of it becomes the ring).
/// Discrete: half of RAM, 4–256 — all of it pageable RAM tier, leaving the
/// other half to the OS, the pinned pipe and whatever else the workstation
/// runs (another compositor, a renderer).
pub fn auto_ram_gb(kind: Kind, phys: Option<u64>) -> f64 {
    let Some(b) = phys else { return 8.0 };
    let gb = b as f64 / GIB;
    match kind {
        Kind::Unified => (gb / 4.0).clamp(4.0, 24.0),
        Kind::Discrete => (gb / 2.0).clamp(4.0, 256.0),
    }
}

/// The pinned upload budget when the user has not set one, in GiB
/// (discrete only): a sixteenth of RAM, 2–8. Enough for a few ticks of
/// several 6K readers in flight (a 6K slot is ~290 MB), far under any
/// driver's pinned-heap ceiling.
pub fn auto_pinned_gb(phys: Option<u64>) -> f64 {
    phys.map_or(4.0, |b| (b as f64 / GIB / 16.0).clamp(2.0, 8.0))
}

/// The budgets in force — one value the app keeps and every sizing site
/// asks, so the two profiles cannot drift apart site by site.
#[derive(Clone, Copy, Debug)]
pub struct Budgets {
    pub kind: Kind,
    /// The RAM budget the user set (or auto).
    pub ram_bytes: usize,
    /// Discrete: the ceiling on each staging ring's pinned bytes.
    pub pinned_bytes: u64,
    /// Every Read caches in the RAM tier, whatever its own toggle says.
    pub all_reads_in_ram: bool,
}

impl Budgets {
    /// The CPU RAM tier's share. Unified: the half the rings do not take.
    /// Discrete: all of it — the rings live on their own pinned budget.
    pub fn ram_tier(&self) -> usize {
        match self.kind {
            Kind::Unified => self.ram_bytes / 2,
            Kind::Discrete => self.ram_bytes,
        }
    }

    /// Does a staging ring double as the frame cache (the unified shape),
    /// or is it only the pipe to the GPU?
    pub fn ring_is_cache(&self) -> bool {
        self.kind == Kind::Unified
    }

    /// Slots for one player's ring of `slot_bytes` slots feeding `readers`
    /// readers. `classic` is the old fixed ring (`--slots`), `floor` the
    /// playback floor (`readers × (ahead + 3)`). Never fewer than two
    /// ticks: the tick on screen is pinned, so one tick's worth deadlocks.
    pub fn ring_slots(&self, slot_bytes: u64, readers: usize, classic: usize, floor: usize) -> usize {
        let two_ticks = 2 * readers.max(1);
        let n = match self.kind {
            Kind::Unified => {
                let by_budget = (self.ram_bytes as u64 / 2 / slot_bytes.max(1)) as usize;
                by_budget.max(classic).max(floor)
            }
            // The pinned budget is a hard ceiling — it may cut the floor
            // (the RAM tier refills a slot with one memcpy, so a short
            // pipe costs prefetch depth, not decodes), never two ticks.
            Kind::Discrete => (self.pinned_bytes / slot_bytes.max(1)) as usize,
        };
        // 4096 bounds buffer objects and per-slot bookkeeping, not memory.
        n.max(two_ticks).min(4096)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: u64 = 1 << 30;
    /// A 6000×6000 RGBA half-float slot, rows padded to 256 bytes.
    const SLOT_6K: u64 = 48_128 * 6000;

    fn budgets(kind: Kind, ram_gb: u64) -> Budgets {
        Budgets {
            kind,
            ram_bytes: (ram_gb * G) as usize,
            pinned_bytes: 8 * G,
            all_reads_in_ram: kind == Kind::Discrete,
        }
    }

    #[test]
    fn auto_follows_the_adapter_and_an_unknown_one_is_discrete() {
        assert_eq!(detect(wgpu::DeviceType::IntegratedGpu), Kind::Unified);
        assert_eq!(detect(wgpu::DeviceType::DiscreteGpu), Kind::Discrete);
        assert_eq!(detect(wgpu::DeviceType::Other), Kind::Discrete);
        assert_eq!(resolve(Pick::Unified, wgpu::DeviceType::DiscreteGpu), Kind::Unified);
        assert_eq!(resolve(Pick::Discrete, wgpu::DeviceType::IntegratedGpu), Kind::Discrete);
    }

    /// The crash: 92 GB budget, 6K plates, discrete card. Unified sizing
    /// lets one ring grow to ~46 GB of pinned memory; discrete keeps it
    /// at the pinned budget and hands the 92 GB to the pageable tier.
    #[test]
    fn a_big_budget_on_a_discrete_card_stays_out_of_pinned_memory() {
        let uni = budgets(Kind::Unified, 92);
        let dis = budgets(Kind::Discrete, 92);
        let floor = 6 * (5 + 3);
        let uni_pinned = uni.ring_slots(SLOT_6K, 6, 12, floor) as u64 * SLOT_6K;
        let dis_pinned = dis.ring_slots(SLOT_6K, 6, 12, floor) as u64 * SLOT_6K;
        assert!(uni_pinned > 40 * G, "unified ring {uni_pinned}");
        assert!(dis_pinned <= 8 * G, "discrete ring {dis_pinned}");
        assert_eq!(dis.ram_tier(), 92 * G as usize);
        assert_eq!(uni.ram_tier(), 46 * G as usize);
    }

    #[test]
    fn a_ring_never_drops_below_two_ticks() {
        let mut dis = budgets(Kind::Discrete, 64);
        dis.pinned_bytes = SLOT_6K; // one slot's worth
        assert_eq!(dis.ring_slots(SLOT_6K, 6, 12, 48), 12);
    }

    #[test]
    fn unified_sizing_is_unchanged() {
        // What `ring_slots_for` computed before profiles existed.
        let uni = budgets(Kind::Unified, 16);
        let slot = 2048 * 8 * 1080;
        let old = ((16 * G / 2 / slot) as usize).max(12).max(8).min(4096);
        assert_eq!(uni.ring_slots(slot, 1, 12, 8), old);
    }

    #[test]
    fn auto_budgets() {
        let ram = Some(128 * G);
        assert_eq!(auto_ram_gb(Kind::Unified, ram), 24.0);
        assert_eq!(auto_ram_gb(Kind::Discrete, ram), 64.0);
        assert_eq!(auto_ram_gb(Kind::Unified, Some(48 * G)), 12.0);
        assert_eq!(auto_pinned_gb(ram), 8.0);
        assert_eq!(auto_pinned_gb(Some(16 * G)), 2.0);
        assert_eq!(auto_ram_gb(Kind::Discrete, None), 8.0);
    }

    #[test]
    fn meminfo_parses() {
        let info = "MemTotal:       131072000 kB\nMemFree:  1 kB\n";
        assert_eq!(parse_meminfo(info), Some(131_072_000 * 1024));
        assert_eq!(parse_meminfo("nothing"), None);
    }

    #[test]
    fn this_machine_reports_its_ram() {
        #[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
        assert!(physical_ram().is_some_and(|b| b > G));
    }
}
