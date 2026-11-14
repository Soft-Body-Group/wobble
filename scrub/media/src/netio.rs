//! Reading frame files, with network storage in mind.
//!
//! - `read_file`: the whole file in one buffer. On macOS the read bypasses
//!   the page cache (`F_NOCACHE`): a 4K EXR sequence streamed once would
//!   otherwise push everything else out of RAM, and our own caches (the
//!   raw-bytes cache, the frame cache, the local disk cache) are where a
//!   frame is kept.
//! - `is_network`: the file lives on a network share (SMB, NFS, AFP,
//!   WebDAV), by the volume's file system type.
//! - A NETWORK SIMULATOR for measuring without a share: `EDIFY_NETSIM` =
//!   `1g` (110 MB/s, 3 ms), `10g` (1.1 GB/s, 2 ms) or `<MB/s>:<ms>`. Every
//!   read pays the latency (overlapping other reads, as round trips do)
//!   and then queues on ONE shared link of that bandwidth. With it set,
//!   every file counts as remote.

use std::io::Read;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// A link of `bytes_per_sec`, `latency` a request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NetSim {
    pub bytes_per_sec: f64,
    pub latency: Duration,
}

impl NetSim {
    pub const GBE_1: NetSim = NetSim { bytes_per_sec: 110e6, latency: Duration::from_millis(3) };
    pub const GBE_10: NetSim = NetSim { bytes_per_sec: 1.1e9, latency: Duration::from_millis(2) };

    pub fn parse(s: &str) -> Option<NetSim> {
        match s.trim().to_ascii_lowercase().as_str() {
            "1g" | "1gbe" => Some(Self::GBE_1),
            "10g" | "10gbe" => Some(Self::GBE_10),
            "" | "0" | "off" => None,
            o => {
                let (mb, ms) = o.split_once(':').unwrap_or((o, "2"));
                Some(NetSim { bytes_per_sec: mb.parse::<f64>().ok()? * 1e6, latency: Duration::from_secs_f64(ms.parse::<f64>().ok()? / 1000.0) })
            }
        }
    }
}

static SIM: Mutex<Option<Option<NetSim>>> = Mutex::new(None);
/// When the simulated link is next free.
static LINK: Mutex<Option<Instant>> = Mutex::new(None);

/// The simulator in force (from `EDIFY_NETSIM`, unless `set_sim` said).
pub fn sim() -> Option<NetSim> {
    let mut g = SIM.lock().unwrap();
    *g.get_or_insert_with(|| crate::sys::var("NETSIM").ok().and_then(|s| NetSim::parse(&s)))
}

/// Set (or clear) the simulator (benches, tests).
pub fn set_sim(s: Option<NetSim>) {
    *SIM.lock().unwrap() = Some(s);
    *LINK.lock().unwrap() = None;
}

/// Pay for `n` bytes on the simulated link.
fn pay(s: NetSim, n: usize) {
    std::thread::sleep(s.latency);
    let dur = Duration::from_secs_f64(n as f64 / s.bytes_per_sec);
    let done = {
        let mut g = LINK.lock().unwrap();
        let now = Instant::now();
        let start = g.filter(|t| *t > now).unwrap_or(now);
        let done = start + dur;
        *g = Some(done);
        done
    };
    std::thread::sleep(done.saturating_duration_since(Instant::now()));
}

/// The whole file, read past the page cache where the OS allows.
pub fn read_file(path: &Path) -> std::io::Result<Vec<u8>> {
    let mut f = std::fs::File::open(path)?;
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: a valid fd we own; F_NOCACHE takes an int flag.
        unsafe {
            libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1);
        }
    }
    let n = f.metadata().map(|m| m.len() as usize).unwrap_or(0);
    let mut buf = Vec::with_capacity(n);
    f.read_to_end(&mut buf)?;
    if let Some(s) = sim() {
        pay(s, buf.len());
    }
    Ok(buf)
}

/// The file is on a network share (or the simulator is on).
pub fn is_network(path: &Path) -> bool {
    if sim().is_some() {
        return true;
    }
    fs_is_network(path)
}

#[cfg(target_os = "macos")]
fn fs_is_network(path: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let Ok(c) = std::ffi::CString::new(path.as_os_str().as_bytes()) else { return false };
    let mut st: libc::statfs = unsafe { std::mem::zeroed() };
    // SAFETY: a NUL-terminated path and a zeroed out-struct.
    if unsafe { libc::statfs(c.as_ptr(), &mut st) } != 0 {
        return false;
    }
    let name: Vec<u8> = st.f_fstypename.iter().take_while(|&&b| b != 0).map(|&b| b as u8).collect();
    matches!(name.as_slice(), b"smbfs" | b"nfs" | b"afpfs" | b"webdav" | b"cifs")
}

#[cfg(target_os = "linux")]
fn fs_is_network(path: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let Ok(c) = std::ffi::CString::new(path.as_os_str().as_bytes()) else { return false };
    let mut st: libc::statfs = unsafe { std::mem::zeroed() };
    // SAFETY: as above.
    if unsafe { libc::statfs(c.as_ptr(), &mut st) } != 0 {
        return false;
    }
    // NFS, SMB, CIFS (smb2): 32-bit magic numbers (f_type is signed and its width varies by arch).
    matches!(st.f_type as u32, 0x6969 | 0x517B | 0xFF53_4D42 | 0xFE53_4D42)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn fs_is_network(path: &Path) -> bool {
    path.to_string_lossy().starts_with(r"\\")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_simulated_link_is_shared_and_pays_latency() {
        let s = NetSim { bytes_per_sec: 100e6, latency: Duration::from_millis(5) };
        set_sim(Some(s));
        let t = Instant::now();
        // Four 5 MB reads at once: 20 MB over one 100 MB/s link = 0.2 s
        // (plus one overlapped latency), not 4 x 0.05 s.
        let hs: Vec<_> = (0..4).map(|_| std::thread::spawn(move || pay(s, 5_000_000))).collect();
        for h in hs {
            h.join().unwrap();
        }
        let e = t.elapsed().as_secs_f64();
        set_sim(None);
        assert!((0.19..0.35).contains(&e), "{e}");
        assert_eq!(NetSim::parse("1g"), Some(NetSim::GBE_1));
        assert_eq!(NetSim::parse("500:4"), Some(NetSim { bytes_per_sec: 500e6, latency: Duration::from_millis(4) }));
    }
}
