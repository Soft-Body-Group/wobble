//! Windows native crash capture — report.rs's half for structured exceptions
//! (an access violation in a GPU driver, an illegal instruction, a stack
//! overflow that got as far as an exception). A Rust panic never comes
//! through here: the panic hook has it.
//!
//! One process-wide unhandled-exception filter. It hands report.rs a `Fatal`
//! — code, faulting module + offset — and report.rs asks it, in order of
//! how likely each is to survive a broken process, for a minidump and then
//! the faulting thread's stack. The stack is walked from the exception's OWN
//! context with `RtlVirtualUnwind` (a backtrace taken inside the filter would
//! start in the exception dispatcher) and symbolised by dbghelp, which finds
//! `wobble.pdb` beside `wobble.exe` — the alpha zip ships both.
//!
//! The filter then lets the search continue, so Windows Error Reporting still
//! records the crash (the event-log fallback in report.rs reads that for the
//! deaths that never reach a filter: `__fastfail`, which is how Rust aborts
//! and how its stack-overflow handler exits on Windows).

use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use windows_sys::Win32::Foundation::HMODULE;
use windows_sys::Win32::System::Diagnostics::Debug::{
    EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS, IMAGEHLP_LINEW64, LPTOP_LEVEL_EXCEPTION_FILTER,
    MINIDUMP_EXCEPTION_INFORMATION, MiniDumpWithThreadInfo, MiniDumpWithUnloadedModules, MiniDumpWriteDump,
    SYMBOL_INFOW, SYMOPT_DEFERRED_LOADS, SYMOPT_LOAD_LINES, SYMOPT_UNDNAME, SetUnhandledExceptionFilter,
    SymFromAddrW, SymGetLineFromAddrW64, SymInitializeW, SymSetOptions,
};
use windows_sys::Win32::System::LibraryLoader::{
    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, GetModuleFileNameW,
    GetModuleHandleExW,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetCurrentProcessId, GetCurrentThreadId};

/// The crash, as the filter saw it.
pub struct Fatal {
    pub code: u32,
    /// File name of the module the faulting address is in (`wobble.exe`,
    /// `nvwgf2umx.dll`, …), "?" when it is in none.
    pub module: String,
    pub offset: usize,
    info: *const EXCEPTION_POINTERS,
}

static HANDLER: OnceLock<fn(&Fatal)> = OnceLock::new();
static PREV: OnceLock<LPTOP_LEVEL_EXCEPTION_FILTER> = OnceLock::new();
static IN_FILTER: AtomicBool = AtomicBool::new(false);

/// Install the filter (once). `handler` runs on the crashing thread.
pub fn install(handler: fn(&Fatal)) {
    if HANDLER.set(handler).is_err() {
        return;
    }
    // SAFETY: plain Win32 call; `filter` matches LPTOP_LEVEL_EXCEPTION_FILTER.
    let prev = unsafe { SetUnhandledExceptionFilter(Some(filter)) };
    let _ = PREV.set(prev);
}

unsafe extern "system" fn filter(info: *const EXCEPTION_POINTERS) -> i32 {
    // A second crash while handling the first (or on another thread at the
    // same moment) goes straight to Windows.
    if !IN_FILTER.swap(true, Ordering::SeqCst) {
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: Windows hands the filter valid exception pointers.
            unsafe { run(info) }
        }));
        drop(caught);
    }
    match PREV.get().copied().flatten() {
        // SAFETY: the filter that was installed before ours, same contract.
        Some(prev) => unsafe { prev(info) },
        None => EXCEPTION_CONTINUE_SEARCH,
    }
}

unsafe fn run(info: *const EXCEPTION_POINTERS) {
    let Some(handler) = HANDLER.get() else { return };
    if info.is_null() {
        return;
    }
    // SAFETY: non-null pointers from the exception dispatcher.
    let rec = unsafe { (*info).ExceptionRecord };
    if rec.is_null() {
        return;
    }
    let (code, address) = unsafe { ((*rec).ExceptionCode as u32, (*rec).ExceptionAddress as usize) };
    let (module, base) = module_at(address);
    handler(&Fatal { code, module, offset: address.wrapping_sub(base), info });
}

/// (file name, load address) of the module containing `addr`.
fn module_at(addr: usize) -> (String, usize) {
    let mut h: HMODULE = std::ptr::null_mut();
    // SAFETY: FROM_ADDRESS reads `addr` as an address, not a string;
    // UNCHANGED_REFCOUNT means there is nothing to free.
    let ok = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            addr as *const u16,
            &mut h,
        )
    };
    if ok == 0 || h.is_null() {
        return ("?".into(), 0);
    }
    let mut buf = [0u16; 520];
    // SAFETY: buffer and its length in u16s.
    let n = unsafe { GetModuleFileNameW(h, buf.as_mut_ptr(), buf.len() as u32) } as usize;
    let path = String::from_utf16_lossy(&buf[..n.min(buf.len())]);
    let name = path.rsplit(['\\', '/']).next().unwrap_or(&path).to_string();
    (name, h as usize) // an HMODULE IS the module's load address
}

impl Fatal {
    /// A minidump of the dying process (thread stacks + module list, a few
    /// hundred KB) for a debugger or `minidump-stackwalk`.
    pub fn write_minidump(&self, path: &Path) -> bool {
        use std::os::windows::io::AsRawHandle;
        let Ok(f) = std::fs::File::create(path) else { return false };
        let mei = MINIDUMP_EXCEPTION_INFORMATION {
            // SAFETY: trivial Win32 getter.
            ThreadId: unsafe { GetCurrentThreadId() },
            ExceptionPointers: self.info as *mut EXCEPTION_POINTERS,
            ClientPointers: 0,
        };
        // SAFETY: valid process + file handles; `mei` outlives the call.
        let ok = unsafe {
            MiniDumpWriteDump(
                GetCurrentProcess(),
                GetCurrentProcessId(),
                f.as_raw_handle(),
                MiniDumpWithThreadInfo | MiniDumpWithUnloadedModules,
                &mei,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        ok != 0
    }

    /// The faulting thread's stack, one entry per frame: `module+0xoffset
    /// symbol` and, when the PDB has it, `at file:line` on the next line.
    pub fn stack(&self) -> Vec<String> {
        symbolise(&self.walk())
    }

    #[cfg(target_arch = "x86_64")]
    fn walk(&self) -> Vec<u64> {
        use windows_sys::Win32::System::Diagnostics::Debug::{
            CONTEXT, RtlLookupFunctionEntry, RtlVirtualUnwind, UNW_FLAG_NHANDLER,
        };
        // SAFETY: the dispatcher's context record, copied so unwinding never
        // writes to the real one (the debugger/WER still want it).
        let ctx_ptr = unsafe { (*self.info).ContextRecord };
        if ctx_ptr.is_null() {
            return Vec::new();
        }
        let mut ctx: CONTEXT = unsafe { *ctx_ptr };
        let mut pcs = Vec::new();
        for _ in 0..96 {
            let pc = ctx.Rip;
            if pc == 0 {
                break;
            }
            pcs.push(pc);
            let mut base = 0u64;
            // SAFETY: looking up unwind data for an address is read-only.
            let entry = unsafe { RtlLookupFunctionEntry(pc, &mut base, std::ptr::null_mut()) };
            if entry.is_null() {
                // A leaf function: the return address is on top of the stack.
                if ctx.Rsp == 0 {
                    break;
                }
                // SAFETY: Rsp of a live frame of this thread.
                ctx.Rip = unsafe { *(ctx.Rsp as *const u64) };
                ctx.Rsp += 8;
            } else {
                let mut handler_data = std::ptr::null_mut();
                let mut establisher = 0u64;
                // SAFETY: unwinds our copy of the context by one frame.
                unsafe {
                    RtlVirtualUnwind(
                        UNW_FLAG_NHANDLER,
                        base,
                        pc,
                        entry,
                        &mut ctx,
                        &mut handler_data,
                        &mut establisher,
                        std::ptr::null_mut(),
                    );
                }
            }
        }
        pcs
    }

    /// No hand-rolled unwinder off x86-64; the minidump still has the stack.
    #[cfg(not(target_arch = "x86_64"))]
    fn walk(&self) -> Vec<u64> {
        Vec::new()
    }
}

/// dbghelp symbolisation. It is not thread-safe, but this only runs in the
/// crash filter, which admits one thread. `SymInitializeW` fails if std's
/// own backtrace code initialised it first — which still leaves it usable.
fn symbolise(pcs: &[u64]) -> Vec<String> {
    const NAME_MAX: usize = 512;
    #[repr(C)]
    struct SymBuf {
        info: SYMBOL_INFOW,
        _room: [u16; NAME_MAX],
    }
    // SAFETY: plain dbghelp calls on our own process.
    let process = unsafe { GetCurrentProcess() };
    unsafe {
        SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
        SymInitializeW(process, std::ptr::null(), 1);
    }
    let mut out = Vec::with_capacity(pcs.len());
    for (i, &pc) in pcs.iter().enumerate() {
        // Return addresses point AFTER the call; step back into it so the
        // line is the call's, not the next statement's.
        let a = if i == 0 { pc } else { pc.saturating_sub(1) };
        let (module, base) = module_at(a as usize);
        let mut line = format!("{i:3}: {module}+0x{:x}", (a as usize).wrapping_sub(base));
        // SAFETY: zeroed POD; SizeOfStruct/MaxNameLen describe the buffer.
        let mut buf: SymBuf = unsafe { std::mem::zeroed() };
        buf.info.SizeOfStruct = std::mem::size_of::<SYMBOL_INFOW>() as u32;
        buf.info.MaxNameLen = NAME_MAX as u32;
        let mut disp = 0u64;
        if unsafe { SymFromAddrW(process, a, &mut disp, &mut buf.info) } != 0 {
            let n = (buf.info.NameLen as usize).min(NAME_MAX);
            // The name runs past SYMBOL_INFOW's one-element array into
            // `_room`; take the pointer from the whole buffer so it may.
            let name_ptr = unsafe {
                (&buf as *const SymBuf as *const u8).add(std::mem::offset_of!(SYMBOL_INFOW, Name)) as *const u16
            };
            let name = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(name_ptr, n) });
            line.push_str(&format!("  {}", super::demangle(&name)));
            let mut li: IMAGEHLP_LINEW64 = unsafe { std::mem::zeroed() };
            li.SizeOfStruct = std::mem::size_of::<IMAGEHLP_LINEW64>() as u32;
            let mut ldisp = 0u32;
            if unsafe { SymGetLineFromAddrW64(process, a, &mut ldisp, &mut li) } != 0 && !li.FileName.is_null() {
                let file = unsafe { wide_cstr(li.FileName) };
                line.push_str(&format!("\n             at {file}:{}", li.LineNumber));
            }
        }
        out.push(line);
    }
    out
}

/// A NUL-terminated UTF-16 string dbghelp owns.
unsafe fn wide_cstr(p: *const u16) -> String {
    let mut n = 0;
    // SAFETY: dbghelp's strings are NUL-terminated; capped anyway.
    while n < 4096 && unsafe { *p.add(n) } != 0 {
        n += 1;
    }
    String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(p, n) })
}
