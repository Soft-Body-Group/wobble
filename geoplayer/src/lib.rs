//! A 2dini drawing evaluated live in a web page — what a "play forever"
//! web export embeds. Plain C-ABI exports, no wasm-bindgen: the page
//! instantiates the module with no imports and talks to it through linear
//! memory.
//!
//! - `alloc(n)` → a buffer for the setup JSON; `load(ptr, len)` parses it
//!   (`{"geo": GeoParams, "noise": NoiseParams|null, "w": .., "h": ..}`).
//! - `frame(t)` evaluates at tick `t` (fractional: noise moves smoothly)
//!   and returns how many i16s it packed at `data_ptr()`, in
//!   `engine::geoweb`'s frame format; that frame's styles are JSON at
//!   `styles_ptr()` / `styles_len()`.
//!
//! The geometry is the layer's: the same evaluator, the same context
//! (seed, aspect, pixels per unit, the field port's Noise at time `t`).

use std::cell::RefCell;

use engine::geo::{eval, field_cx_for, EvalCx};
use engine::geoweb::{pack_frame, Palette};
use engine::graph::{GeoParams, NoiseParams};

#[derive(serde::Deserialize)]
struct Setup {
    geo: GeoParams,
    noise: Option<NoiseParams>,
    w: u32,
    h: u32,
}

#[derive(Default)]
struct State {
    setup: Option<Setup>,
    data: Vec<i16>,
    styles: String,
    pal: Palette,
}

thread_local! {
    static S: RefCell<State> = RefCell::new(State::default());
}

/// A buffer the page writes the setup into (never freed: one per load).
#[no_mangle]
pub extern "C" fn alloc(n: usize) -> *mut u8 {
    let mut v = vec![0u8; n.max(1)];
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// Parse the setup at `ptr`; 1 on success.
///
/// # Safety
/// `ptr..ptr+len` must be a buffer `alloc` handed out.
#[no_mangle]
pub unsafe extern "C" fn load(ptr: *const u8, len: usize) -> i32 {
    let bytes = std::slice::from_raw_parts(ptr, len);
    let Ok(setup) = serde_json::from_slice::<Setup>(bytes) else { return 0 };
    S.with(|s| s.borrow_mut().setup = Some(setup));
    1
}

/// Evaluate at tick `t`; the packed length in i16s.
#[no_mangle]
pub extern "C" fn frame(t: f32) -> usize {
    S.with(|s| {
        let s = &mut *s.borrow_mut();
        let Some(setup) = &s.setup else { return 0 };
        let (w, h) = (setup.w.max(1), setup.h.max(1));
        let field = setup.noise.as_ref().map(|n| {
            let mut n = n.clone();
            n.time = t;
            field_cx_for(&n)
        });
        let cx = EvalCx {
            tick: t,
            field: field.as_ref(),
            seed: setup.geo.seed,
            aspect: h as f32 / w as f32,
            px_per_unit: w as f32,
        };
        let geo = eval(&setup.geo.graph, &cx);
        s.data.clear();
        s.pal.clear();
        pack_frame(&geo, w as f32, &mut s.pal, &mut s.data);
        s.styles = s.pal.json();
        s.data.len()
    })
}

#[no_mangle]
pub extern "C" fn data_ptr() -> *const i16 {
    S.with(|s| s.borrow().data.as_ptr())
}

#[no_mangle]
pub extern "C" fn styles_ptr() -> *const u8 {
    S.with(|s| s.borrow().styles.as_ptr())
}

#[no_mangle]
pub extern "C" fn styles_len() -> usize {
    S.with(|s| s.borrow().styles.len())
}
