//! `wobble://` links reaching the app.
//!
//! macOS delivers a URL to a running (or launching) app as a GetURL Apple
//! Event, and winit does not surface it, so we register our own handler on
//! `NSAppleEventManager` before the event loop runs. Elsewhere the OS
//! launches the binary with the URL as an argument, which `main` hands to
//! `push`. Either way the URL waits in a queue the plugin store drains once
//! per UI pass — the app redraws continuously, so nothing needs waking.
//!
//! The scheme only exists for a BUNDLED app (`tools/bundle_app.sh` writes
//! the `CFBundleURLTypes` entry); a bare `cargo run` binary never gets one.

use std::sync::Mutex;

static QUEUE: Mutex<Vec<String>> = Mutex::new(Vec::new());

pub fn push(url: String) {
    QUEUE.lock().unwrap().push(url);
}

pub fn drain() -> Vec<String> {
    std::mem::take(&mut *QUEUE.lock().unwrap())
}

#[cfg(target_os = "macos")]
pub fn install() {
    use objc2::rc::Retained;
    use objc2::runtime::{AnyClass, AnyObject, NSObject};
    use objc2::{define_class, msg_send, sel, AllocAnyThread};
    use std::ffi::{c_char, CStr};

    /// 'GURL' — both the event class (kInternetEventClass) and id (kAEGetURL).
    const GURL: u32 = u32::from_be_bytes(*b"GURL");
    /// '----' — keyDirectObject, where the URL string sits.
    const DIRECT: u32 = u32::from_be_bytes(*b"----");

    define_class!(
        #[unsafe(super(NSObject))]
        #[name = "WobbleUrlHandler"]
        struct Handler;

        impl Handler {
            #[unsafe(method(handleGetURLEvent:withReplyEvent:))]
            fn handle(&self, event: &AnyObject, _reply: &AnyObject) {
                unsafe {
                    let d: *mut AnyObject = msg_send![event, paramDescriptorForKeyword: DIRECT];
                    let Some(d) = d.as_ref() else { return };
                    let s: *mut AnyObject = msg_send![d, stringValue];
                    let Some(s) = s.as_ref() else { return };
                    let p: *const c_char = msg_send![s, UTF8String];
                    if !p.is_null() {
                        push(CStr::from_ptr(p).to_string_lossy().into_owned());
                    }
                }
            }
        }
    );

    // Registering twice is harmless (it replaces), so this runs before the
    // loop AND again once it is up, in case AppKit's launch installed its
    // own GetURL handler over ours.
    static HANDLER: Mutex<Option<usize>> = Mutex::new(None);
    let mut h = HANDLER.lock().unwrap();
    let ptr = *h.get_or_insert_with(|| {
        let obj: Retained<Handler> = unsafe { msg_send![Handler::alloc(), init] };
        // The event manager does not retain its handler: this one lives
        // for the process.
        Retained::into_raw(obj) as usize
    });
    let Some(cls) = AnyClass::get(c"NSAppleEventManager") else { return };
    unsafe {
        let mgr: *mut AnyObject = msg_send![cls, sharedAppleEventManager];
        let Some(mgr) = mgr.as_ref() else { return };
        let handler = &*(ptr as *const AnyObject);
        let _: () = msg_send![
            mgr,
            setEventHandler: handler,
            andSelector: sel!(handleGetURLEvent:withReplyEvent:),
            forEventClass: GURL,
            andEventID: GURL
        ];
    }
}

#[cfg(not(target_os = "macos"))]
pub fn install() {}
