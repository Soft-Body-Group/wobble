//! Scrub as an app of its own: its icon, and no stray console.
//!
//! The icon files live in `scrub/app/icons/` (drawn by
//! `tools/make_scrub_icons.py`): the `.icns` goes into Scrub.app
//! (`tools/bundle_scrub.sh`), the `.ico` into scrub.exe (`build.rs`), the
//! PNGs into the Linux desktop entry (`app/icons/install-desktop-entry.sh`).
//! At run time the window gets the 256px PNG (Windows title bar + taskbar,
//! X11), and a bare macOS binary sets it as the Dock image; a `.app` gets
//! its icon from the bundle and is left alone.

/// The 256px icon, embedded for the window.
static PNG_256: &[u8] = include_bytes!("../icons/scrub-256.png");

fn decode(bytes: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let mut r = png::Decoder::new(std::io::Cursor::new(bytes)).read_info().ok()?;
    let mut buf = vec![0; r.output_buffer_size()?];
    let info = r.next_frame(&mut buf).ok()?;
    (info.color_type == png::ColorType::Rgba && info.bit_depth == png::BitDepth::Eight)
        .then(|| (info.width, info.height, buf[..info.buffer_size()].to_vec()))
}

/// Give the window its icon (macOS and Wayland ignore a window icon; the
/// Dock gets one from [`set_dock_icon`]).
pub fn apply(window: &winit::window::Window) {
    if let Some((w, h, px)) = decode(PNG_256) {
        window.set_window_icon(winit::window::Icon::from_rgba(px, w, h).ok());
    }
    set_dock_icon();
}

/// A bare binary (`target/release/scrub`) has no bundle for the Dock to read
/// an icon from; hand it the PNG. Call once NSApplication exists.
#[cfg(target_os = "macos")]
fn set_dock_icon() {
    use objc2::msg_send;
    use objc2::runtime::{AnyClass, AnyObject};
    if std::env::current_exe().is_ok_and(|p| p.to_string_lossy().contains(".app/Contents/MacOS/")) {
        return;
    }
    let (Some(app), Some(data), Some(image)) =
        (AnyClass::get(c"NSApplication"), AnyClass::get(c"NSData"), AnyClass::get(c"NSImage"))
    else {
        return;
    };
    unsafe {
        let app: *mut AnyObject = msg_send![app, sharedApplication];
        let bytes: *mut AnyObject =
            msg_send![data, dataWithBytes: PNG_256.as_ptr().cast::<std::ffi::c_void>(), length: PNG_256.len()];
        let img: *mut AnyObject = msg_send![image, alloc];
        let img: *mut AnyObject = msg_send![img, initWithData: bytes];
        if !app.is_null() && !img.is_null() {
            let _: () = msg_send![app, setApplicationIconImage: img];
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn set_dock_icon() {}

/// Windows: scrub.exe is a console program (its CLI modes print), so a
/// double-click from Explorer or the Start menu opens a console beside the
/// window. When no one else shares that console (nothing launched us from a
/// terminal), let it go: it closes, and the window stands alone. From a
/// terminal the console is shared and stays, output and all.
#[cfg(windows)]
pub fn drop_lonely_console() {
    use windows::Win32::System::Console::{FreeConsole, GetConsoleProcessList};
    let mut ids = [0u32; 4];
    let n = unsafe { GetConsoleProcessList(&mut ids) };
    if n == 1 {
        let _ = unsafe { FreeConsole() };
    }
}

#[cfg(not(windows))]
pub fn drop_lonely_console() {}

#[cfg(test)]
mod tests {
    #[test]
    fn the_embedded_icon_decodes() {
        let (w, h, px) = super::decode(super::PNG_256).expect("an 8-bit RGBA PNG");
        assert_eq!((w, h), (256, 256));
        // Transparent corner, opaque middle: it is a tile, not a square.
        assert_eq!(px[3], 0);
        assert_eq!(px[((128 * 256 + 128) * 4 + 3) as usize], 255);
    }
}
