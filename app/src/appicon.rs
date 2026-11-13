//! The app icon: the Read's TV cabinet tuned to nothing (static), filling
//! the tile. ONE drawing, `paint`, is the source of every size: the files in
//! `app/icons/` (macOS `.icns`, Windows `.ico`, Linux PNGs) are rendered
//! from it by `tools/make_icons.sh`, each size drawn natively rather than
//! scaled down from the largest.
//!
//! At run time the window gets the 256px PNG (Windows title bar + taskbar,
//! X11), and a bare macOS binary sets it as the Dock image — a `.app` gets
//! its icon from the bundle's `.icns` and is left alone.

#[cfg(test)]
use egui::{Rect, Stroke, Vec2};

/// The drawing's own size in points: 1024, so egui tessellates the curves
/// finely. The icon is the same picture at every pixel size —
/// `pixels_per_point` is the only thing that changes.
#[cfg(test)]
pub const ICON_PT: f32 = 1024.0;

/// Draw the icon into the `ICON_PT`-square `area`.
#[cfg(test)]
pub fn paint(painter: &egui::Painter, area: Rect) {
    let k = area.width() / 1024.0;
    // The cabinet IS the tile: Apple's grid, an 824 square centred on a
    // 1024 canvas with ~185 corners. No antennas, no feet.
    let tile = Rect::from_center_size(area.center(), Vec2::splat(824.0 * k));
    // The graph's glass is laid out for a 4:3 cabinet; on a square one its
    // bezel would spill off the left edge, so the icon lays out its own.
    let t = tile.width();
    let screen = Rect::from_min_max(
        tile.min + Vec2::splat(0.11 * t),
        tile.max - Vec2::new(0.27 * t, 0.11 * t),
    );
    crate::tv::body(
        painter,
        tile,
        screen,
        185.0 * k,
        crate::theme::node_read(),
        Stroke::NONE,
        None,
        None,
        SEED,
        0.0,
        // ~10 cells across: chunky enough to read as static at 16 px.
        50.0 * k,
        false,
    )
}

/// The static's shuffle, picked by eye.
#[cfg(test)]
const SEED: u64 = 7;

/// The 256px icon, embedded for the window.
#[cfg(not(target_arch = "wasm32"))]
static PNG_256: &[u8] = include_bytes!("../icons/wobble-256.png");

#[cfg(not(target_arch = "wasm32"))]
fn decode(bytes: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let mut r = png::Decoder::new(std::io::Cursor::new(bytes)).read_info().ok()?;
    let mut buf = vec![0; r.output_buffer_size()?];
    let info = r.next_frame(&mut buf).ok()?;
    (info.color_type == png::ColorType::Rgba && info.bit_depth == png::BitDepth::Eight)
        .then(|| (info.width, info.height, buf[..info.buffer_size()].to_vec()))
}

/// The window icon (ignored by macOS and Wayland, which take it elsewhere).
#[cfg(not(target_arch = "wasm32"))]
pub fn window_icon() -> Option<winit::window::Icon> {
    let (w, h, px) = decode(PNG_256)?;
    winit::window::Icon::from_rgba(px, w, h).ok()
}

/// A bare binary (`target/release/wobble`) has no bundle for the Dock to
/// read an icon from; hand it the PNG. Call once NSApplication exists.
#[cfg(target_os = "macos")]
pub fn set_dock_icon() {
    use objc2::runtime::{AnyClass, AnyObject};
    use objc2::msg_send;
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
pub fn set_dock_icon() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_icon_decodes() {
        let (w, h, px) = decode(PNG_256).expect("an 8-bit RGBA PNG");
        assert_eq!((w, h), (256, 256));
        // Transparent corner, opaque middle: it is a tile, not a square.
        assert_eq!(px[3], 0);
        assert_eq!(px[((128 * 256 + 128) * 4 + 3) as usize], 255);
    }

    /// `tools/make_icons.sh` runs this: every size of the icon into
    /// `app/icons/` (PNGs, the `.ico`, and `wobble.iconset/` for iconutil).
    #[test]
    #[ignore]
    fn app_icon_render() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("icons");
        let set = dir.join("wobble.iconset");
        std::fs::create_dir_all(&set).unwrap();
        let mut ico = Vec::new();
        for px in [16u32, 24, 32, 48, 64, 128, 256, 512, 1024] {
            let Some((w, h, rgba)) = crate::uishot::render_rgba(ICON_PT, ICON_PT, px as f32 / ICON_PT, 2, |ui, _| {
                paint(ui.painter(), ui.max_rect());
            }) else {
                println!("SKIP: no adapter");
                return;
            };
            assert_eq!((w, h), (px, px));
            let mut bytes = Vec::new();
            {
                let mut enc = png::Encoder::new(&mut bytes, w, h);
                enc.set_color(png::ColorType::Rgba);
                enc.set_depth(png::BitDepth::Eight);
                enc.write_header().unwrap().write_image_data(&rgba).unwrap();
            }
            std::fs::write(dir.join(format!("wobble-{px}.png")), &bytes).unwrap();
            for (n, scale) in [(px, 1), (px / 2, 2)] {
                if [16, 32, 128, 256, 512].contains(&n) && n * scale == px {
                    let name = if scale == 1 { format!("icon_{n}x{n}.png") } else { format!("icon_{n}x{n}@2x.png") };
                    std::fs::write(set.join(name), &bytes).unwrap();
                }
            }
            if px <= 256 {
                ico.push((px, bytes));
            }
        }
        // ICO with PNG entries (Vista on): header, a directory, the PNGs.
        let mut out = vec![0u8, 0, 1, 0];
        out.extend((ico.len() as u16).to_le_bytes());
        let mut off = 6 + 16 * ico.len() as u32;
        for (px, b) in &ico {
            let d = if *px >= 256 { 0 } else { *px as u8 };
            out.extend([d, d, 0, 0]);
            out.extend(1u16.to_le_bytes());
            out.extend(32u16.to_le_bytes());
            out.extend((b.len() as u32).to_le_bytes());
            out.extend(off.to_le_bytes());
            off += b.len() as u32;
        }
        for (_, b) in &ico {
            out.extend(b);
        }
        std::fs::write(dir.join("wobble.ico"), out).unwrap();
        eprintln!("wrote {}", dir.display());
    }
}
