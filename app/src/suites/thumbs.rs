//! `--test-thumbs`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// Thumbnails and the type filter: a mixed directory shows only the
    /// decodable sequences by default, "all files" reveals the rest, and the
    /// grid decodes a thumbnail per sequence (not just the selected one).
    pub(crate) fn drive_test_thumbs(&mut self) {
        let Some((stage, mut frames)) = self.test_thumbs else { return };
        frames += 1;
        if frames > 2400 {
            println!("THUMBS TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let dir = std::env::temp_dir().join("wobble-thumbs-test");
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                // A mixed directory: two exr sequences plus files we can't
                // decode.
                let _ = std::fs::remove_dir_all(&dir);
                if std::fs::create_dir_all(&dir).is_err() {
                    println!("THUMBS TEST FAIL: could not make {}", dir.display());
                    std::process::exit(1);
                }
                let src = std::path::Path::new("data/mix");
                let mut copied = 0;
                for e in std::fs::read_dir(src).into_iter().flatten().flatten() {
                    let p = e.path();
                    if engine::decode::is_supported(&p) {
                        if let Some(n) = p.file_name() {
                            if std::fs::copy(&p, dir.join(n)).is_ok() {
                                copied += 1;
                            }
                        }
                    }
                }
                let _ = std::fs::write(dir.join("notes.txt"), b"not an image");
                // An image extension we still do not read. This used to be
                // a .jpg — the point was always "looks like media, is not
                // decodable", and jpg stopped qualifying once we gained a
                // JPEG reader.
                let _ = std::fs::write(dir.join("ref.psd"), b"not decodable either");
                // A REAL still, so the filter is checked both ways round:
                // it must hide what we cannot decode AND show what we can.
                if let Ok(file) = std::fs::File::create(dir.join("still.png")) {
                    let mut enc = png::Encoder::new(file, 8, 8);
                    enc.set_color(png::ColorType::Rgba);
                    enc.set_depth(png::BitDepth::Eight);
                    let _ = enc
                        .write_header()
                        .and_then(|mut w| w.write_image_data(&[128u8; 8 * 8 * 4]));
                }
                if copied == 0 {
                    println!("THUMBS TEST FAIL: no exrs copied from data/mix");
                    std::process::exit(1);
                }
                self.browser = Some(browser::SeqBrowser::new(
                    None,
                    dir.clone(),
                    self.decoder.clone(),
                ));
                if let Some(b) = &mut self.browser {
                    b.test_set_grid(true);
                    b.test_set_tile(220.0);
                }
                self.test_thumbs = Some((1, frames));
            }
            // Default filter: the exr sequences and the png, none of the
            // junk.
            1 if frames > 20 => {
                let Some(b) = &self.browser else {
                    println!("THUMBS TEST FAIL: browser closed");
                    std::process::exit(1);
                };
                let (seqs, other, _) = b.test_counts();
                if seqs != 3 || other != 0 {
                    println!(
                        "THUMBS TEST FAIL: default filter showed {seqs} sequences + {other} loose files, want 3 + 0"
                    );
                    std::process::exit(1);
                }
                self.browser.as_mut().unwrap().test_set_filter(browser::TypeFilter::All);
                self.test_thumbs = Some((2, frames));
            }
            // "all files" reveals the two undecodable ones.
            2 if frames > 30 => {
                let (seqs, other, _) = self.browser.as_ref().unwrap().test_counts();
                if other != 2 {
                    println!(
                        "THUMBS TEST FAIL: 'all files' showed {other} loose files, want 2 (seqs {seqs})"
                    );
                    std::process::exit(1);
                }
                self.browser
                    .as_mut()
                    .unwrap()
                    .test_set_filter(browser::TypeFilter::Supported);
                self.test_thumbs = Some((3, frames));
            }
            // The grid decodes a thumbnail per sequence, not one at a time.
            3 => {
                let (_, _, thumbs) = self.browser.as_ref().unwrap().test_counts();
                if thumbs >= 2 {
                    // Tiles honour the size control (and clamp sanely).
                    let b = self.browser.as_mut().unwrap();
                    let tile = b.test_tile();
                    b.test_set_tile(9999.0);
                    self.test_thumbs = Some((4, frames));
                    log::info!("thumbs: {thumbs} decoded at tile {tile}");
                    return;
                }
                self.test_thumbs = Some((3, frames));
            }
            4 if frames > 10 => {
                let b = self.browser.as_ref().unwrap();
                let (_, _, thumbs) = b.test_counts();
                let tile = b.test_tile();
                if tile > browser::TILE_MAX {
                    println!("THUMBS TEST FAIL: tile size {tile} exceeds the max");
                    std::process::exit(1);
                }
                let _ = std::fs::remove_dir_all(&dir);
                println!(
                    "THUMBS TEST PASS: {thumbs} thumbnails decoded in the tile view (tile {tile:.0}pt); default filter showed the exr sequences + a png and hid 2 undecodable files, 'all files' showed them"
                );
                std::process::exit(0);
            }
            _ => self.test_thumbs = Some((stage, frames)),
        }
    }
}
