//! What a saved clip gets laid over it: the pointer (the OS cursor is never
//! in the surface, so a clip without it is a UI moving by itself) and the
//! "Made with Scrub" mark. Both are CPU sprites, made once and cached.

use std::sync::Mutex;

pub const MARK: &str = "Made with Scrub";

/// A coverage sprite: white `fill` over a black `edge`, both 0..255.
struct Sprite {
    w: usize,
    h: usize,
    fill: Vec<u8>,
    edge: Vec<u8>,
}

static CURSOR: Mutex<Option<(u32, Sprite)>> = Mutex::new(None);
static MARK_SPRITE: Mutex<Option<(u32, Sprite)>> = Mutex::new(None);

/// Lay the overlays onto an RGBA frame in place.
pub fn apply(px: &mut [u8], w: u32, h: u32, cursor: Option<[f32; 2]>, px_per_pt: f32, mark: bool) {
    if mark {
        let size = ((h as f32 * 0.026).round() as u32).max(11);
        let mut g = MARK_SPRITE.lock().unwrap();
        if g.as_ref().map(|(s, _)| *s) != Some(size) {
            *g = Some((size, mark_sprite(size as f32)));
        }
        let s = &g.as_ref().unwrap().1;
        let margin = (size as f32 * 0.9) as i64;
        let x = w as i64 - s.w as i64 - margin;
        let y = h as i64 - s.h as i64 - margin * 2 / 3;
        blit(px, w, h, s, x, y, 0.8);
    }
    if let Some([cx, cy]) = cursor {
        let scale = ((px_per_pt * 4.0).round() as u32).max(2);
        let mut g = CURSOR.lock().unwrap();
        if g.as_ref().map(|(s, _)| *s) != Some(scale) {
            *g = Some((scale, cursor_sprite(scale as f32 / 4.0)));
        }
        let s = &g.as_ref().unwrap().1;
        // The arrow's tip is the sprite's (1, 1): the outline's margin.
        let pad = (scale as f32 / 4.0).round().max(1.0) as i64;
        blit(px, w, h, s, cx.round() as i64 - pad, cy.round() as i64 - pad, 1.0);
    }
}

fn blit(px: &mut [u8], w: u32, h: u32, s: &Sprite, x0: i64, y0: i64, alpha: f32) {
    for sy in 0..s.h {
        let y = y0 + sy as i64;
        if y < 0 || y >= h as i64 {
            continue;
        }
        for sx in 0..s.w {
            let x = x0 + sx as i64;
            if x < 0 || x >= w as i64 {
                continue;
            }
            let i = sy * s.w + sx;
            let (e, f) = (s.edge[i] as f32 / 255.0 * alpha, s.fill[i] as f32 / 255.0 * alpha);
            if e == 0.0 && f == 0.0 {
                continue;
            }
            let o = ((y as u32 * w + x as u32) * 4) as usize;
            for c in 0..3 {
                let mut v = px[o + c] as f32;
                v *= 1.0 - e * 0.6; // the edge is a soft shade, not ink
                v = v * (1.0 - f) + 255.0 * f;
                px[o + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Max filter of radius `r` (a square): the shadow / outline.
fn dilate(src: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    let mut out = vec![0u8; src.len()];
    for y in 0..h {
        for x in 0..w {
            let mut m = 0;
            for yy in y.saturating_sub(r)..(y + r + 1).min(h) {
                for xx in x.saturating_sub(r)..(x + r + 1).min(w) {
                    m = m.max(src[yy * w + xx]);
                }
            }
            out[y * w + x] = m;
        }
    }
    out
}

fn mark_sprite(size: f32) -> Sprite {
    use ab_glyph::{point, Font, FontRef, PxScale, ScaleFont};
    let font = FontRef::try_from_slice(epaint_default_fonts::UBUNTU_LIGHT).expect("bundled font");
    let sf = font.as_scaled(PxScale::from(size));
    let pad = 2usize;
    let mut glyphs = Vec::new();
    let mut caret = 0.0f32;
    let mut prev = None;
    for c in MARK.chars() {
        let id = font.glyph_id(c);
        if let Some(p) = prev {
            caret += sf.kern(p, id);
        }
        glyphs.push(id.with_scale_and_position(sf.scale(), point(caret, sf.ascent())));
        caret += sf.h_advance(id);
        prev = Some(id);
    }
    let w = caret.ceil() as usize + pad * 2;
    let h = (sf.ascent() - sf.descent()).ceil() as usize + pad * 2;
    let mut fill = vec![0u8; w * h];
    for g in glyphs {
        if let Some(og) = font.outline_glyph(g) {
            let b = og.px_bounds();
            og.draw(|x, y, cov| {
                let xx = b.min.x as i64 + x as i64 + pad as i64;
                let yy = b.min.y as i64 + y as i64 + pad as i64;
                if xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h {
                    let i = yy as usize * w + xx as usize;
                    fill[i] = fill[i].max((cov * 255.0).round() as u8);
                }
            });
        }
    }
    let edge = dilate(&fill, w, h, 1);
    Sprite { w, h, fill, edge }
}

/// The classic arrow, `s` output pixels to the point.
fn cursor_sprite(s: f32) -> Sprite {
    const ARROW: [(f32, f32); 7] =
        [(0.0, 0.0), (0.0, 16.5), (4.0, 12.8), (6.9, 19.2), (9.4, 18.1), (6.6, 11.8), (11.6, 11.8)];
    let pad = s.round().max(1.0) as usize;
    let w = (12.0 * s).ceil() as usize + pad * 2 + 1;
    let h = (20.0 * s).ceil() as usize + pad * 2 + 1;
    let inside = |x: f32, y: f32| {
        let mut c = false;
        let n = ARROW.len();
        for i in 0..n {
            let (ax, ay) = (ARROW[i].0 * s, ARROW[i].1 * s);
            let (bx, by) = (ARROW[(i + 1) % n].0 * s, ARROW[(i + 1) % n].1 * s);
            if (ay > y) != (by > y) && x < (bx - ax) * (y - ay) / (by - ay) + ax {
                c = !c;
            }
        }
        c
    };
    let mut fill = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut n = 0;
            for sy in 0..4 {
                for sx in 0..4 {
                    let fx = x as f32 - pad as f32 + (sx as f32 + 0.5) / 4.0;
                    let fy = y as f32 - pad as f32 + (sy as f32 + 0.5) / 4.0;
                    n += inside(fx, fy) as u32;
                }
            }
            fill[y * w + x] = (n * 255 / 16) as u8;
        }
    }
    let mut edge = dilate(&fill, w, h, pad);
    // The outline is ink, not a shade: full strength.
    for e in &mut edge {
        *e = ((*e as u32 * 5 / 3).min(255)) as u8;
    }
    Sprite { w, h, fill, edge }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mark_lands_bottom_right_and_the_cursor_where_it_was() {
        let (w, h) = (640u32, 360u32);
        let mut px = vec![40u8; (w * h * 4) as usize];
        apply(&mut px, w, h, Some([100.0, 100.0]), 1.0, true);
        let at = |x: u32, y: u32| px[((y * w + x) * 4) as usize];
        // Somewhere in the lower-right corner is now brighter than the plate.
        let lit = (w / 2..w).flat_map(|x| (h * 3 / 4..h).map(move |y| (x, y))).filter(|&(x, y)| at(x, y) > 120).count();
        assert!(lit > 50, "{lit} bright pixels in the mark's corner");
        // The top-left quarter (bar the cursor) is untouched.
        assert_eq!(at(20, 20), 40);
        // Just inside the arrow's tip is white; the tip's outline dark.
        assert!(at(102, 108) > 200, "{}", at(102, 108));
        assert!(at(99, 99) < 40);
    }
}
