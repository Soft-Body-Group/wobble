//! A 1920x1080 half-float EXR of 6x4 flat colour patches (scene-linear,
//! some past 1.0), for checking colour management against wobble.
fn main() {
    use exr::prelude::*;
    let out = std::env::args().nth(1).expect("out path");
    let pal: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 0.0], [0.01, 0.01, 0.01], [0.05, 0.05, 0.05], [0.18, 0.18, 0.18], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [0.8, 0.1, 0.05], [0.05, 0.6, 0.1], [0.05, 0.1, 0.7], [0.9, 0.8, 0.1],
        [0.1, 0.7, 0.8], [0.7, 0.1, 0.7], [3.0, 0.5, 0.2], [0.2, 2.5, 0.4], [0.3, 0.4, 3.5], [0.25, 0.15, 0.1],
        [0.12, 0.2, 0.3], [0.6, 0.45, 0.35], [8.0, 6.0, 4.0], [0.02, 0.05, 0.01], [1.5, 0.05, 0.05], [0.05, 0.05, 1.5],
    ];
    let (w, h) = (1920usize, 1080usize);
    let ch = SpecificChannels::rgb(|p: Vec2<usize>| {
        let (cx, cy) = (p.x() * 6 / w, p.y() * 4 / h);
        let c = pal[cy * 6 + cx];
        (f16::from_f32(c[0]), f16::from_f32(c[1]), f16::from_f32(c[2]))
    });
    Image::from_channels((w, h), ch).write().to_file(&out).expect("write");
}
