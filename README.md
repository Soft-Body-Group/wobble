# wobble

GPU compositing, (soon) open source.

A node-based 2D compositor written in Rust on wgpu.

- **The whole comp is one GPU pass.** The node graph compiles to a single
  fused shader, so a stack of grades, keys and transforms costs about what
  one does. Frames stay on the GPU.
- **Real-time at real resolutions.** 2K EXR plays and scrubs at display
  rate; 4K and 8K are designed in, with GPU, RAM and NVMe caches that scale
  down rather than fail.
- **You can see what costs time.** Fast wires are cyan, slow ones red, a
  node with its own pass glows. A Bake clears it. Nothing is slow silently.
- **Scene-linear with a proper working space** (ACEScg or Rec.709),
  transforms at Load and Render, a readout that labels scene vs display
  values.
- **Direct manipulation everywhere.** Gizmos on the picture, HUD sliders,
  held-key gestures, rate keys, remappable hotkeys, Blender's navigation
  chords.
- **Blender as the 3D system, live.** Scrub in one and the other follows;
  Blender's viewport or Cycles frame lands in the comp over shared memory,
  plates go back the other way.
- **Trackers and solvers built in.** Point, planar and mesh tracking,
  stabilisation, a 3D camera solve, optical flow driving motion blur, warps
  and retiming.
- **ML nodes as explicit, cached jobs**: neural keying, click-to-roto,
  depth, upscaling, HDR reconstruction. Weights download only on request,
  licences are shown, non-commercial models are flagged through to the
  Render.
- **Procedural toolkits inside the comp**: 2dini vector graphics with light
  and fluid sims, SDF effects, noise fields, 3D cards.
- **Extensible from outside the build.** A node is a JSON manifest plus a
  WGSL expression; subwobs package a graph as a node; addons drive the app
  over a JSON API, with a Python client and an MCP bridge.
- **Runs in a browser too** (WebGPU), and will be GPL open source.

## Download

Builds for macOS, Windows and Linux: [download.wobble.ing](https://download.wobble.ing).
Guide, quick start and a browser demo: [wobble.ing](https://wobble.ing).

## Build

```sh
cargo build --release
./target/release/wobble
```

Add `-p app --features app/movie` for ProRes/H.264 (needs a system ffmpeg).
Test media is generated, not committed:

```sh
target/release/gen_exr data/2k_zip_bg 2048 1080 24 zip bg
./target/release/wobble data/2k_zip_bg
```

The full user and architecture reference is in [docs/reference.md](docs/reference.md).

## License

GPL-3.0-or-later, with an additional permission for plugins and addons.
See [LICENSE](LICENSE) and [LICENSING.md](LICENSING.md).
