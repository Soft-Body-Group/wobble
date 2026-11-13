# Spektrafilm wobbin

Film stock emulation for wobble, from [spektrafilm](https://github.com/andreavolpato/spektrafilm)
by Andrea Volpato, through the Rust port spektrafilm-rs. Pick a negative (or slide) stock and a
print paper; the spectral film -> print -> scan chain is baked into a 3D LUT and lands on a
**Shader node** in the comp.

- **exposure** and **mix** are the node's own params: uniforms, keyable, never a re-bake.
- Stock, paper, print exposure, filtration, contrast and DIR couplers re-bake (about 0.2 s).
- The recipe rides in the node's source (`// spektrafilm: {...}`), so a saved comp keeps it, and
  **edit selected node** reads it back into the panel.
- Not included: grain, halation, diffusion, glare. They are spatial or random, so no LUT holds
  them; use wobble's Grain and Glow nodes after the film node.
- The output is the print as scanned (display-referred, linear in the working space). View it with
  the viewer's transform set to none (or standard) to see the print as printed.

## How it is built

`lutgen/` is a small Rust program, `spektra-lut`, that links spektrafilm-rs and writes the
shader. The LUT's input is the working space, log2-shaped per channel over -12..+6 stops (the
print has shouldered well before +6); its output is stored as `v^(1/2.4)` in half floats and read
back with tetrahedral interpolation. Against the chain itself, on random plate colours, a 33³
table lands 0.3-0.4 sRGB code values off on average, 2-6 at worst on saturated colours
(`spektra-lut check`). It is one pass on the GPU, so the node is red (a slow path) like any
Shader node.

```sh
cd plugins/spektrafilm/lutgen && cargo build --release     # or the panel's "build the baker"
target/release/spektra-lut list  --data ~/src/devl/spektrafilm-rs/data
target/release/spektra-lut check --data ~/src/devl/spektrafilm-rs/data --recipe r.json
cd .. && python3 -m unittest test_core
```

It expects the spektrafilm-rs checkout next to the wobble checkout (`../spektrafilm-rs`, branch
`spektro-experimental`), the same one spektrorganize builds against. `$SPEKTRAFILM_DATA`
points it at another data directory.

## Licence

spektrafilm and spektrafilm-rs are GPL-3.0, and the film and paper profiles ship inside that
repository under the same licence. The baker is GPL-3.0 too and lives in its own Cargo workspace
on purpose: wobble's own binaries never link spektrafilm code. A baked LUT is the program's
output, generated from the GPL profile data; whether shipping baked LUTs in a non-GPL product is
allowed is a question for the upstream author, not something this wobbin decides.
