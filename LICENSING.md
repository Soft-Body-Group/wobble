# Licensing

## Wobble

Wobble is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. The full text is in [LICENSE](LICENSE).

Wobble is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for details.

Copyright (C) 2026 The Wobble authors.

This covers every crate in this repository (`engine`, `app`, `api`,
`genai`, `colors`, `web`, `geoplayer`, `blender_native`) and the Blender
bridge script (`engine/src/blender/wobble_blender.py`). The patch to Blender
in `blender/patches/` is a modification of Blender and is licensed like
Blender, GPL-2.0-or-later.

`imfram`, the UI crate Wobble is built with, lives in its own repository
under the MIT licence, so its widgets can be used without the rest of
Wobble.

### Additional permission: plugins and addons

As an additional permission under section 7 of the GNU GPL version 3, you
may create and distribute plugins for Wobble under terms of your choice,
including proprietary terms, without those plugins being subject to the GNU
GPL. A plugin here means:

- a folder Wobble loads from its plugins directory (`plugin.json`,
  `tool.json` or `addon.json` and the WGSL, JSON and assets it names), and
- a separate program that drives Wobble only through its scripting protocol
  (the local socket API, or the `wobble-mcp` bridge), or through its
  remote-panel protocol (the meshes-out, pointer-in stream on the API's
  port + 1, `app/src/apiserve/remote.rs`).

This permission does not extend to modified versions of Wobble itself, or to
code linked into the Wobble binary. If you modify Wobble, you may remove this
permission from your version.

## Components shipped in Wobble builds

Every Rust crate a build links, with its licence text, is listed in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) (regenerate it with
`tools/third_party_licenses.sh`). All of them are compatible with the GPL.
Other components:

- **FFmpeg** (macOS builds with movie support): built with only its LGPL
  parts, licensed under the GNU Lesser General Public License version 2.1
  or later ([licenses/LGPL-2.1.txt](licenses/LGPL-2.1.txt)). No GPL-only
  FFmpeg parts and no x264 or x265 are included. FFmpeg's source is at
  <https://ffmpeg.org/download.html>; the exact version and configure line
  of a build are recorded in that build's archive.
- **OpenDRT** (only with the `opendrt` cargo feature): GPLv3, by Jed Smith.
  See `engine/src/viewtf/opendrt.rs`.
- **egui's default fonts**: SIL Open Font License 1.1 and the Ubuntu Font
  Licence, listed in THIRD_PARTY_LICENSES.md.
- **egui-winit**, vendored and modified in `vendor/egui-winit/`: MIT or
  Apache-2.0, with its licence files beside it.
- **Code ported from other projects** (the ACES 1.3 output transform and
  the neural networks re-implemented in WGSL: SEA-RAFT, RIFE, Real-ESRGAN,
  SPAN, HDRCNN): each upstream's notice is kept in [NOTICE.md](NOTICE.md).

## Name and logo

The GPL covers the code, not the Wobble name, logo or app icons. See
[TRADEMARKS.md](TRADEMARKS.md) for how they may be used.

## Not shipped with Wobble

These are fetched or run only when you click for them, and each comes with
its own licence, which Wobble shows before anything is downloaded:

- **Model weights** in the models catalog (`engine/src/models.rs`).
  Weights under research-only licences are tagged, and anything rendered
  through them is flagged. Weights that ask for credit are credited in
  [NOTICE.md](NOTICE.md).
- **CorridorKey**, installed from its own repository by the CorridorKey
  node. It is CC BY-NC-SA 4.0, with an explicit grant for commercial use of
  its output.
- **DiffusionLight-Turbo** (MIT), installed from its own repository by the
  DiffusionLight node, with the weights it runs on: Stable Diffusion XL and
  its depth ControlNet (CreativeML Open RAIL++-M), the DiffusionLight LoRAs
  (MIT) and DPT-Large (Apache-2.0).
- **Blender**, which Wobble launches as a separate program. Blender is
  GPL-2.0-or-later.
- **Remote generative-AI services**, which have their own terms of service.
