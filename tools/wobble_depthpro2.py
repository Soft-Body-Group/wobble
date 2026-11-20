"""wobble's driver for Depth Pro 2 (Apple,
github.com/apple-aiml-research/ml-depth-pro-2; code under Apple's sample
licence, WEIGHTS under the Apple Machine Learning Research Model licence:
research only, no commercial use).

Written next to an installed checkout by the app on every run, together
with `wobble_depth.py` (whose EXR reader and writer it shares), and
executed from that directory with the checkout's own `uv` environment:

    uv run --project <checkout> python wobble_depthpro2.py --shot <dir> \
        --weights <depthpro2-s.safetensors> --focal <depthpro2-focal.safetensors> \
        --preset depthpro2-s [--size 1088]

The weights are NOT fetched here: the app downloads them through its own
models catalog (pinned by hash) and passes the paths. Nothing in this file
touches the network.

The shot directory is what wobble rendered: `Input/f_NNNNN.exr`, linear
Rec.709, premultiplied. The network is display-referred, so each frame is
clamped to 0..1 and sRGB-encoded to 8 bits on the way in (what the tool's
own transform takes), then shrunk to `--size` on its short side. The clip
goes through the tool's `infer_video` — overlapping 16-frame windows,
cross-faded, one focal length estimated from the first window for the
whole clip — which reads frames lazily and hands each prediction back once,
in order, so memory does not grow with the clip.

Each result frame lands in `Depth/f_NNNNN.exr`, same names, in wobble's
depth layout (`app::depth::LAYOUT`): ONE half-float channel `Z`, metres
along the optical axis, capped at FAR (the sky is infinitely far), scaled
back to the plate's size. `Depth/meta.json`:

    {"layout": 2, "top": <disparity clip maximum, 99.9th pct>, "metric": true}

Progress is one machine-readable line per event on stdout:

    WOBBLE frame <done> <total>
    WOBBLE done
    WOBBLE error <message>

Everything else printed is shown as a log tail.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# The shared reader / writer sits beside this file; the tool's package is
# installed into the environment.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wobble_depth import (  # noqa: E402
    FAR,
    LAYOUT,
    Unreadable,
    first_line,
    free_gpu,
    out_of_memory,
    read_rgb,
    say,
    srgb_encode,
    vram_words,
    write_z,
)

# The smallest short side the driver steps down to on running out of memory.
SMALLEST = 256


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--focal", required=True)
    ap.add_argument("--preset", default="depthpro2-s")
    ap.add_argument("--size", type=int, default=1088, help="network input short side, a multiple of 32")
    a = ap.parse_args()

    import json
    import time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch
    import torch.nn.functional as F

    inputs = sorted(glob.glob(os.path.join(a.shot, "Input", "*.exr")))
    if not inputs:
        say("error", "no Input frames")
        return 1
    out_dir = os.path.join(a.shot, "Depth")
    os.makedirs(out_dir, exist_ok=True)
    total = len(inputs)

    if torch.cuda.is_available():
        device, where = "cuda", torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device, where = "mps", "Apple Silicon (MPS)"
    else:
        device, where = "cpu", "the CPU (slow: no CUDA or MPS here)"
    # bfloat16 on a GPU, as the models were trained and evaluated (the
    # tool's own default); float32 on the CPU, where bf16 is slow.
    precision = torch.float32 if device == "cpu" else torch.bfloat16
    print(f"running on {where}, {a.preset}, input {a.size}", flush=True)
    if device == "cuda":
        # An overflow is an out-of-memory error (which steps the size down),
        # never a silent spill into shared system memory.
        torch.cuda.set_per_process_memory_fraction(0.9)

    t0 = time.time()
    print("loading the model...", flush=True)
    from safetensors.torch import load_file

    from depth_pro_2.model import PRESETS, DepthPro2
    from depth_pro_2.network.fov import FOVEstimator

    cfg = PRESETS[a.preset]
    model = DepthPro2(
        backbone_preset=cfg.backbone_preset,
        reassembled_blocks=cfg.reassembled_blocks,
        cross_frame_blocks=cfg.cross_frame_blocks,
        dims_reassemble=cfg.dims_reassemble,
        dim_decoder=cfg.dim_decoder,
        fov=False,
        precision=precision,
    )
    model.load_state_dict(load_file(a.weights), strict=True)
    model.fov = FOVEstimator(precision)
    model.fov.load_state_dict(load_file(a.focal), strict=True)
    model = model.to(device).eval()
    print(f"model loaded in {time.time() - t0:.1f} s", flush=True)

    first = read_rgb(inputs[0])
    H, W = first.shape[:2]
    del first
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(3, 1, 1)
    pool = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))

    def net_size(short: int) -> tuple[int, int]:
        s = min(short, min(H, W))  # never upsample the plate
        if H <= W:
            return s, max(2, round(W * s / H))
        return max(2, round(H * s / W)), s

    def solve(short: int) -> list:
        h, w = net_size(short)

        def load(i: int):
            rgb = read_rgb(inputs[i])
            u8 = (srgb_encode(rgb) * 255.0 + 0.5).astype(np.uint8)
            del rgb
            return u8

        def frames():
            # Read ahead on the pool; normalise and shrink on the device.
            ahead = [pool.submit(load, i) for i in range(min(8, total))]
            for i in range(total):
                u8 = ahead[i].result()
                ahead[i] = None
                if i + 8 < total:
                    ahead.append(pool.submit(load, i + 8))
                x = torch.from_numpy(u8).to(device).permute(2, 0, 1).float().div_(255.0)
                x = x.sub_(mean).div_(std)
                if (h, w) != (H, W):
                    x = F.interpolate(x[None], size=(h, w), mode="bilinear", antialias=True, align_corners=False)[0]
                yield x

        print(f"{total} frames at {w}x{h} through the network, 16 at a time", flush=True)
        stats = []
        writes = []
        tw = time.time()
        for i, pred in enumerate(model.infer_video(frames())):
            inv = pred["inverse_depth"].float()
            f_px = pred["focallength_px"]
            if i == 0:
                print(f"focal length {float(f_px):.1f} px at {w} wide; first window {time.time() - tw:.1f} s{vram_words(torch, device)}", flush=True)
            # Back to the plate's size in INVERSE depth (smooth, the sky a
            # finite 0), then metres: f / (width * inverse depth).
            if (h, w) != (H, W):
                inv = F.interpolate(inv[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            depth = (float(f_px) / (w * inv.clamp_min(1e-12))).clamp(max=FAR).cpu().numpy()
            stats.append(np.asarray(1.0 / np.maximum(depth[::8, ::8], 1e-3), np.float32).ravel())
            writes.append(pool.submit(write_z, os.path.join(out_dir, os.path.basename(inputs[i])), depth))
            del depth, inv
            say("frame", i + 1, total)
        for f in writes:
            f.result()
        return stats

    t0 = time.time()
    size = a.size
    try:
        while True:
            try:
                stats = solve(size)
                break
            except RuntimeError as e:
                if not out_of_memory(e):
                    raise
                free_gpu(torch, device)
                smaller = int(size * 0.8) // 32 * 32
                if smaller < SMALLEST:
                    say("error", f"out of GPU memory even at input {size} with {a.preset}: pick a smaller size or model, or free the GPU")
                    return 1
                # Scale down, never fail outright - and say so.
                print(f"out of GPU memory at input {size}: starting again at {smaller}", flush=True)
                size = smaller
    except Unreadable as e:
        bad = []
        for p in inputs:
            try:
                read_rgb(p)
            except Unreadable:
                bad.append(p)
        for p in bad:
            try:
                os.remove(p)
            except OSError:
                pass
        say("error", f"could not read {len(bad)} input frame(s), removed them - press run again to re-render. First: {e}")
        return 1
    dt = time.time() - t0
    print(f"depth: {dt:.2f} s for {total} frames ({1000.0 * dt / total:.0f} ms a frame, reads and writes included)", flush=True)

    sample = np.concatenate(stats) if stats else np.ones(1, np.float32)
    top = float(np.percentile(sample, 99.9)) if sample.size else 1.0
    top = top if top > 1e-8 else 1.0
    print(f"disparity clip maximum {top:.4g} (nearest ~{1.0 / top:.2f} m)", flush=True)
    with open(os.path.join(out_dir, "meta.json.part"), "w") as f:
        json.dump({"layout": LAYOUT, "top": top, "metric": True}, f)
    os.replace(os.path.join(out_dir, "meta.json.part"), os.path.join(out_dir, "meta.json"))
    say("frame", total, total)
    say("done")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        say("error", f"{type(e).__name__}: {first_line(e)}")
        raise
