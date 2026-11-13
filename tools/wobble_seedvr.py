"""wobble's driver for SeedVR2 (ByteDance-Seed), through the numz CLI
(github.com/numz/ComfyUI-SeedVR2_VideoUpscaler, Apache-2.0).

Written next to an installed checkout by the app on every run and executed
from that directory with its own `uv` environment:

    uv run --project <checkout> python wobble_seedvr.py --shot <dir> ...

The shot directory is what wobble rendered: `Input/f_NNNNN.exr`, linear
Rec.709, premultiplied. SeedVR2 is a display-referred model, so each frame
is unpremultiplied, clamped to 0..1 and sRGB-encoded on the way in; on the
way out the result is decoded, the part of the input the model never saw
(highlights past 1, negatives) is resized bilinearly and added back, alpha
is resized beside it and the colour premultiplied again — the same contract
as the Upscale node. Results land in `Output/f_NNNNN.exr`, same names.

Frames go through the tool's own four-phase pipeline in chunks of whole
batches (SeedVR2 is consistent within a batch of `--batch` frames and
independent across them, so a chunk boundary on a batch boundary changes
nothing), with the models kept loaded between chunks. A chunk whose frames
are all on disk is skipped, so an interrupted run resumes.

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

# OpenCV only reads / writes EXR with this set BEFORE it is imported.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# Executed from the checkout, but sys.path[0] is this file's directory:
# the tool's packages live in the working directory.
sys.path.insert(0, os.getcwd())

# Which DiT file, by model and device. Apple Silicon has unified memory and
# no fp8 arithmetic, so fp16; a CUDA card under 20 GB takes the fp8 builds.
DIT = {
    ("3b", "fp16"): "seedvr2_ema_3b_fp16.safetensors",
    ("3b", "fp8"): "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    ("7b", "fp16"): "seedvr2_ema_7b_fp16.safetensors",
    ("7b", "fp8"): "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    ("7b_sharp", "fp16"): "seedvr2_ema_7b_sharp_fp16.safetensors",
    ("7b_sharp", "fp8"): "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
}


def say(*words: object) -> None:
    print("WOBBLE", *words, flush=True)


def srgb_encode(v):
    import numpy as np

    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)


def srgb_decode(v):
    import numpy as np

    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", required=True)
    ap.add_argument("--model", choices=["3b", "7b", "7b_sharp"], default="3b")
    ap.add_argument("--resolution", type=int, required=True, help="output short side")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--colour", default="lab")
    ap.add_argument("--models-dir", default=os.path.join(os.getcwd(), "models", "SEEDVR2"))
    a = ap.parse_args()

    import cv2
    import numpy as np
    import torch

    inputs = sorted(glob.glob(os.path.join(a.shot, "Input", "*.exr")))
    if not inputs:
        say("error", "no Input frames")
        return 1
    out_dir = os.path.join(a.shot, "Output")
    os.makedirs(out_dir, exist_ok=True)

    if torch.backends.mps.is_available():
        precision, where = "fp16", "Apple Silicon (MPS)"
    elif torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30
        precision = "fp16" if vram >= 20 else "fp8"
        where = f"{torch.cuda.get_device_name(0)} ({vram:.0f} GB)"
    else:
        say("error", "SeedVR2 needs an NVIDIA GPU (CUDA) or Apple Silicon (MPS); this machine has neither")
        return 1
    dit = DIT[(a.model, precision)]
    print(f"running on {where}: {dit}", flush=True)

    # The tool's own argument parser, so every default it relies on is set.
    sys.argv = [
        "inference_cli.py",
        a.shot,
        "--dit_model", dit,
        "--model_dir", a.models_dir,
        "--resolution", str(a.resolution),
        "--batch_size", str(a.batch),
        "--seed", str(a.seed),
        "--color_correction", a.colour,
        "--cache_dit",
        "--cache_vae",
        # Tiled VAE: the same speed and about half the memory — measured on
        # an M4 Pro at 1080p out, 50 GB at the peak untiled, 28 GB tiled.
        "--vae_encode_tiled",
        "--vae_decode_tiled",
    ]
    import inference_cli as cli

    args = cli.parse_arguments()
    if not cli.download_weight(dit_model=args.dit_model, vae_model=cli.DEFAULT_VAE, model_dir=args.model_dir, debug=cli.debug):
        say("error", "could not download the SeedVR2 weights (see the log)")
        return 1

    def read(path: str):
        m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if m is None:
            raise RuntimeError(f"could not read {path}")
        m = m.astype(np.float32)
        if m.ndim == 2:
            m = np.stack([m, m, m, np.ones_like(m)], -1)
        elif m.shape[2] == 3:
            m = np.concatenate([m[:, :, ::-1], np.ones_like(m[:, :, :1])], -1)
        else:
            m = cv2.cvtColor(m, cv2.COLOR_BGRA2RGBA)
        rgb, alpha = m[:, :, :3], m[:, :, 3:4]
        straight = np.where(alpha > 1e-5, rgb / np.maximum(alpha, 1e-5), rgb)
        return straight, alpha

    total = len(inputs)
    chunk = max(1, a.batch) * 4
    cache: dict = {}
    say("frame", 0, total)
    for start in range(0, total, chunk):
        paths = inputs[start : start + chunk]
        outs = [os.path.join(out_dir, os.path.basename(p)) for p in paths]
        if all(os.path.exists(o) for o in outs):
            say("frame", start + len(paths), total)
            continue
        frames = [read(p) for p in paths]
        x = torch.from_numpy(np.stack([srgb_encode(s) for s, _ in frames]).astype(np.float32))
        y = cli._process_frames_core(frames_tensor=x, args=args, device_id="0", debug=cli.debug, runner_cache=cache)
        y = y.float().numpy()
        for (straight, alpha), o, img in zip(frames, outs, y):
            h, w = img.shape[:2]
            lin = srgb_decode(img[:, :, :3])
            # What the model never saw, carried around it.
            excess = straight - np.clip(straight, 0.0, 1.0)
            lin = lin + cv2.resize(excess, (w, h), interpolation=cv2.INTER_LINEAR)
            al = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
            al = al[:, :, None] if al.ndim == 2 else al
            rgba = np.concatenate([lin * al, al], -1).astype(np.float32)
            tmp = o + ".part.exr"
            cv2.imwrite(tmp, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
            os.replace(tmp, o)
        say("frame", start + len(paths), total)
    say("done")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        say("error", str(e).splitlines()[0] if str(e) else type(e).__name__)
        raise
