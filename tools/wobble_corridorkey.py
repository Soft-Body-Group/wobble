"""wobble's driver for CorridorKey (github.com/nikopueringer/CorridorKey).

Written next to an installed CorridorKey checkout by the app on every run
and executed from that directory with its own `uv` environment:

    uv run --project <checkout> python wobble_corridorkey.py --shot <dir> ...

The shot directory is what wobble rendered: `Input/f_NNNNN.exr` (the
plate, linear Rec.709) and `AlphaHint/f_NNNNN.exr` (a coarse matte in every
channel). Each frame goes through the tool's own engine and its `processed`
result — despilled foreground premultiplied by the cleaned matte, linear
Rec.709 — lands in `Processed/f_NNNNN.exr`, one file per input frame, same
name. Frames already there are skipped, so an interrupted run resumes.

Progress is one machine-readable line per event on stdout:

    WOBBLE frame <done> <total>
    WOBBLE done
    WOBBLE error <message>

Everything else printed (the tool's own logging) is shown as a log tail.
The tool's licence is CC BY-NC-SA 4.0; this file is wobble's, a thin
client of its published `create_engine` API, and ships no part of it.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys

# OpenCV only reads / writes EXR with this set BEFORE it is imported.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# Executed from the checkout, but sys.path[0] is this file's directory:
# the tool's packages live in the working directory.
sys.path.insert(0, os.getcwd())


def say(*words: object) -> None:
    print("WOBBLE", *words, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", required=True)
    ap.add_argument("--screen", choices=["auto", "green", "blue"], default="auto")
    ap.add_argument("--despill", type=float, default=0.5)
    ap.add_argument("--refiner", type=float, default=1.0)
    ap.add_argument("--despeckle", type=int, default=1)
    ap.add_argument("--despeckle-px", type=int, default=400)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backend", default="auto")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    import cv2
    import numpy as np

    from backend.frame_io import EXR_WRITE_FLAGS, read_image_frame
    from CorridorKeyModule.backend import create_engine
    from CorridorKeyModule.core import color_utils as cu
    from device_utils import resolve_device, setup_rocm_env

    setup_rocm_env()

    inputs = sorted(glob.glob(os.path.join(a.shot, "Input", "*.exr")))
    if not inputs:
        say("error", "no Input frames")
        return 1
    out_dir = os.path.join(a.shot, "Processed")
    os.makedirs(out_dir, exist_ok=True)

    def hint_for(stem: str) -> np.ndarray | None:
        p = os.path.join(a.shot, "AlphaHint", stem + ".exr")
        m = cv2.imread(p, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        if m is None:
            return None
        if m.ndim == 3:
            m = m[:, :, 0]
        return np.clip(m.astype(np.float32), 0.0, 1.0)

    # Which model: the plate's own screen pixels decide under "auto", read
    # the way the tool reads them (sRGB-encoded, outside the hint).
    screen = a.screen
    if screen == "auto":
        stem = os.path.splitext(os.path.basename(inputs[0]))[0]
        img = read_image_frame(inputs[0], gamma_correct_exr=True)
        hint = hint_for(stem)
        screen = cu.estimate_screen_color(img, hint) if img is not None and hint is not None else "green"
        say("screen", screen)
    backend = a.backend
    if screen == "blue" and backend in ("auto", "mlx"):
        # The MLX port has no blue weights yet.
        backend = "torch"
    device = resolve_device(None if a.device == "auto" else a.device)
    try:
        engine = create_engine(backend=backend, device=device, screen_color=screen)
    except Exception as e:  # noqa: BLE001 — the message is the point
        say("error", f"loading the model: {e}")
        return 1
    channel = cu.screen_channel_for_color(screen)

    total = len(inputs)
    say("frame", 0, total)
    for i, path in enumerate(inputs):
        stem = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(out_dir, stem + ".exr")
        if os.path.exists(out):
            say("frame", i + 1, total)
            continue
        img = read_image_frame(path, gamma_correct_exr=False)
        hint = hint_for(stem)
        if img is None or hint is None:
            say("error", f"could not read frame {stem}")
            return 1
        if hint.shape[:2] != img.shape[:2]:
            hint = cv2.resize(hint, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        res = engine.process_frame(
            img,
            hint,
            input_is_linear=True,
            fg_is_straight=True,
            despill_strength=a.despill,
            auto_despeckle=bool(a.despeckle),
            despeckle_size=a.despeckle_px,
            refiner_scale=a.refiner,
            generate_comp=False,
            screen_channel=channel,
        )
        rgba = res["processed"]
        tmp = out + ".part.exr"
        cv2.imwrite(tmp, cv2.cvtColor(rgba.astype(np.float32), cv2.COLOR_RGBA2BGRA), EXR_WRITE_FLAGS)
        os.replace(tmp, out)
        say("frame", i + 1, total)
    say("done")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        say("error", str(e))
        raise
