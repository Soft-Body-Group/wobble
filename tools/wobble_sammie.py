"""wobble's driver for the models Sammie-Roto 2 ships (github.com/Zarxrax/Sammie-Roto-2).

Written next to an installed Sammie checkout by the app on every run and
executed from that directory with its own `uv` environment:

    uv run --project <checkout> python wobble_sammie.py --shot <dir> ...

The shot directory is what wobble rendered: `Input/f_NNNNN.exr` (the plate,
linear Rec.709, named by comp frame) and `points.json` (the clicks:
`{"points": [{"frame": <comp frame>, "x": 0..1, "y": 0..1, "positive": bool}]}`,
y down). SAM 2 (or EfficientTAM) segments the object from the clicks and
tracks it through every frame; with `--matting matanyone2` MatAnyone 2 then
turns the segment into a soft matte, seeded at each clicked frame. The matte
lands in `Matte/f_NNNNN.exr`, one per input frame, same name, the matte in
every channel.

Progress is one machine-readable line per event on stdout:

    WOBBLE frame <done> <total>
    WOBBLE done
    WOBBLE error <message>

Everything else printed is shown as a log tail.

Licences: this file is wobble's. It imports only the `sam2` (Apache-2.0)
and `matanyone` (NTU S-Lab License 1.0, NON-COMMERCIAL) packages the
checkout vendors, never Sammie's own GPL-3.0 application code; checkpoints
are fetched by running Sammie's `model_downloader.py` as a program.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

# OpenCV only reads / writes EXR with this set BEFORE it is imported.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# Ops MPS lacks fall back to the CPU instead of failing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Executed from the checkout, but sys.path[0] is this file's directory:
# the vendored packages live in the working directory.
sys.path.insert(0, os.getcwd())

# The checkpoint each model loads, the model key the tool's downloader
# knows it by, and (for SAM) the config its builder looks up.
SAM = {
    "Base": ("checkpoints/sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "Large": ("checkpoints/sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
    "Efficient": ("checkpoints/efficienttam_s_512x512.pt", "configs/sam2.1/efficienttam_s_512x512.yaml"),
}
MATANYONE2 = ("checkpoints/matanyone2.pth", "matanyone2")


def say(*words: object) -> None:
    print("WOBBLE", *words, flush=True)


def fetch(path: str, key: str) -> None:
    """The tool's own downloader, as a program: it checks the md5."""
    if os.path.exists(path):
        return
    print(f"fetching {os.path.basename(path)} (first run only)", flush=True)
    r = subprocess.run([sys.executable, os.path.join("sammie", "model_downloader.py"), key])
    if r.returncode != 0 or not os.path.exists(path):
        raise RuntimeError(f"could not download the {key} weights")


def device_and_autocast():
    import contextlib

    import torch

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        torch.backends.cuda.matmul.allow_tf32 = True
        return torch.device("cuda"), lambda: torch.autocast("cuda", dtype=dtype)
    if torch.backends.mps.is_available():
        return torch.device("mps"), lambda: torch.autocast("mps", dtype=torch.float16)
    return torch.device("cpu"), contextlib.nullcontext


def srgb8(lin):
    import numpy as np

    c = np.clip(lin, 0.0, 1.0)
    c = np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)
    return (c * 255.0 + 0.5).astype(np.uint8)


def serve(model: str) -> int:
    """The preview segmenter: SAM 2's image predictor, loaded once. Each
    stdin line is `{"image": <exr>, "out": <exr>, "points": [...]}` (points
    as in points.json); the picture is embedded once per image, so the
    next click on the same frame is a mask decode. One `WOBBLE done` or
    `WOBBLE error` a line."""
    import cv2
    import numpy as np
    import torch

    device, autocast = device_and_autocast()
    print(f"segmenter on {device}", flush=True)
    ckpt, cfg = SAM[model]
    fetch(ckpt, model)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    pred = SAM2ImagePredictor(build_sam2(cfg, ckpt, device=device))
    say("ready")
    last, h, w = None, 0, 0
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            with torch.inference_mode(), autocast():
                if req["image"] != last:
                    img = cv2.imread(req["image"], cv2.IMREAD_UNCHANGED)
                    if img is None:
                        raise RuntimeError(f"could not read {os.path.basename(req['image'])}")
                    rgb = srgb8(img[:, :, 2::-1].astype(np.float32))
                    pred.set_image(np.ascontiguousarray(rgb))
                    last, (h, w) = req["image"], rgb.shape[:2]
                pts = req["points"]
                xy = np.array([[p["x"] * w, p["y"] * h] for p in pts], dtype=np.float32)
                lb = np.array([1 if p["positive"] else 0 for p in pts], dtype=np.int32)
                # One click is ambiguous (a shirt or the person in it):
                # take SAM's best-scored of three. More clicks settle it.
                masks, scores, _ = pred.predict(point_coords=xy, point_labels=lb, multimask_output=len(pts) == 1)
                m = np.asarray(masks[int(np.argmax(scores))], dtype=np.float32)
            out = req["out"]
            os.makedirs(os.path.dirname(out), exist_ok=True)
            tmp = out + ".part.exr"
            cv2.imwrite(tmp, np.dstack([m, m, m, m]), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
            os.replace(tmp, out)
            say("done")
        except Exception as e:  # noqa: BLE001 — the reply is the point
            say("error", str(e).splitlines()[0] if str(e) else type(e).__name__)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--shot")
    ap.add_argument("--model", choices=sorted(SAM), default="Base")
    ap.add_argument("--matting", choices=["off", "matanyone2"], default="off")
    # Matting's internal resolution (Sammie's matany_res): the shorter side
    # is cut to this before MatAnyone sees it; 0 = the plate's own size.
    ap.add_argument("--matting-res", type=int, default=0)
    a = ap.parse_args()
    if a.serve:
        return serve(a.model)
    if not a.shot:
        say("error", "--shot is required")
        return 1

    import cv2
    import numpy as np
    import torch

    inputs = sorted(glob.glob(os.path.join(a.shot, "Input", "f_*.exr")))
    if not inputs:
        say("error", "no Input frames")
        return 1
    stems = [os.path.splitext(os.path.basename(p))[0] for p in inputs]
    index_of = {int(s[2:]): i for i, s in enumerate(stems)}
    with open(os.path.join(a.shot, "points.json")) as f:
        points = json.load(f)["points"]
    by_frame: dict[int, list] = {}
    for p in points:
        i = index_of.get(int(p["frame"]))
        if i is not None:
            by_frame.setdefault(i, []).append(p)
    if not by_frame:
        say("error", "no clicks on a frame of the plate")
        return 1
    keys = sorted(by_frame)
    n = len(inputs)
    total = n * (2 if a.matting != "off" else 1)
    say("frame", 0, total)

    # The plate, the way SAM sees pictures: sRGB 8-bit, numbered from 0.
    frames = os.path.join(a.shot, "frames")
    os.makedirs(frames, exist_ok=True)
    h = w = 0
    for i, path in enumerate(inputs):
        out = os.path.join(frames, f"{i:05d}.jpg")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            say("error", f"could not read {stems[i]}")
            return 1
        h, w = img.shape[:2]
        if not os.path.exists(out):
            cv2.imwrite(out, srgb8(img[:, :, :3].astype(np.float32)), [cv2.IMWRITE_JPEG_QUALITY, 95])

    device, autocast = device_and_autocast()
    print(f"running on {device}", flush=True)
    ckpt, cfg = SAM[a.model]
    fetch(ckpt, a.model)
    from sam2.build_sam import build_sam2_video_predictor

    masks: dict[int, np.ndarray] = {}
    done = 0
    with torch.inference_mode(), autocast():
        pred = build_sam2_video_predictor(cfg, ckpt, device=device)

        def prompted():
            state = pred.init_state(video_path=frames, offload_video_to_cpu=True)
            for i in keys:
                ps = by_frame[i]
                xy = np.array([[p["x"] * w, p["y"] * h] for p in ps], dtype=np.float32)
                lb = np.array([1 if p["positive"] else 0 for p in ps], dtype=np.int32)
                pred.add_new_points_or_box(state, frame_idx=i, obj_id=1, points=xy, labels=lb)
            return state

        def run(reverse: bool) -> None:
            # A fresh state each way: a backward pass over a state the
            # forward one has filled comes back empty.
            nonlocal done
            state = prompted()
            for fi, _ids, logits in pred.propagate_in_video(state, start_frame_idx=keys[0], reverse=reverse):
                if fi not in masks:
                    done += 1
                    say("frame", done, total)
                masks[fi] = (logits[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8) * 255

        run(False)
        if keys[0] > 0:
            run(True)
        del pred

    out_dir = os.path.join(a.shot, "Matte")
    os.makedirs(out_dir, exist_ok=True)

    def write(i: int, m) -> None:
        m = m.astype(np.float32)
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        out = os.path.join(out_dir, stems[i] + ".exr")
        tmp = out + ".part.exr"
        rgba = np.dstack([m, m, m, m])
        cv2.imwrite(tmp, rgba, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        os.replace(tmp, out)

    if a.matting == "off":
        for i in range(n):
            write(i, masks.get(i, np.zeros((h, w), np.uint8)) / 255.0)
        say("done")
        return 0

    # MatAnyone 2: seeded with the segment at each clicked frame, run from
    # it to the next clicked frame (and backwards from the first), a fresh
    # memory per run so one seed never leaks into another's frames.
    ckpt, key = MATANYONE2
    fetch(ckpt, key)
    from matanyone.inference.inference_core import InferenceCore
    from matanyone.utils.get_default_model import get_matanyone_model

    model = get_matanyone_model(ckpt, device=device)
    scale = a.matting_res / min(h, w) if 0 < a.matting_res < min(h, w) else 1.0
    h8, w8 = max(8, int(h * scale) // 8 * 8), max(8, int(w * scale) // 8 * 8)

    def image(i: int):
        img = cv2.cvtColor(cv2.imread(os.path.join(frames, f"{i:05d}.jpg")), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w8, h8), interpolation=cv2.INTER_AREA)
        return torch.tensor(img / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)

    def segment(seed: int, order, seeded_elsewhere: bool = False) -> None:
        nonlocal done
        m = masks.get(seed)
        if m is None or not m.any():
            print(f"no segment at frame {stems[seed]}: matting skipped from there", flush=True)
            for k, i in enumerate(order):
                if k == 0 and seeded_elsewhere:
                    continue
                write(i, masks.get(i, np.zeros((h, w), np.uint8)) / 255.0)
                done += 1
                say("frame", done, total)
            return
        proc = InferenceCore(model, cfg=model.cfg, device=device)
        seed_mask = torch.tensor(
            cv2.resize(m, (w8, h8), interpolation=cv2.INTER_NEAREST), dtype=torch.float32, device=device
        )
        for k, i in enumerate(order):
            img = image(i)
            if k == 0:
                prob = proc.step(img, seed_mask, objects=[1])
                for _ in range(10):  # the warm-up the model is trained with
                    prob = proc.step(img, first_frame_pred=True)
            else:
                prob = proc.step(img)
            if k == 0 and seeded_elsewhere:
                continue  # the forward run from this seed writes it
            write(i, proc.output_prob_to_mask(prob).float().clamp(0, 1).cpu().numpy())
            done += 1
            say("frame", done, total)

    with torch.inference_mode(), autocast():
        if keys[0] > 0:
            segment(keys[0], range(keys[0], -1, -1), seeded_elsewhere=True)
        for k, s in enumerate(keys):
            end = keys[k + 1] if k + 1 < len(keys) else n
            segment(s, range(s, end))
    say("done")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        say("error", str(e).splitlines()[0] if str(e) else type(e).__name__)
        raise
