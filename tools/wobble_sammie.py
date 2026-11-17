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
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

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
    # Its progress is a tqdm bar on stderr, redrawn with CR and never a
    # newline: relayed here as a line every 5%, or a big checkpoint
    # downloads for minutes with nothing in the log.
    p = subprocess.Popen(
        [sys.executable, os.path.join("sammie", "model_downloader.py"), key],
        stdin=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    shown, buf = -5, b""
    for chunk in iter(lambda: p.stderr.read1(4096), b""):
        buf += chunk
        *parts, buf = re.split(rb"[\r\n]", buf)
        for part in parts:
            m = re.match(rb"\s*(\d+)%", part)
            if m and int(m.group(1)) >= shown + 5:
                shown = int(m.group(1))
                print(f"fetching {os.path.basename(path)}: {shown}%", flush=True)
            elif not m and part.strip():
                print(part.decode(errors="replace"), flush=True)
    r = p.wait()
    if r != 0 or not os.path.exists(path):
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


_SRGB8 = None


def srgb8(lin):
    """Linear to sRGB 8-bit through a 16-bit table: a lookup a pixel instead
    of a `pow`, which on a 2K plate cost more than SAM's whole frame. The
    table's step is a twentieth of a code value at its steepest."""
    import numpy as np

    global _SRGB8
    if _SRGB8 is None:
        c = np.linspace(0.0, 1.0, 65536)
        c = np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)
        _SRGB8 = (c * 255.0 + 0.5).astype(np.uint8)
    return _SRGB8[(np.clip(lin, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)]


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
    # The predictor hard-codes SAM 2's 1024 px backbone grid (256/128/64);
    # EfficientTAM runs at 512, and a 64 vs 32 mismatch fails every mask.
    s = pred.model.image_size
    pred._bb_feat_sizes = [(s // 4, s // 4), (s // 8, s // 8), (s // 16, s // 16)]
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

    # The plate, the way SAM sees pictures: sRGB 8-bit, numbered from 0, and
    # no bigger than its 1024 x 1024 input on either axis (it resizes every
    # frame to that; the masks come back at plate size from the logits).
    # One frame per worker: the EXR decode and the encode drop the GIL, and
    # on one thread this step cost several times the tracking itself.
    frames = os.path.join(a.shot, "frames")
    os.makedirs(frames, exist_ok=True)
    first = cv2.imread(inputs[0], cv2.IMREAD_UNCHANGED)
    if first is None:
        say("error", f"could not read {stems[0]}")
        return 1
    h, w = first.shape[:2]
    del first

    def to_jpeg(i: int) -> str | None:
        out = os.path.join(frames, f"{i:05d}.jpg")
        if os.path.exists(out):
            return None
        img = cv2.imread(inputs[i], cv2.IMREAD_UNCHANGED)
        if img is None:
            return f"could not read {stems[i]}"
        img = img[:, :, :3]
        size = (min(img.shape[1], 1024), min(img.shape[0], 1024))
        if size != (img.shape[1], img.shape[0]):
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", srgb8(img.astype(np.float32)), [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            return f"could not encode {stems[i]}"
        # Not `.jpg` until whole: SAM lists the directory by extension.
        with open(out + ".part", "wb") as f:
            f.write(jpg.tobytes())
        os.replace(out + ".part", out)
        return None

    workers = min(16, os.cpu_count() or 4)
    with ThreadPoolExecutor(workers) as pool:
        for e in pool.map(to_jpeg, range(n)):
            if e:
                say("error", e)
                return 1

    device, autocast = device_and_autocast()
    print(f"running on {device}", flush=True)
    ckpt, cfg = SAM[a.model]
    fetch(ckpt, a.model)
    from sam2.build_sam import build_sam2_video_predictor

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

    # Without matting the segment IS the matte: each frame is written by a
    # worker as it is tracked, not all at the end. With matting the
    # segments are kept for MatAnyone to seed from.
    keep = a.matting != "off"
    masks: dict[int, np.ndarray] = {}
    seen: set[int] = set()
    writes = []
    done = 0
    with ThreadPoolExecutor(workers) as pool, torch.inference_mode(), autocast():
        pred = build_sam2_video_predictor(cfg, ckpt, device=device)
        # The frames load once and serve both ways. Not SAM's async loader:
        # measured, it halved the tracking rate for the whole run.
        state = pred.init_state(video_path=frames, offload_video_to_cpu=True)

        def run(reverse: bool) -> None:
            # Prompted afresh each way: a backward pass over a state the
            # forward one has filled comes back empty.
            nonlocal done
            pred.reset_state(state)
            for i in keys:
                ps = by_frame[i]
                xy = np.array([[p["x"] * w, p["y"] * h] for p in ps], dtype=np.float32)
                lb = np.array([1 if p["positive"] else 0 for p in ps], dtype=np.int32)
                pred.add_new_points_or_box(state, frame_idx=i, obj_id=1, points=xy, labels=lb)
            for fi, _ids, logits in pred.propagate_in_video(state, start_frame_idx=keys[0], reverse=reverse):
                if fi in seen:
                    continue
                seen.add(fi)
                # SAM's own last step (logits up to size, then > 0), at the
                # plate's size rather than the frames' it was given.
                up = torch.nn.functional.interpolate(logits[:1].float(), size=(h, w), mode="bilinear", align_corners=False)
                m = (up[0, 0] > 0.0).to(torch.uint8).cpu().numpy()
                if keep:
                    masks[fi] = m * 255
                else:
                    writes.append(pool.submit(write, fi, m))
                done += 1
                say("frame", done, total)

        run(False)
        if keys[0] > 0:
            run(True)
        del pred, state
        if not keep:
            for i in range(n):
                if i not in seen:
                    writes.append(pool.submit(write, i, np.zeros((h, w), np.uint8)))
            for f in writes:
                f.result()

    if not keep:
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
    h8, w8 = max(8, h // 8 * 8), max(8, w // 8 * 8)

    # MatAnyone sees the plate at its own size, not SAM's smaller frames;
    # decoded a few frames ahead on the pool, since a frame is a full EXR.
    def picture(i: int):
        img = cv2.imread(inputs[i], cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"could not read {stems[i]}")
        img = cv2.resize(srgb8(img[:, :, 2::-1].astype(np.float32)), (w8, h8), interpolation=cv2.INTER_AREA)
        return img

    def pictures(order):
        order, ahead = list(order), []
        with ThreadPoolExecutor(min(4, workers)) as pool:
            for k, i in enumerate(order):
                while len(ahead) < 4 and k + len(ahead) < len(order):
                    ahead.append(pool.submit(picture, order[k + len(ahead)]))
                img = ahead.pop(0).result()
                yield i, torch.tensor(img / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)

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
        for k, (i, img) in enumerate(pictures(order)):
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
