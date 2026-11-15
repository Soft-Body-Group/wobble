"""wobble's driver for NormalCrafter (Bin et al., ICCV 2025,
github.com/Binyr/NormalCrafter, MIT code).

Written next to an installed checkout by the app on every run and executed
from that directory with its own `uv` environment:

    uv run --project <checkout> python wobble_normals.py --shot <dir> \
        [--max-res 1024] [--window 14] [--stride 10]

The shot directory is what wobble rendered: `Input/f_NNNNN.exr`, linear
Rec.709, premultiplied. The network is a video diffusion model fine-tuned
from Stable Video Diffusion and is display-referred, so each frame is
clamped to 0..1 and sRGB-encoded on the way in, then shrunk so its long
side is `--max-res` (the pipeline pads to a multiple of 64 itself).

The weights are the TOOL's: the first run fetches `Yanrui95/NormalCrafter`
from Hugging Face into the user's Hugging Face cache (the whole pipeline
lives in that one repository: UNet, temporal VAE, CLIP image encoder;
~5 GB). They are an SVD derivative, under the Stability AI Community
Licence; the app tags the node accordingly.

The clip goes through the tool's own windowing — `--window` frames at a
time, each window starting `--stride` frames after the last, the overlap
taken from the previous window's latents and linearly blended (what keeps
the normals steady over time) — re-done here so the decode STREAMS: the
tool's `__call__` holds every input frame and every decoded result in
memory at once. Here frames are read, encoded (CLIP + VAE latents, which
are 1/8 size) chunk by chunk, and each result frame is decoded, scaled
back to the plate's size and written as soon as no later window can touch
it.

Each result frame lands in `Normals/f_NNNNN.exr`, same names: `R G B` =
the unit normal's x y z in the network's camera space (x right, y up, z
towards the camera; -1..1), half float, ZIP-compressed. `Normals/meta.json`
says so:

    {"layout": 1, "space": "camera, x right, y up, z towards the camera"}

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
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Executed from the checkout, but sys.path[0] is this file's directory:
# the tool's packages live in the working directory.
sys.path.insert(0, os.getcwd())

# Matches app::normals::LAYOUT.
LAYOUT = 1
REPO_ID = "Yanrui95/NormalCrafter"
# The Hugging Face revision the driver loads: a later push to the repo
# must not silently change what an old comp renders.
REVISION = "7e24d68d86ae008fe08ef50b4e51cd2fc2c8cf57"


def say(*words: object) -> None:
    print("WOBBLE", *words, flush=True)


def first_line(e: BaseException) -> str:
    s = str(e).strip()
    return s.splitlines()[0] if s else type(e).__name__


def srgb_encode(v):
    import numpy as np

    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)


class Unreadable(RuntimeError):
    """An Input frame neither reader could open. The driver removes it, so
    the next run renders it again rather than failing on it forever."""


def read_rgb(path: str):
    """An Input frame as float32 rgb: the OpenEXR package first, OpenCV's
    EXR codec as the fallback, both readers' reasons on failure."""
    import numpy as np

    why = []
    try:
        import OpenEXR

        with OpenEXR.File(path, separate_channels=True) as f:
            by = {k.rsplit(".", 1)[-1]: v.pixels for k, v in f.channels().items()}
            if all(c in by for c in "RGB"):
                return np.stack([by["R"], by["G"], by["B"]], -1).astype(np.float32)
            if by:
                one = next(iter(by.values())).astype(np.float32)
                return np.stack([one, one, one], -1)
            why.append("OpenEXR: no channels")
    except Exception as e:  # noqa: BLE001
        why.append(f"OpenEXR: {first_line(e)}")
    try:
        import cv2

        m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if m is not None:
            m = m.astype(np.float32)
            if m.ndim == 2:
                return np.stack([m, m, m], -1)
            return m[:, :, 2::-1] if m.shape[2] >= 3 else np.repeat(m[:, :, :1], 3, -1)
        why.append("OpenCV: imread returned nothing")
    except Exception as e:  # noqa: BLE001
        why.append(f"OpenCV: {first_line(e)}")
    try:
        size = f"{os.path.getsize(path)} bytes"
    except OSError as e:
        size = first_line(e)
    raise Unreadable(f"{os.path.basename(path)} ({size}): {'; '.join(why)}")


def write_rgb(path: str, rgb) -> None:
    """R G B half float, ZIP-compressed, written beside `path` and renamed
    into place, so a stopped run never leaves a frame that looks done."""
    import numpy as np
    import OpenEXR

    tmp = path + ".part"
    chans = {c: np.ascontiguousarray(rgb[:, :, i], np.float16) for i, c in enumerate("RGB")}
    with OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}, chans) as f:
        f.write(tmp)
    os.replace(tmp, path)


def windows(n: int, window: int, stride: int):
    """The tool's `get_ses`: [start, end) per window, the last one pulled
    back to end on the clip's last frame."""
    ses = [[i, i + window] for i in range(0, n, stride)]
    ses = [s for s in ses if s[1] <= n]
    if not ses or ses[-1][1] < n:
        ses.append([n - window, n])
    return ses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", required=True)
    ap.add_argument("--max-res", type=int, default=1024, help="the network's long side")
    ap.add_argument("--window", type=int, default=14)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=7, help="frames per (temporal) VAE decode")
    a = ap.parse_args()

    import json
    import time

    import numpy as np
    import torch

    inputs = sorted(glob.glob(os.path.join(a.shot, "Input", "*.exr")))
    if not inputs:
        say("error", "no Input frames")
        return 1
    out_dir = os.path.join(a.shot, "Normals")
    os.makedirs(out_dir, exist_ok=True)
    total = len(inputs)
    window = max(2, a.window)
    stride = min(max(1, a.stride), window)

    if torch.cuda.is_available():
        device, where = "cuda", torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device, where = "mps", "Apple Silicon (MPS)"
    else:
        device, where = "cpu", "the CPU (very slow: no CUDA or MPS here)"
    dtype = torch.float32 if device == "cpu" else torch.float16
    print(f"running on {where}, long side {a.max_res}, window {window} / stride {stride}", flush=True)
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.92)

    t0 = time.time()
    print(f"loading NormalCrafter ({REPO_ID}; the first run downloads ~5 GB)...", flush=True)
    from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    from normalcrafter.normal_crafter_ppl import NormalCrafterPipeline
    from normalcrafter.unet import DiffusersUNetSpatioTemporalConditionModelNormalCrafter

    kw = {"revision": REVISION}
    unet = DiffusersUNetSpatioTemporalConditionModelNormalCrafter.from_pretrained(
        REPO_ID, subfolder="unet", low_cpu_mem_usage=True, torch_dtype=dtype, **kw
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(REPO_ID, subfolder="vae", torch_dtype=dtype, **kw)
    enc = CLIPVisionModelWithProjection.from_pretrained(
        REPO_ID, subfolder="image_encoder", variant="fp16", torch_dtype=dtype, **kw
    )
    fe = CLIPImageProcessor.from_pretrained(REPO_ID, subfolder="feature_extractor", **kw)
    sched = EulerDiscreteScheduler.from_pretrained(REPO_ID, subfolder="scheduler", **kw)
    pipe = NormalCrafterPipeline(vae=vae, image_encoder=enc, unet=unet, scheduler=sched, feature_extractor=fe)
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe = pipe.to(device)
    except RuntimeError as e:
        if device != "cuda" or not out_of_memory(e):
            raise
        # Scale down, never refuse: the GPU is busy or small, so keep the
        # models in system memory and move each one over as it runs.
        pipe.to("cpu")
        free_gpu(torch, device)
        print("not enough free GPU memory to hold the models: offloading each to the CPU between uses (slower)", flush=True)
        pipe.enable_model_cpu_offload()
    print(f"model loaded in {time.time() - t0:.1f} s", flush=True)

    from concurrent.futures import ThreadPoolExecutor

    import cv2
    from PIL import Image

    pool = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))
    first = read_rgb(inputs[0])
    H, W = first.shape[:2]
    del first

    def solve(max_res: int) -> None:
        s = min(1.0, max_res / max(H, W))
        h, w = max(64, round(H * s)), max(64, round(W * s))
        nh, nw = -(-h // 64) * 64, -(-w // 64) * 64
        pt, pl = (nh - h) // 2, (nw - w) // 2
        print(f"network size {w}x{h} (padded to {nw}x{nh})", flush=True)

        def load(i: int):
            rgb = read_rgb(inputs[min(i, total - 1)])
            u8 = (srgb_encode(rgb) * 255.0 + 0.5).astype(np.uint8)
            u8 = cv2.resize(u8, (w, h), interpolation=cv2.INTER_AREA)
            # The tool's own padding: white, centred.
            pad = np.full((nh, nw, 3), 255, np.uint8)
            pad[pt : pt + h, pl : pl + w] = u8
            return Image.fromarray(pad)

        # A clip shorter than one window repeats its last frame.
        n = max(total, window)
        ses = windows(n, window, stride)
        t1 = time.time()
        embeds, latents = [], []
        # The VAE's ENCODER is per frame (only its decoder is temporal), so
        # encoding two at a time changes nothing and keeps the fp32 upcast
        # it does small.
        step = 2
        with torch.inference_mode():
            for c in range(0, n, step):
                imgs = list(pool.map(load, range(c, min(n, c + step))))
                embeds.append(pipe._encode_image(imgs, device, 1, False))
                latents.append(pipe.ecnode_video_vae(imgs, chunk_size=step).to(embeds[-1].dtype))
                if (c // step) % 8 == 7 or c + step >= n:
                    print(f"encoded {min(n, c + step)} of {n} frames", flush=True)
        embeds = torch.cat(embeds)
        latents = torch.cat(latents).unsqueeze(0)
        print(f"encode took {time.time() - t1:.1f} s{vram_words(torch, device)}", flush=True)
        added = pipe._get_add_time_ids(7, 127, 0.0, embeds.dtype, 1, 1, False).to(device)

        written = 0

        def emit(frames, start: int) -> None:
            """Decoded frames for clip indices start.. : unpad, back to the
            plate's size, unit length, written."""

            def one(k: int) -> None:
                i = start + k
                if i >= total:
                    return
                v = frames[k][pt : pt + h, pl : pl + w]
                v = cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR)
                v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)
                write_rgb(os.path.join(out_dir, os.path.basename(inputs[i])), v)

            list(pool.map(one, range(len(frames))))

        chunk = max(1, a.chunk)

        def decode(part):
            """The temporal VAE decode, `chunk` frames at a time; out of
            memory halves the chunk (for the rest of the run) before it
            gives up on the resolution."""
            nonlocal chunk
            while True:
                try:
                    return pipe.decode_latents(part, part.shape[1], chunk)
                except RuntimeError as e:
                    if not out_of_memory(e) or chunk == 1:
                        raise
                    free_gpu(torch, device)
                    chunk = max(1, chunk // 2)
                    print(f"out of GPU memory decoding: {chunk} frames at a time from here", flush=True)

        prev = None
        with torch.inference_mode():
            for wi, (s0, s1) in enumerate(ses):
                tw = time.time()
                replace = None
                if prev is not None and stride != window:
                    replace = prev[:, -(ses[wi - 1][1] - s0) :]
                cur = pipe.generate(
                    1, device, 1, 1, window, nh, nw, embeds[s0:s1], None, True, replace, False,
                    latents[:, s0:s1], added,
                )
                if replace is not None:
                    k = replace.shape[1]
                    wt = torch.linspace(1.0, 0.0, k + 2)[1:-1].to(device)[None, :, None, None, None]
                    cur[:, :k] = replace * wt + cur[:, :k] * (1 - wt)
                prev = cur
                # Everything before the next window's start is final.
                end = ses[wi + 1][0] if wi + 1 < len(ses) else s1
                if end > written:
                    part = cur[:, written - s0 : end - s0]
                    fr = decode(part.to(pipe.vae.dtype))
                    fr = pipe.video_processor.postprocess_video(video=fr, output_type="np")[0] * 2.0 - 1.0
                    emit(fr.astype(np.float32), written)
                    written = end
                if wi == 0:
                    print(f"window 1 of {len(ses)} took {time.time() - tw:.1f} s{vram_words(torch, device)}", flush=True)
                say("frame", min(written, total), total)

    size = a.max_res
    try:
        while True:
            try:
                solve(size)
                break
            except RuntimeError as e:
                if not out_of_memory(e):
                    raise
                # An out-of-memory inside the VAE's fp32 upcast leaves it
                # in fp32: put it back.
                pipe.vae.to(dtype=dtype)
                free_gpu(torch, device)
                smaller = int(size * 0.8) // 64 * 64
                if smaller < 256:
                    say("error", f"out of GPU memory even at {size}: lower the resolution, or free the GPU")
                    return 1
                # Scale down, never fail outright - and say so.
                print(f"out of GPU memory at {size}: starting again at {smaller}", flush=True)
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
    print(f"done in {time.time() - t0:.1f} s ({size} long side)", flush=True)
    with open(os.path.join(out_dir, "meta.json.part"), "w") as f:
        json.dump({"layout": LAYOUT, "space": "camera, x right, y up, z towards the camera", "max_res": size}, f)
    os.replace(os.path.join(out_dir, "meta.json.part"), os.path.join(out_dir, "meta.json"))
    say("frame", total, total)
    say("done")
    return 0


def out_of_memory(e: BaseException) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def free_gpu(torch, device: str) -> None:
    import gc

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def vram_words(torch, device: str) -> str:
    if device == "cuda":
        return f", peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
    return ""


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        say("error", f"{type(e).__name__}: {first_line(e)}")
        raise
