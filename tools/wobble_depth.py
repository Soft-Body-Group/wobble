"""wobble's driver for Video Depth Anything (ByteDance,
github.com/DepthAnything/Video-Depth-Anything, Apache-2.0 code).

Written next to an installed checkout by the app on every run and executed
from that directory with its own `uv` environment:

    uv run --project <checkout> python wobble_depth.py --shot <dir> \
        --weights <file.pth> --encoder vits [--metric] [--size 518]

The weights are NOT fetched here: the app downloads them through its own
models catalog (pinned by hash) and passes the path. Nothing in this file
touches the network.

The shot directory is what wobble rendered: `Input/f_NNNNN.exr`, linear
Rec.709, premultiplied. The network is display-referred, so each frame is
clamped to 0..1 and sRGB-encoded on the way in. The clip goes through the
tool's offline windowing — overlapping 32-frame windows, each aligned to
the last, which is what keeps the depth steady across frames (the
experimental streaming mode measured worse upstream, and is not used) —
re-done here window by window: `infer_video_depth` holds every frame at
full size and every result at the plate's size, ~50 GB for 700 4K frames.
Here a window's frames are read and shrunk to the network's size as it
needs them, aligned results wait on disk at that size (`depth_raw.npy`,
removed at the end) and are scaled back up as they are written, so memory
does not grow with the clip. Same pixels as the tool's own call to 0.02%
(hcr080, measured).

EXRs are read and written with the OpenEXR package; OpenCV's codec is only
the fallback (opencv-python 5.x's Windows wheel has none).

Each result frame lands in `Depth/f_NNNNN.exr`, same names: ONE half-float
channel `Z`, the network's own output (a relative model's disparity, a
metric model's metres), ZIP-compressed. `Depth/meta.json` holds what the
node needs to derive its three outputs in the shader (no re-run to switch):

    {"layout": 2, "top": <disparity clip maximum, 99.9th pct>, "metric": bool}

(Half, not float32: wobble's reader decodes to half whatever the file
holds. Layout 1 wrote three float32 channels, ~74 MB a 4K frame.)

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
# MPS lacks a few ops; let them fall back to the CPU rather than fail.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Executed from the checkout, but sys.path[0] is this file's directory:
# the tool's packages live in the working directory.
sys.path.insert(0, os.getcwd())

# Matches engine::nodes::depth::FAR (the shader caps relative depth there).
FAR = 1000.0
# Matches app::depth::LAYOUT: one `Z` channel + meta.json.
LAYOUT = 2

CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def say(*words: object) -> None:
    print("WOBBLE", *words, flush=True)


def srgb_encode(v):
    import numpy as np

    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)


class Unreadable(RuntimeError):
    """An Input frame neither reader could open. The driver removes it, so
    the next run renders it again rather than failing on it forever."""


def read_rgb(path: str):
    """An Input frame as float32 rgb. The OpenEXR package first (the
    reference library, wheels on every OS); OpenCV's EXR codec as the
    fallback. On failure, the reason from BOTH, and the file's size, so the
    panel says why instead of just "could not read"."""
    import numpy as np

    why = []
    try:
        import OpenEXR

        with OpenEXR.File(path, separate_channels=True) as f:
            ch = f.channels()
            # Channel names may carry a layer prefix ("rgba.R").
            by = {k.rsplit(".", 1)[-1]: v.pixels for k, v in ch.items()}
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


def write_z(path: str, z) -> None:
    """One half-float channel `Z`, ZIP-compressed, written beside `path` and
    renamed into place, so a run stopped mid-write never leaves a frame that
    looks finished. (OpenCV cannot name a channel: no fallback here, and the
    pyproject installs OpenEXR.)"""
    import numpy as np
    import OpenEXR

    tmp = path + ".part"
    chans = {"Z": np.ascontiguousarray(z, np.float16)}
    with OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}, chans) as f:
        f.write(tmp)
    os.replace(tmp, path)


def first_line(e: BaseException) -> str:
    s = str(e).strip()
    return s.splitlines()[0] if s else type(e).__name__


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--encoder", choices=sorted(CONFIGS), default="vits")
    ap.add_argument("--metric", action="store_true")
    ap.add_argument("--size", type=int, default=518, help="network input short side, a multiple of 14")
    a = ap.parse_args()

    import time

    import numpy as np
    import torch

    inputs = sorted(glob.glob(os.path.join(a.shot, "Input", "*.exr")))
    if not inputs:
        say("error", "no Input frames")
        return 1
    out_dir = os.path.join(a.shot, "Depth")
    os.makedirs(out_dir, exist_ok=True)
    total = len(inputs)

    if torch.backends.mps.is_available():
        device, where = "mps", "Apple Silicon (MPS)"
    elif torch.cuda.is_available():
        device, where = "cuda", torch.cuda.get_device_name(0)
    else:
        device, where = "cpu", "the CPU (slow: no CUDA or MPS here)"
    # Half precision on a GPU (the tool's default); the CPU has no fast fp16.
    fp32 = device == "cpu"
    print(f"running on {where}, {a.encoder}{' metric' if a.metric else ''}, input {a.size}", flush=True)
    if device == "cuda":
        # Never let the driver spill into shared system memory (Windows
        # does, silently, and CUDA then crawls): over 90% of the card is an
        # out-of-memory error, which steps the input size down below.
        torch.cuda.set_per_process_memory_fraction(0.9)

    t0 = time.time()
    print("loading the model...", flush=True)
    import video_depth_anything.dinov2_layers.attention as att
    import video_depth_anything.video_depth as vd
    from video_depth_anything.video_depth import VideoDepthAnything

    if not att.XFORMERS_AVAILABLE:
        # Without xformers DINOv2 builds the whole attention matrix: 32
        # frames x 16 heads x 2311^2 at a 4K plate's input, ~5.5 GB a layer
        # (45 GB peak for the large model, measured on MPS). The same
        # attention through torch's fused kernel: 19 GB, half the time.
        att.Attention.forward = sdpa_forward
        att.MemEffAttention.forward = sdpa_forward

    model = VideoDepthAnything(**CONFIGS[a.encoder], metric=a.metric)
    state = torch.load(a.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    print(f"model loaded in {time.time() - t0:.1f} s", flush=True)

    import torch.nn.functional as F
    from torchvision.transforms import Compose

    from utils.util import compute_scale_and_shift, get_interpolate_frames
    from video_depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize

    # The tool's own window constants: the result must be the offline
    # mode's, only without the whole clip in memory.
    L, OV, KF, IL = vd.INFER_LEN, vd.OVERLAP, vd.KEYFRAMES, vd.INTERP_LEN
    step = L - OV
    align_len = OV - IL
    kf_align = KF[:align_len]
    windows = (total + step - 1) // step

    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))

    # The input size, exactly as `infer_video_depth` picks it.
    first = read_rgb(inputs[0])
    H, W = first.shape[:2]
    del first
    size = a.size
    ratio = max(H, W) / min(H, W)
    if ratio > 1.78:
        size = round(int(size * 1.777 / ratio) / 14) * 14

    raw_path = os.path.join(a.shot, "depth_raw.npy")
    half = not fp32

    def infer(x):
        nonlocal half
        while True:
            try:
                with torch.inference_mode():
                    with torch.autocast(device_type=device, enabled=half):
                        return model.forward(x.to(device)).float()[0].cpu().numpy()
            except RuntimeError as e:
                if not half or out_of_memory(e):
                    raise
                print(f"half precision failed ({first_line(e)}); continuing in fp32", flush=True)
                half = False

    def solve(size: int):
        """The whole clip at one network input size. Aligned results at the
        network's size wait on DISK (`depth_raw.npy`): memory does not grow
        with the clip; they are upsampled to the plate's size when written.
        Only the current window is ever on the GPU."""
        transform = Compose([
            Resize(width=size, height=size, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
                   resize_method="lower_bound", image_interpolation_method=cv2_cubic()),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        def load(i: int):
            """Frame i of the clip (the last one repeats past the end, as
            the tool pads), as the network's input: read, sRGB-encoded to 8
            bits like a video frame, resized. Only this small tensor is kept."""
            rgb = read_rgb(inputs[min(i, total - 1)])
            u8 = (srgb_encode(rgb) * 255.0 + 0.5).astype(np.uint8)
            del rgb
            x = transform({"image": u8.astype(np.float32) / 255.0})["image"]
            return torch.from_numpy(x).unsqueeze(0).unsqueeze(0)

        raw = None
        stats = []  # a sparse sample of every frame, for the clip maximum
        pending = []  # aligned frames the next window may still blend
        written = 0

        def flush(keep: int) -> None:
            nonlocal written
            while len(pending) > keep and written < total:
                d = pending.pop(0)
                raw[written] = d
                stats.append(np.asarray(d[::8, ::8], np.float32).ravel())
                written += 1
            if len(pending) > keep:
                del pending[: len(pending) - keep]

        pre = None
        ref = []
        for w, fid in enumerate(range(0, total, step)):
            tw = time.time()
            if w == 0:
                print(f"window 1 of {windows}: reading {L} frames", flush=True)
            xs = list(pool.map(load, range(fid, fid + L)))
            cur = torch.cat(xs, dim=1)
            del xs
            if pre is not None:
                cur[:, :OV] = pre[:, KF]
            if w == 0:
                print(f"window 1 of {windows}: {L} frames at {cur.shape[-1]}x{cur.shape[-2]} through the network", flush=True)
            d = infer(cur)
            pre = cur
            if raw is None:
                raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float32, shape=(total,) + d.shape[1:])
            if fid == 0:
                pending.extend(d[i] for i in range(L))
                ref = [d[k] for k in kf_align]
            else:
                if a.metric:
                    scale, shift = 1.0, 0.0
                else:
                    curr = [d[i] for i in range(len(kf_align))]
                    scale, shift = compute_scale_and_shift(
                        np.concatenate(curr), np.concatenate(ref), np.concatenate(np.ones_like(ref) == 1)
                    )
                fix = lambda x: np.maximum(x * scale + shift, 0.0)  # noqa: E731
                post = [fix(d[i]) for i in range(align_len, OV)]
                pending[-IL:] = get_interpolate_frames(pending[-IL:], post)
                pending.extend(fix(d[i]) for i in range(OV, L))
                ref = ref[:1] + [fix(d[k]) for k in kf_align[1:]]
            del d
            # Everything but the frames the next window blends is final.
            flush(IL)
            if w == 0:
                print(f"window 1 took {time.time() - tw:.1f} s ({windows} windows){vram_words(torch, device)}", flush=True)
            say("frame", min(fid + step, total), total)
        flush(0)
        return raw, stats

    t0 = time.time()
    try:
        while True:
            try:
                raw, stats = solve(size)
                break
            except RuntimeError as e:
                if not out_of_memory(e):
                    raise
                free_gpu(torch, device)
                smaller = round(size * 0.8 / 14) * 14
                if smaller < 140:
                    say("error", f"out of GPU memory even at input {size} with {a.encoder}: pick the small model, or free the GPU")
                    return 1
                # Scale down, never fail outright - and say so.
                print(f"out of GPU memory at input {size}: starting again at {smaller}", flush=True)
                size = smaller
    except Unreadable as e:
        # A frame left half-written by an interrupted render: remove every
        # one that does not open, so the next run renders them again.
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
    print(f"inference: {dt:.2f} s for {total} frames ({1000.0 * dt / total:.0f} ms a frame, reads included)", flush=True)

    # A robust clip maximum: a few hot pixels at an edge should not set the
    # scale for the whole shot.
    sample = np.concatenate(stats) if stats else np.ones(1, np.float32)
    if a.metric:
        sample = 1.0 / np.maximum(sample, 1e-3)
    top = float(np.percentile(sample, 99.9)) if sample.size else 1.0
    top = top if top > 1e-8 else 1.0
    print(f"disparity clip maximum {top:.4g}", flush=True)

    def write(i: int) -> None:
        # Back to the plate's size the way the tool does it.
        t = torch.from_numpy(np.array(raw[i], np.float32))[None, None]
        v = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=True)[0, 0].numpy()
        write_z(os.path.join(out_dir, os.path.basename(inputs[i])), v)

    t0 = time.time()
    list(pool.map(write, range(total)))
    print(f"wrote {total} frames in {time.time() - t0:.1f} s", flush=True)
    del raw
    try:
        os.remove(raw_path)
    except OSError:
        pass
    import json

    with open(os.path.join(out_dir, "meta.json.part"), "w") as f:
        json.dump({"layout": LAYOUT, "top": top, "metric": bool(a.metric)}, f)
    os.replace(os.path.join(out_dir, "meta.json.part"), os.path.join(out_dir, "meta.json"))
    say("frame", total, total)
    say("done")
    return 0


def sdpa_forward(self, x, attn_bias=None):
    """DINOv2's attention, exactly (`Attention.forward`'s maths), through
    torch's fused kernel instead of a materialised attention matrix."""
    import torch.nn.functional as F

    assert attn_bias is None, "nested tensors need xformers"
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2]).transpose(1, 2).reshape(B, N, C)
    return self.proj_drop(self.proj(x))


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
    if device == "mps":
        return f", GPU memory {torch.mps.driver_allocated_memory() / 2**30:.1f} GB"
    return ""


def cv2_cubic() -> int:
    import cv2

    return cv2.INTER_CUBIC

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        say("error", f"{type(e).__name__}: {first_line(e)}")
        raise
