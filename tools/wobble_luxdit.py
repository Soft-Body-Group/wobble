"""wobble's driver for LuxDiT (Liang et al., NVIDIA, github.com/nv-tlabs/LuxDiT):
an HDR latlong environment estimated from a clip by a video diffusion
transformer (a CogVideoX-5B fine-tune) that generates the environment as
two tone-mapped panoramas, merged to HDR by a small network of the authors'.

Written next to an installed checkout by the app on every run (with
`wobble_difflight.py`, whose smoothing pass and EXR helpers it shares) and
executed from that directory with its own `uv` environment. Two modes,
each started by an explicit click in the app:

    uv run --project <checkout> python wobble_luxdit.py --fetch <models dir>

        Downloads the weights from Hugging Face at pinned revisions into
        `<models dir>/<repo name>`: LuxDiT's video and image transformers,
        their LoRAs and the HDR merger, and only the VAE and scheduler of
        CogVideoX-5B-I2V (not its 9 GB text encoder: LuxDiT conditions on
        no prompt). About 24 GB. This is the only network use.

    uv run --project <checkout> python wobble_luxdit.py --shot <dir> \
        --models <models dir> --out <name> [--seed 33] [--steps 40] \
        [--exposure 0] [--height 256] [--window 25] [--stride 1] \
        [--smooth 0] [--no-follow]

        `<dir>/Input/*.exr` is what wobble rendered, scene-linear Rec.709.

        1. **estimate** — the frames picked by `--stride` go through the
           model in WINDOWS of `--window` frames (4k+1: the VAE packs four
           frames a latent), each window overlapping the last by a quarter
           and cross-faded into it, one seed for every window. A window is
           a video to the model: the environment is consistent within it,
           which is the point of a video model; the cross-fade carries it
           across windows. A one-frame range runs the IMAGE checkpoint.
           The frame is exposed, sRGB encoded and resized and centre
           cropped to 480x720 (the model's own preprocessing; a wider plate
           loses its sides). The two 128x256 panoramas are merged by the
           authors' MLP, rotated and flipped as their `hdr_merger.py`
           does, resized to `--height` rows and divided by the exposure.
           Lands in `Raw/` under the input's name; a window whose frames
           are all there is not run again.
        2. **smooth** — `wobble_difflight.smooth`, unchanged: the same
           follow-the-plate maths over the estimates.

The latlong is Blender's equirectangular, as the DiffusionLight driver
writes it: the centre is where the camera looks. It is relative to each
frame's CAMERA.

Licence: LuxDiT's code and weights are under the NVIDIA OneWay
Noncommercial License (research and evaluation only); CogVideoX's VAE
under the CogVideoX License. wobble tags the node accordingly.

Progress is one machine-readable line per event on stdout:

    WOBBLE frame <done> <total>
    WOBBLE done
    WOBBLE error <message>
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.getcwd())

COGVIDEOX = "zai-org/CogVideoX-5b-I2V"
LUXDIT = "nvidia/LuxDiT"
# (repo, revision); the files a run needs are FILES[repo].
PINS = {
    COGVIDEOX: "a6f0f4858a8395e7429d82493864ce92bf73af11",
    LUXDIT: "6c10bcbbe11439660d4bad514250e6d30a32999d",
}
FILES = {
    COGVIDEOX: ["vae/*", "scheduler/*", "model_index.json", "LICENSE"],
    LUXDIT: ["hdr_merge_mlp/*", "luxdit_video/*", "luxdit_image/*", "LICENSE.md"],
}
SKIP = ["*.DS_Store", "*/.gitattributes"]
READY = "READY"
CONFIG = "configs/luxdit_base.yaml"
HEIGHT, WIDTH = 480, 720
ENV = (128, 256)
GUIDANCE = 2.5
LORA_SCALE = 0.8
# The view the 480x720 crop is taken to span, for the level calibration.
HFOV = 60.0
LUMA = (0.212671, 0.715160, 0.072169)


def say(*words: object) -> None:
    print("WOBBLE", *words, flush=True)


def first_line(e: BaseException) -> str:
    s = str(e).strip()
    return s.splitlines()[0] if s else type(e).__name__


def ready_text() -> str:
    return "".join(f"{repo} {rev}\n" for repo, rev in PINS.items())


def local(models: str, repo: str) -> str:
    return os.path.join(models, repo.split("/")[-1])


def fetch(models: str) -> int:
    from huggingface_hub import snapshot_download

    os.makedirs(models, exist_ok=True)
    say("frame", 0, len(PINS))
    for i, (repo, rev) in enumerate(PINS.items()):
        print(f"fetching {repo} @ {rev[:12]}", flush=True)
        try:
            snapshot_download(
                repo_id=repo, revision=rev, allow_patterns=FILES[repo], ignore_patterns=SKIP, local_dir=local(models, repo)
            )
        except Exception as e:  # noqa: BLE001 - the message is the product here
            say("error", f"fetching {repo}: {first_line(e)}")
            return 1
        say("frame", i + 1, len(PINS))
    with open(os.path.join(models, READY), "w") as f:
        f.write(ready_text())
    say("done")
    return 0


def windows(n: int, size: int) -> list[list[int]]:
    """Positions 0..n-1 cut into windows of `size` (4k+1) overlapping by a
    quarter; the last window ends on the last position. Short clips are one
    window."""
    if n <= size:
        return [list(range(n))]
    step = max(1, size - max(1, size // 4))
    out, a = [], 0
    while True:
        b = min(a + size, n)
        out.append(list(range(max(0, b - size), b)))
        if b == n:
            return out
        a += step


def padded(k: int) -> int:
    """The frame count the model takes for `k` frames: the next 4k+1."""
    return k if (k - 1) % 4 == 0 else k + (4 - (k - 1) % 4)


class Model:
    def __init__(self, models: str, video: bool):
        import torch
        from omegaconf import ListConfig, OmegaConf

        from diffusers.schedulers import CogVideoXDPMScheduler
        from luxdit_config import TrainingConfig
        from src.data.rendering_utils import envmap_vec
        from src.models.custom_autoencoder_kl_cogvideox import AutoencoderKLCogVideoX
        from src.models.custom_cogvideox_transformer_3d import CustomCogVideoXTransformer3DModel
        from src.pipelines.pipeline_cogvideox_rgbxenv import RGBXEnvCogVideoXPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("LuxDiT needs an NVIDIA GPU (CUDA)")
        self.device = "cuda"
        dtype = torch.bfloat16
        cfg = OmegaConf.merge(OmegaConf.structured(TrainingConfig), OmegaConf.load(CONFIG))
        self.pcfg = cfg.model_pipeline
        kw = {k: tuple(v) if isinstance(v, (list, ListConfig)) else v for k, v in dict(self.pcfg.transformer_kwargs).items()}
        base = local(models, COGVIDEOX)
        ckpt = os.path.join(local(models, LUXDIT), "luxdit_video" if video else "luxdit_image")
        vae = AutoencoderKLCogVideoX.from_pretrained(base, subfolder="vae", torch_dtype=dtype)
        transformer = CustomCogVideoXTransformer3DModel.from_pretrained(ckpt, torch_dtype=dtype, **kw)
        sched = CogVideoXDPMScheduler.from_pretrained(base, subfolder="scheduler")
        self.pipe = RGBXEnvCogVideoXPipeline(vae=vae, tokenizer=None, text_encoder=None, transformer=transformer, scheduler=sched)
        self.pipe.load_lora_weights(os.path.join(ckpt, "lora"), weight_name="model.safetensors", adapter_name="env-lora")
        self.pipe.to(self.device)
        for f in ("enable_slicing", "enable_tiling"):
            if hasattr(self.pipe.vae, f):
                getattr(self.pipe.vae, f)()
        self.env_nrm = (envmap_vec(ENV) * 0.5 + 0.5).numpy()
        cond = dict(self.pcfg.cond_images)
        cond.pop("env_ldr", None)
        cond.pop("env_log", None)
        self.cond = cond

    def estimate(self, frames, steps: int, seed: int):
        """sRGB frames [F, H, W, 3] in [0, 1] at 480x720 -> two lists of F
        PIL panoramas (LDR, log)."""
        import numpy as np
        import torch

        n = frames.shape[0]
        k = padded(n)
        if k > n:
            frames = np.concatenate([frames, np.repeat(frames[-1:], k - n, 0)], 0)
        example = {"rgb": frames, "env_nrm": np.repeat(self.env_nrm[None], k, 0)}
        additional = list(self.pcfg.get("additional_target_image", []))
        _, cond_images = self.pipe.example2input(example, additional, self.cond)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        out = self.pipe(
            prompt=None,
            cond_images=cond_images,
            cond_mapping=self.cond,
            guidance_scale=GUIDANCE,
            num_inference_steps=steps,
            generator=gen,
            height=HEIGHT,
            width=WIDTH,
            num_frames=k,
            additional_cond_labels=list(self.pcfg.get("additional_cond_labels", [])),
            attention_kwargs={"scale": LORA_SCALE},
            denoise_type=self.pcfg.get("denoise_type", "additional_hidden_states"),
            target_labels=list(self.pcfg.target_image),
            additional_target_labels=additional,
            num_latents=len(additional),
            separate_timesteps=self.pcfg.get("separate_timesteps", False),
            additional_rope_time_only=self.pcfg.get("additional_rope_time_only", True),
        ).frames
        return out[0][0][:n], out[1][0][:n]


class Merge:
    """The authors' `hdr_merger.py` for one pair: the MLP, the seam blend,
    the half-turn roll and the flip, then a resize to the latlong's size."""

    def __init__(self, models: str, rows: int, device: str):
        from src.models.hdr_model import HDR_MLP

        self.net = HDR_MLP.from_pretrained(os.path.join(local(models, LUXDIT), "hdr_merge_mlp")).to(device).eval()
        self.rows, self.device = rows, device

    def __call__(self, ldr, log):
        import cv2
        import numpy as np
        import torch

        from hdr_merger import seam_blending

        a = torch.from_numpy(np.asarray(ldr, np.float32) / 255.0 * 2.0 - 1.0).to(self.device)
        b = torch.from_numpy(np.asarray(log, np.float32) / 255.0 * 2.0 - 1.0).to(self.device)
        hdr = self.net(a, b).float().cpu().numpy()
        w = hdr.shape[1]
        hdr = seam_blending(hdr, blend_width=int(w * 0.02))
        hdr = np.flip(np.roll(hdr, w // 2, axis=1), axis=1)
        return cv2.resize(np.ascontiguousarray(hdr), (2 * self.rows, self.rows), interpolation=cv2.INTER_LINEAR)


def view_level(latlong, gain: float) -> float:
    """Mean luminance of the part of a latlong the camera sees (centre,
    HFOV x the crop's VFOV), clipped where the plate would clip."""
    import math

    import numpy as np

    h, w = latlong.shape[:2]
    vfov = 2 * math.degrees(math.atan(math.tan(math.radians(HFOV / 2)) * HEIGHT / WIDTH))
    dx = max(1, round(w * HFOV / 720.0))
    dy = max(1, round(h * vfov / 360.0))
    patch = np.clip(latlong[h // 2 - dy : h // 2 + dy, w // 2 - dx : w // 2 + dx], 0.0, 1.0 / gain)
    return float((patch.reshape(-1, 3) @ np.array(LUMA)).mean())


def plate_level(srgb, gain: float) -> float:
    """Mean luminance of what the model was shown, in the plate's units."""
    import numpy as np

    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return float((lin.reshape(-1, 3) @ np.array(LUMA)).mean()) / gain


def model_input(path: str, gain: float):
    """The frame as the model is shown it: exposed, sRGB, 480x720 crop."""
    import numpy as np

    from src.data.rendering_utils import resize_crop
    from wobble_difflight import read_rgb, srgb_encode

    return resize_crop(srgb_encode(read_rgb(path) * gain).astype(np.float32), (HEIGHT, WIDTH))


def run(a: argparse.Namespace) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    from wobble_difflight import read_rgb, sampled, smooth, write_rgb

    inputs = sorted(glob.glob(os.path.join(a.shot, "Input", "*.exr")))
    if not inputs:
        say("error", "no input frames")
        return 1
    try:
        with open(os.path.join(a.models, READY)) as f:
            fetched = f.read() == ready_text()
    except OSError:
        fetched = False
    if not fetched:
        say("error", "the weights are not fetched: press fetch weights…")
        return 1
    n = len(inputs)
    gain = 2.0**a.exposure
    raw_dir = os.path.join(a.shot, "Raw")
    os.makedirs(raw_dir, exist_ok=True)
    picks = sampled(n, a.stride)
    size = padded(max(1, min(a.window, 49)))
    wins = windows(len(picks), size)
    have = lambda k: os.path.isfile(os.path.join(raw_dir, os.path.basename(inputs[k])))  # noqa: E731
    todo = [w for w in wins if not all(have(picks[i]) for i in w)]
    est_total = sum(len(w) for w in todo)
    total = est_total + n
    say("frame", 0, total)
    print(
        f"{n} frames: {len(picks)} to estimate in {len(wins)} window(s) of up to {size} "
        f"({len(wins) - len(todo)} already there), every one smoothed",
        flush=True,
    )
    if todo:
        import time

        import numpy as np
        import torch

        with torch.inference_mode():
            model = Model(a.models, video=len(picks) > 1)
            merge = Merge(a.models, a.height, model.device)
            done = 0
            prev: set[int] = set()
            for wi, w in enumerate(todo):
                t0 = time.time()
                frames = np.stack([model_input(inputs[picks[i]], gain) for i in w])
                ldrs, logs = model.estimate(frames, a.steps, a.seed)
                hdrs = [merge(ldrs[j], logs[j]) / gain for j in range(len(w))]
                # The merger's HDR has a level of its own (several times the
                # plate's). Bring it to the plate's: the part of the latlong
                # the camera sees should read as bright as the frames do.
                # One factor for the window, so it adds no flicker.
                seen = float(np.mean([view_level(h, gain) for h in hdrs]))
                shown = float(np.mean([plate_level(f, gain) for f in frames]))
                k = shown / seen if seen > 1e-9 and shown > 0 else 1.0
                print(f"window {wi + 1}: level x{k:.3f}", flush=True)
                # Cross-fade into the frames the previous window wrote.
                over = [i for i in w if i in prev]
                for j, i in enumerate(w):
                    hdr = hdrs[j] * k
                    path = os.path.join(raw_dir, os.path.basename(inputs[picks[i]]))
                    if i in over and os.path.isfile(path):
                        t = (over.index(i) + 1) / (len(over) + 1)
                        hdr = (1.0 - t) * read_rgb(path) + t * hdr
                    write_rgb(path, hdr)
                prev = set(w)
                done += len(w)
                if wi == 0:
                    print(
                        f"first window ({len(w)} frames) in {time.time() - t0:.1f} s, "
                        f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB of VRAM",
                        flush=True,
                    )
                say("frame", done, total)
            del model, merge
    print(f"smoothing (sigma {a.smooth:g} frames{', following the plate' if not a.no_follow else ''})", flush=True)
    smooth(
        inputs,
        raw_dir,
        os.path.join(a.shot, a.out),
        picks,
        a.smooth,
        not a.no_follow,
        gain,
        lambda d: say("frame", est_total + d, total),
    )
    say("done")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", metavar="MODELS_DIR")
    ap.add_argument("--shot")
    ap.add_argument("--models")
    ap.add_argument("--out", default="Env")
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--exposure", type=float, default=0.0, help="stops applied to the plate before the model sees it")
    ap.add_argument("--height", type=int, default=256, help="the latlong's rows (twice as many columns)")
    ap.add_argument("--window", type=int, default=25, help="frames the model sees at once (4k+1)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--smooth", type=float, default=0.0, help="temporal Gaussian, sigma in frames")
    ap.add_argument("--no-follow", action="store_true")
    a = ap.parse_args()
    try:
        if a.fetch:
            return fetch(a.fetch)
        if not a.shot or not a.models:
            say("error", "--shot and --models are required")
            return 2
        return run(a)
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        say("error", first_line(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
