"""wobble's driver for DiffusionLight-Turbo (Chinchuthakun, Phongthawee et
al., github.com/DiffusionLight/DiffusionLight-Turbo): an HDR latlong
environment estimated from one frame, by inpainting a chrome ball into it
at three exposures and unwrapping the ball.

Written next to an installed checkout by the app on every run and executed
from that directory with its own `uv` environment. Two modes, each started
by an explicit click in the app:

    uv run --project <checkout> python wobble_difflight.py --fetch <models dir>

        Downloads the weights from Hugging Face at pinned revisions into
        `<models dir>` (a Hugging Face cache of its own, not ~/.cache):
        SDXL base, its depth ControlNet, the Turbo and Exposure LoRAs and
        DPT-Large, fp16 safetensors only. This is the only network use;
        a run is offline.

    uv run --project <checkout> python wobble_difflight.py --shot <dir> \
        --models <models dir> --out <name> [--seed 200] [--steps 30] \
        [--exposure 0] [--height 256] [--stride 1] [--smooth 0] [--no-follow]

        The shot directory is what wobble rendered: `Input/f_NNNNN.exr`,
        scene-linear Rec.709. Two passes:

        1. **estimate** — every `--stride`th frame (and the last): the
           frame is exposed, sRGB encoded, letterboxed to 1024x1024 (the
           tool's own advice), a ball inpainted at EV 0, -2.5 and -5 (one
           seed for all three and for every frame), each unwrapped with
           the tool's own `ball2envmap` maths and merged with its
           `exposure2hdr` maths. Lands in `Raw/` under the INPUT's name,
           and a frame already there is not estimated again: a cancelled
           run resumes, and a new stride or smoothing costs no diffusion.
        2. **smooth** — every input frame gets a latlong in `<out>/`,
           half float, scene-linear Rec.709, in the PLATE's units (the
           exposure is divided back out). See `smooth()`.

The latlong is the tool's (Blender's equirectangular): the centre is where
the camera looks, the left and right edges are behind it, up is up. It is
relative to the CAMERA of each frame.

The inpainting is the authors' own diffusers port
(github.com/DiffusionLight/Diffusionlight-turbo-diffusers, MIT), kept
step for step: UniPC, 30 steps, the Turbo LoRA down to timestep 800 and
the Exposure LoRA at 0.75 after it, the bilinear mask latents, depth
ControlNet at 0.5. The research checkout pins torch 2.0 + CUDA 11.8, which
no current GPU driver runs, so the environment is ours (`PYPROJECT` in the
app) and only the projection is imported from the checkout.

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
sys.path.insert(0, os.getcwd())

# Pinned: the same revisions every run, whatever upstream does next.
# (repo, revision, the files a run needs)
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET = "diffusers/controlnet-depth-sdxl-1.0"
TURBO_LORA = "DiffusionLight/Flickr2K"
EXPOSURE_LORA = "DiffusionLight/DiffusionLight"
DPT = "Intel/dpt-large"
PINS = {
    SDXL: "462165984030d82259a11f4367a4eed129e94a7b",
    CONTROLNET: "17bb97973f29801224cd66f192c5ffacf82648b4",
    TURBO_LORA: "ed635477e5a742e363e54a6ed2193a9a43153113",
    EXPOSURE_LORA: "bda34823e932601b3eb9aff0276f00925cdeea46",
    DPT: "bc15f29aa3a80d532f2ed650b5e16ac48d8958f9",
}
FILES = {
    SDXL: [
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "tokenizer_2/*",
        "text_encoder/config.json",
        "text_encoder/model.fp16.safetensors",
        "text_encoder_2/config.json",
        "text_encoder_2/model.fp16.safetensors",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    ],
    CONTROLNET: ["config.json", "diffusion_pytorch_model.fp16.safetensors"],
    TURBO_LORA: ["pytorch_lora_weights.safetensors"],
    EXPOSURE_LORA: ["pytorch_lora_weights.safetensors"],
    DPT: ["config.json", "preprocessor_config.json", "model.safetensors"],
}
LORA_FILE = "pytorch_lora_weights.safetensors"
# What `--fetch` leaves behind when everything is there; the app reads it.
READY = "READY"

# The authors' settings (their diffusers port).
ALL_EVS = [0.0, -2.5, -5.0]
LOWEST_EV = -5.0
SWITCH_LORA_TIMESTEP = 800
PROMPT = "a perfect mirrored reflective chrome ball sphere"
PROMPT_BLACK = "a perfect black dark mirrored reflective chrome ball sphere"
NEGATIVE_PROMPT = "matte, diffuse, flat, dull"
SIDE = 1024
BALL = 256
BALL_AT = (SIDE - BALL) // 2
# Latents of context decoded round the ball (see `decode_ball`).
MARGIN = 16
GAMMA = 2.4
LUMA = (0.212671, 0.715160, 0.072169)


def say(*words: object) -> None:
    print("WOBBLE", *words, flush=True)


def first_line(e: BaseException) -> str:
    s = str(e).strip()
    return s.splitlines()[0] if s else type(e).__name__


def ready_text() -> str:
    return "".join(f"{repo} {rev}\n" for repo, rev in PINS.items())


def fetch(models: str) -> int:
    os.environ["HF_HOME"] = models
    from huggingface_hub import snapshot_download

    os.makedirs(models, exist_ok=True)
    say("frame", 0, len(PINS))
    for i, (repo, rev) in enumerate(PINS.items()):
        print(f"fetching {repo} @ {rev[:12]}", flush=True)
        try:
            snapshot_download(repo_id=repo, revision=rev, allow_patterns=FILES[repo])
        except Exception as e:  # noqa: BLE001 - the message is the product here
            say("error", f"fetching {repo}: {first_line(e)}")
            return 1
        say("frame", i + 1, len(PINS))
    with open(os.path.join(models, READY), "w") as f:
        f.write(ready_text())
    say("done")
    return 0


def read_rgb(path: str):
    """An Input frame as float32 rgb."""
    import numpy as np
    import OpenEXR

    with OpenEXR.File(path, separate_channels=True) as f:
        by = {k.rsplit(".", 1)[-1]: v.pixels for k, v in f.channels().items()}
    if not all(c in by for c in "RGB"):
        raise RuntimeError(f"{os.path.basename(path)}: no R, G and B channels")
    px = np.stack([by["R"], by["G"], by["B"]], -1).astype(np.float32)
    return np.nan_to_num(px, nan=0.0, posinf=0.0, neginf=0.0)


def write_rgb(path: str, rgb) -> None:
    """Half-float RGBA (alpha 1), ZIP, written beside `path` and renamed
    into place, so a run stopped mid-write never leaves a frame that looks
    finished."""
    import numpy as np
    import OpenEXR

    px = np.clip(np.nan_to_num(rgb, nan=0.0, posinf=65000.0, neginf=0.0), 0.0, 65000.0).astype(np.float16)
    chans = {c: np.ascontiguousarray(px[:, :, i]) for i, c in enumerate("RGB")}
    chans["A"] = np.ones(px.shape[:2], np.float16)
    tmp = path + ".part"
    with OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}, chans) as f:
        f.write(tmp)
    os.replace(tmp, path)


def srgb_encode(v):
    import numpy as np

    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)


def plate_mean(path: str, gain: float):
    """What the model was shown of this frame, per channel, in the plate's
    units: the mean of the frame clipped at the exposure's white. This is
    the plate's own word on how bright and what colour the frame is."""
    import numpy as np

    px = np.clip(read_rgb(path), 0.0, 1.0 / gain)
    return px.reshape(-1, 3).mean(0).astype(np.float64)


def letterbox(rgb, gain: float):
    """The frame exposed and sRGB encoded, fitted inside 1024x1024 on
    black (the tool's advice for a frame that is not square)."""
    import numpy as np
    from PIL import Image

    ldr = (srgb_encode(rgb * gain) * 255.0 + 0.5).astype(np.uint8)
    im = Image.fromarray(ldr)
    k = SIDE / max(im.size)
    w, h = max(1, round(im.size[0] * k)), max(1, round(im.size[1] * k))
    im = im.resize((w, h), Image.LANCZOS)
    out = Image.new("RGB", (SIDE, SIDE))
    out.paste(im, ((SIDE - w) // 2, (SIDE - h) // 2))
    return out


def circle(size: int):
    import torch

    x = torch.linspace(-1, 1, size)
    y = torch.linspace(1, -1, size)
    y, x = torch.meshgrid(y, x, indexing="ij")
    return ((1 - x**2 - y**2) >= 0).numpy()


def lora_weights(repo: str):
    """A LoRA's weights off the local cache, as diffusers names them. The
    published files carry the prefix twice (`unet.unet.…`, the trainer's
    wrapper), which only some diffusers releases strip on their own."""
    from huggingface_hub import try_to_load_from_cache
    from safetensors.torch import load_file

    path = try_to_load_from_cache(repo, LORA_FILE, revision=PINS[repo])
    if not isinstance(path, str):
        raise RuntimeError(f"{repo} is not fetched: press fetch weights…")
    return {(k[5:] if k.startswith("unet.unet.") else k): v for k, v in load_file(path).items()}


class Estimator:
    """The authors' diffusers port, as an object: built once, called per
    frame."""

    def __init__(self, models: str, steps: int, seed: int):
        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline, UniPCMultistepScheduler
        from transformers import pipeline

        self.torch = torch
        self.steps = steps
        self.seed = seed
        cuda = torch.cuda.is_available()
        mps = not cuda and torch.backends.mps.is_available()
        self.device = "cuda" if cuda else ("mps" if mps else "cpu")
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        name = {"cuda": "CUDA", "mps": "Apple Silicon (MPS)", "cpu": "the CPU (very slow)"}[self.device]
        print(f"loading SDXL + the depth ControlNet + the DiffusionLight LoRAs on {name}", flush=True)
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET, revision=PINS[CONTROLNET], variant="fp16", use_safetensors=True, torch_dtype=dtype
        )
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            SDXL,
            revision=PINS[SDXL],
            controlnet=controlnet,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=dtype,
            add_watermarker=False,
        ).to(self.device)
        pipe.load_lora_weights(lora_weights(TURBO_LORA), adapter_name="turbo")
        pipe.load_lora_weights(lora_weights(EXPOSURE_LORA), adapter_name="exposure")
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)

        # DiffusionLight needs the mask latents interpolated bilinearly
        # (diffusers uses nearest): the authors' monkey patch.
        original = pipe.prepare_mask_latents

        def bilinear_mask_latents(self_, *args, **kwargs):
            args = list(args)
            args[0] = torch.nn.functional.interpolate(
                args[0],
                size=(args[3] // self_.vae_scale_factor, args[4] // self_.vae_scale_factor),
                mode="bilinear",
                align_corners=False,
            )
            return original(*args, **kwargs)

        pipe.prepare_mask_latents = bilinear_mask_latents.__get__(pipe, pipe.__class__)
        self.pipe = pipe
        self.depth = pipeline(
            task="depth-estimation", model=DPT, revision=PINS[DPT], device=self.device, torch_dtype=dtype
        )

        # The two prompts, blended by exposure: 0 EV the ball, the lowest
        # the black ball.
        pe, ne, ppe, npe = pipe.encode_prompt(PROMPT, negative_prompt=NEGATIVE_PROMPT)
        pb, _, ppb, _ = pipe.encode_prompt(PROMPT_BLACK, negative_prompt=NEGATIVE_PROMPT)
        self.embeds = []
        for ev in ALL_EVS:
            r = (ev - LOWEST_EV) / (0 - LOWEST_EV)
            self.embeds.append((pe * r + pb * (1 - r), ppe * r + ppb * (1 - r)))
        self.negative = (ne, npe)
        # The prompts never change, so the text encoders are done: 1.6 GB
        # of VRAM back for a card that is shared with wobble itself.
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
        if self.device == "cuda":
            torch.cuda.empty_cache()

        import numpy as np
        from PIL import Image

        # The inpainting mask: the ball's disc, grown 10 px all round.
        m = np.zeros((SIDE, SIDE), np.uint8)
        m[BALL_AT - 10 : BALL_AT + BALL + 10, BALL_AT - 10 : BALL_AT + BALL + 10] = circle(BALL + 20) * 255
        self.mask = Image.fromarray(m)
        self.disc = circle(BALL)

    def lora(self, name: str, scale: float) -> None:
        self.pipe.unfuse_lora()
        self.pipe.set_adapters(name)
        self.pipe.fuse_lora(lora_scale=scale)

    def balls(self, image):
        """`image` (a 1024x1024 PIL picture) -> the ball at each exposure,
        float [3, 3, 256, 256] on the device."""
        import numpy as np
        from PIL import Image

        torch = self.torch
        depth = np.asarray(self.depth(images=image)["depth"]).copy()
        s = slice(BALL_AT, BALL_AT + BALL)
        # The ball is the nearest thing in the picture.
        depth[s, s] = depth[s, s] * (1 - self.disc) + self.disc * 255
        control = Image.fromarray(depth)
        out = []
        for (embeds, pooled) in self.embeds:
            swapped = [False]

            def swap(pipe, i, t, kw, swapped=swapped):
                # t runs from 999 down; the Exposure LoRA takes over once.
                if not swapped[0] and t <= SWITCH_LORA_TIMESTEP:
                    self.lora("exposure", 0.75)
                    swapped[0] = True
                return kw

            self.lora("turbo", 1.0)
            # One seed for every exposure and every frame: the method is
            # sensitive to it, and a fixed one is the steadiest over time.
            gen = torch.Generator(device=self.device if self.device == "cuda" else "cpu").manual_seed(self.seed)
            res = self.pipe(
                prompt_embeds=embeds,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=self.negative[0],
                negative_pooled_prompt_embeds=self.negative[1],
                num_inference_steps=self.steps,
                image=image,
                mask_image=self.mask,
                control_image=control,
                controlnet_conditioning_scale=0.5,
                callback_on_step_end=swap,
                generator=gen,
                output_type="latent",
            ).images
            out.append(self.decode_ball(res))
        return torch.stack(out)

    def decode_ball(self, latents):
        """The ball alone, decoded from the latents around it. Only the
        ball is ever read, and SDXL's VAE has to run in float32: decoding
        the whole 1024x1024 there takes more memory than the diffusion
        itself and is what overflowed a 16 GB card. A margin of `MARGIN`
        latents (8 px each) of the picture round the ball goes in with it,
        so the decoder sees the ball's surroundings as it would have."""
        torch = self.torch
        vae = self.pipe.vae
        a, b = BALL_AT // 8 - MARGIN, (BALL_AT + BALL) // 8 + MARGIN
        crop = latents[:, :, a:b, a:b]
        half = vae.dtype == torch.float16 and vae.config.force_upcast
        if half:
            self.pipe.upcast_vae()
        crop = crop.to(next(iter(vae.post_quant_conv.parameters())).dtype)
        px = vae.decode(crop / vae.config.scaling_factor, return_dict=False)[0]
        if half:
            vae.to(dtype=torch.float16)
        m = MARGIN * 8
        return (px[0, :, m : m + BALL, m : m + BALL].float() / 2 + 0.5).clamp(0, 1)


class Unwrap:
    """Ball -> latlong, with the tool's own `ball2envmap` maths (imported
    from the checkout), the lookup grid built once."""

    def __init__(self, height: int, device: str):
        import numpy as np
        import torch
        from ball2envmap import create_envmap_grid, get_cartesian_from_spherical, get_normal_vector

        self.height = height
        # The tool samples at 4x its 256 rows and filters down: 1024 rows.
        rows = max(1024, height)
        g = create_envmap_grid(rows)
        reflect = get_cartesian_from_spherical(g[..., 1], g[..., 0])
        normal = get_normal_vector(np.array([1, 0, 0])[None, None], reflect)
        pos = (1.0 - (normal + 1.0) / 2)[..., 1:]
        self.grid = (torch.from_numpy(pos)[None].float() * 2 - 1).to(device)

    def __call__(self, balls):
        import torch.nn.functional as F

        n = balls.shape[0]
        env = F.grid_sample(balls, self.grid.expand(n, -1, -1, -1), mode="bilinear", padding_mode="border", align_corners=True)
        if env.shape[2] != self.height:
            env = F.interpolate(env, size=(self.height, self.height * 2), mode="area")
        return env


def merge_hdr(envs):
    """Three LDR latlongs (EV 0, -2.5, -5; [3, 3, H, W] in [0, 1]) -> one
    linear HDR [H, W, 3]. The tool's `exposure2hdr.py`, line for line: each
    is decoded at gamma 2.4 and scaled by its exposure, and luminance is
    taken from the darker bracket wherever the brighter one nears its
    white."""
    import torch

    luma = torch.tensor(LUMA, device=envs.device).view(1, 3, 1, 1)
    lin = [envs[i : i + 1] ** GAMMA / (2.0 ** ALL_EVS[i]) for i in range(len(ALL_EVS))]
    lum = [(x * luma).sum(1, keepdim=True) for x in lin]
    out = lum[-1]
    for i in range(len(ALL_EVS) - 1, 0, -1):
        white = 1.0 / (2.0 ** ALL_EVS[i - 1])
        p1 = ((lum[i - 1] - 0.9 * white) / (0.1 * white)).clamp(0, 1)
        p2 = (out > lum[i - 1]).float()
        mask = p1 * p2
        out = lum[i - 1] * (1 - mask) + out * mask
    hdr = lin[0] * (out / (lum[0] + 1e-10))
    return hdr[0].permute(1, 2, 0)


def sampled(n: int, stride: int) -> list[int]:
    """The frames that are estimated: every `stride`th, and the last."""
    return sorted(set(range(0, n, max(1, stride))) | {n - 1})


def smooth(inputs, raw_dir, out_dir, picks, sigma, follow, gain, progress):
    """Every input frame's latlong, from the estimated ones around it.

    The estimate flickers because each frame's ball is its own
    hallucination of what is behind the camera; the LIGHTING flickers
    because it really does (a strobe, a colour wash). A plain average over
    time cannot tell them apart and would smear the strobe. So, with
    `follow`:

        latlong(t) = G[E / mean E](t)  x  G[mean E / P](t)  x  P(t)

    where E is an estimated latlong, `mean` its per-channel mean, P the
    PLATE's per-channel mean on that frame and G a Gaussian over the
    estimated frames (sigma in frames). The picture of the surroundings
    and how much brighter they are than the frame are steadied; how bright
    and what colour THIS frame is comes from the plate, unsmoothed. With no
    smoothing and every frame estimated it is the estimate itself.

    Without `follow` it is the plain Gaussian, G[E](t).

    Frames between estimates (`--stride`) are filled the same way, with at
    least the Gaussian that reaches the neighbours.
    """
    import numpy as np

    n = len(inputs)
    stride = max(1, max((b - a for a, b in zip(picks, picks[1:])), default=1))
    s = max(sigma, 0.5 * stride if stride > 1 else 0.0)
    reach = int(np.ceil(3 * s)) if s > 0 else 0
    cache: dict[int, object] = {}

    def raw(k: int):
        if k not in cache:
            cache[k] = read_rgb(os.path.join(raw_dir, os.path.basename(inputs[k]))).astype(np.float64)
        return cache[k]

    plate = [plate_mean(p, gain) for p in inputs] if follow else None
    # A black frame has no colour to follow: a floor well under the clip's
    # own level keeps the ratio finite.
    floor = (1e-3 * float(np.mean([p @ np.array(LUMA) for p in plate])) + 1e-7) if follow else 0.0
    os.makedirs(out_dir, exist_ok=True)
    for t in range(n):
        if s > 0:
            near = [k for k in picks if abs(k - t) <= reach]
            w = np.array([np.exp(-((k - t) ** 2) / (2 * s * s)) for k in near])
        else:
            near, w = [t], np.array([1.0])
        w = w / w.sum()
        for k in [k for k in cache if k < t - reach]:
            del cache[k]
        if follow:
            shape = 0.0
            ratio = np.zeros(3)
            for k, wk in zip(near, w):
                e = raw(k)
                m = np.maximum(e.reshape(-1, 3).mean(0), 1e-9)
                shape = shape + wk * (e / m)
                ratio += wk * m / (plate[k] + floor)
            out = shape * (ratio * (plate[t] + floor))
        else:
            out = sum(wk * raw(k) for k, wk in zip(near, w))
        write_rgb(os.path.join(out_dir, os.path.basename(inputs[t])), out)
        progress(t + 1)


def run(a: argparse.Namespace) -> int:
    os.environ["HF_HOME"] = a.models
    os.environ["HF_HUB_OFFLINE"] = "1"
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
    todo = [k for k in picks if not os.path.isfile(os.path.join(raw_dir, os.path.basename(inputs[k])))]
    total = len(todo) + n
    say("frame", 0, total)
    print(f"{n} frames: {len(picks)} to estimate ({len(picks) - len(todo)} already there), every one smoothed", flush=True)
    if todo:
        import time

        import torch

        with torch.inference_mode():
            est = Estimator(a.models, a.steps, a.seed)
            unwrap = Unwrap(a.height, est.device)
            for i, k in enumerate(todo):
                t0 = time.time()
                balls = est.balls(letterbox(read_rgb(inputs[k]), gain))
                hdr = merge_hdr(unwrap(balls)) / gain
                write_rgb(os.path.join(raw_dir, os.path.basename(inputs[k])), hdr.cpu().numpy())
                if i == 0:
                    used = f", {torch.cuda.max_memory_allocated() / 2**30:.1f} GB of VRAM" if est.device == "cuda" else ""
                    print(f"first frame in {time.time() - t0:.1f} s{used}", flush=True)
                say("frame", i + 1, total)
            del est, unwrap
    print(f"smoothing (sigma {a.smooth:g} frames{', following the plate' if not a.no_follow else ''})", flush=True)
    smooth(
        inputs,
        raw_dir,
        os.path.join(a.shot, a.out),
        picks,
        a.smooth,
        not a.no_follow,
        gain,
        lambda done: say("frame", len(todo) + done, total),
    )
    say("done")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", metavar="MODELS_DIR")
    ap.add_argument("--shot")
    ap.add_argument("--models")
    ap.add_argument("--out", default="Env")
    ap.add_argument("--seed", type=int, default=200)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--exposure", type=float, default=0.0, help="stops applied to the plate before the model sees it")
    ap.add_argument("--height", type=int, default=256, help="the latlong's rows (twice as many columns)")
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
