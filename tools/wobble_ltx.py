"""wobble's driver for LTX-2.5 video generation on this machine (Lightricks,
github.com/Lightricks/LTX-2, the DistilledPipeline in ltx_pipelines.distilled).

Written next to an installed checkout by the app on every run and executed
with that checkout's own environment (`.venv`, never `uv run`, which would
re-sync the environment and drop the optional NVFP4 kernels). Two modes,
each started by an explicit click in the app:

    python wobble_ltx.py --fetch <models dir>

        Downloads the five weight files from Hugging Face at a pinned
        revision. The repository is GATED: the user accepts the LTX-2.x
        Community Licence on the model page and logs in (`hf auth login`,
        or HF_TOKEN) first. This is the only network use.

    python wobble_ltx.py --job <job.json> --models <models dir> --out <file.mp4>

        The job is the GenAI step's request: `prompt`, `image` (a path to
        the start frame, or null for text-to-video), `seconds`, `width`,
        `height`, `seed`, and optionally `fps` (24). The clip is written
        to --out as an H.264 mp4.

The transformer is the NVFP4 distilled checkpoint: on an RTX 50-series it
runs natively (`--quantization nvfp4-prequant`), its weights streamed from
system RAM (`--offload cpu`) so a 16 GB card is enough. Elsewhere it cannot
run, and the driver says so rather than falling back to something slower
behind the user's back.

Progress is one machine-readable line per event on stdout:

    WOBBLE frame <done> <total>
    WOBBLE done
    WOBBLE error <message>

Everything else printed is shown as a log tail.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())

# Pinned: the same revision every run, whatever upstream does next.
REPO = "Lightricks/LTX-2.5"
REV = "49b093a39c19169a8a81721dcc541e07e945232f"
TRANSFORMER = "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"
TEXT_ENCODER = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
VIDEO_VAE = "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
AUDIO_VAE = "vae/ltx-2.5-audio-vae-bf16.safetensors"
UPSAMPLER = "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
FILES = [TRANSFORMER, TEXT_ENCODER, VIDEO_VAE, AUDIO_VAE, UPSAMPLER]

# The two-stage pipeline denoises at half size and upsamples by two, so the
# output must be a multiple of 64 on each side.
GRID = 64


def say(*words: object) -> None:
    print("WOBBLE", *words, flush=True)


def fetch(models: str) -> int:
    from huggingface_hub import hf_hub_download

    os.makedirs(models, exist_ok=True)
    say("frame", 0, len(FILES))
    for i, name in enumerate(FILES):
        print(f"fetching {REPO} {name} @ {REV[:12]}", flush=True)
        try:
            hf_hub_download(repo_id=REPO, filename=name, revision=REV, local_dir=models)
        except Exception as e:  # noqa: BLE001 - the message is the product here
            text = str(e)
            if "401" in text or "403" in text or "gated" in text.lower() or "restricted" in text.lower():
                say(
                    "error",
                    f"{REPO} is gated: accept the licence at https://huggingface.co/{REPO} and log in "
                    "(hf auth login, or set HF_TOKEN), then fetch again",
                )
            else:
                say("error", f"fetching {name}: {text.splitlines()[0] if text else e!r}")
            return 1
        say("frame", i + 1, len(FILES))
    say("done")
    return 0


def frames_for(seconds: float, fps: float) -> int:
    # 8k+1 frames, at least nine.
    n = max(9, round(seconds * fps))
    return (n - 1) // 8 * 8 + 1


def snap(v: int) -> int:
    return max(GRID, round(v / GRID) * GRID)


def run(a: argparse.Namespace) -> int:
    with open(a.job) as f:
        job = json.load(f)
    paths = [os.path.join(a.models, f) for f in FILES]
    for p in paths:
        if not os.path.isfile(p):
            say("error", f"missing weights: {p} (fetch the LTX weights first)")
            return 1

    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        say("error", "local LTX needs an NVIDIA RTX 50-series (Blackwell) GPU: the NVFP4 transformer runs nowhere else")
        return 1
    try:
        import ltx_kernels.nvfp4  # noqa: F401
    except ImportError:
        say("error", "the NVFP4 kernels are not built in this environment: reinstall LTX-2 from the panel")
        return 1

    fps = float(job.get("fps") or 24)
    num_frames = frames_for(float(job.get("seconds") or 5), fps)
    width, height = snap(int(job["width"])), snap(int(job["height"]))
    argv = [
        "wobble_ltx",
        "--transformer-path", paths[0],
        "--text-encoder-path", paths[1],
        "--video-vae-path", paths[2],
        "--audio-vae-path", paths[3],
        "--spatial-upsampler-path", paths[4],
        "--quantization", "nvfp4-prequant",
        "--offload", a.offload,
        "--prompt", job.get("prompt") or "",
        "--seed", str(int(job.get("seed") or 0)),
        "--num-frames", str(num_frames),
        "--frame-rate", f"{fps:g}",
        "--width", str(width),
        "--height", str(height),
        "--output-path", a.out,
    ]
    image = job.get("image")
    if image:
        if not os.path.isfile(image):
            say("error", f"the start frame is missing: {image}")
            return 1
        argv += ["--image", image, "0", "1.0"]
    what = "image-to-video" if image else "text-to-video"
    print(f"LTX-2.5 distilled {what}: {width}x{height}, {num_frames} frames at {fps:g} fps, seed {job.get('seed')}", flush=True)
    say("frame", 0, 1)

    from ltx_pipelines import distilled

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.argv = argv
    with torch.inference_mode():
        distilled.main()
    if not os.path.isfile(a.out):
        say("error", "the pipeline finished without writing a clip")
        return 1
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"peak VRAM {peak:.1f} GiB", flush=True)
    say("frame", 1, 1)
    say("done")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", metavar="MODELS", help="download the weights into MODELS and exit")
    ap.add_argument("--job")
    ap.add_argument("--models")
    ap.add_argument("--out")
    ap.add_argument("--offload", default="cpu", choices=["none", "cpu", "disk"])
    a = ap.parse_args()
    if a.fetch:
        return fetch(a.fetch)
    if not (a.job and a.models and a.out):
        ap.error("--job, --models and --out are required (or --fetch)")
    try:
        return run(a)
    except Exception as e:  # noqa: BLE001 - report, then let the app show the log
        import traceback

        traceback.print_exc()
        say("error", f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
