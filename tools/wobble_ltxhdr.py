"""wobble's driver for LTX-2.5 SDR-to-HDR (Lightricks,
github.com/Lightricks/LTX-2, the HDRICLoraPipeline in ltx_pipelines.hdr_ic_lora).

Written next to an installed checkout by the app on every run and executed
from that directory with its own `uv` environment. Two modes, each started
by an explicit click in the app:

    uv run --project <checkout> python wobble_ltxhdr.py --fetch <models dir>

        Downloads the four weight files from Hugging Face at pinned
        revisions. Both repositories are GATED: the user accepts the
        LTX-2 Community Licence on each model page and logs in
        (`hf auth login`, or HF_TOKEN) first. This is the only network use.

    uv run --project <checkout> python wobble_ltxhdr.py --shot <dir> \
        --models <models dir> --fps 24 [--seed 42] [--scale 0.5] \
        [--high-quality] [--no-keyframes] [--offload none|cpu|disk]

        The shot directory is what wobble rendered: `Input/f_NNNNN.exr`,
        scene-linear ACEScg. The tool wants 8k+1 frames, so the clip is
        padded by holding its last frame (`Padded/`) and the result trimmed
        back. `--scale` under 1 runs the model on a smaller copy (area
        filtered; wobble stretches the result back over the plate). Each
        result frame lands in `HDR/` under the INPUT's name, half float,
        scene-linear ACEScg.

The pipeline is built exactly as the tool's own `main()` builds it, from
its own argument parser; the EXRs are then written here with the tool's
own ACEScct -> ACEScg decode, rather than through its HLG mp4 encoder (a
missing x265 there would lose the EXRs too).

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

# MPS lacks a few ops; let them fall back to the CPU rather than fail.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.getcwd())

# Pinned: the same revisions every run, whatever upstream does next.
BASE_REPO = "Lightricks/LTX-2.5"
BASE_REV = "49b093a39c19169a8a81721dcc541e07e945232f"
HDR_REPO = "Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR"
HDR_REV = "c7578ebddb54f4210e534e4d8cad698a5a520692"
FILES = [
    (BASE_REPO, BASE_REV, "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
    (BASE_REPO, BASE_REV, "vae/ltx-2.5-video-vae-bf16.safetensors"),
    (HDR_REPO, HDR_REV, "ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors"),
    (HDR_REPO, HDR_REV, "ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors"),
]


def say(*words: object) -> None:
    print("WOBBLE", *words, flush=True)


def model_path(models: str, name: str) -> str:
    return os.path.join(models, name)


def fetch(models: str) -> int:
    from huggingface_hub import hf_hub_download

    os.makedirs(models, exist_ok=True)
    say("frame", 0, len(FILES))
    for i, (repo, rev, name) in enumerate(FILES):
        print(f"fetching {repo} {name} @ {rev[:12]}", flush=True)
        try:
            hf_hub_download(repo_id=repo, filename=name, revision=rev, local_dir=models)
        except Exception as e:  # noqa: BLE001 - the message is the product here
            text = str(e)
            if "401" in text or "403" in text or "gated" in text.lower() or "restricted" in text.lower():
                say(
                    "error",
                    f"{repo} is gated: accept the licence at https://huggingface.co/{repo} and log in "
                    "(hf auth login, or set HF_TOKEN), then fetch again",
                )
            else:
                say("error", f"fetching {name}: {text.splitlines()[0] if text else e!r}")
            return 1
        say("frame", i + 1, len(FILES))
    say("done")
    return 0


def padded_count(n: int) -> int:
    # Matches engine::nodes::ltxhdr::LtxHdrParams::padded_frames.
    if n <= 1:
        return 1
    return -(-(n - 1) // 8) * 8 + 1


def run(a: argparse.Namespace) -> int:
    inputs = sorted(glob.glob(os.path.join(a.shot, "Input", "*.exr")))
    if not inputs:
        say("error", "no input frames")
        return 1
    n = len(inputs)
    m = padded_count(n)
    pad = os.path.join(a.shot, "Padded")
    os.makedirs(pad, exist_ok=True)
    for f in glob.glob(os.path.join(pad, "*.exr")):
        os.remove(f)
    if a.scale >= 0.999:
        for i in range(m):
            os.symlink(os.path.abspath(inputs[min(i, n - 1)]), os.path.join(pad, f"p_{i:05d}.exr"))
    else:
        import torch.nn.functional as F

        from ltx_pipelines.utils.media_io.exr import exr_colorspace_tag, read_exr, save_exr_tensor
        from ltx_pipelines.utils.media_io.color_config import EXRColorSpace

        cs = EXRColorSpace.ACESCG
        for i in range(n):
            px = read_exr(inputs[i]).float()  # [H, W, 3]
            hw = (max(32, round(px.shape[0] * a.scale)), max(32, round(px.shape[1] * a.scale)))
            small = F.interpolate(px.permute(2, 0, 1)[None], size=hw, mode="area")[0].permute(1, 2, 0)
            save_exr_tensor(small, os.path.join(pad, f"p_{i:05d}.exr"), True, cs.source_primaries, exr_colorspace_tag(cs))
        for i in range(n, m):
            os.symlink(os.path.join(pad, f"p_{n - 1:05d}.exr"), os.path.join(pad, f"p_{i:05d}.exr"))
    print(f"{n} frames, padded to {m} (8k+1) by holding the last, at {a.scale:g} of the plate", flush=True)
    say("frame", 0, n)

    names = [f for _, _, f in FILES]
    paths = [model_path(a.models, f) for f in names]
    for p in paths:
        if not os.path.isfile(p):
            say("error", f"missing weights: {p} (press fetch weights…)")
            return 1

    import torch

    from ltx_pipelines import hdr_ic_lora as h
    from ltx_pipelines.utils.media_io.color_config import EXRColorSpace, decode_hdr_video
    from ltx_pipelines.utils.media_io.exr import exr_colorspace_tag, save_exr_tensor

    cuda = torch.cuda.is_available()
    argv = [
        "--input", pad,
        "--input-colorspace", "acescg",
        "--frame-rate", str(a.fps),
        "--output-path", os.path.join(a.shot, "ltx", "hdr.mp4"),
        "--exr-colorspace", "acescg",
        "--transformer-path", paths[0],
        "--video-vae-path", paths[1],
        "--hdr-lora", paths[2],
        "--text-embeddings", paths[3],
        "--seed", str(a.seed),
        "--offload", a.offload,
    ]
    if a.high_quality:
        argv.append("--high-quality")
    if a.no_keyframes:
        argv.append("--no-keyframes")
    # The fp8 cast is a CUDA thing; elsewhere the transformer stays bf16.
    if not cuda:
        argv.append("--no-quantization")
    args = h.hdr_ic_lora_arg_parser().parse_args(argv)
    device = "CUDA" if cuda else ("Apple Silicon (MPS)" if torch.backends.mps.is_available() else "the CPU")
    print(f"loading LTX-2.5 + the HDR IC-LoRA on {device}", flush=True)

    with torch.inference_mode():
        pipeline = h.HDRICLoraPipeline(
            model_paths=args.model_paths,
            hdr_lora=args.hdr_lora,
            text_embeddings_path=args.text_embeddings,
            offload_mode=args.offload_mode,
            quantization=None if args.no_quantization else h._DEFAULT_QUANTIZATION,
            diffvae_optimization=args.diffvae_optimization,
        )
        print("generating (one pass over the whole clip)", flush=True)
        acescct, _fps = pipeline(
            video=args.input,
            seed=args.seed,
            tiling_config=h.AUTO_TILING,
            high_quality_hdr=args.high_quality,
            keyframe_strength=None if args.no_keyframes else args.keyframe_strength,
        )
        cs = EXRColorSpace.ACESCG
        out = os.path.join(a.shot, "HDR")
        os.makedirs(out, exist_ok=True)
        for i in range(n):
            frame = decode_hdr_video(acescct[i : i + 1], out_primaries=cs.source_primaries)[0]
            save_exr_tensor(
                frame.cpu(),
                os.path.join(out, os.path.basename(inputs[i])),
                True,
                cs.source_primaries,
                exr_colorspace_tag(cs),
            )
            say("frame", i + 1, n)
    say("done")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", metavar="MODELS_DIR")
    ap.add_argument("--shot")
    ap.add_argument("--models")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--high-quality", action="store_true")
    ap.add_argument("--no-keyframes", action="store_true")
    ap.add_argument("--offload", choices=["none", "cpu", "disk"], default="none")
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
        msg = str(e).splitlines()[0] if str(e) else repr(e)
        say("error", msg)
        return 1


if __name__ == "__main__":
    sys.exit(main())
