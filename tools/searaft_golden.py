#!/usr/bin/env python3
"""Run the REAL SEA-RAFT (PyTorch) on a deterministic frame pair and store
its inputs, intermediates and outputs, so `cargo test -p engine searaft`
can check the WGSL port end to end against torch.

usage:
  python3 tools/searaft_golden.py --repo <SEA-RAFT checkout> \
      --weights <model.safetensors> --cfg <repo>/config/eval/spring-M.json \
      --out data/searaft/golden.wifw

Needs torch and safetensors (no torchvision: the ImageNet init the repo
does at construction is skipped; the checkpoint overwrites it anyway).
The output is Wobble's .wifw container (engine/src/rife.rs), arch 1.
"""

import argparse
import json
import struct
import sys
import types
from pathlib import Path

import numpy as np
import torch


def write_wifw(path, tensors):
    with open(path, "wb") as f:
        f.write(b"WIFW")
        f.write(struct.pack("<IIII", 1, 1, 0, len(tensors)))
        for name, arr in tensors:
            arr = np.ascontiguousarray(arr, dtype="<f4")
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", arr.ndim))
            f.write(struct.pack(f"<{arr.ndim}I", *arr.shape))
            f.write(arr.tobytes())


def texture(h, w, dx, dy):
    """Value noise in three channels, moved by (dx, dy): the same family
    of picture the RIFE tests use."""
    rng = np.random.default_rng(7)
    out = np.zeros((3, h, w), np.float32)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    for c in range(3):
        acc = np.zeros((h, w), np.float32)
        for scale, amp in [(16, 0.5), (7, 0.3), (3, 0.2)]:
            g = rng.random((h // scale + 3, w // scale + 3)).astype(np.float32)
            fx = (xs - dx) / scale + 1
            fy = (ys - dy) / scale + 1
            x0 = np.floor(fx).astype(int)
            y0 = np.floor(fy).astype(int)
            tx = fx - x0
            ty = fy - y0
            tx = tx * tx * (3 - 2 * tx)
            ty = ty * ty * (3 - 2 * ty)
            x0 = np.clip(x0, 0, g.shape[1] - 2)
            y0 = np.clip(y0, 0, g.shape[0] - 2)
            a = g[y0, x0] * (1 - tx) + g[y0, x0 + 1] * tx
            b = g[y0 + 1, x0] * (1 - tx) + g[y0 + 1, x0 + 1] * tx
            acc += amp * (a * (1 - ty) + b * ty)
        out[c] = acc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.repo) / "core"))
    # utils.utils imports scipy for a helper the forward pass never calls.
    sys.modules.setdefault("scipy", types.ModuleType("scipy"))
    sys.modules.setdefault("scipy.interpolate", types.ModuleType("scipy.interpolate"))
    sys.modules["scipy"].interpolate = sys.modules["scipy.interpolate"]
    import extractor

    # The pretrained-trunk init needs torchvision and a download; skip it.
    extractor.ResNetFPN._init_weights = lambda self, a: None
    from raft import RAFT
    from safetensors.torch import load_file

    cfg = types.SimpleNamespace(**json.load(open(args.cfg)))
    model = RAFT(cfg)
    sd = load_file(args.weights)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # `downsample.1` IS `bn3` (one module, two names); the file stores bn3.
    missing = [k for k in missing if ".downsample.1." not in k]
    if missing:
        sys.exit(f"missing keys: {missing[:6]}")
    model.eval()

    # The 1/8 grid is pooled four times: it must survive 16x.
    h, w = 192, 256
    img1 = texture(h, w, 0.0, 0.0)
    img2 = texture(h, w, 3.0, -2.0)
    # The model takes 0..255.
    t1 = torch.from_numpy(img1 * 255.0)[None]
    t2 = torch.from_numpy(img2 * 255.0)[None]

    taps = {}
    with torch.no_grad():
        i1 = 2 * (t1 / 255.0) - 1.0
        i2 = 2 * (t2 / 255.0) - 1.0
        taps["fmap1"] = model.fnet(i1)[0]
        cnet = model.init_conv(model.cnet(torch.cat([i1, i2], dim=1)))
        taps["cnet"] = cnet[0]
        out = model(t1, t2, iters=cfg.iters, test_mode=True)
    flows = out["flow"]
    infos = out["info"]
    g = [
        ("golden.img1", img1),
        ("golden.img2", img2),
        ("golden.iters", np.array([cfg.iters], np.float32)),
        ("golden.fmap1", taps["fmap1"].numpy()),
        ("golden.cnet", taps["cnet"].numpy()),
        ("golden.flow0", flows[0][0].numpy()),
        ("golden.flow", flows[-1][0].numpy()),
        ("golden.info", infos[-1][0].numpy()),
    ]
    write_wifw(args.out, g)
    f = flows[-1][0].numpy()
    print(f"wrote {args.out}: {h}x{w}, mean flow ({f[0].mean():.3f}, {f[1].mean():.3f}), truth (3, -2)")


if __name__ == "__main__":
    main()
