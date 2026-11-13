"""Reference numbers for engine/tests/hdrcnn.rs, from the upstream HDRCNN
graph (github.com/gabrieleilertsen/hdrcnn, network.py, BSD-3) written out in
NumPy and run on the same .npz files wobble reads.

TensorFlow 1 / TensorLayer do not install on a current machine, so this is
the graph re-implemented from network.py, op for op, with TF's semantics
spelled out where they matter:

- conv2d: cross-correlation, NHWC, filter [kh, kw, in, out], SAME padding;
- max_pool 2x2 stride 2, SAME (the sizes here are even: no padding);
- conv2d_transpose 4x4 stride 2, SAME, filter [kh, kw, out, in]: the
  gradient of a SAME conv of the 2N input, whose padding is 2 split 1/1,
  so output o takes input i through tap o - 2i + 1;
- TensorLayer's BatchNormLayer at inference: gamma (x - mean) /
  sqrt(var + 1e-5) + beta, parameters in the order beta, gamma,
  moving_mean, moving_variance;
- the skip connection's in-place `skip_layer.outputs = log(...)`: the map
  applies to what the concat sees, and the input layer's skip is the
  255-scaled RGB picture (not the BGR mean-subtracted one VGG reads);
- the leaky ReLUs have alpha 0: plain ReLU.

It is written independently of the Rust builder (engine/src/hdrcnn.rs) on
purpose: the test holds the two against each other.

    uv run --with numpy python tools/hdrcnn_reference.py \
        ~/.wobble/models > engine/tests/hdrcnn_ref.rs

Run once; the output is committed. np.load(allow_pickle=True) is fine here
— a developer's own run on a file whose hash the catalog pins — and is
exactly what the Rust reader refuses to do in the app. The input is a 64x64
picture the test builds with the same formula.
"""
import math
import os
import sys

import numpy as np

W, H, N = 64, 64, 256
MODELS = [("REG", "hdrcnn_params_compr_regularized.npz"), ("ORIG", "hdrcnn_params.npz")]
VGG_MEAN_BGR = [103.939, 116.779, 123.68]


def picture():
    """The network input x, 0..1: smooth colour, and a bright disc whose
    channels run past 1 before the clamp (a clipped highlight)."""
    a = np.zeros((H, W, 3), dtype=np.float64)
    for y in range(H):
        for x in range(W):
            d = math.hypot(x - 40.0, y - 24.0)
            for c in range(3):
                v = 0.45 + 0.3 * math.sin(0.21 * x + 1.3 * c) * math.cos(0.17 * y - 0.4 * c)
                v += 1.4 * math.exp(-d * d / 60.0) * (1.0 - 0.25 * c)
                a[y, x, c] = min(max(v, 0.0), 1.0)
    return a.astype(np.float32)


def conv(x, w, b):
    kh, kw, cin, cout = w.shape
    ph, pw = (kh - 1) // 2, (kw - 1) // 2
    xp = np.pad(x, ((ph, kh - 1 - ph), (pw, kw - 1 - pw), (0, 0)))
    h, ww = x.shape[:2]
    out = np.zeros((h, ww, cout), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += xp[i : i + h, j : j + ww, :] @ w[i, j].astype(np.float64)
    return out + b


def pool(x):
    h, w, c = x.shape
    return x.reshape(h // 2, 2, w // 2, 2, c).max(axis=(1, 3))


def deconv(x, w, b):
    h, ww, cin = x.shape
    kh, kw, cout, _ = w.shape
    out = np.zeros((2 * h, 2 * ww, cout), dtype=np.float64)
    for ky in range(kh):
        oy = 2 * np.arange(h) + ky - 1
        vy = (oy >= 0) & (oy < 2 * h)
        for kx in range(kw):
            ox = 2 * np.arange(ww) + kx - 1
            vx = (ox >= 0) & (ox < 2 * ww)
            c = x @ w[ky, kx].astype(np.float64).T
            out[np.ix_(oy[vy], ox[vx])] += c[np.ix_(vy, vx)]
    return out + b


def bn(x, p):
    beta, gamma, mean, var = p
    return gamma * (x - mean) / np.sqrt(var + 1e-5) + beta


def relu(x):
    return np.maximum(x, 0.0)


def skip_map(s):
    return np.log((s / 255.0) ** 2 + 1.0 / 255.0)


def fuse(dec, skip, w, b):
    return conv(np.concatenate([dec, skip_map(skip)], axis=2), w, b)


def run(params, x):
    it = iter(params)

    def nxt():
        return next(it).astype(np.float64)

    x_in = 255.0 * x.astype(np.float64)
    r, g, bl = x_in[..., 0], x_in[..., 1], x_in[..., 2]
    net = np.stack([bl - VGG_MEAN_BGR[0], g - VGG_MEAN_BGR[1], r - VGG_MEAN_BGR[2]], axis=2)
    skips = [x_in]
    for n in [2, 2, 3, 3, 3]:
        for _ in range(n):
            net = relu(conv(net, nxt(), nxt()))
        skips.append(net)
        net = pool(net)
    net = relu(bn(conv(net, nxt(), nxt()), [nxt() for _ in range(4)]))
    # h1: deconv; h2..h5: fuse, deconv.
    net = relu(bn(deconv(net, nxt(), nxt()), [nxt() for _ in range(4)]))
    for s in [5, 4, 3, 2]:
        net = fuse(net, skips[s], nxt(), nxt())
        net = relu(bn(deconv(net, nxt(), nxt()), [nxt() for _ in range(4)]))
    net = fuse(net, skips[1], nxt(), nxt())
    net = relu(bn(conv(net, nxt(), nxt()), [nxt() for _ in range(4)]))
    net = fuse(net, skips[0], nxt(), nxt())
    try:
        next(it)
        raise SystemExit("parameters left over")
    except StopIteration:
        pass
    return net


def final(x, y):
    thr = 0.05
    a = np.clip((x.max(axis=2, keepdims=True) - 1.0 + thr) / thr, 0.0, 1.0)
    return (1 - a) * x.astype(np.float64) ** 2 + a * (np.exp(y) - 1.0 / 255.0)


def main():
    d = sys.argv[1]
    x = picture()
    print("// Generated by tools/hdrcnn_reference.py: numpy", np.__version__, "— do not edit.")
    print(f"pub const W: usize = {W};\npub const H: usize = {H};\npub const N: usize = {N};")
    for name, f in MODELS:
        path = os.path.join(d, f)
        if not os.path.isfile(path):
            print(f"// {f}: not found, skipped", file=sys.stderr)
            continue
        params = list(np.load(path, allow_pickle=True, encoding="latin1")["params"])
        y = run(params, x)
        fin = final(x, y)
        for tag, arr in [("Y", y), ("FINAL", fin)]:
            # Planar [3][H][W], like the Rust reference's output.
            flat = arr.transpose(2, 0, 1).reshape(-1)
            idx = [(k * 2654435761) % flat.size for k in range(N)]
            vals = ", ".join(f"{flat[i]:.7}" for i in idx)
            print(f"/// {f}: {tag} [3, {H}, {W}], mean {flat.mean():.7}")
            print(f"pub const {name}_{tag}_MEAN: f64 = {flat.mean():.9};")
            print(f"pub const {name}_{tag}: [f32; N] = [{vals}];")


main()
