#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
# SPDX-License-Identifier: Apache-2.0
"""Train and quantize the demo network, then save everything the RTL test needs.

The network is a 16 -> 8 -> 10 multilayer perceptron over 4x4 handwritten digit
images, trained in float32 with plain NumPy (no deep-learning framework), then
quantized to the exact integer pipeline the accelerator implements:

  per-channel symmetric INT8 weights, per-tensor INT8 activations,
  INT32-domain bias, one Q0.16 multiplier and one rounding shift per channel

The input zero point is handled the way a real compiler does it, by folding
-zp_in * sum_r W[r][c] into the bias, so layer 2 can consume the ReLU output of
layer 1 with its zero point at -128 while the hardware only ever computes a
plain signed dot product.

Outputs (all committed so the demo runs with no network access and no
scikit-learn):
  test/data/digits4x4.npz    the dataset, downsampled to 4x4
  test/data/demo_model.npz   quantized weights, quantization parameters,
                             the INT8 test activations and the reference
                             INT8 layer outputs
  docs/demo_results.json     accuracy numbers used by the README and the plots

Dataset provenance: scikit-learn's `load_digits`, which is the UCI ML
hand-written digits test set of Alpaydin and Kaynak, 1797 8x8 images. Only the
downsampled copy is committed here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "test" / "data"
DOCS = REPO / "docs"
sys.path.insert(0, str(REPO / "test"))

import golden as g  # noqa: E402

CFG = g.Cfg(rows=4, cols=2, s_max=6, acc_w=24, m_w=16, sh_w=5)

N_IN, N_HID, N_OUT = 16, 8, 10
SEED = 20260725


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    cached = DATA / "digits4x4.npz"
    if cached.is_file():
        z = np.load(cached)
        print(f"using committed dataset {cached.name}: {z['x'].shape}")
        return z["x"], z["y"]

    try:
        from sklearn.datasets import load_digits
    except ImportError:
        sys.exit("no committed dataset and scikit-learn is unavailable; "
                 "install scikit-learn once to generate test/data/digits4x4.npz")

    d = load_digits()
    imgs = d.data.reshape(-1, 8, 8)
    # 2x2 mean pooling to 4x4, kept as integers in 0..16.
    small = imgs.reshape(-1, 4, 2, 4, 2).mean(axis=(2, 4))
    x = np.round(small).astype(np.uint8).reshape(-1, N_IN)
    y = d.target.astype(np.uint8)
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cached, x=x, y=y)
    print(f"generated {cached} from scikit-learn digits: {x.shape}")
    return x, y


def split(x: np.ndarray, y: np.ndarray):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(x))
    n_test = len(x) // 5
    test, train = perm[:n_test], perm[n_test:]
    return (x[train], y[train]), (x[test], y[test])


# ---------------------------------------------------------------------------
# Float training
# ---------------------------------------------------------------------------
def train(xtr: np.ndarray, ytr: np.ndarray, epochs: int = 400):
    rng = np.random.default_rng(SEED + 1)
    w1 = rng.normal(0, np.sqrt(2.0 / N_IN), (N_IN, N_HID))
    b1 = np.zeros(N_HID)
    w2 = rng.normal(0, np.sqrt(2.0 / N_HID), (N_HID, N_OUT))
    b2 = np.zeros(N_OUT)

    onehot = np.eye(N_OUT)[ytr]
    lr, batch = 0.15, 32
    v = [np.zeros_like(t) for t in (w1, b1, w2, b2)]
    for ep in range(epochs):
        idx = rng.permutation(len(xtr))
        for s in range(0, len(idx), batch):
            sl = idx[s:s + batch]
            x, t = xtr[sl], onehot[sl]
            h_pre = x @ w1 + b1
            h = np.maximum(h_pre, 0.0)
            logit = h @ w2 + b2
            logit -= logit.max(axis=1, keepdims=True)
            p = np.exp(logit)
            p /= p.sum(axis=1, keepdims=True)
            d_logit = (p - t) / len(sl)
            g_w2 = h.T @ d_logit
            g_b2 = d_logit.sum(axis=0)
            d_h = (d_logit @ w2.T) * (h_pre > 0)
            g_w1 = x.T @ d_h
            g_b1 = d_h.sum(axis=0)
            for i, (param, grad) in enumerate(
                    zip((w1, b1, w2, b2), (g_w1, g_b1, g_w2, g_b2))):
                v[i] = 0.9 * v[i] - lr * grad
                param += v[i]
    return w1, b1, w2, b2


def float_forward(x, w1, b1, w2, b2):
    h = np.maximum(x @ w1 + b1, 0.0)
    return h @ w2 + b2, h


# ---------------------------------------------------------------------------
# Quantization to the accelerator's exact integer pipeline
# ---------------------------------------------------------------------------
def quantize_layer(w, b, s_in, s_out, zp_in, cfg=CFG):
    """Symmetric per-channel weights, bias in the accumulator domain.

    Returns (q_w, q_bias, mult, shift). q_bias already contains the input
    zero-point correction, so the hardware never needs to know about zp_in.
    """
    n_out = w.shape[1]
    s_w = np.abs(w).max(axis=0) / 127.0
    s_w[s_w == 0] = 1e-12
    q_w = np.clip(np.rint(w / s_w), -127, 127).astype(np.int32)

    acc_scale = s_in * s_w                       # float value of one acc LSB
    q_bias = np.rint(b / acc_scale).astype(np.int64)
    q_bias -= zp_in * q_w.sum(axis=0).astype(np.int64)

    mult, shift = [], []
    for c in range(n_out):
        m, sh = g.quantize_multiplier(float(acc_scale[c] / s_out), cfg)
        mult.append(m)
        shift.append(sh)
    return q_w.astype(np.int8), q_bias.astype(np.int64), \
        np.array(mult, dtype=np.int64), np.array(shift, dtype=np.int64)


def int_layer(q_x, q_w, q_bias, mult, shift, zp_out, relu, cfg=CFG):
    """Integer reference for one fully connected layer, exactly as the RTL."""
    n_s, n_out = q_x.shape[0], q_w.shape[1]
    acc = q_x.astype(np.int64) @ q_w.astype(np.int64) + q_bias[None, :]
    out = np.zeros((n_s, n_out), dtype=np.int8)
    for s in range(n_s):
        for c in range(n_out):
            q, _ = g.requantize(int(acc[s, c]), int(mult[c]), int(shift[c]),
                                zp_out, cfg)
            sel = g.ACT_RELU if relu else g.ACT_IDENTITY
            out[s, c] = g.activation(q, sel, 0, zp_out, -128, 127)
    return out, acc


def main() -> int:
    x, y = load_dataset()
    (xtr_u, ytr), (xte_u, yte) = split(x, y)
    # Train on [0, 1] so the input scale is a clean 1/127 after quantization.
    xtr = xtr_u.astype(np.float64) / 16.0
    xte = xte_u.astype(np.float64) / 16.0

    w1, b1, w2, b2 = train(xtr, ytr)
    logits_f, hidden_f = float_forward(xte, w1, b1, w2, b2)
    acc_f32 = float((logits_f.argmax(axis=1) == yte).mean())

    # Activation scales measured on the training set, as a calibration pass does.
    logits_tr, hidden_tr = float_forward(xtr, w1, b1, w2, b2)
    s_x = float(xtr.max() / 127.0)
    zp_x = 0
    s_h = float(hidden_tr.max() / 255.0)          # ReLU output, zp at -128
    zp_h = -128
    s_o = float(np.abs(logits_tr).max() / 127.0)  # logits, symmetric
    zp_o = 0

    q_w1, q_b1, m1, sh1 = quantize_layer(w1, b1, s_x, s_h, zp_x)
    q_w2, q_b2, m2, sh2 = quantize_layer(w2, b2, s_h, s_o, zp_h)

    q_xte = np.clip(np.rint(xte / s_x), -128, 127).astype(np.int8)
    q_h, acc1 = int_layer(q_xte, q_w1, q_b1, m1, sh1, zp_h, relu=True)
    q_o, acc2 = int_layer(q_h, q_w2, q_b2, m2, sh2, zp_o, relu=False)
    pred_i8 = q_o.argmax(axis=1)
    acc_i8 = float((pred_i8 == yte).mean())

    per_class = {}
    for cls in range(N_OUT):
        sel = yte == cls
        per_class[str(cls)] = {
            "n": int(sel.sum()),
            "float32": float((logits_f[sel].argmax(axis=1) == cls).mean()),
            "int8": float((pred_i8[sel] == cls).mean()),
        }

    conf_f = np.zeros((N_OUT, N_OUT), dtype=int)
    conf_i = np.zeros((N_OUT, N_OUT), dtype=int)
    for t, pf, pi in zip(yte, logits_f.argmax(axis=1), pred_i8):
        conf_f[t, pf] += 1
        conf_i[t, pi] += 1

    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA / "demo_model.npz",
        q_w1=q_w1, q_b1=q_b1, m1=m1, sh1=sh1, zp_h=np.int64(zp_h),
        q_w2=q_w2, q_b2=q_b2, m2=m2, sh2=sh2, zp_o=np.int64(zp_o),
        q_xte=q_xte, yte=yte, q_h_ref=q_h, q_o_ref=q_o,
        acc1_ref=acc1, acc2_ref=acc2,
        s_x=np.float64(s_x), s_h=np.float64(s_h), s_o=np.float64(s_o),
        w1=w1, b1=b1, w2=w2, b2=b2,
        hidden_f=hidden_f, logits_f=logits_f,
    )

    results = {
        "network": f"{N_IN}-{N_HID}-{N_OUT} MLP, ReLU hidden layer",
        "dataset": "UCI hand-written digits (scikit-learn load_digits), "
                   "2x2 mean pooled to 4x4",
        "train_samples": int(len(xtr)),
        "test_samples": int(len(xte)),
        "accuracy_float32": acc_f32,
        "accuracy_int8": acc_i8,
        "accuracy_delta": acc_i8 - acc_f32,
        "quantization": {
            "input_scale": s_x, "input_zero_point": zp_x,
            "hidden_scale": s_h, "hidden_zero_point": zp_h,
            "output_scale": s_o, "output_zero_point": zp_o,
            "layer1_mult": m1.tolist(), "layer1_shift": sh1.tolist(),
            "layer2_mult": m2.tolist(), "layer2_shift": sh2.tolist(),
            "m_w": CFG.m_w,
        },
        "per_class": per_class,
        "confusion_float32": conf_f.tolist(),
        "confusion_int8": conf_i.tolist(),
        "acc1_range": [int(acc1.min()), int(acc1.max())],
        "acc2_range": [int(acc2.min()), int(acc2.max())],
    }
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "demo_results.json").write_text(json.dumps(results, indent=2))

    print(f"train {len(xtr)} / test {len(xte)} samples")
    print(f"float32 accuracy {acc_f32:.4f}")
    print(f"INT8    accuracy {acc_i8:.4f}  (delta {acc_i8 - acc_f32:+.4f})")
    print(f"layer 1 accumulator range {acc1.min()} .. {acc1.max()}")
    print(f"layer 2 accumulator range {acc2.min()} .. {acc2.max()}")
    print(f"layer 1 multipliers {m1.tolist()} shifts {sh1.tolist()}")
    print(f"wrote {DATA / 'demo_model.npz'} and {DOCS / 'demo_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
