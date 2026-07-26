#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
# SPDX-License-Identifier: Apache-2.0
"""Draw the quantization figures from the demo model and the RTL trace.

Produces:
  docs/img/demo_confusion.png    float32 and INT8 confusion matrices side by
                                 side, plus the decision margins that explain
                                 why the two agree
  docs/img/demo_per_class.png    per-class accuracy, float32 against INT8
  docs/img/demo_histograms.png   activation distributions before and after
                                 quantization, for the input, the hidden layer
                                 and the logits
  docs/img/requant_error.png     RTL requantizer output against the exact real
                                 valued product, from docs/data/requant_sweep.json

Everything except the last comes from scripts/train_demo.py; the last comes from
`make trace`, which reads the values out of the running RTL.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
IMG = REPO / "docs" / "img"
DATA = REPO / "test" / "data"
TRACE = REPO / "docs" / "data"

C_F32 = "#7aa6c2"
C_I8 = "#2f6f9f"
C_ERR = "#c26b3f"
C_GRID = "#d8dee3"
MUTED_ANN = "#5c6672"
C_TEXT = "#22282d"


def style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=10.5, color=C_TEXT, pad=8, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=C_TEXT)
    ax.set_ylabel(ylabel, fontsize=9, color=C_TEXT)
    ax.grid(axis="y", color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT, labelsize=8.5, length=0)


def decision_margin(z):
    """Per (sample, rival class) pair: the float32 gap and what quantization
    does to it.

    For the float32 winner `t` and any rival `j`, the INT8 network changes the
    prediction exactly when the dequantized output of `j` overtakes `t`, that
    is when `pert[j] - pert[t]` exceeds the float32 gap `logit[t] - logit[j]`.
    Returns the gap, that adverse swing, and the per-class shift.
    """
    lg = z["logits_f"]
    deq = (z["q_o_ref"].astype(float) - float(z["zp_o"])) * float(z["s_o"])
    pert = deq - lg
    top1 = lg.argmax(1)
    gap, swing = [], []
    for i in range(lg.shape[0]):
        t = top1[i]
        for j in range(lg.shape[1]):
            if j != t:
                gap.append(lg[i, t] - lg[i, j])
                swing.append(pert[i, j] - pert[i, t])
    return np.array(gap), np.array(swing), pert


def plot_margin_panel(ax, z, diff, lsb: float) -> None:
    """Third confusion panel: why quantization changes no prediction.

    The obvious panel here is the confusion difference as a heat map, but the
    difference is the all-zero matrix, and an all-zero heat map is a uniform
    grey grid with no labels: correct, and indistinguishable from a broken
    figure. The mechanism is the interesting part. Quantization does move the
    output layer; the decision survives because the float32 gap to every rival
    class is wider than the movement, and this panel shows the two against
    each other.
    """
    gap, swing, pert = decision_margin(z)
    tie = swing >= gap
    flip = swing > gap

    xs = np.logspace(np.log10(gap.min()), np.log10(gap.max()), 200)
    ax.fill_between(xs, xs, max(swing.max(), gap.max()) * 1.2,
                    color=C_ERR, alpha=0.09, linewidth=0)
    ax.plot(xs, xs, color=C_ERR, linewidth=1.1)
    ax.axhline(0, color=C_GRID, linewidth=0.8)
    ax.scatter(gap[~tie], swing[~tie], s=5, color=C_I8, alpha=0.45,
               linewidth=0)
    ax.scatter(gap[tie], swing[tie], s=30, facecolor="none", edgecolor=C_ERR,
               linewidth=1.2, zorder=3, label="exact INT8 tie")
    ax.set_xscale("log")
    # Headroom above the cloud for the annotation block, so it never sits on
    # top of a data point.
    ax.set_ylim(swing.min() - 0.25, swing.max() + 1.6)
    style(ax, "Why no prediction changes",
          "float32 gap, winner minus rival (log)", "INT8 swing of that pair")
    # Top right: the point cloud stays below y = 1.05 and the shaded flip
    # region hugs the left edge, so this is the only corner that is free.
    ax.annotate(
        f"{gap.size} (sample, rival) pairs\n"
        f"{flip.sum()} cross the line, so no prediction flips\n"
        f"{tie.sum()} land on it: an INT8 tie, broken by lower class index\n"
        f"largest output shift {np.abs(pert).max():.2f} = "
        f"{np.abs(pert).max() / lsb:.2f} LSB\n"
        f"INT8 minus float32 confusion: 0 in all {diff.size} cells",
        (0.97, 0.97), xycoords="axes fraction", va="top", ha="right",
        fontsize=8, color=MUTED_ANN)
    ax.annotate("rival wins", (0.045, 0.55), xycoords="axes fraction",
                fontsize=8, color=C_ERR)
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def plot_confusion(res, z) -> None:
    cf = np.array(res["confusion_float32"])
    ci = np.array(res["confusion_int8"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6),
                             gridspec_kw={"width_ratios": [1, 1, 1]})
    for ax, mat, name, acc in ((axes[0], cf, "float32", res["accuracy_float32"]),
                               (axes[1], ci, "INT8", res["accuracy_int8"])):
        im = ax.imshow(mat, cmap="Blues")
        ax.set_title(f"{name}: {acc:.3f} accuracy", fontsize=10.5,
                     color=C_TEXT, loc="left", pad=8)
        ax.set_xlabel("predicted", fontsize=9, color=C_TEXT)
        ax.set_ylabel("true", fontsize=9, color=C_TEXT)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.tick_params(colors=C_TEXT, labelsize=8, length=0)
        for i in range(10):
            for j in range(10):
                if mat[i, j]:
                    ax.text(j, i, mat[i, j], ha="center", va="center",
                            fontsize=7,
                            color="white" if mat[i, j] > mat.max() * 0.6
                            else C_TEXT)
        fig.colorbar(im, ax=ax, fraction=0.046)

    plot_margin_panel(axes[2], z, ci - cf, res["quantization"]["output_scale"])
    fig.suptitle(f"Confusion matrices, {res['test_samples']} held-out images, "
                 f"{res['network']}", fontsize=11.5, color=C_TEXT, x=0.012,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(IMG / "demo_confusion.png", dpi=150)
    plt.close(fig)


def plot_per_class(res) -> None:
    classes = [str(c) for c in range(10)]
    f32 = [res["per_class"][c]["float32"] for c in classes]
    i8 = [res["per_class"][c]["int8"] for c in classes]
    n = [res["per_class"][c]["n"] for c in classes]
    x = np.arange(10)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.bar(x - 0.19, f32, width=0.38, color=C_F32, label="float32")
    ax.bar(x + 0.19, i8, width=0.38, color=C_I8, label="INT8")
    for i, (a, b) in enumerate(zip(f32, i8)):
        if abs(a - b) > 1e-9:
            ax.annotate(f"{b - a:+.3f}", (i, max(a, b)), ha="center",
                        va="bottom", fontsize=7.5, color=C_ERR,
                        xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\nn={m}" for c, m in zip(classes, n)])
    ax.set_ylim(0, 1.12)
    style(ax, "Per-class recall, float32 against INT8 "
              "(orange = change from quantization)", "digit class", "recall")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(IMG / "demo_per_class.png", dpi=150)
    plt.close(fig)


def plot_histograms(z, res) -> None:
    q = res["quantization"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.4))
    sets = [
        ("input", z["q_xte"].astype(float) * q["input_scale"],
         z["q_xte"].astype(float), q["input_scale"], q["input_zero_point"]),
        ("hidden after ReLU", z["hidden_f"], z["q_h_ref"].astype(float),
         q["hidden_scale"], q["hidden_zero_point"]),
        ("output logits", z["logits_f"], z["q_o_ref"].astype(float),
         q["output_scale"], q["output_zero_point"]),
    ]
    for j, (name, real, quant, scale, zp) in enumerate(sets):
        axes[0][j].hist(np.asarray(real).ravel(), bins=60, color=C_F32)
        style(axes[0][j], f"{name}: float32 values", "value",
              "count" if j == 0 else "")
        axes[1][j].hist(np.asarray(quant).ravel(), bins=60, color=C_I8)
        style(axes[1][j], f"{name}: INT8", "INT8 code",
              "count" if j == 0 else "")
        axes[1][j].annotate(f"scale {scale:.3g}, zero point {zp}",
                            (0.02, 0.93), xycoords="axes fraction",
                            fontsize=8.5, color=MUTED_ANN)
        axes[1][j].set_xlim(-132, 132)
    fig.suptitle("Activation distributions before and after quantization",
                 fontsize=11.5, color=C_TEXT, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(IMG / "demo_histograms.png", dpi=150)
    plt.close(fig)


def plot_requant_error() -> None:
    path = TRACE / "requant_sweep.json"
    if not path.is_file():
        print(f"skipping requantization error plot: {path} missing "
              f"(run `make trace` in test/)")
        return
    d = json.loads(path.read_text())
    acc = np.array([p["acc"] for p in d["points"]], dtype=float)
    rtl = np.array([p["rtl"] for p in d["points"]], dtype=float)
    exact = np.array([p["exact"] for p in d["points"]], dtype=float)
    # Round to nearest with ties away from zero, which is what the hardware
    # does; numpy's round would break ties to even and disagree at exact halves.
    ideal = np.clip(np.sign(exact) * np.floor(np.abs(exact) + 0.5), -128, 127)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    ax.plot(acc, exact, color=C_ERR, linewidth=1.2,
            label=f"exact acc x {d['scale']:.6g}")
    ax.step(acc, rtl, color=C_I8, linewidth=1.2, where="mid",
            label="RTL INT8 output")
    ax.axhline(127, color=C_GRID, linewidth=0.9)
    ax.axhline(-128, color=C_GRID, linewidth=0.9)
    style(ax, f"Requantization at M={d['m']}, shift={d['shift']} "
              f"(scale {d['scale']:.6g}), measured on the RTL", "",
          "INT8 output")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    err = rtl - exact
    ax2.plot(acc, err, color=C_I8, linewidth=1.0)
    inside = np.abs(ideal) < 127
    ax2.axhline(0.5, color=C_GRID, linewidth=0.9, linestyle="--")
    ax2.axhline(-0.5, color=C_GRID, linewidth=0.9, linestyle="--")
    style(ax2, f"RTL minus exact, at most "
               f"{np.abs(err[inside]).max():.3f} LSB inside the unsaturated "
               f"range (dashed = half an LSB)", "accumulator value",
          "error (LSB)")
    fig.tight_layout()
    fig.savefig(IMG / "requant_error.png", dpi=150)
    plt.close(fig)
    bad = np.flatnonzero(rtl[inside] != ideal[inside])
    assert bad.size == 0, (
        "RTL requantization is not the correctly rounded result at "
        f"acc={acc[inside][bad][:5].tolist()}: rtl {rtl[inside][bad][:5].tolist()} "
        f"expected {ideal[inside][bad][:5].tolist()}")
    print(f"requantization error plot: {len(acc)} RTL points, "
          f"max error inside range {np.abs(err[inside]).max():.4f} LSB")


def main() -> int:
    IMG.mkdir(parents=True, exist_ok=True)
    res = json.loads((REPO / "docs" / "demo_results.json").read_text())
    z = np.load(DATA / "demo_model.npz")
    plot_confusion(res, z)
    plot_per_class(res)
    plot_histograms(z, res)
    plot_requant_error()
    print(f"wrote demo figures to {IMG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
