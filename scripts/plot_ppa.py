#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
# SPDX-License-Identifier: Apache-2.0
"""Draw the PPA charts from docs/synth/ppa.json (written by scripts/run_ppa.py).

Produces:
  docs/img/ppa_adders.png     area, cells and depth per adder architecture
  docs/img/ppa_mults.png      the same for the signed 8x8 multipliers
  docs/img/area_scaling.png   measured top-level area against array geometry,
                              with the Tiny Tapeout tile budgets drawn in
  docs/img/requant_width.png  area cost of the requantization multiplier width
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PPA = REPO / "docs" / "synth" / "ppa.json"
IMG = REPO / "docs" / "img"

# One hue per metric, used consistently across every figure.
C_AREA = "#2f6f9f"
C_CELL = "#7aa6c2"
C_DEPTH = "#c26b3f"
C_WIN = "#1f7a4d"
C_GRID = "#d8dee3"
C_TEXT = "#22282d"

ADDERS = ["ripple-carry", "Brent-Kung", "Kogge-Stone", "Sklansky",
          "Han-Carlson"]
MULTS = ["Baugh-Wooley array", "Baugh-Wooley + Wallace",
         "Booth radix-4 + Wallace"]


def style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10.5, color=C_TEXT, pad=8, loc="left")
    ax.set_ylabel(ylabel, fontsize=9, color=C_TEXT)
    ax.grid(axis="y", color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT, labelsize=8.5, length=0)


def bars(ax, labels, values, color, fmt="{:.0f}", winner="min"):
    best = min(values) if winner == "min" else max(values)
    cols = [C_WIN if v == best else color for v in values]
    b = ax.bar(labels, values, color=cols, width=0.66)
    for rect, v in zip(b, values):
        ax.annotate(fmt.format(v), (rect.get_x() + rect.get_width() / 2,
                                    rect.get_height()),
                    ha="center", va="bottom", fontsize=8, color=C_TEXT,
                    xytext=(0, 2), textcoords="offset points")
    ax.set_ylim(0, max(values) * 1.18)
    return b


def short(label: str) -> str:
    return label.replace("Baugh-Wooley", "BW").replace(" + ", "\n+ ")


def plot_adders(data) -> None:
    widths = [19, 25, 26, 42]
    fig, axes = plt.subplots(3, len(widths), figsize=(13, 8.2))
    for j, w in enumerate(widths):
        group = [data["adders"][f"fast/{w}/{a}"] for a in range(5)]
        names = [n.replace("-", "-\n") for n in ADDERS]
        bars(axes[0][j], names, [r["area_um2"] for r in group], C_AREA)
        style(axes[0][j], f"{w}-bit adder area", "um2" if j == 0 else "")
        bars(axes[1][j], names, [r["cell_count"] for r in group], C_CELL)
        style(axes[1][j], f"{w}-bit cell count", "cells" if j == 0 else "")
        bars(axes[2][j], names, [r["logic_depth"] for r in group], C_DEPTH)
        style(axes[2][j], f"{w}-bit logic depth", "mapped cells" if j == 0 else "")
        for ax in axes[:, j]:
            ax.tick_params(axis="x", labelrotation=0, labelsize=7)
    fig.suptitle("Adder architectures on IHP sg13g2, structure-preserving "
                 "mapping (green = best in column)", fontsize=11.5,
                 color=C_TEXT, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(IMG / "ppa_adders.png", dpi=150)
    plt.close(fig)


def plot_mults(data) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.9))
    group = [data["mults"][f"fast/mul/{k}"] for k in range(3)]
    names = [short(n) for n in MULTS]
    bars(axes[0], names, [r["area_um2"] for r in group], C_AREA)
    style(axes[0], "signed 8x8 multiplier area", "um2")
    bars(axes[1], names, [r["cell_count"] for r in group], C_CELL)
    style(axes[1], "cell count", "cells")
    bars(axes[2], names, [r["logic_depth"] for r in group], C_DEPTH)
    style(axes[2], "logic depth", "mapped cells")

    cpa = [data["mults"][f"fast/cpa/{k}"] for k in range(5)]
    bars(axes[3], [n.replace("-", "-\n") for n in ADDERS],
         [r["area_um2"] for r in cpa], C_AREA)
    style(axes[3], "Wallace tree by final adder", "um2")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7.5)
    fig.suptitle("Multiplier architectures on IHP sg13g2, structure-preserving "
                 "mapping (green = best)", fontsize=11.5, color=C_TEXT,
                 x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(IMG / "ppa_mults.png", dpi=150)
    plt.close(fig)


def plot_scaling(data) -> None:
    geoms = data["geometries"]
    if not geoms:
        return
    items = sorted(geoms.values(), key=lambda r: r["area_um2"])
    labels = [f"{r['rows']}x{r['cols']}\nS={r['s_max']}" for r in items]
    areas = [r["area_um2"] for r in items]
    macs = [r["rows"] * r["cols"] for r in items]
    ship = data["shipped"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2),
                                  gridspec_kw={"width_ratios": [2.1, 1]})
    colours = [C_WIN if abs(a - ship["area_um2"]) < 1.0 else C_AREA
               for a in areas]
    ax.bar(labels, areas, color=colours, width=0.68)
    for tile, die in sorted(data["tiles"].items(), key=lambda kv: kv[1]):
        budget = die * data["target_density"]
        if budget > max(areas) * 1.25:
            continue
        ax.axhline(budget, color=C_DEPTH, linewidth=1.0, linestyle="--")
        ax.annotate(f"{tile} tile at {data['target_density']:.0%} density",
                    (len(labels) - 0.42, budget), fontsize=7.6, color=C_DEPTH,
                    va="bottom", ha="right")
    style(ax, "Measured top-level cell area by array geometry "
              "(green = shipped configuration)", "um2")
    ax.tick_params(axis="x", labelsize=7.5)

    ax2.scatter(macs, areas, color=C_AREA, s=34, zorder=3)
    ax2.scatter([ship_macs(ship, data)], [ship["area_um2"]], color=C_WIN, s=70,
                zorder=4, label="shipped")
    for m, a, r in zip(macs, areas, items):
        ax2.annotate(f"S={r['s_max']}", (m, a), fontsize=6.8, color=C_TEXT,
                     xytext=(4, -2), textcoords="offset points")
    style(ax2, "Area against MACs per cycle", "um2")
    ax2.set_xlabel("PEs (MACs per cycle)", fontsize=9, color=C_TEXT)
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG / "area_scaling.png", dpi=150)
    plt.close(fig)


def ship_macs(ship, data) -> int:
    for r in data["geometries"].values():
        if abs(r["area_um2"] - ship["area_um2"]) < 1.0:
            return r["rows"] * r["cols"]
    return 8


def plot_requant_width(data) -> None:
    widths = sorted(data["requant_widths"].values(), key=lambda r: r["m_w"])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labels = [str(r["m_w"]) for r in widths]
    bars(ax, labels, [r["area_um2"] for r in widths], C_AREA)
    style(ax, "Requantizer area against multiplier width M_W", "um2")
    ax.set_xlabel("M_W (bits of fixed-point multiplier)", fontsize=9,
                  color=C_TEXT)
    ax2 = ax.twinx()
    ax2.plot(labels, [2.0 ** -r["m_w"] for r in widths], color=C_DEPTH,
             marker="o", linewidth=1.4)
    ax2.set_yscale("log")
    ax2.set_ylabel("worst-case relative scale error", fontsize=9, color=C_DEPTH)
    ax2.tick_params(colors=C_DEPTH, labelsize=8.5, length=0)
    for side in ("top",):
        ax2.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG / "requant_width.png", dpi=150)
    plt.close(fig)


def main() -> int:
    data = json.loads(PPA.read_text())
    IMG.mkdir(parents=True, exist_ok=True)
    plot_adders(data)
    plot_mults(data)
    plot_scaling(data)
    plot_requant_width(data)
    print(f"wrote PPA figures to {IMG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
