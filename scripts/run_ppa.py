#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
# SPDX-License-Identifier: Apache-2.0
"""Run the full PPA comparison and write the reports under docs/synth/.

Three sweeps:

1. Adders: five architectures at the four widths the accelerator instantiates
   (19-bit partial-sum chain, 25-bit accumulator add, 26-bit requantizer,
   42-bit for reference), at both mapping efforts.
2. Multipliers: three architectures for signed 8x8, and the Wallace tree paired
   with each of the five final adders.
3. Top level: the shipped configuration plus an array-geometry sweep, which is
   what the tile-size decision is based on.

Everything is measured with Yosys against the real IHP sg13g2 liberty. Nothing
here is estimated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SYNTH = REPO / "scripts" / "synth.py"
OUT = REPO / "docs" / "synth"

ADDERS = [(0, "ripple-carry"), (1, "Brent-Kung"), (2, "Kogge-Stone"),
          (3, "Sklansky"), (4, "Han-Carlson")]
MULTS = [(0, "Baugh-Wooley array"), (1, "Baugh-Wooley + Wallace"),
         (2, "Booth radix-4 + Wallace")]
ADDER_WIDTHS = [19, 25, 26, 42]

# Tile geometry for Tiny Tapeout on IHP sg13g2: a 1x1 tile is 167 x 108 um, and
# an "N x 2" tile is N tiles wide by 2 tall.
TILE_UM2 = {
    "1x1": 167.0 * 108.0,
    "1x2": 167.0 * 216.0,
    "2x2": 334.0 * 216.0,
    "3x2": 501.0 * 216.0,
    "4x2": 668.0 * 216.0,
    "6x2": 1002.0 * 216.0,
    "8x2": 1336.0 * 216.0,
}
TARGET_DENSITY = 0.60   # PL_TARGET_DENSITY_PCT in src/config.json

# Array geometries to measure for the scaling plot.
GEOMETRIES = [(2, 2, 2), (2, 2, 4), (2, 2, 8), (4, 2, 2), (4, 2, 4), (4, 2, 5),
              (4, 2, 6), (4, 2, 8), (6, 2, 4), (8, 2, 2), (8, 2, 4), (2, 4, 4),
              (4, 4, 2), (4, 4, 4)]

SHIPPED = dict(ROWS=4, COLS=2, S_MAX=6)


REUSE = False


def synth(top: str, name: str, effort: str, params: dict[str, int],
          netlist: bool = False) -> dict:
    cmd = [sys.executable, str(SYNTH), "--top", top, "--name", name,
           "--effort", effort, "--out", str(OUT), "--quiet"]
    if REUSE:
        cmd.append("--reuse")
    for k, v in params.items():
        cmd += ["--param", f"{k}={v}"]
    if netlist:
        cmd.append("--netlist")
    subprocess.run(cmd, check=True)
    return json.loads((OUT / f"{name}.{effort}.json").read_text())


def md_table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(str(r[i])) for r in [header] + rows)
              for i in range(len(header))]
    def fmt(cells):
        return "| " + " | ".join(str(c).ljust(widths[i])
                                 for i, c in enumerate(cells)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([fmt(header), sep] + [fmt(r) for r in rows])


def best(results: list[dict], key: str, lower_is_better: bool = True):
    vals = [r[key] for r in results]
    return (min(vals) if lower_is_better else max(vals))


def run_adders() -> dict:
    data = {}
    for effort in ("fast", "full"):
        for width in ADDER_WIDTHS:
            for arch, label in ADDERS:
                name = f"adder_w{width}_a{arch}"
                r = synth("npu_adder", name, effort,
                          {"WIDTH": width, "ARCH": arch})
                r["label"] = label
                r["width"] = width
                data[f"{effort}/{width}/{arch}"] = r
    return data


def run_mults() -> dict:
    data = {}
    for effort in ("fast", "full"):
        for arch, label in MULTS:
            name = f"mult8x8_m{arch}"
            r = synth("npu_mult", name, effort,
                      {"A_W": 8, "B_W": 8, "MUL_ARCH": arch, "ADD_ARCH": 4})
            r["label"] = label
            data[f"{effort}/mul/{arch}"] = r
        for arch, label in ADDERS:
            name = f"mult8x8_wallace_cpa{arch}"
            r = synth("npu_mult", name, effort,
                      {"A_W": 8, "B_W": 8, "MUL_ARCH": 1, "ADD_ARCH": arch})
            r["label"] = f"Wallace, {label} CPA"
            data[f"{effort}/cpa/{arch}"] = r
    return data


def run_blocks() -> dict:
    blocks = {
        "npu_pe": ("pe", {"PSUM_W": 19, "MUL_ARCH": 1, "ADD_ARCH": 4}),
        "npu_array": ("array", dict(ROWS=4, COLS=2, PSUM_W=19, MUL_ARCH=1,
                                    ADD_ARCH=4)),
        "npu_requant": ("requant", {"T_W": 24, "M_W": 16, "SH_W": 5,
                                    "ADD_ARCH": 4}),
        "npu_activation": ("activation", {}),
        "npu_host_if": ("host_if", {}),
        "npu_core": ("core", dict(ROWS=4, COLS=2, S_MAX=6, MUL_ARCH=1,
                                  ADD_ARCH=4)),
    }
    out = {}
    for top, (name, params) in blocks.items():
        out[name] = synth(top, f"block_{name}", "full", params)
    return out


def run_geometries() -> dict:
    out = {}
    for rows, cols, smax in GEOMETRIES:
        key = f"{rows}x{cols}s{smax}"
        out[key] = synth("tt_um_danieltyukov_int8_npu", f"scale_{key}", "full",
                         dict(ROWS=rows, COLS=cols, S_MAX=smax))
        out[key].update(rows=rows, cols=cols, s_max=smax)
    return out


def run_requant_widths() -> dict:
    """Cost of the requantization multiplier width, which sets scale precision."""
    out = {}
    for m_w in (8, 12, 16, 20, 24):
        out[str(m_w)] = synth("npu_requant", f"requant_m{m_w}", "full",
                              {"T_W": 24, "M_W": m_w, "SH_W": 5, "ADD_ARCH": 4})
        out[str(m_w)]["m_w"] = m_w
    return out


def smallest_tile(area: float, density: float = TARGET_DENSITY):
    for tile, die in sorted(TILE_UM2.items(), key=lambda kv: kv[1]):
        if area <= die * density:
            return tile, area / die
    return None, area / TILE_UM2["8x2"]


def write_reports(all_data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ppa.json").write_text(json.dumps(all_data, indent=2))

    lines = ["# PPA comparison", "",
             "Measured with Yosys 0.33 against "
             "`sg13g2_stdcell_typ_1p20V_25C.lib` (IHP open PDK). Area is the",
             "sum of standard-cell areas in square micrometres; depth is the",
             "longest topological path through mapped cells, which is the",
             "critical-path proxy used throughout this project.", "",
             "Two mapping efforts are reported:", "",
             "- **fast**: ABC maps the netlist as written, so the architecture",
             "  a generator describes is what gets measured.",
             "- **full**: ABC's default script with `dc2` and `&dch`",
             "  resynthesis, which is what the Tiny Tapeout hardening flow runs.",
             ""]

    # Adders
    for effort in ("fast", "full"):
        lines += [f"## Adders, {effort} mapping", ""]
        for width in ADDER_WIDTHS:
            rows = []
            group = [all_data["adders"][f"{effort}/{width}/{a}"]
                     for a, _ in ADDERS]
            a_min = best(group, "area_um2")
            d_min = best(group, "logic_depth")
            for (arch, label), r in zip(ADDERS, group):
                mark = []
                if r["area_um2"] == a_min:
                    mark.append("smallest")
                if r["logic_depth"] == d_min:
                    mark.append("shortest path")
                rows.append([label, r["cell_count"], f"{r['area_um2']:.1f}",
                             r["logic_depth"], ", ".join(mark) or ""])
            lines += [f"### {width}-bit", "",
                      md_table(rows, ["architecture", "cells", "area (um2)",
                                      "depth", "note"]), ""]

    # Multipliers
    for effort in ("fast", "full"):
        lines += [f"## Signed 8x8 multipliers, {effort} mapping", ""]
        group = [all_data["mults"][f"{effort}/mul/{a}"] for a, _ in MULTS]
        a_min = best(group, "area_um2")
        d_min = best(group, "logic_depth")
        rows = []
        for (arch, label), r in zip(MULTS, group):
            mark = []
            if r["area_um2"] == a_min:
                mark.append("smallest")
            if r["logic_depth"] == d_min:
                mark.append("shortest path")
            rows.append([label, r["cell_count"], f"{r['area_um2']:.1f}",
                         r["logic_depth"], ", ".join(mark) or ""])
        lines += [md_table(rows, ["architecture", "cells", "area (um2)",
                                  "depth", "note"]), "",
                  "Wallace tree with each final carry-propagate adder:", ""]
        rows = []
        for arch, label in ADDERS:
            r = all_data["mults"][f"{effort}/cpa/{arch}"]
            rows.append([label, r["cell_count"], f"{r['area_um2']:.1f}",
                         r["logic_depth"]])
        lines += [md_table(rows, ["final adder", "cells", "area (um2)",
                                  "depth"]), ""]

    # Blocks
    lines += ["## Block breakdown, shipped configuration", "",
              "ROWS=4, COLS=2, S_MAX=6, ACC_W=24, M_W=16, Wallace multipliers,",
              "Han-Carlson adders.", ""]
    rows = []
    for name, r in all_data["blocks"].items():
        rows.append([name, r["cell_count"], f"{r['area_um2']:.1f}",
                     r["flop_count"], r["logic_depth"]])
    lines += [md_table(rows, ["module", "cells", "area (um2)", "flops",
                              "depth"]), ""]

    # Requantizer width
    lines += ["## Requantization multiplier width", "",
              "M_W sets how precisely a float scale can be represented",
              "(relative error below 2^-M_W) and is the main precision knob.", ""]
    rows = []
    for m_w, r in sorted(all_data["requant_widths"].items(),
                         key=lambda kv: int(kv[0])):
        ndig = (int(m_w) + 3) // 2
        rows.append([m_w, r["cell_count"], f"{r['area_um2']:.1f}",
                     r["flop_count"], ndig])
    lines += [md_table(rows, ["M_W", "cells", "area (um2)", "flops",
                              "Booth steps"]), ""]

    # Geometry scaling
    lines += ["## Array geometry scaling", "",
              "Full top-level area for each geometry, with the smallest Tiny",
              f"Tapeout tile whose die area is at least the measured area",
              f"divided by {TARGET_DENSITY:.0%} (the `PL_TARGET_DENSITY_PCT`",
              "in `src/config.json`).", ""]
    rows = []
    for key, r in sorted(all_data["geometries"].items(),
                         key=lambda kv: kv[1]["area_um2"]):
        tile, dens = smallest_tile(r["area_um2"])
        macs = r["rows"] * r["cols"]
        rows.append([f"{r['rows']}x{r['cols']}", r["s_max"], macs,
                     r["cell_count"], f"{r['area_um2']:.0f}", r["flop_count"],
                     r["logic_depth"], tile or "over 8x2",
                     f"{TILE_UM2[tile] and r['area_um2'] / TILE_UM2[tile]:.1%}"
                     if tile else ""])
    lines += [md_table(rows, ["array", "S_MAX", "MACs/cycle", "cells",
                              "area (um2)", "flops", "depth", "smallest tile",
                              "density"]), ""]

    ship = all_data["shipped"]
    tile, dens = smallest_tile(ship["area_um2"])
    lines += ["## Shipped configuration", "",
              f"- geometry: ROWS={SHIPPED['ROWS']}, COLS={SHIPPED['COLS']}, "
              f"S_MAX={SHIPPED['S_MAX']}",
              f"- cells: {ship['cell_count']}",
              f"- registers: {ship['flop_count']}",
              f"- cell area: {ship['area_um2']:.1f} um2",
              f"- logic depth: {ship['logic_depth']} mapped cells",
              f"- tile: {tile} ({TILE_UM2[tile]:.0f} um2 die, "
              f"{ship['area_um2'] / TILE_UM2[tile]:.1%} cell density)",
              f"- inferred latches: "
              f"{ship['latches'] if ship['latches'] else 'none'}",
              f"- unmapped cells: "
              f"{ship['blackboxes'] if ship['blackboxes'] else 'none'}", ""]

    (OUT / "ppa.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'ppa.md'} and {OUT / 'ppa.json'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the geometry sweep")
    ap.add_argument("--reuse", action="store_true",
                    help="re-parse existing logs instead of re-synthesizing")
    args = ap.parse_args()
    global REUSE
    REUSE = args.reuse

    all_data = {}
    print("== adders ==")
    all_data["adders"] = run_adders()
    print("== multipliers ==")
    all_data["mults"] = run_mults()
    print("== blocks ==")
    all_data["blocks"] = run_blocks()
    print("== requantizer widths ==")
    all_data["requant_widths"] = run_requant_widths()
    print("== shipped top level ==")
    all_data["shipped"] = synth("tt_um_danieltyukov_int8_npu", "top_shipped",
                                "full", SHIPPED, netlist=True)
    all_data["shipped_fast"] = synth("tt_um_danieltyukov_int8_npu",
                                     "top_shipped", "fast", SHIPPED)
    all_data["tiles"] = TILE_UM2
    all_data["target_density"] = TARGET_DENSITY
    if not args.quick:
        print("== geometry sweep ==")
        all_data["geometries"] = run_geometries()
    else:
        all_data["geometries"] = {}
    write_reports(all_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
