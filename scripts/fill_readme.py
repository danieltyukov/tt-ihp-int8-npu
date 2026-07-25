#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
# SPDX-License-Identifier: Apache-2.0
"""Fill the measured tables in README.md from the artefacts that produced them.

The README carries marker comments; this script replaces the block after each
marker with a table generated from `docs/synth/ppa.json`,
`docs/demo_results.json` and the cocotb `results*.xml` files. Running it twice is
a no-op, and running it after `make ppa` or `make test` is how the numbers in the
README stay honest.

  markers: PPA_ADDERS, PPA_MULTS, PPA_SCALING, DEMO_RESULTS, TEST_RESULTS
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"
PPA = REPO / "docs" / "synth" / "ppa.json"
DEMO = REPO / "docs" / "demo_results.json"
TESTS = REPO / "test"

ADDERS = ["ripple-carry", "Brent-Kung", "Kogge-Stone", "Sklansky", "Han-Carlson"]
MULTS = ["Baugh-Wooley array", "Baugh-Wooley + Wallace", "Booth radix-4 + Wallace"]


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def mark(name: str, body: str, text: str) -> str:
    pattern = re.compile(rf"(<!--{name}-->\n)(.*?)(?=\n<!--|\n!\[|\n#{{2,3}} |\Z)",
                         re.S)
    m = pattern.search(text)
    if not m:
        raise SystemExit(f"marker {name} not found in README.md")
    return text[:m.start()] + m.group(1) + body + "\n" + text[m.end():]


def adders_block(d) -> str:
    width = 25
    rows = []
    group = [d["adders"][f"fast/{width}/{a}"] for a in range(5)]
    a_min = min(r["area_um2"] for r in group)
    d_min = min(r["logic_depth"] for r in group)
    for name, r in zip(ADDERS, group):
        note = []
        if r["area_um2"] == a_min:
            note.append("smallest")
        if r["logic_depth"] == d_min:
            note.append("shortest path")
        rows.append([name, r["cell_count"], f"{r['area_um2']:.0f}",
                     r["logic_depth"], ", ".join(note) or ""])
    return ("25-bit adders, the width of the accumulator path:\n\n"
            + table(["architecture", "cells", "area (um2)", "logic depth",
                     ""], rows))


def mults_block(d) -> str:
    group = [d["mults"][f"fast/mul/{k}"] for k in range(3)]
    a_min = min(r["area_um2"] for r in group)
    d_min = min(r["logic_depth"] for r in group)
    rows = []
    for name, r in zip(MULTS, group):
        note = []
        if r["area_um2"] == a_min:
            note.append("smallest")
        if r["logic_depth"] == d_min:
            note.append("shortest path")
        rows.append([name, r["cell_count"], f"{r['area_um2']:.0f}",
                     r["logic_depth"], ", ".join(note) or ""])
    extra = [[ADDERS[k],
              d["mults"][f"fast/cpa/{k}"]["cell_count"],
              f"{d['mults'][f'fast/cpa/{k}']['area_um2']:.0f}",
              d["mults"][f"fast/cpa/{k}"]["logic_depth"]] for k in range(5)]
    return ("Signed 8x8 multipliers, the width every PE instantiates:\n\n"
            + table(["architecture", "cells", "area (um2)", "logic depth", ""],
                    rows)
            + "\n\nThe same Wallace tree with each final carry-propagate adder, "
              "which shows the two choices are independent:\n\n"
            + table(["final adder", "cells", "area (um2)", "logic depth"],
                    extra))


def scaling_block(d) -> str:
    tiles = d["tiles"]
    dens = d["target_density"]
    items = sorted(d["geometries"].values(), key=lambda r: r["area_um2"])
    rows = []
    for r in items:
        best = None
        for tile, die in sorted(tiles.items(), key=lambda kv: kv[1]):
            if r["area_um2"] <= die * dens:
                best = tile
                break
        ship = (r["rows"], r["cols"], r["s_max"]) == (4, 2, 6)
        rows.append([
            f"**{r['rows']}x{r['cols']}**" if ship else f"{r['rows']}x{r['cols']}",
            r["s_max"], r["rows"] * r["cols"], r["cell_count"],
            f"{r['area_um2']:.0f}", r["flop_count"], r["logic_depth"],
            best or "over 8x2",
            f"{r['area_um2'] / tiles[best]:.1%}" if best else "",
            "shipped" if ship else "",
        ])
    return ("Every geometry below was synthesized and measured, not estimated:\n\n"
            + table(["array", "S_MAX", "MACs/cycle", "cells", "area (um2)",
                     "registers", "depth", "smallest tile", "density", ""],
                    rows))


def demo_block(d) -> str:
    q = d["quantization"]
    lines = [
        table(["", ""], [
            ["network", d["network"]],
            ["dataset", d["dataset"]],
            ["train / test images", f"{d['train_samples']} / {d['test_samples']}"],
            ["float32 accuracy", f"**{d['accuracy_float32']:.4f}**"],
            ["INT8 accuracy", f"**{d['accuracy_int8']:.4f}**"],
            ["accuracy change from quantization", f"{d['accuracy_delta']:+.4f}"],
            ["layer 1 accumulator range", f"{d['acc1_range'][0]} .. {d['acc1_range'][1]}"],
            ["layer 2 accumulator range", f"{d['acc2_range'][0]} .. {d['acc2_range'][1]}"],
            ["input quantization", f"scale {q['input_scale']:.6g}, "
                                   f"zero point {q['input_zero_point']}"],
            ["hidden quantization", f"scale {q['hidden_scale']:.6g}, "
                                    f"zero point {q['hidden_zero_point']}"],
            ["output quantization", f"scale {q['output_scale']:.6g}, "
                                    f"zero point {q['output_zero_point']}"],
        ]),
    ]
    return "\n".join(lines)


def test_block() -> str:
    rows = []
    total_t = total_f = 0
    # cocotb writes a flat JUnit file; counting the tags avoids pulling an XML
    # parser in just to read two numbers out of our own output.
    for path in sorted(TESTS.glob("results*.xml")):
        body = path.read_text(errors="replace")
        n = len(re.findall(r"<testcase\b", body))
        f = len(re.findall(r"<failure\b", body))
        if n == 0:
            continue
        total_t += n
        total_f += f
        rows.append([f"`{path.name}`", n, n - f, f])
    if not rows:
        return "_Run `make test-all` to populate this table._"
    rows.append(["**total**", total_t, total_t - total_f, total_f])
    return (table(["results file", "tests", "passed", "failed"], rows)
            + f"\n\nProduced by `make test-all` on this machine with Icarus "
              f"Verilog 12.0 and cocotb 2.0.1.")


def main() -> int:
    text = README.read_text()
    if PPA.is_file():
        d = json.loads(PPA.read_text())
        text = mark("PPA_ADDERS", adders_block(d), text)
        text = mark("PPA_MULTS", mults_block(d), text)
        if d.get("geometries"):
            text = mark("PPA_SCALING", scaling_block(d), text)
    if DEMO.is_file():
        text = mark("DEMO_RESULTS", demo_block(json.loads(DEMO.read_text())),
                    text)
    text = mark("TEST_RESULTS", test_block(), text)
    README.write_text(text)
    print(f"updated {README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
