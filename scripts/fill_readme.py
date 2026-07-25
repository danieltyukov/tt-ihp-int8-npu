#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
# SPDX-License-Identifier: Apache-2.0
"""Fill the measured tables in README.md from the artefacts that produced them.

The README carries marker comments; this script replaces the block after each
marker with a table generated from `docs/synth/ppa.json`,
`docs/demo_results.json` and the cocotb `results*.xml` files. Running it twice is
a no-op, and running it after `make ppa` or `make test` is how the numbers in the
README stay honest.

  markers: PPA_ADDERS, PPA_MULTS, PPA_SCALING, PNR_RESULTS, FORMAL_RESULTS,
           DEMO_RESULTS, TEST_RESULTS
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"
PPA = REPO / "docs" / "synth" / "ppa.json"
DEMO = REPO / "docs" / "demo_results.json"
PNR = REPO / "docs" / "pnr" / "metrics.json"
PLACEMENT = REPO / "docs" / "pnr" / "placement.json"
FORMAL = REPO / "docs" / "formal" / "summary.json"
TESTS = REPO / "test"

ADDERS = ["ripple-carry", "Brent-Kung", "Kogge-Stone", "Sklansky", "Han-Carlson"]
MULTS = ["Baugh-Wooley array", "Baugh-Wooley + Wallace", "Booth radix-4 + Wallace"]

# Tiny Tapeout ihp-sg13g2 die sizes, keyed by (width, height) in um, from the
# tt_block_<tile>_pgvdd.def templates. Used to name a harvested run by its tile.
TILE_BY_DIE = {
    (202.08, 154.98): "1x1", (202.08, 313.74): "1x2",
    (419.52, 313.74): "2x2", (636.96, 313.74): "3x2",
    (854.40, 313.74): "4x2", (1289.28, 313.74): "6x2",
    (1724.16, 313.74): "8x2", (636.96, 710.64): "3x4",
    (854.40, 710.64): "4x4", (1071.84, 710.64): "5x4",
    (1289.28, 710.64): "6x4", (1724.16, 710.64): "8x4",
}


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


def smallest_tile(area: float, tiles: dict, density: float):
    for tile, die in sorted(tiles.items(), key=lambda kv: kv[1]):
        if area <= die * density:
            return tile, area / die
    return None, None


def route_inflation(d) -> float | None:
    """Post-route cell area divided by synthesis cell area, measured once.

    Placement and routing insert timing-repair and hold buffers, so a tile has
    to hold noticeably more cell area than Yosys reports. The shipped
    configuration is the one geometry that has been hardened, so its ratio is
    the only measured factor available; applying it to the other rows is an
    extrapolation and is labelled as one.
    """
    if not PNR.is_file():
        return None
    m = json.loads(PNR.read_text())
    return m["design__instance__area__stdcell"] / d["shipped"]["area_um2"]


def scaling_block(d) -> str:
    tiles = d["tiles"]
    dens = d["target_density"]
    infl = route_inflation(d)
    items = sorted(d["geometries"].values(), key=lambda r: r["area_um2"])
    rows = []
    for r in items:
        best, frac = smallest_tile(r["area_um2"], tiles, dens)
        ship = (r["rows"], r["cols"], r["s_max"]) == (4, 2, 6)
        row = [
            f"**{r['rows']}x{r['cols']}**" if ship else f"{r['rows']}x{r['cols']}",
            r["s_max"], r["rows"] * r["cols"], r["cell_count"],
            f"{r['area_um2']:.0f}", r["flop_count"], r["logic_depth"],
            f"{best} at {frac:.1%}" if best else "over 8x4",
        ]
        if infl is not None:
            routed, r_frac = smallest_tile(r["area_um2"] * infl, tiles, dens)
            row.append(f"{routed} at {r_frac:.1%}" if routed else "over 8x4")
        row.append("shipped" if ship else "")
        rows.append(row)

    header = ["array", "S_MAX", "MACs/cycle", "cells", "synth area (um2)",
              "registers", "depth", "smallest tile, synth area"]
    lead = "Every geometry below was synthesized and measured, not estimated:"
    if infl is not None:
        header.append(f"smallest tile, x{infl:.2f} route")
        lead += (f"\n\nThe last two columns apply the 60% criterion twice: once"
                 f" to the Yosys cell area, and once to that area scaled by"
                 f" {infl:.2f}, which is the synthesis-to-post-route cell area"
                 f" ratio measured on the shipped configuration. Only the"
                 f" shipped row has been hardened, so the scaled column is an"
                 f" extrapolation from that one data point.")
    header.append("")
    return lead + "\n\n" + table(header, rows)


def pnr_block() -> str:
    m = json.loads(PNR.read_text())
    p = json.loads(PLACEMENT.read_text())
    prov = p["provenance"]
    die = m["design__die__area"]
    stdcell = m["design__instance__area__stdcell"]
    tile = TILE_BY_DIE.get((p["die_width_um"], p["die_height_um"]), "custom")
    rows = [
        ["die", f"{p['die_width_um']} x {p['die_height_um']} um, "
                f"{die} um2 ({tile} tile)"],
        ["standard cells", f"{stdcell} um2 in "
                           f"{m['design__instance__count__stdcell']} instances"],
        ["core utilization", f"**{m['design__instance__utilization']:.2%}**"],
        ["cell area vs the die", f"{stdcell / die:.2%}"],
        ["decap and fill", f"{m['design__instance__area__class:fill_cell']:.0f} "
                           f"um2 in {p['instances_fill']} instances"],
        ["registers", f"{m['design__instance__count__class:sequential_cell']}"],
        ["buffers inserted for timing repair",
         f"{m['design__instance__count__class:timing_repair_buffer']}"],
        ["buffers inserted for hold",
         f"{m['design__instance__count__hold_buffer']}"],
        ["clock buffers and inverters",
         f"{m['design__instance__count__class:clock_buffer']} + "
         f"{m['design__instance__count__class:clock_inverter']}"],
        ["logic placed between",
         f"x = {p['logic_x_min_um']} um and x = {p['logic_x_max_um']} um, "
         f"{p['logic_x_span_fraction']:.1%} of the die width"],
        ["routed wirelength", f"{m['route__wirelength']} um"],
        ["setup slack, slow corner (1.08 V, 125 C)",
         f"**+{m['timing__setup__ws__corner:nom_slow_1p08V_125C']:.2f} ns** "
         f"at a 25 ns period"],
        ["hold slack, fast corner (1.32 V, -40 C)",
         f"+{m['timing__hold__ws__corner:nom_fast_1p32V_m40C']:.3f} ns"],
        ["worst clock skew, setup",
         f"{m['clock__skew__worst_setup__corner:nom_slow_1p08V_125C']:.3f} ns"],
        ["Magic DRC errors", f"**{m['magic__drc_error__count']}**"],
        ["Netgen LVS errors", f"**{m['design__lvs_error__count']}**"],
        ["detailed-route DRC errors", f"**{m['route__drc_errors']}**"],
        ["antenna violations", f"{m['route__antenna_violation__count']}"],
        ["total power estimate", f"{1000 * m['power__total']:.1f} mW"],
    ]
    return (
        f"Signoff metrics from the `gds` workflow, run "
        f"[{prov['github_run_id']}]"
        f"(https://github.com/danieltyukov/tt-ihp-int8-npu/actions/runs/"
        f"{prov['github_run_id']}), hardened with {prov['flow']} against "
        f"`{prov['pdk']}` at PDK commit `{prov['pdk_version'][:12]}`. Copied "
        f"verbatim into [docs/pnr/metrics.json](docs/pnr/metrics.json) by "
        f"`scripts/harvest_pnr.py`.\n\n"
        + table(["", ""], rows))


def tile_runs_block() -> str:
    """Every tile size this RTL has actually been hardened at, side by side."""
    runs = [(PNR.parent, True)]
    runs += [(d, False) for d in PNR.parent.glob("alt-*")
             if (d / "metrics.json").is_file()]

    cols = []
    for d, shipped in runs:
        m = json.loads((d / "metrics.json").read_text())
        p = json.loads((d / "placement.json").read_text())
        tile = TILE_BY_DIE.get((p["die_width_um"], p["die_height_um"]),
                               f"{p['die_width_um']}x{p['die_height_um']} um")
        cols.append((tile + (" (shipped)" if shipped else ""), m, p))
    cols.sort(key=lambda c: c[1]["design__die__area"])

    def row(label, fn):
        return [label] + [fn(m, p) for _, m, p in cols]

    rows = [
        row("die", lambda m, p: f"{p['die_width_um']} x {p['die_height_um']} um"),
        row("die area", lambda m, p: f"{m['design__die__area']} um2"),
        row("standard cells",
            lambda m, p: f"{m['design__instance__area__stdcell']} um2"),
        row("cell instances",
            lambda m, p: f"{m['design__instance__count__stdcell']}"),
        row("core utilization",
            lambda m, p: f"{m['design__instance__utilization']:.2%}"),
        row("logic reaches",
            lambda m, p: f"x = {p['logic_x_max_um']} um "
                         f"({p['logic_x_span_fraction']:.0%} of the width)"),
        row("decap and fill instances", lambda m, p: f"{p['instances_fill']}"),
        row("routed wirelength", lambda m, p: f"{m['route__wirelength']} um"),
        row("routing iterations to 0 DRC",
            lambda m, p: str(1 + max(int(k.split(":")[1]) for k in m
                                     if k.startswith("route__drc_errors__iter")))),
        row("setup slack, slow corner",
            lambda m, p: f"+{m['timing__setup__ws__corner:nom_slow_1p08V_125C']:.2f} ns"),
        row("hold slack, fast corner",
            lambda m, p: f"+{m['timing__hold__ws__corner:nom_fast_1p32V_m40C']:.3f} ns"),
        row("total power",
            lambda m, p: f"{1000 * m['power__total']:.1f} mW"),
        row("Magic DRC / Netgen LVS / route DRC / antenna",
            lambda m, p: f"{m['magic__drc_error__count']} / "
                         f"{m['design__lvs_error__count']} / "
                         f"{m['route__drc_errors']} / "
                         f"{m['route__antenna_violation__count']}"),
    ]
    ids = ", ".join(
        f"[{t}]"
        f"(https://github.com/danieltyukov/tt-ihp-int8-npu/actions/runs/"
        f"{p['provenance']['github_run_id']})" for t, _m, p in cols)
    return (f"The same RTL, hardened at each tile size and taken all the way to "
            f"signoff. Runs: {ids}.\n\n"
            + table([""] + [t for t, _m, _p in cols], rows))


def formal_block() -> str:
    d = json.loads(FORMAL.read_text())
    return (
        f"`scripts/run_formal.py` proves each arithmetic variant equal to its "
        f"behavioral reference: `a + b + cin` for adders, `a * b` for "
        f"multipliers. Engine: {d['engine']}. Both miters are combinational "
        f"and hold no state, so depth 1 reaches every input: a pass is a "
        f"correctness proof against the reference expression over the whole "
        f"input space, not an agreement check between two implementations, and "
        f"it is what makes the area and depth differences in the PPA tables "
        f"the only differences between the instances proved.\n\n"
        f"**{d['passed']} of {d['proofs']} proofs pass.** The `formal` workflow "
        f"reruns them on every push and fails if "
        f"[docs/formal/summary.md](docs/formal/summary.md) no longer matches, "
        f"so this is a checked result rather than a committed one.\n\n"
        + table(["what is proved", "variants", "inputs per proof", "result"], [
            ["`npu_adder` equals `a + b + cin`",
             "5 architectures x 19, 25, 26 and 42 bits",
             "2**39 to 2**85",
             f"{sum(r['status'] == 'pass' for r in d['results'] if r['kind'] == 'adder')}"
             f"/{sum(r['kind'] == 'adder' for r in d['results'])} pass"],
            ["`npu_mult` equals `a * b`",
             "3 partial-product styles x 5 final adders",
             "all 65536 signed 8x8 pairs",
             f"{sum(r['status'] == 'pass' for r in d['results'] if r['kind'] == 'mult')}"
             f"/{sum(r['kind'] == 'mult' for r in d['results'])} pass"],
        ])
        + f"\n\nWhat is not proved matters as much as what is. The miters "
        f"instantiate `npu_adder` and `npu_mult` and nothing else, so the "
        f"proofs say nothing about the sequential design: `npu_pe`, "
        f"`npu_array`, `npu_requant`, `npu_activation`, `npu_host_if` and "
        f"`npu_core` have no proof, and reset behaviour, the accumulator bank, "
        f"the requantization pipeline and the host protocol are covered by the "
        f"cocotb suite against `test/golden.py` instead. Each proof also fixes "
        f"its parameters, so what holds is {d['proofs']} instances rather than "
        f"the generators at arbitrary parameters: the adders at 19, 25, 26 and "
        f"42 bits, which are the widths `npu_core` instantiates, and the "
        f"multipliers at `A_W = B_W = 8`. Nothing here is proved about the "
        f"synthesized netlist either. The proofs run on the RTL; the "
        f"gate-level netlist is checked by `gl_test` and by LVS.")


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
    if PNR.is_file() and PLACEMENT.is_file():
        text = mark("PNR_RESULTS", pnr_block(), text)
        text = mark("TILE_RUNS", tile_runs_block(), text)
    if FORMAL.is_file():
        text = mark("FORMAL_RESULTS", formal_block(), text)
    if DEMO.is_file():
        text = mark("DEMO_RESULTS", demo_block(json.loads(DEMO.read_text())),
                    text)
    text = mark("TEST_RESULTS", test_block(), text)
    README.write_text(text)
    print(f"updated {README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
