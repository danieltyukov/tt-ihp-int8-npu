#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
# SPDX-License-Identifier: Apache-2.0
"""Synthesize one module with Yosys and extract area, cell count and logic depth.

The IHP sg13g2 standard-cell liberty file is used for both technology mapping
and area reporting, so the numbers are real cell areas in square micrometres
rather than abstract gate counts.

Two mapping efforts are supported because they answer different questions:

  fast  ABC maps the netlist as written (structural hashing plus technology
        mapping). Architectural differences between, say, a ripple-carry and a
        Kogge-Stone adder survive, so this is the mode used for the
        architecture comparison.
  full  ABC's default script, with `dc2` and `&dch` resynthesis. This is what
        the real Tiny Tapeout hardening flow runs. It partially erases the
        architectural differences, which is itself worth reporting.

Usage:
  scripts/synth.py --top npu_adder --param WIDTH=25 --param ARCH=2 \\
      --name adder_w25_kogge_stone --effort fast
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
DEFAULT_OUT = REPO / "docs" / "synth"

LIBERTY_CANDIDATES = [
    os.environ.get("SG13G2_LIB", ""),
    str(REPO / "pdk" / "sg13g2_stdcell_typ_1p20V_25C.lib"),
    os.path.expandvars("$PDK_ROOT/ihp-sg13g2/libs.ref/sg13g2_stdcell/lib/"
                       "sg13g2_stdcell_typ_1p20V_25C.lib"),
]



def find_liberty() -> Path:
    for cand in LIBERTY_CANDIDATES:
        if cand and Path(cand).is_file():
            return Path(cand)
    sys.exit(
        "No sg13g2 liberty file found. Set SG13G2_LIB, or place\n"
        "sg13g2_stdcell_typ_1p20V_25C.lib in ./pdk/ (see docs/ADAPTING.md).\n"
        "It ships with the IHP open PDK: "
        "https://github.com/IHP-GmbH/IHP-Open-PDK"
    )


def source_files() -> list[Path]:
    return sorted(SRC.glob("*.sv"))


def build_script(top: str, params: dict[str, int], liberty: Path,
                 effort: str, netlist: Path | None) -> str:
    # Reading the liberty as blackboxes first gives Yosys the cell interfaces,
    # without which `ltp` cannot walk the mapped netlist.
    lines = [f"read_liberty -lib {liberty}",
             f"read_verilog -sv {' '.join(str(p) for p in source_files())}"]
    if params:
        sets = " ".join(f"-set {k} {v}" for k, v in params.items())
        lines.append(f"chparam {sets} {top}")
    abc = f"abc {'-fast ' if effort == 'fast' else ''}-liberty {liberty}"
    lines += [
        f"hierarchy -top {top} -check",
        f"synth -top {top} -flatten",
        "check -assert",
        "design -save prepped",
        # Depth pass: map only the combinational logic, so the registers are
        # still generic $_DFF_ cells and `ltp -noff` can stop at them.
        abc,
        "setundef -zero",
        "opt_clean -purge",
        "ltp -noff",
        # Area pass, in the order a real flow uses: map the registers first,
        # then run ABC, so the multiplexers dfflibmap emits for enable flops
        # (sg13g2 has no enable flop) get mapped and counted too.
        "design -load prepped",
        f"dfflibmap -liberty {liberty}",
        abc,
        "setundef -zero",
        "opt_clean -purge",
        f"stat -liberty {liberty}",
    ]
    if netlist:
        lines.append(f"write_verilog -noattr {netlist}")
    return "; ".join(lines)


def parse_report(text: str) -> dict:
    """Extract the numbers from a Yosys log.

    Yosys prints `stat` more than once: the `synth` script ends with one over
    generic cells, and the explicit `stat -liberty` at the end of the flow
    reports the mapped netlist. Only the last one describes real cells, so every
    figure here is taken from the final block.
    """
    out: dict = {"cells": {}}

    # The final statistics block starts at the last "Number of cells:".
    starts = [m.start() for m in re.finditer(r"Number of cells:", text)]
    tail = text[starts[-1]:] if starts else text

    m = re.search(r"Number of cells:\s+(\d+)", tail)
    out["cell_count"] = int(m.group(1)) if m else None

    m = re.search(r"Chip area for module [^:]*:\s+([0-9.]+)", text)
    out["area_um2"] = float(m.group(1)) if m else None

    m = re.search(r"Longest topological path.*?length=(\d+)", text, re.S)
    out["logic_depth"] = int(m.group(1)) if m else None

    wires = re.findall(r"Number of wires:\s+(\d+)", text)
    out["wires"] = int(wires[-1]) if wires else None

    for line in tail.splitlines()[1:]:
        mm = re.match(r"\s+(\S+)\s+(\d+)\s*$", line)
        if mm:
            out["cells"][mm.group(1)] = int(mm.group(2))
        elif line.strip() == "":
            break
    # A sequential cell count separates flop area from combinational area.
    out["flop_count"] = sum(n for c, n in out["cells"].items()
                            if re.search(r"_s?dfr|_s?dfb|_dl[hl]", c))
    out["blackboxes"] = sorted(c for c in out["cells"] if c.startswith("$"))
    out["latches"] = sorted(c for c in out["cells"] if "_dl" in c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", required=True)
    ap.add_argument("--param", action="append", default=[],
                    help="NAME=VALUE, repeatable")
    ap.add_argument("--name", default=None, help="report basename")
    ap.add_argument("--effort", choices=["fast", "full"], default="fast")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--netlist", action="store_true",
                    help="also write the mapped gate-level netlist")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="re-parse an existing log instead of running Yosys")
    args = ap.parse_args()

    if not shutil.which("yosys"):
        sys.exit("yosys not found on PATH")

    params = {}
    for p in args.param:
        k, _, v = p.partition("=")
        params[k.strip()] = int(v, 0)

    liberty = find_liberty()
    name = args.name or args.top
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    log = outdir / f"{name}.{args.effort}.log"
    netlist = (outdir / f"{name}.{args.effort}.netlist.v") if args.netlist else None

    if args.reuse and log.is_file():
        out = log.read_text()
    else:
        script = build_script(args.top, params, liberty, args.effort, netlist)
        proc = subprocess.run(["yosys", "-p", script],
                              capture_output=True, text=True)
        log.write_text(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
            sys.exit(f"yosys failed for {name} (see {log})")
        out = proc.stdout

    res = parse_report(out)
    res.update({"name": name, "top": args.top, "params": params,
                "effort": args.effort, "liberty": liberty.name})
    (outdir / f"{name}.{args.effort}.json").write_text(json.dumps(res, indent=2))

    if not args.quiet:
        print(f"{name:34s} effort={args.effort:4s} "
              f"cells={res['cell_count']:6d} "
              f"area={res['area_um2']:10.2f} um2 "
              f"depth={res['logic_depth']:4d} "
              f"flops={res['flop_count']:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
