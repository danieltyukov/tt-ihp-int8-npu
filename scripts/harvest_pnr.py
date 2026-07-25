# SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
# SPDX-License-Identifier: Apache-2.0
"""Harvest post-route results from a LibreLane run into docs/pnr/.

Synthesis area is not what a tile has to hold. Placement and routing add
timing-repair and hold buffers, so the number that decides the tile size is the
post-route standard-cell area, not the Yosys figure. This pulls both the signoff
metrics and the placement extent out of a finished run so the README can quote
them from a committed artefact.

The run directory is what the `gds` workflow uploads as its GDS_logs artifact:

    gh run download <id> -R danieltyukov/tt-ihp-int8-npu -n GDS_logs
    .venv/bin/python scripts/harvest_pnr.py runs/wokwi --run-id <id>

Writes docs/pnr/metrics.json (the flat LibreLane metric dict, verbatim) and
docs/pnr/placement.json (die box, cell extent and the x histogram of non-fill
instances, derived from final/def).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "pnr"

COMPONENT_RE = re.compile(
    r"^\s+- (\S+) (\S+) \+ .*?PLACED \( (-?\d+) (-?\d+) \)", re.M)
# decap and fill cells are placed to satisfy density rules, not by the design.
FILL_RE = re.compile(r"decap|fill", re.I)

BINS = 20


def read_def(path: Path) -> dict:
    text = path.read_text()
    units = int(re.search(r"^UNITS DISTANCE MICRONS (\d+)", text, re.M).group(1))
    dx1, dy1, dx2, dy2 = (
        int(v) / units for v in
        re.search(r"^DIEAREA \( (-?\d+) (-?\d+) \) \( (-?\d+) (-?\d+) \)",
                  text, re.M).groups())

    logic: list[tuple[float, float]] = []
    fill = 0
    for _name, cell, x, y in COMPONENT_RE.findall(text):
        if FILL_RE.search(cell):
            fill += 1
        else:
            logic.append((int(x) / units, int(y) / units))
    if not logic:
        raise SystemExit(f"no non-fill components found in {path}")

    xs = [p[0] for p in logic]
    ys = [p[1] for p in logic]
    width = dx2 - dx1
    hist = [0] * BINS
    for x in xs:
        hist[min(BINS - 1, int(BINS * (x - dx1) / width))] += 1

    return {
        "die_box_um": [dx1, dy1, dx2, dy2],
        "die_width_um": round(width, 2),
        "die_height_um": round(dy2 - dy1, 2),
        "instances_total": len(logic) + fill,
        "instances_fill": fill,
        "instances_logic": len(logic),
        "logic_x_min_um": round(min(xs), 2),
        "logic_x_max_um": round(max(xs), 2),
        "logic_y_min_um": round(min(ys), 2),
        "logic_y_max_um": round(max(ys), 2),
        "logic_width_um": round(max(xs) - min(xs), 2),
        "logic_x_span_fraction": round((max(xs) - min(xs)) / width, 4),
        "x_histogram_bins": BINS,
        "x_histogram": hist,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path,
                    help="LibreLane run directory, the one holding final/")
    ap.add_argument("--run-id", default=None,
                    help="GitHub Actions run id this came from")
    # Deliberately not a commit sha: the run id pins the exact tree GitHub
    # hardened and survives a history rewrite, a sha does not.
    ap.add_argument("--branch", default=None,
                    help="branch the run was triggered from")
    ap.add_argument("--out-dir", type=Path, default=OUT,
                    help="where to write, default docs/pnr. Use a subdirectory "
                         "for a run that is not the shipped configuration.")
    ap.add_argument("--label", default=None,
                    help="what this run is, recorded in the provenance block")
    args = ap.parse_args()
    out = args.out_dir

    final = args.run / "final"
    metrics = json.loads((final / "metrics.json").read_text())
    defs = list((final / "def").glob("*.def"))
    if len(defs) != 1:
        raise SystemExit(f"expected one DEF in {final / 'def'}, found {len(defs)}")

    placement = read_def(defs[0])
    pdk = json.loads((args.run / "pdk.json").read_text())
    placement["provenance"] = {
        "label": args.label,
        "github_run_id": args.run_id,
        "branch": args.branch,
        "flow": f"{pdk.get('FLOW_NAME')} {pdk.get('FLOW_VERSION')}",
        "pdk": pdk.get("PDK"),
        "pdk_version": pdk.get("PDK_VERSION"),
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out / "placement.json").write_text(json.dumps(placement, indent=2) + "\n")

    print(f"die           {placement['die_width_um']} x "
          f"{placement['die_height_um']} um "
          f"({metrics['design__die__area']} um2)")
    print(f"stdcell area  {metrics['design__instance__area__stdcell']} um2 "
          f"in {metrics['design__instance__count__stdcell']} cells")
    print(f"utilization   {metrics['design__instance__utilization']:.2%} "
          f"of the core")
    print(f"logic extent  x {placement['logic_x_min_um']} .. "
          f"{placement['logic_x_max_um']} um "
          f"({placement['logic_x_span_fraction']:.1%} of the die width)")
    print(f"wrote {out / 'metrics.json'} and {out / 'placement.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
