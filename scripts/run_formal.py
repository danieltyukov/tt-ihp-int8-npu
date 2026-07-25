# SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
# SPDX-License-Identifier: Apache-2.0
"""Prove every arithmetic variant equal to its behavioral reference.

The PPA comparison in README.md compares five adder architectures and three
multiplier architectures and attributes the differences to area and logic
depth. That only holds if the variants compute the same function. The
randomized suite in test/test_arith.py samples that; this proves it.

Each job is a combinational miter checked with SymbiYosys in bmc mode at depth
1: the design under test against `a + b + cin` or `a * b`. A pass is a proof
over the entire input space, so the multiplier results cover all 65536 signed
8x8 pairs and the 42-bit adder covers all 2**85 inputs.

    .venv/bin/python scripts/run_formal.py            # everything
    .venv/bin/python scripts/run_formal.py --jobs 3   # 3 solvers at a time
    .venv/bin/python scripts/run_formal.py --only adder

Writes docs/formal/summary.json and docs/formal/summary.md.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FORMAL = REPO / "formal"
OUT = REPO / "docs" / "formal"
WORK = REPO / "build" / "formal"

ADDERS = {0: "ripple-carry", 1: "Brent-Kung", 2: "Kogge-Stone",
          3: "Sklansky", 4: "Han-Carlson"}
MULTS = {0: "Baugh-Wooley array", 1: "Baugh-Wooley + Wallace",
         2: "Booth radix-4 + Wallace"}

# The widths npu_core actually instantiates: the PE accumulator adder, the
# requantizer paths and the widest Booth-step adder.
ADDER_WIDTHS = [19, 25, 26, 42]

MULT_SOURCES = ["npu_adder.sv", "npu_csa_reduce.sv", "npu_pp_baugh_wooley.sv",
                "npu_pp_booth4.sv", "npu_mult.sv"]

SBY_TEMPLATE = """\
[options]
mode bmc
depth 1

[engines]
smtbmc z3

[script]
{reads}
chparam {params} {top}
prep -top {top}

[files]
{files}
"""


def job_name(kind: str, **params: int) -> str:
    tail = "_".join(f"{k.lower()}{v}" for k, v in params.items())
    return f"{kind}_{tail}"


def build_jobs(only: str | None) -> list[dict]:
    jobs: list[dict] = []
    if only in (None, "adder"):
        for width in ADDER_WIDTHS:
            for arch in sorted(ADDERS):
                jobs.append({
                    "kind": "adder",
                    "name": job_name("adder", WIDTH=width, ARCH=arch),
                    "top": "miter_adder",
                    "sources": ["npu_adder.sv"],
                    "miter": "miter_adder.sv",
                    "params": {"WIDTH": width, "ARCH": arch},
                    "label": f"{ADDERS[arch]}, {width}-bit",
                    "space": f"2**{2 * width + 1}",
                })
    if only in (None, "mult"):
        for mul in sorted(MULTS):
            for add in sorted(ADDERS):
                jobs.append({
                    "kind": "mult",
                    "name": job_name("mult", MUL=mul, ADD=add),
                    "top": "miter_mult",
                    "sources": MULT_SOURCES,
                    "miter": "miter_mult.sv",
                    "params": {"A_W": 8, "B_W": 8, "MUL_ARCH": mul,
                               "ADD_ARCH": add},
                    "label": f"{MULTS[mul]}, {ADDERS[add]} final adder",
                    "space": "2**16",
                })
    return jobs


def write_sby(job: dict) -> Path:
    reads = "\n".join(f"read -formal {s}" for s in job["sources"])
    reads += f"\nread -formal {job['miter']}"
    params = " ".join(f"-set {k} {v}" for k, v in job["params"].items())
    files = "\n".join(f"{REPO / 'src' / s}" for s in job["sources"])
    files += f"\n{FORMAL / job['miter']}"
    path = WORK / f"{job['name']}.sby"
    path.write_text(SBY_TEMPLATE.format(reads=reads, params=params,
                                        top=job["top"], files=files))
    return path


def run(job: dict, timeout: int) -> dict:
    sby = write_sby(job)
    shutil.rmtree(sby.with_suffix(""), ignore_errors=True)
    start = time.monotonic()
    try:
        proc = subprocess.run(
            ["sby", "-f", sby.name],
            cwd=WORK, capture_output=True, text=True, timeout=timeout)
        out = proc.stdout + proc.stderr
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "")
        rc = -1
    elapsed = time.monotonic() - start

    status = "unknown"
    if re.search(r"DONE \(PASS", out):
        status = "pass"
    elif re.search(r"DONE \(FAIL", out):
        status = "fail"
    elif rc == -1:
        status = "timeout"
    result = dict(job)
    result.pop("sources", None)
    result.pop("miter", None)
    result.update(status=status, seconds=round(elapsed, 1), returncode=rc)
    print(f"  {status:8s} {elapsed:6.1f}s  {job['name']:24s} {job['label']}",
          flush=True)
    if status != "pass":
        (WORK / f"{job['name']}.log").write_text(out)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None,
                    help="solvers to run at once (default: cores minus 2, "
                         "capped at 4 so a shared machine stays usable)")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per-proof timeout in seconds")
    ap.add_argument("--only", choices=["adder", "mult"])
    args = ap.parse_args()

    if shutil.which("sby") is None:
        print("sby (SymbiYosys) is not on PATH", file=sys.stderr)
        return 1

    workers = args.jobs or min(4, max(1, (os.cpu_count() or 4) - 2))
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args.only)
    print(f"{len(jobs)} proofs, {workers} at a time")
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda j: run(j, args.timeout), jobs))
    elapsed = time.monotonic() - start

    passed = sum(r["status"] == "pass" for r in results)
    if args.only:
        # A partial run must not leave a summary that looks like a full one.
        print(f"\n{passed}/{len(results)} passed in {elapsed:.0f}s "
              f"(--only {args.only}, reports not written)")
        return 0 if passed == len(results) else 1

    summary = {
        "engine": "SymbiYosys bmc, smtbmc z3, depth 1",
        "proofs": len(results),
        "passed": passed,
        "wall_seconds": round(elapsed, 1),
        "results": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Formal equivalence of the arithmetic variants",
        "",
        "Generated by `scripts/run_formal.py`. Each row is a combinational",
        "miter proved with SymbiYosys (`mode bmc`, `depth 1`) against the",
        "z3 SMT solver: the variant is checked against the behavioral",
        "reference (`a + b + cin` for adders, `a * b` for multipliers) over",
        "the whole input space, not over sampled vectors.",
        "",
        f"{passed} of {len(results)} proofs pass "
        f"({elapsed:.0f}s wall, {workers} solvers at a time).",
        "",
        "| variant | inputs proved | result | solver time |",
        "| --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(f"| {r['label']} | {r['space']} | {r['status']} | "
                     f"{r['seconds']:.1f}s |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")

    print(f"\n{passed}/{len(results)} passed in {elapsed:.0f}s")
    print(f"wrote {OUT / 'summary.json'} and {OUT / 'summary.md'}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
