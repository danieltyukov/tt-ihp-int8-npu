# SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
# SPDX-License-Identifier: Apache-2.0
"""Capture real simulation data for the figures under docs/img/.

Nothing here is decorative: the dataflow animation, the pipeline timing diagram,
the protocol timing diagram and the requantization error plot are all drawn from
the JSON these tests write, so the pictures cannot drift away from the hardware.

Run with `make trace` in test/, or `make images` at the top level.
"""

from __future__ import annotations

import json
from pathlib import Path

import cocotb
from cocotb.triggers import FallingEdge

import golden as g
from npu_driver import Npu

CFG = g.Cfg(rows=4, cols=2, s_max=6, acc_w=24, m_w=16, sh_w=5)
M_HALF = 1 << (CFG.m_w - 1)
OUT = Path(__file__).resolve().parent.parent / "docs" / "data"


def write(name: str, payload: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=1))
    return path


@cocotb.test()
async def test_capture_dataflow(dut):
    """Per-cycle activation and partial-sum state of every PE during one run."""
    npu = Npu(dut, CFG)
    await npu.start_clock()
    await npu.reset()
    model = g.Model(CFG)

    rows, cols, s_count = CFG.rows, CFG.cols, CFG.s_max
    # Distinct small weights and activations so each PE's contribution is
    # identifiable in the animation.
    weights = [[r + 1, -(r + 1)] for r in range(rows)]
    acts = [[10 * (s + 1) + r for r in range(rows)] for s in range(s_count)]

    await npu.load_all(model, weights, acts, [0] * cols, [M_HALF] * cols,
                       [0] * cols, s_count=s_count)

    array = dut.user_project.u_core.u_array
    pes = [[array.g_row[r].g_col[c].u_pe for c in range(cols)]
           for r in range(rows)]
    core = dut.user_project.u_core

    await npu.cmd(g.OP_RUN, 0b10)
    frames = []
    seen_busy = False
    for cycle in range(s_count + rows + cols + 8):
        await FallingEdge(dut.clk)
        busy = npu.busy
        if busy:
            seen_busy = True
        elif seen_busy:
            break
        state = int(core.state.value)
        frames.append({
            "cycle": cycle,
            "state": state,
            "row_in": [int(core.act_row.value) >> (8 * r) & 0xFF
                       for r in range(rows)],
            "a_reg": [[to_signed(int(pes[r][c].a_reg.value), 8)
                       for c in range(cols)] for r in range(rows)],
            "psum": [[to_signed(int(pes[r][c].psum_reg.value), 19)
                      for c in range(cols)] for r in range(rows)],
            "w_reg": [[to_signed(int(pes[r][c].w_reg.value), 8)
                       for c in range(cols)] for r in range(rows)],
            "col_valid": [int(core.g_bank[c].col_valid.value)
                          if hasattr(core, "g_bank") else 0
                          for c in range(cols)],
        })
        if state == 2:      # ST_REQ: the array has drained
            break

    await npu.wait_done()
    results = await npu.read_results(s_count * cols)
    model.run(weights, acts, [0] * cols, [M_HALF] * cols, [0] * cols,
              s_count=s_count, accumulate=False, requant=True)
    assert results == model.result[:s_count * cols], "trace run must be correct"

    path = write("dataflow.json", {
        "rows": rows, "cols": cols, "s_count": s_count,
        "weights": weights, "acts": acts,
        "results": results,
        "frames": frames,
    })
    dut._log.info(f"captured {len(frames)} cycles of array state to {path}")


def to_signed(v: int, width: int) -> int:
    return v - (1 << width) if v >> (width - 1) else v


@cocotb.test()
async def test_capture_requant_sweep(dut):
    """RTL requantizer output against the exact real-valued product.

    A single weight of 1 in row 0 puts the activation straight into the
    accumulator, and the bias walks the accumulator across the range that maps
    to the whole INT8 output range, so each run yields s_count*COLS points.
    """
    npu = Npu(dut, CFG)
    await npu.start_clock()
    await npu.reset()
    model = g.Model(CFG)

    mult = [M_HALF, M_HALF]
    shift = [7, 7]
    scale = g.effective_scale(mult[0], shift[0], CFG)   # 1/256
    weights = [[1, 1], [0, 0], [0, 0], [0, 0]]

    points = []
    # The INT8 output range covers acc in [-128/scale, 127/scale].
    lo, hi = int(-140 / scale), int(140 / scale)
    step = (hi - lo) // 40
    for base in range(lo, hi, step * CFG.s_max):
        acts = []
        biases = []
        for s in range(CFG.s_max):
            acts.append([0, 0, 0, 0])
        for c in range(CFG.cols):
            biases.append(base + c * step // 2)
        # Sweep the activation as the fine step within each run.
        for s in range(CFG.s_max):
            acts[s] = [s * (step // CFG.s_max), 0, 0, 0]
        model.reset()
        await npu.load_all(model, weights, acts, biases, mult, shift,
                           s_count=CFG.s_max, zp=0)
        await npu.run()
        model.run(weights, acts, biases, mult, shift, s_count=CFG.s_max,
                  accumulate=False, requant=True)
        hw = await npu.read_results(CFG.s_max * CFG.cols)
        assert hw == model.result[:CFG.s_max * CFG.cols], \
            f"requant sweep mismatch at base {base}: {hw}"
        for s in range(CFG.s_max):
            for c in range(CFG.cols):
                acc = model.acc[c][s]
                points.append({"acc": acc, "rtl": hw[s * CFG.cols + c],
                               "exact": acc * scale})

    points.sort(key=lambda p: p["acc"])
    path = write("requant_sweep.json", {
        "m": mult[0], "shift": shift[0], "scale": scale, "m_w": CFG.m_w,
        "points": points,
    })
    dut._log.info(f"captured {len(points)} requantization points to {path}")


@cocotb.test()
async def test_capture_protocol(dut):
    """Pin-level trace of a weight load followed by an inference trigger."""
    npu = Npu(dut, CFG)
    await npu.start_clock()
    await npu.reset()
    model = g.Model(CFG)

    weights = [[3, -3], [2, -2], [1, -1], [4, -4]]
    acts = [[5, 6, 7, 8]]
    await npu.frame(g.OP_CFG, 0, [0])
    await npu.frame(g.OP_LD_ACT, 0, model.act_bytes(acts))
    await npu.frame(g.OP_LD_BIAS, 0, model.bias_bytes([0, 0]))
    await npu.frame(g.OP_LD_QUANT, 0, model.quant_bytes([M_HALF] * CFG.cols,
                                                        [4, 4]))
    await npu.frame(g.OP_LD_POST, 0, model.post_bytes(0, -128, 127, 0, 0))

    samples = []

    async def snap(label: str):
        samples.append({
            "cycle": len(samples),
            "label": label,
            "ui_in": int(dut.ui_in.value),
            "wr": int(dut.uio_in.value) & 1,
            "is_cmd": (int(dut.uio_in.value) >> 1) & 1,
            "rd": (int(dut.uio_in.value) >> 2) & 1,
            "uo_out": int(dut.uo_out.value),
            "busy": int(npu.busy),
            "done": int(npu.done),
        })

    # Drive the frame by hand so every cycle can be sampled mid-cycle.
    byte_stream = [(0x2 << 4, 1, "LD_W")] + \
        [(b, 0, f"W{i}") for i, b in enumerate(model.weight_bytes(weights))]
    await FallingEdge(dut.clk)
    await snap("idle")
    for value, is_cmd, label in byte_stream:
        dut.ui_in.value = value
        dut.uio_in.value = 1 | (2 if is_cmd else 0)
        await FallingEdge(dut.clk)
        await snap(label)
    dut.ui_in.value = (0x7 << 4) | 0b10
    dut.uio_in.value = 3
    await FallingEdge(dut.clk)
    await snap("RUN")
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    for i in range(16):
        await FallingEdge(dut.clk)
        await snap("compute" if npu.busy else "idle")

    await npu.wait_done()
    path = write("protocol.json", {"samples": samples})
    dut._log.info(f"captured {len(samples)} protocol cycles to {path}")
