# SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
# SPDX-License-Identifier: Apache-2.0
"""End-to-end neural network demo on the real RTL.

A 16-12-10 MLP over 4x4 handwritten digit images, quantized to INT8 by
scripts/train_demo.py, is executed layer by layer on the accelerator: weight
tiles are streamed in, activations are batched S_MAX at a time, K > ROWS is
handled by accumulating passes, and the requantized INT8 activations are read
back and fed into the next layer.

Every INT8 value the hardware produces, for both layers, must equal the NumPy
INT8 reference exactly. The number of images pushed through the RTL is set by
NPU_DEMO_IMAGES (default 18, which is 3 full batches); the reference accuracy
over the whole 359-image test set comes from docs/demo_results.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cocotb
import numpy as np

import golden as g
from npu_driver import Npu

CFG = g.Cfg(rows=4, cols=2, s_max=6, acc_w=24, m_w=16, sh_w=5)

DATA = Path(__file__).resolve().parent / "data" / "demo_model.npz"
RESULTS = Path(__file__).resolve().parent.parent / "docs" / "demo_results.json"
N_IMAGES = int(os.environ.get("NPU_DEMO_IMAGES", "18"))


async def run_layer(npu: Npu, model: g.Model, q_x, q_w, q_bias, mult, shift,
                    zp_out: int, relu: bool):
    """One fully connected layer for a batch of samples.

    q_x is (S, K) INT8, q_w is (K, N) INT8. K must be a multiple of ROWS and N a
    multiple of COLS: the layer is tiled into K/ROWS accumulating passes and
    N/COLS channel groups.
    """
    s_count, k = q_x.shape
    n_out = q_w.shape[1]
    assert k % CFG.rows == 0, f"K={k} must be a multiple of ROWS={CFG.rows}"
    assert n_out % CFG.cols == 0, f"N={n_out} must be a multiple of COLS"
    passes = k // CFG.rows
    groups = n_out // CFG.cols

    out = np.zeros((s_count, n_out), dtype=np.int64)
    act_sel = g.ACT_RELU if relu else g.ACT_IDENTITY

    for grp in range(groups):
        lo = grp * CFG.cols
        cols = slice(lo, lo + CFG.cols)
        await npu.frame(g.OP_CFG, 0, [s_count - 1])
        await npu.frame(g.OP_LD_BIAS, 0, model.bias_bytes(q_bias[cols]))
        await npu.frame(g.OP_LD_QUANT, 0,
                        model.quant_bytes(mult[cols], shift[cols]))
        await npu.frame(g.OP_LD_POST, 0,
                        model.post_bytes(zp_out, -128, 127, act_sel, 0))
        for p in range(passes):
            rows = slice(p * CFG.rows, (p + 1) * CFG.rows)
            await npu.frame(g.OP_LD_W, 0, model.weight_bytes(q_w[rows, cols]))
            await npu.frame(g.OP_LD_ACT, 0, model.act_bytes(q_x[:, rows]))
            await npu.run(accumulate=(p > 0), requant=(p == passes - 1))
        res = await npu.read_results(s_count * CFG.cols)
        out[:, cols] = np.array(res, dtype=np.int64).reshape(s_count, CFG.cols)
    return out


@cocotb.test()
async def test_mlp_end_to_end(dut):
    """Run the quantized MLP on the RTL and demand bit-exact agreement."""
    assert DATA.is_file(), (
        f"{DATA} missing: run `python scripts/train_demo.py` first")
    z = np.load(DATA)
    ref = json.loads(RESULTS.read_text())

    q_xte = z["q_xte"].astype(np.int64)
    yte = z["yte"].astype(np.int64)
    q_h_ref = z["q_h_ref"].astype(np.int64)
    q_o_ref = z["q_o_ref"].astype(np.int64)

    n = min(N_IMAGES, len(q_xte))
    npu = Npu(dut, CFG)
    await npu.start_clock()
    await npu.reset()
    model = g.Model(CFG)

    checked = 0
    correct = 0
    preds: list[int] = []
    for base in range(0, n, CFG.s_max):
        batch = slice(base, min(base + CFG.s_max, n))
        x = q_xte[batch]
        hidden = await run_layer(npu, model, x, z["q_w1"].astype(np.int64),
                                 z["q_b1"].astype(np.int64),
                                 z["m1"].astype(np.int64),
                                 z["sh1"].astype(np.int64),
                                 int(z["zp_h"]), relu=True)
        assert np.array_equal(hidden, q_h_ref[batch]), (
            f"layer 1 mismatch on images {base}..{batch.stop - 1}\n"
            f"  rtl {hidden.tolist()}\n  ref {q_h_ref[batch].tolist()}")

        logits = await run_layer(npu, model, hidden.astype(np.int64),
                                 z["q_w2"].astype(np.int64),
                                 z["q_b2"].astype(np.int64),
                                 z["m2"].astype(np.int64),
                                 z["sh2"].astype(np.int64),
                                 int(z["zp_o"]), relu=False)
        assert np.array_equal(logits, q_o_ref[batch]), (
            f"layer 2 mismatch on images {base}..{batch.stop - 1}\n"
            f"  rtl {logits.tolist()}\n  ref {q_o_ref[batch].tolist()}")

        checked += hidden.size + logits.size
        pred = logits.argmax(axis=1)
        preds += pred.tolist()
        correct += int((pred == yte[batch]).sum())

    ref_pred = q_o_ref[:n].argmax(axis=1)
    assert preds == ref_pred.tolist(), \
        f"RTL predictions {preds} differ from the INT8 reference {ref_pred.tolist()}"

    dut._log.info(
        f"end-to-end demo: {n} images through the RTL, {checked} INT8 layer "
        f"outputs bit-exact against the NumPy INT8 reference")
    dut._log.info(
        f"RTL accuracy on those {n} images: {correct}/{n} = {correct / n:.3f}")
    dut._log.info(
        f"reference accuracy over all {ref['test_samples']} test images: "
        f"float32 {ref['accuracy_float32']:.4f}, INT8 "
        f"{ref['accuracy_int8']:.4f} (delta {ref['accuracy_delta']:+.4f})")
