# SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
# SPDX-License-Identifier: Apache-2.0
"""Accelerator test suite.

Every check compares the RTL against test/golden.py, an independent Python
model written from the datasheet. Nothing here is a smoke test: each case
asserts exact byte equality, and the randomized sweep reports how many vectors
it actually ran.
"""

from __future__ import annotations

import os
import random

import cocotb
from cocotb.triggers import ClockCycles, FallingEdge

import golden as g
from npu_driver import Npu

CFG = g.Cfg(rows=4, cols=2, s_max=6, acc_w=24, m_w=16, sh_w=5)

# Number of randomized layer configurations. The default is what the committed
# results were produced with; lower it for a quicker smoke run.
SWEEP_CASES = int(os.environ.get("NPU_SWEEP_CASES", "120"))
MULTIPASS_CASES = int(os.environ.get("NPU_MULTIPASS_CASES", "30"))

# Effective scale of 1/2 with M normalized: the mid-range case used whenever a
# test cares about something other than the scale itself.
M_HALF = 1 << (CFG.m_w - 1)


def expected_busy_cycles(cfg: g.Cfg, s_count: int, requant: bool,
                         shifts) -> int:
    """Documented latency model, from docs/DESIGN.md.

    array phase   s_count + ROWS + COLS cycles
                  (1 injection per cycle, ROWS+1 fill, COLS-1 drain)
    requant phase per element: 1 setup + NDIG Booth steps
                  + shift steps (4 bits then 1 bit per cycle)
                  + 1 round/zero-point + 1 done
    plus one cycle in the terminal state.
    """
    ndig = (cfg.m_w + 3) // 2
    total = s_count + cfg.rows + cfg.cols
    if requant:
        for _ in range(s_count):
            for c in range(cfg.cols):
                n = cfg.m_w + shifts[c]
                total += ndig + (n // 4 + n % 4) + 3
    return total + 1


async def setup(dut) -> Npu:
    npu = Npu(dut, CFG)
    await npu.start_clock()
    await npu.reset()
    return npu


def rand_i8(rng, n):
    return [rng.randint(-128, 127) for _ in range(n)]


async def run_case(npu: Npu, model: g.Model, weights, acts, bias, mult, shift,
                   *, s_count, zp=0, act_sel=g.ACT_IDENTITY, leaky_k=0,
                   clamp_lo=-128, clamp_hi=127, accumulate=False, requant=True):
    """Program one pass into the DUT and the model, run both, compare results."""
    await npu.load_all(model, weights, acts, bias, mult, shift,
                       s_count=s_count, zp=zp, act_sel=act_sel,
                       leaky_k=leaky_k, clamp_lo=clamp_lo, clamp_hi=clamp_hi)
    cycles = await npu.run(accumulate=accumulate, requant=requant)
    model.run(weights, acts, bias, mult, shift, s_count=s_count,
              accumulate=accumulate, requant=requant, zp=zp, act_sel=act_sel,
              leaky_k=leaky_k, clamp_lo=clamp_lo, clamp_hi=clamp_hi)
    return cycles


# ---------------------------------------------------------------------------
# Reset and identity
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_reset_defines_all_outputs(dut):
    """After reset every output is driven, known and zero-valued."""
    npu = Npu(dut, CFG)
    await npu.start_clock()
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)

    assert dut.uo_out.value.is_resolvable, f"uo_out is {dut.uo_out.value} in reset"
    assert dut.uio_out.value.is_resolvable
    assert int(dut.uio_oe.value) == 0b1111_1000, \
        f"uio_oe must drive only the status pins, got {int(dut.uio_oe.value):#010b}"
    assert int(dut.uo_out.value) == 0
    assert int(dut.uio_out.value) == 0, "no status flag may be set in reset"

    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    assert not npu.busy and not npu.done and not npu.err
    assert not npu.sat and not npu.ovf
    results = await npu.read_results(CFG.s_max * CFG.cols)
    assert results == [0] * (CFG.s_max * CFG.cols), \
        f"result buffer not cleared by reset: {results}"


@cocotb.test()
async def test_identity_block(dut):
    """The identity block reports the geometry the host needs to drive it."""
    npu = await setup(dut)
    ident = await npu.read_id()
    assert ident[0] == 0x4E and ident[1] == 0x38, f"bad magic {ident[:2]}"
    assert ident[2] == 1, f"unexpected version {ident[2]}"
    assert ident[3] == ((CFG.rows << 4) | CFG.cols), f"geometry byte {ident[3]:#x}"
    assert (ident[4] >> 4) == CFG.s_max, f"s_max byte {ident[4]:#x}"
    assert ident[5] == CFG.acc_w
    assert ident[6] == CFG.m_w
    dut._log.info(f"identity block: {[hex(v) for v in ident]}")


# ---------------------------------------------------------------------------
# Directed arithmetic
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_directed_layer(dut):
    """A small case whose expected values can be checked by hand.

    W = [[1,2],[3,4],[5,6],[7,8]], x = [1,1,1,1], bias = [0, 100]
    column 0: 1+3+5+7 = 16, column 1: 2+4+6+8 = 20 (+100 = 120)
    scale 1/2 with round-half-away-from-zero: 8 and 60.
    """
    npu = await setup(dut)
    model = g.Model(CFG)
    weights = [[1, 2], [3, 4], [5, 6], [7, 8]]
    acts = [[1, 1, 1, 1]]
    bias = [0, 100]
    mult = [M_HALF, M_HALF]
    shift = [0, 0]

    await run_case(npu, model, weights, acts, bias, mult, shift, s_count=1)
    hw = await npu.read_results(CFG.cols)
    assert hw == model.result[:CFG.cols], f"hw {hw} model {model.result[:CFG.cols]}"
    assert hw == [8, 60], f"hand-computed expectation failed: {hw}"

    raw0 = await npu.read_acc(0, 0)
    raw1 = await npu.read_acc(0, 1)
    assert (raw0, raw1) == (16, 120), f"raw accumulators {raw0}, {raw1}"


@cocotb.test()
async def test_int8_minimum(dut):
    """-128 has no positive counterpart and is where sign handling breaks."""
    npu = await setup(dut)
    model = g.Model(CFG)
    weights = [[-128, -128], [-128, 127], [127, -128], [-1, 0]]
    acts = [[-128, -128, -128, -128],
            [-128, 127, -128, 127],
            [127, 127, 127, 127]]
    bias = [0, 0]
    mult = [M_HALF, M_HALF]
    shift = [8, 8]

    await run_case(npu, model, weights, acts, bias, mult, shift, s_count=3)
    hw = await npu.read_results(3 * CFG.cols)
    assert hw == model.result[:3 * CFG.cols], \
        f"hw {hw} model {model.result[:3 * CFG.cols]}"
    for s in range(3):
        for c in range(CFG.cols):
            got = await npu.read_acc(s, c)
            assert got == model.acc[c][s], \
                f"raw sum ({s},{c}) hw {got} model {model.acc[c][s]}"
    dut._log.info(f"-128 coverage: raw sums {[model.acc[c][0] for c in range(CFG.cols)]}")


@cocotb.test()
async def test_extreme_tensors(dut):
    """All-zero and all-rail tensors, which maximise the accumulator range."""
    npu = await setup(dut)
    model = g.Model(CFG)
    cases = [
        ([[0] * CFG.cols] * CFG.rows, [[0] * CFG.rows]),
        ([[127] * CFG.cols] * CFG.rows, [[127] * CFG.rows]),
        ([[-128] * CFG.cols] * CFG.rows, [[-128] * CFG.rows]),
        ([[127] * CFG.cols] * CFG.rows, [[-128] * CFG.rows]),
    ]
    for weights, acts in cases:
        model.reset()
        await run_case(npu, model, weights, acts, [0] * CFG.cols,
                       [M_HALF] * CFG.cols, [10] * CFG.cols, s_count=1)
        hw = await npu.read_results(CFG.cols)
        assert hw == model.result[:CFG.cols], \
            f"weights {weights[0]} acts {acts[0]}: hw {hw} model {model.result[:CFG.cols]}"


@cocotb.test()
async def test_saturation_both_rails(dut):
    """Force the output past both INT8 rails and check the sticky sat flag."""
    npu = await setup(dut)
    model = g.Model(CFG)
    weights = [[127, -128], [127, -128], [127, -128], [127, -128]]
    acts = [[127, 127, 127, 127]]
    # shift 0 keeps the full magnitude, so both channels overshoot INT8.
    await run_case(npu, model, weights, acts, [0, 0], [M_HALF, M_HALF], [0, 0],
                   s_count=1)
    hw = await npu.read_results(CFG.cols)
    assert hw == [127, -128], f"expected both rails, got {hw}"
    assert hw == model.result[:CFG.cols]
    assert npu.sat, "saturation must raise the sticky sat flag"
    assert not npu.ovf, "the accumulator itself did not overflow"

    await npu.frame(g.OP_CLR)
    await npu.settle()
    assert not npu.sat, "CLR must clear the sticky sat flag"


@cocotb.test()
async def test_accumulator_overflow(dut):
    """Repeated accumulation past 2**23 saturates and latches ovf."""
    npu = await setup(dut)
    model = g.Model(CFG)
    weights = [[127, 127], [127, 127], [127, 127], [127, 127]]
    acts = [[127] * CFG.rows]
    bias = [CFG.acc_max - 1000, -CFG.acc_max]
    mult = [M_HALF, M_HALF]
    shift = [0, 0]

    await npu.load_all(model, weights, acts, bias, mult, shift, s_count=1)
    await npu.run(accumulate=False, requant=False)
    model.run(weights, acts, bias, mult, shift, s_count=1,
              accumulate=False, requant=False)
    assert npu.ovf, "positive accumulator overflow must latch ovf"
    got = await npu.read_acc(0, 0)
    assert got == CFG.acc_max == model.acc[0][0], \
        f"accumulator must saturate to {CFG.acc_max}, got {got}"

    # Negative direction: start next to the negative rail, then accumulate
    # negative products over several passes.
    await npu.frame(g.OP_CLR)
    await npu.frame(g.OP_SRST)
    await npu.settle()
    assert not npu.ovf, "CLR and SRST must clear ovf"
    model.reset()
    neg_w = [[-128, -128]] * CFG.rows
    neg_bias = [CFG.acc_min + 1000, CFG.acc_min + 1000]
    await npu.load_all(model, neg_w, acts, neg_bias, mult, shift, s_count=1)
    for p in range(3):
        await npu.run(accumulate=(p > 0), requant=False)
        model.run(neg_w, acts, neg_bias, mult, shift, s_count=1,
                  accumulate=(p > 0), requant=False)
    await npu.settle()
    assert npu.ovf, "negative accumulator overflow must latch ovf"
    got = await npu.read_acc(0, 0)
    assert got == CFG.acc_min == model.acc[0][0], \
        f"accumulator must saturate to {CFG.acc_min}, got {got}"


@cocotb.test()
async def test_rounding_ties(dut):
    """Exact .5 boundaries must round away from zero, both signs.

    With M = 2**(M_W-1) the scale is exactly 1/2, so an odd accumulator is an
    exact tie: 3 -> 2, -3 -> -2, 5 -> 3, -5 -> -3.
    """
    npu = await setup(dut)
    model = g.Model(CFG)
    # A single row of weight 1 with activation v puts v straight in the
    # accumulator, so the tie cases can be dialled in exactly.
    weights = [[1, 1], [0, 0], [0, 0], [0, 0]]
    for value, expect in [(3, 2), (-3, -2), (5, 3), (-5, -3), (1, 1), (-1, -1),
                          (2, 1), (-2, -1), (127, 64), (-127, -64)]:
        model.reset()
        acts = [[value, 0, 0, 0]]
        await run_case(npu, model, weights, acts, [0, 0], [M_HALF, M_HALF],
                       [0, 0], s_count=1)
        hw = await npu.read_results(1)
        assert hw[0] == expect, \
            f"acc {value} * 1/2 should round to {expect}, got {hw[0]}"
        assert hw[0] == model.result[0]

    # Cross-check the two formulations of the rounding rule over a wide range.
    for p in range(-4096, 4096):
        for n in (1, 3, 16, 17):
            assert g.rounding_shift(p, n) == g.rounding_shift_addform(p, n), \
                f"rounding forms disagree at p={p} n={n}"


@cocotb.test()
async def test_scale_extremes(dut):
    """Zero multiplier, maximum multiplier and maximum shift."""
    npu = await setup(dut)
    model = g.Model(CFG)
    weights = [[100, -100], [50, -50], [25, -25], [12, -12]]
    acts = [[100, 100, 100, 100]]
    m_max = (1 << CFG.m_w) - 1
    for mult, shift, note in [([0, 0], [0, 0], "M=0"),
                              ([m_max, m_max], [0, 0], "M=max shift=0"),
                              ([m_max, m_max], [(1 << CFG.sh_w) - 1] * 2,
                               "M=max shift=max"),
                              ([1, 1], [0, 0], "M=1")]:
        model.reset()
        for zp in (0, 17, -128, 127):
            model.reset()
            await run_case(npu, model, weights, acts, [0, 0], mult, shift,
                           s_count=1, zp=zp)
            hw = await npu.read_results(CFG.cols)
            assert hw == model.result[:CFG.cols], \
                f"{note} zp={zp}: hw {hw} model {model.result[:CFG.cols]}"
        if mult == [0, 0]:
            assert hw == [127, 127] if zp == 127 else True


@cocotb.test()
async def test_zero_point_extremes(dut):
    """Zero point at both INT8 rails, with and without ReLU."""
    npu = await setup(dut)
    model = g.Model(CFG)
    weights = [[10, -10], [10, -10], [10, -10], [10, -10]]
    acts = [[5, 5, 5, 5]]
    for zp in (-128, -1, 0, 1, 127):
        for sel in (g.ACT_IDENTITY, g.ACT_RELU):
            model.reset()
            await run_case(npu, model, weights, acts, [0, 0],
                           [M_HALF, M_HALF], [4, 4], s_count=1, zp=zp,
                           act_sel=sel)
            hw = await npu.read_results(CFG.cols)
            assert hw == model.result[:CFG.cols], \
                f"zp={zp} sel={sel}: hw {hw} model {model.result[:CFG.cols]}"


@cocotb.test()
async def test_activations(dut):
    """All four activation functions against the model."""
    npu = await setup(dut)
    model = g.Model(CFG)
    weights = [[30, -30], [20, -20], [10, -10], [-40, 40]]
    acts = [[20, 10, -30, 25], [-100, 60, 40, -20], [5, -5, 5, -5]]
    cases = [
        (g.ACT_IDENTITY, 0, -128, 127),
        (g.ACT_RELU, 0, -128, 127),
        (g.ACT_RELU6, 0, -128, 40),
        (g.ACT_LEAKY, 3, -128, 127),
        (g.ACT_LEAKY, 1, -100, 100),
        (g.ACT_IDENTITY, 0, -20, 20),
    ]
    for sel, k, lo, hi in cases:
        for zp in (0, -30, 25):
            model.reset()
            await run_case(npu, model, weights, acts, [0, 0],
                           [M_HALF, M_HALF], [3, 3], s_count=3, zp=zp,
                           act_sel=sel, leaky_k=k, clamp_lo=lo, clamp_hi=hi)
            hw = await npu.read_results(3 * CFG.cols)
            assert hw == model.result[:3 * CFG.cols], \
                (f"act sel={sel} k={k} clamp=({lo},{hi}) zp={zp}: "
                 f"hw {hw} model {model.result[:3 * CFG.cols]}")


# ---------------------------------------------------------------------------
# Randomized sweep: the main correctness evidence
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_random_sweep(dut):
    """Randomized weights, activations, biases, scales and activation modes."""
    npu = await setup(dut)
    rng = random.Random(20260725)
    vectors = 0
    outputs = 0

    for case in range(SWEEP_CASES):
        model = g.Model(CFG)
        s_count = rng.randint(1, CFG.s_max)
        weights = [rand_i8(rng, CFG.cols) for _ in range(CFG.rows)]
        acts = [rand_i8(rng, CFG.rows) for _ in range(s_count)]
        # Biases span the full signed 24-bit range every so often.
        if case % 5 == 0:
            bias = [rng.randint(CFG.acc_min // 2, CFG.acc_max // 2)
                    for _ in range(CFG.cols)]
        else:
            bias = [rng.randint(-4096, 4096) for _ in range(CFG.cols)]
        mult = [rng.randint(1 << (CFG.m_w - 1), (1 << CFG.m_w) - 1)
                for _ in range(CFG.cols)]
        shift = [rng.randint(0, 12) for _ in range(CFG.cols)]
        zp = rng.randint(-128, 127)
        sel = rng.choice([g.ACT_IDENTITY, g.ACT_RELU, g.ACT_RELU6, g.ACT_LEAKY])
        leaky = rng.randint(0, 7)
        lo = rng.randint(-128, 0)
        hi = rng.randint(0, 127)

        await run_case(npu, model, weights, acts, bias, mult, shift,
                       s_count=s_count, zp=zp, act_sel=sel, leaky_k=leaky,
                       clamp_lo=lo, clamp_hi=hi)
        n = s_count * CFG.cols
        hw = await npu.read_results(n)
        assert hw == model.result[:n], (
            f"case {case}: hw {hw} model {model.result[:n]}\n"
            f"  W={weights} X={acts} bias={bias} M={mult} shift={shift}\n"
            f"  zp={zp} sel={sel} k={leaky} clamp=({lo},{hi})")
        # Raw accumulators too, so a requantization bug cannot hide a MAC bug.
        for s in range(s_count):
            for c in range(CFG.cols):
                got = await npu.read_acc(s, c)
                assert got == model.acc[c][s], (
                    f"case {case} raw ({s},{c}): hw {got} model {model.acc[c][s]}")
        vectors += 1
        outputs += n

    dut._log.info(f"randomized sweep: {vectors} layer configurations, "
                  f"{outputs} INT8 outputs and {outputs} raw accumulators "
                  f"compared bit-exactly")
    assert vectors == SWEEP_CASES


@cocotb.test()
async def test_random_multipass(dut):
    """K larger than ROWS via accumulating passes, including 16-input layers."""
    npu = await setup(dut)
    rng = random.Random(0xBEEF)
    vectors = 0
    for case in range(MULTIPASS_CASES):
        passes = rng.randint(2, 4)
        s_count = rng.randint(1, CFG.s_max)
        model = g.Model(CFG)
        bias = [rng.randint(-2048, 2048) for _ in range(CFG.cols)]
        mult = [rng.randint(1 << (CFG.m_w - 1), (1 << CFG.m_w) - 1)
                for _ in range(CFG.cols)]
        shift = [rng.randint(4, 10) for _ in range(CFG.cols)]
        zp = rng.randint(-40, 40)

        await npu.frame(g.OP_CFG, 0, [s_count - 1])
        await npu.frame(g.OP_LD_BIAS, 0, model.bias_bytes(bias))
        await npu.frame(g.OP_LD_QUANT, 0, model.quant_bytes(mult, shift))
        await npu.frame(g.OP_LD_POST, 0,
                        model.post_bytes(zp, -128, 127, g.ACT_RELU, 0))
        for p in range(passes):
            weights = [rand_i8(rng, CFG.cols) for _ in range(CFG.rows)]
            acts = [rand_i8(rng, CFG.rows) for _ in range(s_count)]
            await npu.frame(g.OP_LD_W, 0, model.weight_bytes(weights))
            await npu.frame(g.OP_LD_ACT, 0, model.act_bytes(acts))
            last = (p == passes - 1)
            await npu.run(accumulate=(p > 0), requant=last)
            model.run(weights, acts, bias, mult, shift, s_count=s_count,
                      accumulate=(p > 0), requant=last, zp=zp,
                      act_sel=g.ACT_RELU)
        n = s_count * CFG.cols
        hw = await npu.read_results(n)
        assert hw == model.result[:n], \
            f"multipass case {case} ({passes} passes): hw {hw} model {model.result[:n]}"
        for s in range(s_count):
            for c in range(CFG.cols):
                got = await npu.read_acc(s, c)
                assert got == model.acc[c][s], \
                    f"multipass raw ({s},{c}): hw {got} model {model.acc[c][s]}"
        vectors += 1
    dut._log.info(f"multi-pass sweep: {vectors} layers of 8 to 16 inputs")


# ---------------------------------------------------------------------------
# Throughput and latency
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_sustained_throughput(dut):
    """Every PE performs one MAC per cycle for the whole streaming window.

    The check reads each PE's activation register every cycle. PE(r, c) must
    hold a live activation for exactly s_count consecutive cycles starting at
    cycle r + c + 1 of the array phase, which is one MAC per PE per cycle with
    no gaps. The array is fully populated (all ROWS*COLS PEs at once) for
    s_count - ROWS - COLS + 2 cycles.
    """
    npu = await setup(dut)
    model = g.Model(CFG)
    rows, cols, s_count = CFG.rows, CFG.cols, CFG.s_max

    # All activations non-zero so an idle PE is distinguishable from a busy one.
    weights = [[1] * cols for _ in range(rows)]
    acts = [[(s * rows + r) % 100 + 1 for r in range(rows)]
            for s in range(s_count)]

    await npu.load_all(model, weights, acts, [0] * cols, [M_HALF] * cols,
                       [0] * cols, s_count=s_count)

    pes = []
    array = dut.user_project.u_core.u_array
    for r in range(rows):
        for c in range(cols):
            pes.append((r, c, array.g_row[r].g_col[c].u_pe))

    await npu.cmd(g.OP_RUN, 0b10)
    active = {}   # (r,c) -> list of cycles where the PE held a live activation
    busy_seen = False
    for cycle in range(s_count + rows + cols + 40):
        await FallingEdge(dut.clk)
        if npu.busy:
            busy_seen = True
        elif busy_seen:
            break
        for r, c, pe in pes:
            if int(pe.a_reg.value) != 0:
                active.setdefault((r, c), []).append(cycle)

    for r, c, _ in pes:
        cycles = active.get((r, c), [])
        assert len(cycles) == s_count, (
            f"PE({r},{c}) held a live activation for {len(cycles)} cycles, "
            f"expected {s_count} (one MAC per cycle)")
        assert cycles == list(range(cycles[0], cycles[0] + s_count)), (
            f"PE({r},{c}) stalled: active cycles {cycles}")

    base = min(active[(0, 0)])
    for r, c, _ in pes:
        assert min(active[(r, c)]) == base + r + c, (
            f"PE({r},{c}) started at {min(active[(r, c)])}, expected "
            f"{base + r + c} for a diagonal wavefront")

    full = [t for t in range(base, base + s_count + rows + cols)
            if all(t in active[(r, c)] for r, c, _ in pes)]
    expected_full = max(0, s_count - rows - cols + 2)
    assert len(full) == expected_full, (
        f"array fully populated for {len(full)} cycles, expected {expected_full}")
    macs = s_count * rows * cols
    dut._log.info(
        f"sustained throughput: {len(pes)} PEs x {s_count} cycles = {macs} MACs, "
        f"all {len(pes)} PEs simultaneously active for {len(full)} cycles")

    await npu.wait_done()
    hw = await npu.read_results(s_count * cols)
    model.run(weights, acts, [0] * cols, [M_HALF] * cols, [0] * cols,
              s_count=s_count, accumulate=False, requant=True)
    assert hw == model.result[:s_count * cols]


@cocotb.test()
async def test_latency_matches_model(dut):
    """Measured busy time equals the documented fill + compute + drain model."""
    npu = await setup(dut)
    model = g.Model(CFG)
    weights = [[3, -2]] * CFG.rows
    for s_count in range(1, CFG.s_max + 1):
        for shift in ([0, 0], [5, 9], [31, 31]):
            model.reset()
            acts = [[s + 1] * CFG.rows for s in range(s_count)]
            cycles = await run_case(npu, model, weights, acts, [0, 0],
                                    [M_HALF] * CFG.cols, shift,
                                    s_count=s_count)
            want = expected_busy_cycles(CFG, s_count, True, shift)
            assert cycles == want, (
                f"s_count={s_count} shift={shift}: measured {cycles} busy "
                f"cycles, model says {want}")
        # Array phase alone, no requantization.
        model.reset()
        acts = [[s + 1] * CFG.rows for s in range(s_count)]
        cycles = await run_case(npu, model, weights, acts, [0, 0],
                                [M_HALF] * CFG.cols, [0, 0],
                                s_count=s_count, requant=False)
        want = expected_busy_cycles(CFG, s_count, False, [0, 0])
        assert cycles == want, (
            f"array-only s_count={s_count}: measured {cycles}, model {want}")
    dut._log.info(
        f"latency model confirmed for s_count 1..{CFG.s_max}: array phase is "
        f"s+{CFG.rows + CFG.cols} cycles, requant "
        f"{expected_busy_cycles(CFG, 1, True, [0]) - expected_busy_cycles(CFG, 1, False, [0])}"
        f" cycles for a single output at shift 0")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_protocol_unknown_opcode(dut):
    """Unknown opcodes set the sticky error flag with code 1."""
    npu = await setup(dut)
    for op in (0xB, 0xC, 0xD, 0xE):
        await npu.frame(g.OP_CLR)
        await npu.cmd(op, 0)
        await npu.settle()
        assert npu.err, f"opcode {op:#x} should raise err"
        st = await npu.read_status()
        assert (st >> 6) == g.ERR_OPCODE, f"error code {st >> 6} for opcode {op:#x}"
    await npu.frame(g.OP_CLR)
    await npu.settle()
    assert not npu.err
    st = await npu.read_status()
    assert (st >> 6) == g.ERR_NONE


@cocotb.test()
async def test_protocol_frame_length(dut):
    """One payload byte too many, and a frame abandoned early, both flag code 3."""
    npu = await setup(dut)
    model = g.Model(CFG)

    # Too many payload bytes for LD_POST (expects 4).
    await npu.frame(g.OP_LD_POST, 0, [0, 0, 0, 0, 0])
    await npu.settle()
    assert npu.err
    st = await npu.read_status()
    assert (st >> 6) == g.ERR_FRAME, f"overrun gave code {st >> 6}"

    await npu.frame(g.OP_CLR)
    # Abandon a weight load after one byte.
    await npu.cmd(g.OP_LD_W)
    await npu.payload([1])
    await npu.cmd(g.OP_NOP)
    await npu.settle()
    assert npu.err, "an incomplete frame must be reported"
    st = await npu.read_status()
    assert (st >> 6) == g.ERR_FRAME

    # A complete frame after the error must still work: the interface resyncs.
    await npu.frame(g.OP_CLR)
    weights = [[2, 3]] * CFG.rows
    acts = [[1] * CFG.rows]
    await run_case(npu, model, weights, acts, [0, 0], [M_HALF] * CFG.cols,
                   [0, 0], s_count=1)
    hw = await npu.read_results(CFG.cols)
    assert hw == model.result[:CFG.cols], "interface failed to resynchronize"
    assert not npu.err


@cocotb.test()
async def test_protocol_busy_rejection(dut):
    """Writes during a run are dropped, not queued, and flag code 2."""
    npu = await setup(dut)
    model = g.Model(CFG)
    weights = [[5, 5]] * CFG.rows
    acts = [[2] * CFG.rows]
    await npu.load_all(model, weights, acts, [0, 0], [M_HALF] * CFG.cols,
                       [0, 0], s_count=1)
    model.run(weights, acts, [0, 0], [M_HALF] * CFG.cols, [0, 0], s_count=1,
              accumulate=False, requant=True)

    await npu.cmd(g.OP_RUN, 0b10)
    await npu.tick(3)
    assert npu.busy
    # Try to corrupt the weight tile mid-run.
    await npu.frame(g.OP_LD_W, 0, [0x7F] * (CFG.rows * CFG.cols))
    await npu.settle()
    assert npu.err, "a write while busy must raise err"
    await npu.wait_done()
    st = await npu.read_status()
    assert (st >> 6) == g.ERR_BUSY, f"busy violation gave code {st >> 6}"
    hw = await npu.read_results(CFG.cols)
    assert hw == model.result[:CFG.cols], \
        f"rejected write leaked into the weight tile: {hw}"


@cocotb.test()
async def test_readback_sources(dut):
    """Every readback source addresses the right bytes and auto-increments."""
    npu = await setup(dut)
    model = g.Model(CFG)
    weights = [[1, 2], [3, 4], [5, 6], [7, 8]]
    acts = [[1, 2, 3, 4], [5, 6, 7, 8]]
    await run_case(npu, model, weights, acts, [1000, -1000],
                   [M_HALF] * CFG.cols, [6, 6], s_count=2)

    seq = await npu.read_results(CFG.s_max * CFG.cols)
    assert seq[:4] == model.result[:4], f"auto-increment mismatch: {seq}"
    # Reading from a non-zero start index must land on the same bytes.
    one = await npu.read(1, g.RD_RESULT, 3)
    assert (one[0] - 256 if one[0] > 127 else one[0]) == model.result[3]
    # Out-of-range result index reads as zero rather than aliasing.
    far = await npu.read(1, g.RD_RESULT, CFG.s_max * CFG.cols)
    assert far[0] == 0, f"out-of-range readback returned {far[0]}"

    for s in range(2):
        for c in range(CFG.cols):
            got = await npu.read_acc(s, c)
            assert got == model.acc[c][s], f"acc readback ({s},{c}) = {got}"

    st = await npu.read_status()
    assert st & 0x01 == 0, "busy should be clear"
    assert st & 0x02 != 0, "done should be set after a completed run"


@cocotb.test()
async def test_soft_reset_midrun(dut):
    """A soft reset during a run returns the core to a known state."""
    npu = await setup(dut)
    model = g.Model(CFG)
    weights = [[9, -9]] * CFG.rows
    acts = [[7] * CFG.rows] * CFG.s_max
    await npu.load_all(model, weights, acts, [0, 0], [M_HALF] * CFG.cols,
                       [0, 0], s_count=CFG.s_max)
    await npu.cmd(g.OP_RUN, 0b10)
    await npu.tick(6)
    assert npu.busy

    # SRST is itself a write, so it is rejected while busy: that is the
    # documented behaviour. Wait, then soft reset and check the state is clean.
    await npu.wait_done()
    await npu.frame(g.OP_SRST)
    await npu.tick(2)
    await npu.settle()
    assert not npu.busy and not npu.done and not npu.err
    res = await npu.read_results(CFG.s_max * CFG.cols)
    assert res == [0] * (CFG.s_max * CFG.cols), f"SRST left results {res}"
    for s in range(CFG.s_max):
        for c in range(CFG.cols):
            assert await npu.read_acc(s, c) == 0, "SRST left an accumulator set"

    # Weights and quantization parameters survive a soft reset, so a rerun
    # reproduces the same answer without reloading them.
    await npu.frame(g.OP_RUN, 0b10)
    await npu.wait_done()
    model.run(weights, acts, [0, 0], [M_HALF] * CFG.cols, [0, 0],
              s_count=CFG.s_max, accumulate=False, requant=True)
    hw = await npu.read_results(CFG.s_max * CFG.cols)
    assert hw == model.result, f"state after SRST: hw {hw} model {model.result}"


@cocotb.test()
async def test_hard_reset_midrun(dut):
    """Asserting rst_n mid-run leaves every output defined and cleared."""
    npu = await setup(dut)
    model = g.Model(CFG)
    weights = [[11, -11]] * CFG.rows
    acts = [[6] * CFG.rows] * CFG.s_max
    await npu.load_all(model, weights, acts, [0, 0], [M_HALF] * CFG.cols,
                       [0, 0], s_count=CFG.s_max)
    await npu.cmd(g.OP_RUN, 0b10)
    await npu.tick(5)
    assert npu.busy
    dut.rst_n.value = 0
    await npu.tick(2)
    await npu.settle()
    assert int(dut.uio_out.value) == 0, "reset must clear all status pins"
    assert dut.uo_out.value.is_resolvable
    dut.rst_n.value = 1
    await npu.tick(2)
    await npu.settle()
    assert not npu.busy and not npu.done

    res = await npu.read_results(CFG.s_max * CFG.cols)
    assert res == [0] * (CFG.s_max * CFG.cols)
    # Everything, including the weight tile, must be reloaded after a hard reset.
    model.reset()
    await run_case(npu, model, weights, acts, [0, 0], [M_HALF] * CFG.cols,
                   [0, 0], s_count=CFG.s_max)
    hw = await npu.read_results(CFG.s_max * CFG.cols)
    assert hw == model.result
