# SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
# SPDX-License-Identifier: Apache-2.0
"""Independent INT8 reference model for tt_um_danieltyukov_int8_npu.

This file is written from the datasheet in docs/info.md, not from the RTL, and
is what every hardware result is compared against. It uses plain Python integers
so there is no doubt about overflow behaviour, plus NumPy only for generating
and reshaping tensors.

Requantization semantics implemented here (and in the RTL):

    n = M_W + shift
    p = acc * M                              exact, arbitrary precision
    y = (p >> n) + round_up                  arithmetic shift, floor
    round_up = R                if p >= 0
               R and S          if p < 0
    R = bit n-1 of p, S = OR of bits n-2..0 of p
    q = saturate_int8(y + zero_point)

which is exactly gemmlowp's RoundingDivideByPOT, that is, round-to-nearest with
ties broken away from zero. TFLite reaches the same rule in two steps
(SaturatingRoundingDoublingHighMul then RoundingDivideByPOT) with a Q0.31
multiplier; this design uses a single Q0.M_W multiplier and one rounding step.
See docs/DESIGN.md for why, and for the measured effect on accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INT8_MIN, INT8_MAX = -128, 127

ACT_IDENTITY, ACT_RELU, ACT_RELU6, ACT_LEAKY = 0, 1, 2, 3

OP_NOP, OP_CFG, OP_LD_W, OP_LD_ACT = 0x0, 0x1, 0x2, 0x3
OP_LD_BIAS, OP_LD_QUANT, OP_LD_POST = 0x4, 0x5, 0x6
OP_RUN, OP_RDSEL, OP_CLR, OP_SRST, OP_ID = 0x7, 0x8, 0x9, 0xA, 0xF

RD_RESULT, RD_ACC, RD_STATUS, RD_ID = 0, 1, 2, 3

ERR_NONE, ERR_OPCODE, ERR_BUSY, ERR_FRAME = 0, 1, 2, 3


@dataclass(frozen=True)
class Cfg:
    """Hardware geometry. Must match the parameters the DUT was built with."""

    rows: int = 4
    cols: int = 2
    s_max: int = 4
    acc_w: int = 24
    m_w: int = 16
    sh_w: int = 5

    @property
    def psum_w(self) -> int:
        return 16 + (self.rows).bit_length()  # 16 + ceil(log2(rows+1))

    @property
    def acc_min(self) -> int:
        return -(1 << (self.acc_w - 1))

    @property
    def acc_max(self) -> int:
        return (1 << (self.acc_w - 1)) - 1

    @property
    def bias_bytes(self) -> int:
        return (self.acc_w + 7) // 8

    @property
    def m_bytes(self) -> int:
        return (self.m_w + 7) // 8

    @property
    def quant_bytes(self) -> int:
        return self.m_bytes + 1


def sat_int8(v: int) -> tuple[int, bool]:
    """Clamp to the INT8 range, reporting whether a rail was hit."""
    if v > INT8_MAX:
        return INT8_MAX, True
    if v < INT8_MIN:
        return INT8_MIN, True
    return v, False


def rounding_shift(p: int, n: int) -> int:
    """Arithmetic right shift by n, rounding to nearest with ties away from zero.

    Implemented the way the hardware does it, from the round and sticky bits, so
    that a divergence in the hardware shows up as a mismatch rather than being
    masked by a different but equivalent formula.
    """
    if n <= 0:
        return p
    shifted = p >> n  # Python floors, which is an arithmetic shift
    round_bit = (p >> (n - 1)) & 1
    sticky = 1 if (p & ((1 << (n - 1)) - 1)) != 0 else 0
    if p < 0:
        return shifted + (round_bit & sticky)
    return shifted + round_bit


def rounding_shift_addform(p: int, n: int) -> int:
    """Same function expressed as add-then-shift, used to cross-check the above."""
    if n <= 0:
        return p
    return (p + (1 << (n - 1)) - (1 if p < 0 else 0)) >> n


def requantize(acc: int, m: int, shift: int, zp: int, cfg: Cfg) -> tuple[int, bool]:
    """Integer-only affine requantization of one accumulator value."""
    n = cfg.m_w + shift
    y = rounding_shift(acc * m, n)
    return sat_int8(y + zp)


def activation(q: int, sel: int, leaky_k: int, zp: int,
               clamp_lo: int, clamp_hi: int) -> int:
    """Activation in the quantized domain. Mirrors src/npu_activation.sv."""
    if sel == ACT_LEAKY:
        cand = q if q >= zp else zp + ((q - zp) >> leaky_k)
        cand, _ = sat_int8(cand)
    else:
        cand = q

    lo = zp if sel in (ACT_RELU, ACT_RELU6) else clamp_lo
    hi = INT8_MAX if sel == ACT_RELU else clamp_hi

    if cand < lo:
        return lo
    if cand > hi:
        return hi
    return cand


def sat_acc(v: int, cfg: Cfg) -> tuple[int, bool]:
    """Saturating accumulate into an ACC_W-bit signed accumulator."""
    if v > cfg.acc_max:
        return cfg.acc_max, True
    if v < cfg.acc_min:
        return cfg.acc_min, True
    return v, False


def raw_sums(weights, acts, cfg: Cfg) -> list[list[int]]:
    """Exact integer matrix product of one weight tile with the activations.

    weights: (rows, cols) signed INT8
    acts:    (s_count, rows) signed INT8
    returns  [s][c] exact integer dot products
    """
    w = np.asarray(weights, dtype=np.int64)
    x = np.asarray(acts, dtype=np.int64)
    assert w.shape == (cfg.rows, cfg.cols), w.shape
    assert x.shape[1] == cfg.rows, x.shape
    return (x @ w).tolist()


class Model:
    """Cycle-free functional model of the accelerator's programmer view."""

    def __init__(self, cfg: Cfg = Cfg()):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        c = self.cfg
        self.acc = [[0] * c.s_max for _ in range(c.cols)]
        self.result = [0] * (c.s_max * c.cols)
        self.ovf = False
        self.sat = False

    def run(self, weights, acts, bias, mult, shift, *, s_count,
            accumulate: bool, requant: bool, zp: int = 0,
            act_sel: int = ACT_IDENTITY, leaky_k: int = 0,
            clamp_lo: int = INT8_MIN, clamp_hi: int = INT8_MAX) -> None:
        """One RUN command.

        bias/mult/shift are per output channel; acts is (s_count, rows).
        """
        c = self.cfg
        sums = raw_sums(weights, acts[:s_count], c)
        for s in range(s_count):
            for ch in range(c.cols):
                base = bias[ch] if not accumulate else self.acc[ch][s]
                v, o = sat_acc(base + sums[s][ch], c)
                self.acc[ch][s] = v
                self.ovf |= o
        if not requant:
            return
        for s in range(s_count):
            for ch in range(c.cols):
                q, st = requantize(self.acc[ch][s], mult[ch], shift[ch], zp, c)
                self.sat |= st
                self.result[s * c.cols + ch] = activation(
                    q, act_sel, leaky_k, zp, clamp_lo, clamp_hi)

    # ------------------------------------------------------------------
    # Byte-stream helpers, so tests and the model agree on wire format.
    # ------------------------------------------------------------------
    def weight_bytes(self, weights) -> list[int]:
        w = np.asarray(weights, dtype=np.int64)
        return [int(v) & 0xFF for v in w.reshape(-1)]  # row-major

    def act_bytes(self, acts) -> list[int]:
        c = self.cfg
        x = np.zeros((c.s_max, c.rows), dtype=np.int64)
        a = np.asarray(acts, dtype=np.int64)
        x[: a.shape[0], :] = a
        return [int(v) & 0xFF for v in x.reshape(-1)]  # sample-major

    def bias_bytes(self, bias) -> list[int]:
        c = self.cfg
        out = []
        for v in bias:
            u = int(v) & ((1 << c.acc_w) - 1)
            out += [(u >> (8 * i)) & 0xFF for i in range(c.bias_bytes)]
        return out

    def quant_bytes(self, mult, shift) -> list[int]:
        c = self.cfg
        out = []
        for m, sh in zip(mult, shift):
            out += [(int(m) >> (8 * i)) & 0xFF for i in range(c.m_bytes)]
            out += [int(sh) & 0xFF]
        return out

    def post_bytes(self, zp, clamp_lo, clamp_hi, act_sel, leaky_k) -> list[int]:
        return [int(zp) & 0xFF, int(clamp_lo) & 0xFF, int(clamp_hi) & 0xFF,
                (int(act_sel) & 0x3) | ((int(leaky_k) & 0x7) << 2)]


def quantize_multiplier(scale: float, cfg: Cfg = Cfg()) -> tuple[int, int]:
    """Split a positive float scale into (M, shift) with M normalized.

    The hardware computes scale = M / 2**(M_W + shift) with shift >= 0, so the
    representable range is 0 < scale < 1. M is pushed as high as it will go for
    maximum precision, exactly like TFLite's QuantizeMultiplier but with a
    Q0.M_W mantissa instead of Q0.31.
    """
    if not (0.0 < scale < 1.0):
        raise ValueError(f"scale {scale} outside the representable range (0, 1)")
    m_hi = 1 << cfg.m_w
    m_lo = 1 << (cfg.m_w - 1)
    for shift in range(1 << cfg.sh_w):
        m = int(round(scale * (1 << (cfg.m_w + shift))))
        if m >= m_hi:
            continue
        if m >= m_lo:
            return m, shift
        # scale is too small for this shift to normalize; keep increasing it
    raise ValueError(f"scale {scale} needs a shift beyond SH_W={cfg.sh_w}")


def effective_scale(m: int, shift: int, cfg: Cfg = Cfg()) -> float:
    """The exact scale the hardware applies for a given (M, shift)."""
    return m / float(1 << (cfg.m_w + shift))
