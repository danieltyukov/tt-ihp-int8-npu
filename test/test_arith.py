# SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
# SPDX-License-Identifier: Apache-2.0
"""Arithmetic-variant equivalence tests.

Every adder architecture must produce bit-identical sums and carry-outs, and
every multiplier architecture bit-identical products, for the same operands.
Divergence between variants is a bug by construction: the whole point of having
five adders and three multipliers is that they are interchangeable.

The reference is Python integer arithmetic, so the tests check both
"all variants agree" and "all variants are right".
"""

from __future__ import annotations

import random

import cocotb
from cocotb.triggers import Timer

ADDER_NAMES = ["ripple-carry", "Brent-Kung", "Kogge-Stone", "Sklansky",
               "Han-Carlson"]
MULT_NAMES = ["Baugh-Wooley array", "Baugh-Wooley Wallace", "Booth-4 Wallace"]
ADD_WIDTHS = [19, 25, 26, 42]


def _sig(dut, name, idx):
    return getattr(dut, name)[idx]


async def _settle():
    await Timer(2, unit="ns")


async def check_adders(dut, a: int, b: int, cin: int, errors: list[str]) -> int:
    dut.a.value = a
    dut.b.value = b
    dut.cin.value = cin
    await _settle()
    checks = 0
    for w in ADD_WIDTHS:
        mask = (1 << w) - 1
        exp_sum = (a & mask) + (b & mask) + cin
        exp_cout = exp_sum >> w
        exp_sum &= mask
        for k, name in enumerate(ADDER_NAMES):
            got_sum = int(_sig(dut, f"sum{w}", k).value)
            got_cout = int(_sig(dut, f"cout{w}", k).value)
            checks += 2
            if got_sum != exp_sum or got_cout != exp_cout:
                errors.append(
                    f"adder w={w} {name}: a={a & mask:#x} b={b & mask:#x} "
                    f"cin={cin} got {got_cout}:{got_sum:#x} "
                    f"expected {exp_cout}:{exp_sum:#x}")
    return checks


async def check_mults(dut, ma: int, mb: int, errors: list[str]) -> int:
    dut.ma.value = ma
    dut.mb.value = mb
    await _settle()
    exp = ma * mb
    checks = 0
    for k, name in enumerate(MULT_NAMES):
        got = int(_sig(dut, "prod", k).value.signed_integer)
        checks += 1
        if got != exp:
            errors.append(f"mult {name}: {ma} * {mb} gave {got}, expected {exp}")
    for k, name in enumerate(ADDER_NAMES):
        got = int(_sig(dut, "prod_addarch", k).value.signed_integer)
        checks += 1
        if got != exp:
            errors.append(
                f"mult Wallace with {name} CPA: {ma} * {mb} gave {got}, "
                f"expected {exp}")
    return checks


@cocotb.test()
async def test_multiplier_exhaustive(dut):
    """All 65536 signed 8x8 operand pairs through all multiplier variants."""
    errors: list[str] = []
    checks = 0
    for ma in range(-128, 128):
        for mb in range(-128, 128):
            checks += await check_mults(dut, ma, mb, errors)
            if errors:
                break
        if errors:
            break
    dut._log.info(f"multiplier equivalence: {checks} checks over 65536 operand "
                  f"pairs x {len(MULT_NAMES) + len(ADDER_NAMES)} variants")
    assert not errors, "\n".join(errors[:10])


@cocotb.test()
async def test_multiplier_int8_min(dut):
    """-128 is the asymmetric INT8 minimum and the classic sign-handling bug."""
    errors: list[str] = []
    corners = [-128, -127, -1, 0, 1, 126, 127]
    for ma in corners:
        for mb in corners:
            await check_mults(dut, ma, mb, errors)
    await check_mults(dut, -128, -128, errors)   # +16384, needs 16 bits
    await check_mults(dut, -128, 127, errors)    # -16256
    assert not errors, "\n".join(errors[:10])


@cocotb.test()
async def test_adder_directed(dut):
    """Carry-propagation corners: all zeros, all ones, alternating patterns."""
    errors: list[str] = []
    ones = (1 << 42) - 1
    vectors = [
        (0, 0, 0), (0, 0, 1), (ones, 0, 1), (ones, ones, 1), (ones, ones, 0),
        (0xAAAAAAAAAAA, 0x55555555555, 1), (0x55555555555, 0xAAAAAAAAAAA, 0),
        (1, ones, 0), (ones, 1, 0), (1 << 41, 1 << 41, 0),
    ]
    checks = 0
    for a, b, cin in vectors:
        checks += await check_adders(dut, a, b, cin, errors)
    dut._log.info(f"adder directed: {checks} checks")
    assert not errors, "\n".join(errors[:10])


@cocotb.test()
async def test_adder_random(dut):
    """Randomized sweep across all five architectures and four widths."""
    rng = random.Random(0xA11CE)
    errors: list[str] = []
    checks = 0
    vectors = 600
    for _ in range(vectors):
        a = rng.getrandbits(42)
        b = rng.getrandbits(42)
        checks += await check_adders(dut, a, b, rng.getrandbits(1), errors)
    dut._log.info(f"adder random: {vectors} vectors, {checks} checks")
    assert not errors, "\n".join(errors[:10])
