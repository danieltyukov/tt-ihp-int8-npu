# SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
# SPDX-License-Identifier: Apache-2.0
"""cocotb driver for the framed byte protocol on the Tiny Tapeout pins.

Pin usage, from src/tt_um_danieltyukov_int8_npu.sv:
    ui_in[7:0]  data byte in        uo_out[7:0]  readback byte
    uio_in[0]   wr                  uio_out[3]   busy
    uio_in[1]   is_cmd              uio_out[4]   done
    uio_in[2]   rd                  uio_out[5]   err (sticky)
                                    uio_out[6]   sat (sticky)
                                    uio_out[7]   ovf (sticky)
"""

from __future__ import annotations

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge

import golden as g

WR = 1 << 0
IS_CMD = 1 << 1
RD = 1 << 2

BUSY_BIT = 1 << 3
DONE_BIT = 1 << 4
ERR_BIT = 1 << 5
SAT_BIT = 1 << 6
OVF_BIT = 1 << 7


class Npu:
    """Host-side model of the pin protocol. Every method is one or more cycles."""

    def __init__(self, dut, cfg: g.Cfg = g.Cfg(), clk_period_ns: int = 20):
        self.dut = dut
        self.cfg = cfg
        self.clk_period_ns = clk_period_ns
        self.cycles = 0  # cycles consumed since reset, for latency assertions

    async def start_clock(self) -> None:
        clock = Clock(self.dut.clk, self.clk_period_ns, unit="ns")
        cocotb.start_soon(clock.start())

    async def tick(self, n: int = 1) -> None:
        await ClockCycles(self.dut.clk, n)
        self.cycles += n

    async def sample(self) -> None:
        """Move to mid-cycle so registered outputs are settled before reading.

        cocotb resumes on a clock edge before the non-blocking updates of that
        edge are visible, so status pins must be sampled away from the edge.
        """
        await FallingEdge(self.dut.clk)

    async def reset(self, cycles: int = 4) -> None:
        self.dut.ena.value = 1
        self.dut.ui_in.value = 0
        self.dut.uio_in.value = 0
        self.dut.rst_n.value = 0
        await ClockCycles(self.dut.clk, cycles)
        self.dut.rst_n.value = 1
        await ClockCycles(self.dut.clk, 1)
        self.cycles = 0

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    async def _write(self, byte: int, is_cmd: bool) -> None:
        self.dut.ui_in.value = byte & 0xFF
        self.dut.uio_in.value = WR | (IS_CMD if is_cmd else 0)
        await self.tick()
        self.dut.uio_in.value = 0
        self.dut.ui_in.value = 0

    async def cmd(self, op: int, arg: int = 0) -> None:
        await self._write(((op & 0xF) << 4) | (arg & 0xF), True)

    async def payload(self, data) -> None:
        for b in data:
            await self._write(b, False)

    async def frame(self, op: int, arg: int = 0, data=()) -> None:
        await self.cmd(op, arg)
        await self.payload(data)

    # ------------------------------------------------------------------
    # Status and readback
    # ------------------------------------------------------------------
    def status_pins(self) -> int:
        return int(self.dut.uio_out.value)

    @property
    def busy(self) -> bool:
        return bool(self.status_pins() & BUSY_BIT)

    @property
    def done(self) -> bool:
        return bool(self.status_pins() & DONE_BIT)

    @property
    def err(self) -> bool:
        return bool(self.status_pins() & ERR_BIT)

    @property
    def sat(self) -> bool:
        return bool(self.status_pins() & SAT_BIT)

    @property
    def ovf(self) -> bool:
        return bool(self.status_pins() & OVF_BIT)

    async def wait_done(self, timeout: int = 6000) -> int:
        """Wait out a command, returning the number of cycles busy was high.

        RUN takes two cycles to reach the sequencer, so this first waits for
        busy to assert and only then counts, which makes the returned number
        directly comparable with the latency model in docs/DESIGN.md.
        """
        busy_cycles = 0
        seen = False
        for _ in range(timeout):
            await FallingEdge(self.dut.clk)
            self.cycles += 1
            if self.busy:
                seen = True
                busy_cycles += 1
            elif seen:
                return busy_cycles
        raise TimeoutError(f"busy stuck high for {timeout} cycles")

    async def read(self, n: int, src: int | None = None,
                   start: int | None = None) -> list[int]:
        """Read n bytes from a readback source, advancing the pointer."""
        if src is not None:
            await self.frame(g.OP_RDSEL, src, [start or 0])
        out = []
        for _ in range(n):
            await FallingEdge(self.dut.clk)
            out.append(int(self.dut.uo_out.value))
            self.dut.uio_in.value = RD
            await RisingEdge(self.dut.clk)
            self.cycles += 1
            self.dut.uio_in.value = 0
        return out

    async def read_status(self) -> int:
        vals = await self.read(1, g.RD_STATUS, 0)
        return vals[0]

    async def read_results(self, n: int) -> list[int]:
        raw = await self.read(n, g.RD_RESULT, 0)
        return [v - 256 if v > 127 else v for v in raw]

    async def read_acc(self, sample: int, chan: int) -> int:
        """Read one raw accumulator as a signed int32."""
        c_w = max(1, (self.cfg.cols - 1).bit_length())
        idx = ((sample << c_w) | chan) * 4
        b = await self.read(4, g.RD_ACC, idx)
        v = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)
        return v - (1 << 32) if v & (1 << 31) else v

    async def read_id(self) -> list[int]:
        return await self.read(8, g.RD_ID, 0)

    # ------------------------------------------------------------------
    # Higher level: program one layer pass
    # ------------------------------------------------------------------
    async def load_all(self, model: g.Model, weights, acts, bias, mult, shift,
                       *, s_count, zp=0, act_sel=g.ACT_IDENTITY, leaky_k=0,
                       clamp_lo=-128, clamp_hi=127) -> None:
        await self.frame(g.OP_CFG, 0, [s_count - 1])
        await self.frame(g.OP_LD_W, 0, model.weight_bytes(weights))
        await self.frame(g.OP_LD_ACT, 0, model.act_bytes(acts))
        await self.frame(g.OP_LD_BIAS, 0, model.bias_bytes(bias))
        await self.frame(g.OP_LD_QUANT, 0, model.quant_bytes(mult, shift))
        await self.frame(g.OP_LD_POST, 0,
                         model.post_bytes(zp, clamp_lo, clamp_hi, act_sel, leaky_k))

    async def run(self, *, accumulate: bool = False, requant: bool = True) -> int:
        arg = (1 if accumulate else 0) | (2 if requant else 0)
        await self.cmd(g.OP_RUN, arg)
        return await self.wait_done()
