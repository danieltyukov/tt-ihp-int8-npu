// SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
// SPDX-License-Identifier: Apache-2.0
//
// Formal miter for npu_mult. Proves that each partial-product and reduction
// style computes the exact two's-complement product, for every one of the
// 65536 signed 8x8 input pairs, and for every final-adder architecture.
//
// This is what makes the multiplier rows of the PPA table a like-for-like
// comparison: the variants are provably the same function, so the area and
// depth differences are the only differences.

`default_nettype none

module miter_mult #(
    parameter int A_W      = 8,
    parameter int B_W      = 8,
    parameter int MUL_ARCH = 0,
    parameter int ADD_ARCH = 0
) (
    input wire signed [A_W-1:0] a,
    input wire signed [B_W-1:0] b
);

  wire signed [A_W+B_W-1:0] p_dut;

  npu_mult #(
      .A_W     (A_W),
      .B_W     (B_W),
      .MUL_ARCH(MUL_ARCH),
      .ADD_ARCH(ADD_ARCH)
  ) u_dut (
      .a(a),
      .b(b),
      .p(p_dut)
  );

  wire signed [A_W+B_W-1:0] p_ref = a * b;

  always_comb begin
    assert (p_dut == p_ref);
  end

endmodule

`default_nettype wire
