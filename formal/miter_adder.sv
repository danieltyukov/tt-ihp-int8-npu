// SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
// SPDX-License-Identifier: Apache-2.0
//
// Formal miter for npu_adder. The PPA comparison in README.md rests on the
// claim that all five carry-network architectures compute the same function
// and differ only in area and logic depth. This proves the stronger statement:
// each architecture equals a + b + cin over every input, not just over the
// random vectors test/test_arith.py samples.
//
// The reference is the behavioral sum, so a proof here is a correctness proof
// and not merely an agreement proof between two implementations.

`default_nettype none

module miter_adder #(
    parameter int WIDTH = 19,
    parameter int ARCH  = 0
) (
    input wire [WIDTH-1:0] a,
    input wire [WIDTH-1:0] b,
    input wire             cin
);

  wire [WIDTH-1:0] sum_dut;
  wire             cout_dut;

  npu_adder #(
      .WIDTH(WIDTH),
      .ARCH (ARCH)
  ) u_dut (
      .a   (a),
      .b   (b),
      .cin (cin),
      .sum (sum_dut),
      .cout(cout_dut)
  );

  wire [WIDTH:0] ref_sum = {1'b0, a} + {1'b0, b} + {{WIDTH{1'b0}}, cin};

  always_comb begin
    assert (sum_dut == ref_sum[WIDTH-1:0]);
    assert (cout_dut == ref_sum[WIDTH]);
  end

endmodule

`default_nettype wire
