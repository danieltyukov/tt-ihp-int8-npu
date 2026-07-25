// SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
// SPDX-License-Identifier: Apache-2.0
//
// Testbench for the arithmetic-variant equivalence tests: every adder
// architecture and every multiplier architecture in parallel on the same
// operands, so a divergence between variants is directly observable.
//
// Widths chosen to match what the NPU actually instantiates: 19-bit partial-sum
// adders, 25-bit accumulator adders, 26-bit requantizer adders, plus 8x8 signed
// multipliers.

`default_nettype none
`timescale 1ns / 1ps

module tb_arith ();

  initial begin
    $dumpfile("tb_arith.fst");
    $dumpvars(0, tb_arith);
    #1;
  end

  reg [41:0] a;
  reg [41:0] b;
  reg        cin;

  // Adders: 5 architectures at 4 widths.
  wire [18:0] sum19 [0:4];
  wire        cout19 [0:4];
  wire [24:0] sum25 [0:4];
  wire        cout25 [0:4];
  wire [25:0] sum26 [0:4];
  wire        cout26 [0:4];
  wire [41:0] sum42 [0:4];
  wire        cout42 [0:4];

  genvar k;
  generate
    for (k = 0; k < 5; k = k + 1) begin : g_add
      npu_adder #(.WIDTH(19), .ARCH(k)) u19 (.a(a[18:0]), .b(b[18:0]), .cin(cin), .sum(sum19[k]), .cout(cout19[k]));
      npu_adder #(.WIDTH(25), .ARCH(k)) u25 (.a(a[24:0]), .b(b[24:0]), .cin(cin), .sum(sum25[k]), .cout(cout25[k]));
      npu_adder #(.WIDTH(26), .ARCH(k)) u26 (.a(a[25:0]), .b(b[25:0]), .cin(cin), .sum(sum26[k]), .cout(cout26[k]));
      npu_adder #(.WIDTH(42), .ARCH(k)) u42 (.a(a[41:0]), .b(b[41:0]), .cin(cin), .sum(sum42[k]), .cout(cout42[k]));
    end
  endgenerate

  // Multipliers: 3 architectures, signed 8x8, one per adder architecture for
  // the final carry-propagate stage.
  reg signed [7:0] ma;
  reg signed [7:0] mb;
  wire signed [15:0] prod [0:2];
  wire signed [15:0] prod_addarch [0:4];

  generate
    for (k = 0; k < 3; k = k + 1) begin : g_mul
      npu_mult #(.A_W(8), .B_W(8), .MUL_ARCH(k), .ADD_ARCH(2)) um (.a(ma), .b(mb), .p(prod[k]));
    end
    for (k = 0; k < 5; k = k + 1) begin : g_mul_add
      npu_mult #(.A_W(8), .B_W(8), .MUL_ARCH(1), .ADD_ARCH(k)) uma (.a(ma), .b(mb), .p(prod_addarch[k]));
    end
  endgenerate

endmodule
