// SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
// SPDX-License-Identifier: Apache-2.0
//
// Standalone self-checking bench for the arithmetic library. Runs under plain
// iverilog (no cocotb) so the adder and multiplier generators can be checked
// across widths and architectures in one shot:
//
//   iverilog -g2012 -o build/tb_arith test/tb_arith.sv src/npu_*.sv && vvp build/tb_arith
//
// The cocotb suite re-checks the same modules through the DUT wrapper; this
// bench exists to sweep widths that the ASIC configuration does not use.

`default_nettype none
`timescale 1ns / 1ps

module tb_arith;

  int errors = 0;
  int checks = 0;

  // ---------------------------------------------------------------------------
  // Adders: every architecture, several widths.
  // ---------------------------------------------------------------------------
  logic [63:0] add_a, add_b;
  logic        add_cin;

  logic [3:0]  s_w4  [0:4];
  logic        c_w4  [0:4];
  logic [7:0]  s_w8  [0:4];
  logic        c_w8  [0:4];
  logic [15:0] s_w16 [0:4];
  logic        c_w16 [0:4];
  logic [24:0] s_w25 [0:4];
  logic        c_w25 [0:4];
  logic [41:0] s_w42 [0:4];
  logic        c_w42 [0:4];

  generate
    for (genvar k = 0; k < 5; k++) begin : g_add
      npu_adder #(.WIDTH(4),  .ARCH(k)) u4  (.a(add_a[3:0]),  .b(add_b[3:0]),  .cin(add_cin), .sum(s_w4[k]),  .cout(c_w4[k]));
      npu_adder #(.WIDTH(8),  .ARCH(k)) u8  (.a(add_a[7:0]),  .b(add_b[7:0]),  .cin(add_cin), .sum(s_w8[k]),  .cout(c_w8[k]));
      npu_adder #(.WIDTH(16), .ARCH(k)) u16 (.a(add_a[15:0]), .b(add_b[15:0]), .cin(add_cin), .sum(s_w16[k]), .cout(c_w16[k]));
      npu_adder #(.WIDTH(25), .ARCH(k)) u25 (.a(add_a[24:0]), .b(add_b[24:0]), .cin(add_cin), .sum(s_w25[k]), .cout(c_w25[k]));
      npu_adder #(.WIDTH(42), .ARCH(k)) u42 (.a(add_a[41:0]), .b(add_b[41:0]), .cin(add_cin), .sum(s_w42[k]), .cout(c_w42[k]));
    end
  endgenerate

  // ---------------------------------------------------------------------------
  // Multipliers: every architecture, the widths the NPU instantiates.
  // ---------------------------------------------------------------------------
  logic signed [7:0]  ma8;
  logic signed [7:0]  mb8;
  logic signed [24:0] ma25;
  logic signed [8:0]  mb9;

  logic signed [15:0] p8  [0:2];
  logic signed [33:0] p25 [0:2];

  generate
    for (genvar k = 0; k < 3; k++) begin : g_mul
      npu_mult #(.A_W(8),  .B_W(8), .MUL_ARCH(k), .ADD_ARCH(2)) m8  (.a(ma8),  .b(mb8), .p(p8[k]));
      npu_mult #(.A_W(25), .B_W(9), .MUL_ARCH(k), .ADD_ARCH(2)) m25 (.a(ma25), .b(mb9), .p(p25[k]));
    end
  endgenerate

  task automatic check(input string what, input logic [63:0] got, input logic [63:0] exp);
    checks++;
    if (got !== exp) begin
      errors++;
      if (errors < 25) $display("FAIL %s: got %0d (0x%0h) expected %0d (0x%0h)", what, got, got, exp, exp);
    end
  endtask

  task automatic do_add_checks;
    for (int k = 0; k < 5; k++) begin
      check($sformatf("add W4 ARCH%0d",  k), {60'b0, s_w4[k]},  (add_a[3:0]  + add_b[3:0]  + add_cin) & 64'hF);
      check($sformatf("cout W4 ARCH%0d", k), {63'b0, c_w4[k]},  ({1'b0, add_a[3:0]}  + {1'b0, add_b[3:0]}  + add_cin) >> 4);
      check($sformatf("add W8 ARCH%0d",  k), {56'b0, s_w8[k]},  (add_a[7:0]  + add_b[7:0]  + add_cin) & 64'hFF);
      check($sformatf("cout W8 ARCH%0d", k), {63'b0, c_w8[k]},  ({1'b0, add_a[7:0]}  + {1'b0, add_b[7:0]}  + add_cin) >> 8);
      check($sformatf("add W16 ARCH%0d", k), {48'b0, s_w16[k]}, (add_a[15:0] + add_b[15:0] + add_cin) & 64'hFFFF);
      check($sformatf("cout W16 ARCH%0d",k), {63'b0, c_w16[k]}, ({1'b0, add_a[15:0]} + {1'b0, add_b[15:0]} + add_cin) >> 16);
      check($sformatf("add W25 ARCH%0d", k), {39'b0, s_w25[k]}, (add_a[24:0] + add_b[24:0] + add_cin) & 64'h1FFFFFF);
      check($sformatf("cout W25 ARCH%0d",k), {63'b0, c_w25[k]}, ({1'b0, add_a[24:0]} + {1'b0, add_b[24:0]} + add_cin) >> 25);
      check($sformatf("add W42 ARCH%0d", k), {22'b0, s_w42[k]}, (add_a[41:0] + add_b[41:0] + add_cin) & 64'h3FFFFFFFFFF);
      check($sformatf("cout W42 ARCH%0d",k), {63'b0, c_w42[k]}, ({1'b0, add_a[41:0]} + {1'b0, add_b[41:0]} + add_cin) >> 42);
    end
  endtask

  task automatic do_mul_checks;
    logic signed [15:0] exp8;
    logic signed [33:0] exp25;
    exp8  = ma8 * mb8;
    exp25 = ma25 * mb9;
    for (int k = 0; k < 3; k++) begin
      check($sformatf("mul 8x8 ARCH%0d a=%0d b=%0d", k, ma8, mb8),
            {{48{p8[k][15]}}, p8[k]}, {{48{exp8[15]}}, exp8});
      check($sformatf("mul 25x9 ARCH%0d a=%0d b=%0d", k, ma25, mb9),
            {{30{p25[k][33]}}, p25[k]}, {{30{exp25[33]}}, exp25});
    end
  endtask

  initial begin
    // Exhaustive 8x8 signed multiply: all 65536 operand pairs.
    for (int i = -128; i <= 127; i++) begin
      for (int j = -128; j <= 127; j++) begin
        ma8 = 8'(i);
        mb8 = 8'(j);
        ma25 = 25'(i * 7919);
        mb9  = 9'(j >> 1);
        #1;
        do_mul_checks();
      end
    end
    $display("after exhaustive 8x8 multiply sweep: %0d checks, %0d errors", checks, errors);

    // Randomized wide multiply corners.
    for (int t = 0; t < 4000; t++) begin
      ma8  = 8'($urandom());
      mb8  = 8'($urandom());
      ma25 = 25'($urandom());
      mb9  = 9'($urandom());
      if (t < 8) begin
        ma25 = (t & 1) ? 25'sh0FFFFFF : 25'sh1000000;  // widest positive / negative
        mb9  = (t & 2) ? 9'sh0FF      : 9'sh100;
      end
      #1;
      do_mul_checks();
    end

    // Adder sweep: directed corners then random.
    for (int t = 0; t < 6000; t++) begin
      case (t)
        0: begin add_a = 64'h0;                add_b = 64'h0;                add_cin = 1'b0; end
        1: begin add_a = 64'h0;                add_b = 64'h0;                add_cin = 1'b1; end
        2: begin add_a = 64'hFFFFFFFFFFFFFFFF; add_b = 64'h0;                add_cin = 1'b1; end
        3: begin add_a = 64'hFFFFFFFFFFFFFFFF; add_b = 64'hFFFFFFFFFFFFFFFF; add_cin = 1'b1; end
        4: begin add_a = 64'hAAAAAAAAAAAAAAAA; add_b = 64'h5555555555555555; add_cin = 1'b1; end
        5: begin add_a = 64'h5555555555555555; add_b = 64'hAAAAAAAAAAAAAAAA; add_cin = 1'b0; end
        default: begin
          add_a   = {$urandom(), $urandom()};
          add_b   = {$urandom(), $urandom()};
          add_cin = 1'($urandom());
        end
      endcase
      #1;
      do_add_checks();
    end

    $display("tb_arith: %0d checks, %0d errors", checks, errors);
    if (errors == 0) $display("TB_ARITH PASS");
    else $display("TB_ARITH FAIL");
    $finish;
  end

endmodule

`default_nettype wire
