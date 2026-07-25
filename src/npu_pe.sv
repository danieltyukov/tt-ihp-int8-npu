// SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
// SPDX-License-Identifier: Apache-2.0
//
// npu_pe: one processing element of the weight-stationary systolic array.
//
//   a_in  ->[a_reg]-> a_out        activations move west to east, one PE/cycle
//            |
//   w_reg (resident, loaded once through the w_in/w_out shift chain)
//            |
//   psum_in -+-> [psum_reg] -> psum_out   partial sums move north to south
//
// Every cycle the PE computes
//   psum_reg <= psum_in + w_reg * a_reg
// so one MAC per PE per cycle is sustained for as long as activations keep
// arriving. There is deliberately no compute enable: when the array is idle the
// row inputs are driven to zero, the products are zero and the psum chain
// flushes itself, which removes ROWS*COLS*(PSUM_W+8) enable multiplexers from
// the design.
//
// a_reg holds the activation that the multiplier sees, so the product a PE
// contributes in cycle T uses the activation that arrived in cycle T-1. That
// one-cycle offset is what makes the diagonal wavefront line up: see
// docs/DESIGN.md for the full timing derivation.

`default_nettype none

module npu_pe #(
    parameter int PSUM_W   = 19,
    parameter int MUL_ARCH = 0,
    parameter int ADD_ARCH = 0
) (
    input  wire                     clk,
    input  wire                     rst_n,
    // Weight shift chain.
    input  wire                     w_en,
    input  wire  [7:0]              w_in,
    output wire  [7:0]              w_out,
    // Activation, west to east.
    input  wire  [7:0]              a_in,
    output wire  [7:0]              a_out,
    // Partial sum, north to south.
    input  wire signed [PSUM_W-1:0] psum_in,
    output wire signed [PSUM_W-1:0] psum_out
);

  initial begin
    if (PSUM_W < 17) $fatal(1, "npu_pe: PSUM_W must be >= 17 to hold a signed 8x8 product");
  end

  logic signed [7:0]        w_reg;
  logic signed [7:0]        a_reg;
  logic signed [PSUM_W-1:0] psum_reg;

  wire signed [15:0] prod;

  npu_mult #(
      .A_W     (8),
      .B_W     (8),
      .MUL_ARCH(MUL_ARCH),
      .ADD_ARCH(ADD_ARCH)
  ) u_mul (
      .a(a_reg),
      .b(w_reg),
      .p(prod)
  );

  // Sign-extend the 16-bit product into the accumulator chain width.
  wire signed [PSUM_W-1:0] prod_ext = {{(PSUM_W - 16) {prod[15]}}, prod};

  wire [PSUM_W-1:0] psum_sum;
  wire              psum_cout;

  npu_adder #(
      .WIDTH(PSUM_W),
      .ARCH (ADD_ARCH)
  ) u_add (
      .a   (psum_in),
      .b   (prod_ext),
      .cin (1'b0),
      .sum (psum_sum),
      .cout(psum_cout)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      w_reg    <= 8'sd0;
      a_reg    <= 8'sd0;
      psum_reg <= '0;
    end else begin
      a_reg    <= a_in;
      psum_reg <= psum_sum;
      if (w_en) w_reg <= w_in;
    end
  end

  assign w_out    = w_reg;
  assign a_out    = a_reg;
  assign psum_out = psum_reg;

  // The chain never overflows by construction (see npu_array), so the adder
  // carry-out is unused on purpose.
  wire _unused = &{1'b0, psum_cout};

endmodule

`default_nettype wire
