// SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
// SPDX-License-Identifier: Apache-2.0
//
// npu_activation: runtime-selectable activation function in the quantized
// domain, applied after requantization.
//
// In integer inference "zero" is the output zero point, not 0, so every
// activation is expressed relative to zp:
//
//   SEL  name       result
//   0    identity   clamp(q, clamp_lo, clamp_hi)
//   1    ReLU       clamp(q, zp, 127)
//   2    ReLU6      clamp(q, zp, clamp_hi)   clamp_hi holds quantize(6.0)
//   3    leaky ReLU clamp(q >= zp ? q : zp + ((q - zp) >>> k),
//                         clamp_lo, clamp_hi)
//
// ReLU6 uses clamp_hi because the integer representation of 6.0 depends on the
// output scale, which only the host knows: clamp_hi = round(6/scale) + zp.
// The leaky slope is 2^-k with k = leaky_k, so k = 0 leaks with slope 1 (which
// is the identity) and k = 3 gives the common slope of 0.125. The shift is
// arithmetic, so it rounds toward negative infinity.

`default_nettype none

module npu_activation (
    input  wire signed [7:0] q_in,
    input  wire        [1:0] sel,
    input  wire        [2:0] leaky_k,
    input  wire signed [7:0] zp,
    input  wire signed [7:0] clamp_lo,
    input  wire signed [7:0] clamp_hi,
    output wire signed [7:0] y
);

  localparam logic [1:0] ACT_IDENTITY = 2'd0;
  localparam logic [1:0] ACT_RELU     = 2'd1;
  localparam logic [1:0] ACT_RELU6    = 2'd2;
  localparam logic [1:0] ACT_LEAKY    = 2'd3;

  // Leaky path: scale the distance below the zero point, then move back.
  wire signed [8:0] delta    = {q_in[7], q_in} - {zp[7], zp};
  wire signed [8:0] leaked  = delta >>> leaky_k;
  wire signed [9:0] leak_q  = {leaked[8], leaked} + {{2{zp[7]}}, zp};
  wire signed [7:0] leak_s  = (leak_q >  $signed(10'sd127)) ?  8'sd127
                            : (leak_q < -$signed(10'sd128)) ? -8'sd128
                            : leak_q[7:0];

  wire signed [7:0] cand = (sel == ACT_LEAKY) ? ((q_in >= zp) ? q_in : leak_s)
                                              : q_in;

  wire signed [7:0] lo = ((sel == ACT_RELU) || (sel == ACT_RELU6)) ? zp : clamp_lo;
  wire signed [7:0] hi = (sel == ACT_RELU) ? 8'sd127 : clamp_hi;

  // A host that programs clamp_lo > clamp_hi gets the low bound, which keeps the
  // function total instead of undefined.
  assign y = (cand < lo) ? lo : (cand > hi) ? hi : cand;

  wire _unused = &{1'b0, ACT_IDENTITY};

endmodule

`default_nettype wire
