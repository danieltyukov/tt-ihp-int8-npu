// SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
// SPDX-License-Identifier: Apache-2.0
//
// ---------------------------------------------------------------------------
// Radix-4 modified Booth partial-product generation.
//
// Each overlapping triple (b[2k+1], b[2k], b[2k-1]) selects a digit in
// {-2,-1,0,1,2}. Negative digits are produced by complementing the magnitude
// and injecting a +1 at bit 2k; because those injection points are all at even
// positions they pack into a single extra row instead of one row each.
// ---------------------------------------------------------------------------
module npu_pp_booth4 #(
    parameter int A_W = 8,
    parameter int B_W = 8
) (
    input  wire [A_W-1:0]                    a,
    input  wire [B_W-1:0]                    b,
    output wire [(((B_W+2)/2)+1)*(A_W+B_W)-1:0] rows_o
);

  localparam int W    = A_W + B_W;
  localparam int NDIG = (B_W + 2) / 2;

  initial begin
    if (A_W < 2 || B_W < 2) $fatal(1, "npu_pp_booth4: operands must be >= 2 bits");
  end

  // b sign-extended by one bit above the top and by one zero below bit 0, so
  // that every triple can be indexed uniformly.
  wire [NDIG*2:0] bx = {{(NDIG*2 + 1 - B_W - 1) {b[B_W-1]}}, b, 1'b0};

  wire [W-1:0] a_sext = {{(B_W) {a[A_W-1]}}, a};

  wire [NDIG-1:0] neg;

  generate
    for (genvar k = 0; k < NDIG; k++) begin : g_dig
      wire t0 = bx[2*k];      // b[2k-1]
      wire t1 = bx[2*k+1];    // b[2k]
      wire t2 = bx[2*k+2];    // b[2k+1]

      wire single = t1 ^ t0;
      wire dbl    = (t2 & ~t1 & ~t0) | (~t2 & t1 & t0);

      assign neg[k] = t2;

      wire [W-1:0] mag = dbl    ? {a_sext[W-2:0], 1'b0}
                       : single ? a_sext
                       :          {W{1'b0}};

      // Complement first, then align: -(mag) * 2^(2k) = (~mag + 1) * 2^(2k).
      wire [W-1:0] signed_mag = mag ^ {W{neg[k]}};
      assign rows_o[k*W +: W] = signed_mag << (2 * k);
    end

    // The +1 injections, one per negated digit, all land on distinct even bits.
    wire [W-1:0] corr;
    for (genvar i = 0; i < W; i++) begin : g_corr
      if ((i % 2 == 0) && (i / 2 < NDIG)) begin : g_bit
        assign corr[i] = neg[i/2];
      end else begin : g_zero
        assign corr[i] = 1'b0;
      end
    end
    assign rows_o[NDIG*W +: W] = corr;
  endgenerate

endmodule
`default_nettype wire
