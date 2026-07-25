// SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
// SPDX-License-Identifier: Apache-2.0
//
// Signed multiplier with a selectable architecture. Three styles are provided,
// all producing the exact two's-complement product a*b:
//
//   MUL_ARCH  partial products      reduction              depth
//   0         Baugh-Wooley (B_W+1)  linear carry-save row  O(B_W)
//   1         Baugh-Wooley (B_W+1)  Wallace tree           O(log B_W)
//   2         Booth radix-4         Wallace tree           O(log B_W)
//             (B_W/2+2 rows)
//
// Everything below works modulo 2**(A_W+B_W). The true product of two signed
// operands is representable in A_W+B_W bits, so carries out of the top bit can
// be discarded and sign handling collapses into which rows get complemented.

`default_nettype none

// ---------------------------------------------------------------------------
// Multiplier top level: pick a partial-product style, reduce, then add.
// ---------------------------------------------------------------------------
module npu_mult #(
    parameter int A_W      = 8,
    parameter int B_W      = 8,
    parameter int MUL_ARCH = 0,
    parameter int ADD_ARCH = 0
) (
    input  wire signed [A_W-1:0]       a,
    input  wire signed [B_W-1:0]       b,
    output wire signed [A_W+B_W-1:0]   p
);

  localparam int W = A_W + B_W;

  initial begin
    if (MUL_ARCH < 0 || MUL_ARCH > 2) $fatal(1, "npu_mult: MUL_ARCH must be 0..2");
  end

  wire [W-1:0] final_a;
  wire [W-1:0] final_b;

  generate
    if (MUL_ARCH == 2) begin : g_booth
      localparam int NROWS = ((B_W + 2) / 2) + 1;
      wire [NROWS*W-1:0] rows;

      npu_pp_booth4 #(
          .A_W(A_W),
          .B_W(B_W)
      ) u_pp (
          .a     (a),
          .b     (b),
          .rows_o(rows)
      );

      wire [2*W-1:0] two_rows;
      npu_csa_reduce #(
          .WIDTH(W),
          .N    (NROWS)
      ) u_tree (
          .rows_i(rows),
          .rows_o(two_rows)
      );
      assign final_a = two_rows[0 +: W];
      assign final_b = two_rows[W +: W];

    end else begin : g_bw
      localparam int NROWS = B_W + 1;
      wire [NROWS*W-1:0] rows;

      npu_pp_baugh_wooley #(
          .A_W(A_W),
          .B_W(B_W)
      ) u_pp (
          .a     (a),
          .b     (b),
          .rows_o(rows)
      );

      if (MUL_ARCH == 1) begin : g_wallace
        wire [2*W-1:0] two_rows;
        npu_csa_reduce #(
            .WIDTH(W),
            .N    (NROWS)
        ) u_tree (
            .rows_i(rows),
            .rows_o(two_rows)
        );
        assign final_a = two_rows[0 +: W];
        assign final_b = two_rows[W +: W];

      end else begin : g_array
        // Classic array multiplier: one carry-save row per partial product,
        // accumulated in sequence rather than in a tree. Flat vectors are used
        // instead of unpacked arrays so that Yosys keeps the drivers; stage k
        // reads only stage k-1, which Verilator flags as a loop regardless.
        /* verilator lint_off UNOPTFLAT */
        wire [NROWS*W-1:0] chain_s;
        wire [NROWS*W-1:0] chain_c;
        /* verilator lint_on UNOPTFLAT */

        assign chain_s[0 +: W] = rows[0 +: W];
        assign chain_c[0 +: W] = {W{1'b0}};

        for (genvar k = 1; k < NROWS; k++) begin : g_stage
          wire [W-1:0] x = chain_s[(k-1)*W +: W];
          wire [W-1:0] y = chain_c[(k-1)*W +: W];
          wire [W-1:0] z = rows[k*W +: W];
          // Bit W-1 of the carry row would land outside the product width and
          // is dropped: exact, because only the sum modulo 2**W is needed.
          /* verilator lint_off UNUSEDSIGNAL */
          /* verilator lint_off UNOPTFLAT */
          wire [W-1:0] c = (x & y) | (x & z) | (y & z);
          /* verilator lint_on UNOPTFLAT */
          /* verilator lint_on UNUSEDSIGNAL */
          assign chain_s[k*W +: W] = x ^ y ^ z;
          assign chain_c[k*W +: W] = {c[W-2:0], 1'b0};
        end

        assign final_a = chain_s[(NROWS-1)*W +: W];
        assign final_b = chain_c[(NROWS-1)*W +: W];
      end
    end
  endgenerate

  // Final carry-propagate addition of the two remaining rows.
  wire unused_cout;
  npu_adder #(
      .WIDTH(W),
      .ARCH (ADD_ARCH)
  ) u_cpa (
      .a   (final_a),
      .b   (final_b),
      .cin (1'b0),
      .sum (p),
      .cout(unused_cout)
  );

  wire _unused = &{unused_cout, 1'b0};

endmodule

`default_nettype wire
