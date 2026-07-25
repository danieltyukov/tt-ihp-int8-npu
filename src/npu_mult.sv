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
// Baugh-Wooley partial-product generation.
//
// Writing A = -a[m-1]*2^(m-1) + A' and B = -b[n-1]*2^(n-1) + B' and folding the
// two negative cross terms with -X = ~X - 2^(k) + 1 gives
//
//   A*B = sum_{i<m-1, j<n-1} a_i b_j 2^(i+j)
//       + sum_{j<n-1} ~(a_{m-1} b_j) 2^(m-1+j)
//       + sum_{i<m-1} ~(a_i b_{n-1}) 2^(n-1+i)
//       + a_{m-1} b_{n-1} 2^(m+n-2)
//       + 2^(m+n-1) + 2^(m-1) + 2^(n-1)          (mod 2^(m+n))
//
// which is B_W rows of AND/NAND terms plus one constant row.
// ---------------------------------------------------------------------------
module npu_pp_baugh_wooley #(
    parameter int A_W = 8,
    parameter int B_W = 8
) (
    input  wire [A_W-1:0]              a,
    input  wire [B_W-1:0]              b,
    output wire [(B_W+1)*(A_W+B_W)-1:0] rows_o
);

  localparam int W = A_W + B_W;

  initial begin
    if (A_W < 2 || B_W < 2) $fatal(1, "npu_pp_baugh_wooley: operands must be >= 2 bits");
  end

  generate
    // Rows 0 .. B_W-2: plain AND terms, with the top term inverted because it
    // involves a's sign bit.
    for (genvar j = 0; j < B_W - 1; j++) begin : g_row
      assign rows_o[j*W +: W] = {
        {(W - A_W - j) {1'b0}},
        ~(a[A_W-1] & b[j]),
        a[A_W-2:0] & {(A_W - 1) {b[j]}},
        {j{1'b0}}
      };
    end

    // Row B_W-1 involves b's sign bit, so all but its top term are inverted.
    assign rows_o[(B_W-1)*W +: W] = {
      1'b0,
      a[A_W-1] & b[B_W-1],
      ~(a[A_W-2:0] & {(A_W - 1) {b[B_W-1]}}),
      {(B_W - 1) {1'b0}}
    };

    // Constant correction row.
    // Note the additions: for a square multiplier 2^(A_W-1) and 2^(B_W-1)
    // land on the same bit and must carry, so these cannot be OR-ed.
    localparam logic [W-1:0] BW_CONST =
        (({{(W-1){1'b0}}, 1'b1}) << (W - 1)) +
        (({{(W-1){1'b0}}, 1'b1}) << (A_W - 1)) +
        (({{(W-1){1'b0}}, 1'b1}) << (B_W - 1));
    assign rows_o[B_W*W +: W] = BW_CONST;
  endgenerate

endmodule

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
        // instead of unpacked arrays so that Yosys keeps the drivers.
        wire [NROWS*W-1:0] chain_s;
        wire [NROWS*W-1:0] chain_c;

        assign chain_s[0 +: W] = rows[0 +: W];
        assign chain_c[0 +: W] = {W{1'b0}};

        for (genvar k = 1; k < NROWS; k++) begin : g_stage
          wire [W-1:0] x = chain_s[(k-1)*W +: W];
          wire [W-1:0] y = chain_c[(k-1)*W +: W];
          wire [W-1:0] z = rows[k*W +: W];
          wire [W-1:0] c = (x & y) | (x & z) | (y & z);
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
