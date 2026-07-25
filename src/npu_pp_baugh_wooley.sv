// SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
// SPDX-License-Identifier: Apache-2.0
//
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
`default_nettype wire
