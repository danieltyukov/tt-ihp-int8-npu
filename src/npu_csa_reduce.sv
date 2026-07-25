// SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
// SPDX-License-Identifier: Apache-2.0
//
// npu_csa_reduce: Wallace-style carry-save reduction tree.
//
// Takes N rows of WIDTH bits and reduces them to exactly two rows whose sum,
// taken modulo 2**WIDTH, equals the sum of the input rows modulo 2**WIDTH.
// Every group of three rows is compressed by a row of full adders (a 3:2
// compressor); rows that do not fill a group of three are forwarded unchanged.
// The module then instantiates itself on the shorter row set until two rows
// remain, which gives the O(log N) depth of a Wallace tree.
//
// Carries leaving bit WIDTH-1 are dropped. That is exact, not an
// approximation: the caller only ever needs the sum modulo 2**WIDTH.

`default_nettype none

module npu_csa_reduce #(
    parameter int WIDTH = 16,
    parameter int N     = 4
) (
    input  wire [N*WIDTH-1:0] rows_i,
    output wire [2*WIDTH-1:0] rows_o
);

  initial begin
    if (N < 1) $fatal(1, "npu_csa_reduce: N must be >= 1");
  end

  generate
    if (N == 1) begin : g_one
      assign rows_o = {{WIDTH{1'b0}}, rows_i};

    end else if (N == 2) begin : g_two
      assign rows_o = rows_i;

    end else begin : g_reduce
      localparam int GROUPS = N / 3;               // full 3:2 compressor rows
      localparam int PASS   = N - 3 * GROUPS;      // 0..2 rows forwarded as-is
      localparam int NEXT_N = 2 * GROUPS + PASS;

      wire [NEXT_N*WIDTH-1:0] next_rows;

      for (genvar g = 0; g < GROUPS; g++) begin : g_grp
        wire [WIDTH-1:0] x = rows_i[(3*g+0)*WIDTH +: WIDTH];
        wire [WIDTH-1:0] y = rows_i[(3*g+1)*WIDTH +: WIDTH];
        wire [WIDTH-1:0] z = rows_i[(3*g+2)*WIDTH +: WIDTH];

        wire [WIDTH-1:0] s = x ^ y ^ z;                    // full-adder sums
        wire [WIDTH-1:0] c = (x & y) | (x & z) | (y & z);  // full-adder carries

        assign next_rows[(2*g+0)*WIDTH +: WIDTH] = s;
        // A carry generated at bit i belongs at bit i+1 of the next row.
        assign next_rows[(2*g+1)*WIDTH +: WIDTH] = {c[WIDTH-2:0], 1'b0};
      end

      for (genvar r = 0; r < PASS; r++) begin : g_pass
        assign next_rows[(2*GROUPS+r)*WIDTH +: WIDTH] =
            rows_i[(3*GROUPS+r)*WIDTH +: WIDTH];
      end

      npu_csa_reduce #(
          .WIDTH(WIDTH),
          .N    (NEXT_N)
      ) u_next (
          .rows_i(next_rows),
          .rows_o(rows_o)
      );
    end
  endgenerate

endmodule

`default_nettype wire
