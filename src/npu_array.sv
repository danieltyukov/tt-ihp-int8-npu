// SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
// SPDX-License-Identifier: Apache-2.0
//
// npu_array: ROWS x COLS weight-stationary systolic array.
//
// Row r holds the weights for input r, column c holds the weights for output
// channel c, so PE(r, c) is resident with W[r][c]. Activations enter at the
// west edge, one element per row per cycle, and shift east. Partial sums start
// at zero on the north edge and accumulate south, so the bottom of column c
// emits the complete dot product sum_r W[r][c] * x[r].
//
// Weights are loaded through a single byte-wide shift chain that snakes through
// every PE from the last element back to the first. The chain runs backwards so
// that a host sending W[0][0], W[0][1], ... W[ROWS-1][COLS-1] in plain
// row-major order ends up with each byte in the right PE after exactly
// ROWS*COLS shifts, with no address decoder anywhere.
//
// Timing (all cycle numbers relative to the sample entering row 0):
//   activation x[r] must be presented at row r in cycle r     (diagonal skew)
//   PE(r, c) latches it in cycle r + c
//   PE(r, c) adds its product in cycle r + c + 1
//   the column-c dot product is in the bottom psum register at
//     end of cycle ROWS + c
// so results leave the array one column per cycle, staggered by one cycle per
// column. docs/DESIGN.md derives this and the fill/drain latency.

`default_nettype none

module npu_array #(
    parameter int ROWS     = 4,
    parameter int COLS     = 2,
    parameter int PSUM_W   = 19,
    parameter int MUL_ARCH = 0,
    parameter int ADD_ARCH = 0
) (
    input  wire                       clk,
    input  wire                       rst_n,
    // Weight shift chain: one byte per cycle while w_en is high.
    input  wire                       w_en,
    input  wire  [7:0]                w_byte,
    // Activations, one byte per row, already skewed by the caller.
    input  wire  [ROWS*8-1:0]         act_row,
    // Bottom-of-column partial sums.
    output wire  [COLS*PSUM_W-1:0]    psum_col
);

  localparam int NPE = ROWS * COLS;

  initial begin
    if (ROWS < 1) $fatal(1, "npu_array: ROWS must be >= 1");
    if (COLS < 1) $fatal(1, "npu_array: COLS must be >= 1");
    if (PSUM_W < 16 + $clog2(ROWS + 1))
      $fatal(1, "npu_array: PSUM_W too narrow for ROWS rows of signed 8x8 products");
  end

  // Per-PE nets, indexed by pe = r * COLS + c.
  wire [NPE*8-1:0]      w_out;
  wire [NPE*8-1:0]      a_out;
  wire [NPE*PSUM_W-1:0] psum_out;

  generate
    for (genvar r = 0; r < ROWS; r++) begin : g_row
      for (genvar c = 0; c < COLS; c++) begin : g_col
        localparam int PE = r * COLS + c;

        wire [7:0]               w_src;
        wire [7:0]               a_src;
        wire signed [PSUM_W-1:0] psum_src;

        // Weight chain: the last PE takes the incoming byte, everyone else
        // takes the byte from the PE after it, so byte k lands in PE k.
        if (PE == NPE - 1) begin : g_w_head
          assign w_src = w_byte;
        end else begin : g_w_link
          assign w_src = w_out[(PE+1)*8 +: 8];
        end

        // Activation source: west neighbour, or the row input at column 0.
        if (c == 0) begin : g_a_edge
          assign a_src = act_row[r*8 +: 8];
        end else begin : g_a_link
          assign a_src = a_out[(PE-1)*8 +: 8];
        end

        // Partial-sum source: north neighbour, or zero at row 0.
        if (r == 0) begin : g_p_edge
          assign psum_src = {PSUM_W{1'b0}};
        end else begin : g_p_link
          assign psum_src = psum_out[(PE-COLS)*PSUM_W +: PSUM_W];
        end

        npu_pe #(
            .PSUM_W  (PSUM_W),
            .MUL_ARCH(MUL_ARCH),
            .ADD_ARCH(ADD_ARCH)
        ) u_pe (
            .clk     (clk),
            .rst_n   (rst_n),
            .w_en    (w_en),
            .w_in    (w_src),
            .w_out   (w_out[PE*8 +: 8]),
            .a_in    (a_src),
            .a_out   (a_out[PE*8 +: 8]),
            .psum_in (psum_src),
            .psum_out(psum_out[PE*PSUM_W +: PSUM_W])
        );
      end
    end

    for (genvar c = 0; c < COLS; c++) begin : g_drain
      assign psum_col[c*PSUM_W +: PSUM_W] =
          psum_out[((ROWS-1)*COLS + c)*PSUM_W +: PSUM_W];
    end
  endgenerate

  // The east-edge activation registers and the head of the weight chain have no
  // consumer, which is the whole point of a systolic edge; naming them here
  // keeps the lint clean.
  wire _unused = &{1'b0, w_out[0 +: 8], a_out};

endmodule

`default_nettype wire
