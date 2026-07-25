// SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
// SPDX-License-Identifier: Apache-2.0
//
// npu_adder: parameterized carry-propagate adder with a selectable
// prefix-network architecture. All five architectures compute exactly the same
// function, sum = a + b + cin, and differ only in the structure of the carry
// network, so they trade area against logic depth.
//
//   ARCH  network        carry-network cells        logic depth
//   0     ripple carry   W - 1                     O(W)
//   1     Brent-Kung     ~2W - log2(W) - 2         2*log2(W) - 1
//   2     Kogge-Stone    ~W*log2(W) - W + 1        log2(W)
//   3     Sklansky       ~(W/2)*log2(W)            log2(W)
//   4     Han-Carlson    ~(W/2)*log2(W) + W/2      log2(W) + 1
//
// The four parallel-prefix variants share one skeleton: a pre-processing stage
// turns (a, b, cin) into per-bit generate/propagate pairs, a prefix network
// computes the group generate G[i] ("a carry leaves bit i"), and a
// post-processing stage forms sum[i] = p[i] ^ G[i-1].
//
// The prefix operator
//   (G, P) o (G', P') = (G | (P & G'), P & P')
// is associative, so every valid network yields bit-identical results. The
// equivalence test in test/test_arith.py checks that on random vectors.
//
// The prefix network is carried in two flat vectors indexed as
// [stage*WIDTH + bit] rather than an unpacked array of wires, because Yosys
// turns arrays of wires into memories and then loses the continuous drivers.

`default_nettype none

module npu_adder #(
    parameter int WIDTH = 16,
    parameter int ARCH  = 0
) (
    input  wire [WIDTH-1:0] a,
    input  wire [WIDTH-1:0] b,
    input  wire             cin,
    output wire [WIDTH-1:0] sum,
    output wire             cout
);

  // Number of prefix levels needed to span WIDTH bits.
  localparam int LEVELS = (WIDTH <= 1) ? 1 : $clog2(WIDTH);
  // Brent-Kung needs a forward and a backward sweep; the others need LEVELS,
  // plus one for the Han-Carlson recombination stage.
  localparam int STAGES = 2 * LEVELS + 1;
  localparam int NNODE  = (STAGES + 1) * WIDTH;

  initial begin
    if (WIDTH < 1) $fatal(1, "npu_adder: WIDTH must be >= 1");
    if (ARCH < 0 || ARCH > 4) $fatal(1, "npu_adder: ARCH must be 0..4");
  end

  // ---------------------------------------------------------------------------
  // Pre-processing. The incoming carry is folded into bit 0's generate term so
  // the prefix network needs no separate carry-in input.
  // ---------------------------------------------------------------------------
  wire [WIDTH-1:0] p_pre = a ^ b;
  wire [WIDTH-1:0] g_raw = a & b;
  wire [WIDTH-1:0] g_pre = {g_raw[WIDTH-1:1], g_raw[0] | (p_pre[0] & cin)};

  wire [NNODE-1:0] gnet;  // group generate, [stage*WIDTH + bit]
  wire [NNODE-1:0] pnet;  // group propagate

  assign gnet[0 +: WIDTH] = g_pre;
  assign pnet[0 +: WIDTH] = p_pre;

  generate
    if (ARCH == 0) begin : g_ripple
      // Serial carry chain: G[i] = g[i] | (p[i] & G[i-1]).
      wire [WIDTH:0] carry;
      assign carry[0] = cin;
      for (genvar i = 0; i < WIDTH; i++) begin : g_bit
        assign carry[i+1] = g_raw[i] | (p_pre[i] & carry[i]);
      end
      for (genvar s = 1; s <= STAGES; s++) begin : g_fill
        assign gnet[s*WIDTH +: WIDTH] = carry[WIDTH:1];
        assign pnet[s*WIDTH +: WIDTH] = {WIDTH{1'b0}};
      end

    end else if (ARCH == 1) begin : g_brent_kung
      // Forward sweep: level s completes the prefix at every index where
      // (i+1) is a multiple of 2^s. Backward sweep fills in the gaps.
      for (genvar s = 1; s <= LEVELS; s++) begin : g_fwd
        localparam int SPAN = 1 << s;
        localparam int HALF = SPAN >> 1;
        for (genvar i = 0; i < WIDTH; i++) begin : g_bit
          if (((i + 1) % SPAN == 0) && (i >= HALF)) begin : g_comb
            assign gnet[s*WIDTH+i] = gnet[(s-1)*WIDTH+i]
                                   | (pnet[(s-1)*WIDTH+i] & gnet[(s-1)*WIDTH+i-HALF]);
            assign pnet[s*WIDTH+i] = pnet[(s-1)*WIDTH+i] & pnet[(s-1)*WIDTH+i-HALF];
          end else begin : g_keep
            assign gnet[s*WIDTH+i] = gnet[(s-1)*WIDTH+i];
            assign pnet[s*WIDTH+i] = pnet[(s-1)*WIDTH+i];
          end
        end
      end
      for (genvar s = 1; s < LEVELS; s++) begin : g_bwd
        localparam int STG  = LEVELS + s;
        localparam int SPAN = 1 << (LEVELS - s);
        localparam int HALF = SPAN >> 1;
        for (genvar i = 0; i < WIDTH; i++) begin : g_bit
          if (((i + 1) % SPAN == HALF) && (i >= HALF)) begin : g_comb
            assign gnet[STG*WIDTH+i] = gnet[(STG-1)*WIDTH+i]
                                     | (pnet[(STG-1)*WIDTH+i] & gnet[(STG-1)*WIDTH+i-HALF]);
            assign pnet[STG*WIDTH+i] = pnet[(STG-1)*WIDTH+i] & pnet[(STG-1)*WIDTH+i-HALF];
          end else begin : g_keep
            assign gnet[STG*WIDTH+i] = gnet[(STG-1)*WIDTH+i];
            assign pnet[STG*WIDTH+i] = pnet[(STG-1)*WIDTH+i];
          end
        end
      end
      for (genvar s = 2 * LEVELS; s <= STAGES; s++) begin : g_fill
        assign gnet[s*WIDTH +: WIDTH] = gnet[(s-1)*WIDTH +: WIDTH];
        assign pnet[s*WIDTH +: WIDTH] = pnet[(s-1)*WIDTH +: WIDTH];
      end

    end else if (ARCH == 2) begin : g_kogge_stone
      // Every node combines with the node 2^(s-1) positions to its right.
      for (genvar s = 1; s <= LEVELS; s++) begin : g_lvl
        localparam int DIST = 1 << (s - 1);
        for (genvar i = 0; i < WIDTH; i++) begin : g_bit
          if (i >= DIST) begin : g_comb
            assign gnet[s*WIDTH+i] = gnet[(s-1)*WIDTH+i]
                                   | (pnet[(s-1)*WIDTH+i] & gnet[(s-1)*WIDTH+i-DIST]);
            assign pnet[s*WIDTH+i] = pnet[(s-1)*WIDTH+i] & pnet[(s-1)*WIDTH+i-DIST];
          end else begin : g_keep
            assign gnet[s*WIDTH+i] = gnet[(s-1)*WIDTH+i];
            assign pnet[s*WIDTH+i] = pnet[(s-1)*WIDTH+i];
          end
        end
      end
      for (genvar s = LEVELS + 1; s <= STAGES; s++) begin : g_fill
        assign gnet[s*WIDTH +: WIDTH] = gnet[(s-1)*WIDTH +: WIDTH];
        assign pnet[s*WIDTH +: WIDTH] = pnet[(s-1)*WIDTH +: WIDTH];
      end

    end else if (ARCH == 3) begin : g_sklansky
      // The upper half of every 2^s block combines with the block midpoint:
      // minimum depth, at the cost of high fanout on the midpoint nodes.
      for (genvar s = 1; s <= LEVELS; s++) begin : g_lvl
        localparam int BLK  = 1 << s;
        localparam int HALF = BLK >> 1;
        for (genvar i = 0; i < WIDTH; i++) begin : g_bit
          if ((i % BLK) >= HALF) begin : g_comb
            localparam int SRC = (i / BLK) * BLK + HALF - 1;
            assign gnet[s*WIDTH+i] = gnet[(s-1)*WIDTH+i]
                                   | (pnet[(s-1)*WIDTH+i] & gnet[(s-1)*WIDTH+SRC]);
            assign pnet[s*WIDTH+i] = pnet[(s-1)*WIDTH+i] & pnet[(s-1)*WIDTH+SRC];
          end else begin : g_keep
            assign gnet[s*WIDTH+i] = gnet[(s-1)*WIDTH+i];
            assign pnet[s*WIDTH+i] = pnet[(s-1)*WIDTH+i];
          end
        end
      end
      for (genvar s = LEVELS + 1; s <= STAGES; s++) begin : g_fill
        assign gnet[s*WIDTH +: WIDTH] = gnet[(s-1)*WIDTH +: WIDTH];
        assign pnet[s*WIDTH +: WIDTH] = pnet[(s-1)*WIDTH +: WIDTH];
      end

    end else begin : g_han_carlson
      // Stage 1 pairs each odd bit with its even neighbour, stages 2..LEVELS
      // run Kogge-Stone over the odd positions only, and a final stage
      // recombines the even positions: half the wiring of Kogge-Stone for one
      // extra level of depth.
      for (genvar i = 0; i < WIDTH; i++) begin : g_pair
        if (i % 2 == 1) begin : g_comb
          assign gnet[WIDTH+i] = gnet[i] | (pnet[i] & gnet[i-1]);
          assign pnet[WIDTH+i] = pnet[i] & pnet[i-1];
        end else begin : g_keep
          assign gnet[WIDTH+i] = gnet[i];
          assign pnet[WIDTH+i] = pnet[i];
        end
      end
      for (genvar s = 2; s <= LEVELS; s++) begin : g_lvl
        localparam int DIST = 1 << (s - 1);
        for (genvar i = 0; i < WIDTH; i++) begin : g_bit
          if ((i % 2 == 1) && (i >= DIST)) begin : g_comb
            assign gnet[s*WIDTH+i] = gnet[(s-1)*WIDTH+i]
                                   | (pnet[(s-1)*WIDTH+i] & gnet[(s-1)*WIDTH+i-DIST]);
            assign pnet[s*WIDTH+i] = pnet[(s-1)*WIDTH+i] & pnet[(s-1)*WIDTH+i-DIST];
          end else begin : g_keep
            assign gnet[s*WIDTH+i] = gnet[(s-1)*WIDTH+i];
            assign pnet[s*WIDTH+i] = pnet[(s-1)*WIDTH+i];
          end
        end
      end
      for (genvar i = 0; i < WIDTH; i++) begin : g_merge
        if ((i % 2 == 0) && (i > 0)) begin : g_comb
          assign gnet[(LEVELS+1)*WIDTH+i] = gnet[LEVELS*WIDTH+i]
                                          | (pnet[LEVELS*WIDTH+i] & gnet[LEVELS*WIDTH+i-1]);
          assign pnet[(LEVELS+1)*WIDTH+i] = pnet[LEVELS*WIDTH+i] & pnet[LEVELS*WIDTH+i-1];
        end else begin : g_keep
          assign gnet[(LEVELS+1)*WIDTH+i] = gnet[LEVELS*WIDTH+i];
          assign pnet[(LEVELS+1)*WIDTH+i] = pnet[LEVELS*WIDTH+i];
        end
      end
      for (genvar s = LEVELS + 2; s <= STAGES; s++) begin : g_fill
        assign gnet[s*WIDTH +: WIDTH] = gnet[(s-1)*WIDTH +: WIDTH];
        assign pnet[s*WIDTH +: WIDTH] = pnet[(s-1)*WIDTH +: WIDTH];
      end
    end
  endgenerate

  // ---------------------------------------------------------------------------
  // Post-processing: the carry into bit i is the group generate of bits i-1..0.
  // ---------------------------------------------------------------------------
  wire [WIDTH-1:0] g_out = gnet[STAGES*WIDTH +: WIDTH];
  wire [WIDTH-1:0] carry_in_bit;

  generate
    if (WIDTH == 1) begin : g_carry_w1
      assign carry_in_bit = cin;
    end else begin : g_carry
      assign carry_in_bit = {g_out[WIDTH-2:0], cin};
    end
  endgenerate

  assign sum  = p_pre ^ carry_in_bit;
  assign cout = g_out[WIDTH-1];

  // The unused propagate outputs of the last stage are intentional.
  wire _unused_pnet = &{1'b0, pnet[STAGES*WIDTH +: WIDTH]};

endmodule

`default_nettype wire
