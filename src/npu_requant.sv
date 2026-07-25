// SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
// SPDX-License-Identifier: Apache-2.0
//
// npu_requant: integer-only affine requantization of one accumulator value.
//
// Computes, exactly and with no intermediate truncation,
//
//   y = clamp_int8( round_half_away_from_zero( acc * M / 2^(M_W + shift) )
//                   + zero_point )
//
// where acc already contains the bias (the accumulator bank is initialised with
// the bias, exactly as a TFLite kernel does) and M is an unsigned M_W-bit
// fixed-point multiplier. The rounding rule is the one gemmlowp and TFLite use
// in RoundingDivideByPOT: ties are broken away from zero. See docs/DESIGN.md
// for the derivation and for how M and shift relate to the float scales.
//
// Area, not throughput, sets the structure here. A parallel 24x16 multiplier
// costs about 20000 um2 in sg13g2, four times a full processing element, so the
// multiply is done serially: radix-4 Booth, two multiplier bits per cycle,
// sharing one adder with the final zero-point addition. The rounding shift
// reuses the same register, four bits per cycle plus a one-bit fine step, and
// tracks the round and sticky bits as they leave the bottom of the register, so
// no barrel shifter is needed either.
//
// Cycle cost per output element, with n = M_W + shift + 1:
//   1 (setup) + NDIG (Booth steps) + floor(n/4) + n mod 4 (shift)
//   + 1 (round and zero point) + 1 (done)
// which is 17 cycles at the default M_W of 16 with shift 0, and 25 at the
// maximum shift. The systolic array is never stalled by this: it writes raw
// sums into the accumulator bank and requantization runs afterwards.

`default_nettype none

module npu_requant #(
    parameter int T_W      = 24,
    parameter int M_W      = 16,
    parameter int SH_W     = 5,
    parameter int ADD_ARCH = 0
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  start,
    input  wire signed [T_W-1:0] acc_in,
    input  wire        [M_W-1:0] m_in,
    input  wire        [SH_W-1:0] shift_in,
    input  wire signed [7:0]     zp_in,
    output wire                  busy,
    output logic                 done,
    output logic signed [7:0]    q_out,
    output logic                 sat_out
);

  // Radix-4 Booth digits needed to cover M as an (M_W+1)-bit signed value.
  localparam int NDIG    = (M_W + 2) / 2;  // ceil((M_W+1)/2)
  localparam int A_W     = T_W + 2;        // room for 2*acc plus sign growth
  // Digit NDIG-1 inspects multiplier bit 2*NDIG-1, which sits at field
  // position 2*NDIG once the b[-1] pad is prepended, so the field, and hence
  // Q, needs 2*NDIG+1 bits. The loop only shifts 2*NDIG of them out, which
  // leaves the product in the register scaled by two; the rounding shift below
  // absorbs that with one extra bit of shift.
  localparam int Q_W     = 2 * NDIG + 1;
  localparam int REG_W   = A_W + Q_W;
  localparam int MAXSH   = M_W + (1 << SH_W);
  localparam int SHCNT_W = $clog2(MAXSH + 1);
  localparam int DIGCNT_W = $clog2(NDIG + 1);

  initial begin
    if (T_W < 10) $fatal(1, "npu_requant: T_W must be >= 10");
    if (M_W < 4)  $fatal(1, "npu_requant: M_W must be >= 4");
    if (Q_W < M_W + 2) $fatal(1, "npu_requant: Q_W too narrow for the multiplier");
    if (2 * NDIG < M_W + 1) $fatal(1, "npu_requant: too few Booth digits for M_W");
  end

  typedef enum logic [1:0] {S_IDLE, S_MUL, S_SHIFT, S_FIN} state_e;
  state_e state;

  logic [REG_W-1:0]   pr;      // {A, Q}
  logic [DIGCNT_W-1:0] digcnt;
  logic [SHCNT_W-1:0]  shcnt;
  logic                rbit;   // bit that left the register last
  logic                sbit;   // sticky OR of everything below rbit

  wire [A_W-1:0] a_part = pr[REG_W-1 -: A_W];

  // ---------------------------------------------------------------------------
  // Radix-4 Booth recoding of the two multiplier bits at the bottom of Q.
  // ---------------------------------------------------------------------------
  wire b_m1 = pr[0];
  wire b_0  = pr[1];
  wire b_1  = pr[2];

  wire boo_single = b_0 ^ b_m1;
  wire boo_double = (b_1 & ~b_0 & ~b_m1) | (~b_1 & b_0 & b_m1);
  wire boo_neg    = b_1;

  wire signed [A_W-1:0] t_ext = {{(A_W - T_W) {acc_in[T_W-1]}}, acc_in};
  wire        [A_W-1:0] boo_mag = boo_double ? {t_ext[A_W-2:0], 1'b0}
                                : boo_single ? t_ext
                                :              {A_W{1'b0}};
  wire        [A_W-1:0] boo_addend = boo_mag ^ {A_W{boo_neg}};

  // ---------------------------------------------------------------------------
  // One shared adder: Booth accumulation during S_MUL, then the zero-point add
  // (with the rounding increment folded into its carry-in) during S_FIN.
  // ---------------------------------------------------------------------------
  wire        prod_neg  = pr[REG_W-1];
  wire        round_up  = prod_neg ? (rbit & sbit) : rbit;

  wire [A_W-1:0] add_a = (state == S_FIN) ? pr[A_W-1:0] : a_part;
  wire [A_W-1:0] add_b = (state == S_FIN) ? {{(A_W - 8) {zp_in[7]}}, zp_in}
                                          : boo_addend;
  wire           add_ci = (state == S_FIN) ? round_up : boo_neg;

  wire [A_W-1:0] add_s;
  wire           add_co;

  npu_adder #(
      .WIDTH(A_W),
      .ARCH (ADD_ARCH)
  ) u_add (
      .a   (add_a),
      .b   (add_b),
      .cin (add_ci),
      .sum (add_s),
      .cout(add_co)
  );

  // Saturating narrow of the final sum to signed 8 bits.
  wire signed [A_W-1:0] y_full  = add_s;
  wire                  y_hi    = (y_full > $signed({{(A_W - 8) {1'b0}}, 8'sd127}));
  wire                  y_lo    = (y_full < $signed({{(A_W - 8) {1'b1}}, 8'sd128}));
  wire signed [7:0]     y_clamp = y_hi ? 8'sd127 : y_lo ? -8'sd128 : y_full[7:0];

  // ---------------------------------------------------------------------------
  // Sequencing.
  // ---------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state   <= S_IDLE;
      pr      <= '0;
      digcnt  <= '0;
      shcnt   <= '0;
      rbit    <= 1'b0;
      sbit    <= 1'b0;
      done    <= 1'b0;
      q_out   <= 8'sd0;
      sat_out <= 1'b0;
    end else begin
      done <= 1'b0;
      case (state)
        S_IDLE: begin
          if (start) begin
            // A = 0, Q = {pad, M, 0}: the extra low zero is Booth's b[-1].
            pr     <= {{A_W{1'b0}}, {(Q_W - M_W - 1) {1'b0}}, m_in, 1'b0};
            digcnt <= DIGCNT_W'(NDIG);
            // M_W + shift, plus one for the factor of two described above.
            shcnt  <= SHCNT_W'(M_W + 1) + SHCNT_W'(shift_in);
            rbit   <= 1'b0;
            sbit   <= 1'b0;
            state  <= S_MUL;
          end
        end

        S_MUL: begin
          // Accumulate one Booth digit, then shift {A, Q} right by two.
          pr     <= {{2{add_s[A_W-1]}}, add_s, pr[Q_W-1:2]};
          digcnt <= digcnt - 1'b1;
          if (digcnt == DIGCNT_W'(1)) state <= (shcnt == '0) ? S_FIN : S_SHIFT;
        end

        S_SHIFT: begin
          // Rounding shift, four bits per cycle then one at a time, carrying
          // the round bit and the sticky OR out of the bottom of the register.
          if (shcnt >= SHCNT_W'(4)) begin
            pr    <= {{4{pr[REG_W-1]}}, pr[REG_W-1:4]};
            rbit  <= pr[3];
            sbit  <= sbit | rbit | pr[2] | pr[1] | pr[0];
            shcnt <= shcnt - SHCNT_W'(4);
            if (shcnt == SHCNT_W'(4)) state <= S_FIN;
          end else begin
            pr    <= {pr[REG_W-1], pr[REG_W-1:1]};
            rbit  <= pr[0];
            sbit  <= sbit | rbit;
            shcnt <= shcnt - SHCNT_W'(1);
            if (shcnt == SHCNT_W'(1)) state <= S_FIN;
          end
        end

        default: begin  // S_FIN
          q_out   <= y_clamp;
          sat_out <= y_hi | y_lo;
          done    <= 1'b1;
          state   <= S_IDLE;
        end
      endcase
    end
  end

  assign busy = (state != S_IDLE);

  wire _unused = &{1'b0, add_co};

endmodule

`default_nettype wire
