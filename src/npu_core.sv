// SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
// SPDX-License-Identifier: Apache-2.0
//
// npu_core: storage, sequencer and result path around the systolic array.
//
// One RUN computes, for s = 0..S_COUNT-1 and c = 0..COLS-1,
//
//   acc[c][s] = saturate( (first_pass ? bias[c] : acc[c][s])
//                         + sum_r W[r][c] * X[s][r] )
//
// and, when the requantize flag is set, also
//
//   result[s][c] = activation( requant(acc[c][s], M[c], shift[c], zp) )
//
// Phases
//   ARRAY  S_COUNT + ROWS + COLS cycles. Activations are read out of the buffer
//          with a diagonal skew (row r sees sample cyc-r) so the array is fed
//          one full activation vector per cycle. Every PE performs one MAC per
//          cycle for the whole streaming window; the ROWS+1 leading and COLS-1
//          trailing cycles are the pipeline fill and drain.
//   REQ    S_COUNT*COLS requantizations on the shared serial unit, 17 to 25
//          cycles each.
//
// Multi-pass reduction: a layer with K > ROWS inputs is computed as
// ceil(K/ROWS) runs. The first uses RUN arg 0 (accumulator starts from the
// bias), the middle ones RUN arg 1 (accumulate), and the last RUN arg 3
// (accumulate and requantize), with a fresh weight tile loaded in between.
//
// Everything is registers: there is no SRAM on a Tiny Tapeout tile, so the
// buffer sizes are what set the area. docs/DESIGN.md has the breakdown.

`default_nettype none

module npu_core #(
    parameter int ROWS     = 4,
    parameter int COLS     = 2,
    parameter int S_MAX    = 4,
    parameter int ACC_W    = 24,
    parameter int M_W      = 16,
    parameter int SH_W     = 5,
    parameter int MUL_ARCH = 0,
    parameter int ADD_ARCH = 0
) (
    input  wire       clk,
    input  wire       rst_n,
    // Host bus.
    input  wire [7:0] data_in,
    input  wire       wr,
    input  wire       is_cmd,
    input  wire       rd,
    output wire [7:0] data_out,
    // Status.
    output wire       busy,
    output wire       done,
    output wire       err,
    output wire       sat,
    output wire       ovf
);

  // ---------------------------------------------------------------------------
  // Derived geometry.
  // ---------------------------------------------------------------------------
  localparam int PSUM_W     = 16 + $clog2(ROWS + 1);
  localparam int S_W        = (S_MAX <= 1) ? 1 : $clog2(S_MAX);
  localparam int C_W        = (COLS  <= 1) ? 1 : $clog2(COLS);
  localparam int SKEW       = ROWS + COLS;
  localparam int CYC_W      = $clog2(S_MAX + ROWS + COLS + 2);
  localparam int W_LEN      = ROWS * COLS;
  localparam int ACT_LEN    = S_MAX * ROWS;
  localparam int BB         = (ACC_W + 7) / 8;         // bias bytes per channel
  localparam int BIAS_LEN   = COLS * BB;
  localparam int MB         = (M_W + 7) / 8;           // multiplier bytes
  localparam int QB         = MB + 1;                  // plus the shift byte
  localparam int QUANT_LEN  = COLS * QB;
  localparam int RES_LEN    = S_MAX * COLS;
  localparam int RIDX_W     = (RES_LEN <= 1) ? 1 : $clog2(RES_LEN);
  localparam int ENT_W      = S_W + C_W;
  localparam int RPTR_W     = ENT_W + 2;               // 4 readback bytes/entry
  localparam int VERSION    = 1;

  initial begin
    if (ROWS < 1 || COLS < 1) $fatal(1, "npu_core: ROWS and COLS must be >= 1");
    if (S_MAX < 1) $fatal(1, "npu_core: S_MAX must be >= 1");
    if (ACC_W < PSUM_W)
      $fatal(1, "npu_core: ACC_W (%0d) must be >= PSUM_W (%0d)", ACC_W, PSUM_W);
    if (ACC_W > 32) $fatal(1, "npu_core: ACC_W > 32 exceeds the readback format");
    if (M_W < 4 || M_W > 24) $fatal(1, "npu_core: M_W must be 4..24");
    if (SH_W < 1 || SH_W > 6) $fatal(1, "npu_core: SH_W must be 1..6");
    if (RPTR_W > 8) $fatal(1, "npu_core: readback pointer wider than one byte");
    if (ROWS > 15 || COLS > 15 || S_MAX > 15)
      $fatal(1, "npu_core: geometry must fit the 4-bit identity block fields");
  end

  localparam logic [ACC_W-1:0] ACC_MAX = {1'b0, {(ACC_W - 1) {1'b1}}};
  localparam logic [ACC_W-1:0] ACC_MIN = {1'b1, {(ACC_W - 1) {1'b0}}};

  // ---------------------------------------------------------------------------
  // Host interface.
  // ---------------------------------------------------------------------------
  wire [7:0]        hb;
  wire              wr_w, wr_act, wr_bias, wr_quant, wr_post, wr_cfg;
  wire [1:0]        post_idx;
  wire              run_pulse, run_acc_arg, run_req_arg, clr_flags, soft_rst;
  wire [1:0]        rd_src;
  wire [RPTR_W-1:0] rd_ptr;
  wire              err_set;
  wire [1:0]        err_code_new;

  npu_host_if #(
      .W_LEN    (W_LEN),
      .ACT_LEN  (ACT_LEN),
      .BIAS_LEN (BIAS_LEN),
      .QUANT_LEN(QUANT_LEN),
      .RPTR_W   (RPTR_W)
  ) u_host (
      .clk      (clk),
      .rst_n    (rst_n),
      .data_in  (data_in),
      .wr       (wr),
      .is_cmd   (is_cmd),
      .rd       (rd),
      .core_busy(busy),
      .byte_out (hb),
      .wr_w     (wr_w),
      .wr_act   (wr_act),
      .wr_bias  (wr_bias),
      .wr_quant (wr_quant),
      .wr_post  (wr_post),
      .wr_cfg   (wr_cfg),
      .post_idx (post_idx),
      .run      (run_pulse),
      .run_acc  (run_acc_arg),
      .run_req  (run_req_arg),
      .clr_flags(clr_flags),
      .soft_rst (soft_rst),
      .rd_src   (rd_src),
      .rd_ptr   (rd_ptr),
      .err_set  (err_set),
      .err_code (err_code_new)
  );

  // ---------------------------------------------------------------------------
  // Byte-stream storage. Each bulk load is a plain shift chain: a new byte
  // enters at the top index and everything moves down, so after a full-length
  // load byte k sits at index k with no address decoder anywhere.
  // ---------------------------------------------------------------------------
  logic [ACT_LEN*8-1:0]   act_buf;
  logic [BIAS_LEN*8-1:0]  bias_buf;
  logic [QUANT_LEN*8-1:0] quant_buf;

  logic signed [7:0] zp_reg;
  logic signed [7:0] clamp_lo_reg;
  logic signed [7:0] clamp_hi_reg;
  logic [1:0]        act_sel_reg;
  logic [2:0]        leaky_k_reg;
  logic [S_W-1:0]    s_cnt_m1;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      act_buf      <= '0;
      bias_buf     <= '0;
      quant_buf    <= '0;
      zp_reg       <= 8'sd0;
      clamp_lo_reg <= -8'sd128;
      clamp_hi_reg <= 8'sd127;
      act_sel_reg  <= 2'd0;
      leaky_k_reg  <= 3'd0;
      s_cnt_m1     <= '0;
    end else begin
      if (wr_act)   act_buf   <= {hb, act_buf[ACT_LEN*8-1:8]};
      if (wr_bias)  bias_buf  <= {hb, bias_buf[BIAS_LEN*8-1:8]};
      if (wr_quant) quant_buf <= {hb, quant_buf[QUANT_LEN*8-1:8]};
      // An out-of-range sample count is clamped rather than wrapped, which
      // keeps every buffer index provably inside its array.
      if (wr_cfg)   s_cnt_m1  <= (hb[S_W-1:0] > S_W'(S_MAX - 1)) ? S_W'(S_MAX - 1)
                                                                 : hb[S_W-1:0];
      if (wr_post) begin
        case (post_idx)
          2'd0: zp_reg       <= $signed(hb);
          2'd1: clamp_lo_reg <= $signed(hb);
          2'd2: clamp_hi_reg <= $signed(hb);
          default: begin
            act_sel_reg <= hb[1:0];
            leaky_k_reg <= hb[4:2];
          end
        endcase
      end
    end
  end

  // ---------------------------------------------------------------------------
  // Sequencer state, declared early because the activation feed depends on it.
  // ---------------------------------------------------------------------------
  typedef enum logic [1:0] {ST_IDLE, ST_ARRAY, ST_REQ, ST_DONE} state_e;
  state_e state;

  logic [CYC_W-1:0] cyc;
  logic             acc_first;
  logic             do_req;
  logic [S_W-1:0]   req_s;
  logic [C_W-1:0]   req_c;
  logic             req_start;

  wire [CYC_W-1:0] s_top = {{(CYC_W - S_W) {1'b0}}, s_cnt_m1};

  // ---------------------------------------------------------------------------
  // Skewed activation feed: row r must see the sample that entered row 0 r
  // cycles ago, which is exactly what makes the wavefront line up.
  // ---------------------------------------------------------------------------
  wire [ROWS*8-1:0] act_row;

  generate
    for (genvar r = 0; r < ROWS; r++) begin : g_feed
      wire [CYC_W-1:0] si = cyc - CYC_W'(r);
      wire             in_win;
      if (r == 0) begin : g_first
        assign in_win = (state == ST_ARRAY) && (si <= s_top);
      end else begin : g_later
        assign in_win = (state == ST_ARRAY) && (cyc >= CYC_W'(r)) && (si <= s_top);
      end
      wire [S_W-1:0] sidx = in_win ? si[S_W-1:0] : '0;
      assign act_row[r*8 +: 8] = in_win ? act_buf[(sidx*ROWS + r)*8 +: 8] : 8'h00;
    end
  endgenerate

  wire [COLS*PSUM_W-1:0] psum_col;

  npu_array #(
      .ROWS    (ROWS),
      .COLS    (COLS),
      .PSUM_W  (PSUM_W),
      .MUL_ARCH(MUL_ARCH),
      .ADD_ARCH(ADD_ARCH)
  ) u_array (
      .clk     (clk),
      .rst_n   (rst_n),
      .w_en    (wr_w),
      .w_byte  (hb),
      .act_row (act_row),
      .psum_col(psum_col)
  );

  // Valid/sample-index pipeline. Stage ROWS+c is the cycle in which the
  // column-c result of that sample stands at the bottom of the array.
  logic [SKEW-1:0]     vpipe;
  logic [SKEW*S_W-1:0] ipipe;
  logic [S_W-1:0]      inj_idx;

  wire inject = (state == ST_ARRAY) && (cyc <= s_top);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      vpipe   <= '0;
      ipipe   <= '0;
      inj_idx <= '0;
    end else begin
      vpipe <= {vpipe[SKEW-2:0], inject};
      ipipe <= {ipipe[(SKEW-1)*S_W-1:0], inj_idx};
      if (run_pulse)   inj_idx <= '0;
      else if (inject) inj_idx <= inj_idx + S_W'(1);
    end
  end

  // ---------------------------------------------------------------------------
  // Accumulator bank: COLS independent banks of S_MAX entries. Up to COLS
  // results retire in the same cycle (different columns, different samples), so
  // each bank owns its own accumulate adder.
  // ---------------------------------------------------------------------------
  wire [COLS*S_MAX*ACC_W-1:0] acc;
  wire [COLS*ACC_W-1:0]       acc_rd;
  wire [COLS-1:0]             col_valid;
  wire [COLS*S_W-1:0]         col_idx;
  wire [COLS*ACC_W-1:0]       acc_next;
  wire [COLS-1:0]             col_ovf;

  generate
    for (genvar c = 0; c < COLS; c++) begin : g_bank
      assign col_valid[c]          = vpipe[ROWS + c];
      assign col_idx[c*S_W +: S_W] = ipipe[(ROWS + c)*S_W +: S_W];

      // One read port per bank, shared between the retiring sample during
      // ST_ARRAY and the requantization walk during ST_REQ.
      wire [S_W-1:0] rd_i = (state == ST_REQ) ? req_s : col_idx[c*S_W +: S_W];
      // 32-bit index arithmetic keeps the flat-vector select portable across
      // Icarus, Verilator and Yosys; the upper bits fold away to constants.
      wire [31:0] rd_flat = c * S_MAX + {{(32 - S_W) {1'b0}}, rd_i};
      assign acc_rd[c*ACC_W +: ACC_W] = acc[rd_flat*ACC_W +: ACC_W];

      wire [ACC_W-1:0] base = acc_first ? bias_buf[c*BB*8 +: ACC_W]
                                        : acc_rd[c*ACC_W +: ACC_W];
      wire signed [PSUM_W-1:0] ps     = psum_col[c*PSUM_W +: PSUM_W];
      wire        [ACC_W-1:0]  ps_ext = {{(ACC_W - PSUM_W) {ps[PSUM_W-1]}}, ps};

      wire [ACC_W:0] sum_ext;
      wire           sum_co;
      npu_adder #(
          .WIDTH(ACC_W + 1),
          .ARCH (ADD_ARCH)
      ) u_acc_add (
          .a   ({ps_ext[ACC_W-1], ps_ext}),
          .b   ({base[ACC_W-1], base}),
          .cin (1'b0),
          .sum (sum_ext),
          .cout(sum_co)
      );

      // Signed overflow: the two sign bits of the widened sum disagree.
      assign col_ovf[c] = sum_ext[ACC_W] ^ sum_ext[ACC_W-1];
      assign acc_next[c*ACC_W +: ACC_W] =
          col_ovf[c] ? (sum_ext[ACC_W] ? ACC_MIN : ACC_MAX) : sum_ext[ACC_W-1:0];

      // One register per entry with its own enable, which synthesises into a
      // small decoder instead of a barrel-shifted write mask.
      for (genvar s = 0; s < S_MAX; s++) begin : g_ent
        logic [ACC_W-1:0] q;
        wire wen = (state == ST_ARRAY) && col_valid[c]
                   && (col_idx[c*S_W +: S_W] == S_W'(s));
        always_ff @(posedge clk or negedge rst_n) begin
          if (!rst_n)        q <= '0;
          else if (soft_rst) q <= '0;
          else if (wen)      q <= acc_next[c*ACC_W +: ACC_W];
        end
        assign acc[(c*S_MAX + s)*ACC_W +: ACC_W] = q;
      end

      wire _unused_bank = &{1'b0, sum_co};
    end
  endgenerate

  wire [ACC_W-1:0] acc_req_val = acc_rd[req_c*ACC_W +: ACC_W];

  // ---------------------------------------------------------------------------
  // Shared serial requantizer plus the activation function.
  // ---------------------------------------------------------------------------
  wire [M_W-1:0]  m_sel  = quant_buf[req_c*QB*8 +: M_W];
  wire [SH_W-1:0] sh_sel = quant_buf[(req_c*QB + MB)*8 +: SH_W];

  wire              rq_busy, rq_done, rq_sat;
  wire signed [7:0] rq_q;

  npu_requant #(
      .T_W     (ACC_W),
      .M_W     (M_W),
      .SH_W    (SH_W),
      .ADD_ARCH(ADD_ARCH)
  ) u_requant (
      .clk     (clk),
      .rst_n   (rst_n),
      .start   (req_start),
      .acc_in  (acc_req_val),
      .m_in    (m_sel),
      .shift_in(sh_sel),
      .zp_in   (zp_reg),
      .busy    (rq_busy),
      .done    (rq_done),
      .q_out   (rq_q),
      .sat_out (rq_sat)
  );

  wire signed [7:0] act_y;

  npu_activation u_act (
      .q_in    (rq_q),
      .sel     (act_sel_reg),
      .leaky_k (leaky_k_reg),
      .zp      (zp_reg),
      .clamp_lo(clamp_lo_reg),
      .clamp_hi(clamp_hi_reg),
      .y       (act_y)
  );

  // ---------------------------------------------------------------------------
  // Result bytes, one register per (sample, channel).
  // ---------------------------------------------------------------------------
  wire [RES_LEN*8-1:0] res;

  generate
    for (genvar e = 0; e < RES_LEN; e++) begin : g_res
      localparam int E_S = e / COLS;
      localparam int E_C = e % COLS;
      logic [7:0] q;
      wire wen = (state == ST_REQ) && rq_done
                 && (req_s == S_W'(E_S)) && (req_c == C_W'(E_C));
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)        q <= 8'h00;
        else if (soft_rst) q <= 8'h00;
        else if (wen)      q <= act_y;
      end
      assign res[e*8 +: 8] = q;
    end
  endgenerate

  // ---------------------------------------------------------------------------
  // Sequencer.
  // ---------------------------------------------------------------------------
  logic       done_reg, err_reg, sat_reg, ovf_reg;
  logic [1:0] err_code_reg;

  wire array_last = (cyc == s_top + CYC_W'(ROWS + COLS));
  wire req_last   = (req_s == s_cnt_m1) && (req_c == C_W'(COLS - 1));
  wire any_ovf    = |(col_valid & col_ovf);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state        <= ST_IDLE;
      cyc          <= '0;
      acc_first    <= 1'b1;
      do_req       <= 1'b0;
      req_s        <= '0;
      req_c        <= '0;
      req_start    <= 1'b0;
      done_reg     <= 1'b0;
      err_reg      <= 1'b0;
      sat_reg      <= 1'b0;
      ovf_reg      <= 1'b0;
      err_code_reg <= 2'd0;
    end else begin
      req_start <= 1'b0;

      if (err_set) begin
        err_reg      <= 1'b1;
        err_code_reg <= err_code_new;
      end
      if (clr_flags) begin
        err_reg      <= 1'b0;
        sat_reg      <= 1'b0;
        ovf_reg      <= 1'b0;
        err_code_reg <= 2'd0;
        done_reg     <= 1'b0;
      end

      if (soft_rst) begin
        state        <= ST_IDLE;
        cyc          <= '0;
        req_s        <= '0;
        req_c        <= '0;
        req_start    <= 1'b0;
        done_reg     <= 1'b0;
        err_reg      <= 1'b0;
        sat_reg      <= 1'b0;
        ovf_reg      <= 1'b0;
        err_code_reg <= 2'd0;
      end else begin
        case (state)
          ST_IDLE: begin
            if (run_pulse) begin
              state     <= ST_ARRAY;
              cyc       <= '0;
              acc_first <= !run_acc_arg;
              do_req    <= run_req_arg;
              done_reg  <= 1'b0;
            end
          end

          ST_ARRAY: begin
            cyc <= cyc + CYC_W'(1);
            if (any_ovf) ovf_reg <= 1'b1;
            if (array_last) begin
              if (do_req) begin
                state     <= ST_REQ;
                req_s     <= '0;
                req_c     <= '0;
                req_start <= 1'b1;
              end else begin
                state <= ST_DONE;
              end
            end
          end

          ST_REQ: begin
            if (rq_done) begin
              if (rq_sat) sat_reg <= 1'b1;
              if (req_last) begin
                state <= ST_DONE;
              end else begin
                if (req_c == C_W'(COLS - 1)) begin
                  req_c <= '0;
                  req_s <= req_s + S_W'(1);
                end else begin
                  req_c <= req_c + C_W'(1);
                end
                req_start <= 1'b1;
              end
            end
          end

          default: begin  // ST_DONE
            done_reg <= 1'b1;
            state    <= ST_IDLE;
          end
        endcase
      end
    end
  end

  assign busy = (state != ST_IDLE) || rq_busy;
  assign done = done_reg;
  assign err  = err_reg;
  assign sat  = sat_reg;
  assign ovf  = ovf_reg;

  // ---------------------------------------------------------------------------
  // Readback multiplexer. Accumulator entries are addressed as
  // ((sample << ceil(log2(COLS))) | channel) * 4 + byte, little endian, sign
  // extended to 32 bits, so a host can read the raw sums as plain int32.
  // ---------------------------------------------------------------------------
  wire [7:0] status_byte = {err_code_reg, 1'b0, ovf_reg, sat_reg, err_reg,
                            done_reg, busy};

  wire [ENT_W-1:0] acc_ent = rd_ptr[RPTR_W-1:2];
  wire [S_W-1:0]   acc_e_s = acc_ent[ENT_W-1:C_W];
  wire [C_W-1:0]   acc_e_c = acc_ent[C_W-1:0];
  wire [31:0]      acc_rb_i = {{(32 - C_W) {1'b0}}, acc_e_c} * S_MAX
                           + {{(32 - S_W) {1'b0}}, acc_e_s};
  wire [ACC_W-1:0] acc_rb  = acc[acc_rb_i*ACC_W +: ACC_W];
  wire [31:0]      acc_rb32 = {{(32 - ACC_W) {acc_rb[ACC_W-1]}}, acc_rb};

  wire             res_in  = (rd_ptr < RPTR_W'(RES_LEN));
  wire [RIDX_W-1:0] res_i  = res_in ? rd_ptr[RIDX_W-1:0] : '0;

  logic [7:0] id_byte;
  always_comb begin
    case (rd_ptr[2:0])
      3'd0:    id_byte = 8'h4E;                 // 'N'
      3'd1:    id_byte = 8'h38;                 // '8'
      3'd2:    id_byte = 8'(VERSION);
      3'd3:    id_byte = {4'(ROWS), 4'(COLS)};
      3'd4:    id_byte = {4'(S_MAX), 4'(0)};
      3'd5:    id_byte = 8'(ACC_W);
      3'd6:    id_byte = 8'(M_W);
      default: id_byte = {4'(MUL_ARCH), 4'(ADD_ARCH)};
    endcase
  end

  logic [7:0] rb;
  always_comb begin
    case (rd_src)
      2'd0:    rb = res_in ? res[res_i*8 +: 8] : 8'h00;
      2'd1:    rb = acc_rb32[rd_ptr[1:0]*8 +: 8];
      2'd2:    rb = status_byte;
      default: rb = id_byte;
    endcase
  end

  assign data_out = rb;

  wire _unused = &{1'b0, col_ovf, acc_rd, do_req};

endmodule

`default_nettype wire
