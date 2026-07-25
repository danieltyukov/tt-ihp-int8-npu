// SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
// SPDX-License-Identifier: Apache-2.0
//
// npu_host_if: framed byte protocol over the Tiny Tapeout pins.
//
// A frame is one command byte followed by a fixed number of payload bytes:
//
//   cycle:   wr=1 is_cmd=1  data={opcode[3:0], arg[3:0]}
//   cycle:   wr=1 is_cmd=0  data=payload[0]
//   ...      wr=1 is_cmd=0  data=payload[len-1]
//
// Frames may be spread over any number of idle cycles. Payload length is fixed
// per opcode, which lets the interface detect both a payload byte too many and a
// frame abandoned early, and report them in the sticky error code.
//
//   op    name       payload            effect
//   0x0   NOP        0
//   0x1   CFG        1                  sample count minus one
//   0x2   LD_W       ROWS*COLS          weight tile, row-major
//   0x3   LD_ACT     S_MAX*ROWS         activations, sample-major
//   0x4   LD_BIAS    COLS*3             per channel, 24-bit little endian
//   0x5   LD_QUANT   COLS*QBYTES        per channel multiplier then shift
//   0x6   LD_POST    4                  zp, clamp_lo, clamp_hi, {leaky, act}
//   0x7   RUN        0                  arg[0] accumulate, arg[1] requantize
//   0x8   RDSEL      1                  arg[1:0] source, payload = start index
//   0x9   CLR        0                  clear sticky flags and done
//   0xA   SRST       0                  soft reset (keeps weights and params)
//   0xF   ID         0                  select the identity block, pointer 0
//
// Error codes: 1 unknown opcode, 2 write while busy, 3 frame length violation.
// Any write attempted while the core is busy is dropped, not queued.

`default_nettype none

module npu_host_if #(
    parameter int W_LEN     = 8,   // LD_W payload length
    parameter int ACT_LEN   = 16,  // LD_ACT payload length
    parameter int BIAS_LEN  = 6,   // LD_BIAS payload length
    parameter int QUANT_LEN = 6,   // LD_QUANT payload length
    parameter int RPTR_W    = 6    // readback pointer width
) (
    input  wire        clk,
    input  wire        rst_n,
    // Pin-level host bus.
    input  wire [7:0]  data_in,
    input  wire        wr,
    input  wire        is_cmd,
    input  wire        rd,
    // Core status.
    input  wire        core_busy,
    // Byte-stream write strobes. byte_out carries the payload byte and is
    // registered alongside them, so it is valid in the same cycle as a strobe.
    output logic [7:0] byte_out,
    output logic       wr_w,
    output logic       wr_act,
    output logic       wr_bias,
    output logic       wr_quant,
    output logic       wr_post,
    output logic       wr_cfg,
    output logic [1:0] post_idx,
    // Control pulses.
    output logic       run,
    output logic       run_acc,
    output logic       run_req,
    output logic       clr_flags,
    output logic       soft_rst,
    // Readback addressing.
    output logic [1:0] rd_src,
    output logic [RPTR_W-1:0] rd_ptr,
    // Sticky error reporting.
    output logic       err_set,
    output logic [1:0] err_code
);

  localparam int PLEN_W = 8;

  localparam logic [3:0] OP_NOP      = 4'h0;
  localparam logic [3:0] OP_CFG      = 4'h1;
  localparam logic [3:0] OP_LD_W     = 4'h2;
  localparam logic [3:0] OP_LD_ACT   = 4'h3;
  localparam logic [3:0] OP_LD_BIAS  = 4'h4;
  localparam logic [3:0] OP_LD_QUANT = 4'h5;
  localparam logic [3:0] OP_LD_POST  = 4'h6;
  localparam logic [3:0] OP_RUN      = 4'h7;
  localparam logic [3:0] OP_RDSEL    = 4'h8;
  localparam logic [3:0] OP_CLR      = 4'h9;
  localparam logic [3:0] OP_SRST     = 4'hA;
  localparam logic [3:0] OP_ID       = 4'hF;

  localparam logic [1:0] ERR_OPCODE = 2'd1;
  localparam logic [1:0] ERR_BUSY   = 2'd2;
  localparam logic [1:0] ERR_FRAME  = 2'd3;

  logic [3:0]        op;
  logic [PLEN_W-1:0] pcount;


  // Expected payload length, as a function of the opcode being processed.
  function automatic [PLEN_W-1:0] payload_len(input logic [3:0] o);
    case (o)
      OP_CFG:      payload_len = PLEN_W'(1);
      OP_LD_W:     payload_len = PLEN_W'(W_LEN);
      OP_LD_ACT:   payload_len = PLEN_W'(ACT_LEN);
      OP_LD_BIAS:  payload_len = PLEN_W'(BIAS_LEN);
      OP_LD_QUANT: payload_len = PLEN_W'(QUANT_LEN);
      OP_LD_POST:  payload_len = PLEN_W'(4);
      OP_RDSEL:    payload_len = PLEN_W'(1);
      default:     payload_len = PLEN_W'(0);
    endcase
  endfunction

  function automatic logic known_op(input logic [3:0] o);
    case (o)
      OP_NOP, OP_CFG, OP_LD_W, OP_LD_ACT, OP_LD_BIAS, OP_LD_QUANT,
      OP_LD_POST, OP_RUN, OP_RDSEL, OP_CLR, OP_SRST, OP_ID: known_op = 1'b1;
      default: known_op = 1'b0;
    endcase
  endfunction

  wire [3:0] new_op  = data_in[7:4];
  wire [3:0] new_arg = data_in[3:0];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      op        <= OP_NOP;
      pcount    <= '0;
      byte_out  <= 8'h00;
      post_idx  <= 2'd0;
      rd_src    <= 2'd0;
      rd_ptr    <= '0;
      err_set   <= 1'b0;
      err_code  <= 2'd0;
      wr_w      <= 1'b0;
      wr_act    <= 1'b0;
      wr_bias   <= 1'b0;
      wr_quant  <= 1'b0;
      wr_post   <= 1'b0;
      wr_cfg    <= 1'b0;
      run       <= 1'b0;
      run_acc   <= 1'b0;
      run_req   <= 1'b0;
      clr_flags <= 1'b0;
      soft_rst  <= 1'b0;
    end else begin
      wr_w      <= 1'b0;
      wr_act    <= 1'b0;
      wr_bias   <= 1'b0;
      wr_quant  <= 1'b0;
      wr_post   <= 1'b0;
      wr_cfg    <= 1'b0;
      run       <= 1'b0;
      clr_flags <= 1'b0;
      soft_rst  <= 1'b0;
      err_set   <= 1'b0;

      if (wr && core_busy) begin
        // Rejected, not queued. The host is expected to poll busy.
        err_set  <= 1'b1;
        err_code <= ERR_BUSY;
      end else if (wr && is_cmd) begin
        // Starting a new frame while the previous one is unfinished is an error,
        // but the new frame is still accepted so the host can resynchronize.
        if ((pcount != '0) && (pcount != payload_len(op))) begin
          err_set  <= 1'b1;
          err_code <= ERR_FRAME;
        end
        if (!known_op(new_op)) begin
          err_set  <= 1'b1;
          err_code <= ERR_OPCODE;
          op       <= OP_NOP;
        end else begin
          op <= new_op;
        end
        pcount <= '0;

        case (new_op)
          OP_RUN: begin
            run     <= 1'b1;
            run_acc <= new_arg[0];
            run_req <= new_arg[1];
          end
          OP_CLR:   clr_flags <= 1'b1;
          OP_SRST:  soft_rst  <= 1'b1;
          OP_RDSEL: rd_src    <= new_arg[1:0];
          OP_ID: begin
            rd_src <= 2'd3;
            rd_ptr <= '0;
          end
          default: ;  // no immediate effect
        endcase

      end else if (wr && !is_cmd) begin
        if (pcount >= payload_len(op)) begin
          err_set  <= 1'b1;
          err_code <= ERR_FRAME;
        end else begin
          pcount   <= pcount + PLEN_W'(1);
          // The strobes below are registered, so the byte and the register
          // index have to be captured with them or the destination would see
          // whatever the host drives in the following cycle.
          byte_out <= data_in;
          post_idx <= pcount[1:0];
          case (op)
            OP_LD_W:     wr_w     <= 1'b1;
            OP_LD_ACT:   wr_act   <= 1'b1;
            OP_LD_BIAS:  wr_bias  <= 1'b1;
            OP_LD_QUANT: wr_quant <= 1'b1;
            OP_LD_POST:  wr_post  <= 1'b1;
            OP_CFG:      wr_cfg   <= 1'b1;
            OP_RDSEL:    rd_ptr   <= data_in[RPTR_W-1:0];
            default: ;
          endcase
        end
      end else if (rd) begin
        rd_ptr <= rd_ptr + RPTR_W'(1);
      end
    end
  end

  wire _unused = &{1'b0, data_in, new_arg[3:2]};

endmodule

`default_nettype wire
