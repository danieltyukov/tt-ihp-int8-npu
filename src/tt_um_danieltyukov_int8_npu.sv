// SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
// SPDX-License-Identifier: Apache-2.0
//
// tt_um_danieltyukov_int8_npu: Tiny Tapeout top level for the signed INT8
// systolic inference accelerator.
//
// Pin map
//   ui_in[7:0]   host data byte in
//   uio_in[0]    wr      write strobe, samples ui_in on this rising edge
//   uio_in[1]    is_cmd  1 = command byte, 0 = payload byte
//   uio_in[2]    rd      advance the readback pointer on this rising edge
//   uo_out[7:0]  readback byte at the current pointer
//   uio_out[3]   busy
//   uio_out[4]   done
//   uio_out[5]   err     sticky, cleared by CLR
//   uio_out[6]   sat     sticky, an output saturated at an INT8 rail
//   uio_out[7]   ovf     sticky, an accumulator saturated
//
// The configuration below is the one that was hardened: see README.md for the
// measured area and the tile-size justification, and docs/ADAPTING.md for how to
// resize the array.

`default_nettype none

module tt_um_danieltyukov_int8_npu #(
    // Array geometry and datapath widths. The defaults are the configuration
    // that was hardened; see README.md for the measured area and the tile-size
    // justification, and docs/ADAPTING.md for how to resize. Every value is
    // checked at elaboration inside the submodules.
    parameter int ROWS     = 4,   // reduction depth resident in the array
    parameter int COLS     = 2,   // output channels computed in parallel
    parameter int S_MAX    = 6,   // activation vectors held on chip
    parameter int ACC_W    = 24,  // accumulator and bias width
    parameter int M_W      = 16,  // requantization multiplier width
    parameter int SH_W     = 5,   // requantization shift width
    parameter int MUL_ARCH = 1,   // 0 array, 1 Wallace, 2 Booth radix-4
    parameter int ADD_ARCH = 4    // 0 RCA, 1 BK, 2 KS, 3 Sklansky, 4 HC
) (
    input  wire [7:0] ui_in,    // dedicated inputs
    output wire [7:0] uo_out,   // dedicated outputs
    input  wire [7:0] uio_in,   // IOs: input path
    output wire [7:0] uio_out,  // IOs: output path
    output wire [7:0] uio_oe,   // IOs: enable path (1 = drive out)
    input  wire       ena,      // high when the design is selected
    input  wire       clk,
    input  wire       rst_n
);

  wire busy, done, err, sat, ovf;

  npu_core #(
      .ROWS    (ROWS),
      .COLS    (COLS),
      .S_MAX   (S_MAX),
      .ACC_W   (ACC_W),
      .M_W     (M_W),
      .SH_W    (SH_W),
      .MUL_ARCH(MUL_ARCH),
      .ADD_ARCH(ADD_ARCH)
  ) u_core (
      .clk     (clk),
      .rst_n   (rst_n),
      .data_in (ui_in),
      .wr      (uio_in[0]),
      .is_cmd  (uio_in[1]),
      .rd      (uio_in[2]),
      .data_out(uo_out),
      .busy    (busy),
      .done    (done),
      .err     (err),
      .sat     (sat),
      .ovf     (ovf)
  );

  assign uio_out = {ovf, sat, err, done, busy, 3'b000};
  assign uio_oe  = 8'b1111_1000;

  // ena is driven by the Tiny Tapeout mux and is not used by the design.
  wire _unused = &{ena, uio_in[7:3], 1'b0};

endmodule

`default_nettype wire
