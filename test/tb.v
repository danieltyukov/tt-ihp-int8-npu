// SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
// SPDX-License-Identifier: Apache-2.0
//
// Testbench wrapper for the cocotb suite: instantiates the Tiny Tapeout top
// level and exposes its pins as plain signals.

`default_nettype none
`timescale 1ns / 1ps

module tb ();

  // Dumping every signal of an 8000-cell design on every cycle costs more than
  // the simulation itself, so it is opt-out: run with +nodump (the Makefile's
  // NODUMP=1) when you only care about the assertions.
  initial begin
    if (!$test$plusargs("nodump")) begin
      $dumpfile("tb.fst");
      $dumpvars(0, tb);
    end
    #1;
  end

  reg        clk;
  reg        rst_n;
  reg        ena;
  reg  [7:0] ui_in;
  reg  [7:0] uio_in;
  wire [7:0] uo_out;
  wire [7:0] uio_out;
  wire [7:0] uio_oe;

  tt_um_danieltyukov_int8_npu user_project (
      .ui_in  (ui_in),
      .uo_out (uo_out),
      .uio_in (uio_in),
      .uio_out(uio_out),
      .uio_oe (uio_oe),
      .ena    (ena),
      .clk    (clk),
      .rst_n  (rst_n)
  );

endmodule
