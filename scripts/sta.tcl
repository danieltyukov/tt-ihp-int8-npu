# SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
# SPDX-License-Identifier: Apache-2.0
#
# Static timing analysis of the mapped netlist with OpenSTA against the IHP
# sg13g2 standard cells. This is what backs the clock_hz in info.yaml, so it is
# a real measurement and not a delay estimate.
#
#   sta -no_splash -exit scripts/sta.tcl
#
# Overrides: write build/sta_config.tcl setting any of `period`, `netlist`,
# `top`, `corner` or `pdk`. A file is used rather than environment variables
# because the local `sta` is a Docker wrapper and does not forward the
# environment into the container.
#
# Wire load is not modelled: there is no placement here, so these numbers are
# the cell-delay component of the path. The hardening flow's own STA, with real
# parasitics, is the authority; this bounds the design's own logic depth.

set pdk     "/home/danieltyukov/.local/share/pdk/IHP-Open-PDK/ihp-sg13g2"
set netlist "docs/synth/top_shipped.full.netlist.v"
set top     "tt_um_danieltyukov_int8_npu"
set period  25.0
set corner  "sg13g2_stdcell_typ_1p20V_25C"

if {[file exists build/sta_config.tcl]} {
  source build/sta_config.tcl
}

# Under OpenROAD both LEF reads must precede read_verilog (ORD-2010); plain
# OpenSTA has no read_lef at all, so only do it when the command exists.
if {[llength [info commands read_lef]]} {
  read_lef $pdk/libs.ref/sg13g2_stdcell/lef/sg13g2_tech.lef
  read_lef $pdk/libs.ref/sg13g2_stdcell/lef/sg13g2_stdcell.lef
}
read_liberty $pdk/libs.ref/sg13g2_stdcell/lib/$corner.lib
read_verilog $netlist
link_design $top

create_clock -name clk -period $period {clk}

# Inputs arrive from the Tiny Tapeout mux, outputs drive it. Keep the external
# assumptions explicit and modest rather than silently zero.
set inputs [get_ports {ui_in[*] uio_in[*] ena rst_n}]
set_input_delay -clock clk 2.0 $inputs
set_output_delay -clock clk 2.0 [get_ports {uo_out[*] uio_out[*] uio_oe[*]}]
set_load 0.05 [get_ports {uo_out[*] uio_out[*] uio_oe[*]}]
set_driving_cell -lib_cell sg13g2_buf_2 $inputs
set_clock_uncertainty 0.25 [get_clocks clk]
set_clock_transition 0.15 [get_clocks clk]

puts "=== setup timing, period ${period} ns, corner $corner ==="
report_checks -path_delay max -format full_clock_expanded -digits 4
puts "=== hold timing ==="
report_checks -path_delay min -digits 4
puts "=== worst slack ==="
report_worst_slack -max -digits 4
report_worst_slack -min -digits 4
puts "=== design summary ==="
if {[llength [info commands report_design_area]]} { report_design_area }
report_tns -digits 4
report_wns -digits 4
puts "=== five worst setup paths ==="
report_checks -path_delay max -digits 4 -group_count 5 -format slack_only
