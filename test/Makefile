# SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
# SPDX-License-Identifier: Apache-2.0
#
# cocotb makefile, following the Tiny Tapeout template layout.
#   make           run the accelerator suite against the RTL
#   make arith     run the arithmetic-variant equivalence suite
#   make GATES=yes run the accelerator suite against the hardened netlist
#
# See https://docs.cocotb.org/en/stable/quickstart.html

SIM ?= icarus
FST ?= -fst
TOPLEVEL_LANG ?= verilog
SRC_DIR = $(PWD)/../src
PROJECT_SOURCES = npu_adder.sv npu_csa_reduce.sv npu_pp_baugh_wooley.sv \
                  npu_pp_booth4.sv npu_mult.sv npu_pe.sv npu_array.sv \
                  npu_requant.sv npu_activation.sv npu_host_if.sv \
                  npu_core.sv tt_um_danieltyukov_int8_npu.sv

ifneq ($(GATES),yes)

# RTL simulation:
SIM_BUILD = sim_build/rtl
VERILOG_SOURCES += $(addprefix $(SRC_DIR)/,$(PROJECT_SOURCES))

else

# Gate level simulation:
SIM_BUILD = sim_build/gl
COMPILE_ARGS += -DGL_TEST
COMPILE_ARGS += -DFUNCTIONAL
COMPILE_ARGS += -DSIM
VERILOG_SOURCES += $(PDK_ROOT)/ihp-sg13g2/libs.ref/sg13g2_io/verilog/sg13g2_io.v
VERILOG_SOURCES += $(PDK_ROOT)/ihp-sg13g2/libs.ref/sg13g2_stdcell/verilog/sg13g2_stdcell.v
# this gets copied in by the GDS action workflow
VERILOG_SOURCES += $(PWD)/gate_level_netlist.v

endif

COMPILE_ARGS += -I$(SRC_DIR)

VERILOG_SOURCES += $(PWD)/tb.v
TOPLEVEL = tb

COCOTB_TEST_MODULES = test_npu

include $(shell cocotb-config --makefiles)/Makefile.sim

# Arithmetic-variant equivalence runs against its own testbench, which brings up
# every adder and multiplier architecture side by side.
.PHONY: arith
arith:
	$(MAKE) -f Makefile.arith

.PHONY: everything
everything:
	$(MAKE) clean && $(MAKE)
	$(MAKE) -f Makefile.arith clean && $(MAKE) -f Makefile.arith

.PHONY: trace
trace:
	$(MAKE) COCOTB_TEST_MODULES=test_trace COCOTB_RESULTS_FILE=results_trace.xml

.PHONY: demo
demo:
	$(MAKE) COCOTB_TEST_MODULES=test_demo COCOTB_RESULTS_FILE=results_demo.xml
