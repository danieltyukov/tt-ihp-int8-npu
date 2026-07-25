# SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
# SPDX-License-Identifier: Apache-2.0
#
# Top-level convenience targets. Everything runs locally with iverilog,
# verilator, yosys and a repo-local Python venv; nothing needs network access
# once the venv exists and the sg13g2 liberty is in ./pdk.

PY      := .venv/bin/python
PIP     := .venv/bin/pip
VENV_OK := .venv/.stamp
SRC     := $(wildcard src/*.sv)
TOP     := tt_um_danieltyukov_int8_npu
LINT_TOPS := $(TOP) npu_core npu_array npu_pe npu_requant npu_mult npu_adder \
             npu_activation npu_host_if

.PHONY: help
help:
	@echo "make venv       create .venv and install test requirements"
	@echo "make lint       verilator --lint-only -Wall on every module"
	@echo "make arith      exhaustive iverilog bench for the arithmetic library"
	@echo "make test       cocotb accelerator suite"
	@echo "make test-all   accelerator suite, variant equivalence and the NN demo"
	@echo "make trace      capture simulation data for the figures"
	@echo "make formal     prove every arithmetic variant against a + b / a * b"
	@echo "make synth      Yosys area and depth for the shipped configuration"
	@echo "make ppa        full PPA comparison, writes docs/synth/ppa.{md,json}"
	@echo "make images     regenerate every figure in docs/img"
	@echo "make demo-model retrain and requantize the demo network"
	@echo "make clean      remove build artifacts"

$(VENV_OK): test/requirements.txt
	python3 -m venv .venv
	$(PIP) install -q -r test/requirements.txt
	@touch $@

.PHONY: venv
venv: $(VENV_OK)

.PHONY: lint
lint:
	@for top in $(LINT_TOPS); do \
	  echo "verilator --lint-only -Wall --top-module $$top"; \
	  verilator --lint-only -Wall --top-module $$top $(SRC) || exit 1; \
	done
	@echo "lint clean"

.PHONY: arith
arith:
	@mkdir -p build
	iverilog -g2012 -o build/tb_arith test/tb_arith.sv $(SRC)
	vvp build/tb_arith | tail -4

.PHONY: test
test: $(VENV_OK)
	cd test && PATH=$(CURDIR)/.venv/bin:$$PATH $(MAKE) clean && \
	  PATH=$(CURDIR)/.venv/bin:$$PATH $(MAKE)

.PHONY: test-all
test-all: test
	cd test && PATH=$(CURDIR)/.venv/bin:$$PATH $(MAKE) -f Makefile.arith clean && \
	  PATH=$(CURDIR)/.venv/bin:$$PATH $(MAKE) -f Makefile.arith
	cd test && PATH=$(CURDIR)/.venv/bin:$$PATH $(MAKE) demo

.PHONY: trace
trace: $(VENV_OK)
	cd test && PATH=$(CURDIR)/.venv/bin:$$PATH $(MAKE) trace

.PHONY: formal
formal: $(VENV_OK)
	$(PY) scripts/run_formal.py

.PHONY: synth
synth:
	$(PY) scripts/synth.py --top $(TOP) --name top_shipped --effort full --netlist

.PHONY: ppa
ppa: $(VENV_OK)
	$(PY) scripts/run_ppa.py

.PHONY: demo-model
demo-model: $(VENV_OK)
	$(PY) scripts/train_demo.py

.PHONY: images
images: $(VENV_OK)
	$(PY) scripts/gen_diagrams.py
	$(PY) scripts/gen_dataflow_gif.py
	$(PY) scripts/plot_ppa.py
	$(PY) scripts/plot_demo.py

.PHONY: clean
clean:
	rm -rf build test/sim_build test/results*.xml test/*.fst
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: readme
readme: $(VENV_OK)
	$(PY) scripts/fill_readme.py

.PHONY: sta
sta:
	@mkdir -p build docs/synth
	printf 'set period 25.0\n' > build/sta_config.tcl
	sta -no_splash -exit scripts/sta.tcl | grep -vE "unsupported expression|Warning 503" \
	  > docs/synth/sta_typ.txt
	printf 'set period 25.0\nset corner "sg13g2_stdcell_slow_1p08V_125C"\n' > build/sta_config.tcl
	sta -no_splash -exit scripts/sta.tcl | grep -vE "unsupported expression|Warning 503" \
	  > docs/synth/sta_slow.txt
	@rm -f build/sta_config.tcl
	@grep -H "worst slack" docs/synth/sta_*.txt
