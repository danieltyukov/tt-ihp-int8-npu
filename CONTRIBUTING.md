# Contributing

Contributions are welcome, especially new arithmetic architectures, additional
quantized layer types, and tighter area.

## Before you open a pull request

```
make lint        # verilator --lint-only -Wall, must be silent
make arith       # exhaustive arithmetic bench, must print TB_ARITH PASS
make test        # cocotb accelerator suite, no failures
make formal      # only if you touched src/npu_adder.sv or the multiplier
make ppa         # only if you touched arithmetic or geometry
```

`make formal` needs `sby` and `z3` and takes about ten minutes. The `formal`
workflow runs it on every push and fails if the regenerated
`docs/formal/summary.md` differs from the committed one, so commit the
regenerated file with the change that caused it.

If a change affects area or timing, include the measured numbers. `make ppa`
writes `docs/synth/ppa.md`; quote the lines that changed in the pull request
rather than saying "slightly smaller".

## What the tests expect

- Every new arithmetic variant must be bit-identical to the existing ones.
  `test/test_arith.py` runs all 65536 signed 8x8 operand pairs and randomized
  adder vectors at four widths; a new variant is not done until it is in those
  loops. `docs/ADAPTING.md` documents the exact interface a new adder or
  multiplier has to implement.
- Every functional change must be checked against `test/golden.py`, not against
  the RTL's own previous behaviour. If the golden model needs to change, that is
  a specification change and the reasoning belongs in `docs/DESIGN.md`.
- New tests must assert something specific. A test that runs the design without
  comparing against the model is not a test.

## Style

- SystemVerilog: `default_nettype none` in every file, one module per file with
  a matching name, explicit widths, no unpacked arrays of wires (Yosys turns
  them into memories and drops the drivers), asynchronous reset.
- Parameters are checked at elaboration in the module that depends on them, and
  every module's defaults must satisfy its own checks.
- Comments explain why, not what. If a line exists because of a tool quirk or a
  bug that was fixed, say so, and say which tool.
- No emoji, and no em dashes as sentence punctuation, anywhere in the repository.

## Reporting a bug

A failing case is worth far more than a description. The most useful report is a
new case added to `test/test_npu.py` with the exact weights, activations, bias,
multiplier, shift and zero point, plus what the golden model says it should be.
