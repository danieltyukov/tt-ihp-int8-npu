# Adapting this project

This repository is meant to be forked. Everything that defines the accelerator
is a parameter on one module, every arithmetic architecture is selected by a
number, and the whole PPA and figure pipeline re-runs from a single `make`
target. This guide covers the changes people actually want to make.

## Contents

- [Getting set up](#getting-set-up)
- [Resizing the array](#resizing-the-array)
- [Retargeting the tile size](#retargeting-the-tile-size)
- [Swapping in a different quantized layer](#swapping-in-a-different-quantized-layer)
- [Adding a new adder architecture](#adding-a-new-adder-architecture)
- [Adding a new multiplier architecture](#adding-a-new-multiplier-architecture)
- [Re-running the PPA comparison](#re-running-the-ppa-comparison)
- [Changing the requantization precision](#changing-the-requantization-precision)
- [Going faster](#going-faster)
- [Things that will bite you](#things-that-will-bite-you)

## Getting set up

```
git clone https://github.com/danieltyukov/tt-ihp-int8-npu
cd tt-ihp-int8-npu
make venv                       # .venv with cocotb, numpy, matplotlib, pillow
make lint                       # verilator -Wall on every module
make arith                      # exhaustive arithmetic bench, no venv needed
make test                       # cocotb accelerator suite
```

Tools used: `iverilog` 12.0, `verilator` 5.020, `yosys` 0.33, Python 3.12.
Anything that reports area needs the IHP standard-cell liberty. It is not
vendored here; put it in `pdk/` or point `SG13G2_LIB` at it:

```
git clone --depth 1 https://github.com/IHP-GmbH/IHP-Open-PDK
cp IHP-Open-PDK/ihp-sg13g2/libs.ref/sg13g2_stdcell/lib/\
sg13g2_stdcell_typ_1p20V_25C.lib pdk/
```

`scripts/synth.py` also finds it under `$PDK_ROOT/ihp-sg13g2/...` if you already
have a PDK installed.

## Resizing the array

Every geometry parameter lives on the top module, and each one is checked at
elaboration inside the submodule that depends on it:

| parameter | default | meaning | constraints |
| --- | --- | --- | --- |
| `ROWS` | 4 | reduction depth resident in the array | 1 to 15 |
| `COLS` | 2 | output channels computed in parallel | 1 to 15 |
| `S_MAX` | 6 | activation vectors held on chip | 1 to 15 |
| `ACC_W` | 24 | accumulator and bias width | `PSUM_W` to 32 |
| `M_W` | 16 | requantization multiplier width | 4 to 24 |
| `SH_W` | 5 | requantization shift width | 1 to 6 |
| `MUL_ARCH` | 1 | 0 array, 1 Wallace, 2 Booth radix-4 | 0 to 2 |
| `ADD_ARCH` | 4 | 0 RCA, 1 Brent-Kung, 2 KS, 3 Sklansky, 4 Han-Carlson | 0 to 4 |

Change them in `src/tt_um_danieltyukov_int8_npu.sv`. `PSUM_W` is derived as
`16 + ceil(log2(ROWS+1))` and never needs setting by hand.

Then update the test configuration to match, because the tests check the
hardware against a model of the same shape:

```python
# test/test_npu.py, test/test_demo.py, test/test_trace.py
CFG = g.Cfg(rows=4, cols=2, s_max=6, acc_w=24, m_w=16, sh_w=5)
```

The identity block (`RDSEL` source 3, or opcode `0xF0`) reports `ROWS`, `COLS`,
`S_MAX`, `ACC_W` and `M_W`, so host software can discover the geometry instead of
hard-coding it. `test_identity_block` asserts it matches.

What the numbers do to the design:

- `ROWS` sets the reduction length per pass and the array fill latency
  (`ROWS + 1` cycles). Larger `ROWS` means fewer passes per layer.
- `COLS` sets how many output channels retire per cycle. Every extra column adds
  a bank of `S_MAX` accumulators, an accumulate adder, a bias register and a
  per-channel multiplier and shift.
- `S_MAX` sets how many samples can be in flight. The array is only fully
  populated when `S_COUNT >= ROWS + COLS - 1`, so if you want a cycle in which
  every PE is busy, keep `S_MAX >= ROWS + COLS - 1`. It is also the most
  expensive parameter: it multiplies both the activation buffer and the
  accumulator bank.
- Non-power-of-two `S_MAX` is supported. The sample count field is clamped
  rather than wrapped so buffer indices stay in range.

Measured areas for thirteen geometries are in `docs/synth/ppa.md`, and
`docs/img/area_scaling.png` plots them against the tile budgets.

## Retargeting the tile size

The criterion used here is cell area at or below 60 percent of the die area,
matching `PL_TARGET_DENSITY_PCT` in `src/config.json`. Tile die areas are not
a pitch multiplied out: each one is the `DIEAREA` of the corresponding
`tt_block_<tile>_pgvdd.def` floorplan template in `TinyTapeout/tt-support-tools`,
which is the DEF the hardening flow actually applies.

| tile | die | die area | budget at 60% |
| --- | --- | --- | --- |
| 1x1 | 202.08 x 154.98 um | 31318 um2 | 18791 um2 |
| 1x2 | 202.08 x 313.74 um | 63401 um2 | 38040 um2 |
| 2x2 | 419.52 x 313.74 um | 131620 um2 | 78972 um2 |
| 3x2 | 636.96 x 313.74 um | 199840 um2 | 119904 um2 |
| 4x2 | 854.40 x 313.74 um | 268059 um2 | 160836 um2 |
| 6x2 | 1289.28 x 313.74 um | 404499 um2 | 242699 um2 |
| 8x2 | 1724.16 x 313.74 um | 540938 um2 | 324563 um2 |
| 3x4 | 636.96 x 710.64 um | 452649 um2 | 271590 um2 |
| 4x4 | 854.40 x 710.64 um | 607171 um2 | 364302 um2 |
| 5x4 | 1071.84 x 710.64 um | 761692 um2 | 457015 um2 |
| 6x4 | 1289.28 x 710.64 um | 916215 um2 | 549729 um2 |
| 8x4 | 1724.16 x 710.64 um | 1225257 um2 | 735154 um2 |

Budget against the post-route area, not the Yosys area. On the shipped
configuration placement and routing add 2530 timing-repair buffers and 1474 hold
buffers, and cell area goes from 142403 um2 to 190566 um2, a factor of 1.34.
Sizing a tile from the synthesis number alone picks a tile too small: it says
4x2 for this design, where the routed netlist would sit at about 73% and above
the placement target.

Treat all of that as the shortlist, not the answer. Hardening is a 45 minute CI
job on a throwaway branch: change `tiles:` in `info.yaml`, push the branch, read
the numbers back with `scripts/harvest_pnr.py --out-dir docs/pnr/alt-<tile>`.
This project settled 6x2 by hardening 4x2, 6x2 and 8x2 and comparing signoff,
which is the only comparison that is not an extrapolation. All three passed, and
the estimate had ruled 4x2 out.

To move to a smaller tile, pick a geometry from the scaling table that fits, set
the parameters, then:

```
make synth                       # measured area for your configuration
```

and update three things: `tiles:` in `info.yaml`, the area sentence in
`README.md`, and `CLOCK_PERIOD` in `src/config.json` if your depth changed.
`scripts/run_ppa.py` prints the smallest tile for every geometry it measures,
against both the synthesis area and that area scaled by the measured routing
factor, so you do not have to do the arithmetic yourself. Then push and read the
real numbers back out of the `gds` workflow with
`scripts/harvest_pnr.py`, because one hardening run settles what the estimate
only brackets.

## Swapping in a different quantized layer

The hardware computes a tiled matrix multiply with requantization, so any
fully connected layer, and any convolution that has been lowered to one, fits.
The host is responsible for tiling.

Start from `scripts/train_demo.py`, which trains a small MLP in NumPy and
quantizes it, and `test/test_demo.py`, which executes it on the RTL. The pieces
you need to reproduce for your own layer:

1. **Symmetric per-channel weights.** `s_w[c] = max|W[:,c]| / 127`,
   `q_w = round(W / s_w)`. Zero point must be zero: the hardware computes a plain
   signed dot product.
2. **Bias in the accumulator domain.** `q_b[c] = round(b[c] / (s_x * s_w[c]))`.
3. **Fold the input zero point into the bias.** If the input tensor has a
   non-zero zero point, subtract `z_x * sum_r q_w[r][c]` from `q_b[c]`. The demo
   does this for layer 2, whose input is the ReLU output of layer 1 with its zero
   point at -128.
4. **Split each effective scale.** `M_eff[c] = s_x * s_w[c] / s_y`, then
   `golden.quantize_multiplier(M_eff[c])` gives `(M, shift)`. `M_eff` must be
   strictly between 0 and 1; if it is not, your output scale is too small.
5. **Tile the layer.** `K` inputs become `ceil(K/ROWS)` accumulating passes,
   `N` channels become `ceil(N/COLS)` groups. `K` must be a multiple of `ROWS`
   and `N` of `COLS`, or you pad the weights with zeros, which contribute nothing
   to the sum.
6. **Drive it.** Per channel group: `CFG`, `LD_BIAS`, `LD_QUANT`, `LD_POST`, then
   per pass `LD_W`, `LD_ACT`, `RUN` with `accumulate = (pass > 0)` and
   `requant = (pass == last)`. Read the results with `RDSEL` source 0.

`run_layer` in `test/test_demo.py` is 30 lines and does exactly that; it is the
reference host implementation.

## Adding a new adder architecture

`npu_adder` is one module with a `generate` branch per architecture. A new one
needs to fill in the group generate and propagate arrays, nothing else:

```systemverilog
end else if (ARCH == 5) begin : g_my_network
  // gnet and pnet are flat vectors indexed [stage*WIDTH + bit].
  // Stage 0 already holds the per-bit terms (with cin folded into bit 0).
  // Produce, for every bit i, the prefix over bits i..0 at stage STAGES.
  for (genvar s = 1; s <= LEVELS; s++) begin : g_lvl
    for (genvar i = 0; i < WIDTH; i++) begin : g_bit
      if (/* this node combines at this level */) begin : g_comb
        assign gnet[s*WIDTH+i] = gnet[(s-1)*WIDTH+i]
                               | (pnet[(s-1)*WIDTH+i] & gnet[(s-1)*WIDTH+j]);
        assign pnet[s*WIDTH+i] = pnet[(s-1)*WIDTH+i] & pnet[(s-1)*WIDTH+j];
      end else begin : g_keep
        assign gnet[s*WIDTH+i] = gnet[(s-1)*WIDTH+i];
        assign pnet[s*WIDTH+i] = pnet[(s-1)*WIDTH+i];
      end
    end
  end
  // Copy the last computed stage up to STAGES so post-processing finds it.
  for (genvar s = LEVELS + 1; s <= STAGES; s++) begin : g_fill
    assign gnet[s*WIDTH +: WIDTH] = gnet[(s-1)*WIDTH +: WIDTH];
    assign pnet[s*WIDTH +: WIDTH] = pnet[(s-1)*WIDTH +: WIDTH];
  end
end
```

The contract a new architecture must meet:

- Ports do not change: `a`, `b`, `cin`, `sum`, `cout`, parameters `WIDTH` and
  `ARCH`.
- Correct for any `WIDTH >= 1`, not only powers of two.
- Every bit of `gnet[STAGES*WIDTH +: WIDTH]` must be the complete prefix over
  bits `i..0`, because post-processing forms `sum[i] = p[i] ^ G[i-1]` and
  `cout = G[WIDTH-1]`.
- Purely combinational, no state.

Then widen the range check (`ARCH must be 0..4`), add the name to `ADDERS` in
`scripts/run_ppa.py` and `scripts/plot_ppa.py`, add it to `ADDER_NAMES` in
`test/test_arith.py`, and extend the `generate` loops in `test/tb_arith.v` and
`test/tb_arith.sv` from 5 to 6. The equivalence test then compares your network
against the other five and against Python integers at four widths.

## Adding a new multiplier architecture

Partial-product generation and reduction are separate modules, so a new
architecture is usually one new `npu_pp_*` module plus a branch in `npu_mult`:

```systemverilog
module npu_pp_mine #(parameter int A_W = 8, parameter int B_W = 8) (
    input  wire [A_W-1:0]              a,
    input  wire [B_W-1:0]              b,
    output wire [NROWS*(A_W+B_W)-1:0]  rows_o   // NROWS packed rows
);
```

The contract:

- Every row is `A_W+B_W` bits wide, and the sum of all rows modulo
  `2^(A_W+B_W)` must equal `a*b` interpreted as signed. Carries out of the top
  bit may be dropped, which is what lets sign handling be free.
- Rows may be constants; Yosys folds them away.
- `A_W, B_W >= 2`.

Then add a branch to `npu_mult` that instantiates your generator and either
`npu_csa_reduce` (tree) or the linear chain, widen the `MUL_ARCH` range check,
and register the name in `scripts/run_ppa.py`, `scripts/plot_ppa.py` and
`test/test_arith.py`. `test_multiplier_exhaustive` will then run all 65536
signed operand pairs through it.

## Re-running the PPA comparison

```
make ppa        # ~150 Yosys runs, writes docs/synth/ppa.{md,json} plus one
                # log and one JSON per variant
make images     # redraws every chart from those JSONs
```

`scripts/synth.py` is also usable directly for one-off measurements:

```
.venv/bin/python scripts/synth.py --top npu_adder --param WIDTH=32 \
    --param ARCH=2 --name adder_w32_ks --effort fast
```

`--effort fast` maps the netlist as written, which is what makes architectures
comparable. `--effort full` runs ABC's default resynthesis, which is what the
hardening flow does and which partly erases the differences; both are reported
in `docs/synth/ppa.md` so the erasure is visible rather than hidden.

## Changing the requantization precision

`M_W` is the knob. It sets the worst-case relative error of a representable
scale (`2^-M_W`), the Booth step count (`ceil((M_W+1)/2)`), and the width of the
requantizer register. Measured cost is in `docs/img/requant_width.png`.

If you need bit-exact TFLite behaviour rather than exact arithmetic on a
narrower multiplier, `M_W = 24` gets the scale error to `6e-8`, or you can
replace the serial loop with a parallel multiplier and a barrel shifter, which
costs roughly 20000 um2 at these widths (about four processing elements) and
brings the requantizer down to a couple of cycles per output.

## Going faster

The critical path is a multiply and an add in one cycle. Two ways to shorten it:

- **Pipeline the PE.** Register the product, then add in the next cycle. This
  changes the array timing: each row adds one more cycle of latency, so the
  valid/index pipeline in `npu_core` (`vpipe`, `ipipe`) needs `2*ROWS + COLS`
  stages instead of `ROWS + COLS`, and `col_valid[c]` moves to
  `vpipe[2*ROWS + c]`. `test_latency_matches_model` will tell you immediately if
  you get it wrong.
- **Change `ADD_ARCH`.** The partial-sum adder is on the critical path. The
  measured depths in `docs/synth/ppa.md` show what each network buys.

Both are cheaper than raising the clock by other means, and the PPA table tells
you what each costs before you commit.

## Things that will bite you

- **Bulk loads must be full length.** Every load is a shift chain, so a short
  `LD_W`, `LD_ACT`, `LD_BIAS` or `LD_QUANT` leaves the data misaligned. The
  interface reports the abandoned frame in the error code, but it cannot fix the
  alignment. Send exactly `ROWS*COLS`, `S_MAX*ROWS`, `COLS*3` and
  `COLS*(ceil(M_W/8)+1)` bytes.
- **`LD_ACT` is always `S_MAX*ROWS` bytes**, even when `S_COUNT` is smaller.
  Pad; the extra samples are simply not streamed.
- **Writes during `busy` are dropped**, not queued, and set error code 2. Poll
  `busy` on `uio_out[3]`.
- **The effective scale must be below 1.** `quantize_multiplier` raises if it is
  not, which is nearly always a sign that the output scale was calibrated wrong.
- **Unused rows and columns need zero weights**, not stale ones: a stale weight
  still contributes a product.
- **Keep the test `Cfg` in step with the RTL parameters.** The suite compares
  against a model of a specific shape; a mismatch shows up as a flood of
  failures rather than a clear message.
