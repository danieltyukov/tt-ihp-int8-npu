# Design notes

How the accelerator works and why it is built this way. The README is the
datasheet; this file is the reasoning behind it.

## Contents

- [Dataflow choice](#dataflow-choice)
- [PE microarchitecture and array timing](#pe-microarchitecture-and-array-timing)
- [Accumulator bank and multi-pass reduction](#accumulator-bank-and-multi-pass-reduction)
- [Requantization mathematics](#requantization-mathematics)
- [Why the requantizer is serial](#why-the-requantizer-is-serial)
- [Activation functions](#activation-functions)
- [Adder architectures and their cost models](#adder-architectures-and-their-cost-models)
- [Multiplier architectures and their cost models](#multiplier-architectures-and-their-cost-models)
- [Area budget](#area-budget)
- [Clock target](#clock-target)
- [Verification plan](#verification-plan)

## Dataflow choice

An INT8 layer is `Y = act(requant(X W + b))` with `X` of shape (S, K) and `W` of
shape (K, N). There are three classic ways to map that onto a 2-D array of
multiply-accumulate cells, and they differ in what stays still:

| dataflow | resident | streams | reuse |
| --- | --- | --- | --- |
| weight-stationary | one weight per PE | activations and partial sums | each weight is reused by every sample |
| output-stationary | one partial sum per PE | weights and activations | each partial sum is accumulated in place |
| row-stationary | a mix | all three | balances all three reuses |

Weight-stationary is the right choice here because of the pin budget, not
because of arithmetic. The design has eight bits of input bandwidth: one byte
per cycle, and no SRAM anywhere on the tile. Weights are the largest operand set
(`ROWS*COLS` bytes per tile) and the one that is reused most, so making them
resident means the host pays for them once per tile and then streams only
activations. An output-stationary array would have to re-send weights for every
sample, which the pins cannot sustain.

The consequence is the one every systolic array lives with: the array is only
busy when several samples are in flight, and several samples in flight produce
`COLS` results per cycle. That is what sizes the accumulator bank, and it is why
`S_MAX` costs as much area as it does.

## PE microarchitecture and array timing

Each PE holds three registers and does one signed multiply and one add per cycle:

```
              w_reg  (8 bit, resident)
                |
  a_in -> [a_reg] -> a_out          activations move east
                |
      psum_in -(+)- [psum_reg] -> psum_out    partial sums move south
```

with the update

```
a_reg    <= a_in
psum_reg <= psum_in + w_reg * a_reg
```

There is deliberately no compute enable. When the array is idle the row inputs
are driven to zero, so the products are zero and the partial-sum chain flushes
itself. That removes `ROWS*COLS*(PSUM_W+8)` enable multiplexers, which at 18 um2
each is over 6000 um2 in the shipped configuration.

The one-cycle offset between `a_reg` and the product is what makes the wavefront
line up. Take a sample that enters row 0 in cycle `s`, with row `r` fed the same
sample in cycle `s+r` (the diagonal skew, which the activation buffer does by
addressing, not by shift registers):

| event | cycle |
| --- | --- |
| `a_reg(r, c)` latches the sample | `s + r + c` |
| `PE(r, c)` adds its product | `s + r + c + 1` |
| `psum_reg(r, c)` holds the sum of rows 0..r | end of `s + r + c + 1` |
| column `c` result complete at the array bottom | end of `s + ROWS + c` |
| accumulator bank writes it | end of `s + ROWS + c + 1` |

`psum_reg(r-1, c)` is written at the end of cycle `s + r + c` and read during
cycle `s + r + c + 1`, exactly when `PE(r, c)` needs it: the skew and the
register delay cancel. Results therefore leave the array one column per cycle,
staggered by one cycle per column, and the array phase is

```
ARRAY cycles = S_COUNT + ROWS + COLS
```

of which `ROWS + 1` are fill and `COLS - 1` are drain. Every PE performs exactly
`S_COUNT` MACs, one per cycle with no gaps, and all `ROWS*COLS` PEs hold a live
activation simultaneously for `S_COUNT - ROWS - COLS + 2` cycles. With the
shipped `ROWS=4, COLS=2, S_MAX=6` that is two fully populated cycles;
`test_sustained_throughput` reads every PE's activation register every cycle and
asserts all of it, and `docs/img/pipeline_timing.svg` is drawn from that trace.

## Accumulator bank and multi-pass reduction

The bank is `COLS` independent banks of `S_MAX` entries, `ACC_W` bits each. Up
to `COLS` results retire in the same cycle (different columns, different
samples), so each bank owns its own accumulate adder and its own saturation
logic; a single shared adder would need `COLS` write ports on one register file.

Each write is

```
acc[c][s] <= saturate( (first_pass ? bias[c] : acc[c][s]) + column_sum )
```

Initialising the accumulator with the bias, rather than adding the bias later,
is what a TFLite kernel does and it costs nothing: the multiplexer in front of
the adder is cheaper than a separate bias adder in the requantization path, and
it makes the accumulator value the exact quantity the requantizer needs.

A layer with `K > ROWS` inputs is `ceil(K / ROWS)` runs against the same
accumulator: `RUN` with `arg=0` for the first (bias), `arg=1` for the middle
ones, `arg=3` for the last (accumulate and requantize). The host reloads the
weight tile between passes, which is host-bandwidth bound, not array bound: a
tile is `ROWS*COLS` bytes and the pins carry one byte per cycle. A shadow weight
bank would let the next tile load during the current pass, at the cost of
`ROWS*COLS*8` extra flip-flops; `docs/ADAPTING.md` describes where to add it.

`ACC_W = 24` bounds the useful reduction length. With INT8 operands the largest
single product magnitude is `128*127 = 16256`, so `P` passes of `ROWS` rows can
reach `P*ROWS*16384`. At `ACC_W = 24` and `ROWS = 4` that allows `P <= 128`, so
`K` up to 512 inputs before the accumulator can saturate, and saturation is
detected rather than silently wrapped: the sum is computed one bit wider and the
two sign bits are compared, then the value is clamped and the sticky `ovf` flag
is raised.

Because no representable input can overflow a 24-bit accumulator at these array
sizes, the result is bit-identical to an INT32 accumulator for every input the
hardware accepts; the requantization path itself is 26 bits wide. That is the
sense in which this is an "INT32 accumulator" design: the arithmetic is exact,
and the width is the smallest that provably cannot lose a bit.

## Requantization mathematics

Start from the affine quantization of a real value `v`:

```
v = s * (q - z)
```

with scale `s` and zero point `z`. For a layer with input scale `s_x`, weight
scale `s_w` (per output channel) and output scale `s_y`, and with symmetric
weights (`z_w = 0`):

```
y_real[c] = sum_r (s_x (q_x[r] - z_x)) (s_w[c] q_w[r][c]) + b[c]
          = s_x s_w[c] ( sum_r q_x[r] q_w[r][c]
                         - z_x sum_r q_w[r][c]
                         + b[c] / (s_x s_w[c]) )
```

Two things fall out of that. First, the hardware only ever needs the plain
signed dot product `sum_r q_x q_w`: the input zero point contributes
`- z_x sum_r q_w[r][c]`, a per-channel constant that is folded into the bias
offline, which is exactly what a quantizing compiler does. Second, the bias
belongs in the accumulator domain, `q_b[c] = round(b[c] / (s_x s_w[c]))`.

Quantizing the output, `q_y = round(y_real / s_y) + z_y`, gives the requantization
the hardware performs:

```
q_y[c] = saturate_int8( round( acc[c] * M_eff[c] ) + z_y )
M_eff[c] = s_x * s_w[c] / s_y
```

`M_eff` is a positive real number smaller than one for any sane layer (the
accumulator has far more range than the output), so it is represented as a
fixed-point fraction

```
M_eff = M / 2^(M_W + shift),   M in [2^(M_W-1), 2^M_W),   shift >= 0
```

`M` is normalized to the top bit so all `M_W` bits carry information; `shift` is
the binary exponent. `golden.quantize_multiplier` does that split and raises if
the scale is outside `(0, 1)`.

The rounding rule is the one gemmlowp uses in `RoundingDivideByPOT`, which TFLite
inherits: round to nearest, ties away from zero. With `n = M_W + shift` and
`p = acc * M`:

```
mask      = 2^n - 1
remainder = p & mask
threshold = (mask >> 1) + (p < 0 ? 1 : 0)
result    = (p >> n) + (remainder > threshold)
```

The hardware computes the same thing from two bits it captures as they leave the
bottom of the shifting register: the round bit `R` (bit `n-1` of `p`) and the
sticky bit `S` (the OR of bits `n-2..0`).

```
round_up = (p >= 0) ? R : (R and S)
result   = (p >>> n) + round_up
```

For `p >= 0`, `remainder > 2^(n-1) - 1` is exactly `R = 1`, so a tie rounds up,
away from zero. For `p < 0`, `remainder > 2^(n-1)` is exactly `R and S`, so a tie
does not round up, and because `>>>` floors, that also moves away from zero. An
equivalent add-then-shift form, `(p + 2^(n-1) - (p < 0)) >>> n`, is checked
against the bit form over 32768 values in `test_rounding_ties`.

### Worked example

`acc = 16` (the directed test's column 0), `M = 32768`, `shift = 0`, `M_W = 16`,
`z_y = 0`:

```
n = 16 + 0 = 16
p = 16 * 32768 = 524288 = 0x80000
p >> 16 = 8,  R = bit 15 of p = 0,  S = OR bits 14..0 = 0
round_up = 0
q = saturate_int8(8 + 0) = 8
```

and for the tie case `acc = 3`, `M = 32768` (an effective scale of exactly 1/2):

```
p = 98304 = 0x18000,  p >> 16 = 1,  R = bit 15 = 1,  S = 0
p >= 0 so round_up = R = 1
q = 1 + 1 = 2          (1.5 rounds away from zero)
```

and its negative counterpart `acc = -3`:

```
p = -98304,  p >>> 16 = -2 (floor),  R = 1, S = 0
p < 0 so round_up = R and S = 0
q = -2                 (-1.5 rounds away from zero)
```

### Difference from TFLite

TFLite splits the multiply into `SaturatingRoundingDoublingHighMul` with a Q0.31
multiplier followed by `RoundingDivideByPOT`, so it rounds twice and represents
the scale to 31 bits. This design uses one Q0.`M_W` multiplier and rounds once.
The arithmetic is exact for the multiplier it is given; the difference is the
precision of the scale itself, bounded by a relative error of `2^-M_W`
(`1.5e-5` at the default `M_W = 16`). The area cost of that choice is measured
in `docs/synth/ppa.md` and plotted in `docs/img/requant_width.png`: going from
`M_W = 16` to 24 buys 8 more bits of scale precision for about 1400 um2 and four
more cycles per output. On the demo network the INT8 accuracy is identical to
float32, so 16 bits is not the limiting factor there.

## Why the requantizer is serial

A parallel requantizer would need a 24x16 multiplier. Measured on this PDK an
8x8 signed multiplier is about 3300 um2 (full-effort mapping), and partial
product count scales with the operand product, so 24x16 lands near 20000 um2:
four processing elements, for a unit that runs once per output element rather
than once per MAC. The array produces `ROWS` MACs per output element, so a
parallel requantizer is 4x over-provisioned by construction at `ROWS = 4`.

So the multiply is serial: radix-4 Booth, two multiplier bits per cycle, over
`NDIG = ceil((M_W+1)/2)` steps. The register is `{A, Q}` with the multiplier
loaded into `Q`; each step adds `d_k * acc` into `A` and shifts the pair right by
two, which consumes one Booth digit and moves two product bits into `Q`.

One subtlety is worth recording because it was a real bug during development.
Digit `NDIG-1` inspects multiplier bit `2*NDIG-1`, which sits at field position
`2*NDIG` once the `b[-1]` pad is prepended, so `Q` needs `2*NDIG+1` bits, not
`2*NDIG`. With `Q` one bit too narrow the top digit reads a product bit that has
already been shifted in and the sign of the result is wrong for some multipliers.
The extra bit means the loop leaves `2*acc*M` in the register rather than
`acc*M`, which the rounding shift absorbs by shifting one extra bit: the bit
pattern is identical, so the round and sticky bits are unaffected.

The rounding shift then reuses the same register, four bits per cycle plus a
one-bit fine step, tracking `R` and `S` as bits leave the bottom. A barrel
shifter for a 45-bit register would be about 4000 um2; the serial shift is a
2:1 multiplexer per bit.

Finally the same adder that accumulates Booth digits performs the zero-point
addition, with the rounding increment folded into its carry-in, so the whole
requantizer contains exactly one adder.

| stage | cycles |
| --- | --- |
| setup | 1 |
| Booth multiply | `NDIG` = 9 at `M_W = 16` |
| rounding shift | `floor(n/4) + n mod 4`, `n = M_W + shift + 1` |
| round, zero point, saturate | 1 |
| done | 1 |

which is 17 cycles per output at `shift = 0` and 25 at the maximum shift.
`test_latency_matches_model` asserts the measured busy time equals this model
for every sample count and several shifts.

## Activation functions

In integer inference "zero" is the output zero point, so every activation is
expressed relative to `zp`:

| select | function |
| --- | --- |
| 0 identity | `clamp(q, clamp_lo, clamp_hi)` |
| 1 ReLU | `clamp(q, zp, 127)` |
| 2 ReLU6 | `clamp(q, zp, clamp_hi)` where `clamp_hi = quantize(6.0)` |
| 3 leaky ReLU | `clamp(q >= zp ? q : zp + ((q - zp) >>> k), clamp_lo, clamp_hi)` |

ReLU6 has to use a programmable bound because the integer representation of 6.0
depends on the output scale, which only the host knows. The leaky slope is
`2^-k` for a 3-bit `k`, using an arithmetic shift, so `k = 3` gives the common
0.125 and `k = 0` degenerates to the identity. Everything is one 9-bit subtract,
one shift and two comparators; the whole block is under 400 um2.

## Adder architectures and their cost models

All five share one skeleton: pre-process `(a, b, cin)` into per-bit generate and
propagate, run a prefix network for the group generate `G[i]`, then
`sum[i] = p[i] ^ G[i-1]`. The prefix operator
`(G, P) o (G', P') = (G | (P & G'), P & P')` is associative, which is why every
network gives bit-identical results and why the equivalence test is meaningful
rather than a formality.

| ARCH | network | prefix cells | levels | wiring |
| --- | --- | --- | --- | --- |
| 0 | ripple carry | `W - 1` | `W` | nearest neighbour |
| 1 | Brent-Kung | `2W - log2(W) - 2` | `2 log2(W) - 1` | sparse, short |
| 2 | Kogge-Stone | `W log2(W) - W + 1` | `log2(W)` | dense, long |
| 3 | Sklansky | `(W/2) log2(W)` | `log2(W)` | high fanout at block midpoints |
| 4 | Han-Carlson | `(W/2) log2(W) + W/2` | `log2(W) + 1` | Kogge-Stone on odd bits only |

Brent-Kung trades depth for cells by doing a forward reduction and a backward
expansion; Kogge-Stone spends cells and wire to reach minimum depth; Sklansky
reaches the same depth with half the cells but concentrates fanout; Han-Carlson
runs Kogge-Stone over the odd positions and recombines, giving Kogge-Stone's
depth plus one for about half its wiring. The measured numbers in
`docs/synth/ppa.md` follow exactly this ordering, which is a useful check that
the generators really do describe the networks they claim to.

Every network here is written for arbitrary `W`, not just powers of two, with
guards on the index arithmetic. The 19, 25, 26 and 42-bit widths in the test
bench are the widths the accelerator actually instantiates.

## Multiplier architectures and their cost models

Signed multiplication is done modulo `2^(A_W+B_W)`. The true product of two
signed operands is representable in that many bits, so carries out of the top
can be dropped and sign handling collapses into which partial-product rows get
complemented.

**Baugh-Wooley** rewrites the two negative cross terms of
`A*B = A'B' - a_{m-1} 2^{m-1} B' - b_{n-1} 2^{n-1} A' + a_{m-1} b_{n-1} 2^{m+n-2}`
using `-X = ~X - 2^k + 1`, which turns them into inverted AND terms plus a
compile-time constant row:

```
A*B = sum_{i<m-1, j<n-1} a_i b_j 2^(i+j)
    + sum_{j<n-1} ~(a_{m-1} b_j) 2^(m-1+j)
    + sum_{i<m-1} ~(a_i b_{n-1}) 2^(n-1+i)
    + a_{m-1} b_{n-1} 2^(m+n-2)
    + 2^(m+n-1) + 2^(m-1) + 2^(n-1)        (mod 2^(m+n))
```

That is `B_W` rows plus one constant row, with no sign-extension logic at all.

**Radix-4 Booth** recodes the multiplier into `ceil((B_W+1)/2)` digits in
`{-2,-1,0,1,2}`, roughly halving the row count. Negative digits are produced by
complementing the magnitude and injecting a `+1` at bit `2k`; because those
injection points are all at even positions they pack into one extra row instead
of one row each, which is what makes Booth actually cheaper here rather than
just differently shaped.

Reduction is either a linear carry-save chain (the classic array multiplier,
depth `O(rows)`) or a Wallace tree (`npu_csa_reduce`, which compresses every
group of three rows with a row of full adders and recurses, depth
`O(log rows)`). Both end in one carry-propagate adder whose architecture is a
separate parameter, so the multiplier and adder choices are orthogonal and are
measured that way.

| MUL_ARCH | rows for 8x8 | reduction | expected |
| --- | --- | --- | --- |
| 0 | 9 | linear carry-save | smallest wiring, deepest |
| 1 | 9 | Wallace tree | shallower for a few more cells |
| 2 | 6 | Wallace tree | fewest rows, extra recoding logic |

## Area budget

Registers dominate. On this PDK a resettable flip-flop is 48.99 um2 and a 2:1
multiplexer is 18.14 um2, so a flip-flop with an enable costs about the same as
a full adder bit. The shipped configuration is 1126 flip-flops out of 11278
cells, and the storage that exists purely to keep the array fed (activation
buffer, accumulator bank, result registers) is 40 percent of them. That is the
real cost of a systolic array on a tile with no SRAM, and it is why the geometry
sweep in `docs/synth/ppa.md` shows area growing faster with `S_MAX` than
intuition suggests.

Design decisions that came out of measuring rather than guessing:

- No compute enable in the PE (self-flushing array): saves about 6000 um2.
- Bulk loads are plain shift chains with no address decoders. A shift chain
  needs no multiplexer at all when every element shifts, so the weight,
  activation, bias and quantization loads cost one enable per byte lane instead
  of a decoder plus per-entry enables.
- The accumulator bank is built from per-entry registers with explicit enables
  rather than a variable part-select write into one flat vector, which Yosys
  turns into a barrel-shifted write mask.
- Asynchronous reset throughout, because the only flip-flops in sg13g2 have an
  asynchronous reset. A synchronous reset would add a multiplexer per bit, about
  20000 um2 across the design.

## Clock target

`clock_hz` is 40 MHz (25 ns). No static timing analysis tool is available in
this environment, so the number is derived from the measured logic depth and the
liberty's own delay tables, with margin, and it must be confirmed by the STA in
the hardening flow.

The critical path is one 8x8 signed multiply followed by a 19-bit add inside a
PE, or the accumulator path (sign extend, 25-bit add, saturate multiplexer);
Yosys reports 36 mapped cells for the deepest path in the shipped
configuration. Taking the mean `cell_rise` of eight representative cells from
`sg13g2_stdcell_typ_1p20V_25C.lib`:

| load per stage | mean cell delay | 36 stages | implied clock |
| --- | --- | --- | --- |
| 20 fF (a few fanouts) | 0.173 ns | 6.2 ns | 160 MHz |
| 150 fF (middle of the library's load grid) | 0.457 ns | 16.6 ns | 60 MHz |
| chosen target | 0.694 ns | 25.0 ns | 40 MHz |

40 MHz leaves 1.5x margin over the pessimistic column and 4x over the
optimistic one, which is where a design with a multiply-add in a single cycle
should sit before real extraction. Halving `M_W` or splitting the PE into
multiply and accumulate stages are the two obvious ways to go faster; both are
described in `docs/ADAPTING.md`.

## Verification plan

The independent reference is `test/golden.py`, written from the datasheet rather
than from the RTL, in plain Python integers so overflow behaviour is never in
doubt. Every hardware result is compared against it.

| area | test | what it proves |
| --- | --- | --- |
| arithmetic | `tb_arith.sv` (iverilog) | all 65536 signed 8x8 operand pairs against every multiplier architecture, plus randomized adder sweeps at four widths |
| arithmetic | `test_arith.py` (cocotb) | every adder and multiplier architecture is bit-identical to the others and to Python integers |
| signedness | `test_int8_minimum` | `-128`, the asymmetric INT8 minimum, in weights and activations, including `-128 * -128` |
| range | `test_extreme_tensors` | all-zero and all-rail tensors |
| saturation | `test_saturation_both_rails` | both INT8 rails and the sticky `sat` flag, and that `CLR` clears it |
| overflow | `test_accumulator_overflow` | accumulator saturation in both directions and the sticky `ovf` flag |
| rounding | `test_rounding_ties` | exact `.5` boundaries at both signs, and the two formulations of the rounding rule against each other |
| scale | `test_scale_extremes` | `M = 0`, `M = 2^M_W - 1`, maximum shift, `M = 1`, crossed with zero-point extremes |
| activations | `test_activations` | all four functions, several zero points and clamp windows |
| correctness | `test_random_sweep` | randomized weights, activations, biases, scales, zero points and activation modes, checking both the INT8 outputs and the raw accumulators |
| reduction | `test_random_multipass` | `K` up to 16 through accumulating passes |
| throughput | `test_sustained_throughput` | per-PE activation registers every cycle: one MAC per PE per cycle, diagonal wavefront, fully populated cycles |
| latency | `test_latency_matches_model` | measured busy cycles equal the documented fill, compute, drain and requantization model |
| protocol | `test_protocol_*` | unknown opcode, payload overrun, abandoned frame, write while busy, resynchronization, and the error codes |
| readback | `test_readback_sources` | all four readback sources, auto-increment, and out-of-range behaviour |
| reset | `test_reset_defines_all_outputs`, `test_soft_reset_midrun`, `test_hard_reset_midrun` | outputs defined and deterministic after reset, mid-run reset returns to a known state |
| end to end | `test_demo.py` | a quantized 16-12-10 MLP executed layer by layer on the RTL, every INT8 value bit-exact against the NumPy INT8 model |

`make trace` re-runs the RTL to regenerate the data behind the figures, so the
pipeline diagram, the dataflow animation and the requantization error plot are
always the hardware's own behaviour rather than an illustration of it.
