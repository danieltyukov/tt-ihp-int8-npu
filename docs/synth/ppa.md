# PPA comparison

Measured with Yosys 0.33 against `sg13g2_stdcell_typ_1p20V_25C.lib` (IHP open PDK). Area is the
sum of standard-cell areas in square micrometres; depth is the
longest topological path through mapped cells, which is the
critical-path proxy used throughout this project.

Two mapping efforts are reported:

- **fast**: ABC maps the netlist as written, so the architecture
  a generator describes is what gets measured.
- **full**: ABC's default script with `dc2` and `&dch`
  resynthesis, which is what the Tiny Tapeout hardening flow runs.

## Adders, fast mapping

### 19-bit

| architecture | cells | area (um2) | depth | note          |
|--------------|-------|------------|-------|---------------|
| ripple-carry | 86    | 929.0      | 20    | smallest      |
| Brent-Kung   | 118   | 1210.2     | 11    |               |
| Kogge-Stone  | 217   | 2010.4     | 9     |               |
| Sklansky     | 127   | 1268.3     | 8     | shortest path |
| Han-Carlson  | 145   | 1406.2     | 10    |               |

### 25-bit

| architecture | cells | area (um2) | depth | note          |
|--------------|-------|------------|-------|---------------|
| ripple-carry | 113   | 1222.9     | 26    | smallest      |
| Brent-Kung   | 162   | 1614.8     | 11    |               |
| Kogge-Stone  | 341   | 3062.7     | 9     |               |
| Sklansky     | 183   | 1796.3     | 8     | shortest path |
| Han-Carlson  | 231   | 2148.2     | 11    |               |

### 26-bit

| architecture | cells | area (um2) | depth | note          |
|--------------|-------|------------|-------|---------------|
| ripple-carry | 159   | 1518.6     | 28    | smallest      |
| Brent-Kung   | 169   | 1685.6     | 11    |               |
| Kogge-Stone  | 383   | 3469.1     | 10    |               |
| Sklansky     | 190   | 1888.8     | 8     | shortest path |
| Han-Carlson  | 260   | 2444.0     | 11    |               |

### 42-bit

| architecture | cells | area (um2) | depth | note          |
|--------------|-------|------------|-------|---------------|
| ripple-carry | 255   | 2447.6     | 44    | smallest      |
| Brent-Kung   | 288   | 2826.8     | 14    |               |
| Kogge-Stone  | 584   | 5419.6     | 10    | shortest path |
| Sklansky     | 367   | 3474.6     | 12    |               |
| Han-Carlson  | 448   | 4144.1     | 12    |               |

## Adders, full mapping

### 19-bit

| architecture | cells | area (um2) | depth | note          |
|--------------|-------|------------|-------|---------------|
| ripple-carry | 95    | 990.7      | 20    |               |
| Brent-Kung   | 90    | 948.9      | 20    | smallest      |
| Kogge-Stone  | 102   | 1099.3     | 18    | shortest path |
| Sklansky     | 93    | 981.6      | 20    |               |
| Han-Carlson  | 90    | 965.3      | 20    |               |

### 25-bit

| architecture | cells | area (um2) | depth | note          |
|--------------|-------|------------|-------|---------------|
| ripple-carry | 125   | 1300.9     | 26    |               |
| Brent-Kung   | 117   | 1275.5     | 26    | smallest      |
| Kogge-Stone  | 125   | 1300.9     | 26    |               |
| Sklansky     | 132   | 1373.4     | 24    |               |
| Han-Carlson  | 141   | 1475.1     | 17    | shortest path |

### 26-bit

| architecture | cells | area (um2) | depth | note          |
|--------------|-------|------------|-------|---------------|
| ripple-carry | 130   | 1344.5     | 27    |               |
| Brent-Kung   | 130   | 1344.5     | 27    |               |
| Kogge-Stone  | 130   | 1339.0     | 27    | smallest      |
| Sklansky     | 140   | 1433.3     | 25    |               |
| Han-Carlson  | 140   | 1478.7     | 19    | shortest path |

### 42-bit

| architecture | cells | area (um2) | depth | note          |
|--------------|-------|------------|-------|---------------|
| ripple-carry | 210   | 2166.4     | 43    | smallest      |
| Brent-Kung   | 210   | 2177.3     | 43    |               |
| Kogge-Stone  | 210   | 2166.4     | 43    | smallest      |
| Sklansky     | 214   | 2222.6     | 43    |               |
| Han-Carlson  | 212   | 2358.7     | 27    | shortest path |

## Signed 8x8 multipliers, fast mapping

| architecture            | cells | area (um2) | depth | note                    |
|-------------------------|-------|------------|-------|-------------------------|
| Baugh-Wooley array      | 689   | 5913.5     | 24    |                         |
| Baugh-Wooley + Wallace  | 674   | 5839.8     | 20    | smallest, shortest path |
| Booth radix-4 + Wallace | 786   | 6956.0     | 23    |                         |

Wallace tree with each final carry-propagate adder:

| final adder  | cells | area (um2) | depth |
|--------------|-------|------------|-------|
| ripple-carry | 534   | 4813.0     | 23    |
| Brent-Kung   | 675   | 5999.4     | 20    |
| Kogge-Stone  | 677   | 5945.1     | 19    |
| Sklansky     | 675   | 5954.1     | 18    |
| Han-Carlson  | 674   | 5839.8     | 20    |

## Signed 8x8 multipliers, full mapping

| architecture            | cells | area (um2) | depth | note          |
|-------------------------|-------|------------|-------|---------------|
| Baugh-Wooley array      | 339   | 3488.8     | 21    |               |
| Baugh-Wooley + Wallace  | 324   | 3374.2     | 18    | shortest path |
| Booth radix-4 + Wallace | 302   | 3232.3     | 20    | smallest      |

Wallace tree with each final carry-propagate adder:

| final adder  | cells | area (um2) | depth |
|--------------|-------|------------|-------|
| ripple-carry | 331   | 3428.7     | 18    |
| Brent-Kung   | 340   | 3501.3     | 18    |
| Kogge-Stone  | 337   | 3437.8     | 18    |
| Sklansky     | 339   | 3477.6     | 15    |
| Han-Carlson  | 324   | 3374.2     | 18    |

## Block breakdown, shipped configuration

ROWS=4, COLS=2, S_MAX=6, ACC_W=24, M_W=16, Wallace multipliers,
Han-Carlson adders.

| module     | cells | area (um2) | flops | depth |
|------------|-------|------------|-------|-------|
| pe         | 472   | 6284.6     | 35    | 25    |
| array      | 3492  | 47108.6    | 274   | 25    |
| requant    | 693   | 9371.5     | 69    | 29    |
| activation | 223   | 2358.5     | 0     | 23    |
| host_if    | 192   | 3644.9     | 44    | 8     |
| core       | 8089  | 141979.7   | 1137  | 36    |

## Requantization multiplier width

M_W sets how precisely a float scale can be represented
(relative error below 2^-M_W) and is the main precision knob.

| M_W | cells | area (um2) | flops | Booth steps |
|-----|-------|------------|-------|-------------|
| 8   | 634   | 8499.1     | 60    | 5           |
| 12  | 664   | 8896.3     | 64    | 7           |
| 16  | 693   | 9371.5     | 69    | 9           |
| 20  | 718   | 9786.9     | 73    | 11          |
| 24  | 740   | 10211.8    | 77    | 13          |

## Array geometry scaling

Full top-level area for each geometry, with the smallest Tiny
Tapeout tile whose die area is at least the measured area
divided by 60% (the `PL_TARGET_DENSITY_PCT`
in `src/config.json`).

| array | S_MAX | MACs/cycle | cells | area (um2) | flops | depth | smallest tile | density |
|-------|-------|------------|-------|------------|-------|-------|---------------|---------|
| 2x2   | 2     | 4          | 3918  | 65508      | 554   | 35    | 2x2           | 49.8%   |
| 2x2   | 4     | 4          | 4705  | 83687      | 723   | 35    | 3x2           | 41.9%   |
| 4x2   | 2     | 8          | 5899  | 92995      | 733   | 32    | 3x2           | 46.5%   |
| 4x2   | 4     | 8          | 6718  | 114070     | 935   | 29    | 3x2           | 57.1%   |
| 2x2   | 8     | 4          | 6604  | 126700     | 1051  | 38    | 4x2           | 47.3%   |
| 4x2   | 5     | 8          | 7785  | 132631     | 1041  | 34    | 4x2           | 49.5%   |
| 4x2   | 6     | 8          | 8118  | 142403     | 1137  | 35    | 4x2           | 53.1%   |
| 2x4   | 4     | 8          | 7966  | 145147     | 1215  | 29    | 4x2           | 54.1%   |
| 6x2   | 4     | 12         | 8896  | 145372     | 1145  | 29    | 4x2           | 54.2%   |
| 8x2   | 2     | 16         | 9826  | 148825     | 1099  | 31    | 4x2           | 55.5%   |
| 4x2   | 8     | 8          | 8788  | 162109     | 1329  | 35    | 6x2           | 40.1%   |
| 4x4   | 2     | 16         | 10665 | 166112     | 1237  | 29    | 6x2           | 41.1%   |
| 8x2   | 4     | 16         | 10751 | 174925     | 1369  | 29    | 6x2           | 43.2%   |
| 4x4   | 4     | 16         | 11778 | 199360     | 1569  | 29    | 6x2           | 49.3%   |

## Shipped configuration

- geometry: ROWS=4, COLS=2, S_MAX=6
- cells: 8118
- registers: 1137
- cell area: 142403.1 um2
- logic depth: 35 mapped cells
- tile: 4x2 (268059 um2 die, 53.1% cell density)
- inferred latches: none
- unmapped cells: none

