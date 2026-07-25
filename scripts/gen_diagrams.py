#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
# SPDX-License-Identifier: Apache-2.0
"""Emit the hand-written SVG diagrams.

  docs/img/architecture.svg     datapath: host interface, activation buffer with
                                its diagonal skew, the weight-stationary PE
                                array with one PE opened up, the accumulator
                                bank, the serial requantizer and the readback
                                path
  docs/img/pipeline_timing.svg  fill, steady state and drain with the cycle
                                numbers taken from docs/data/dataflow.json
  docs/img/protocol_timing.svg  pin-level weight load and inference trigger,
                                from docs/data/protocol.json

The two timing diagrams are drawn from real simulation traces, so they cannot
disagree with the hardware. Regenerate with `make images`.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IMG = REPO / "docs" / "img"
TRACE = REPO / "docs" / "data"

INK = "#22282d"
MUTED = "#5c6672"
LINE = "#8b97a3"
GRID = "#dce2e8"
BLUE = "#2f6f9f"
BLUE_L = "#e6eff6"
ORANGE = "#c26b3f"
ORANGE_L = "#fbeee6"
GREEN = "#1f7a4d"
GREEN_L = "#e6f2eb"
PURPLE = "#6b4f9e"
PURPLE_L = "#efeaf7"
FONT = ("font-family=\"Inter, 'DejaVu Sans', Helvetica, Arial, sans-serif\"")


def esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


class Svg:
    """Minimal SVG builder: explicit coordinates, no external dependencies."""

    def __init__(self, w: int, h: int, title: str):
        self.w, self.h = w, h
        self.parts: list[str] = []
        self.title = title

    def rect(self, x, y, w, h, fill="none", stroke=LINE, sw=1.2, rx=6,
             dash=None, opacity=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
            f' opacity="{opacity}"{d}/>')

    def text(self, x, y, s, size=12, fill=INK, anchor="start", weight="400",
             mono=False, style=""):
        fam = ('font-family="ui-monospace, \'DejaVu Sans Mono\', monospace"'
               if mono else FONT)
        self.parts.append(
            f'<text x="{x}" y="{y}" {fam} font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" '
            f'style="{style}">{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke=LINE, sw=1.2, dash=None, arrow=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        a = f' marker-end="url(#{arrow})"' if arrow else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}"{d}{a}/>')

    def path(self, d, stroke=LINE, sw=1.2, fill="none", dash=None, arrow=None):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        a = f' marker-end="url(#{arrow})"' if arrow else ""
        self.parts.append(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}"{da}{a}/>')

    def render(self) -> str:
        defs = "".join(
            f'<marker id="arrow-{name}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 1 L 9 5 L 0 9 z" fill="{col}"/></marker>'
            for name, col in (("line", LINE), ("blue", BLUE),
                              ("orange", ORANGE), ("green", GREEN),
                              ("purple", PURPLE)))
        return (f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}" role="img" '
                f'aria-label="{esc(self.title)}">'
                f'<defs>{defs}</defs>'
                f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>'
                + "".join(self.parts) + "</svg>")

    def save(self, path: Path) -> None:
        path.write_text(self.render())
        print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Architecture diagram
# ---------------------------------------------------------------------------
def architecture(rows: int = 4, cols: int = 2, s_max: int = 6) -> None:
    s = Svg(1180, 860, "INT8 systolic accelerator datapath")
    s.text(24, 34, "tt_um_danieltyukov_int8_npu", 19, INK, weight="600")
    s.text(24, 54,
           f"weight-stationary systolic array, ROWS={rows} COLS={cols} "
           f"S_MAX={s_max}, signed INT8 in, INT8 out", 12, MUTED)

    # Host interface -------------------------------------------------------
    s.rect(24, 78, 210, 208, BLUE_L, BLUE)
    s.text(36, 100, "host interface", 13, BLUE, weight="600")
    pins = [
        ("ui_in[7:0]", "data byte in"),
        ("uio_in[0] wr", "write strobe"),
        ("uio_in[1] is_cmd", "1 = opcode byte"),
        ("uio_in[2] rd", "advance read pointer"),
        ("uo_out[7:0]", "readback byte"),
        ("uio_out[3] busy", "run in progress"),
        ("uio_out[4] done", "results ready"),
        ("uio_out[7:5]", "ovf / sat / err sticky"),
    ]
    for i, (pin, note) in enumerate(pins):
        y = 122 + i * 21
        s.text(36, y, pin, 10.5, INK, mono=True)
        s.text(150, y, note, 9.5, MUTED)
    s.text(36, 300, "frame = opcode byte + fixed payload;", 10, MUTED)
    s.text(36, 314, "length checked, errors are sticky", 10, MUTED)

    # Activation buffer ----------------------------------------------------
    ax, ay = 268, 78
    s.rect(ax, ay, 236, 96, GREEN_L, GREEN)
    s.text(ax + 12, ay + 22, f"activation buffer  {s_max} x {rows} bytes",
           12, GREEN, weight="600")
    s.text(ax + 12, ay + 40, "byte shift chain, no address decoder", 9.5, MUTED)
    s.text(ax + 12, ay + 58, "read with a diagonal skew:", 10, INK)
    s.text(ax + 12, ay + 74, "row r sees sample (cycle - r)", 10.5, INK,
           mono=True)

    # PE array -------------------------------------------------------------
    px, py, pw, ph, gap = 300, 232, 96, 62, 22
    s.text(px, py - 14, "systolic PE array", 13, INK, weight="600")
    for r in range(rows):
        for c in range(cols):
            x = px + c * (pw + gap)
            y = py + r * (ph + gap)
            s.rect(x, y, pw, ph, "#ffffff", LINE)
            s.text(x + pw / 2, y + 20, f"PE({r},{c})", 11, INK, anchor="middle",
                   weight="600")
            s.text(x + pw / 2, y + 36, f"w = W[{r}][{c}]", 9.5, PURPLE,
                   anchor="middle", mono=True)
            s.text(x + pw / 2, y + 50, "psum += w*a", 9.5, MUTED,
                   anchor="middle", mono=True)
            # activation east
            if c < cols - 1:
                s.line(x + pw, y + ph / 2, x + pw + gap, y + ph / 2, GREEN, 1.6,
                       arrow="arrow-green")
            else:
                s.line(x + pw, y + ph / 2, x + pw + 16, y + ph / 2, GREEN, 1.6)
                s.text(x + pw + 20, y + ph / 2 + 4, "drop", 8.5, MUTED)
            # psum south
            if r < rows - 1:
                s.line(x + pw / 2, y + ph, x + pw / 2, y + ph + gap, BLUE, 1.6,
                       arrow="arrow-blue")
    # row inputs with skew annotation
    for r in range(rows):
        y = py + r * (ph + gap) + ph / 2
        s.line(ax + 60, ay + 96, ax + 60, y, GREEN, 1.4)
        s.line(ax + 60, y, px, y, GREEN, 1.6, arrow="arrow-green")
        s.text(px - 78, y - 6, f"x[s-{r}][{r}]" if r else "x[s][0]", 9.5, GREEN,
               mono=True)
    # north edge zeros
    for c in range(cols):
        x = px + c * (pw + gap) + pw / 2
        s.line(x, py - 34, x, py, BLUE, 1.4, arrow="arrow-blue")
        s.text(x, py - 40, "0", 10, BLUE, anchor="middle", mono=True)

    s.text(px + 2 * (pw + gap) + 40, py + 8, "activations stream east,", 11,
           GREEN, weight="600")
    s.text(px + 2 * (pw + gap) + 40, py + 24, "one PE per cycle", 11, GREEN)
    s.text(px + 2 * (pw + gap) + 40, py + 52, "partial sums accumulate", 11,
           BLUE, weight="600")
    s.text(px + 2 * (pw + gap) + 40, py + 68, "south, one row per cycle", 11,
           BLUE)
    s.text(px + 2 * (pw + gap) + 40, py + 96, "weights stay resident:", 11,
           PURPLE, weight="600")
    s.text(px + 2 * (pw + gap) + 40, py + 112,
           f"{rows * cols} bytes shifted in once,", 11, PURPLE)
    s.text(px + 2 * (pw + gap) + 40, py + 128, "reused by every sample", 11,
           PURPLE)

    # weight chain (snake, entering the last PE)
    wx = px + (cols - 1) * (pw + gap) + pw
    s.path(f"M 246 {py + rows * (ph + gap) + 26} "
           f"H {wx + 14} V {py + (rows - 1) * (ph + gap) + ph - 12} "
           f"H {wx}", PURPLE, 1.6, arrow="arrow-purple")
    s.text(250, py + rows * (ph + gap) + 22, "weight byte chain (reverse "
           "raster order)", 10, PURPLE)

    # PE internals blow-up -------------------------------------------------
    bx, by = 700, 232
    s.rect(bx, by, 250, 176, "#ffffff", INK, 1.4, dash="4 3")
    s.text(bx + 12, by + 22, "inside one PE", 12, INK, weight="600")
    s.rect(bx + 16, by + 36, 74, 30, PURPLE_L, PURPLE, rx=4)
    s.text(bx + 53, by + 56, "w_reg 8b", 9.5, PURPLE, anchor="middle",
           mono=True)
    s.rect(bx + 16, by + 78, 74, 30, GREEN_L, GREEN, rx=4)
    s.text(bx + 53, by + 98, "a_reg 8b", 9.5, GREEN, anchor="middle", mono=True)
    s.rect(bx + 116, by + 52, 56, 42, "#ffffff", ORANGE, rx=4)
    s.text(bx + 144, by + 70, "signed", 9, ORANGE, anchor="middle")
    s.text(bx + 144, by + 84, "8x8 mul", 9, ORANGE, anchor="middle")
    s.rect(bx + 116, by + 112, 56, 34, "#ffffff", BLUE, rx=4)
    s.text(bx + 144, by + 133, "19b add", 9, BLUE, anchor="middle")
    s.rect(bx + 190, by + 112, 46, 34, BLUE_L, BLUE, rx=4)
    s.text(bx + 213, by + 133, "psum", 9, BLUE, anchor="middle", mono=True)
    s.line(bx + 90, by + 51, bx + 116, by + 62, PURPLE, 1.3,
           arrow="arrow-purple")
    s.line(bx + 90, by + 93, bx + 116, by + 84, GREEN, 1.3, arrow="arrow-green")
    s.line(bx + 144, by + 94, bx + 144, by + 112, ORANGE, 1.3,
           arrow="arrow-orange")
    s.line(bx + 172, by + 129, bx + 190, by + 129, BLUE, 1.3, arrow="arrow-blue")
    s.text(bx + 12, by + 166, "one MAC per cycle, no compute enable", 9.5,
           MUTED)

    # Result path ----------------------------------------------------------
    ry = 560
    s.text(24, ry - 16, "result path, one output element at a time", 13, INK,
           weight="600")
    blocks = [
        (24, "accumulator bank", f"{cols} banks x {s_max} x 24b",
         "bias on the first pass,\nsaturating accumulate", BLUE, BLUE_L),
        (258, "serial requantizer", "radix-4 Booth, 9 steps",
         "acc x M, one shared adder", ORANGE, ORANGE_L),
        (492, "rounding shift", "4 bits then 1 per cycle",
         "ties away from zero", ORANGE, ORANGE_L),
        (726, "zero point + saturate", "+zp, clamp to INT8",
         "sticky sat flag", ORANGE, ORANGE_L),
        (960, "activation", "id / ReLU / ReLU6 / leaky",
         "clamp_lo, clamp_hi", GREEN, GREEN_L),
    ]
    for x, title, sub, note, col, fillc in blocks:
        s.rect(x, ry, 196, 104, fillc, col)
        s.text(x + 12, ry + 24, title, 11.5, col, weight="600")
        s.text(x + 12, ry + 42, sub, 9.5, INK, mono=True)
        for i, ln in enumerate(note.split("\n")):
            s.text(x + 12, ry + 62 + i * 14, ln, 9.5, MUTED)
    for x in (220, 454, 688, 922):
        s.line(x, ry + 52, x + 38, ry + 52, LINE, 1.5, arrow="arrow-line")
    for c in range(cols):
        x = px + c * (pw + gap) + pw / 2
        s.path(f"M {x} {py + rows * (ph + gap) - gap} V {ry - 26} H 122 V {ry}",
               BLUE, 1.5, arrow="arrow-blue")
    s.text(126, ry - 32, f"{cols} column sums, one per cycle, staggered by "
                         f"one cycle per column", 10, BLUE)

    s.rect(960, ry + 132, 196, 74, "#ffffff", LINE)
    s.text(972, ry + 154, "result registers", 11.5, INK, weight="600")
    s.text(972, ry + 172, f"{s_max} x {cols} bytes", 9.5, INK, mono=True)
    s.text(972, ry + 190, "read back over uo_out", 9.5, MUTED)
    s.line(1058, ry + 104, 1058, ry + 132, GREEN, 1.5, arrow="arrow-green")
    s.path(f"M 960 {ry + 169} H 150 V 292", LINE, 1.4, arrow="arrow-line")
    s.text(160, 306, "readback mux: results, raw accumulators, status, "
                     "identity block", 10, MUTED)
    s.save(IMG / "architecture.svg")


# ---------------------------------------------------------------------------
# Pipeline timing diagram, from the captured trace
# ---------------------------------------------------------------------------
def pipeline_timing() -> None:
    path = TRACE / "dataflow.json"
    if not path.is_file():
        print(f"skipping pipeline timing: {path} missing (run `make trace`)")
        return
    d = json.loads(path.read_text())
    rows, cols, s_count = d["rows"], d["cols"], d["s_count"]
    frames = d["frames"]
    acts = d["acts"]

    # Which sample each PE holds in each captured cycle, recovered from the
    # activation register value.
    lookup = {}
    for si, vec in enumerate(acts):
        for r, v in enumerate(vec):
            lookup[(r, v)] = si

    cellw, cellh = 46, 26
    left, top = 190, 96
    n = len(frames)
    s = Svg(left + cellw * n + 240, top + cellh * (rows * cols + 3) + 120,
            "systolic pipeline fill, steady state and drain")
    s.text(24, 34, "Pipeline occupancy, captured from simulation", 18, INK,
           weight="600")
    s.text(24, 54, f"ROWS={rows} COLS={cols}, {s_count} activation vectors "
                   f"streamed back to back; numbers are the sample index each "
                   f"PE is working on", 11.5, MUTED)

    for i, f in enumerate(frames):
        x = left + i * cellw
        s.text(x + cellw / 2, top - 10, str(f["cycle"]), 10, MUTED,
               anchor="middle")
    s.text(left - 12, top - 10, "cycle", 10, MUTED, anchor="end")

    busy_cycles = {}
    for k, (r, c) in enumerate([(r, c) for r in range(rows)
                                for c in range(cols)]):
        y = top + k * cellh
        s.text(left - 12, y + 17, f"PE({r},{c})", 11, INK, anchor="end",
               mono=True)
        for i, f in enumerate(frames):
            x = left + i * cellw
            val = f["a_reg"][r][c]
            si = lookup.get((r, val))
            live = val != 0 and si is not None
            s.rect(x, y, cellw - 3, cellh - 4,
                   BLUE_L if live else "#f7f9fa",
                   BLUE if live else GRID, 1.0, rx=3)
            if live:
                s.text(x + (cellw - 3) / 2, y + 16, str(si), 10.5, BLUE,
                       anchor="middle", weight="600")
                busy_cycles.setdefault(i, 0)
                busy_cycles[i] += 1

    # MACs per cycle strip
    y = top + rows * cols * cellh + 12
    s.text(left - 12, y + 17, "MACs", 11, INK, anchor="end", mono=True)
    full = 0
    for i in range(len(frames)):
        x = left + i * cellw
        v = busy_cycles.get(i, 0)
        if v == rows * cols:
            full += 1
        s.rect(x, y, cellw - 3, cellh - 4,
               GREEN_L if v == rows * cols else "#ffffff",
               GREEN if v == rows * cols else GRID, 1.0, rx=3)
        s.text(x + (cellw - 3) / 2, y + 16, str(v), 10.5,
               GREEN if v == rows * cols else MUTED, anchor="middle")

    # Phase bars
    first = min(min(v) for v in [[i for i in busy_cycles]] ) if busy_cycles else 0
    last = max(busy_cycles) if busy_cycles else 0
    fill_end = first + rows + cols - 2
    y2 = y + cellh + 18
    def band(i0, i1, label, col, fillc):
        x0 = left + i0 * cellw
        x1 = left + (i1 + 1) * cellw - 3
        s.rect(x0, y2, x1 - x0, 22, fillc, col, 1.0, rx=4)
        s.text((x0 + x1) / 2, y2 + 16, label, 10, col, anchor="middle",
               weight="600")
    band(0, first - 1, "fill", ORANGE, ORANGE_L)
    if fill_end >= first:
        band(first, fill_end, "ramp", BLUE, BLUE_L)
    if last > fill_end:
        band(fill_end + 1, last, "steady state and drain", GREEN, GREEN_L)

    s.text(24, y2 + 62,
           f"All {rows * cols} PEs hold a live activation simultaneously for "
           f"{full} cycles; every PE performs exactly {s_count} MACs, one per "
           f"cycle, with no gaps.", 11.5, INK)
    s.text(24, y2 + 82,
           f"Array phase is s_count + ROWS + COLS = {s_count} + {rows} + "
           f"{cols} = {s_count + rows + cols} cycles, of which "
           f"{rows + 1} are fill and {cols - 1} drain.", 11.5, MUTED)
    s.save(IMG / "pipeline_timing.svg")


# ---------------------------------------------------------------------------
# Host protocol timing diagram, from the captured trace
# ---------------------------------------------------------------------------
def protocol_timing() -> None:
    path = TRACE / "protocol.json"
    if not path.is_file():
        print(f"skipping protocol timing: {path} missing (run `make trace`)")
        return
    samples = json.loads(path.read_text())["samples"]
    cw = 42
    left, top = 150, 110
    lanes = [("wr", "wr"), ("is_cmd", "is_cmd"), ("busy", "busy"),
             ("done", "done")]
    h = top + 40 + len(lanes) * 42 + 150
    s = Svg(left + cw * len(samples) + 60, h,
            "host protocol: weight load then inference trigger")
    s.text(24, 34, "Host protocol, captured from simulation", 18, INK,
           weight="600")
    s.text(24, 54, "one weight-tile load (LD_W plus 8 payload bytes) followed "
                   "by RUN with requantization", 11.5, MUTED)

    for i, smp in enumerate(samples):
        x = left + i * cw
        s.text(x + cw / 2, top - 42, str(i), 9, MUTED, anchor="middle")
        s.line(x, top - 34, x, top + 30 + len(lanes) * 42, GRID, 0.8)
    s.text(left - 12, top - 42, "cycle", 9, MUTED, anchor="end")

    # ui_in byte lane
    s.text(left - 12, top - 8, "ui_in", 11, INK, anchor="end", mono=True)
    for i, smp in enumerate(samples):
        x = left + i * cw
        active = smp["wr"] == 1
        s.rect(x + 2, top - 26, cw - 6, 24,
               ORANGE_L if smp["is_cmd"] else (BLUE_L if active else "#ffffff"),
               ORANGE if smp["is_cmd"] else (BLUE if active else GRID), 1.0,
               rx=3)
        if active:
            s.text(x + cw / 2, top - 9, f"{smp['ui_in']:02X}", 9.5,
                   ORANGE if smp["is_cmd"] else BLUE, anchor="middle",
                   mono=True)
    # digital lanes
    for k, (key, label) in enumerate(lanes):
        y = top + 22 + k * 42
        s.text(left - 12, y + 14, label, 11, INK, anchor="end", mono=True)
        prev = None
        for i, smp in enumerate(samples):
            x = left + i * cw
            v = smp[key]
            yv = y if v else y + 22
            s.line(x, yv, x + cw, yv, BLUE if v else LINE, 2.0)
            if prev is not None and prev != v:
                s.line(x, y, x, y + 22, LINE, 1.2)
            prev = v
    # labels for the interesting cycles
    y = top + 22 + len(lanes) * 42 + 26
    notes = []
    for i, smp in enumerate(samples):
        if smp["label"] in ("LD_W", "RUN"):
            notes.append((i, smp["label"]))
        if smp["label"] == "compute" and (not notes or notes[-1][1] != "busy"):
            notes.append((i, "busy"))
    for i, label in notes:
        x = left + i * cw + cw / 2
        s.line(x, top + 22 + len(lanes) * 42, x, y - 12, MUTED, 1.0,
               dash="3 3")
        s.text(x, y, label, 10, MUTED, anchor="middle")
    s.text(24, y + 34, "Orange = command byte (is_cmd high), blue = payload "
                       "byte. Writes while busy are rejected and set the "
                       "sticky error flag.", 11, INK)
    s.text(24, y + 54, "busy rises two cycles after the RUN byte: one to "
                       "register the command, one for the sequencer to start.",
           11, MUTED)
    s.save(IMG / "protocol_timing.svg")


def main() -> int:
    IMG.mkdir(parents=True, exist_ok=True)
    architecture()
    pipeline_timing()
    protocol_timing()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
