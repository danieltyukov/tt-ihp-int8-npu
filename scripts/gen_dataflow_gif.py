#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
# SPDX-License-Identifier: Apache-2.0
"""Animate the systolic array from a real simulation trace.

Reads docs/data/dataflow.json (written by test/test_trace.py against the RTL)
and renders docs/img/dataflow.gif: one frame per clock cycle, showing the
activation register and partial sum of every PE, the skewed row inputs and the
column results as they retire.

Also renders docs/img/dataflow_still.png, the same trace as a static contact
sheet. docs/info.md becomes a PDF datasheet and a PDF cannot animate: typst
embeds the GIF's first frame, which is cycle 0 with the array idle, so the
datasheet ended up illustrating the dataflow with a picture of nothing
happening. The contact sheet picks its cycles out of the trace instead.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
IMG = REPO / "docs" / "img"
TRACE = REPO / "docs" / "data" / "dataflow.json"

W, H = 760, 470
BG = (255, 255, 255)
INK = (34, 40, 45)
MUTED = (92, 102, 114)
GRID = (220, 226, 232)
BLUE = (47, 111, 159)
BLUE_L = (230, 239, 246)
GREEN = (31, 122, 77)
GREEN_L = (230, 242, 235)
ORANGE = (194, 107, 63)
PURPLE = (107, 79, 158)


def font(size: int, bold: bool = False):
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
             "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"]
    for name in names:
        for base in ("/usr/share/fonts/truetype/dejavu/",
                     "/usr/share/fonts/truetype/liberation/", ""):
            try:
                return ImageFont.truetype(base + name, size)
            except OSError:
                continue
    return ImageFont.load_default()


F_TITLE = font(19, True)
F_SUB = font(12)
F_CELL = font(12, True)
F_SMALL = font(10)
F_TINY = font(9)


def draw_frame(d: dict, frame: dict, sample_of) -> Image.Image:
    rows, cols = d["rows"], d["cols"]
    img = Image.new("RGB", (W, H), BG)
    g = ImageDraw.Draw(img)

    g.text((22, 16), "Weight-stationary systolic array, cycle "
                     f"{frame['cycle']}", INK, font=F_TITLE)
    g.text((22, 42), f"ROWS={rows} COLS={cols}, {d['s_count']} activation "
                     f"vectors streamed one per cycle", MUTED, font=F_SUB)

    cw, ch, gap = 150, 66, 26
    x0, y0 = 250, 84
    for r in range(rows):
        # skewed row input
        val = frame["row_in"][r]
        val = val - 256 if val > 127 else val
        y = y0 + r * (ch + gap)
        live = val != 0
        g.rectangle([120, y + 14, 205, y + 44],
                    fill=GREEN_L if live else BG,
                    outline=GREEN if live else GRID)
        g.text((162, y + 22), f"x={val}" if live else "idle",
               GREEN if live else MUTED, font=F_CELL, anchor="ma")
        g.line([205, y + 29, x0 - 6, y + 29], fill=GREEN if live else GRID,
               width=2)
        g.text((22, y + 22), f"row {r}", INK, font=F_CELL)

        for c in range(cols):
            x = x0 + c * (cw + gap)
            a = frame["a_reg"][r][c]
            p = frame["psum"][r][c]
            w = frame["w_reg"][r][c]
            si = sample_of.get((r, a))
            live_pe = a != 0 and si is not None
            g.rectangle([x, y, x + cw, y + ch],
                        fill=BLUE_L if live_pe else BG,
                        outline=BLUE if live_pe else GRID, width=2)
            g.text((x + 8, y + 6), f"PE({r},{c})", INK, font=F_CELL)
            g.text((x + cw - 8, y + 6), f"w={w}", PURPLE, font=F_SMALL,
                   anchor="ra")
            g.text((x + 8, y + 26), f"a={a}", GREEN if live_pe else MUTED,
                   font=F_SMALL)
            g.text((x + 8, y + 44), f"psum={p}", BLUE, font=F_SMALL)
            if live_pe:
                g.text((x + cw - 8, y + 44), f"sample {si}", BLUE,
                       font=F_TINY, anchor="ra")
            # arrows
            if c < cols - 1:
                g.line([x + cw, y + ch / 2, x + cw + gap, y + ch / 2],
                       fill=GREEN, width=2)
            if r < rows - 1:
                g.line([x + cw / 2, y + ch, x + cw / 2, y + ch + gap],
                       fill=BLUE, width=2)

    # column results leaving the bottom
    ybot = y0 + rows * (ch + gap) - gap + 14
    for c in range(cols):
        x = x0 + c * (cw + gap)
        p = frame["psum"][rows - 1][c]
        g.text((x + cw / 2, ybot + 6), f"column {c} sum: {p}", BLUE,
               font=F_CELL, anchor="ma")
    state = {0: "idle", 1: "array phase", 2: "requantizing", 3: "done"}.get(
        frame["state"], "?")
    g.text((22, H - 30), f"sequencer: {state}", ORANGE, font=F_CELL)
    macs = sum(1 for r in range(rows) for c in range(cols)
               if frame["a_reg"][r][c] != 0
               and (r, frame["a_reg"][r][c]) in sample_of)
    g.text((W - 22, H - 30), f"MACs this cycle: {macs} of {rows * cols}",
           INK if macs < rows * cols else GREEN, font=F_CELL, anchor="ra")
    return img


def live_macs(d: dict, frame: dict, sample_of) -> int:
    """PEs holding a live activation this cycle, by the same rule draw_frame
    uses for the MAC counter it prints."""
    return sum(1 for r in range(d["rows"]) for c in range(d["cols"])
               if frame["a_reg"][r][c] != 0
               and (r, frame["a_reg"][r][c]) in sample_of)


def still_cycles(d: dict, sample_of) -> list[int]:
    """Frame indices for the contact sheet: fill, mid-fill, peak, drain.

    Read out of the trace rather than hard-coded, so a fork that resizes the
    array still gets four cycles that show the wavefront arriving, every PE
    busy, and results retiring.
    """
    macs = [live_macs(d, f, sample_of) for f in d["frames"]]
    peak = macs.index(max(macs))
    start = next(i for i, m in enumerate(macs) if m > 0)
    # The last cycle that still has a live MAC, not the first one below peak:
    # one PE short of full looks identical to full at a glance, whereas the
    # tail shows the array emptying with the column results already retired.
    tail = max(i for i, m in enumerate(macs) if m > 0)
    picks = [start, (start + peak) // 2, peak, tail]
    seen, out = set(), []
    for i in picks:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def contact_sheet(images: list[Image.Image]) -> Image.Image:
    cols = 2 if len(images) > 1 else 1
    rows = (len(images) + cols - 1) // cols
    pad = 12
    sheet = Image.new("RGB", (cols * W + (cols + 1) * pad,
                              rows * H + (rows + 1) * pad), BG)
    g = ImageDraw.Draw(sheet)
    for i, im in enumerate(images):
        x = pad + (i % cols) * (W + pad)
        y = pad + (i // cols) * (H + pad)
        sheet.paste(im, (x, y))
        g.rectangle([x, y, x + W - 1, y + H - 1], outline=GRID)
    return sheet


def main() -> int:
    if not TRACE.is_file():
        print(f"{TRACE} missing: run `make trace` in test/ first")
        return 1
    d = json.loads(TRACE.read_text())
    sample_of = {}
    for si, vec in enumerate(d["acts"]):
        for r, v in enumerate(vec):
            sample_of[(r, v)] = si

    frames = [draw_frame(d, f, sample_of) for f in d["frames"]]
    IMG.mkdir(parents=True, exist_ok=True)
    out = IMG / "dataflow.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=800,
                   loop=0, optimize=True)
    print(f"wrote {out} ({len(frames)} frames from {len(d['frames'])} "
          f"simulated cycles)")

    picks = still_cycles(d, sample_of)
    still = IMG / "dataflow_still.png"
    contact_sheet([frames[i] for i in picks]).save(still, optimize=True)
    print(f"wrote {still} (cycles "
          f"{', '.join(str(d['frames'][i]['cycle']) for i in picks)}, "
          f"{', '.join(str(live_macs(d, d['frames'][i], sample_of)) for i in picks)}"
          f" of {d['rows'] * d['cols']} PEs live)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
