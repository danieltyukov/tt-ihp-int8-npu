# SPDX-FileCopyrightText: Copyright (c) 2026 Daniel Tyukov
# SPDX-License-Identifier: Apache-2.0
"""Render a hardened GDS to PNG with KLayout in batch mode.

KLayout's batch mode takes no positional arguments, only -rd name=value pairs
that arrive as script globals:

    klayout -b -rm scripts/render_gds.py \
        -rd gds=runs/.../tt_um_danieltyukov_int8_npu.gds \
        -rd out=docs/img/layout_die.png -rd w=2600 -rd h=520

Optional globals:
    box=x1,y1,x2,y2   render this rectangle in um instead of the whole layout
    frac=fx,fy,fw,fh  same, as fractions of the die bounding box
    hier=n            limit the hierarchy depth (default: everything)
"""

import pya  # noqa: F401  (provided by the klayout interpreter)

gds_path = gds          # noqa: F821
out_path = out          # noqa: F821
width = int(w)          # noqa: F821
height = int(h)         # noqa: F821

lv = pya.LayoutView()
lv.load_layout(gds_path, 0)

try:
    lv.max_hier_levels = int(hier)  # noqa: F821
except NameError:
    lv.max_hier()

bbox = lv.active_cellview().cell.dbbox()
print("die bbox um: %.2f %.2f %.2f %.2f (%.2f x %.2f)"
      % (bbox.left, bbox.bottom, bbox.right, bbox.top,
         bbox.width(), bbox.height()))

region = None
try:
    x1, y1, x2, y2 = (float(v) for v in box.split(","))  # noqa: F821
    region = pya.DBox(x1, y1, x2, y2)
except NameError:
    pass

if region is None:
    try:
        fx, fy, fw, fh = (float(v) for v in frac.split(","))  # noqa: F821
        region = pya.DBox(
            bbox.left + fx * bbox.width(),
            bbox.bottom + fy * bbox.height(),
            bbox.left + (fx + fw) * bbox.width(),
            bbox.bottom + (fy + fh) * bbox.height(),
        )
    except NameError:
        pass

if region is None:
    lv.zoom_fit()
else:
    print("crop um: %.2f %.2f %.2f %.2f (%.2f x %.2f)"
          % (region.left, region.bottom, region.right, region.top,
             region.width(), region.height()))
    lv.zoom_box(region)

lv.save_image(out_path, width, height)
print("wrote %s (%dx%d)" % (out_path, width, height))
