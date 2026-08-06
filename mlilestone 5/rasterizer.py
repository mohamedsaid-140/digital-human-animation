"""
rasterizer.py
-------------
Minimal software rasterizer with a per-pixel z-buffer.
"""

import math
from PIL import Image


def rasterize(width, height, triangles_screen, bg=(16, 18, 24)):
    img = Image.new("RGB", (width, height), bg)
    px = img.load()
    zbuf = [[float("inf")] * width for _ in range(height)]

    for (p0, p1, p2, color) in triangles_screen:
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        min_x = max(int(math.floor(min(x0, x1, x2))), 0)
        max_x = min(int(math.ceil(max(x0, x1, x2))), width - 1)
        min_y = max(int(math.floor(min(y0, y1, y2))), 0)
        max_y = min(int(math.ceil(max(y0, y1, y2))), height - 1)
        if min_x > max_x or min_y > max_y:
            continue

        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-9:
            continue

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                w0 = (x1 - x) * (y2 - y) - (x2 - x) * (y1 - y)
                w1 = (x2 - x) * (y0 - y) - (x0 - x) * (y2 - y)
                w2 = (x0 - x) * (y1 - y) - (x1 - x) * (y0 - y)
                if area > 0:
                    inside = w0 >= 0 and w1 >= 0 and w2 >= 0
                else:
                    inside = w0 <= 0 and w1 <= 0 and w2 <= 0
                if not inside:
                    continue
                b0, b1, b2 = w0 / area, w1 / area, w2 / area
                z = b0 * z0 + b1 * z1 + b2 * z2
                if z < zbuf[y][x]:
                    zbuf[y][x] = z
                    px[x, y] = color

    return img
