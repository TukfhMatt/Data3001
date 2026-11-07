"""
The shipping density raster (World Bank / IMF "Global Shipping Traffic Density",
Commercial layer), read for one box straight from the zip. Its values are
undocumented in scale, so steps use them only as a relative density.

    raw, px, _ = read_window(box)
    per_cell = blocks(raw, res, px).sum(axis=(1, 3))
"""

import os
import zipfile

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from agulhas.config import SHIP_URL, SHIP_ZIP


def read_window(box, dtype=np.float64):
    """Pixels of the box, row 0 at its southern edge, with no-data and negative
    values set to 0: (pixels, pixel size in degrees, file name in the zip)."""
    if not os.path.exists(SHIP_ZIP):
        raise SystemExit(f"{SHIP_ZIP} not found — download it first:\n"
                         f"  curl -L -o {SHIP_ZIP} {SHIP_URL}")
    with zipfile.ZipFile(SHIP_ZIP) as z:
        tif = next(n for n in z.namelist() if n.lower().endswith((".tif", ".tiff")))
    lon_min, lon_max, lat_min, lat_max = box
    with rasterio.open(f"zip://{SHIP_ZIP}!/{tif}") as src:
        window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform).round_offsets().round_lengths()
        raw = src.read(1, window=window).astype(dtype)
        nodata, px = src.nodata, src.res[0]
    if nodata is not None:
        raw[raw == nodata] = 0
    raw[~np.isfinite(raw) | (raw < 0)] = 0
    return raw[::-1], px, tif   # raster rows run north→south


def blocks(a, deg, px):
    """Pixels grouped into deg-sized cells, shape (rows, k, columns, k), ready to sum
    or average over axes (1, 3). A partial last row/column (box side not a multiple
    of deg) is zero-padded so it keeps the pixels it has."""
    k = int(round(deg / px))
    ny, nx = -(-a.shape[0] // k), -(-a.shape[1] // k)
    a = np.pad(a, ((0, ny * k - a.shape[0]), (0, nx * k - a.shape[1])))
    return a.reshape(ny, k, nx, k)
