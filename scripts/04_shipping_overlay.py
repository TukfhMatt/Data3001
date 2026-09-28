"""
Agulhas Region — Shipping Traffic vs Surface Currents
Overlays commercial shipping density (World Bank / IMF AIS, 2015–2021) on
the mean current from drogued and from undrogued drifters, then picks
candidate release points: the busiest shipping cells plus known hotspots,
ranked by how much material each transition matrix (drogued and undrogued)
carries to near-shore cells. Grid, lag and box come from scripts/config.py.

Shipping data: "Global Shipping Traffic Density", World Bank Data Catalog,
dataset 0037580, CC-BY 4.0. Described as counts of AIS positions per 0.005°
cell, but the stored values are far larger than raw position counts could be
(open-ocean pixels ~1e5–1e6), so they are treated as a relative density only.

Candidates are only placed in cells that are reliable in both matrices (≥ 10
distinct drifters, has outgoing data), so the two versions are compared at
the same points. Hotspots in unreliable cells are moved to the nearest
reliable cell and the shift is reported.
"""

import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import rasterio
from rasterio.windows import from_bounds
import xarray as xr

from config import DROGUE_TYPES, MIN_DRIFTERS, grid_shape, operator_path, parse_args
from transport import load_operator, propagate, state_of

args = parse_args(__doc__, grid=True, lag=True)

SHIP_URL = ("https://datacatalogfiles.worldbank.org/ddh-published/0037580/5/"
            "DR0045405/shipdensity_commercial_.zip")
SHIP_ZIP = "data/shipping/shipdensity_commercial.zip"
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
RES = args.res       # analysis grid (matches the transition matrix)
FINE = 0.1           # display grid for the shipping map
MIN_CURRENT_OBS = 300  # cells with fewer hourly obs of a drogue type get no current arrow
RES_TAG = f"{RES:g}".replace(".", "p")

N_TRAFFIC = 8        # busiest-cell candidates to pick
MIN_SPACING = 2.0    # degrees between chosen candidates

# Known hotspots: offshore of each port / anchorage (lon, lat)
HOTSPOTS = {
    "Richards Bay":           (32.6, -28.9),
    "Durban":                 (31.4, -30.0),
    "Algoa Bay (STS bunkering)": (25.9, -33.9),
    "Cape Town":              (18.2, -33.9),
    "Saldanha Bay":           (17.8, -33.0),
}

os.makedirs("figures", exist_ok=True)

# ── 1. Shipping density ──────────────────────────────────────────────────────
if not os.path.exists(SHIP_ZIP):
    raise SystemExit(f"{SHIP_ZIP} not found — download it first:\n"
                     f"  curl -L -o {SHIP_ZIP} {SHIP_URL}")

with zipfile.ZipFile(SHIP_ZIP) as z:
    tif = next(n for n in z.namelist() if n.lower().endswith((".tif", ".tiff")))

with rasterio.open(f"zip://{SHIP_ZIP}!/{tif}") as src:
    window = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, src.transform).round_offsets().round_lengths()
    raw = src.read(1, window=window).astype(np.float64)
    nodata = src.nodata
    px = src.res[0]

if nodata is not None:
    raw[raw == nodata] = 0
raw[~np.isfinite(raw) | (raw < 0)] = 0
raw = raw[::-1]  # raster rows run north→south; flip so row 0 is LAT_MIN


def block_sum(a, deg):
    """Sum pixels into deg-sized cells; a partial last row/column (box side
    not a multiple of deg) is zero-padded so it keeps the pixels it has."""
    k = int(round(deg / px))
    ny, nx = -(-a.shape[0] // k), -(-a.shape[1] // k)
    a = np.pad(a, ((0, ny * k - a.shape[0]), (0, nx * k - a.shape[1])))
    return a.reshape(ny, k, nx, k).sum(axis=(1, 3))


ship_fine = block_sum(raw, FINE)
n_lon, n_lat = grid_shape(args.box, RES)
# Cell centres; a partial last row/column is centred inside the box
CELL_LON = (np.minimum(LON_MIN + np.arange(n_lon) * RES, LON_MAX)
            + np.minimum(LON_MIN + np.arange(1, n_lon + 1) * RES, LON_MAX)) / 2
CELL_LAT = (np.minimum(LAT_MIN + np.arange(n_lat) * RES, LAT_MAX)
            + np.minimum(LAT_MIN + np.arange(1, n_lat + 1) * RES, LAT_MAX)) / 2
ship_grid = block_sum(raw, RES)[:n_lat, :n_lon]   # drop any stray rounding pixels
del raw
print(f"Shipping raster: {tif}, pixel {px:.4f}° (values used as relative density)")

# ── 2. Mean current per drogue type ──────────────────────────────────────────
ds = xr.open_dataset(args.data)
lon = ds["lon"].values.astype(float)
lat = ds["lat"].values.astype(float)
ve = ds["ve"].values.astype(float)
vn = ds["vn"].values.astype(float)
drogue = ds["drogue_status"].values.astype(bool)

in_box =((lon >= LON_MIN) & (lon < LON_MAX) & (lat >= LAT_MIN) & (lat < LAT_MAX)
          & np.isfinite(ve) & np.isfinite(vn))
currents = {}
for name, flag in DROGUE_TYPES.items():
    m = in_box & (drogue == flag)
    cell = ((lat[m] - LAT_MIN) // RES).astype(int) * n_lon + ((lon[m] - LON_MIN) // RES).astype(int)
    cnt = np.bincount(cell, minlength=n_lat * n_lon)
    u = np.bincount(cell, ve[m], n_lat * n_lon) / np.maximum(cnt, 1)
    v = np.bincount(cell, vn[m], n_lat * n_lon) / np.maximum(cnt, 1)
    ok = cnt >= MIN_CURRENT_OBS
    u, v = np.where(ok, u, np.nan).reshape(n_lat, n_lon), np.where(ok, v, np.nan).reshape(n_lat, n_lon)
    currents[name] = (u, v, cnt.reshape(n_lat, n_lon))
    print(f"Mean current ({name}): {ok.sum()} cells, max speed {np.nanmax(np.hypot(u, v)):.2f} m/s")
del lon, lat, ve, vn, drogue

# ── 3. Candidate release points ──────────────────────────────────────────────
ops = {name: load_operator(operator_path(name, RES, args.tau, args.box)) for name in DROGUE_TYPES}
op = ops["drogued"]   # both operators share the same ocean cells
n_cells = len(op["cell_flat"])
is_ocean = np.zeros(n_lat * n_lon, bool)
is_ocean[op["cell_flat"]] = True
ocean_grid = is_ocean.reshape(n_lat, n_lon)

# Near-shore cells: ocean cells with a non-ocean neighbour inside the grid
# (the matrix has no beaching state, so these stand in for coastal impact)
pad = np.pad(~ocean_grid, 1, constant_values=False)
land_nb = np.zeros_like(ocean_grid)
for dj in (-1, 0, 1):
    for di in (-1, 0, 1):
        land_nb |= pad[1 + dj: 1 + dj + n_lat, 1 + di: 1 + di + n_lon]
coastal_grid = ocean_grid & land_nb
coastal_state = coastal_grid.ravel()[op["cell_flat"]]

# Reliable cells: enough drifters behind the row, and real outgoing data,
# in both matrices
reliable = np.zeros(n_lat * n_lon, bool)
reliable[op["cell_flat"]] = np.logical_and.reduce([~o["flagged"] & ~o["empty"] for o in ops.values()])
reliable_grid = reliable.reshape(n_lat, n_lon)


def snap_to_reliable(lo, la):
    """Centre of the nearest reliable cell to (lo, la) and the distance in km."""
    jj, ii = np.nonzero(reliable_grid)
    cx, cy = CELL_LON[ii], CELL_LAT[jj]
    dx = (cx - lo) * 111.32 * np.cos(np.radians(la))
    dy = (cy - la) * 110.57
    k = np.argmin(dx ** 2 + dy ** 2)
    return cx[k], cy[k], float(np.hypot(dx[k], dy[k]))


print(f"\nHotspots moved to the nearest cell reliable in both matrices "
      f"(≥ {MIN_DRIFTERS} drifters, has outgoing data):")
candidates = []
for name, (lo, la) in HOTSPOTS.items():
    j, i = int((la - LAT_MIN) // RES), int((lo - LON_MIN) // RES)
    if reliable_grid[j, i]:
        clo, cla = CELL_LON[i], CELL_LAT[j]
        print(f"  {name:<28} unchanged")
    else:
        clo, cla, km = snap_to_reliable(lo, la)
        print(f"  {name:<28} ({lo}, {la}) → ({clo:.1f}, {cla:.1f}), ~{km:.0f} km")
    candidates.append(("hotspot", name, clo, cla))

traffic = np.where(reliable_grid, ship_grid, -1).ravel()
for flat in np.argsort(traffic)[::-1]:
    if len([c for c in candidates if c[0] == "traffic"]) == N_TRAFFIC:
        break
    j, i = divmod(flat, n_lon)
    clo, cla = CELL_LON[i], CELL_LAT[j]
    if all(max(abs(clo - c[2]), abs(cla - c[3])) >= MIN_SPACING for c in candidates):
        candidates.append(("traffic", f"Lane {clo:.1f}°E {abs(cla):.1f}°S", clo, cla))


def fate(o, s):
    """Coastal exposure over 30 days and state after 1 year, released in state s."""
    tau = float(o["tau_days"])
    week_step = int(round(7 / tau))
    p = np.zeros(o["P"].shape[0]); p[s] = 1.0
    exposure, coast_7, c = 0.0, 0.0, 0.0
    for step in range(1, int(round(30 / tau)) + 1):
        p = p @ o["P"]
        c = p[:n_cells][coastal_state].sum()
        exposure += c * tau
        if step == week_step:
            coast_7 = c
    p365 = propagate(o, np.eye(o["P"].shape[0])[s], days=365)
    return {"coastal_frac_7d": coast_7, "coastal_frac_30d": c,
            "coastal_exposure_days_30d": exposure, "in_R_365d": p365[:n_cells].sum(),
            **{f"exit_{l}_365d": v for l, v in zip(o["exit_labels"], p365[n_cells:])},
            "row_drifters": int(o["row_drifters"][s])}


rows = []
for kind, name, lo, la in candidates:
    s = state_of(op, lo, la)
    j, i = divmod(int(op["cell_flat"][s]), n_lon)
    row = {"kind": kind, "name": name, "lon": lo, "lat": la,
           "ship_density": ship_grid[j, i],
           "ship_density_rel": ship_grid[j, i] / ship_grid[ocean_grid].max(),
           "start_coastal": bool(coastal_grid[j, i])}
    for dname, o in ops.items():
        row.update({f"{dname}_{k}": v for k, v in fate(o, s).items()})
    rows.append(row)

table = pd.DataFrame(rows).sort_values("drogued_coastal_exposure_days_30d", ascending=False)
table.to_csv("data/release_candidates.csv", index=False)

print(f"\nCandidate release points ({RES:g}°, τ = {args.tau:g} d), ranked by drogued coastal exposure")
print("ship = density relative to the busiest cell in R; coastal = ocean cells next to land;")
print("exposure = material-days spent in coastal cells over the first 30 days\n")
show = table[["kind", "name", "lon", "lat", "ship_density_rel", "start_coastal"]
             + [f"{d}_{k}" for k in ("coastal_exposure_days_30d", "in_R_365d") for d in DROGUE_TYPES]]
show.columns = ["kind", "name", "lon", "lat", "ship", "coastal",
                "exp_30d_dro", "exp_30d_und", "inR_365d_dro", "inR_365d_und"]
print(show.to_string(index=False, formatters={
    "ship": "{:.0%}".format,
    "exp_30d_dro": "{:.1f}".format, "exp_30d_und": "{:.1f}".format,
    "inR_365d_dro": "{:.0%}".format, "inR_365d_und": "{:.0%}".format}))
print("\nTable saved → data/release_candidates.csv")

ship_out = f"data/shipping_{RES_TAG}deg.npz"
np.savez_compressed(ship_out, ship=ship_grid, ship_fine=ship_fine,
                    **{f"{k}_{d}": a for d, c in currents.items()
                       for k, a in zip(("u", "v", "current_obs"), c)},
                    coastal=coastal_grid, lon_min=LON_MIN, lat_min=LAT_MIN,
                    res=RES, fine=FINE)
print(f"Shipping and currents saved → {ship_out}")

# ── 4. Maps (one per drogue type) ────────────────────────────────────────────
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    subplot_kw = {"projection": ccrs.PlateCarree()}
except ImportError:
    ccrs = None
    subplot_kw = {}
kw = {"transform": ccrs.PlateCarree()} if ccrs else {}

import matplotlib.patheffects as pe

fine_lon = np.arange(ship_fine.shape[1] + 1) * FINE + LON_MIN
fine_lat = np.arange(ship_fine.shape[0] + 1) * FINE + LAT_MIN
# Relative to the busiest 0.1° cell; colour range clipped to percentiles of
# non-zero cells so the lanes stand out from the broad background
rel = np.where(ship_fine > 0, ship_fine / ship_fine.max(), np.nan)
lo_c, hi_c = np.nanpercentile(rel, [60, 99.5])
cx, cy = CELL_LON, CELL_LAT
thin = max(1, int(round(2 / RES)))   # one arrow per ~2°
sub = (slice(None, None, thin), slice(None, None, thin))
halo = [pe.withStroke(linewidth=3, foreground="white")]
LABELS = {"drogued": "drogued drifters, 15 m current", "undrogued": "undrogued drifters, current + wind slip"}

for dname, (u, v, _) in currents.items():
    fig, ax = plt.subplots(figsize=(13, 9), subplot_kw=subplot_kw)
    mesh = ax.pcolormesh(fine_lon, fine_lat, rel, cmap="Blues",
                         norm=LogNorm(vmin=lo_c, vmax=hi_c), **kw)
    q = ax.quiver(cx[sub[1]], cy[sub[0]], u[sub], v[sub], color="#3a3a3a", alpha=0.85,
                  scale=10, width=0.0025, zorder=4, **kw)
    ax.quiverkey(q, 0.40, 0.93, 0.5, "0.5 m/s mean current", labelpos="E",
                 coordinates="axes", fontproperties={"size": 9})

    for kind, marker, face, label in [("hotspot", "*", "#e8710a", "Known hotspot (moved to nearest reliable cell)"),
                                      ("traffic", "o", "#ffffff", "Busiest shipping cell (reliable cells only)")]:
        sel = table[table["kind"] == kind]
        ax.scatter(sel["lon"], sel["lat"], marker=marker, s=180 if marker == "*" else 70,
                   c=face, edgecolor="#222222", linewidth=1, zorder=6, label=label, **kw)
    col = f"{dname}_coastal_exposure_days_30d"
    for _, r in table.iterrows():
        text = f"{r['name']}  {r[col]:.1f} d" if r["kind"] == "hotspot" else f"{r[col]:.1f} d"
        ax.annotate(text, (r["lon"], r["lat"]), xytext=(7, -12), textcoords="offset points",
                    fontsize=8, color="#222222", zorder=7, path_effects=halo)

    if ccrs:
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, color="#e6e6e6", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
        gl.top_labels = gl.right_labels = False

    cb = fig.colorbar(mesh, ax=ax, shrink=0.7, pad=0.02, extend="both")
    cb.set_label("Commercial ship density relative to busiest 0.1° cell, AIS 2015–2021 (log)")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(f"Commercial shipping and mean current ({LABELS[dname]}) — Agulhas box R\n"
                 f"Labels: material-days spent in near-shore cells over the first 30 days after release "
                 f"({RES:g}°, τ = {args.tau:g} d)", fontsize=11)

    out = f"figures/agulhas_shipping_currents_{dname}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Map saved → {out}")
    plt.close()
