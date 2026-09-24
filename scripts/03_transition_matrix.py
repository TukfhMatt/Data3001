"""
Agulhas Region — Transition Matrix (Ulam's method)
Builds the surface-transport operator P for box R from GDP drifter pairs
separated by τ = 3.5 days, on a 1° grid, separately for drogued (container)
and undrogued (oil proxy) drifters.

P[i, j] = probability that material in state i is in state j after τ.
States: ocean cells of R, then four absorbing "exited R" states (W, E, S, N).
Load and iterate with scripts/transport.py.
"""

import numpy as np
import xarray as xr

from transport import propagate, release, to_grid

GDP_LOCAL = "data/agulhas_gdp6h_subset.nc"

LON_MIN, LON_MAX = 10.0, 40.0
LAT_MIN, LAT_MAX = -45.0, -25.0
RES = 1.0
TAU_DAYS = 3.5
MIN_DRIFTERS = 10          # rows built from fewer distinct drifters are flagged
EXIT_LABELS = np.array(["W", "E", "S", "N"])

# ── 1. Load subset ───────────────────────────────────────────────────────────
ds = xr.open_dataset(GDP_LOCAL)
lon = ds["lon"].values.astype(float)
lat = ds["lat"].values.astype(float)
drogue = ds["drogue_status"].values.astype(bool)
t = ds["time"].values.astype("datetime64[s]").astype(np.int64)
traj_idx = np.repeat(np.arange(ds.sizes["traj"]), ds["rowsize"].values)
years = ds["time"].values.astype("datetime64[Y]").astype(int) + 1970

n_lon = int(round((LON_MAX - LON_MIN) / RES))
n_lat = int(round((LAT_MAX - LAT_MIN) / RES))
lon_edges = np.linspace(LON_MIN, LON_MAX, n_lon + 1)
lat_edges = np.linspace(LAT_MIN, LAT_MAX, n_lat + 1)

in_R = (lon >= LON_MIN) & (lon < LON_MAX) & (lat >= LAT_MIN) & (lat < LAT_MAX)
cell_flat = np.where(
    in_R,
    ((lat - LAT_MIN) // RES).astype(int) * n_lon + ((lon - LON_MIN) // RES).astype(int),
    -1,
)

# ── 2. Pair each obs with the same drifter τ later ───────────────────────────
# Data is sorted by (trajectory, time), so a combined key can be binary-searched.
TAU_S = int(TAU_DAYS * 86400)
key = traj_idx.astype(np.int64) * 10**10 + t
target = key + TAU_S
j = np.searchsorted(key, target)
j = np.minimum(j, len(key) - 1)
has_pair = in_R & (key[j] == target)
start = np.where(has_pair)[0]
end = j[has_pair]


def exit_state(lon_e, lat_e):
    """Which side of R a point outside R lies on (largest excursion wins)."""
    excursion = np.stack([LON_MIN - lon_e, lon_e - LON_MAX,
                          LAT_MIN - lat_e, lat_e - LAT_MAX])
    return np.argmax(excursion, axis=0)


# ── 3. Build P for each drogue subset ────────────────────────────────────────
def build(name, drogue_flag):
    same_type = (drogue[start] == drogue_flag) & (drogue[end] == drogue_flag)
    s, e = start[same_type], end[same_type]

    # Ocean cells: any cell in R where a drifter of either type was observed
    cells = np.unique(cell_flat[cell_flat >= 0])
    n_cells = len(cells)
    n_states = n_cells + len(EXIT_LABELS)
    state_of_cell = np.full(n_lat * n_lon, -1)
    state_of_cell[cells] = np.arange(n_cells)

    from_state = state_of_cell[cell_flat[s]]
    to_state = np.where(
        cell_flat[e] >= 0,
        state_of_cell[np.maximum(cell_flat[e], 0)],
        n_cells + exit_state(lon[e], lat[e]),
    )

    C = np.zeros((n_states, n_states))
    np.add.at(C, (from_state, to_state), 1)

    # Distinct drifters contributing to each row
    pairs = np.unique(np.stack([from_state, traj_idx[s]]), axis=1)
    row_drifters = np.bincount(pairs[0], minlength=n_states)
    row_obs = C.sum(axis=1)

    P = np.zeros_like(C)
    ok = row_obs > 0
    P[ok] = C[ok] / row_obs[ok, None]
    # Ocean cells with no outgoing data: hold mass in place (flagged below)
    empty = np.where(~ok[:n_cells])[0]
    P[empty, empty] = 1.0
    # Exit states are absorbing
    P[n_cells:, n_cells:] = np.eye(len(EXIT_LABELS))

    flagged = row_drifters[:n_cells] < MIN_DRIFTERS
    out = f"data/P_{name}_1deg_3p5d.npz"
    np.savez_compressed(
        out,
        P=P, C=C,
        row_drifters=row_drifters[:n_cells], row_obs=row_obs[:n_cells],
        flagged=flagged, empty=np.isin(np.arange(n_cells), empty),
        cell_flat=cells, lon_edges=lon_edges, lat_edges=lat_edges,
        res=RES, tau_days=TAU_DAYS, exit_labels=EXIT_LABELS,
        drogued=drogue_flag,
        years=np.array([years[s].min(), years[s].max()]),
        source="NOAA GDP hourly v2.01 via CloudDrift, thinned to 6-hourly",
    )

    # ── Checks ──
    print(f"\n=== {name} ===")
    print(f"Transitions used      : {len(s):,}  ({years[s].min()}–{years[s].max()})")
    print(f"Ocean cells / states  : {n_cells} / {n_states}")
    print(f"Rows sum to 1         : {np.allclose(P.sum(axis=1), 1)}")
    print(f"Cells with no data    : {len(empty)}")
    print(f"Cells < {MIN_DRIFTERS} drifters    : {flagged.sum()} ({flagged.mean():.0%})")
    exit_per_step = P[:n_cells, n_cells:].sum(axis=1)
    print(f"Mean exit prob / step : {exit_per_step.mean():.1%}")
    print(f"Stay-in-same-cell prob: {np.diag(P)[:n_cells].mean():.1%} (mean over cells)")
    print(f"Saved → {out}")
    return {"P": P, "cell_flat": cells, "lon_edges": lon_edges,
            "lat_edges": lat_edges, "res": RES, "tau_days": TAU_DAYS}


ops = {name: build(name, flag) for name, flag in [("drogued", True), ("undrogued", False)]}

# ── 4. Sanity check: one release off Durban ──────────────────────────────────
print("\nSanity check — release at 31.5°E, 30.5°S (off Durban)")
for name, op in ops.items():
    n_cells = len(op["cell_flat"])
    p0 = release(op, 31.5, -30.5)
    print(f"  {name}:")
    for days in [7, 30, 365]:
        p = propagate(op, p0, days=days)
        grid = to_grid(op, p)
        j, i = np.unravel_index(np.nanargmax(grid), grid.shape)
        exits = ", ".join(f"{l} {v:.0%}" for l, v in zip(EXIT_LABELS, p[n_cells:]))
        print(f"    {days:>3} d: in R {p[:n_cells].sum():5.1%} | exited {exits} | "
              f"peak cell {lon_edges[i] + RES/2:.1f}°E {lat_edges[j] + RES/2:.1f}°")
