"""
Agulhas Region — Velocity Decorrelation Time
How long a drifter's velocity remembers itself, which sets how short the lag τ
of a transition matrix can be while its steps stay nearly independent. For each
drogue type: velocity anomalies from the mean in each grid cell, their
autocorrelation over matching hourly pairs along each drifter, and the integral
timescale (autocorrelation integrated to MAX_LAG_H). Intervals come from N_BOOT
resamples of whole drifters. In R the core region (10–40°E, 45–25°S) is used, as
it holds the Agulhas Current itself; any other box is used whole.

    .venv/bin/python scripts/27_decorrelation_time.py --res 1
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt
import xarray as xr

from config import BOX, DROGUE_TYPES, TAU_DAYS, grid_tag, parse_args, region_prefix

args = parse_args(__doc__, grid=True)
RES = args.res
AREA = (10.0, 40.0, -45.0, -25.0) if tuple(args.box) == BOX else tuple(args.box)
TAG = grid_tag(RES, args.box)
MAX_LAG_H = 120
N_BOOT = 1000
SEED = 0
SERIES = {"drogued": ("#2a78d6", "-"), "undrogued": ("#eb6834", "--")}

os.makedirs("figures", exist_ok=True)


# ── 1. Load the area ─────────────────────────────────────────────────────────
ds = xr.open_dataset(args.data)
lon = ds["lon"].values.astype(float)
lat = ds["lat"].values.astype(float)
ve = ds["ve"].values.astype(float)
vn = ds["vn"].values.astype(float)
drogue = ds["drogue_status"].values.astype(bool)
hour = ds["time"].values.astype("datetime64[s]").astype(np.int64) // 3600
traj_idx = np.repeat(np.arange(ds.sizes["traj"]), ds["rowsize"].values)
in_area = ((lon >= AREA[0]) & (lon < AREA[1]) & (lat >= AREA[2]) & (lat < AREA[3])
           & np.isfinite(ve) & np.isfinite(vn))
hour_span = int(hour.max() - hour.min()) + MAX_LAG_H + 1


# ── 2. Autocorrelation per drifter, so whole drifters can be resampled ───────
def lag_sums(flag):
    """Per drifter and lag: the sum of anomaly products and the number of pairs."""
    m = np.where(in_area & (drogue == flag))[0]
    cell = ((lat[m] - AREA[2]) // RES).astype(np.int64) * 100_000 + ((lon[m] - AREA[0]) // RES).astype(np.int64)
    frame = pd.DataFrame({"cell": cell, "u": ve[m], "v": vn[m]})
    means = frame.groupby("cell")[["u", "v"]].transform("mean")
    up, vp = (frame["u"] - means["u"]).values, (frame["v"] - means["v"]).values
    # The subset is sorted by (trajectory, time), so these keys are sorted too
    key = traj_idx[m].astype(np.int64) * hour_span + (hour[m] - hour.min())
    drifters, d_idx = np.unique(traj_idx[m], return_inverse=True)
    products = np.zeros((len(drifters), MAX_LAG_H + 1))
    pairs = np.zeros((len(drifters), MAX_LAG_H + 1))
    for lag in range(MAX_LAG_H + 1):
        pos = np.searchsorted(key, key + lag)
        found = pos < len(key)
        found[found] = key[pos[found]] == (key + lag)[found]
        products[:, lag] = np.bincount(d_idx[found], up[found] * up[pos[found]] + vp[found] * vp[pos[found]],
                                       minlength=len(drifters))
        pairs[:, lag] = np.bincount(d_idx[found], minlength=len(drifters))
    return products, pairs, len(m)


def autocorrelation(products, pairs, w=None):
    """Autocorrelation per lag, with drifter weights w."""
    w = np.ones(len(products)) if w is None else w
    mean = (w[:, None] * products).sum(axis=0) / (w[:, None] * pairs).sum(axis=0)
    return mean / mean[0]


def timescale(r):
    """Integral of the autocorrelation over the lags, in days (trapezium rule)."""
    return ((r[0] + r[-1]) / 2 + r[1:-1].sum()) / 24


rng = np.random.default_rng(SEED)
curves, rows = [], []
tau_h = int(round(TAU_DAYS * 24))
for name, flag in DROGUE_TYPES.items():
    products, pairs, n_obs = lag_sums(flag)
    r = autocorrelation(products, pairs)
    n_drifters = len(products)
    draws = np.array([autocorrelation(products, pairs, np.bincount(rng.integers(n_drifters, size=n_drifters),
                                                                   minlength=n_drifters))
                      for _ in range(N_BOOT)])
    r_lo, r_hi = np.percentile(draws, [2.5, 97.5], axis=0)
    t_lo, t_hi = np.percentile([timescale(d) for d in draws], [2.5, 97.5])
    curves.append(pd.DataFrame({"drogue": name, "lag_hours": np.arange(MAX_LAG_H + 1), "autocorrelation": r,
                                "lo": r_lo, "hi": r_hi}))
    rows.append({"drogue": name, "drifters": n_drifters, "positions": n_obs, "timescale_days": timescale(r),
                 "lo": t_lo, "hi": t_hi, "autocorrelation_1d": r[24], "autocorrelation_at_tau": r[tau_h]})
curve = pd.concat(curves, ignore_index=True)
summary = pd.DataFrame(rows)
out_curve, out_summary = f"data/decorrelation_curve_{TAG}.csv", f"data/decorrelation_{TAG}.csv"
curve.to_csv(out_curve, index=False)
summary.to_csv(out_summary, index=False)


# ── 3. Figure ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.axhline(0, color="#777777", lw=1)
for name, (color, style) in SERIES.items():
    g = curve[curve["drogue"] == name]
    s = summary[summary["drogue"] == name].iloc[0]
    ax.fill_between(g["lag_hours"] / 24, g["lo"], g["hi"], color=color, alpha=0.2, lw=0)
    ax.plot(g["lag_hours"] / 24, g["autocorrelation"], color=color, ls=style, lw=2,
            label=f"{name.capitalize()}: {s.timescale_days:.2f} d ({s.lo:.2f}–{s.hi:.2f})")
ax.axvline(TAU_DAYS, color="#777777", lw=1, ls=":")
ax.text(TAU_DAYS + 0.05, 0.9, f"τ = {TAU_DAYS:g} d", fontsize=9, color="#555555")
ax.set_xlabel("Lag (days)")
ax.set_ylabel("Velocity autocorrelation")
ax.set_title(f"Velocity decorrelation over {AREA[0]:g} to {AREA[1]:g} lon, {AREA[2]:g} to {AREA[3]:g} lat", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, title="Integral timescale (95%)")
out_png = f"figures/{region_prefix(args.data)}_decorrelation_{TAG}.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()


# ── 4. Summary ───────────────────────────────────────────────────────────────
pd.set_option("display.width", 200)
print(summary.round(3).to_string(index=False))
print(f"\nSaved → {out_summary}, {out_curve}, {out_png}")
