"""
Agulhas Region — Velocity Decorrelation Time
How long a drifter's velocity remembers itself, which sets how short the lag τ
of a transition matrix can be while its steps stay nearly independent. For each
drogue type: velocity anomalies from the mean in each grid cell, their
autocorrelation over matching hourly pairs along each drifter, and the integral
timescale (autocorrelation integrated to MAX_LAG_H). Intervals come from N_BOOT
resamples of whole drifters. In R the core region (10–40°E, 45–25°S) is used, as
it holds the Agulhas Current itself; any other box is used whole.

    python -m agulhas decorrelation_time --res 1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agulhas.config import BOX, DROGUE_TYPES, TAU_DAYS, grid_tag, parse_args, region_prefix
from agulhas.drifters import Drifters, in_box
from agulhas.plotting import save_figure

CORE_AREA = (10.0, 40.0, -45.0, -25.0)
MAX_LAG_H = 120
N_BOOT = 1000
SEED = 0
SERIES = {"drogued": ("#2a78d6", "-"), "undrogued": ("#eb6834", "--")}


def autocorrelation(products, pairs, w=None):
    """Autocorrelation per lag, with drifter weights w."""
    w = np.ones(len(products)) if w is None else w
    mean = (w[:, None] * products).sum(axis=0) / (w[:, None] * pairs).sum(axis=0)
    return mean / mean[0]


def timescale(r):
    """Integral of the autocorrelation over the lags, in days (trapezium rule)."""
    return ((r[0] + r[-1]) / 2 + r[1:-1].sum()) / 24


def lag_sums(drifters, area, res, flag):
    """Per drifter and lag: the sum of anomaly products and the number of pairs."""
    lon, lat, ve, vn, traj_idx = drifters.lon, drifters.lat, drifters.ve, drifters.vn, drifters.traj_idx
    hour = drifters.t // 3600
    hour_span = int(hour.max() - hour.min()) + MAX_LAG_H + 1
    in_area = in_box(lon, lat, area) & np.isfinite(ve) & np.isfinite(vn)
    m = np.where(in_area & (drifters.drogue == flag))[0]
    cell = ((lat[m] - area[2]) // res).astype(np.int64) * 100_000 + ((lon[m] - area[0]) // res).astype(np.int64)
    frame = pd.DataFrame({"cell": cell, "u": ve[m], "v": vn[m]})
    means = frame.groupby("cell")[["u", "v"]].transform("mean")
    up, vp = (frame["u"] - means["u"]).values, (frame["v"] - means["v"]).values
    # The subset is sorted by (trajectory, time), so these keys are sorted too
    key = traj_idx[m].astype(np.int64) * hour_span + (hour[m] - hour.min())
    ids, d_idx = np.unique(traj_idx[m], return_inverse=True)
    products = np.zeros((len(ids), MAX_LAG_H + 1))
    pairs = np.zeros((len(ids), MAX_LAG_H + 1))
    for lag in range(MAX_LAG_H + 1):
        pos = np.searchsorted(key, key + lag)
        found = pos < len(key)
        found[found] = key[pos[found]] == (key + lag)[found]
        products[:, lag] = np.bincount(d_idx[found], up[found] * up[pos[found]] + vp[found] * vp[pos[found]],
                                       minlength=len(ids))
        pairs[:, lag] = np.bincount(d_idx[found], minlength=len(ids))
    return products, pairs, len(m)


def decorrelation_of(name, products, pairs, n_obs, rng):
    """One drogue type's autocorrelation with intervals: (curve table, summary row)."""
    r = autocorrelation(products, pairs)
    n_drifters = len(products)
    draws = np.array([autocorrelation(products, pairs, np.bincount(rng.integers(n_drifters, size=n_drifters),
                                                                   minlength=n_drifters))
                      for _ in range(N_BOOT)])
    r_lo, r_hi = np.percentile(draws, [2.5, 97.5], axis=0)
    t_lo, t_hi = np.percentile([timescale(d) for d in draws], [2.5, 97.5])
    curve = pd.DataFrame({"drogue": name, "lag_hours": np.arange(MAX_LAG_H + 1), "autocorrelation": r,
                          "lo": r_lo, "hi": r_hi})
    row = {"drogue": name, "drifters": n_drifters, "positions": n_obs, "timescale_days": timescale(r),
           "lo": t_lo, "hi": t_hi, "autocorrelation_1d": r[24], "autocorrelation_at_tau": r[int(round(TAU_DAYS * 24))]}
    return curve, row


def plot_curves(curve, summary, area, path):
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
    ax.set_title(f"Velocity decorrelation over {area[0]:g} to {area[1]:g} lon, {area[2]:g} to {area[3]:g} lat",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, title="Integral timescale (95%)")
    save_figure(path)


def main(argv=None):
    args = parse_args(__doc__, grid=True, argv=argv)
    area = CORE_AREA if tuple(args.box) == BOX else tuple(args.box)
    tag = grid_tag(args.res, args.box)
    out_curve, out_summary = f"data/decorrelation_curve_{tag}.csv", f"data/decorrelation_{tag}.csv"
    out_png = f"figures/{region_prefix(args.data)}_decorrelation_{tag}.png"

    drifters = Drifters(args.data)
    rng = np.random.default_rng(SEED)
    curves, rows = [], []
    for name, flag in DROGUE_TYPES.items():
        curve, row = decorrelation_of(name, *lag_sums(drifters, area, args.res, flag), rng)
        curves.append(curve)
        rows.append(row)
    curve, summary = pd.concat(curves, ignore_index=True), pd.DataFrame(rows)

    curve.to_csv(out_curve, index=False)
    summary.to_csv(out_summary, index=False)
    plot_curves(curve, summary, area, out_png)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    print(f"\nSaved → {out_summary}, {out_curve}, {out_png}")


if __name__ == "__main__":
    main()
