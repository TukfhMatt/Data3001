"""
Agulhas Region — Oil Windage Range
How much the oil results depend on the assumed windage, across 3%, 3.5% and 4%
of the wind (proposal SQ4), against undrogued material. For all traffic, each
route (weights from 24) and each release point from 04: stranding by coast and
in total, eventually and within 30 days, days to strand, oil mass stranded with
surface decay, and exits.

Intervals come from N_BOOT resamples of whole drifters. All three oil matrices
use the same undrogued moves, so each resample draws the same drifters for all
of them, and the interval of the 4% − 3% difference shows whether the effect of
windage goes beyond sampling noise, which the separate intervals cannot.

A release point is flagged as depending on the method where oil and its drogued
check differ by TVD > DISAGREE_TVD at 30 days, at each windage.

Needs 09, 23 and 24.

    .venv/bin/python scripts/28_oil_windage.py --res 1 --tau 3.5
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt

from config import DISAGREE_TVD, OIL_WINDAGE_RANGE, grid_shape, operator_path, parse_args, release_points_path, setting_tag
from transport import (COAST_LABELS, EXIT_LABELS, STRANDED, check_coast_box, load_operator, propagate, release,
                       resampled_counts, row_normalise, state_of, stranding_chain)

args = parse_args(__doc__, grid=True, lag=True)
check_coast_box(args.box)
RES, TAU = args.res, args.tau
TAG = setting_tag(RES, TAU, args.box)
N_BOOT = 1000
SEED = 0
OIL = list(OIL_WINDAGE_RANGE)
LOW, HIGH = "oil_3pct", "oil_4pct"
MATRICES = ["undrogued"] + OIL
SHORT_COASTS = ["Atlantic & Cape", "SA south & east", "S Mozambique", "C & N Mozambique",
                "W & S Madagascar", "E Madagascar"]
SHORT_ROUTES = ["All traffic", "Cape west", "Cape east", "Natal", "Moz. Channel", "S of Madag.", "E of Madag."]
SERIES = {"undrogued": ("#eb6834", "s", "Undrogued"), "oil_3pct": ("#86d6b5", "o", "Oil 3%"),
          "oil": ("#1baf7a", "D", "Oil 3.5%"), "oil_4pct": ("#0b6b49", "^", "Oil 4%")}
HORIZON_DAYS = 30
YEAR_DAYS = 365

os.makedirs("figures", exist_ok=True)


# ── 1. Operators, per-drifter counts and releases ────────────────────────────
ops = {name: load_operator(operator_path(f"{name}_coast", RES, TAU, args.box)) for name in MATRICES}
op0 = ops["oil"]
cells = op0["cell_flat"]
n_cells, n_states = len(cells), op0["P"].shape[0]
n_lon, _ = grid_shape(args.box, RES)
counts_file = np.load(f"data/coast_counts_{TAG}.npz")
if int(counts_file["n_states"]) != n_states:
    raise SystemExit("data/coast_counts_….npz does not match the coast operators; rerun 23.")
counts = {name: tuple(counts_file[f"{name}_{part}"] for part in ("d_idx", "ft", "n"))
          + (int(counts_file[f"{name}_n_drifters"]),) for name in MATRICES}
if len({counts[name][3] for name in OIL}) != 1:
    raise SystemExit("The oil matrices do not share their drifters; rerun 09 and 23.")

traffic = np.load(f"data/traffic_releases_{TAG}.npz", allow_pickle=True)
points = pd.read_csv(release_points_path(args.box))[["name", "lon", "lat"]]
point_rows = np.zeros((len(points), n_cells))
point_rows[np.arange(len(points)), [state_of(op0, r.lon, r.lat) for r in points.itertuples()]] = 1.0
W = np.vstack([traffic["W"], point_rows])
RELEASES = np.r_[traffic["releases"].astype(str), points["name"].values]
kinds = np.r_[["traffic"] * len(traffic["W"]), ["point"] * len(points)]
horizon_steps = int(round(HORIZON_DAYS / TAU))
year_steps = int(round(YEAR_DAYS / TAU))
decay_days = float(op0["decay_days"])


# ── 2. Outcomes of one matrix ────────────────────────────────────────────────
def outcomes(P):
    """Per release: stranding per coast eventually and within HORIZON_DAYS, days
    to strand, exits, and the share of oil mass stranded within a year."""
    fate, visits = stranding_chain(P, n_cells, STRANDED)
    wf = W @ fate
    total = wf[:, STRANDED:].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        days = np.where(total > 0, W @ visits / total * TAU, np.nan)
    p = np.zeros((len(W), n_states))
    p[:, :n_cells] = W
    prev = p[:, n_cells + STRANDED:].copy()
    mass = np.zeros((len(W), len(COAST_LABELS)))
    for step in range(1, year_steps + 1):
        p = p @ P
        new = p[:, n_cells + STRANDED:]
        mass += (new - prev) * np.exp(-step * TAU / decay_days)
        prev = new.copy()
        if step == horizon_steps:
            within = new.copy()
    return {"eventual": wf[:, STRANDED:], "within": within, "days": days, "exit": wf[:, :STRANDED], "mass": mass}


full, draws = {}, {}
for name in MATRICES:
    full[name] = outcomes(ops[name]["P"])
    # The same seed for each matrix redraws the same drifters for all the oil matrices
    rng = np.random.default_rng(SEED)
    collected = {}
    for _ in range(N_BOOT):
        Pb = row_normalise(resampled_counts(counts[name], n_states, rng), cells, n_lon)[0]
        for key, value in outcomes(Pb).items():
            collected.setdefault(key, []).append(value)
    draws[name] = {key: np.array(value) for key, value in collected.items()}
    print(f"{name:<10} {counts[name][3]:,} drifters resampled {N_BOOT} times")


def interval(x):
    with np.errstate(all="ignore"):
        return np.nanpercentile(x, [2.5, 97.5], axis=0)


# ── 3. Long tables ───────────────────────────────────────────────────────────
rows = []
EXIT_NAMES = EXIT_LABELS[:STRANDED]
for name in MATRICES:
    for metric, targets in [("eventual", COAST_LABELS), ("within", COAST_LABELS), ("mass", COAST_LABELS),
                            ("exit", EXIT_NAMES)]:
        est, lo_hi = full[name][metric], interval(draws[name][metric])
        for k, target in enumerate(targets):
            rows += [{"matrix": name, "release": RELEASES[r], "kind": kinds[r], "metric": metric, "target": target,
                      "estimate": est[r, k], "lo": lo_hi[0][r, k], "hi": lo_hi[1][r, k]} for r in range(len(W))]
        if metric != "exit":
            lo_hi = interval(draws[name][metric].sum(axis=2))
            rows += [{"matrix": name, "release": RELEASES[r], "kind": kinds[r], "metric": metric, "target": "any",
                      "estimate": est[r].sum(), "lo": lo_hi[0][r], "hi": lo_hi[1][r]} for r in range(len(W))]
    lo_hi = interval(draws[name]["days"])
    rows += [{"matrix": name, "release": RELEASES[r], "kind": kinds[r], "metric": "days", "target": "any",
              "estimate": full[name]["days"][r], "lo": lo_hi[0][r], "hi": lo_hi[1][r]} for r in range(len(W))]
table = pd.DataFrame(rows)

# 4% − 3% on the same drifters
diff_rows = []
for metric in ["eventual", "within"]:
    delta = draws[HIGH][metric] - draws[LOW][metric]
    est = full[HIGH][metric] - full[LOW][metric]
    for k, target in enumerate(np.r_[COAST_LABELS, ["any"]]):
        d = delta.sum(axis=2) if target == "any" else delta[:, :, k]
        e = est.sum(axis=1) if target == "any" else est[:, k]
        lo, hi = interval(d)
        diff_rows += [{"release": RELEASES[r], "kind": kinds[r], "metric": metric, "target": target, "difference": e[r],
                       "lo": lo[r], "hi": hi[r], "beyond_noise": lo[r] > 0 or hi[r] < 0} for r in range(len(W))]
delta_days = draws[HIGH]["days"] - draws[LOW]["days"]
lo, hi = interval(delta_days)
diff_rows += [{"release": RELEASES[r], "kind": kinds[r], "metric": "days", "target": "any",
               "difference": full[HIGH]["days"][r] - full[LOW]["days"][r], "lo": lo[r], "hi": hi[r],
               "beyond_noise": lo[r] > 0 or hi[r] < 0} for r in range(len(W))]
diffs = pd.DataFrame(diff_rows)


# ── 4. Method flag at each windage ───────────────────────────────────────────
flag_rows = []
for name, alpha in OIL_WINDAGE_RANGE.items():
    oil = load_operator(operator_path(name, RES, TAU, args.box))
    check = load_operator(operator_path(name.replace("oil", "oil_check"), RES, TAU, args.box))
    for r in points.itertuples():
        a = propagate(oil, release(oil, r.lon, r.lat), days=HORIZON_DAYS)
        b = propagate(check, release(check, r.lon, r.lat), days=HORIZON_DAYS)
        tvd = 0.5 * np.abs(a - b).sum()
        flag_rows.append({"name": r.name, "windage": alpha, "tvd_30d": tvd, "method_flag": tvd > DISAGREE_TVD})
flags = pd.DataFrame(flag_rows)

out_table, out_diff, out_flags = [f"data/oil_windage{part}_{TAG}.csv" for part in ("", "_difference", "_method_flags")]
table.to_csv(out_table, index=False)
diffs.to_csv(out_diff, index=False)
flags.to_csv(out_flags, index=False)


# ── 5. Figure: all traffic by coast, and each route ──────────────────────────
routes = [r for r, k in zip(RELEASES, kinds) if k == "traffic"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
offsets = dict(zip(MATRICES, np.linspace(-0.27, 0.27, len(MATRICES))))
for ax, targets, labels, pick, title in [
        (axes[0], list(COAST_LABELS), SHORT_COASTS, lambda g: g[(g["release"] == "all") & (g["metric"] == "eventual")],
         "All traffic: where material eventually strands"),
        (axes[1], routes, SHORT_ROUTES, lambda g: g[(g["metric"] == "eventual") & (g["target"] == "any")],
         "Any coast, by route of release")]:
    for name in MATRICES:
        color, marker, label = SERIES[name]
        g = pick(table[table["matrix"] == name]).set_index("release" if targets is routes else "target").loc[targets]
        y = np.arange(len(targets))[::-1] + offsets[name]
        ax.hlines(y, 100 * g["lo"], 100 * g["hi"], color=color, lw=2)
        ax.scatter(100 * g["estimate"], y, color=color, marker=marker, s=38, edgecolor="white", zorder=3, label=label)
    ax.set_yticks(np.arange(len(targets))[::-1])
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Eventually stranded (%), with 95% interval")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.3)
axes[0].legend(fontsize=9, loc="lower right")
fig.suptitle(f"Oil windage range — {RES:g}°, τ = {TAU:g} d", fontsize=12)
out_png = f"figures/agulhas_oil_windage_{TAG}.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()


# ── 6. Summary ───────────────────────────────────────────────────────────────
pd.set_option("display.width", 220)


def show(metric, target, kind="traffic", scale=100, digits=0):
    g = table[(table["metric"] == metric) & (table["target"] == target) & (table["kind"] == kind)]
    text = g.assign(text=[f"{scale * e:.{digits}f} ({scale * lo:.{digits}f}–{scale * hi:.{digits}f})"
                          for e, lo, hi in zip(g["estimate"], g["lo"], g["hi"])])
    return text.pivot(index="release", columns="matrix", values="text")[MATRICES].reindex(
        [r for r, k in zip(RELEASES, kinds) if k == kind])


print("\nEventually stranded, any coast (%):")
print(show("eventual", "any").to_string())
print(f"\nStranded within {HORIZON_DAYS} days, any coast (%):")
print(show("within", "any").to_string())
print("\nDays to strand, among material that strands:")
print(show("days", "any", scale=1).to_string())
print("\nAll traffic, eventually stranded by coast (%):")
coast_rows = table[(table["release"] == "all") & (table["metric"] == "eventual") & (table["target"] != "any")]
print(coast_rows.assign(text=[f"{100 * e:.1f} ({100 * lo:.1f}–{100 * hi:.1f})" for e, lo, hi
                              in zip(coast_rows["estimate"], coast_rows["lo"], coast_rows["hi"])])
      .pivot(index="target", columns="matrix", values="text")[MATRICES].reindex(COAST_LABELS).to_string())
print(f"\nOil mass stranded within a year, any coast, with exp(−t / {decay_days:g} d) (%):")
print(show("mass", "any", digits=1).to_string())
print("\nRelease points, eventually stranded, any coast (%):")
print(show("eventual", "any", kind="point").to_string())
print("\n4% − 3% on the same drifters, all traffic and routes (points):")
sub = diffs[(diffs["kind"] == "traffic") & (diffs["target"] == "any")]
print(sub.assign(difference=lambda d: d["difference"].where(d["metric"] == "days", 100 * d["difference"]),
                 lo=lambda d: d["lo"].where(d["metric"] == "days", 100 * d["lo"]),
                 hi=lambda d: d["hi"].where(d["metric"] == "days", 100 * d["hi"]))
      [["release", "metric", "difference", "lo", "hi", "beyond_noise"]].round(1).to_string(index=False))
order = points["name"].tolist()
print(f"\nMethod flags (oil against its check, TVD > {DISAGREE_TVD} at {HORIZON_DAYS} d):")
print(flags.pivot(index="name", columns="windage", values="tvd_30d").reindex(order).round(2).to_string())
print("Flagged at each windage: " + ", ".join(f"{100 * a:g}%: {int(g['method_flag'].sum())} of {len(g)}"
                                              for a, g in flags.groupby("windage")))
print(f"\nSaved → {out_table}, {out_diff}, {out_flags}, {out_png}")
