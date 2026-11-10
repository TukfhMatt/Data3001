"""
Agulhas Region — Oil Windage Range
How much the oil results depend on the assumed windage, across 3%, 3.5% and 4%
of the wind (proposal SQ4), against undrogued material. For all traffic, each
route (weights from traffic_exposure) and each release point from
shipping_overlay: stranding by coast and in total, eventually and within 30 days, days to strand, oil mass stranded with
surface decay, and exits.

Intervals come from N_BOOT resamples of whole drifters. All three oil matrices
use the same undrogued moves, so each resample draws the same drifters for all
of them, and the interval of the 4% − 3% difference shows whether the effect of
windage goes beyond sampling noise, which the separate intervals cannot.

A release point is flagged as depending on the method where oil and its drogued
check differ by TVD > DISAGREE_TVD at 30 days, at each windage.

Needs oil_wind, coastal_stranding and traffic_exposure.

    python -m agulhas oil_windage --res 1 --tau 3.5
"""

from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agulhas.config import (DISAGREE_TVD, OIL_RESIDUE_DAYS, OIL_WINDAGE_RANGE, grid_shape, oil_mass_left, operator_path,
                            parse_args, release_points_path, setting_tag)
from agulhas.plotting import save_figure
from agulhas.stats import interval
from agulhas.transport import (COAST_LABELS, EXIT_LABELS, STRANDED, check_coast_box, load_coast_counts,
                               load_operator, propagate, release, resampled_counts, row_normalise, state_of,
                               stranding_chain)

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
EXIT_NAMES = EXIT_LABELS[:STRANDED]

# Rows of W are releases (all traffic, each route, then each release point) spread over the ocean cells
Releases = namedtuple("Releases", "W names kinds")


# ── Operators, per-drifter counts and releases ───────────────────────────────

def load_releases(op, traffic_path, points):
    """Traffic releases from traffic_exposure, then one release per release point."""
    traffic = np.load(traffic_path, allow_pickle=True)
    point_rows = np.zeros((len(points), len(op["cell_flat"])))
    point_rows[np.arange(len(points)), [state_of(op, r.lon, r.lat) for r in points.itertuples()]] = 1.0
    return Releases(W=np.vstack([traffic["W"], point_rows]),
                    names=np.r_[traffic["releases"].astype(str), points["name"].values],
                    kinds=np.r_[["traffic"] * len(traffic["W"]), ["point"] * len(points)])


# ── Outcomes of one matrix ───────────────────────────────────────────────────

def outcomes(P, W, n_cells, tau):
    """Per release: stranding per coast eventually and within HORIZON_DAYS, days
    to strand, exits, and the share of heavy fuel oil mass stranded within a year,
    for each residue time of OIL_RESIDUE_DAYS (mass, mass_30d, mass_120d)."""
    horizon_steps = int(round(HORIZON_DAYS / tau))
    fate, visits = stranding_chain(P, n_cells, STRANDED)
    wf = W @ fate
    total = wf[:, STRANDED:].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        days = np.where(total > 0, W @ visits / total * tau, np.nan)
    p = np.zeros((len(W), P.shape[0]))
    p[:, :n_cells] = W
    prev = p[:, n_cells + STRANDED:].copy()
    mass = {key: np.zeros((len(W), len(COAST_LABELS))) for key in OIL_RESIDUE_DAYS}
    for step in range(1, int(round(YEAR_DAYS / tau)) + 1):
        p = p @ P
        new = p[:, n_cells + STRANDED:]
        for key, residue in OIL_RESIDUE_DAYS.items():
            mass[key] += (new - prev) * oil_mass_left(step * tau, residue)
        prev = new.copy()
        if step == horizon_steps:
            within = new.copy()
    return {"eventual": wf[:, STRANDED:], "within": within, "days": days, "exit": wf[:, :STRANDED],
            **mass}


def resample_all(ops, counts, W, args):
    """Outcomes of every matrix on the full data and on N_BOOT resamples of its drifters.
    The same seed for each matrix redraws the same drifters for all the oil matrices."""
    n_lon, _ = grid_shape(args.box, args.res)
    full, draws = {}, {}
    for name in MATRICES:
        op = ops[name]
        n_cells, n_states = len(op["cell_flat"]), op["P"].shape[0]
        full[name] = outcomes(op["P"], W, n_cells, args.tau)
        rng = np.random.default_rng(SEED)
        collected = {}
        for _ in range(N_BOOT):
            Pb = row_normalise(resampled_counts(counts[name], n_states, rng), op["cell_flat"], n_lon)[0]
            for key, value in outcomes(Pb, W, n_cells, args.tau).items():
                collected.setdefault(key, []).append(value)
        draws[name] = {key: np.array(value) for key, value in collected.items()}
        print(f"{name:<10} {counts[name][3]:,} drifters resampled {N_BOOT} times")
    return full, draws


# ── Long tables ──────────────────────────────────────────────────────────────

def outcome_table(full, draws, rel):
    """Every matrix's outcomes per release and target, with their intervals."""
    n = len(rel.W)

    def rows_of(name, metric, target, est, lo_hi):
        return [{"matrix": name, "release": rel.names[r], "kind": rel.kinds[r], "metric": metric, "target": target,
                 "estimate": est[r], "lo": lo_hi[0][r], "hi": lo_hi[1][r]} for r in range(n)]

    rows = []
    for name in MATRICES:
        for metric, targets in [("eventual", COAST_LABELS), ("within", COAST_LABELS)] \
                + [(key, COAST_LABELS) for key in OIL_RESIDUE_DAYS] + [("exit", EXIT_NAMES)]:
            est, lo_hi = full[name][metric], interval(draws[name][metric])
            for k, target in enumerate(targets):
                rows += rows_of(name, metric, target, est[:, k], (lo_hi[0][:, k], lo_hi[1][:, k]))
            if metric != "exit":
                rows += rows_of(name, metric, "any", est.sum(axis=1), interval(draws[name][metric].sum(axis=2)))
        rows += rows_of(name, "days", "any", full[name]["days"], interval(draws[name]["days"]))
    return pd.DataFrame(rows)


def difference_table(full, draws, rel):
    """4% − 3% on the same drifters, per release and target."""
    def rows_of(metric, target, est, d):
        lo, hi = interval(d)
        return [{"release": rel.names[r], "kind": rel.kinds[r], "metric": metric, "target": target,
                 "difference": est[r], "lo": lo[r], "hi": hi[r], "beyond_noise": lo[r] > 0 or hi[r] < 0}
                for r in range(len(rel.W))]

    rows = []
    for metric in ["eventual", "within"]:
        delta = draws[HIGH][metric] - draws[LOW][metric]
        est = full[HIGH][metric] - full[LOW][metric]
        for k, target in enumerate(np.r_[COAST_LABELS, ["any"]]):
            if target == "any":
                rows += rows_of(metric, target, est.sum(axis=1), delta.sum(axis=2))
            else:
                rows += rows_of(metric, target, est[:, k], delta[:, :, k])
    rows += rows_of("days", "any", full[HIGH]["days"] - full[LOW]["days"], draws[HIGH]["days"] - draws[LOW]["days"])
    return pd.DataFrame(rows)


def method_flags(points, args):
    """Release points where oil and its drogued check differ by TVD > DISAGREE_TVD at 30 days, at each windage."""
    rows = []
    for name, alpha in OIL_WINDAGE_RANGE.items():
        oil = load_operator(operator_path(name, args.res, args.tau, args.box))
        check = load_operator(operator_path(name.replace("oil", "oil_check"), args.res, args.tau, args.box))
        for r in points.itertuples():
            a = propagate(oil, release(oil, r.lon, r.lat), days=HORIZON_DAYS)
            b = propagate(check, release(check, r.lon, r.lat), days=HORIZON_DAYS)
            tvd = 0.5 * np.abs(a - b).sum()
            rows.append({"name": r.name, "windage": alpha, "tvd_30d": tvd, "method_flag": tvd > DISAGREE_TVD})
    return pd.DataFrame(rows)


# ── Figure: all traffic by coast, and each route ─────────────────────────────

def plot_windage(table, rel, args, path):
    routes = [r for r, k in zip(rel.names, rel.kinds) if k == "traffic"]
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
    fig.suptitle(f"Oil windage range — {args.res:g}°, τ = {args.tau:g} d", fontsize=12)
    save_figure(path)


# ── Summary ──────────────────────────────────────────────────────────────────

def pivot(table, rel, metric, target, kind="traffic", scale=100, digits=0):
    """Estimate and interval per release (rows) and matrix (columns), as text."""
    g = table[(table["metric"] == metric) & (table["target"] == target) & (table["kind"] == kind)]
    text = g.assign(text=[f"{scale * e:.{digits}f} ({scale * lo:.{digits}f}–{scale * hi:.{digits}f})"
                          for e, lo, hi in zip(g["estimate"], g["lo"], g["hi"])])
    return text.pivot(index="release", columns="matrix", values="text")[MATRICES].reindex(
        [r for r, k in zip(rel.names, rel.kinds) if k == kind])


def print_stranding(table, rel):
    pd.set_option("display.width", 220)
    print("\nEventually stranded, any coast (%):")
    print(pivot(table, rel, "eventual", "any").to_string())
    print(f"\nStranded within {HORIZON_DAYS} days, any coast (%):")
    print(pivot(table, rel, "within", "any").to_string())
    print("\nDays to strand, among material that strands:")
    print(pivot(table, rel, "days", "any", scale=1).to_string())
    print("\nAll traffic, eventually stranded by coast (%):")
    coast_rows = table[(table["release"] == "all") & (table["metric"] == "eventual") & (table["target"] != "any")]
    print(coast_rows.assign(text=[f"{100 * e:.1f} ({100 * lo:.1f}–{100 * hi:.1f})" for e, lo, hi
                                  in zip(coast_rows["estimate"], coast_rows["lo"], coast_rows["hi"])])
          .pivot(index="target", columns="matrix", values="text")[MATRICES].reindex(COAST_LABELS).to_string())
    for metric, residue in OIL_RESIDUE_DAYS.items():
        print(f"\nHeavy fuel oil mass stranded within a year, any coast, residue lost over {residue:g} d (%):")
        print(pivot(table, rel, metric, "any", digits=1).to_string())
    print("\nRelease points, eventually stranded, any coast (%):")
    print(pivot(table, rel, "eventual", "any", kind="point").to_string())


def print_differences(diffs, flags, order):
    print("\n4% − 3% on the same drifters, all traffic and routes (points):")
    sub = diffs[(diffs["kind"] == "traffic") & (diffs["target"] == "any")]
    print(sub.assign(difference=lambda d: d["difference"].where(d["metric"] == "days", 100 * d["difference"]),
                     lo=lambda d: d["lo"].where(d["metric"] == "days", 100 * d["lo"]),
                     hi=lambda d: d["hi"].where(d["metric"] == "days", 100 * d["hi"]))
          [["release", "metric", "difference", "lo", "hi", "beyond_noise"]].round(1).to_string(index=False))
    print(f"\nMethod flags (oil against its check, TVD > {DISAGREE_TVD} at {HORIZON_DAYS} d):")
    print(flags.pivot(index="name", columns="windage", values="tvd_30d").reindex(order).round(2).to_string())
    print("Flagged at each windage: " + ", ".join(f"{100 * a:g}%: {int(g['method_flag'].sum())} of {len(g)}"
                                                  for a, g in flags.groupby("windage")))


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    check_coast_box(args.box)
    tag = setting_tag(args.res, args.tau, args.box)
    out_table, out_diff, out_flags = [f"data/oil_windage{part}_{tag}.csv" for part in ("", "_difference", "_method_flags")]
    out_png = f"figures/agulhas_oil_windage_{tag}.png"

    ops = {name: load_operator(operator_path(f"{name}_coast", args.res, args.tau, args.box)) for name in MATRICES}
    counts = load_coast_counts(f"data/coast_counts_{tag}.npz", MATRICES, ops["oil"]["P"].shape[0])
    if len({counts[name][3] for name in OIL}) != 1:
        raise SystemExit("The oil matrices do not share their drifters; rerun oil_wind and coastal_stranding.")
    points = pd.read_csv(release_points_path(args.box))[["name", "lon", "lat"]]
    rel = load_releases(ops["oil"], f"data/traffic_releases_{tag}.npz", points)

    full, draws = resample_all(ops, counts, rel.W, args)
    table, diffs = outcome_table(full, draws, rel), difference_table(full, draws, rel)
    flags = method_flags(points, args)
    table.to_csv(out_table, index=False)
    diffs.to_csv(out_diff, index=False)
    flags.to_csv(out_flags, index=False)

    plot_windage(table, rel, args, out_png)
    print_stranding(table, rel)
    print_differences(diffs, flags, points["name"].tolist())
    print(f"\nSaved → {out_table}, {out_diff}, {out_flags}, {out_png}")


if __name__ == "__main__":
    main()
