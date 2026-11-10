"""
Agulhas Region — OpenDrift Extra Checks
Two checks added after opendrift_compare's fixed rule gave its result, reported
as later additions; that rule and its table are left as they are.

  1. Afloat only. OpenDrift strands a particle the first time it touches its
     landmask, and the matrix counts only drifters recorded as aground, so the
     two count stranding differently. The distance over what is still afloat
     (ocean cells and exits, rescaled to sum to one) leaves stranding out.
  2. The method's own error. A matrix built from OpenDrift's own moves over τ
     (opendrift_moves) with the same helpers as every matrix, run from the same
     cells, against OpenDrift's spread: the gap is what the matrix method costs
     (memory lost between steps, mixing within a cell) with the physics held the
     same. The oil matrix against that OpenDrift matrix is then the difference in
     data and physics with the method held the same.

Intervals resample whole start dates, rebuilding the OpenDrift matrix each time.
Resampling adds noise to both sides of a distance, which raises it, so the
intervals are basic bootstrap ones (twice the estimate less each percentile),
which take that rise back out. The OpenDrift matrix is fitted to the same particles it is compared with, so its
distance is a lower bound on the method's error.

    .venv-opendrift/bin/python -m agulhas opendrift_checks
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agulhas.comparison.opendrift_compare import load_runs, reduce_states, tvd
from agulhas.config import DISAGREE_TVD, grid_shape, operator_path, parse_args, release_points_path, setting_tag
from agulhas.opendrift_common import HOURS, N_COASTS, OUT, YEARS
from agulhas.plotting import save_figure
from agulhas.transport import check_coast_box, load_operator, row_normalise, state_of

N_BOOT = 200
SEED = 0
COLORS = {"matrix": "#2a78d6", "od_matrix": "#eb6834", "opendrift": "#10263B"}


def afloat(dist, n_cells):
    """Ocean cells and exits only, rescaled to sum to one (zeros if nothing is afloat)."""
    a = dist[..., :n_cells + 4]
    return a / np.maximum(a.sum(axis=-1, keepdims=True), 1e-12)


def stranded(dist, n_cells):
    return dist[..., n_cells + 4:n_cells + 4 + N_COASTS].sum(axis=-1)


# ── 1. The OpenDrift matrix ──────────────────────────────────────────────────

def load_moves():
    """Each start's moves (starts, ocean cells, states), in the order of opendrift_points' runs."""
    paths = sorted(glob.glob(f"{OUT}/moves/*.npz"))
    runs = sorted(glob.glob(f"{OUT}/points/*.npz"))
    if [os.path.basename(p) for p in paths] != [os.path.basename(p) for p in runs]:
        raise SystemExit(f"{OUT}/moves does not hold a run for every start in {OUT}/points: run opendrift_moves first.")
    files = [np.load(p) for p in paths]
    moves = np.stack([f["moves"] for f in files])
    particles = np.sum([f["particles"] for f in files], axis=0)
    print(f"{len(paths)} starts, {moves.sum():,} moves over τ; {int((particles > 0).sum())} ocean cells have moves")
    return moves, particles


def od_matrix(moves, cells, n_lon):
    """Row-stochastic matrix from summed moves, with the absorbing states and empty rows as transition_matrix makes them."""
    n_cells, n_to = moves.shape
    C = np.zeros((n_to, n_to))
    C[:n_cells] = moves
    P, empty = row_normalise(C, cells, n_lon)
    return P, empty


def spreads(P, states, steps):
    """Each start state's distribution after each number of steps, with a zero for 'no state': (states, horizons, states + 1)."""
    powers = [np.linalg.matrix_power(P, s) for s in steps]
    return np.stack([np.stack([np.append(Pk[st], 0.0) for Pk in powers]) for st in states])


# ── 2. Distances at each point and horizon ───────────────────────────────────

def distances(matrix, od_m, od, n_cells):
    """The three distances between the oil matrix, the OpenDrift matrix and OpenDrift, in full and afloat only."""
    m, o, d = (reduce_states(x, n_cells) for x in (matrix, od_m, od))
    am, ao, ad = (afloat(x, n_cells) for x in (matrix, od_m, od))
    return {"tvd_matrix_opendrift": tvd(m, d), "tvd_afloat_matrix_opendrift": tvd(am, ad),
            "tvd_method": tvd(o, d), "tvd_afloat_method": tvd(ao, ad),
            "tvd_data": tvd(m, o), "tvd_afloat_data": tvd(am, ao)}


def year_to_year_afloat(run, years, n_cells):
    """Mean afloat distance between each year of OpenDrift and its other years."""
    return float(np.mean([tvd(afloat(run[years == y].mean(axis=0), n_cells), afloat(run[years != y].mean(axis=0), n_cells))
                          for y in YEARS if (years == y).any() and (years != y).any()]))


def method_distance(od_m, od, n_cells):
    """The OpenDrift matrix against OpenDrift, full and afloat: (2, points, horizons)."""
    return np.stack([tvd(reduce_states(od_m, n_cells), reduce_states(od, n_cells)), tvd(afloat(od_m, n_cells), afloat(od, n_cells))])


def boot_method(moves, shares, cells, n_lon, states, steps, n_cells, estimate, rng):
    """Basic bootstrap 95% intervals of the method's distance, full and afloat, from resampling
    whole start dates: (2, 2, points, horizons), lower bounds then upper."""
    n = len(moves)
    out = np.empty((N_BOOT, 2, len(states), len(steps)))
    for b in range(N_BOOT):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        P, _ = od_matrix(np.tensordot(w, moves, axes=1), cells, n_lon)
        od_m = spreads(P, states, steps)
        out[b] = method_distance(od_m, np.tensordot(w, shares, axes=1) / n, n_cells)
    q_lo, q_hi = np.percentile(out, [2.5, 97.5], axis=0)
    return np.stack([np.maximum(2 * estimate - q_hi, 0), 2 * estimate - q_lo])


def check_points(points, op, moves, years, shares, args, rng):
    """One row per release point and horizon."""
    n_cells, cells = len(op["cell_flat"]), op["cell_flat"]
    n_lon, _ = grid_shape(args.box, args.res)
    states = [state_of(op, p.lon, p.lat) for _, p in points.iterrows()]
    steps = [int(round(days / args.tau)) for days in HOURS]
    P, empty = od_matrix(moves.sum(axis=0), cells, n_lon)
    print(f"OpenDrift matrix: {len(empty)} of {n_cells} ocean rows have no moves of their own")
    matrix, od_m = spreads(op["P"], states, steps), spreads(P, states, steps)
    od = shares.mean(axis=0)
    ci = boot_method(moves, shares, cells, n_lon, states, steps, n_cells, method_distance(od_m, od, n_cells), rng)
    rows = []
    for k, p in points.iterrows():
        for h, days in enumerate(HOURS):
            row = {"name": p["name"], "lon": p.lon, "lat": p.lat, "days": days}
            row.update(distances(matrix[k, h], od_m[k, h], od[k, h], n_cells))
            row.update({"tvd_method_lo": ci[0, 0, k, h], "tvd_method_hi": ci[1, 0, k, h],
                        "tvd_afloat_method_lo": ci[0, 1, k, h], "tvd_afloat_method_hi": ci[1, 1, k, h],
                        "year_to_year_afloat_tvd": year_to_year_afloat(shares[:, k, h], years, n_cells),
                        "stranded_matrix": stranded(matrix[k, h], n_cells), "stranded_od_matrix": stranded(od_m[k, h], n_cells),
                        "stranded_opendrift": stranded(od[k, h], n_cells)})
            rows.append(row)
    return pd.DataFrame(rows)


# ── 3. Summary ───────────────────────────────────────────────────────────────

def print_summary(final):
    cols = ["name", "tvd_matrix_opendrift", "tvd_afloat_matrix_opendrift", "year_to_year_afloat_tvd", "tvd_method", "tvd_data",
            "stranded_matrix", "stranded_od_matrix", "stranded_opendrift"]
    print(final[cols].round(3).to_string(index=False))
    print(f"\nAt day 31.5, median distance: matrix against OpenDrift {final['tvd_matrix_opendrift'].median():.2f}, "
          f"afloat only {final['tvd_afloat_matrix_opendrift'].median():.2f} (OpenDrift's years against each other {final['year_to_year_afloat_tvd'].median():.2f}).")
    print(f"Afloat distance at most {DISAGREE_TVD} at {(final['tvd_afloat_matrix_opendrift'] <= DISAGREE_TVD).sum()} of {len(final)} points.")
    print(f"The method's own error (OpenDrift matrix against OpenDrift): median {final['tvd_method'].median():.2f}, afloat {final['tvd_afloat_method'].median():.2f}; "
          f"data and physics (oil matrix against OpenDrift matrix): median {final['tvd_data'].median():.2f}, afloat {final['tvd_afloat_data'].median():.2f}.")
    print(f"Stranded, mean over points: oil matrix {final['stranded_matrix'].mean():.1%}, OpenDrift matrix {final['stranded_od_matrix'].mean():.1%}, "
          f"OpenDrift {final['stranded_opendrift'].mean():.1%}.")


def plot_distances(a, final, y):
    bars = [("tvd_matrix_opendrift", "Oil matrix against OpenDrift", "#2a78d6"),
            ("tvd_afloat_matrix_opendrift", "… afloat only", "#9cc3ee"),
            ("tvd_method", "OpenDrift matrix against OpenDrift (method)", "#eb6834"),
            ("tvd_data", "Oil matrix against OpenDrift matrix (data)", "#7a7a7a")]
    h = 0.2
    for i, (col, label, color) in enumerate(bars):
        a.barh(y + (1.5 - i) * h, final[col], height=h, color=color, label=label)
    a.hlines(y - 0.5 * h, final["tvd_method_lo"], final["tvd_method_hi"], color="#10263B", lw=0.8)
    a.axvline(DISAGREE_TVD, color="#10263B", lw=1, ls="--", label=f"Limit {DISAGREE_TVD}")
    a.set_xlabel("Total variation distance at day 31.5")
    a.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2)


def plot_stranding(a, final, y):
    for i, (col, label) in enumerate([("stranded_matrix", "Oil matrix"), ("stranded_od_matrix", "OpenDrift matrix"), ("stranded_opendrift", "OpenDrift")]):
        a.scatter(final[col] * 100, y + (1 - i) * 0.2, s=18, color=list(COLORS.values())[i], label=label)
    a.set_xlabel("Stranded within 31.5 days (%)")
    a.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2)


def plot_checks(final, png):
    y = np.arange(len(final))[::-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 0.45 * len(final) + 2), sharey=True)
    plot_distances(a, final, y)
    plot_stranding(b, final, y)
    a.set_yticks(y)
    a.set_yticklabels(final["name"], fontsize=8)
    fig.suptitle("OpenDrift extra checks: stranding left out, and the matrix method's own error")
    save_figure(png)


def main(argv=None):
    """Run this step; see the module docstring for what it does."""
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    check_coast_box(args.box)
    tag = setting_tag(args.res, args.tau, args.box)
    out = f"{OUT}/checks_{tag}.csv"
    png = f"figures/agulhas_opendrift_checks_{tag}.png"
    points = pd.read_csv(release_points_path(args.box))
    rng = np.random.default_rng(SEED)

    years, shares = load_runs()
    moves, _ = load_moves()
    op = load_operator(operator_path("oil_coast", args.res, args.tau, args.box))
    table = check_points(points, op, moves, years, shares, args, rng)
    table.to_csv(out, index=False)

    final = table[table["days"] == 31.5]
    print_summary(final)
    plot_checks(final, png)
    print(f"Saved → {out}, {png}")


if __name__ == "__main__":
    main()
