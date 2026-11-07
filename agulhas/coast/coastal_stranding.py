"""
Agulhas Region — Stranding by Coast
Splits the stranded state of each operator by the coast where drifters ran
aground, so stranding can be reported per named coast (transport.COAST_LABELS:
six coasts of R, each with ≥ 13 grounded drifters). A stranding move takes the
coast of its drifter's last observation (transport.coast_of); every other
move is unchanged, so the coast columns sum to the operator's stranded column
exactly, which is checked.

Built for drogued and undrogued drifters (from the subset, as transition_matrix) and for oil
and the oil check (from the per-drifter moves saved by oil_wind), and saved as
operator_path("{name}_coast", …) with exit_labels W, E, S, N and the six
coasts. Reports, for every cell and for the release points from
shipping_overlay:
  • eventual stranding on each coast, exact from the absorbing chain
  • stranding on each coast within 7, 30 and 365 days
  • expected time to strand, among material that strands
with 95% intervals for the release points from N_BOOT resamples of whole
drifters (as bootstrap). Drogued stranding is a lower bound (see drogue_loss).

Needs the matrices from transition_matrix and oil_wind, data/oil_moves_….npz
from oil_wind and the release points from shipping_overlay (same box, default
grid and lag).

    python -m agulhas coastal_stranding --res 1 --tau 3.5
"""

from collections import namedtuple
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from agulhas.config import DROGUE_TYPES, MIN_DRIFTERS, operator_path, parse_args, release_points_path, setting_tag
from agulhas.drifters import Drifters
from agulhas.moves import MoveSet
from agulhas.plotting import MAP_KW, TRANSFORM_KW, map_axes, save_figure
from agulhas.stats import interval
from agulhas.transport import (COAST_COLORS, COAST_LABELS, COAST_NAMES, EXIT_LABELS, STRANDED, cell_centres,
                               check_coast_box, coast_of, last_obs_index, load_operator, reliable, resampled_counts,
                               row_normalise, state_of, stranding_chain, to_grid)

N_BOOT = 1000
SEED = 0
HORIZONS = [7, 30, 365]
MATRICES = ["drogued", "undrogued", "oil", "oil_check"]
WINDAGE_MATRICES = ["oil_3pct", "oil_4pct"]   # split for risk_atlas and oil_windage, which report them
COAST_EXIT_LABELS = np.r_[EXIT_LABELS[:STRANDED], COAST_LABELS]
MIN_SHOWN, SOLID = 0.05, 0.20       # map: pale from 5% eventual stranding, solid from 20%

# States of the base operators (n_base) and of the coast operators (n_states), and the first stranded column
Layout = namedtuple("Layout", "n_cells n_base n_states strand_col")


def fmt(e, lo, hi):
    return f"{e:.0%} ({lo:.0%}–{hi:.0%})"


# ── Coast of each grounding ──────────────────────────────────────────────────

def grounding_coasts(drifters, in_R):
    """Per drifter, the coast where it ran aground inside the box, or -1."""
    strand_t = drifters.strand_times(in_R)
    last = np.unique(last_obs_index(drifters.traj_idx))
    grounded = last[strand_t[last] >= 0]
    coast_of_traj = np.full(drifters.ds.sizes["traj"], -1)
    coast_of_traj[drifters.traj_idx[grounded]] = coast_of(drifters.lon[grounded], drifters.lat[grounded])
    return coast_of_traj, strand_t


# ── Coast operators: stranding moves split by the drifter's coast ────────────

def base_moves(name, moves, cells, layout, oil_moves):
    """Moves per (drifter, from, to) of a base operator: from the subset for the
    drogue types, from oil_wind's saved moves for oil. (traj, flat index, moves)."""
    if name in DROGUE_TYPES:
        s, e, st = moves.of_type(DROGUE_TYPES[name])
        f, to = moves.states(cells, s, e, st)
        n_base = layout.n_base
        key = moves.drifters.traj_idx[s].astype(np.int64) * n_base**2 + f * n_base + to
        uniq, n_moves = np.unique(key, return_counts=True)
        return uniq // n_base**2, uniq % n_base**2, n_moves
    if int(oil_moves["n_states"]) != layout.n_base:
        raise SystemExit("data/oil_moves_….npz does not match the matrices; rerun oil_wind.")
    return tuple(oil_moves[f"{name}_{k}"] for k in ("traj", "ft", "n"))


def split_moves(traj, ft, n, coast_of_traj, layout):
    """Moves per (drifter, from, to) on the base layout → coast layout, as
    counts for resampled_counts: (drifter number, flat index, moves, drifters)."""
    f, to = np.divmod(ft, layout.n_base)
    strands = to == layout.strand_col
    if (coast_of_traj[traj[strands]] < 0).any():
        raise SystemExit("A stranding move belongs to a drifter with no grounding coast.")
    to = np.where(strands, layout.strand_col + coast_of_traj[traj], to)
    drifters, d_idx = np.unique(traj, return_inverse=True)
    return d_idx, f * layout.n_states + to, n, len(drifters)


def coast_operator(name, base_op, counts, layout, n_lon, args):
    """The base operator with its stranded state split by coast. Stops unless the
    split reproduces the saved base operator exactly."""
    n_cells, strand_col, n_states = layout.n_cells, layout.strand_col, layout.n_states
    C = np.bincount(counts[1], weights=counts[2], minlength=n_states**2).reshape(n_states, n_states)
    P, empty = row_normalise(C, base_op["cell_flat"], n_lon)
    P0 = base_op["P"]
    same_rest = np.allclose(P[:n_cells, :strand_col], P0[:n_cells, :strand_col])
    same_strand = np.allclose(P[:n_cells, strand_col:].sum(axis=1), P0[:n_cells, strand_col])
    if not (same_rest and same_strand and np.array_equal(np.sort(empty), np.where(base_op["empty"])[0])):
        raise SystemExit(f"The coast split of {name} does not reproduce {operator_path(name, args.res, args.tau, args.box)}.")
    return dict(base_op, P=P, C=C, exit_labels=COAST_EXIT_LABELS, coast_names=COAST_NAMES)


def coast_support(name, traj, ft, coast_of_traj, layout):
    """Grounded drifters behind each coast of one matrix."""
    strand_traj = traj[ft % layout.n_base == layout.strand_col]
    rows = []
    for k, label in enumerate(COAST_LABELS):
        n_dr = len(np.unique(strand_traj[coast_of_traj[strand_traj] == k]))
        rows.append({"matrix": name, "coast": label, "grounded_drifters": n_dr, "thin": n_dr < MIN_DRIFTERS})
    return rows


def build_coast_operators(moves, coast_of_traj, args, tag):
    """Split, check and save every coast operator: (coast operators, their per-drifter
    counts, grounded drifters per coast, layout)."""
    base = {name: load_operator(operator_path(name, args.res, args.tau, args.box)) for name in MATRICES + WINDAGE_MATRICES}
    cells = base["drogued"]["cell_flat"]
    n_cells = len(cells)
    layout = Layout(n_cells, n_cells + len(EXIT_LABELS), n_cells + len(COAST_EXIT_LABELS), n_cells + STRANDED)
    oil_moves = np.load(f"data/oil_moves_{tag}.npz")
    counts, ops, support = {}, {}, []
    for name in MATRICES + WINDAGE_MATRICES:
        traj, ft, n_moves = base_moves(name, moves, cells, layout, oil_moves)
        counts[name] = split_moves(traj, ft, n_moves, coast_of_traj, layout)
        op = coast_operator(name, base[name], counts[name], layout, moves.grid.n_lon, args)
        out = operator_path(f"{name}_coast", args.res, args.tau, args.box)
        np.savez_compressed(out, **op)
        ops[name] = load_operator(out)
        support += coast_support(name, traj, ft, coast_of_traj, layout)
        print(f"{name:<10} split into {len(COAST_LABELS)} coasts; matches the saved operator → {out}")
    return ops, counts, pd.DataFrame(support), layout


def save_counts(path, counts, layout):
    """Per-drifter counts on the coast layout, for the traffic-weighted bootstrap in traffic_exposure."""
    np.savez_compressed(path, n_states=layout.n_states, **{
        f"{name}_{part}": value for name, c in counts.items()
        for part, value in zip(("d_idx", "ft", "n", "n_drifters"), c)})


# ── Stranding by coast: eventual, by horizon, time to strand ────────────────

def strand_summary(P, layout, starts, tau, rows=None):
    """Eventual stranding per coast and expected days to strand (among
    material that strands) for ocean states `rows` (all if None), plus
    stranding per coast within each horizon for the release points."""
    fate, visits = stranding_chain(P, layout.n_cells, STRANDED)
    b = fate[:, STRANDED:].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        days = np.where(b > 0, visits / b * tau, np.nan)
    steps = {d: int(round(d / tau)) for d in HORIZONS}
    p = np.zeros((len(starts), layout.n_states))
    p[np.arange(len(starts)), starts] = 1.0
    by_horizon = {}
    for step in range(1, max(steps.values()) + 1):
        p = p @ P
        for d, k in steps.items():
            if step == k:
                by_horizon[d] = p[:, layout.strand_col:layout.n_states].copy()
    rows = slice(None) if rows is None else rows
    return fate[rows, STRANDED:], days[rows], by_horizon


def resample_points(ops, counts, summary_of, starts, n_lon):
    """Stranding on the full data, and at the release points on N_BOOT resamples of whole drifters."""
    rng = np.random.default_rng(SEED)
    full, boot = {}, {}
    for name in MATRICES:
        full[name] = summary_of(ops[name]["P"])
        draws = {"eventual": [], "days": [], **{d: [] for d in HORIZONS}}
        n_states = ops[name]["P"].shape[0]
        for _ in range(N_BOOT):
            Pb = row_normalise(resampled_counts(counts[name], n_states, rng), ops[name]["cell_flat"], n_lon)[0]
            ev, days, hz = summary_of(Pb, rows=starts)
            draws["eventual"].append(ev)
            draws["days"].append(days)
            for d in HORIZONS:
                draws[d].append(hz[d])
        boot[name] = {k: np.array(v) for k, v in draws.items()}
        print(f"{name:<10} {counts[name][3]:,} drifters resampled {len(boot[name]['days'])} times")
    return full, boot


# ── Tables ───────────────────────────────────────────────────────────────────

def point_table(full, boot, points, starts):
    rows = []
    for name in MATRICES:
        ev_full, days_full, hz_full = full[name]
        b = boot[name]
        estimates = [("eventual", ev_full[starts], b["eventual"])]
        estimates += [(d, hz_full[d], b[d]) for d in HORIZONS]
        for horizon, est, draws in estimates:
            any_est, any_draws = est.sum(axis=1), draws.sum(axis=2)
            for k, label in enumerate(np.r_[COAST_LABELS, ["any"]]):
                e = any_est if label == "any" else est[:, k]
                lo, hi = interval(any_draws if label == "any" else draws[:, :, k])
                for i, r in enumerate(points.itertuples()):
                    rows.append({"name": r.name, "lon": r.lon, "lat": r.lat, "matrix": name,
                                 "horizon": horizon, "coast": label, "estimate": e[i], "lo": lo[i], "hi": hi[i]})
        lo, hi = interval(b["days"])
        for i, r in enumerate(points.itertuples()):
            rows.append({"name": r.name, "lon": r.lon, "lat": r.lat, "matrix": name,
                         "horizon": "days_to_strand", "coast": "any",
                         "estimate": days_full[starts][i], "lo": lo[i], "hi": hi[i]})
    return pd.DataFrame(rows)


def cell_table(full, ops):
    cell_lon, cell_lat = cell_centres(ops["drogued"])
    tables = []
    for name in MATRICES:
        ev, days, _ = full[name]
        table = pd.DataFrame(ev, columns=COAST_LABELS)
        table.insert(0, "matrix", name)
        table.insert(1, "lon", cell_lon)
        table.insert(2, "lat", cell_lat)
        table.insert(3, "reliable", reliable(ops[name]))
        table["any"] = ev.sum(axis=1)
        table["days_to_strand"] = days
        tables.append(table)
    return pd.concat(tables)


# ── Map: where material from each cell strands ──────────────────────────────

def plot_stranding(full, ops, args, path):
    cmap = ListedColormap(COAST_COLORS)
    norm = BoundaryNorm(np.arange(len(COAST_LABELS) + 1) - 0.5, len(COAST_LABELS))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2), subplot_kw=MAP_KW)
    for ax, name in zip(axes.ravel(), MATRICES):
        ev = full[name][0]
        total = ev.sum(axis=1)
        rel = reliable(ops[name])
        for lo, hi, alpha in [(MIN_SHOWN, SOLID, 0.35), (SOLID, np.inf, 1.0)]:
            top = np.where(rel & (total >= lo) & (total < hi), ev.argmax(axis=1), np.nan).astype(float)
            ax.pcolormesh(ops[name]["lon_edges"], ops[name]["lat_edges"], to_grid(ops[name], top),
                          cmap=cmap, norm=norm, alpha=alpha, linewidth=0, **TRANSFORM_KW)
        map_axes(ax, args.box)
        ax.set_title(f"{name}: ≥ {MIN_SHOWN:.0%} strands from {(rel & (total >= MIN_SHOWN)).sum()} "
                     f"of {rel.sum()} reliable cells, ≥ {SOLID:.0%} from {(rel & (total >= SOLID)).sum()}",
                     fontsize=10)
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(COAST_COLORS, COAST_NAMES)],
               title=f"Coast that receives most stranding (solid: ≥ {SOLID:.0%} eventually strands; "
                     f"pale: {MIN_SHOWN:.0%}–{SOLID:.0%}; blank: less, or unreliable cell)",
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"Where material from each cell eventually strands — {args.res:g}°, τ = {args.tau:g} d", fontsize=12)
    fig.subplots_adjust(bottom=0.12, top=0.93, hspace=0.15, wspace=0.05)
    save_figure(path)


# ── Summary ──────────────────────────────────────────────────────────────────

def print_summary(support, full, ops, points_table, points):
    pd.set_option("display.width", 220)
    print(f"\nGrounded drifters behind each coast ({MIN_DRIFTERS}+ = not thin):")
    print(support.pivot(index="coast", columns="matrix", values="grounded_drifters")
          .reindex(index=COAST_LABELS, columns=MATRICES).to_string())

    print("\nEventual stranding by coast, mean over reliable cells:")
    mean_rows = {name: [f"{x:.1%}" for x in full[name][0][reliable(ops[name])].mean(axis=0)]
                 for name in MATRICES}
    print(pd.DataFrame(mean_rows, index=COAST_LABELS).to_string())

    for name in ["drogued", "undrogued", "oil"]:
        sub = points_table[(points_table.matrix == name) & (points_table.horizon == "eventual")]
        show = sub.pivot(index="name", columns="coast", values="estimate")
        lo = sub.pivot(index="name", columns="coast", values="lo")
        hi = sub.pivot(index="name", columns="coast", values="hi")
        cols = [c for c in COAST_LABELS if show[c].max() >= 0.02] + ["any"]
        table = pd.DataFrame({c: [fmt(show.at[n, c], lo.at[n, c], hi.at[n, c]) for n in points.name]
                              for c in cols}, index=points.name)
        days = points_table[(points_table.matrix == name) & (points_table.horizon == "days_to_strand")]
        table["days to strand"] = [f"{e:.0f} ({a:.0f}–{b:.0f})" for e, a, b in
                                   days.set_index("name").loc[points.name, ["estimate", "lo", "hi"]].values]
        print(f"\n{name}: eventual stranding by coast (coasts with ≥ 2% from some point), 95% interval")
        print(table.to_string())


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    check_coast_box(args.box)
    tag = setting_tag(args.res, args.tau, args.box)
    out_counts = f"data/coast_counts_{tag}.npz"
    out_points = f"data/coastal_stranding_points_{tag}.csv"
    out_cells = f"data/coastal_stranding_cells_{tag}.csv"
    out_support = f"data/coastal_stranding_support_{tag}.csv"
    out_png = f"figures/agulhas_coastal_stranding_{tag}.png"

    drifters = Drifters(args.data)
    moves = MoveSet(drifters, args.box, args.res, args.tau)
    coast_of_traj, _ = grounding_coasts(drifters, moves.in_R)
    ops, counts, support, layout = build_coast_operators(moves, coast_of_traj, args, tag)
    save_counts(out_counts, counts, layout)

    points = pd.read_csv(release_points_path(args.box))[["name", "lon", "lat"]]
    starts = np.array([state_of(ops["drogued"], r.lon, r.lat) for r in points.itertuples()])
    summary_of = partial(strand_summary, layout=layout, starts=starts, tau=args.tau)
    full, boot = resample_points(ops, counts, summary_of, starts, moves.grid.n_lon)
    points_table = point_table(full, boot, points, starts)
    points_table.to_csv(out_points, index=False)
    cell_table(full, ops).to_csv(out_cells, index=False)
    support.to_csv(out_support, index=False)

    plot_stranding(full, ops, args, out_png)
    print_summary(support, full, ops, points_table, points)
    print(f"\nSaved → {out_points}, {out_cells}, {out_support}, {out_counts}, {out_png}")
    print("Coast operators → " + ", ".join(operator_path(f"{n}_coast", args.res, args.tau, args.box) for n in MATRICES))


if __name__ == "__main__":
    main()
