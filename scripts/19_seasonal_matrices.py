"""
Agulhas Region — Seasonal Transition Matrices

Builds separate transport operators for the four Southern Hemisphere seasons:

    DJF = Dec-Feb (summer)
    MAM = Mar-May (autumn)
    JJA = Jun-Aug (winter)
    SON = Sep-Nov (spring)

A move is assigned to the season of its START observation. Matrix building
matches 03_transition_matrix.py: the same ocean states, five absorbing states
(W/E/S/N/stranded), neighbour pooling and MIN_DRIFTERS reliability rule.

For each drogue type the script reports:
  - seasonal data support
  - row-wise TVD against the annual operator
  - pairwise TVD between seasons
  - pairwise TVD on cells reliable in all four seasons
  - 30-day release-point outcomes as a within-season sensitivity test

The 30-day calculation repeatedly applies one seasonal operator and is not a
calendar-year seasonal forecast.

    python scripts/19_seasonal_matrices.py --res 1 --tau 3.5
    python scripts/19_seasonal_matrices.py --res 2 --tau 3.5
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from config import (
    DROGUE_TYPES,
    MIN_DRIFTERS,
    grid_shape,
    operator_path,
    parse_args,
    setting_tag,
)
from transport import (
    EXIT_LABELS,
    STRANDED,
    cell_centres,
    load_operator,
    move_indices,
    row_normalise,
    state_of,
    strand_times,
    transition_states,
)


args = parse_args(__doc__, grid=True, lag=True)
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
RES, TAU = args.res, args.tau
TAG = setting_tag(RES, TAU, args.box)

SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}
SEASON_NAMES = {
    "DJF": "Summer",
    "MAM": "Autumn",
    "JJA": "Winter",
    "SON": "Spring",
}
SEASON_LIST = list(SEASONS)

os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)


# ── 1. Load data ──────────────────────────────────────────────────────────────
ds = xr.open_dataset(args.data)

for v in ("typedeath", "end_date"):
    if v not in ds:
        raise SystemExit(
            f"{v} is missing from {args.data}. Refresh the subset metadata first:\n"
            "  python scripts/00_fetch_agulhas_subset.py --metadata-only"
        )

lon = ds["lon"].values.astype(float)
lat = ds["lat"].values.astype(float)
drogue = ds["drogue_status"].values.astype(bool)
time = ds["time"].values
t = time.astype("datetime64[s]").astype(np.int64)
months = time.astype("datetime64[M]").astype(int) % 12 + 1
years = time.astype("datetime64[Y]").astype(int) + 1970

traj_idx = np.repeat(np.arange(ds.sizes["traj"]), ds["rowsize"].values)

in_R = (
    (lon >= LON_MIN) & (lon < LON_MAX)
    & (lat >= LAT_MIN) & (lat < LAT_MAX)
)

strand_t = strand_times(ds, traj_idx, t, in_R)


# ── 2. Grid and transition pairs ─────────────────────────────────────────────
n_lon, n_lat = grid_shape(args.box, RES)

lon_edges = np.minimum(
    LON_MIN + np.arange(n_lon + 1) * RES,
    LON_MAX,
)
lat_edges = np.minimum(
    LAT_MIN + np.arange(n_lat + 1) * RES,
    LAT_MAX,
)

cell_flat = np.where(
    in_R,
    ((lat - LAT_MIN) // RES).astype(int) * n_lon
    + ((lon - LON_MIN) // RES).astype(int),
    -1,
)

# Same state space for all seasons and both drogue types.
cells = np.unique(cell_flat[cell_flat >= 0])
n_cells = len(cells)
n_states = n_cells + len(EXIT_LABELS)

start, end, stranded = move_indices(
    traj_idx, t, in_R, TAU, strand_t
)
start_month = months[start]

print("========================================")
print("SEASONAL TRANSITION MATRICES")
print("========================================")
print(f"Region: {args.box}")
print(f"Grid: {RES:g}°")
print(f"Lag: {TAU:g} days")
print(f"Ocean cells / states: {n_cells} / {n_states}")
print(f"Candidate transitions: {len(start):,}")


# ── 3. Annual operators ───────────────────────────────────────────────────────
baseline_ops = {}

for name in DROGUE_TYPES:
    path = operator_path(name, RES, TAU, args.box)

    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found. Run:\n"
            f"  python scripts/03_transition_matrix.py "
            f"--res {RES:g} --tau {TAU:g}"
        )

    op = load_operator(path)

    # Protect against an old four-exit matrix made before stranded was added.
    if op["P"].shape != (n_states, n_states):
        raise SystemExit(
            f"{path} has shape {op['P'].shape}, but the current model expects "
            f"{(n_states, n_states)}.\n"
            "The annual matrix is probably from an older pipeline version.\n"
            "Rebuild it with:\n"
            f"  python scripts/03_transition_matrix.py "
            f"--res {RES:g} --tau {TAU:g}"
        )

    if not np.array_equal(op["cell_flat"], cells):
        raise SystemExit(
            f"{path} uses a different ocean-cell state space. "
            "Rebuild script 03 with the current data and settings."
        )

    if not np.array_equal(op["exit_labels"].astype(str), EXIT_LABELS.astype(str)):
        raise SystemExit(
            f"{path} has outdated absorbing states. "
            "Rebuild script 03 first."
        )

    baseline_ops[name] = op


# ── 4. Build seasonal matrices ───────────────────────────────────────────────
def build_seasonal(name, drogue_flag, season):
    season_mask = np.isin(start_month, SEASONS[season])

    # Same drogue-type rule as script 03.
    same_type = (
        (drogue[start] == drogue_flag)
        & (drogue[end] == drogue_flag)
    )

    use = season_mask & same_type
    s, e, st = start[use], end[use], stranded[use]

    if len(s) == 0:
        raise SystemExit(f"No transitions for {name} {season}")

    from_state, to_state = transition_states(
        cell_flat,
        cells,
        s,
        e,
        lon,
        lat,
        args.box,
        st,
    )

    C = np.zeros((n_states, n_states))
    np.add.at(C, (from_state, to_state), 1)

    pairs = np.unique(
        np.stack([from_state, traj_idx[s]], axis=1),
        axis=0,
    )
    row_drifters = np.bincount(pairs[:, 0], minlength=n_states)
    row_obs = C.sum(axis=1)

    # Same empty-row pooling as script 03.
    P, empty = row_normalise(C, cells, n_lon)

    empty_mask = np.isin(np.arange(n_cells), empty)
    flagged = row_drifters[:n_cells] < MIN_DRIFTERS
    reliable = ~flagged & ~empty_mask

    out = operator_path(
        f"{name}_{season}",
        RES,
        TAU,
        args.box,
    )

    np.savez_compressed(
        out,
        P=P,
        C=C,
        row_drifters=row_drifters[:n_cells],
        row_obs=row_obs[:n_cells],
        flagged=flagged,
        empty=empty_mask,
        cell_flat=cells,
        lon_edges=lon_edges,
        lat_edges=lat_edges,
        res=RES,
        tau_days=TAU,
        exit_labels=EXIT_LABELS,
        box=np.array(args.box),
        drogued=drogue_flag,
        season=season,
        season_name=SEASON_NAMES[season],
        season_months=np.array(SEASONS[season]),
        season_rule="season of transition start observation",
        years=np.array([years[s].min(), years[s].max()]),
        source="NOAA GDP hourly v2.01 via CloudDrift, hourly, gap <= 6 h",
    )

    active = row_obs[:n_cells] > 0
    n_drifters = len(np.unique(traj_idx[s]))

    exit_prob = P[
        :n_cells,
        n_cells:n_cells + STRANDED
    ].sum(axis=1)

    strand_prob = P[
        :n_cells,
        n_cells + STRANDED
    ]

    print(f"\n=== {name}, {season} ({SEASON_NAMES[season]}) ===")
    print(
        f"Transitions used      : {len(s):,} "
        f"({years[s].min()}–{years[s].max()})"
    )
    print(f"Distinct drifters     : {n_drifters:,}")
    print(f"Active rows           : {active.sum()} / {n_cells}")
    print(
        f"Reliable rows         : {reliable.sum()} / {n_cells} "
        f"({reliable.mean():.1%})"
    )
    print(f"Cells with no moves   : {empty_mask.sum()}")
    print(f"Mean exit prob / step : {exit_prob.mean():.1%}")
    print(f"Mean strand prob/step : {strand_prob.mean():.2%}")
    print(f"Rows sum to 1         : {np.allclose(P.sum(axis=1), 1)}")
    print(f"Saved → {out}")

    summary = {
        "season": season,
        "season_name": SEASON_NAMES[season],
        "drogue": name,
        "transitions": len(s),
        "drifters": n_drifters,
        "active_rows": int(active.sum()),
        "reliable_rows": int(reliable.sum()),
        "reliable_share_all": float(reliable.mean()),
        "reliable_share_active": (
            float(reliable[active].mean()) if active.any() else np.nan
        ),
        "empty_rows": int(empty_mask.sum()),
        "mean_exit_per_step": float(exit_prob.mean()),
        "mean_strand_per_step": float(strand_prob.mean()),
    }

    return load_operator(out), summary


seasonal_ops = {}
support_rows = []

for season in SEASON_LIST:
    for name, flag in DROGUE_TYPES.items():
        op, summary = build_seasonal(name, flag, season)
        seasonal_ops[(season, name)] = op
        support_rows.append(summary)

support = pd.DataFrame(support_rows)
support_file = f"data/seasonal_support_{TAG}.csv"
support.to_csv(support_file, index=False)


# ── 5. Seasonal vs annual TVD ────────────────────────────────────────────────
cell_rows = []

print("\n========================================")
print("SEASONAL VS ANNUAL MATRIX")
print("========================================")

for name in DROGUE_TYPES:
    annual = baseline_ops[name]
    annual_reliable = (
        ~annual["flagged"].astype(bool)
        & ~annual["empty"].astype(bool)
    )
    cell_lon, cell_lat = cell_centres(annual)

    for season in SEASON_LIST:
        seasonal = seasonal_ops[(season, name)]
        seasonal_reliable = (
            ~seasonal["flagged"].astype(bool)
            & ~seasonal["empty"].astype(bool)
        )

        common = annual_reliable & seasonal_reliable

        tvd = 0.5 * np.abs(
            seasonal["P"][:n_cells]
            - annual["P"][:n_cells]
        ).sum(axis=1)

        values = tvd[common]

        print(
            f"{name:10s} {season}: "
            f"{common.sum():4d} common reliable rows | "
            f"median TVD {np.median(values):.3f} | "
            f"90th pct {np.percentile(values, 90):.3f}"
        )

        for i in range(n_cells):
            cell_rows.append({
                "drogue": name,
                "season": season,
                "season_name": SEASON_NAMES[season],
                "state": i,
                "cell_flat": int(cells[i]),
                "lon": float(cell_lon[i]),
                "lat": float(cell_lat[i]),
                "annual_reliable": bool(annual_reliable[i]),
                "season_reliable": bool(seasonal_reliable[i]),
                "common_reliable": bool(common[i]),
                "tvd_vs_annual": float(tvd[i]),
            })

cell_table = pd.DataFrame(cell_rows)
cell_file = f"data/seasonal_tvd_cells_{TAG}.csv"
cell_table.to_csv(cell_file, index=False)


# ── 6. Pairwise seasonal TVD ─────────────────────────────────────────────────
pair_rows = []

print("\n========================================")
print("PAIRWISE SEASON TVD")
print("========================================")

for name in DROGUE_TYPES:
    seasonal_reliable = {
        season: (
            ~seasonal_ops[(season, name)]["flagged"].astype(bool)
            & ~seasonal_ops[(season, name)]["empty"].astype(bool)
        )
        for season in SEASON_LIST
    }

    # Same cells for every seasonal comparison.
    all_four = np.logical_and.reduce(
        [seasonal_reliable[s] for s in SEASON_LIST]
    )

    print(
        f"\n{name}: cells reliable in all four seasons = "
        f"{all_four.sum()} / {n_cells}"
    )

    for i, s1 in enumerate(SEASON_LIST):
        for s2 in SEASON_LIST[i + 1:]:
            op1 = seasonal_ops[(s1, name)]
            op2 = seasonal_ops[(s2, name)]

            pair_common = (
                seasonal_reliable[s1]
                & seasonal_reliable[s2]
            )

            tvd = 0.5 * np.abs(
                op1["P"][:n_cells]
                - op2["P"][:n_cells]
            ).sum(axis=1)

            pair_values = tvd[pair_common]
            four_values = tvd[all_four]

            row = {
                "drogue": name,
                "season_1": s1,
                "season_2": s2,
                "pair_common_rows": int(pair_common.sum()),
                "median_tvd_pair_common": float(np.median(pair_values)),
                "p90_tvd_pair_common": float(np.percentile(pair_values, 90)),
                "all_four_common_rows": int(all_four.sum()),
                "median_tvd_all_four": (
                    float(np.median(four_values))
                    if len(four_values) else np.nan
                ),
                "p90_tvd_all_four": (
                    float(np.percentile(four_values, 90))
                    if len(four_values) else np.nan
                ),
            }
            pair_rows.append(row)

            print(
                f"{s1} vs {s2}: "
                f"pair n={pair_common.sum():4d}, "
                f"median={np.median(pair_values):.3f} | "
                f"all-four n={all_four.sum():4d}, "
                f"median="
                f"{np.median(four_values):.3f}"
                if len(four_values)
                else "no all-four common reliable rows"
            )

pair_table = pd.DataFrame(pair_rows)
pair_file = f"data/seasonal_pairwise_tvd_{TAG}.csv"
pair_table.to_csv(pair_file, index=False)


# ── 7. 30-day release-point seasonal sensitivity ─────────────────────────────
release_file = f"data/seasonal_release_30d_{TAG}.csv"
release_rows = []
release_path = "data/release_candidates.csv"

if os.path.exists(release_path):
    points = pd.read_csv(release_path)

    steps = int(round(30 / TAU))
    model_days = steps * TAU

    print("\n========================================")
    print("30-DAY RELEASE-POINT SEASONAL SCENARIO")
    print("========================================")
    print(
        f"30 requested days = {steps} steps = "
        f"{model_days:g} model days"
    )

    for pt in points.itertuples():
        for name in DROGUE_TYPES:
            for season in SEASON_LIST:
                op = seasonal_ops[(season, name)]

                try:
                    s = state_of(op, pt.lon, pt.lat)
                except ValueError:
                    continue

                p0 = np.zeros(op["P"].shape[0])
                p0[s] = 1.0
                p = p0 @ np.linalg.matrix_power(op["P"], steps)

                row_reliable = (
                    not bool(op["flagged"][s])
                    and not bool(op["empty"][s])
                )

                row = {
                    "name": pt.name,
                    "lon": pt.lon,
                    "lat": pt.lat,
                    "drogue": name,
                    "season": season,
                    "season_name": SEASON_NAMES[season],
                    "requested_days": 30,
                    "steps": steps,
                    "model_days": model_days,
                    "release_row_drifters": int(op["row_drifters"][s]),
                    "release_reliable": row_reliable,
                    "in_R": float(p[:n_cells].sum()),
                }

                for k, label in enumerate(op["exit_labels"].astype(str)):
                    row[f"exit_{label}"] = float(p[n_cells + k])

                release_rows.append(row)

    pd.DataFrame(release_rows).to_csv(release_file, index=False)


# ── 8. Summary figure ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(SEASON_LIST))
width = 0.35

for offset, name in enumerate(DROGUE_TYPES):
    sub = (
        support[support["drogue"] == name]
        .set_index("season")
        .loc[SEASON_LIST]
    )

    axes[0].bar(
        x + (offset - 0.5) * width,
        sub["reliable_share_all"],
        width=width,
        label=name,
    )

axes[0].set_xticks(x)
axes[0].set_xticklabels(SEASON_LIST)
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Share of ocean rows reliable")
axes[0].set_title("Seasonal data support")
axes[0].legend()

tvd_summary = (
    cell_table[cell_table["common_reliable"]]
    .groupby(["drogue", "season"])["tvd_vs_annual"]
    .median()
)

for offset, name in enumerate(DROGUE_TYPES):
    values = [
        tvd_summary.get((name, season), np.nan)
        for season in SEASON_LIST
    ]

    axes[1].bar(
        x + (offset - 0.5) * width,
        values,
        width=width,
        label=name,
    )

axes[1].set_xticks(x)
axes[1].set_xticklabels(SEASON_LIST)
axes[1].set_ylabel("Median row TVD vs annual matrix")
axes[1].set_title("Seasonal change in transport")
axes[1].legend()

fig.suptitle(
    f"Agulhas seasonal transport — {RES:g}°, τ = {TAU:g} d"
)
fig.tight_layout()

figure_file = f"figures/agulhas_seasonal_summary_{TAG}.png"
plt.savefig(figure_file, dpi=150, bbox_inches="tight")
plt.close()


# ── 9. Done ──────────────────────────────────────────────────────────────────
print("\n========================================")
print("DONE")
print("========================================")
print(f"Support     → {support_file}")
print(f"Cell TVD    → {cell_file}")
print(f"Pairwise    → {pair_file}")

if release_rows:
    print(f"30-day fate → {release_file}")

print(f"Figure      → {figure_file}")
print(
    "\n30-day release results are within-season sensitivity scenarios; "
    "they are not calendar-year forecasts."
)
