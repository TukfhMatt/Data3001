"""
Agulhas Region — Does OpenDrift Fall Inside the Matrix's Intervals
Added after opendrift_compare's result; its rule is in docs/opendrift_plan.md and was fixed
before this ran. For each release point, horizon (day 7 and 31.5) and outcome
(still in R, stranded, exit W, E, S, N), counts how often OpenDrift's mean share
lies inside the matrix's 95% bootstrap interval from risk_atlas. Outcomes where
both shares are below MIN_SHARE are left out, since an exit nobody uses would
count as agreement.

The interval holds the matrix's sampling noise only. Inside means the gap is
within it; outside means the data or the physics differ by more. Reported for the
six points west of 40°E (the five ports and the 27.5°E lane) and the seven lanes from 40°E apart.

    .venv-opendrift/bin/python -m agulhas opendrift_coverage
"""

import numpy as np
import pandas as pd

from agulhas.comparison.opendrift_compare import ATLAS_HORIZON, load_runs, reduce_states
from agulhas.comparison.opendrift_points import points_folder
from agulhas.config import OIL_WINDAGE_RANGE, operator_path, parse_args, release_points_path, setting_tag
from agulhas.opendrift_common import HOURS, OUT
from agulhas.transport import check_coast_box, load_operator, state_of

MIN_SHARE = 0.01
OUTCOMES = ["in_R", "W", "E", "S", "N", "stranded_any"]


def outcome_shares(dist, n_cells):
    """In R, the four exits and stranded, from a distribution over the matrix's states."""
    r = reduce_states(dist, n_cells)
    return np.r_[r[:n_cells].sum(), r[n_cells:n_cells + 4], r[n_cells + 4]]


def coverage_rows(points, op, atlas, shares):
    """One row per point, horizon and outcome: OpenDrift's mean share, the matrix's estimate and interval, and whether it is inside."""
    n_cells = len(op["cell_flat"])
    rows = []
    for k, p in points.iterrows():
        state = state_of(op, p.lon, p.lat)
        for h, days in enumerate(HOURS):
            od = outcome_shares(shares[:, k, h].mean(axis=0), n_cells)
            for j, outcome in enumerate(OUTCOMES):
                a = atlas[(atlas["state"] == state) & (atlas["horizon"] == ATLAS_HORIZON[days]) & (atlas["outcome"] == outcome)].iloc[0]
                rows.append({"name": p["name"], "lon": p.lon, "days": days, "outcome": outcome, "opendrift": od[j],
                             "matrix": a["estimate"], "lo": a["lo"], "hi": a["hi"],
                             "judged": bool(max(od[j], a["estimate"]) >= MIN_SHARE), "inside": bool(a["lo"] <= od[j] <= a["hi"])})
    return pd.DataFrame(rows)


def print_summary(table):
    judged = table[table["judged"]]
    print(f"{len(judged)} point–horizon–outcome cases judged of {len(table)} (the rest are below {MIN_SHARE:.0%} on both sides)")
    groups = {"all": judged, "South African points": judged[judged["lon"] < 40], "lanes at 40°E and east": judged[judged["lon"] >= 40]}
    for name, g in groups.items():
        for days, gd in g.groupby("days"):
            print(f"  {name:<24} day {days:>4}: OpenDrift inside the matrix's interval in {gd['inside'].sum()} of {len(gd)} ({gd['inside'].mean():.0%})")
    by_outcome = judged.groupby("outcome")["inside"].agg(["sum", "count"])
    print("By outcome: " + ", ".join(f"{o} {int(r['sum'])}/{int(r['count'])}" for o, r in by_outcome.iterrows()))


def main(argv=None):
    """Run this step; see the module docstring for what it does."""
    def add_options(p):
        p.add_argument("--matrix", default="oil", choices=list(OIL_WINDAGE_RANGE), help="oil matrix, and the runs at its share of the wind")

    args = parse_args(__doc__, grid=True, lag=True, extra=add_options, argv=argv)
    check_coast_box(args.box)
    tag = setting_tag(args.res, args.tau, args.box)
    name = "" if args.matrix == "oil" else f"_{args.matrix}"
    out = f"{OUT}/coverage{name}_{tag}.csv"
    points = pd.read_csv(release_points_path(args.box))
    _, shares = load_runs(points_folder(args.matrix))
    op = load_operator(operator_path(f"{args.matrix}_coast", args.res, args.tau, args.box))
    atlas = pd.read_csv(f"data/risk_atlas_{tag}.csv", low_memory=False)
    table = coverage_rows(points, op, atlas[atlas["matrix"] == args.matrix], shares)
    table.to_csv(out, index=False)
    print_summary(table)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
