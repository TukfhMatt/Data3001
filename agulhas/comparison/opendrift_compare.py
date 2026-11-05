"""
Agulhas Region — OpenDrift Against the Matrix
Sets the oil matrix against the OpenDrift runs at the 13 release points, averaged
over the 60 start dates of opendrift_points. OpenDrift knows each date's real currents
and winds and the matrix only their 1995–2022 average, so this measures what the
average costs and whether the matrix still gets the broad risks right, not which is
better. Apart from Treasure there is no real oil here to judge either against.

Rule, fixed before any full OpenDrift result was seen. A point agrees at day 31.5
if all three hold:

  1. Total variation distance over the ocean cells, the four exits and stranded is
     at most config.DISAGREE_TVD (0.3).
  2. The 95% stranding intervals overlap (OpenDrift: resampling whole start dates;
     matrix: the atlas).
  3. The largest of in R, each exit and stranded is the same.

As a reference, each year of OpenDrift is compared with its other four years: the
distance a single set of years differs from another with the model unchanged. The five
years do not match the matrix's 1995–2022 period, which is a limit of this check.

    .venv-opendrift/bin/python -m agulhas opendrift_compare
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt

from agulhas.opendrift_common import HOURS, N_COASTS, OUT, YEARS
from agulhas.config import DISAGREE_TVD, operator_path, parse_args, release_points_path, setting_tag
from agulhas.transport import COAST_LABELS, check_coast_box, load_operator, state_of

N_BOOT = 2000
SEED = 0
ATLAS_HORIZON = {7: "7", 31.5: "30"}


def reduce_states(dist, n_cells):
    """Ocean cells, four exits, stranded (the six coasts together) and no matching state, on the last axis."""
    return np.concatenate([dist[..., :n_cells + 4], dist[..., n_cells + 4:n_cells + 4 + N_COASTS].sum(axis=-1, keepdims=True),
                           dist[..., n_cells + 4 + N_COASTS:]], axis=-1)


def tvd(a, b):
    return 0.5 * np.abs(a - b).sum(axis=-1)


def largest(dist, n_cells):
    """Which of in R, the exits and stranded holds the most."""
    groups = np.concatenate([[dist[:n_cells].sum()], dist[n_cells:n_cells + 5]])
    return ["in_R", "W", "E", "S", "N", "stranded"][int(np.argmax(groups))]


def main(argv=None):
    """Run this step; see the module docstring for what it does."""
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    check_coast_box(args.box)
    TAG = setting_tag(args.res, args.tau, args.box)
    points = pd.read_csv(release_points_path(args.box))
    rng = np.random.default_rng(SEED)

    # ── 1. The runs: shares for each start, point and horizon ────────────────
    paths = sorted(glob.glob(f"{OUT}/points/*.npz"))
    if not paths:
        raise SystemExit(f"No runs in {OUT}/points: run opendrift_points first.")
    starts = [os.path.basename(p)[:6] for p in paths]
    years = np.array([int(s[:4]) for s in starts])
    counts = np.stack([np.load(p)["counts"] for p in paths])
    shares = counts / counts.sum(axis=-1, keepdims=True)
    print(f"{len(paths)} starts, {counts[0, 0, 0].sum()} particles per point; particles with no matching state: {counts[:, :, :, -1].sum()}")

    # ── 2. The matrix at the same steps, from the same cells ─────────────────
    op = load_operator(operator_path("oil_coast", args.res, args.tau, args.box))
    n_cells = len(op["cell_flat"])
    atlas = pd.read_csv(f"data/risk_atlas_{TAG}.csv", low_memory=False)
    atlas = atlas[atlas["matrix"] == "oil"]
    rows = []
    for k, p in points.iterrows():
        state = state_of(op, p.lon, p.lat)
        for h, (days, hour) in enumerate(HOURS.items()):
            steps = int(round(days / args.tau))
            matrix = np.append(np.linalg.matrix_power(op["P"], steps)[state], 0.0)
            model = shares[:, k, h].mean(axis=0)
            m_red, o_red = reduce_states(matrix, n_cells), reduce_states(model, n_cells)

            # ── 3. Intervals from whole start dates, and year-to-year distance ───
            draw = rng.integers(0, len(starts), (N_BOOT, len(starts)))
            strand = shares[:, k, h, n_cells + 4:n_cells + 4 + N_COASTS].sum(axis=-1)
            lo, hi = np.percentile(strand[draw].mean(axis=1), [2.5, 97.5])
            between = [tvd(reduce_states(shares[years == y, k, h].mean(axis=0), n_cells),
                           reduce_states(shares[years != y, k, h].mean(axis=0), n_cells)) for y in YEARS if (years == y).any() and (years != y).any()]
            row = atlas[(atlas["state"] == state) & (atlas["horizon"] == ATLAS_HORIZON[days]) & (atlas["outcome"] == "stranded_any")].iloc[0]
            coast_m, coast_o = matrix[n_cells + 4:n_cells + 4 + N_COASTS], model[n_cells + 4:n_cells + 4 + N_COASTS]
            d = tvd(m_red, o_red)
            overlap = lo <= row["hi"] and row["lo"] <= hi
            same = largest(m_red, n_cells) == largest(o_red, n_cells)
            rows.append({"name": p["name"], "lon": p.lon, "lat": p.lat, "days": days, "tvd": d, "year_to_year_tvd": float(np.mean(between)) if between else np.nan,
                         "in_R_matrix": matrix[:n_cells].sum(), "in_R_opendrift": model[:n_cells].sum(),
                         "exits_matrix": matrix[n_cells:n_cells + 4].sum(), "exits_opendrift": model[n_cells:n_cells + 4].sum(),
                         "stranded_matrix": coast_m.sum(), "stranded_matrix_lo": row["lo"], "stranded_matrix_hi": row["hi"],
                         "stranded_opendrift": strand.mean(), "stranded_opendrift_lo": lo, "stranded_opendrift_hi": hi,
                         "top_coast_matrix": COAST_LABELS[int(np.argmax(coast_m))] if coast_m.sum() > 0 else "", "top_coast_opendrift": COAST_LABELS[int(np.argmax(coast_o))] if coast_o.sum() > 0 else "",
                         "largest_matrix": largest(m_red, n_cells), "largest_opendrift": largest(o_red, n_cells),
                         "rule_tvd": bool(d <= DISAGREE_TVD), "rule_stranding_overlap": bool(overlap), "rule_same_largest": bool(same), "agrees": bool(d <= DISAGREE_TVD and overlap and same)})
    table = pd.DataFrame(rows)
    out = f"{OUT}/compare_{TAG}.csv"
    table.to_csv(out, index=False)

    # ── 4. Summary ───────────────────────────────────────────────────────────
    final = table[table["days"] == 31.5]
    cols = ["name", "tvd", "year_to_year_tvd", "stranded_matrix", "stranded_opendrift", "stranded_opendrift_lo", "stranded_opendrift_hi", "largest_matrix", "largest_opendrift", "agrees"]
    print(final[cols].round(3).to_string(index=False))
    print(f"\nAt day 31.5: {final['agrees'].sum()} of {len(final)} points agree; TVD ≤ {DISAGREE_TVD} at {final['rule_tvd'].sum()}, stranding intervals overlap at {final['rule_stranding_overlap'].sum()}, "
          f"same largest outcome at {final['rule_same_largest'].sum()}. Median TVD {final['tvd'].median():.2f}, against {final['year_to_year_tvd'].median():.2f} between OpenDrift's own years.")
    if final["stranded_matrix"].std() > 0 and final["stranded_opendrift"].std() > 0:
        print(f"Rank correlation of stranding across points: {final['stranded_matrix'].corr(final['stranded_opendrift'], method='spearman'):.2f}")

    # ── 5. Figure ────────────────────────────────────────────────────────────
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    y = np.arange(len(final))
    a.errorbar(final["stranded_matrix"] * 100, y + 0.15, xerr=[(final["stranded_matrix"] - final["stranded_matrix_lo"]) * 100, (final["stranded_matrix_hi"] - final["stranded_matrix"]) * 100],
               fmt="o", color="#2a78d6", label="Oil matrix (atlas 95%)", capsize=2)
    a.errorbar(final["stranded_opendrift"] * 100, y - 0.15, xerr=[(final["stranded_opendrift"] - final["stranded_opendrift_lo"]) * 100, (final["stranded_opendrift_hi"] - final["stranded_opendrift"]) * 100],
               fmt="s", color="#eb6834", label="OpenDrift (95% over start dates)", capsize=2)
    a.set_yticks(y, final["name"])
    a.invert_yaxis()
    a.set_xlabel("Stranded within 31.5 days (%)")
    a.grid(axis="x", alpha=0.3)
    a.legend(frameon=False)
    b.barh(y + 0.18, final["tvd"], height=0.35, color="#2a78d6", label="Matrix against OpenDrift")
    b.barh(y - 0.18, final["year_to_year_tvd"], height=0.35, color="#9aa7b1", label="OpenDrift: one year against the others")
    b.axvline(DISAGREE_TVD, color="#10263B", lw=1, ls="--", label=f"Limit {DISAGREE_TVD}")
    b.set_xlabel("Total variation distance at day 31.5")
    b.grid(axis="x", alpha=0.3)
    b.legend(frameon=False)
    fig.suptitle(f"Oil matrix against OpenDrift at the release points ({len(paths)} start dates)")
    fig.tight_layout()
    png = f"figures/agulhas_opendrift_compare_{TAG}.png"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}, {png}")


if __name__ == "__main__":
    main()
