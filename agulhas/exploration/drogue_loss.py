"""
Agulhas Region — Drogue Loss Before Grounding
Checks when drifters that ran aground in R lost their drogue, to show how a
stranding move should be assigned to a drogue type. For every drifter whose
record ends in R (last subset observation in R and within a day of GDP's end
date, as in transport.strand_times), the time from its last drogued hour to
the end of its record is compared between drifters that ran aground
(typedeath 1) and drifters that stopped transmitting (typedeath 3).

If drogues were torn off by grounding, grounded drifters would lose them in
their final hours far more often than drifters that stopped transmitting.
The script also counts the stranding moves each drogue type gets when a move
into the stranded state takes the drogue status of its start
(transport.same_drogue), against a rule that needs the status at both ends.

    python -m agulhas drogue_loss --tau 3.5
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt
import xarray as xr

from agulhas.config import DROGUE_TYPES, _tag, box_tag, parse_args, region_prefix
from agulhas.transport import STRAND_MATCH_S, move_indices, same_drogue, strand_times

GROUPS = {1: "ran aground", 3: "stopped transmitting"}


def main(argv=None):
    """Run this step; see the module docstring for what it does."""
    args = parse_args(__doc__, lag=True, argv=argv)
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
    TAU_DAYS = args.tau
    TAG = f"{_tag(TAU_DAYS)}d_{box_tag(args.box)}"

    # ── 1. Load subset ───────────────────────────────────────────────────────
    ds = xr.open_dataset(args.data)
    lon = ds["lon"].values.astype(float)
    lat = ds["lat"].values.astype(float)
    drogue = ds["drogue_status"].values.astype(bool)
    t = ds["time"].values.astype("datetime64[s]").astype(np.int64)
    traj_idx = np.repeat(np.arange(ds.sizes["traj"]), ds["rowsize"].values)
    in_R = (lon >= LON_MIN) & (lon < LON_MAX) & (lat >= LAT_MIN) & (lat < LAT_MAX)

    # ── 2. Drogue status at the end of each record that ends in R ────────────
    row_end = np.cumsum(ds["rowsize"].values) - 1
    row_start = row_end - ds["rowsize"].values + 1
    end_s = ds["end_date"].values.astype("datetime64[s]").astype(np.int64)
    typedeath = ds["typedeath"].values
    ends_in_R = in_R[row_end] & (np.abs(t[row_end] - end_s) <= STRAND_MATCH_S)

    rows = []
    for k in np.where(ends_in_R & np.isin(typedeath, list(GROUPS)))[0]:
        a, b = row_start[k], row_end[k]
        st = drogue[a:b + 1]
        if st[-1]:
            state, gap, km = "drogued at end", 0.0, np.nan
        elif not st.any():
            state, gap, km = "never drogued", np.nan, np.nan
        else:
            i = a + np.where(st)[0][-1]                   # last drogued hour
            state, gap = "lost drogue", (t[b] - t[i]) / 86400
            p1, p2 = np.radians(lat[i]), np.radians(lat[b])
            dl = np.radians(lon[b] - lon[i])
            km = 6371.0 * np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl), -1, 1))
        rows.append({"traj": k, "fate": GROUPS[int(typedeath[k])], "state": state, "gap_days": gap,
                     "km_loss_to_end": km, "end_lon": lon[b], "end_lat": lat[b]})
    df = pd.DataFrame(rows)

    def summary(d, fate):
        lost = d[d.state == "lost drogue"].gap_days
        return {"fate": fate, "drifters": len(d),
                "drogued_at_end": int((d.state == "drogued at end").sum()),
                "never_drogued": int((d.state == "never drogued").sum()),
                "lost_le_1d": int((lost <= 1).sum()),
                "lost_1d_to_tau": int(((lost > 1) & (lost <= TAU_DAYS)).sum()),
                "lost_gt_tau": int((lost > TAU_DAYS).sum()),
                "share_lost_le_1d": (lost <= 1).sum() / len(d),
                "median_gap_days": lost.median()}

    table = pd.DataFrame([summary(df[df.fate == f], f) for f in GROUPS.values()])

    # ── 3. Stranding moves per drogue type under each rule ───────────────────
    strand_t = strand_times(ds, traj_idx, t, in_R)
    start, end, stranded = move_indices(traj_idx, t, in_R, TAU_DAYS, strand_t)
    s, e = start[stranded], end[stranded]
    moves = []
    for name, flag in DROGUE_TYPES.items():
        for rule, keep in [("start", same_drogue(drogue, s, e, True, flag)),
                           ("both ends", (drogue[s] == flag) & (drogue[e] == flag))]:
            moves.append({"drogue": name, "rule": rule, "stranding_moves": int(keep.sum()),
                          "drifters": len(np.unique(traj_idx[s[keep]]))})
    moves = pd.DataFrame(moves)
    n_grounded = int((strand_t[row_end] >= 0).sum())

    out_cells = f"data/drogue_loss_{TAG}.csv"
    out_sum = f"data/drogue_loss_summary_{TAG}.csv"
    out_moves = f"data/drogue_loss_stranding_moves_{TAG}.csv"
    df.to_csv(out_cells, index=False)
    table.to_csv(out_sum, index=False)
    moves.to_csv(out_moves, index=False)

    # ── 4. Figure: time from drogue loss to end of record ────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(-2, np.log10(max(df.gap_days.max(), 10)), 30)
    for fate, colour in zip(GROUPS.values(), ["tab:red", "tab:grey"]):
        g = df[(df.fate == fate) & (df.state == "lost drogue")].gap_days
        ax.hist(np.maximum(g, bins[0]), bins=bins, density=True, histtype="step", lw=2, color=colour,
                label=f"{fate} (n = {len(g)})")
    ax.axvline(1, color="k", ls=":", lw=1)
    ax.axvline(TAU_DAYS, color="k", ls="--", lw=1, label=f"τ = {TAU_DAYS:g} d")
    ax.set_xscale("log")
    ax.set_xlabel("Time from drogue loss to end of record (days)")
    ax.set_ylabel("Density")
    ax.set_title("Drifters whose record ends in R: when was the drogue lost?")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_fig = f"figures/{region_prefix(args.data)}_drogue_loss_{TAG}.png"
    fig.savefig(out_fig, dpi=150)

    # ── 5. Summary ───────────────────────────────────────────────────────────
    pd.set_option("display.width", 200)
    print(f"Drifters whose record ends in R ({n_grounded} ran aground with the subset's end matched):")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    late = df[(df.fate == "ran aground") & (df.state == "lost drogue") & (df.gap_days <= 1)]
    print(f"\nGrounded drifters that lost the drogue ≤ 1 d before grounding: {len(late)}, "
          f"median {late.km_loss_to_end.median():.0f} km from the grounding point")
    print(f"\nStranding moves (τ = {TAU_DAYS:g} d) by drogue type and rule:")
    print(moves.to_string(index=False))
    print(f"\nSaved → {out_cells}, {out_sum}, {out_moves}, {out_fig}")


if __name__ == "__main__":
    main()
