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
import matplotlib.pyplot as plt

from agulhas.config import DROGUE_TYPES, lag_tag, parse_args, region_prefix
from agulhas.drifters import Drifters, in_box
from agulhas.plotting import save_figure
from agulhas.transport import STRAND_MATCH_S, same_drogue

GROUPS = {1: "ran aground", 3: "stopped transmitting"}


def great_circle_km(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    return 6371.0 * np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl), -1, 1))


def record_ends(drifters, in_R):
    """Per drifter whose record ends in R, how long before the end it lost its drogue."""
    ds, drogue, t, lon, lat = drifters.ds, drifters.drogue, drifters.t, drifters.lon, drifters.lat
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
            km = great_circle_km(lon[i], lat[i], lon[b], lat[b])
        rows.append({"traj": k, "fate": GROUPS[int(typedeath[k])], "state": state, "gap_days": gap,
                     "km_loss_to_end": km, "end_lon": lon[b], "end_lat": lat[b]})
    return pd.DataFrame(rows)


def summarise(ends, fate, tau):
    """One fate's drifters, by when they lost the drogue."""
    d = ends[ends.fate == fate]
    lost = d[d.state == "lost drogue"].gap_days
    return {"fate": fate, "drifters": len(d),
            "drogued_at_end": int((d.state == "drogued at end").sum()),
            "never_drogued": int((d.state == "never drogued").sum()),
            "lost_le_1d": int((lost <= 1).sum()),
            "lost_1d_to_tau": int(((lost > 1) & (lost <= tau)).sum()),
            "lost_gt_tau": int((lost > tau).sum()),
            "share_lost_le_1d": (lost <= 1).sum() / len(d),
            "median_gap_days": lost.median()}


def stranding_moves(drifters, in_R, strand_t, tau):
    """Stranding moves each drogue type gets, under each rule for assigning them."""
    drogue, traj_idx = drifters.drogue, drifters.traj_idx
    start, end, stranded = drifters.moves(in_R, tau, strand_t)
    s, e = start[stranded], end[stranded]
    rows = []
    for name, flag in DROGUE_TYPES.items():
        for rule, keep in [("start", same_drogue(drogue, s, e, True, flag)),
                           ("both ends", (drogue[s] == flag) & (drogue[e] == flag))]:
            rows.append({"drogue": name, "rule": rule, "stranding_moves": int(keep.sum()),
                         "drifters": len(np.unique(traj_idx[s[keep]]))})
    return pd.DataFrame(rows)


def plot_gaps(ends, tau, path):
    """Time from drogue loss to the end of the record, per fate."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(-2, np.log10(max(ends.gap_days.max(), 10)), 30)
    for fate, colour in zip(GROUPS.values(), ["tab:red", "tab:grey"]):
        g = ends[(ends.fate == fate) & (ends.state == "lost drogue")].gap_days
        ax.hist(np.maximum(g, bins[0]), bins=bins, density=True, histtype="step", lw=2, color=colour,
                label=f"{fate} (n = {len(g)})")
    ax.axvline(1, color="k", ls=":", lw=1)
    ax.axvline(tau, color="k", ls="--", lw=1, label=f"τ = {tau:g} d")
    ax.set_xscale("log")
    ax.set_xlabel("Time from drogue loss to end of record (days)")
    ax.set_ylabel("Density")
    ax.set_title("Drifters whose record ends in R: when was the drogue lost?")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(path, trim=False)


def print_summary(table, ends, moves, n_grounded, tau):
    pd.set_option("display.width", 200)
    print(f"Drifters whose record ends in R ({n_grounded} ran aground with the subset's end matched):")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    late = ends[(ends.fate == "ran aground") & (ends.state == "lost drogue") & (ends.gap_days <= 1)]
    print(f"\nGrounded drifters that lost the drogue ≤ 1 d before grounding: {len(late)}, "
          f"median {late.km_loss_to_end.median():.0f} km from the grounding point")
    print(f"\nStranding moves (τ = {tau:g} d) by drogue type and rule:")
    print(moves.to_string(index=False))


def main(argv=None):
    args = parse_args(__doc__, lag=True, argv=argv)
    tag = lag_tag(args.tau, args.box)
    out_cells = f"data/drogue_loss_{tag}.csv"
    out_sum = f"data/drogue_loss_summary_{tag}.csv"
    out_moves = f"data/drogue_loss_stranding_moves_{tag}.csv"
    out_fig = f"figures/{region_prefix(args.data)}_drogue_loss_{tag}.png"

    drifters = Drifters(args.data)
    in_R = in_box(drifters.lon, drifters.lat, args.box)
    ends = record_ends(drifters, in_R)
    table = pd.DataFrame([summarise(ends, fate, args.tau) for fate in GROUPS.values()])

    strand_t = drifters.strand_times(in_R)
    moves = stranding_moves(drifters, in_R, strand_t, args.tau)
    n_grounded = int((strand_t[np.cumsum(drifters.ds["rowsize"].values) - 1] >= 0).sum())

    ends.to_csv(out_cells, index=False)
    table.to_csv(out_sum, index=False)
    moves.to_csv(out_moves, index=False)
    plot_gaps(ends, args.tau, out_fig)
    print_summary(table, ends, moves, n_grounded, args.tau)
    print(f"\nSaved → {out_cells}, {out_sum}, {out_moves}, {out_fig}")


if __name__ == "__main__":
    main()
