"""
Agulhas Region — Sensitivity to Lag and Grid
Checks that the results do not hinge on τ = 3.5 days or the 1° grid. The
reference setting (1°, τ = 3.5 d) is compared with τ = 2 and 5 days on the 1°
grid, and with the 0.5° and 2° grids at τ = 3.5 d, for both drogue types:

  • forecast skill — from held_out's validation of each setting, at common
                     horizons (7 and 28 days, linear between whole steps):
                     median position error of the Markov forecast, its skill
                     against persistence from the exact start (a baseline that
                     does not depend on the grid), and the Brier score of the
                     probability of having left R
  • release fates  — each release point from shipping_overlay after 1 year (still in R,
                     exited W, exited E, stranded), against the reference's 95%
                     bootstrap interval from bootstrap, and the shift of the 28-day
                     centre of mass from the reference's

A release point is its 1° reference cell: on the 0.5° grid the mass starts
spread evenly over the ocean cells inside it, on the 2° grid in the cell that
contains the point. Log scores are not compared, since they depend on the
number of cells.

Needs transition_matrix (every setting), bootstrap (reference) and held_out
(every setting) first.

    python -m agulhas sensitivity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agulhas.config import (DROGUE_TYPES, RES, TAU_DAYS, box_tag, operator_path, parse_args, region_prefix,
                            release_points_path, setting_tag)
from agulhas.plotting import save_figure
from agulhas.stats import haversine_km
from agulhas.transport import load_operator, state_of

REFERENCE = (RES, TAU_DAYS)
SETTINGS = [REFERENCE, (1.0, 2.0), (1.0, 5.0), (0.5, 3.5), (2.0, 3.5)]   # (grid °, τ days)
SKILL_DAYS = [7, 28]
COM_DAYS = 28
FATE_DAYS = 365
FATES = ["in_R", "exit_W", "exit_E", "exit_stranded"]
COLORS = ["#1a1a1a", "#2a78d6", "#7b3294", "#eb6834", "#1b9e77"]


def label(res, tau):
    return f"{res:g}°, τ = {tau:g} d"


# ── Forecast skill from held_out ─────────────────────────────────────────────

def load_skill(box):
    """Held-out forecast skill of every setting at the common horizons, and each
    setting's full error curve per drogue type."""
    rows, curves = [], {}
    for res, tau in SETTINGS:
        summary = pd.read_csv(f"data/validation_summary_{setting_tag(res, tau, box)}.csv")
        for name in DROGUE_TYPES:
            curve = summary[summary.type == name].sort_values("days")
            curves[(res, tau, name)] = curve
            rows += [skill_at(curve, days, res, tau, name) for days in SKILL_DAYS]
    return pd.DataFrame(rows), curves


def skill_at(curve, days, res, tau, name):
    """One setting's skill at a horizon, linear between its whole steps."""
    at = lambda col: np.interp(days, curve.days, curve[col])
    err, pers = at("err_markov_km"), at("err_persist_km")
    return {"res": res, "tau": tau, "type": name, "days": days,
            "err_markov_km": err, "err_persist_km": pers,
            "skill_vs_persist": 1 - err / pers, "brier_exit": at("brier_exit"),
            "exit_pred": at("exit_pred"), "exit_obs": at("exit_obs")}


# ── Release fates for every setting ──────────────────────────────────────────

def release_vector(op, lon, lat):
    """Mass on the ocean cells inside the 1° reference cell around (lon, lat),
    or on the cell containing the point when no cell centre lies inside it."""
    n_lon = len(op["lon_edges"]) - 1
    j, i = np.divmod(op["cell_flat"], n_lon)
    c_lon = (op["lon_edges"][i] + op["lon_edges"][i + 1]) / 2
    c_lat = (op["lat_edges"][j] + op["lat_edges"][j + 1]) / 2
    half = REFERENCE[0] / 2
    inside = np.where((np.abs(c_lon - lon) < half) & (np.abs(c_lat - lat) < half))[0]
    p0 = np.zeros(op["P"].shape[0])
    if inside.size:
        p0[inside] = 1.0 / inside.size
    else:
        p0[state_of(op, lon, lat)] = 1.0
    return p0, c_lon, c_lat


def fate_of(op, point, tau):
    """Where one release point's mass is after FATE_DAYS, and its centre of mass
    at COM_DAYS (linear between the whole steps either side)."""
    P, n_cells = op["P"], len(op["cell_flat"])
    k_com = COM_DAYS / tau
    lo, hi = int(np.floor(k_com)), int(np.ceil(k_com))
    p, c_lon, c_lat = release_vector(op, point.lon, point.lat)
    com = {}
    for step in range(1, int(round(FATE_DAYS / tau)) + 1):
        p = p @ P
        if step in (lo, hi):
            w = p[:n_cells] / max(p[:n_cells].sum(), 1e-12)
            com[step] = np.array([w @ c_lon, w @ c_lat])
    com_pos = com[lo] + (k_com - lo) * (com[hi] - com[lo])
    row = {"com_lon": com_pos[0], "com_lat": com_pos[1], "in_R": p[:n_cells].sum()}
    for k, lab in enumerate(op["exit_labels"]):
        row[f"exit_{lab}"] = p[n_cells + k]
    return row


def load_fates(points, box):
    """Fates of every release point under every setting, and each matrix's share of flagged rows."""
    fate_rows, flag_rows = [], []
    for res, tau in SETTINGS:
        for name in DROGUE_TYPES:
            op = load_operator(operator_path(name, res, tau, box))
            n_cells = len(op["cell_flat"])
            flag_rows.append({"res": res, "tau": tau, "type": name, "n_cells": n_cells,
                              "flagged_share": op["flagged"][:n_cells].mean()})
            for _, point in points.iterrows():
                fate_rows.append({"res": res, "tau": tau, "type": name, "name": point["name"],
                                  **fate_of(op, point, tau)})
    return pd.DataFrame(fate_rows), pd.DataFrame(flag_rows)


def compare_to_reference(fates, ci):
    """Add to fates each setting's difference from the reference: the size of each
    fate's difference, whether it lies inside the reference's bootstrap interval,
    and the 28-day centre-of-mass shift. Returns how far the reference's own fates
    are from bootstrap's estimates, as a check that both read the same matrices."""
    ref = fates[(fates.res == REFERENCE[0]) & (fates.tau == REFERENCE[1])].set_index(["type", "name"])
    fates["com_shift_km"] = [haversine_km(r.com_lon, r.com_lat, *ref.loc[(r.type, r["name"]), ["com_lon", "com_lat"]])
                             for _, r in fates.iterrows()]
    for m in FATES:
        lo = np.array([ci.loc[(r.type, r["name"], m), "ci_low"] for _, r in fates.iterrows()])
        hi = np.array([ci.loc[(r.type, r["name"], m), "ci_high"] for _, r in fates.iterrows()])
        fates[f"{m}_diff"] = fates[m] - np.array([ref.loc[(r.type, r["name"]), m] for _, r in fates.iterrows()])
        fates[f"{m}_in_ci"] = (fates[m] >= lo - 1e-9) & (fates[m] <= hi + 1e-9)
    return np.abs(ref["in_R"].values - np.array([ci.loc[(t, n, "in_R"), "estimate"] for t, n in ref.index])).max()


# ── Summary ──────────────────────────────────────────────────────────────────

def print_skill(skill, ref_check):
    print(f"Sensitivity to lag and grid (reference {label(*REFERENCE)}); reference fates match bootstrap "
          f"to {ref_check:.1e}\n")
    print("Held-out forecast skill (from held_out)")
    print(f"{'setting':<17}{'type':<11}{'days':>5}{'Markov km':>11}{'persist km':>11}{'skill':>8}"
          f"{'Brier':>8}{'exit pred':>10}{'exit obs':>9}")
    for _, r in skill.iterrows():
        print(f"{label(r.res, r.tau):<17}{r.type:<11}{r.days:>5g}{r.err_markov_km:>11.0f}{r.err_persist_km:>11.0f}"
              f"{r.skill_vs_persist:>8.0%}{r.brier_exit:>8.3f}{r.exit_pred:>10.1%}{r.exit_obs:>9.1%}")


def print_fates(fates, flags, n_points):
    print(f"\nRelease points after {FATE_DAYS} days against the reference ({n_points} points per type)")
    print(f"{'setting':<17}{'type':<11}{'flagged':>8}" + "".join(f"{m + ' in CI':>13}{'max |Δ|':>9}" for m in FATES)
          + f"{'28 d CoM shift':>16}")
    for (res, tau, name), g in fates.groupby(["res", "tau", "type"], sort=False):
        fl = flags[(flags.res == res) & (flags.tau == tau) & (flags.type == name)].flagged_share.iloc[0]
        cols = "".join(f"{g[f'{m}_in_ci'].sum():>10}/{len(g):<2}{g[f'{m}_diff'].abs().max() * 100:>8.1f}pp"
                       for m in FATES)
        print(f"{label(res, tau):<17}{name:<11}{fl:>8.0%}{cols}{g.com_shift_km.median():>9.0f} km med")
    print("in CI = points whose value lies inside the reference's 95% bootstrap interval; "
          "flagged = rows with < 10 drifters")


# ── Figure: error by horizon, 1-year still-in-R by release point ────────────

def plot_error_curves(axes, curves):
    for ax, name in zip(axes, DROGUE_TYPES):
        ref_curve = curves[(*REFERENCE, name)]
        ax.plot(ref_curve.days, ref_curve.err_persist_km, color="#9e9e9e", lw=1.5, ls="--",
                label="Persistence (reference)")
        for (res, tau), color in zip(SETTINGS, COLORS):
            c = curves[(res, tau, name)]
            ax.plot(c.days, c.err_markov_km, marker="o", ms=3.5, lw=2 if (res, tau) == REFERENCE else 1.3,
                    color=color, label=f"Markov, {label(res, tau)}")
        ax.set_xticks(range(0, 29, 7))
        ax.set_xlabel("Horizon (days)")
        ax.set_ylabel("Median position error on held-out drifters (km)")
        ax.set_title(f"{name.capitalize()}: forecast error by setting", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")


def plot_year_fates(axes, fates, ci, order):
    y = np.arange(len(order))
    offsets = np.linspace(-0.3, 0.3, len(SETTINGS))
    for ax, name in zip(axes, DROGUE_TYPES):
        b = ci.xs((name, "in_R"), level=("drogue", "metric")).reindex(order)
        ax.barh(y, (b.ci_high - b.ci_low) * 100, left=b.ci_low * 100, height=0.8, color="#d9d9d9",
                label="Reference 95% bootstrap interval")
        for (res, tau), color, off in zip(SETTINGS, COLORS, offsets):
            g = fates[(fates.res == res) & (fates.tau == tau) & (fates.type == name)].set_index("name").reindex(order)
            ax.plot(g.in_R * 100, y + off, "o", ms=4, color=color, label=label(res, tau))
        ax.set_yticks(y)
        ax.set_yticklabels(order, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(f"Still in R after {FATE_DAYS} days (%)")
        ax.set_title(f"{name.capitalize()}: 1-year fate by setting", fontsize=10)
        ax.grid(alpha=0.3, axis="x")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)


def plot_sensitivity(curves, fates, ci, order, path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plot_error_curves(axes[0], curves)
    plot_year_fates(axes[1], fates, ci, order)
    fig.suptitle("Sensitivity to lag τ and grid size — box R", fontsize=12)
    fig.tight_layout()
    save_figure(path)


def main(argv=None):
    args = parse_args(__doc__, argv=argv)
    out_skill = f"data/sensitivity_skill_{box_tag(args.box)}.csv"
    out_fates = f"data/sensitivity_fates_{box_tag(args.box)}.csv"
    out_png = f"figures/{region_prefix(args.data)}_sensitivity_{box_tag(args.box)}.png"

    skill, curves = load_skill(args.box)
    points = pd.read_csv(release_points_path(args.box))
    boot = pd.read_csv(f"data/bootstrap_release_{setting_tag(*REFERENCE, args.box)}.csv")
    ci = boot[boot.days == FATE_DAYS].set_index(["drogue", "name", "metric"])
    fates, flags = load_fates(points, args.box)
    ref_check = compare_to_reference(fates, ci)
    skill.to_csv(out_skill, index=False)
    fates.to_csv(out_fates, index=False)

    print_skill(skill, ref_check)
    print_fates(fates, flags, len(points))
    print(f"Saved → {out_skill}, {out_fates}")
    plot_sensitivity(curves, fates, ci, points["name"].tolist(), out_png)
    print(f"Figure saved → {out_png}")


if __name__ == "__main__":
    main()
