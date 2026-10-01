"""
Agulhas Region — Sensitivity to Lag and Grid
Checks that the results do not hinge on τ = 3.5 days or the 1° grid. The
reference setting (1°, τ = 3.5 d) is compared with τ = 2 and 5 days on the 1°
grid, and with the 0.5° and 2° grids at τ = 3.5 d, for both drogue types:

  • forecast skill — from 10's held-out validation of each setting, at common
                     horizons (7 and 28 days, linear between whole steps):
                     median position error of the Markov forecast, its skill
                     against persistence from the exact start (a baseline that
                     does not depend on the grid), and the Brier score of the
                     probability of having left R
  • release fates  — each release point from 04 after 1 year (still in R,
                     exited W, exited E, stranded), against the reference's 95%
                     bootstrap interval from 06, and the shift of the 28-day
                     centre of mass from the reference's

A release point is its 1° reference cell: on the 0.5° grid the mass starts
spread evenly over the ocean cells inside it, on the 2° grid in the cell that
contains the point. Log scores are not compared, since they depend on the
number of cells.

Needs 03 (every setting), 06 (reference) and 10 (every setting) first.

    .venv/bin/python scripts/11_sensitivity.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt

from config import DROGUE_TYPES, RES, TAU_DAYS, box_tag, operator_path, parse_args
from transport import load_operator, state_of

args = parse_args(__doc__)
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
BOX_TAG = box_tag(args.box)

REFERENCE = (RES, TAU_DAYS)
SETTINGS = [REFERENCE, (1.0, 2.0), (1.0, 5.0), (0.5, 3.5), (2.0, 3.5)]   # (grid °, τ days)
SKILL_DAYS = [7, 28]
COM_DAYS = 28
FATE_DAYS = 365
FATES = ["in_R", "exit_W", "exit_E", "exit_stranded"]
COLORS = ["#1a1a1a", "#2a78d6", "#7b3294", "#eb6834", "#1b9e77"]


def tag(res, tau):
    t = lambda x: f"{x:g}".replace(".", "p")
    return f"{t(res)}deg_{t(tau)}d_{BOX_TAG}"


def label(res, tau):
    return f"{res:g}°, τ = {tau:g} d"


def haversine_km(lo1, la1, lo2, la2):
    lo1, la1, lo2, la2 = map(np.radians, (lo1, la1, lo2, la2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# ── 1. Forecast skill from 10, at common horizons ────────────────────────────
skill_rows, curves = [], {}
for res, tau in SETTINGS:
    s = pd.read_csv(f"data/validation_summary_{tag(res, tau)}.csv")
    for name in DROGUE_TYPES:
        sub = s[s.type == name].sort_values("days")
        curves[(res, tau, name)] = sub
        for d in SKILL_DAYS:
            at = lambda col: np.interp(d, sub.days, sub[col])
            err, pers = at("err_markov_km"), at("err_persist_km")
            skill_rows.append({"res": res, "tau": tau, "type": name, "days": d,
                               "err_markov_km": err, "err_persist_km": pers,
                               "skill_vs_persist": 1 - err / pers, "brier_exit": at("brier_exit"),
                               "exit_pred": at("exit_pred"), "exit_obs": at("exit_obs")})
skill = pd.DataFrame(skill_rows)

# ── 2. Release fates for every setting ───────────────────────────────────────
points = pd.read_csv("data/release_candidates.csv")
boot = pd.read_csv(f"data/bootstrap_release_{tag(*REFERENCE)}.csv")


def release_vector(op, lon, lat, n_cells):
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


fate_rows, flag_rows = [], []
for res, tau in SETTINGS:
    for name in DROGUE_TYPES:
        op = load_operator(operator_path(name, res, tau, args.box))
        P, n_cells = op["P"], len(op["cell_flat"])
        flag_rows.append({"res": res, "tau": tau, "type": name, "n_cells": n_cells,
                          "flagged_share": op["flagged"][:n_cells].mean()})
        n_fate = int(round(FATE_DAYS / tau))
        k_com = COM_DAYS / tau                         # centre of mass between whole steps
        for _, pt in points.iterrows():
            p, c_lon, c_lat = release_vector(op, pt.lon, pt.lat, n_cells)
            com = {}
            for step in range(1, n_fate + 1):
                p = p @ P
                if step in (int(np.floor(k_com)), int(np.ceil(k_com))):
                    w = p[:n_cells] / max(p[:n_cells].sum(), 1e-12)
                    com[step] = np.array([w @ c_lon, w @ c_lat])
            lo, hi = int(np.floor(k_com)), int(np.ceil(k_com))
            com_pos = com[lo] + (k_com - lo) * (com[hi] - com[lo])
            row = {"res": res, "tau": tau, "type": name, "name": pt["name"],
                   "com_lon": com_pos[0], "com_lat": com_pos[1], "in_R": p[:n_cells].sum()}
            for k, lab in enumerate(op["exit_labels"]):
                row[f"exit_{lab}"] = p[n_cells + k]
            fate_rows.append(row)
fates = pd.DataFrame(fate_rows)
flags = pd.DataFrame(flag_rows)

# Each setting against the reference: fate inside the reference's bootstrap
# interval, size of the difference, 28-day centre-of-mass shift
ref = fates[(fates.res == REFERENCE[0]) & (fates.tau == REFERENCE[1])].set_index(["type", "name"])
ci = boot[boot.days == FATE_DAYS].set_index(["drogue", "name", "metric"])
fates["com_shift_km"] = [haversine_km(r.com_lon, r.com_lat, *ref.loc[(r.type, r["name"]), ["com_lon", "com_lat"]])
                         for _, r in fates.iterrows()]
for m in FATES:
    lo = np.array([ci.loc[(r.type, r["name"], m), "ci_low"] for _, r in fates.iterrows()])
    hi = np.array([ci.loc[(r.type, r["name"], m), "ci_high"] for _, r in fates.iterrows()])
    fates[f"{m}_diff"] = fates[m] - np.array([ref.loc[(r.type, r["name"]), m] for _, r in fates.iterrows()])
    fates[f"{m}_in_ci"] = (fates[m] >= lo - 1e-9) & (fates[m] <= hi + 1e-9)
ref_check = np.abs(ref["in_R"].values - np.array(
    [ci.loc[(t, n, "in_R"), "estimate"] for t, n in ref.index])).max()

skill.to_csv(f"data/sensitivity_skill_{BOX_TAG}.csv", index=False)
fates.to_csv(f"data/sensitivity_fates_{BOX_TAG}.csv", index=False)

# ── 3. Summary ───────────────────────────────────────────────────────────────
print(f"Sensitivity to lag and grid (reference {label(*REFERENCE)}); reference fates match 06 "
      f"to {ref_check:.1e}\n")
print("Held-out forecast skill (from 10)")
print(f"{'setting':<17}{'type':<11}{'days':>5}{'Markov km':>11}{'persist km':>11}{'skill':>8}"
      f"{'Brier':>8}{'exit pred':>10}{'exit obs':>9}")
for _, r in skill.iterrows():
    print(f"{label(r.res, r.tau):<17}{r.type:<11}{r.days:>5g}{r.err_markov_km:>11.0f}{r.err_persist_km:>11.0f}"
          f"{r.skill_vs_persist:>8.0%}{r.brier_exit:>8.3f}{r.exit_pred:>10.1%}{r.exit_obs:>9.1%}")

print(f"\nRelease points after {FATE_DAYS} days against the reference ({len(points)} points per type)")
print(f"{'setting':<17}{'type':<11}{'flagged':>8}" + "".join(f"{m + ' in CI':>13}{'max |Δ|':>9}" for m in FATES)
      + f"{'28 d CoM shift':>16}")
for (res, tau, name), g in fates.groupby(["res", "tau", "type"], sort=False):
    fl = flags[(flags.res == res) & (flags.tau == tau) & (flags.type == name)].flagged_share.iloc[0]
    cols = "".join(f"{g[f'{m}_in_ci'].sum():>10}/{len(g):<2}{g[f'{m}_diff'].abs().max() * 100:>8.1f}pp"
                   for m in FATES)
    print(f"{label(res, tau):<17}{name:<11}{fl:>8.0%}{cols}{g.com_shift_km.median():>9.0f} km med")
print("in CI = points whose value lies inside the reference's 95% bootstrap interval; "
      "flagged = rows with < 10 drifters")
print(f"Saved → data/sensitivity_skill_{BOX_TAG}.csv, data/sensitivity_fates_{BOX_TAG}.csv")

# ── 4. Figure: error by horizon, 1-year still-in-R by release point ──────────
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for ax, name in zip(axes[0], DROGUE_TYPES):
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
order = points["name"].tolist()
y = np.arange(len(order))
offsets = np.linspace(-0.3, 0.3, len(SETTINGS))
for ax, name in zip(axes[1], DROGUE_TYPES):
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
fig.suptitle("Sensitivity to lag τ and grid size — box R", fontsize=12)
fig.tight_layout()
out = f"figures/agulhas_sensitivity_{BOX_TAG}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Figure saved → {out}")
plt.close()
