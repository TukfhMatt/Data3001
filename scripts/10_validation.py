"""
Agulhas Region — Held-out Validation
Tests how well a transition matrix predicts where drifters it has never seen
actually go. Drifters (never single observations) are split into K_FOLDS
folds per drogue type; each fold's matrix is built from the other folds with
the same helpers as 03, then predicts the held-out drifters.

Test starts are the held-out drifters' positions in R at 00 UTC each day.
The true outcome is followed at whole steps of τ, exactly as the chain moves:
the cell at t0 + kτ, the exit side the first time the drifter is outside R
at a step, or stranded once the drifter has run aground inside R (exits and
stranding are absorbing, as in the matrix). Every whole step up to
HORIZON_DAYS is scored, so runs at different τ share a day axis for 11.

Scores, per drogue type and horizon:
  • position error   — distance from the predicted centre of mass (given the
                       material is still in R) to the actual position, against
                       two baselines: persistence (stays at its start) and
                       advection by the training set's mean current per cell.
                       The matrix only knows a start's cell, so each baseline
                       is also run from the cell centre: that pair isolates
                       the grid's own error from forecast skill
  • log score        — log of the probability given to the true outcome
                       state, against a climatology baseline (training
                       frequency of each outcome at that horizon)
  • exit calibration — predicted vs observed probability of having left R
                       across an edge, and the Brier score of that
                       probability; predicted vs observed share stranded

    .venv/bin/python scripts/10_validation.py --res 1 --tau 3.5
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt
import xarray as xr

from config import DROGUE_TYPES, box_tag, grid_shape, parse_args
from transport import (EXIT_LABELS, STRANDED, last_obs_index, move_indices, row_normalise,
                       strand_times, transition_states)

args = parse_args(__doc__, grid=True, lag=True)
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
RES, TAU = args.res, args.tau
K_FOLDS = 5
SEED = 0
HORIZON_DAYS = 28                    # score every whole step of τ up to this horizon
HORIZON_STEPS = list(range(1, int(np.ceil(HORIZON_DAYS / TAU - 1e-9)) + 1))
LOG_FLOOR = 1e-6                     # probability floor for the log score
EXIT_BINS = np.linspace(0, 1, 11)
COLORS = {"Markov": "#2a78d6", "Mean-current advection": "#eb6834", "Persistence": "#7a7a7a"}
TAG = f"{f'{RES:g}'.replace('.', 'p')}deg_{f'{TAU:g}'.replace('.', 'p')}d_{box_tag(args.box)}"

# ── 1. Load subset, grid, states ─────────────────────────────────────────────
ds = xr.open_dataset(args.data)
lon = ds["lon"].values.astype(float)
lat = ds["lat"].values.astype(float)
ve = ds["ve"].values.astype(float)
vn = ds["vn"].values.astype(float)
drogue = ds["drogue_status"].values.astype(bool)
t = ds["time"].values.astype("datetime64[s]").astype(np.int64)
traj_idx = np.repeat(np.arange(ds.sizes["traj"]), ds["rowsize"].values)

n_lon, n_lat = grid_shape(args.box, RES)
in_R = (lon >= LON_MIN) & (lon < LON_MAX) & (lat >= LAT_MIN) & (lat < LAT_MAX)
cell_flat = np.where(in_R, ((lat - LAT_MIN) // RES).astype(int) * n_lon
                     + ((lon - LON_MIN) // RES).astype(int), -1)
cells = np.unique(cell_flat[in_R])            # same ocean states as 03
n_cells = len(cells)
n_states = n_cells + len(EXIT_LABELS)
state_of_cell = np.full(n_lat * n_lon, -1)
state_of_cell[cells] = np.arange(n_cells)
cj, ci = np.divmod(cells, n_lon)
cell_lon = np.minimum(LON_MIN + (ci + 0.5) * RES, LON_MAX)
cell_lat = np.minimum(LAT_MIN + (cj + 0.5) * RES, LAT_MAX)

key = traj_idx.astype(np.int64) * 10**10 + t
tau_s = int(round(TAU * 86400))
strand_t = strand_times(ds, traj_idx, t, in_R)
last_obs = last_obs_index(traj_idx)
start_all, end_all, strand_all = move_indices(traj_idx, t, in_R, TAU, strand_t)   # all moves from R


def exit_side(lo, la):
    return np.argmax(np.stack([LON_MIN - lo, lo - LON_MAX, LAT_MIN - la, la - LAT_MAX]), axis=0)


def obs_at(idx, dt):
    """Index of the same drifter's observation dt seconds after idx, or -1."""
    target = key[idx] + dt
    j = np.minimum(np.searchsorted(key, target), len(key) - 1)
    return np.where(key[j] == target, j, -1)


def haversine_km(lo1, la1, lo2, la2):
    lo1, la1, lo2, la2 = map(np.radians, (lo1, la1, lo2, la2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# ── 2. Test samples: daily starts, true outcome at each whole step ───────────
def test_samples(flag, test_traj):
    s0 = np.where(in_R & (drogue == flag) & np.isin(traj_idx, test_traj) & (t % 86400 == 0))[0]
    K = max(HORIZON_STEPS)
    state = np.full((len(s0), K + 1), -1)
    state[:, 0] = state_of_cell[cell_flat[s0]]
    pos = np.full((len(s0), K + 1, 2), np.nan)
    pos[:, 0] = np.c_[lon[s0], lat[s0]]
    alive = state[:, 0] >= 0
    exited = np.zeros(len(s0), bool)
    for k in range(1, K + 1):
        # Exits and stranding are absorbing: such a sample keeps its state and
        # needs no further observations
        carry = alive & exited
        state[carry, k] = state[carry, k - 1]
        j = obs_at(s0, k * tau_s)
        has = j >= 0
        has[has] &= drogue[j[has]] == flag        # same drogue type, as in training
        cand = alive & ~exited
        strands = (cand & ~has & (strand_t[s0] >= 0) & (strand_t[s0] < t[s0] + k * tau_s)
                   & (drogue[last_obs[s0]] == flag))
        state[strands, k] = n_cells + STRANDED
        exited |= strands
        alive &= ~(cand & ~has & ~strands)         # lost: no matching observation
        ok = cand & has
        jj = np.where(ok, j, 0)
        outside = ok & (cell_flat[jj] < 0)
        state[outside, k] = n_cells + exit_side(lon[jj[outside]], lat[jj[outside]])
        exited |= outside
        inside = ok & ~outside
        state[inside, k] = state_of_cell[cell_flat[jj[inside]]]
        pos[inside, k] = np.c_[lon[jj[inside]], lat[jj[inside]]]
    return s0, state, pos


# ── 3. Training matrix and baselines for one fold ────────────────────────────
def train_fold(flag, train_traj):
    keep = (np.isin(traj_idx[start_all], train_traj) & (drogue[start_all] == flag)
            & (drogue[end_all] == flag))
    s, e = start_all[keep], end_all[keep]
    fs, ts = transition_states(cell_flat, cells, s, e, lon, lat, args.box, strand_all[keep])
    C = np.zeros((n_states, n_states))
    np.add.at(C, (fs, ts), 1)
    P, empty = row_normalise(C, cells, n_lon)
    # Mean current per cell from training observations of this type
    m = in_R & (drogue == flag) & np.isin(traj_idx, train_traj) & np.isfinite(ve) & np.isfinite(vn)
    st = state_of_cell[cell_flat[m]]
    cnt = np.bincount(st, minlength=n_cells)
    u = np.bincount(st, ve[m], n_cells) / np.maximum(cnt, 1)
    v = np.bincount(st, vn[m], n_cells) / np.maximum(cnt, 1)
    return P, u, v, len(s)


def advect(lo, la, u, v, k_steps):
    """Deterministic forecast: move by the cell's mean current for each step;
    returns final positions and the exit state (−1 while still in R)."""
    lo, la = lo.copy(), la.copy()
    ex = np.full(len(lo), -1)
    for _ in range(k_steps):
        live = ex < 0
        c = np.where(live, cell_flat_of(lo, la), -1)
        stt = np.where(c >= 0, state_of_cell[np.maximum(c, 0)], -1)
        has = stt >= 0
        du = np.where(has, u[np.maximum(stt, 0)], 0.0) * tau_s
        dv = np.where(has, v[np.maximum(stt, 0)], 0.0) * tau_s
        lo = np.where(live, lo + du / (111_320.0 * np.cos(np.radians(la))), lo)
        la = np.where(live, la + dv / 110_574.0, la)
        out = live & ~((lo >= LON_MIN) & (lo < LON_MAX) & (la >= LAT_MIN) & (la < LAT_MAX))
        ex[out] = n_cells + exit_side(lo[out], la[out])
    return lo, la, ex


def cell_flat_of(lo, la):
    inside = (lo >= LON_MIN) & (lo < LON_MAX) & (la >= LAT_MIN) & (la < LAT_MAX)
    return np.where(inside, ((la - LAT_MIN) // RES).astype(int) * n_lon
                    + ((lo - LON_MIN) // RES).astype(int), -1)


# ── 4. Cross-validate ────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
records, exit_records = [], []
for name, flag in DROGUE_TYPES.items():
    trajs = np.unique(traj_idx[in_R & (drogue == flag)])
    folds = np.array_split(rng.permutation(trajs), K_FOLDS)
    for f, test_traj in enumerate(folds):
        train_traj = np.setdiff1d(trajs, test_traj)
        P, u, v, n_train = train_fold(flag, train_traj)
        s0, state, pos = test_samples(flag, test_traj)
        # Climatology baseline: training-set frequency of each outcome state at k
        _, tr_state, _ = test_samples(flag, train_traj)
        Pk = np.eye(n_states)
        for k in HORIZON_STEPS:
            Pk = Pk @ P
            ok = (state[:, 0] >= 0) & (state[:, k] >= 0)
            st0, stk = state[ok, 0], state[ok, k]
            pred = Pk[st0]                                      # (n, n_states)
            p_true = pred[np.arange(len(stk)), stk]
            clim = np.bincount(tr_state[tr_state[:, k] >= 0, k], minlength=n_states).astype(float)
            clim /= clim.sum()
            p_exit = pred[:, n_cells:n_cells + STRANDED].sum(axis=1)
            actual_exit = (stk >= n_cells) & (stk < n_cells + STRANDED)
            actual_strand = stk == n_cells + STRANDED
            exit_records.append(pd.DataFrame({"type": name, "k": k, "p_exit": p_exit,
                                              "exited": actual_exit}))
            # Position error for samples still in R at k
            stay = stk < n_cells
            mass = pred[stay, :n_cells]
            w = mass / np.maximum(mass.sum(axis=1, keepdims=True), 1e-12)
            com_lon, com_lat = w @ cell_lon, w @ cell_lat
            act = pos[ok][stay, k]
            start = pos[ok][stay, 0]
            adv_lo, adv_la, adv_ex = advect(start[:, 0], start[:, 1], u, v, k)
            adv_ok = adv_ex < 0
            c0 = st0[stay]                                       # start state (cell)
            cen_lo, cen_la = cell_lon[c0], cell_lat[c0]
            advc_lo, advc_la, advc_ex = advect(cen_lo, cen_la, u, v, k)
            advc_ok = advc_ex < 0
            records.append({
                "type": name, "fold": f, "k": k, "days": k * TAU, "n_train_pairs": n_train,
                "n_test": int(ok.sum()), "n_test_drifters": len(test_traj),
                "log_markov": np.mean(np.log(np.maximum(p_true, LOG_FLOOR))),
                "log_clim": np.mean(np.log(np.maximum(clim[stk], LOG_FLOOR))),
                "err_markov_km": np.median(haversine_km(com_lon, com_lat, act[:, 0], act[:, 1])),
                "err_persist_km": np.median(haversine_km(start[:, 0], start[:, 1], act[:, 0], act[:, 1])),
                "err_advect_km": np.median(haversine_km(adv_lo[adv_ok], adv_la[adv_ok],
                                                        act[adv_ok, 0], act[adv_ok, 1])),
                "err_persist_centre_km": np.median(haversine_km(cen_lo, cen_la, act[:, 0], act[:, 1])),
                "err_advect_centre_km": np.median(haversine_km(advc_lo[advc_ok], advc_la[advc_ok],
                                                               act[advc_ok, 0], act[advc_ok, 1])),
                "exit_pred": p_exit.mean(), "exit_obs": actual_exit.mean(),
                "brier_exit": np.mean((p_exit - actual_exit) ** 2),
                "strand_pred": pred[:, n_cells + STRANDED].mean(), "strand_obs": actual_strand.mean(),
            })
        print(f"{name:<9} fold {f + 1}/{K_FOLDS}: {len(test_traj)} held-out drifters, "
              f"{int((state[:, 0] >= 0).sum()):,} test starts")

res = pd.DataFrame(records)
res.to_csv(f"data/validation_folds_{TAG}.csv", index=False)
summary = res.groupby(["type", "days"]).agg(
    n_test=("n_test", "sum"),
    err_markov_km=("err_markov_km", "mean"), err_advect_km=("err_advect_km", "mean"),
    err_persist_km=("err_persist_km", "mean"),
    err_advect_centre_km=("err_advect_centre_km", "mean"),
    err_persist_centre_km=("err_persist_centre_km", "mean"),
    log_markov=("log_markov", "mean"), log_clim=("log_clim", "mean"),
    exit_pred=("exit_pred", "mean"), exit_obs=("exit_obs", "mean"),
    brier_exit=("brier_exit", "mean"), strand_pred=("strand_pred", "mean"),
    strand_obs=("strand_obs", "mean")).reset_index()
summary["skill_vs_persist"] = 1 - summary.err_markov_km / summary.err_persist_km
summary["skill_vs_advect"] = 1 - summary.err_markov_km / summary.err_advect_km
summary["skill_vs_advect_centre"] = 1 - summary.err_markov_km / summary.err_advect_centre_km
summary.to_csv(f"data/validation_summary_{TAG}.csv", index=False)

print(f"\nHeld-out validation, {K_FOLDS}-fold by drifter ({RES:g}°, τ = {TAU:g} d); mean over folds")
print(f"{'':<16}{'':>9}{'median error (km)':^45}{'skill vs advect':^18}")
print(f"{'type':<10}{'days':>6}{'tests':>9}{'Markov':>9}{'adv':>9}{'adv@ctr':>9}{'pers':>9}{'pers@ctr':>9}"
      f"{'exact':>9}{'@ctr':>9}{'log Mkv':>9}{'log clim':>9}{'exit pred':>10}{'exit obs':>9}{'Brier':>8}{'strand pred':>12}{'obs':>7}")
for _, r in summary.iterrows():
    print(f"{r.type:<10}{r.days:>6g}{int(r.n_test):>9,}{r.err_markov_km:>9.0f}{r.err_advect_km:>9.0f}"
          f"{r.err_advect_centre_km:>9.0f}{r.err_persist_km:>9.0f}{r.err_persist_centre_km:>9.0f}"
          f"{r.skill_vs_advect:>9.0%}{r.skill_vs_advect_centre:>9.0%}{r.log_markov:>9.2f}{r.log_clim:>9.2f}"
          f"{r.exit_pred:>10.1%}{r.exit_obs:>9.1%}{r.brier_exit:>8.3f}{r.strand_pred:>12.2%}{r.strand_obs:>7.2%}")
print("Errors: median km from forecast to actual position (still-in-R samples). @ctr = baseline "
      "started from the cell centre, the same information the matrix has. Log score: higher is better.")
print(f"Saved → data/validation_folds_{TAG}.csv, data/validation_summary_{TAG}.csv")

# ── 5. Figure: position error by horizon, exit calibration ───────────────────
ex = pd.concat(exit_records)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, name in zip(axes[:2], DROGUE_TYPES):
    sub = summary[summary.type == name]
    for label, col in [("Markov", "err_markov_km"), ("Mean-current advection", "err_advect_km"),
                       ("Persistence", "err_persist_km")]:
        spread = res[res.type == name].groupby("days")[col].agg(["min", "max"])
        ax.plot(sub.days, sub[col], marker="o", lw=2, color=COLORS[label], label=label)
        ax.fill_between(spread.index, spread["min"], spread["max"], color=COLORS[label], alpha=0.15)
    ax.plot(sub.days, sub.err_advect_centre_km, ls="--", lw=1.5, color=COLORS["Mean-current advection"],
            label="Mean-current advection from cell centre")
    ax.set_xticks(np.arange(0, sub.days.max() + 1, 7))
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Median position error (km)")
    ax.set_title(f"{name.capitalize()}: forecast error on held-out drifters\n(band = range over {K_FOLDS} folds)",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
ax = axes[2]
ax.plot([0, 1], [0, 1], color="#9e9e9e", lw=1, ls="--", label="Perfect calibration")
for name, color in [("drogued", "#2a78d6"), ("undrogued", "#eb6834")]:
    e = ex[(ex.type == name) & (ex.k == max(HORIZON_STEPS))]
    b = pd.cut(e.p_exit, EXIT_BINS, include_lowest=True)
    cal = e.groupby(b, observed=True).agg(p=("p_exit", "mean"), o=("exited", "mean"), n=("exited", "size"))
    cal = cal[cal.n >= 30]
    ax.plot(cal.p, cal.o, marker="o", lw=2, color=color, label=name.capitalize())
ax.set_xlabel(f"Predicted probability of having left R across an edge by day {max(HORIZON_STEPS) * TAU:g}")
ax.set_ylabel("Observed share that left")
ax.set_title("Exit calibration (bins with ≥ 30 test starts)", fontsize=10)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle(f"Held-out validation — {RES:g}° grid, τ = {TAU:g} d, box R", fontsize=12, y=1.04)
out = f"figures/agulhas_validation_{TAG}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Figure saved → {out}")
plt.close()
