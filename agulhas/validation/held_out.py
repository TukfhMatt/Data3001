"""
Agulhas Region — Held-out Validation
Tests how well a transition matrix predicts where drifters it has never seen
actually go. Drifters (never single observations) are split into K_FOLDS
folds per drogue type; each fold's matrix is built from the other folds with
the same helpers as transition_matrix, then predicts the held-out drifters.

Test starts are the held-out drifters' positions in R at 00 UTC each day.
The true outcome is followed at whole steps of τ, exactly as the chain moves:
the cell at t0 + kτ, the exit side the first time the drifter is outside R
at a step, or stranded once the drifter has run aground inside R (exits and
stranding are absorbing, as in the matrix). Every whole step up to
HORIZON_DAYS is scored, so runs at different τ share a day axis for whole_box_analysis.

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

    python -m agulhas held_out --res 1 --tau 3.5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agulhas.config import DROGUE_TYPES, parse_args, region_prefix, setting_tag
from agulhas.drifters import Drifters, in_box
from agulhas.moves import MoveSet
from agulhas.plotting import DROGUE_COLORS, save_figure
from agulhas.stats import haversine_km
from agulhas.transport import EXIT_LABELS, STRANDED, row_normalise, same_drogue, transition_states

K_FOLDS = 5
SEED = 0
HORIZON_DAYS = 28                    # score every whole step of τ up to this horizon
LOG_FLOOR = 1e-6                     # probability floor for the log score
EXIT_BINS = np.linspace(0, 1, 11)
COLORS = {"Markov": "#2a78d6", "Mean-current advection": "#eb6834", "Persistence": "#7a7a7a"}


class HeldOut:
    """The subset on one grid and lag, with what every fold needs: the ocean states
    (the same as transition_matrix), a key per observation to find the same drifter
    a whole number of steps later, and all moves from R."""

    def __init__(self, drifters, box, res, tau):
        self.d, self.box, self.tau = drifters, box, tau
        self.strand_t = drifters.strand_times(in_box(drifters.lon, drifters.lat, box))
        self.moves = MoveSet(drifters, box, res, tau, self.strand_t)
        self.grid, self.in_R, self.cell_flat = self.moves.grid, self.moves.in_R, self.moves.cell_flat
        self.cells = np.unique(self.cell_flat[self.in_R])
        self.n_cells = len(self.cells)
        self.n_states = self.n_cells + len(EXIT_LABELS)
        self.state_of_cell = np.full(self.grid.n_cells, -1)
        self.state_of_cell[self.cells] = np.arange(self.n_cells)
        cj, ci = np.divmod(self.cells, self.grid.n_lon)
        self.cell_lon = np.minimum(self.grid.lon_min + (ci + 0.5) * res, self.grid.lon_max)
        self.cell_lat = np.minimum(self.grid.lat_min + (cj + 0.5) * res, self.grid.lat_max)
        self.key = drifters.traj_idx.astype(np.int64) * 10**10 + drifters.t
        self.tau_s = int(round(tau * 86400))
        self.horizon_steps = list(range(1, int(np.ceil(HORIZON_DAYS / tau - 1e-9)) + 1))

    def exit_side(self, lo, la):
        lon_min, lon_max, lat_min, lat_max = self.box
        return np.argmax(np.stack([lon_min - lo, lo - lon_max, lat_min - la, la - lat_max]), axis=0)

    def obs_at(self, idx, dt):
        """Index of the same drifter's observation dt seconds after idx, or -1."""
        target = self.key[idx] + dt
        j = np.minimum(np.searchsorted(self.key, target), len(self.key) - 1)
        return np.where(self.key[j] == target, j, -1)

    # ── Test samples: daily starts, true outcome at each whole step ──────────

    def test_samples(self, flag, test_traj):
        d = self.d
        s0 = np.where(self.in_R & (d.drogue == flag) & np.isin(d.traj_idx, test_traj) & (d.t % 86400 == 0))[0]
        K = max(self.horizon_steps)
        state = np.full((len(s0), K + 1), -1)
        state[:, 0] = self.state_of_cell[self.cell_flat[s0]]
        pos = np.full((len(s0), K + 1, 2), np.nan)
        pos[:, 0] = np.c_[d.lon[s0], d.lat[s0]]
        alive = state[:, 0] >= 0
        exited = np.zeros(len(s0), bool)
        for k in range(1, K + 1):
            # Exits and stranding are absorbing: such a sample keeps its state and
            # needs no further observations
            carry = alive & exited
            state[carry, k] = state[carry, k - 1]
            j = self.obs_at(s0, k * self.tau_s)
            has = j >= 0
            has[has] &= d.drogue[j[has]] == flag        # same drogue type, as in training
            cand = alive & ~exited
            strands = cand & ~has & (self.strand_t[s0] >= 0) & (self.strand_t[s0] < d.t[s0] + k * self.tau_s)
            state[strands, k] = self.n_cells + STRANDED
            exited |= strands
            alive &= ~(cand & ~has & ~strands)         # lost: no matching observation
            ok = cand & has
            jj = np.where(ok, j, 0)
            outside = ok & (self.cell_flat[jj] < 0)
            state[outside, k] = self.n_cells + self.exit_side(d.lon[jj[outside]], d.lat[jj[outside]])
            exited |= outside
            inside = ok & ~outside
            state[inside, k] = self.state_of_cell[self.cell_flat[jj[inside]]]
            pos[inside, k] = np.c_[d.lon[jj[inside]], d.lat[jj[inside]]]
        return s0, state, pos

    # ── Training matrix and baselines for one fold ───────────────────────────

    def train_matrix(self, flag, train_traj):
        """The fold's matrix from its training drifters' moves, and the number of moves."""
        d, mv = self.d, self.moves
        keep = (np.isin(d.traj_idx[mv.start], train_traj)
                & same_drogue(d.drogue, mv.start, mv.end, mv.stranded, flag))
        s, e = mv.start[keep], mv.end[keep]
        fs, ts = transition_states(self.cell_flat, self.cells, s, e, d.lon, d.lat, self.box, mv.stranded[keep])
        C = np.zeros((self.n_states, self.n_states))
        np.add.at(C, (fs, ts), 1)
        P, _ = row_normalise(C, self.cells, self.grid.n_lon)
        return P, len(s)

    def train_fold(self, flag, train_traj):
        """The fold's matrix, the mean current per cell from its training observations,
        and the number of training moves."""
        d = self.d
        P, n_moves = self.train_matrix(flag, train_traj)
        ve, vn = d.ve, d.vn
        m = self.in_R & (d.drogue == flag) & np.isin(d.traj_idx, train_traj) & np.isfinite(ve) & np.isfinite(vn)
        st = self.state_of_cell[self.cell_flat[m]]
        cnt = np.bincount(st, minlength=self.n_cells)
        u = np.bincount(st, ve[m], self.n_cells) / np.maximum(cnt, 1)
        v = np.bincount(st, vn[m], self.n_cells) / np.maximum(cnt, 1)
        return P, u, v, n_moves

    def advect(self, lo, la, u, v, k_steps):
        """Deterministic forecast: move by the cell's mean current for each step;
        returns final positions and the exit state (−1 while still in R)."""
        lo, la = lo.copy(), la.copy()
        ex = np.full(len(lo), -1)
        for _ in range(k_steps):
            live = ex < 0
            c = np.where(live, self.grid.cell_of(lo, la), -1)
            stt = np.where(c >= 0, self.state_of_cell[np.maximum(c, 0)], -1)
            has = stt >= 0
            du = np.where(has, u[np.maximum(stt, 0)], 0.0) * self.tau_s
            dv = np.where(has, v[np.maximum(stt, 0)], 0.0) * self.tau_s
            lo = np.where(live, lo + du / (111_320.0 * np.cos(np.radians(la))), lo)
            la = np.where(live, la + dv / 110_574.0, la)
            out = live & ~in_box(lo, la, self.box)
            ex[out] = self.n_cells + self.exit_side(lo[out], la[out])
        return lo, la, ex

    # ── Scores at one horizon ────────────────────────────────────────────────

    def score(self, Pk, k, state, pos, tr_state, current):
        """Scores of the test samples at step k against their actual outcome: (number
        scored, scores, predicted and actual exit per sample)."""
        u, v = current
        n_cells, n_states = self.n_cells, self.n_states
        ok = (state[:, 0] >= 0) & (state[:, k] >= 0)
        st0, stk = state[ok, 0], state[ok, k]
        pred = Pk[st0]                                      # (n, n_states)
        p_true = pred[np.arange(len(stk)), stk]
        # Climatology baseline: training-set frequency of each outcome state at k
        clim = np.bincount(tr_state[tr_state[:, k] >= 0, k], minlength=n_states).astype(float)
        clim /= clim.sum()
        p_exit = pred[:, n_cells:n_cells + STRANDED].sum(axis=1)
        actual_exit = (stk >= n_cells) & (stk < n_cells + STRANDED)
        actual_strand = stk == n_cells + STRANDED
        # Position error for samples still in R at k
        stay = stk < n_cells
        mass = pred[stay, :n_cells]
        w = mass / np.maximum(mass.sum(axis=1, keepdims=True), 1e-12)
        com_lon, com_lat = w @ self.cell_lon, w @ self.cell_lat
        act = pos[ok][stay, k]
        start = pos[ok][stay, 0]
        adv_lo, adv_la, adv_ex = self.advect(start[:, 0], start[:, 1], u, v, k)
        adv_ok = adv_ex < 0
        c0 = st0[stay]                                       # start state (cell)
        cen_lo, cen_la = self.cell_lon[c0], self.cell_lat[c0]
        advc_lo, advc_la, advc_ex = self.advect(cen_lo, cen_la, u, v, k)
        advc_ok = advc_ex < 0
        scores = {
            "log_markov": np.mean(np.log(np.maximum(p_true, LOG_FLOOR))),
            "log_clim": np.mean(np.log(np.maximum(clim[stk], LOG_FLOOR))),
            "err_markov_km": np.median(haversine_km(com_lon, com_lat, act[:, 0], act[:, 1])),
            "err_persist_km": np.median(haversine_km(start[:, 0], start[:, 1], act[:, 0], act[:, 1])),
            "err_advect_km": np.median(haversine_km(adv_lo[adv_ok], adv_la[adv_ok], act[adv_ok, 0], act[adv_ok, 1])),
            "err_persist_centre_km": np.median(haversine_km(cen_lo, cen_la, act[:, 0], act[:, 1])),
            "err_advect_centre_km": np.median(haversine_km(advc_lo[advc_ok], advc_la[advc_ok],
                                                           act[advc_ok, 0], act[advc_ok, 1])),
            "exit_pred": p_exit.mean(), "exit_obs": actual_exit.mean(),
            "brier_exit": np.mean((p_exit - actual_exit) ** 2),
            "strand_pred": pred[:, n_cells + STRANDED].mean(), "strand_obs": actual_strand.mean(),
        }
        return int(ok.sum()), scores, p_exit, actual_exit


# ── Cross-validate ───────────────────────────────────────────────────────────

def run_fold(ho, name, flag, f, folds_traj):
    """Scores of one fold at every horizon, and its exit predictions at each."""
    trajs, test_traj = folds_traj
    train_traj = np.setdiff1d(trajs, test_traj)
    P, u, v, n_train = ho.train_fold(flag, train_traj)
    _, state, pos = ho.test_samples(flag, test_traj)
    _, tr_state, _ = ho.test_samples(flag, train_traj)
    records, exit_records = [], []
    Pk = np.eye(ho.n_states)
    for k in ho.horizon_steps:
        Pk = Pk @ P
        n_test, scores, p_exit, exited = ho.score(Pk, k, state, pos, tr_state, (u, v))
        exit_records.append(pd.DataFrame({"type": name, "k": k, "p_exit": p_exit, "exited": exited}))
        records.append({"type": name, "fold": f, "k": k, "days": k * ho.tau, "n_train_pairs": n_train,
                        "n_test": n_test, "n_test_drifters": len(test_traj), **scores})
    print(f"{name:<9} fold {f + 1}/{K_FOLDS}: {len(test_traj)} held-out drifters, "
          f"{int((state[:, 0] >= 0).sum()):,} test starts")
    return records, exit_records


def cross_validate(ho, rng):
    """K_FOLDS folds of whole drifters per drogue type."""
    d = ho.d
    records, exit_records = [], []
    for name, flag in DROGUE_TYPES.items():
        trajs = np.unique(d.traj_idx[ho.in_R & (d.drogue == flag)])
        for f, test_traj in enumerate(np.array_split(rng.permutation(trajs), K_FOLDS)):
            r, e = run_fold(ho, name, flag, f, (trajs, test_traj))
            records += r
            exit_records += e
    return pd.DataFrame(records), pd.concat(exit_records)


def summarise(res):
    """Mean over folds per drogue type and horizon."""
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
    return summary


def print_summary(summary, args):
    print(f"\nHeld-out validation, {K_FOLDS}-fold by drifter ({args.res:g}°, τ = {args.tau:g} d); mean over folds")
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


# ── Figure: position error by horizon, exit calibration ─────────────────────

def plot_errors(ax, name, summary, res):
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


def plot_calibration(ax, ex, last_step, tau):
    ax.plot([0, 1], [0, 1], color="#9e9e9e", lw=1, ls="--", label="Perfect calibration")
    for name in DROGUE_TYPES:
        e = ex[(ex.type == name) & (ex.k == last_step)]
        b = pd.cut(e.p_exit, EXIT_BINS, include_lowest=True)
        cal = e.groupby(b, observed=True).agg(p=("p_exit", "mean"), o=("exited", "mean"), n=("exited", "size"))
        cal = cal[cal.n >= 30]
        ax.plot(cal.p, cal.o, marker="o", lw=2, color=DROGUE_COLORS[name], label=name.capitalize())
    ax.set_xlabel(f"Predicted probability of having left R by day {last_step * tau:g}")
    ax.set_ylabel("Observed share that left")
    ax.set_title("Exit calibration (bins with ≥ 30 test starts)", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def plot_validation(summary, res, ex, ho, args, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, name in zip(axes[:2], DROGUE_TYPES):
        plot_errors(ax, name, summary, res)
    plot_calibration(axes[2], ex, max(ho.horizon_steps), args.tau)
    fig.suptitle(f"Held-out validation — {args.res:g}° grid, τ = {args.tau:g} d, box R", fontsize=12, y=1.04)
    save_figure(path)


def main(argv=None):
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    tag = setting_tag(args.res, args.tau, args.box)
    out_folds = f"data/validation_folds_{tag}.csv"
    out_summary = f"data/validation_summary_{tag}.csv"
    out_png = f"figures/{region_prefix(args.data)}_validation_{tag}.png"

    ho = HeldOut(Drifters(args.data), args.box, args.res, args.tau)
    res, ex = cross_validate(ho, np.random.default_rng(SEED))
    summary = summarise(res)
    res.to_csv(out_folds, index=False)
    summary.to_csv(out_summary, index=False)
    print_summary(summary, args)
    print(f"Saved → {out_folds}, {out_summary}")

    plot_validation(summary, res, ex, ho, args, out_png)
    print(f"Figure saved → {out_png}")


if __name__ == "__main__":
    main()
