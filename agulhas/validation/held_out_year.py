"""
Agulhas Region — Held-out Validation to One Year
Tests the matrices' long-range fates against drifters they have never seen, out
to 364 days (104 steps at τ = 3.5 d). Folds and test starts are those of held_out:
whole drifters split into K_FOLDS folds per drogue type, each fold's matrix
built from the others, test starts at 00 UTC in R.

The outcome is a fate: still in R, exited W, E, S or N, or stranded, followed at
whole steps of τ as the chain moves, with exits and stranding absorbing. A step
with no observation (a gap in the record) leaves the fate open and the drifter
is followed at later steps. Its fate is censored when its record ends without
grounding, or, for drogued starts, when the drogue is lost.

Few records last a year, so censoring has to be handled: keeping only starts
whose fate is known over-counts exits, which are known as soon as they happen,
against staying in R, which needs a record that lasts the whole horizon.
  • observed  — Aalen–Johansen estimate of each fate's share, the standard
                estimator for competing outcomes under censoring. It assumes a
                record's end says nothing about the fate it would have had
  • checks    — the same estimate within start-longitude bands and within start
                eras, which relaxes that to "within each band / era", and the
                biased known-only share for comparison
  • predicted — mean over the test starts of the fold matrix's P^k row
  • intervals — 95% from resampling whole held-out drifters, with the fold
                matrices held fixed, so training noise is not included
  • skill     — multi-class Brier score against the training folds' mean fate
                (a prediction that ignores the start cell), with
                inverse-probability-of-censoring weights
  • calibration — predicted against observed within deciles of each fate's
                predicted probability, and within start-longitude bands
  • release cells — the same for test starts in each release point's own cell
                from shipping_overlay, where the number of drifters allows it

There is no direct test of the oil matrix, since no drifter moves as it does.

    python -m agulhas held_out_year --res 1 --tau 3.5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.sparse import csr_matrix

from agulhas.config import DROGUE_TYPES, parse_args, region_prefix, release_points_path, setting_tag
from agulhas.drifters import Drifters
from agulhas.plotting import save_figure
from agulhas.transport import EXIT_LABELS, STRANDED, last_obs_index
from agulhas.validation.held_out import HeldOut

K_FOLDS = 5
SEED = 0                             # same folds as held_out
BOOT_SEED = 1
N_BOOT = 1000
REPORT_DAYS = [28, 91, 182, 273, 364]
INTERVAL_DAYS = [182, 364]           # calibration and band intervals are computed at these horizons
CALIB_BINS = 10
MIN_BIN_DRIFTERS = 10                # calibration bins need this many drifters at risk at the horizon
BAND_WIDTH = 10.0                    # ° of start longitude for the censoring check
ERA_SPLIT = np.datetime64("2010-01-01")
FATES = ["in_R"] + [f"exit_{x}" for x in EXIT_LABELS[:STRANDED]] + ["stranded"]
SHOWN = {"in_R": ("Still in R", "#2a78d6"), "exit_E": ("Exited east", "#eb6834"),
         "exit_W": ("Exited west", "#1b9e77"), "stranded": ("Stranded", "#7b3294")}
CALIB_PANEL_DAYS = {"drogued": 182, "undrogued": 364}   # too few drogued drifters last a year to bin
TABLES = ["summary", "support", "calibration", "bands", "points"]
NO_INTERVAL = (np.nan,) * 4


# ── Estimators under censoring ───────────────────────────────────────────────

def risk_counts(d, n_d, event, cause, known, K):
    """Per drifter: starts at risk at each step and events by cause at each step."""
    last = np.where(event > 0, event, known)
    at_risk = np.zeros((n_d, K + 2))
    np.add.at(at_risk, (d, 1), 1)
    np.add.at(at_risk, (d, last + 1), -1)
    at_risk = np.cumsum(at_risk, axis=1)[:, :K + 1]
    ev = np.zeros((n_d, K + 1, STRANDED + 1))
    hit = event > 0
    np.add.at(ev, (d[hit], event[hit], cause[hit]), 1)
    return at_risk, ev


def aalen_johansen(at_risk, ev):
    """Fate shares per step from summed counts, each (…, K+1) and (…, K+1, causes).
    Returns (…, K+1, 1 + causes): in R first, then each absorbing state."""
    K = at_risk.shape[-1] - 1
    shape = at_risk.shape[:-1]
    out = np.zeros(shape + (K + 1, STRANDED + 2))
    surv = np.ones(shape)
    out[..., 0, 0] = 1.0
    for k in range(1, K + 1):
        n = np.maximum(at_risk[..., k], 1e-12)[..., None]
        out[..., k, 1:] = out[..., k - 1, 1:] + surv[..., None] * ev[..., k, :] / n
        surv = surv * (1 - ev[..., k, :].sum(axis=-1) / n[..., 0])
        out[..., k, 0] = surv
    return out


def censor_survival(event, known, K):
    """G[k] = P(the record is still followed after step k), Kaplan–Meier, for the
    inverse-probability-of-censoring weights."""
    no_event = event == 0
    last = np.where(event > 0, event, known)
    n = np.array([(last >= k).sum() for k in range(K + 1)], float)
    n_ev = np.bincount(event[event > 0], minlength=K + 1)
    cens = np.bincount(known[no_event], minlength=K + 1).astype(float)
    cens[K] = 0                                 # still followed at the last step
    return np.cumprod(1 - cens / np.maximum(n - n_ev, 1))


# ── Fate of every test start ─────────────────────────────────────────────────

class YearHeldOut(HeldOut):
    """held_out's subset, states and fold matrices, with each start followed to a year."""

    def __init__(self, drifters, box, res, tau):
        super().__init__(drifters, box, res, tau)
        self.K = int(round(max(REPORT_DAYS) / tau))
        self.report_steps = [int(round(d / tau)) for d in REPORT_DAYS]
        self.record_end = drifters.t[last_obs_index(drifters.traj_idx)]

    def fates(self, flag):
        """Daily starts in R of one drogue type, with each start's event step (0 =
        no event seen), event cause (index into the absorbing states) and last step
        at which it was known to be still in R."""
        d = self.d
        s0 = np.where(self.in_R & (d.drogue == flag) & (d.t % 86400 == 0))[0]
        event = np.zeros(len(s0), int)
        cause = np.full(len(s0), -1)
        known = np.zeros(len(s0), int)
        live = np.ones(len(s0), bool)
        n_gap = 0
        for k in range(1, self.K + 1):
            tk = d.t[s0] + k * self.tau_s
            j = self.obs_at(s0, k * self.tau_s)
            has = j >= 0
            same = has.copy()
            same[has] = d.drogue[j[has]] == flag
            live &= ~(has & ~same)                     # drogue lost: censored
            strands = live & ~has & (self.strand_t[s0] >= 0) & (self.strand_t[s0] < tk)
            event[strands], cause[strands] = k, STRANDED
            live &= ~strands
            live &= ~(~has & (tk > self.record_end[s0]))    # record over, no grounding: censored
            n_gap += int((live & ~has).sum())          # a gap: stays open
            ok = live & same
            jj = j[ok]
            outside = self.cell_flat[jj] < 0
            idx = np.where(ok)[0]
            event[idx[outside]] = k
            cause[idx[outside]] = self.exit_side(d.lon[jj[outside]], d.lat[jj[outside]])
            known[idx[~outside]] = k
            live[idx[outside]] = False
        return s0, event, cause, known, n_gap

    def release_state(self, lon, lat):
        """Ocean state of the cell holding a release point, or -1."""
        g = self.grid
        return self.state_of_cell[int((lat - g.lat_min) // g.res) * g.n_lon + int((lon - g.lon_min) // g.res)]


class TypeFates:
    """One drogue type's held-out starts: their fates, the fold matrices' predictions,
    and the whole-drifter resamples every table draws on. `drifter` is each start's
    drifter, numbered within this type."""

    def __init__(self, h, name, flag, rng):
        self.h, self.name, self.flag, self.K = h, name, flag, h.K
        d = h.d
        self.trajs = np.unique(d.traj_idx[h.in_R & (d.drogue == flag)])
        self.n_d = len(self.trajs)
        self.fold_trajs = np.array_split(rng.permutation(self.trajs), K_FOLDS)
        self.s0, self.event, self.cause, self.known, self.n_gap = h.fates(flag)
        self.drifter = np.searchsorted(self.trajs, d.traj_idx[self.s0])
        self.st0 = h.state_of_cell[h.cell_flat[self.s0]]
        self.fold_of = np.zeros(self.n_d, int)
        for f, test_traj in enumerate(self.fold_trajs):
            self.fold_of[np.searchsorted(self.trajs, test_traj)] = f
        self.at_risk, self.ev = risk_counts(self.drifter, self.n_d, self.event, self.cause, self.known, self.K)
        self.n_starts_d = np.bincount(self.drifter, minlength=self.n_d).astype(float)
        self.last = np.where(self.event > 0, self.event, self.known)

    def risk(self, m):
        """risk_counts of the starts in mask m."""
        return risk_counts(self.drifter[m], self.n_d, self.event[m], self.cause[m], self.known[m], self.K)

    def observed(self, m):
        """Aalen–Johansen fate shares of the starts in mask m."""
        a, e = self.risk(m)
        return aalen_johansen(a.sum(axis=0), e.sum(axis=0))

    # ── Predictions of the fold matrices ─────────────────────────────────────

    def predict(self):
        """Each drifter's predicted fate mass per step, and each start's prediction
        and training mean fate at the report steps."""
        h, K = self.h, self.K
        self.pred_mass = np.zeros((self.n_d, K + 1, len(FATES)))
        self.pred_rep = {k: np.zeros((len(self.s0), len(FATES)), np.float32) for k in h.report_steps}
        self.clim_rep = {k: np.zeros((len(self.s0), len(FATES)), np.float32) for k in h.report_steps}
        for f in range(K_FOLDS):
            P, _ = h.train_matrix(self.flag, np.setdiff1d(self.trajs, self.fold_trajs[f]))
            self.predict_fold(f, P)
        self.pred_share = self.pred_mass.sum(axis=0) / self.n_starts_d.sum()
        self.obs = aalen_johansen(self.at_risk.sum(axis=0), self.ev.sum(axis=0))

    def predict_fold(self, f, P):
        h, K = self.h, self.K
        test = self.fold_of[self.drifter] == f
        N = csr_matrix((np.ones(test.sum()), (self.drifter[test], self.st0[test])), shape=(self.n_d, h.n_states))
        train = self.fold_of != f
        clim = aalen_johansen(self.at_risk[train].sum(axis=0), self.ev[train].sum(axis=0))
        V = np.zeros((h.n_states, len(FATES)))      # V[i, o] = P(state i has fate o after k steps)
        V[:h.n_cells, 0] = 1.0
        V[h.n_cells + np.arange(len(EXIT_LABELS)), 1 + np.arange(len(EXIT_LABELS))] = 1.0
        self.pred_mass[:, 0] += N @ V
        for k in range(1, K + 1):
            V = P @ V
            self.pred_mass[:, k] += N @ V
            if k in self.pred_rep:
                self.pred_rep[k][test] = V[self.st0[test]]
                self.clim_rep[k][test] = clim[k]

    # ── Intervals and censoring checks ───────────────────────────────────────

    def resample(self, boot_rng):
        """Whole-drifter bootstrap of the held-out set, fold matrices fixed."""
        K, n_d = self.K, self.n_d
        self.W = W = boot_rng.multinomial(n_d, np.full(n_d, 1 / n_d), size=N_BOOT).astype(float)
        b_risk = W @ self.at_risk
        b_ev = (W @ self.ev.reshape(n_d, -1)).reshape(N_BOOT, K + 1, STRANDED + 1)
        b_obs = aalen_johansen(b_risk, b_ev)
        b_n = (W @ self.n_starts_d)[:, None]
        b_pred = (W @ self.pred_mass.reshape(n_d, -1)).reshape(N_BOOT, K + 1, len(FATES)) / b_n[:, :, None]
        self.obs_lo, self.obs_hi = np.percentile(b_obs, [2.5, 97.5], axis=0)
        self.diff_lo, self.diff_hi = np.percentile(b_pred - b_obs, [2.5, 97.5], axis=0)

    def check_censoring(self):
        """Spread over folds, the estimate within start-longitude bands and eras, and
        the censoring survival for the Brier weights."""
        self.fold_diff = np.zeros((K_FOLDS, self.K + 1, len(FATES)))
        for f in range(K_FOLDS):
            m = self.fold_of == f
            self.fold_diff[f] = (self.pred_mass[m].sum(axis=0) / self.n_starts_d[m].sum()
                                 - aalen_johansen(self.at_risk[m].sum(axis=0), self.ev[m].sum(axis=0)))
        h = self.h
        self.band_of = ((h.d.lon[self.s0] - h.grid.lon_min) // BAND_WIDTH).astype(int)
        era_of = (h.d.ds["time"].values[self.s0] >= ERA_SPLIT).astype(int)
        self.by_band, self.by_era = self.stratified(self.band_of), self.stratified(era_of)
        self.G = censor_survival(self.event, self.known, self.K)

    def stratified(self, labels):
        """Fate shares estimated within each label, weighted by its share of starts."""
        total = np.zeros_like(self.obs)
        for lab in np.unique(labels):
            m = labels == lab
            total += m.mean() * self.observed(m)
        return total

    def group_interval(self, m, values, k, o):
        """95% interval of the observed share and of predicted − observed for
        the starts in mask m, resampling whole drifters."""
        d, n_d = self.drifter, self.n_d
        u = np.unique(d[m])
        a, e = self.risk(m)
        Wu = self.W[:, u]
        b = aalen_johansen(Wu @ a[u], (Wu @ e[u].reshape(len(u), -1)).reshape(N_BOOT, self.K + 1, STRANDED + 1))[:, k, o]
        n = np.maximum(Wu @ np.bincount(d[m], minlength=n_d)[u].astype(float), 1e-12)
        bp = (Wu @ np.bincount(d[m], values, n_d)[u]) / n
        return (*np.percentile(b, [2.5, 97.5]), *np.percentile(bp - b, [2.5, 97.5]))

    def drifters_at(self, m, k):
        """Drifters among the starts in mask m still followed at step k."""
        return int(np.unique(self.drifter[m & (self.last >= k)]).size)

    # ── Table rows at one horizon ────────────────────────────────────────────

    def rows(self, days, k, release):
        """Rows of every table at one report horizon."""
        has_event = (self.event > 0) & (self.event <= k)
        known_k = has_event | (self.last >= k)
        fate = np.where(has_event, self.cause + 1, 0)
        return {"summary": self.summary_rows(days, k, known_k, fate),
                "support": [self.support_row(days, k, has_event, known_k, fate)],
                "calibration": self.calibration_rows(days, k),
                "bands": self.band_rows(days, k),
                "points": self.point_rows(days, k, release)}

    def summary_rows(self, days, k, known_k, fate):
        naive = np.bincount(fate[known_k], minlength=len(FATES)) / known_k.sum()
        return [{"type": self.name, "days": days, "fate": fate_name,
                 "predicted": self.pred_share[k, o], "observed": self.obs[k, o],
                 "observed_lo": self.obs_lo[k, o], "observed_hi": self.obs_hi[k, o],
                 "difference": self.pred_share[k, o] - self.obs[k, o],
                 "difference_lo": self.diff_lo[k, o], "difference_hi": self.diff_hi[k, o],
                 "fold_difference_min": self.fold_diff[:, k, o].min(),
                 "fold_difference_max": self.fold_diff[:, k, o].max(),
                 "observed_by_band": self.by_band[k, o], "observed_by_era": self.by_era[k, o],
                 "observed_known_only": naive[o]}
                for o, fate_name in enumerate(FATES)]

    def support_row(self, days, k, has_event, known_k, fate):
        """Support and the Brier score with inverse-probability-of-censoring weights,
        each drifter's sum kept so that the skill can be resampled by drifter."""
        d, n_d = self.drifter, self.n_d
        when = np.where(has_event, self.event, k)
        w = np.where(known_k, 1 / np.maximum(np.r_[1.0, self.G][when], 1e-6), 0.0)
        y = np.eye(len(FATES))[fate]
        bs_model = w * ((self.pred_rep[k] - y) ** 2).sum(axis=1)
        bs_clim = w * ((self.clim_rep[k] - y) ** 2).sum(axis=1)
        d_model = np.bincount(d, bs_model, n_d)
        d_clim = np.bincount(d, bs_clim, n_d)
        b_skill = 1 - (self.W @ d_model) / (self.W @ d_clim)
        at_k = (self.last >= k) & ~has_event
        return {"type": self.name, "days": days, "starts": len(self.s0), "drifters": n_d,
                "starts_followed_in_R": int(at_k.sum()), "drifters_followed_in_R": int(np.unique(d[at_k]).size),
                "share_fate_known": known_k.mean(), "gap_steps": self.n_gap,
                "ipcw_weight_mean": w.mean(),
                "brier_markov": bs_model.mean(), "brier_climatology": bs_clim.mean(),
                "brier_skill": 1 - d_model.sum() / d_clim.sum(),
                "brier_skill_lo": np.percentile(b_skill, 2.5), "brier_skill_hi": np.percentile(b_skill, 97.5)}

    def calibration_rows(self, days, k):
        """Predicted against observed within deciles of each shown fate's predicted probability."""
        rows = []
        for o, fate_name in enumerate(FATES):
            if fate_name not in SHOWN:
                continue
            p = self.pred_rep[k][:, o]
            edges = np.unique(np.quantile(p, np.linspace(0, 1, CALIB_BINS + 1)))
            if len(edges) < 2:
                continue
            bin_of = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
            for b in np.unique(bin_of):
                m = bin_of == b
                lims = self.group_interval(m, p[m], k, o) if days in INTERVAL_DAYS else NO_INTERVAL
                rows.append({"type": self.name, "days": days, "fate": fate_name, "bin": int(b),
                             "predicted": p[m].mean(), "observed": self.observed(m)[k, o],
                             "observed_lo": lims[0], "observed_hi": lims[1],
                             "difference_lo": lims[2], "difference_hi": lims[3],
                             "starts": int(m.sum()), "drifters_at_horizon": self.drifters_at(m, k)})
        return rows

    def band_rows(self, days, k):
        """Predicted against observed within each start-longitude band."""
        g, rows = self.h.grid, []
        for b in np.unique(self.band_of):
            m = self.band_of == b
            ob = self.observed(m)[k]
            band = f"{g.lon_min + b * BAND_WIDTH:g}–{min(g.lon_min + (b + 1) * BAND_WIDTH, g.lon_max):g}°E"
            for o, fate_name in enumerate(FATES):
                lims = (self.group_interval(m, self.pred_rep[k][m, o], k, o)
                        if days in INTERVAL_DAYS and fate_name in SHOWN else NO_INTERVAL)
                rows.append({"type": self.name, "days": days, "fate": fate_name, "band": band,
                             "predicted": self.pred_rep[k][m, o].mean(), "observed": ob[o],
                             "observed_lo": lims[0], "observed_hi": lims[1],
                             "difference_lo": lims[2], "difference_hi": lims[3],
                             "starts": int(m.sum()), "drifters": int(np.unique(self.drifter[m]).size),
                             "drifters_at_horizon": self.drifters_at(m, k)})
        return rows

    def point_rows(self, days, k, release):
        """Predicted against observed for the starts in each release point's own cell."""
        rows = []
        for _, pt in release.iterrows():
            m = self.st0 == self.h.release_state(pt.lon, pt.lat)
            if not m.any():
                continue
            ob = self.observed(m)[k]
            rows += [{"type": self.name, "days": days, "point": pt["name"], "fate": fate_name,
                      "predicted": self.pred_rep[k][m, o].mean(), "observed": ob[o],
                      "starts": int(m.sum()), "drifters": int(np.unique(self.drifter[m]).size),
                      "drifters_at_horizon": self.drifters_at(m, k)}
                     for o, fate_name in enumerate(FATES) if fate_name in SHOWN]
        return rows


# ── Cross-validate ───────────────────────────────────────────────────────────

def cross_validate(h, release):
    """Every table, and the predicted and observed fate curves of each drogue type."""
    rng = np.random.default_rng(SEED)
    boot_rng = np.random.default_rng(BOOT_SEED)
    tables = {t: [] for t in TABLES}
    curves = {}
    for name, flag in DROGUE_TYPES.items():
        tf = TypeFates(h, name, flag, rng)
        tf.predict()
        curves[name] = (tf.pred_share, tf.obs)
        tf.resample(boot_rng)
        tf.check_censoring()
        for days, k in zip(REPORT_DAYS, h.report_steps):
            for table, rows in tf.rows(days, k, release).items():
                tables[table] += rows
        print(f"{name:<9}: {tf.n_d:,} drifters, {len(tf.s0):,} test starts, {tf.n_gap:,} steps open across record gaps")
    return {t: pd.DataFrame(rows) for t, rows in tables.items()}, curves


# ── Report ───────────────────────────────────────────────────────────────────

def pct(x):
    return f"{100 * x:.1f}%"


def print_fates(summary, args):
    print(f"\nHeld-out fates, {K_FOLDS}-fold by drifter ({args.res:g}°, τ = {args.tau:g} d); predicted − observed with 95% interval")
    for name in DROGUE_TYPES:
        print(f"\n{name}")
        print(f"{'days':>5} {'fate':<10}{'pred':>8}{'obs':>8}{'95%':>16}{'pred−obs':>10}{'95%':>18}{'folds':>18}{'by band':>9}{'by era':>8}{'known only':>12}")
        for _, r in summary[(summary.type == name) & summary.fate.isin(SHOWN)].iterrows():
            print(f"{r.days:>5} {r.fate:<10}{pct(r.predicted):>8}{pct(r.observed):>8}"
                  f"{pct(r.observed_lo) + '–' + pct(r.observed_hi):>16}{100 * r.difference:>+10.1f}"
                  f"{f'{100 * r.difference_lo:+.1f} to {100 * r.difference_hi:+.1f}':>18}"
                  f"{f'{100 * r.fold_difference_min:+.1f} to {100 * r.fold_difference_max:+.1f}':>18}"
                  f"{pct(r.observed_by_band):>9}{pct(r.observed_by_era):>8}{pct(r.observed_known_only):>12}")


def print_support(support):
    print("\nSupport and Brier skill against the training mean fate (above 0 = the start cell adds skill)")
    print(support[["type", "days", "starts_followed_in_R", "drifters_followed_in_R", "share_fate_known",
                   "ipcw_weight_mean", "brier_markov", "brier_climatology", "brier_skill", "brier_skill_lo",
                   "brier_skill_hi"]].round(3).to_string(index=False))


def print_calibration(calibration, bands):
    for name in DROGUE_TYPES:
        print(f"\n{name}: calibration of still in R (predicted, observed, 95% interval of predicted − observed)")
        for days in INTERVAL_DAYS:
            c = calibration[(calibration.type == name) & (calibration.fate == "in_R") & (calibration.days == days)]
            print(f"  day {days}: " + "; ".join(
                f"{100 * r.predicted:.0f}/{100 * r.observed:.0f} ({100 * r.difference_lo:+.0f} to {100 * r.difference_hi:+.0f}, n={r.drifters_at_horizon})"
                for _, r in c.iterrows()))
        print(f"{name}: by start longitude at day {max(REPORT_DAYS)}")
        b = bands[(bands.type == name) & (bands.days == max(REPORT_DAYS)) & bands.fate.isin(SHOWN)]
        for _, r in b.iterrows():
            print(f"  {r.band:<9} {r.fate:<9} predicted {pct(r.predicted):>6} observed {pct(r.observed):>6} "
                  f"difference {100 * r.difference_lo:+.1f} to {100 * r.difference_hi:+.1f}; {r.drifters_at_horizon} drifters at the horizon")


def print_release_cells(points):
    print(f"\nDrifters still followed in R at day {max(REPORT_DAYS)}, by release cell (drogued / undrogued):")
    at = points[(points.days == max(REPORT_DAYS)) & (points.fate == "in_R")].pivot_table(
        index="point", columns="type", values="drifters_at_horizon", sort=False).fillna(0).astype(int)
    print(at.to_string())


# ── Figure: fates over the year, calibration at one year ─────────────────────

def plot_fates(ax, name, curves, summary, h):
    days_axis = np.arange(h.K + 1) * h.tau
    pred, ob = curves[name]
    for fate_name, (label, color) in SHOWN.items():
        o = FATES.index(fate_name)
        ax.plot(days_axis, ob[:, o], color=color, lw=2, label=label)
        ax.plot(days_axis, pred[:, o], color=color, lw=2, ls=(0, (4, 2)))
        rows = summary[(summary.type == name) & (summary.fate == fate_name)]
        ax.errorbar(rows.days, rows.observed, yerr=[rows.observed - rows.observed_lo, rows.observed_hi - rows.observed],
                    fmt="none", ecolor=color, elinewidth=1.2, capsize=3)
    ax.plot([], [], color="#555555", lw=2, label="Observed (held out), 95% interval")
    ax.plot([], [], color="#555555", lw=2, ls=(0, (4, 2)), label="Predicted by the matrix")
    ax.set_xlim(0, max(REPORT_DAYS) + 6)
    ax.set_ylim(0, 1)
    ax.set_xticks(REPORT_DAYS)
    ax.set_xlabel("Days after release")
    ax.set_ylabel("Share of test starts")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(alpha=0.25)
    ax.set_title(f"{name.capitalize()}: fate of held-out drifters", fontsize=11)
    ax.legend(fontsize=8, loc="upper right", frameon=False)


def plot_calibration(ax, name, calibration):
    panel_days = CALIB_PANEL_DAYS[name]
    ax.plot([0, 1], [0, 1], color="#9e9e9e", lw=1, ls="--", label="Perfect calibration")
    for fate_name, (label, color) in SHOWN.items():
        c = calibration[(calibration.type == name) & (calibration.fate == fate_name)
                        & (calibration.days == panel_days)
                        & (calibration.drifters_at_horizon >= MIN_BIN_DRIFTERS)]
        ax.errorbar(c.predicted, c.observed, yerr=[c.observed - c.observed_lo, c.observed_hi - c.observed],
                    color=color, lw=2, marker="o", ms=7, mec="white", mew=1, elinewidth=1, capsize=2, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"Predicted probability at day {panel_days}")
    ax.set_ylabel("Observed share")
    ax.grid(alpha=0.25)
    ax.set_title(f"{name.capitalize()}: calibration at day {panel_days} (bins with ≥ {MIN_BIN_DRIFTERS} drifters at the horizon)",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper left", frameon=False)


def plot_validation(tables, curves, h, args, out):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    for ax, name in zip(axes[0], DROGUE_TYPES):
        plot_fates(ax, name, curves, tables["summary"], h)
    for ax, name in zip(axes[1], DROGUE_TYPES):
        plot_calibration(ax, name, tables["calibration"])
    fig.suptitle(f"Held-out validation to one year — {args.res:g}° grid, τ = {args.tau:g} d, box R", fontsize=13)
    fig.tight_layout()
    save_figure(out)


def main(argv=None):
    """Run this step; see the module docstring for what it does."""
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    tag = setting_tag(args.res, args.tau, args.box)
    paths = {t: f"data/validation_year_{t}_{tag}.csv" for t in TABLES}
    out_png = f"figures/{region_prefix(args.data)}_validation_year_{tag}.png"

    h = YearHeldOut(Drifters(args.data), args.box, args.res, args.tau)
    release = pd.read_csv(release_points_path(args.box))[["name", "lon", "lat"]]
    tables, curves = cross_validate(h, release)
    for t, path in paths.items():
        tables[t].to_csv(path, index=False)

    print_fates(tables["summary"], args)
    print_support(tables["support"])
    print_calibration(tables["calibration"], tables["bands"])
    print_release_cells(tables["points"])
    print(f"\nSaved → {', '.join(paths.values())}")

    plot_validation(tables, curves, h, args, out_png)
    print(f"Figure saved → {out_png}")


if __name__ == "__main__":
    main()
