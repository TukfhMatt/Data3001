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
import matplotlib
matplotlib.use("Agg")  # non-interactive, headless
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.sparse import csr_matrix
import xarray as xr

from agulhas.config import DROGUE_TYPES, box_tag, grid_shape, parse_args, region_prefix, release_points_path
from agulhas.transport import (EXIT_LABELS, STRANDED, last_obs_index, move_indices, row_normalise, same_drogue,
                               strand_times, transition_states)

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


def main(argv=None):
    """Run this step; see the module docstring for what it does."""
    args = parse_args(__doc__, grid=True, lag=True, argv=argv)
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = args.box
    RES, TAU = args.res, args.tau
    TAG = f"{f'{RES:g}'.replace('.', 'p')}deg_{f'{TAU:g}'.replace('.', 'p')}d_{box_tag(args.box)}"
    K = int(round(max(REPORT_DAYS) / TAU))
    REPORT_STEPS = [int(round(d / TAU)) for d in REPORT_DAYS]

    # ── 1. Load subset, grid, states ─────────────────────────────────────────────
    ds = xr.open_dataset(args.data)
    lon = ds["lon"].values.astype(float)
    lat = ds["lat"].values.astype(float)
    drogue = ds["drogue_status"].values.astype(bool)
    t = ds["time"].values.astype("datetime64[s]").astype(np.int64)
    traj_idx = np.repeat(np.arange(ds.sizes["traj"]), ds["rowsize"].values)

    n_lon, n_lat = grid_shape(args.box, RES)
    in_R = (lon >= LON_MIN) & (lon < LON_MAX) & (lat >= LAT_MIN) & (lat < LAT_MAX)
    cell_flat = np.where(in_R, ((lat - LAT_MIN) // RES).astype(int) * n_lon
                         + ((lon - LON_MIN) // RES).astype(int), -1)
    cells = np.unique(cell_flat[in_R])            # same ocean states as transition_matrix
    n_cells = len(cells)
    n_states = n_cells + len(EXIT_LABELS)
    state_of_cell = np.full(n_lat * n_lon, -1)
    state_of_cell[cells] = np.arange(n_cells)

    key = traj_idx.astype(np.int64) * 10**10 + t
    tau_s = int(round(TAU * 86400))
    record_end = t[last_obs_index(traj_idx)]
    strand_t = strand_times(ds, traj_idx, t, in_R)
    start_all, end_all, strand_all = move_indices(traj_idx, t, in_R, TAU, strand_t)   # all moves from R


    def exit_side(lo, la):
        return np.argmax(np.stack([LON_MIN - lo, lo - LON_MAX, LAT_MIN - la, la - LAT_MAX]), axis=0)


    def obs_at(idx, dt):
        """Index of the same drifter's observation dt seconds after idx, or -1."""
        target = key[idx] + dt
        j = np.minimum(np.searchsorted(key, target), len(key) - 1)
        return np.where(key[j] == target, j, -1)


    # ── 2. Fate of every test start ──────────────────────────────────────────────
    def fates(flag):
        """Daily starts in R of one drogue type, with each start's event step (0 =
        no event seen), event cause (index into the absorbing states) and last step
        at which it was known to be still in R."""
        s0 = np.where(in_R & (drogue == flag) & (t % 86400 == 0))[0]
        event = np.zeros(len(s0), int)
        cause = np.full(len(s0), -1)
        known = np.zeros(len(s0), int)
        live = np.ones(len(s0), bool)
        n_gap = 0
        for k in range(1, K + 1):
            tk = t[s0] + k * tau_s
            j = obs_at(s0, k * tau_s)
            has = j >= 0
            same = has.copy()
            same[has] = drogue[j[has]] == flag
            live &= ~(has & ~same)                     # drogue lost: censored
            strands = live & ~has & (strand_t[s0] >= 0) & (strand_t[s0] < tk)
            event[strands], cause[strands] = k, STRANDED
            live &= ~strands
            live &= ~(~has & (tk > record_end[s0]))    # record over, no grounding: censored
            n_gap += int((live & ~has).sum())          # a gap: stays open
            ok = live & same
            jj = j[ok]
            outside = cell_flat[jj] < 0
            idx = np.where(ok)[0]
            event[idx[outside]] = k
            cause[idx[outside]] = exit_side(lon[jj[outside]], lat[jj[outside]])
            known[idx[~outside]] = k
            live[idx[outside]] = False
        return s0, event, cause, known, n_gap


    def risk_counts(d, n_d, event, cause, known):
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


    def train_matrix(flag, train_traj):
        keep = (np.isin(traj_idx[start_all], train_traj)
                & same_drogue(drogue, start_all, end_all, strand_all, flag))
        fs, ts = transition_states(cell_flat, cells, start_all[keep], end_all[keep], lon, lat,
                                   args.box, strand_all[keep])
        C = np.zeros((n_states, n_states))
        np.add.at(C, (fs, ts), 1)
        return row_normalise(C, cells, n_lon)[0]


    def censor_survival(event, known):
        """G[k] = P(the record is still followed after step k), Kaplan–Meier, for the
        inverse-probability-of-censoring weights."""
        no_event = event == 0
        last = np.where(event > 0, event, known)
        n = np.array([(last >= k).sum() for k in range(K + 1)], float)
        n_ev = np.bincount(event[event > 0], minlength=K + 1)
        cens = np.bincount(known[no_event], minlength=K + 1).astype(float)
        cens[K] = 0                                 # still followed at the last step
        return np.cumprod(1 - cens / np.maximum(n - n_ev, 1))


    # ── 3. Cross-validate ────────────────────────────────────────────────────────
    release = pd.read_csv(release_points_path(args.box))[["name", "lon", "lat"]]
    rng = np.random.default_rng(SEED)
    boot_rng = np.random.default_rng(BOOT_SEED)
    summary, support, calibration, bands, points = [], [], [], [], []
    curves = {}
    for name, flag in DROGUE_TYPES.items():
        trajs = np.unique(traj_idx[in_R & (drogue == flag)])
        n_d = len(trajs)
        fold_trajs = np.array_split(rng.permutation(trajs), K_FOLDS)
        s0, event, cause, known, n_gap = fates(flag)
        d = np.searchsorted(trajs, traj_idx[s0])
        st0 = state_of_cell[cell_flat[s0]]
        fold_of = np.zeros(n_d, int)
        for f, test_traj in enumerate(fold_trajs):
            fold_of[np.searchsorted(trajs, test_traj)] = f
        at_risk, ev = risk_counts(d, n_d, event, cause, known)
        n_starts_d = np.bincount(d, minlength=n_d).astype(float)

        pred_mass = np.zeros((n_d, K + 1, len(FATES)))
        pred_rep = {k: np.zeros((len(s0), len(FATES)), np.float32) for k in REPORT_STEPS}
        clim_rep = {k: np.zeros((len(s0), len(FATES)), np.float32) for k in REPORT_STEPS}
        for f in range(K_FOLDS):
            P = train_matrix(flag, np.setdiff1d(trajs, fold_trajs[f]))
            test = fold_of[d] == f
            N = csr_matrix((np.ones(test.sum()), (d[test], st0[test])), shape=(n_d, n_states))
            clim = aalen_johansen(at_risk[fold_of != f].sum(axis=0), ev[fold_of != f].sum(axis=0))
            V = np.zeros((n_states, len(FATES)))      # V[i, o] = P(state i has fate o after k steps)
            V[:n_cells, 0] = 1.0
            V[n_cells + np.arange(len(EXIT_LABELS)), 1 + np.arange(len(EXIT_LABELS))] = 1.0
            pred_mass[:, 0] += N @ V
            for k in range(1, K + 1):
                V = P @ V
                pred_mass[:, k] += N @ V
                if k in pred_rep:
                    pred_rep[k][test] = V[st0[test]]
                    clim_rep[k][test] = clim[k]
        pred_mass_share = pred_mass.sum(axis=0) / n_starts_d.sum()

        obs = aalen_johansen(at_risk.sum(axis=0), ev.sum(axis=0))
        curves[name] = (pred_mass_share, obs)

        # Whole-drifter bootstrap of the held-out set, fold matrices fixed
        W = boot_rng.multinomial(n_d, np.full(n_d, 1 / n_d), size=N_BOOT).astype(float)
        b_risk = W @ at_risk
        b_ev = (W @ ev.reshape(n_d, -1)).reshape(N_BOOT, K + 1, STRANDED + 1)
        b_obs = aalen_johansen(b_risk, b_ev)
        b_n = (W @ n_starts_d)[:, None]
        b_pred = (W @ pred_mass.reshape(n_d, -1)).reshape(N_BOOT, K + 1, len(FATES)) / b_n[:, :, None]
        obs_lo, obs_hi = np.percentile(b_obs, [2.5, 97.5], axis=0)
        diff_lo, diff_hi = np.percentile(b_pred - b_obs, [2.5, 97.5], axis=0)

        # Spread over folds, and the censoring checks within bands and eras
        fold_diff = np.zeros((K_FOLDS, K + 1, len(FATES)))
        for f in range(K_FOLDS):
            m = fold_of == f
            fold_diff[f] = (pred_mass[m].sum(axis=0) / n_starts_d[m].sum()
                            - aalen_johansen(at_risk[m].sum(axis=0), ev[m].sum(axis=0)))

        def stratified(labels):
            """Fate shares estimated within each label, weighted by its share of starts."""
            total = np.zeros_like(obs)
            for lab in np.unique(labels):
                m = labels == lab
                a, e = risk_counts(d[m], n_d, event[m], cause[m], known[m])
                total += m.mean() * aalen_johansen(a.sum(axis=0), e.sum(axis=0))
            return total

        band_of = ((lon[s0] - LON_MIN) // BAND_WIDTH).astype(int)
        era_of = (ds["time"].values[s0] >= ERA_SPLIT).astype(int)
        by_band, by_era = stratified(band_of), stratified(era_of)

        def group_interval(m, values, k, o):
            """95% interval of the observed share and of predicted − observed for
            the starts in mask m, resampling whole drifters."""
            u = np.unique(d[m])
            a, e = risk_counts(d[m], n_d, event[m], cause[m], known[m])
            Wu = W[:, u]
            b = aalen_johansen(Wu @ a[u], (Wu @ e[u].reshape(len(u), -1)).reshape(N_BOOT, K + 1, STRANDED + 1))[:, k, o]
            n = np.maximum(Wu @ np.bincount(d[m], minlength=n_d)[u].astype(float), 1e-12)
            bp = (Wu @ np.bincount(d[m], values, n_d)[u]) / n
            return (*np.percentile(b, [2.5, 97.5]), *np.percentile(bp - b, [2.5, 97.5]))

        G = censor_survival(event, known)
        last = np.where(event > 0, event, known)
        for days, k in zip(REPORT_DAYS, REPORT_STEPS):
            has_event = (event > 0) & (event <= k)
            known_k = has_event | (last >= k)
            fate = np.where(has_event, cause + 1, 0)
            naive = np.bincount(fate[known_k], minlength=len(FATES)) / known_k.sum()
            for o, fate_name in enumerate(FATES):
                summary.append({
                    "type": name, "days": days, "fate": fate_name,
                    "predicted": pred_mass_share[k, o], "observed": obs[k, o],
                    "observed_lo": obs_lo[k, o], "observed_hi": obs_hi[k, o],
                    "difference": pred_mass_share[k, o] - obs[k, o],
                    "difference_lo": diff_lo[k, o], "difference_hi": diff_hi[k, o],
                    "fold_difference_min": fold_diff[:, k, o].min(),
                    "fold_difference_max": fold_diff[:, k, o].max(),
                    "observed_by_band": by_band[k, o], "observed_by_era": by_era[k, o],
                    "observed_known_only": naive[o]})

            # Brier score with inverse-probability-of-censoring weights, each drifter's
            # sum kept so that the skill can be resampled by drifter
            when = np.where(has_event, event, k)
            w = np.where(known_k, 1 / np.maximum(np.r_[1.0, G][when], 1e-6), 0.0)
            y = np.eye(len(FATES))[fate]
            bs_model = w * ((pred_rep[k] - y) ** 2).sum(axis=1)
            bs_clim = w * ((clim_rep[k] - y) ** 2).sum(axis=1)
            d_model = np.bincount(d, bs_model, n_d)
            d_clim = np.bincount(d, bs_clim, n_d)
            b_skill = 1 - (W @ d_model) / (W @ d_clim)
            at_k = (last >= k) & ~has_event
            support.append({
                "type": name, "days": days, "starts": len(s0), "drifters": n_d,
                "starts_followed_in_R": int(at_k.sum()), "drifters_followed_in_R": int(np.unique(d[at_k]).size),
                "share_fate_known": known_k.mean(), "gap_steps": n_gap,
                "ipcw_weight_mean": w.mean(),
                "brier_markov": bs_model.mean(), "brier_climatology": bs_clim.mean(),
                "brier_skill": 1 - d_model.sum() / d_clim.sum(),
                "brier_skill_lo": np.percentile(b_skill, 2.5), "brier_skill_hi": np.percentile(b_skill, 97.5)})

            # Calibration within deciles of each fate's predicted probability
            for o, fate_name in enumerate(FATES):
                if fate_name not in SHOWN:
                    continue
                p = pred_rep[k][:, o]
                edges = np.unique(np.quantile(p, np.linspace(0, 1, CALIB_BINS + 1)))
                if len(edges) < 2:
                    continue
                bin_of = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
                for b in np.unique(bin_of):
                    m = bin_of == b
                    a, e = risk_counts(d[m], n_d, event[m], cause[m], known[m])
                    n_follow = np.unique(d[m & (last >= k)]).size
                    lims = group_interval(m, p[m], k, o) if days in INTERVAL_DAYS else (np.nan,) * 4
                    calibration.append({
                        "type": name, "days": days, "fate": fate_name, "bin": int(b),
                        "predicted": p[m].mean(),
                        "observed": aalen_johansen(a.sum(axis=0), e.sum(axis=0))[k, o],
                        "observed_lo": lims[0], "observed_hi": lims[1],
                        "difference_lo": lims[2], "difference_hi": lims[3],
                        "starts": int(m.sum()), "drifters_at_horizon": int(n_follow)})

            # By start-longitude band
            for b in np.unique(band_of):
                m = band_of == b
                a, e = risk_counts(d[m], n_d, event[m], cause[m], known[m])
                ob = aalen_johansen(a.sum(axis=0), e.sum(axis=0))[k]
                for o, fate_name in enumerate(FATES):
                    lims = (group_interval(m, pred_rep[k][m, o], k, o)
                            if days in INTERVAL_DAYS and fate_name in SHOWN else (np.nan,) * 4)
                    bands.append({
                        "type": name, "days": days, "fate": fate_name,
                        "band": f"{LON_MIN + b * BAND_WIDTH:g}–{min(LON_MIN + (b + 1) * BAND_WIDTH, LON_MAX):g}°E",
                        "predicted": pred_rep[k][m, o].mean(), "observed": ob[o],
                        "observed_lo": lims[0], "observed_hi": lims[1],
                        "difference_lo": lims[2], "difference_hi": lims[3],
                        "starts": int(m.sum()), "drifters": int(np.unique(d[m]).size),
                        "drifters_at_horizon": int(np.unique(d[m & (last >= k)]).size)})

            # In each release point's own cell
            for _, pt in release.iterrows():
                state = state_of_cell[int((pt.lat - LAT_MIN) // RES) * n_lon + int((pt.lon - LON_MIN) // RES)]
                m = st0 == state
                if not m.any():
                    continue
                a, e = risk_counts(d[m], n_d, event[m], cause[m], known[m])
                ob = aalen_johansen(a.sum(axis=0), e.sum(axis=0))[k]
                for o, fate_name in enumerate(FATES):
                    if fate_name in SHOWN:
                        points.append({
                            "type": name, "days": days, "point": pt["name"], "fate": fate_name,
                            "predicted": pred_rep[k][m, o].mean(), "observed": ob[o],
                            "starts": int(m.sum()), "drifters": int(np.unique(d[m]).size),
                            "drifters_at_horizon": int(np.unique(d[m & (last >= k)]).size)})
        print(f"{name:<9}: {n_d:,} drifters, {len(s0):,} test starts, {n_gap:,} steps open across record gaps")

    summary = pd.DataFrame(summary)
    support = pd.DataFrame(support)
    calibration = pd.DataFrame(calibration)
    bands = pd.DataFrame(bands)
    points = pd.DataFrame(points)
    summary.to_csv(f"data/validation_year_summary_{TAG}.csv", index=False)
    support.to_csv(f"data/validation_year_support_{TAG}.csv", index=False)
    calibration.to_csv(f"data/validation_year_calibration_{TAG}.csv", index=False)
    bands.to_csv(f"data/validation_year_bands_{TAG}.csv", index=False)
    points.to_csv(f"data/validation_year_points_{TAG}.csv", index=False)

    # ── 4. Report ────────────────────────────────────────────────────────────────
    pct = lambda x: f"{100 * x:.1f}%"
    print(f"\nHeld-out fates, {K_FOLDS}-fold by drifter ({RES:g}°, τ = {TAU:g} d); predicted − observed with 95% interval")
    for name in DROGUE_TYPES:
        print(f"\n{name}")
        print(f"{'days':>5} {'fate':<10}{'pred':>8}{'obs':>8}{'95%':>16}{'pred−obs':>10}{'95%':>18}{'folds':>18}{'by band':>9}{'by era':>8}{'known only':>12}")
        for _, r in summary[(summary.type == name) & summary.fate.isin(SHOWN)].iterrows():
            print(f"{r.days:>5} {r.fate:<10}{pct(r.predicted):>8}{pct(r.observed):>8}"
                  f"{pct(r.observed_lo) + '–' + pct(r.observed_hi):>16}{100 * r.difference:>+10.1f}"
                  f"{f'{100 * r.difference_lo:+.1f} to {100 * r.difference_hi:+.1f}':>18}"
                  f"{f'{100 * r.fold_difference_min:+.1f} to {100 * r.fold_difference_max:+.1f}':>18}"
                  f"{pct(r.observed_by_band):>9}{pct(r.observed_by_era):>8}{pct(r.observed_known_only):>12}")
    print("\nSupport and Brier skill against the training mean fate (above 0 = the start cell adds skill)")
    print(support[["type", "days", "starts_followed_in_R", "drifters_followed_in_R", "share_fate_known",
                   "ipcw_weight_mean", "brier_markov", "brier_climatology", "brier_skill", "brier_skill_lo",
                   "brier_skill_hi"]].round(3).to_string(index=False))
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
    print(f"\nDrifters still followed in R at day {max(REPORT_DAYS)}, by release cell (drogued / undrogued):")
    at = points[(points.days == max(REPORT_DAYS)) & (points.fate == "in_R")].pivot_table(
        index="point", columns="type", values="drifters_at_horizon", sort=False).fillna(0).astype(int)
    print(at.to_string())
    print(f"\nSaved → data/validation_year_summary_{TAG}.csv, data/validation_year_support_{TAG}.csv, "
          f"data/validation_year_calibration_{TAG}.csv, data/validation_year_bands_{TAG}.csv, data/validation_year_points_{TAG}.csv")

    # ── 5. Figure: fates over the year, calibration at one year ──────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    days_axis = np.arange(K + 1) * TAU
    for ax, name in zip(axes[0], DROGUE_TYPES):
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
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(alpha=0.25)
        ax.set_title(f"{name.capitalize()}: fate of held-out drifters", fontsize=11)
        ax.legend(fontsize=8, loc="upper right", frameon=False)
    CALIB_PANEL_DAYS = {"drogued": 182, "undrogued": 364}   # too few drogued drifters last a year to bin
    for ax, name in zip(axes[1], DROGUE_TYPES):
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
    fig.suptitle(f"Held-out validation to one year — {RES:g}° grid, τ = {TAU:g} d, box R", fontsize=13)
    fig.tight_layout()
    out = f"figures/{region_prefix(args.data)}_validation_year_{TAG}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {out}")
    plt.close()


if __name__ == "__main__":
    main()
