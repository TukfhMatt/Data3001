"""
Agulhas Region — OpenDrift Run for the Treasure Spill
The matrix cannot judge the MV Treasure spill of 23 June 2000 at 1° (spill_hindcasts).
This runs OceanDrift from the wreck with that date's currents and winds and applies
the same three rules: over 50% stranded within 30 days, the Atlantic and Cape coast
receiving the most, and material still afloat at 14 days having moved north and
west. It also reports particles reaching Robben and Dassen islands. Castillo de
Bellver (1983) is before GLORYS12 begins.

The main result releases all the oil at once on 23 June. --leak-days 5 is an
exploratory run that releases it evenly over 23–28 June. Those dates are an
assumption and not a leak record, and that run was added after the first result,
so it is reported as a sensitivity and not as the result.

    .venv-opendrift/bin/python -m agulhas opendrift_treasure
    .venv-opendrift/bin/python -m agulhas opendrift_treasure --leak-days 5
"""

from datetime import datetime
import json
import os

import numpy as np

from agulhas.config import parse_args
from agulhas.opendrift_common import OUT, merged_wind, simulate
from agulhas.transport import COAST_LABELS, coast_of
from agulhas.validation.spill_hindcasts import ISLANDS, SPILLS

N_PARTICLES = 5000
HOURS = 30 * 24
DIRECTION_HOURS = 14 * 24
NEAR_KM = 5
SEED = 2000


def km_from(x, y, x0, y0):
    return np.hypot((x - x0) * 111.32 * np.cos(np.radians(y0)), (y - y0) * 110.57)


# ── 1. The run ───────────────────────────────────────────────────────────────

def run_spill(spill, leak_days):
    """Hourly lon and lat of every particle, whether it stranded, and its last hour."""
    lon0, lat0 = spill["lon"], spill["lat"]
    lon, lat, _, stranded, last = simulate(datetime.fromisoformat(spill["date"]), np.full(N_PARTICLES, lon0), np.full(N_PARTICLES, lat0),
                                           np.zeros(N_PARTICLES, int), HOURS, SEED, wind_file=merged_wind(2000, next_january=False),
                                           release_hours=leak_days * 24)
    return lon, lat, stranded, last


# ── 2. The three rules and the islands ───────────────────────────────────────

def fates(spill, lon, lat, stranded, last, leak_days):
    """Stranding by day and coast, drift at 14 days and the islands reached."""
    lon0, lat0 = spill["lon"], spill["lat"]
    ends = np.arange(N_PARTICLES), last
    end_lon, end_lat = lon[ends], lat[ends]
    coasts = coast_of(end_lon, end_lat)
    out = {"release": f"evenly over {leak_days} days from {spill['date']}" if leak_days else f"all at once on {spill['date']}",
           "stranded_30d": float(stranded.mean()),
           "stranded_by_day": {d: float((stranded & (last <= d * 24)).mean()) for d in (1, 3, 7, 14, 30)},
           "coast_shares": {str(c): float((stranded & (coasts == k)).mean()) for k, c in enumerate(COAST_LABELS)}}
    afloat = last >= DIRECTION_HOURS
    out["afloat_at_14d"] = float(afloat.mean())
    east = north = None
    if afloat.any():
        east = float((np.nanmean(lon[afloat, DIRECTION_HOURS]) - lon0) * 111.32 * np.cos(np.radians(lat0)))
        north = float((np.nanmean(lat[afloat, DIRECTION_HOURS]) - lat0) * 110.57)
    out.update(east_km_14d=east, north_km_14d=north, northernmost_lat=float(np.nanmax(lat)))
    for name, (x0, y0) in ISLANDS.items():
        out[f"within_{NEAR_KM}km_{name}"] = float((np.nanmin(km_from(lon, lat, x0, y0), axis=1) <= NEAR_KM).mean())
        out[f"stranded_within_{NEAR_KM}km_{name}"] = float((stranded & (km_from(end_lon, end_lat, x0, y0) <= NEAR_KM)).mean())
    return out


def judge(out, lon, stranded, leak_days):
    """Add the verdict on each rule, and for a leak, stranding by day of release."""
    east, north = out["east_km_14d"], out["north_km_14d"]
    out["rule_stranding"] = "fits" if out["stranded_30d"] > 0.5 else "does not fit"
    out["rule_coast"] = "fits" if max(out["coast_shares"], key=out["coast_shares"].get) == "atlantic_cape" else "does not fit"
    out["rule_direction"] = "cannot be judged: nothing afloat" if east is None else "fits" if east < 0 and north > 0 else "does not fit"
    if leak_days:
        first = np.argmax(~np.isnan(lon), axis=1)
        out["stranded_by_release_day"] = {d: float(stranded[(first >= d * 24) & (first < (d + 1) * 24)].mean()) for d in range(leak_days)}
    return out


# ── 3. Save ──────────────────────────────────────────────────────────────────

def save(out, leak_days):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/treasure{f'_leak{leak_days}d' if leak_days else ''}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print(f"Saved → {path}")


def main(argv=None):
    """Run this step; see the module docstring for what it does."""
    def add_options(p):
        p.add_argument("--leak-days", type=int, default=0, help="release evenly over this many days (exploratory)")

    args = parse_args(__doc__, extra=add_options, argv=argv)
    spill = next(s for s in SPILLS if s["name"] == "Treasure")
    lon, lat, stranded, last = run_spill(spill, args.leak_days)
    out = fates(spill, lon, lat, stranded, last, args.leak_days)
    save(judge(out, lon, stranded, args.leak_days), args.leak_days)


if __name__ == "__main__":
    main()
