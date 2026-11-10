"""
Agulhas Region — Pipeline
The steps in run order, and the command line that runs them, from the repository
root.

    python -m agulhas list                           print the steps
    python -m agulhas all                            every step, in order
    python -m agulhas all --from coastal_stranding   resume at a step
    python -m agulhas all --box -80 -40 25 45 --data data/gulfstream_gdp1h_subset.nc
                                                     another region: the steps that do not
                                                     need the Agulhas coasts, routes or winds
    python -m agulhas bootstrap --res 1 --tau 3.5    one step, with its own options

A full run keeps one log per step in data/logs/ and stops at the first step that
fails. Downloads run only when their file is missing (ERA5 needs ~/.cdsapirc and
takes many hours). Add --help after a step's name for its options.
"""

import argparse
from collections import namedtuple
import importlib
import os
import subprocess
import sys
import time

import requests

from agulhas.config import GDP_LOCAL, SHIP_URL, SHIP_ZIP, wind_paths

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = "data/logs"

Step = namedtuple("Step", "name module options agulhas_only", defaults=((), False))

# ── 1. The steps, in run order; a step reads what the ones before it wrote ───
STEPS = [
    Step("fetch_gdp_subset", "sources.fetch_gdp_subset", agulhas_only=True),
    Step("gdp_exploration", "exploration.gdp_exploration"),
    Step("grid_coverage", "exploration.grid_coverage"),
    Step("transition_matrix", "operators.transition_matrix"),
    Step("transition_matrix_lags", "operators.transition_matrix", ("--res", "1", "--tau", "2", "5", "7")),
    Step("shipping_overlay", "release_points.shipping_overlay"),
    Step("box_selection", "exploration.box_selection", agulhas_only=True),
    Step("bootstrap", "validation.bootstrap"),
    Step("wind_slip", "exploration.wind_slip"),
    Step("fetch_era5", "sources.fetch_era5", agulhas_only=True),
    Step("oil_wind", "operators.oil_wind", agulhas_only=True),
    Step("held_out", "validation.held_out"),
    Step("held_out_tau2", "validation.held_out", ("--res", "1", "--tau", "2")),
    Step("held_out_tau5", "validation.held_out", ("--res", "1", "--tau", "5")),
    Step("held_out_res05", "validation.held_out", ("--res", "0.5", "--tau", "3.5")),
    Step("held_out_res2", "validation.held_out", ("--res", "2", "--tau", "3.5")),
    Step("held_out_year", "validation.held_out_year"),
    Step("whole_box_analysis", "whole_box.whole_box_analysis"),
    Step("coastal_coverage", "whole_box.coastal_coverage"),
    Step("trap_stability", "whole_box.trap_stability"),
    Step("residence_time", "whole_box.residence_time"),
    Step("exit_zones", "whole_box.exit_zones"),
    Step("almost_invariant_regions", "whole_box.almost_invariant_regions"),
    Step("release_point_maps", "release_points.release_point_maps"),
    Step("sensitivity", "validation.sensitivity"),
    Step("seasonal_matrices", "operators.seasonal_matrices"),
    Step("seasonal_matrices_res2", "operators.seasonal_matrices", ("--res", "2", "--tau", "3.5")),
    Step("chapman_kolmogorov", "validation.chapman_kolmogorov"),
    Step("coastal_pooling", "coast.coastal_pooling"),
    Step("drogue_loss", "exploration.drogue_loss"),
    Step("coastal_stranding", "coast.coastal_stranding", agulhas_only=True),
    Step("traffic_exposure", "coast.traffic_exposure", agulhas_only=True),
    Step("risk_atlas", "coast.risk_atlas", agulhas_only=True),
    Step("hypothesis_tests", "validation.hypothesis_tests"),
    Step("decorrelation_time", "exploration.decorrelation_time"),
    Step("oil_windage", "operators.oil_windage", agulhas_only=True),
    Step("spill_hindcasts", "validation.spill_hindcasts", agulhas_only=True),
    Step("source_sink", "whole_box.source_sink"),
    Step("known_features", "validation.known_features", agulhas_only=True),
]
# Not part of a full run: the tools only read the pipeline's outputs, and the
# OpenDrift comparison needs its own environment (requirements-opendrift.txt)
TOOLS = [
    Step("worked_example", "tools.worked_example"),
    Step("oil_explorer", "tools.oil_explorer"),
    Step("opendrift_forcing", "comparison.opendrift_forcing"),
    Step("opendrift_points", "comparison.opendrift_points"),
    Step("opendrift_treasure", "comparison.opendrift_treasure"),
    Step("opendrift_compare", "comparison.opendrift_compare"),
    Step("opendrift_moves", "comparison.opendrift_moves"),
    Step("opendrift_checks", "comparison.opendrift_checks"),
    Step("opendrift_drifters", "comparison.opendrift_drifters"),
    Step("opendrift_score", "comparison.opendrift_score"),
    Step("opendrift_coverage", "comparison.opendrift_coverage"),
]
NEEDS_SHIPPING = {"shipping_overlay", "box_selection"}


def fetch_shipping():
    """Download the shipping zip to a temporary name, so an interrupted download
    never looks like the finished file."""
    os.makedirs(os.path.dirname(SHIP_ZIP), exist_ok=True)
    print(f"== shipping data ({SHIP_ZIP}, 458 MB)", flush=True)
    part = SHIP_ZIP + ".part"
    with requests.get(SHIP_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(part, "wb") as f:
            for block in r.iter_content(chunk_size=1 << 20):
                f.write(block)
    os.replace(part, SHIP_ZIP)


def ready(step):
    """Fetch what a step needs. False if its output exists and the step can be skipped."""
    wind = wind_paths(GDP_LOCAL)[1]
    if step.name == "fetch_gdp_subset" and os.path.exists(GDP_LOCAL):
        print(f"== {step.name} skipped: {GDP_LOCAL} exists", flush=True)
        return False
    if step.name in NEEDS_SHIPPING and not os.path.exists(SHIP_ZIP):
        fetch_shipping()
    if step.name == "fetch_era5":
        if os.path.exists(wind):
            print(f"== {step.name} skipped: {wind} exists", flush=True)
            return False
        if not os.path.exists(os.path.expanduser("~/.cdsapirc")):
            raise SystemExit(f"{step.name} needs ERA5 winds: create ~/.cdsapirc (see README), or supply {wind}.")
    return True


def run_all(argv):
    """Run every step in its own process, so each starts with its memory free."""
    p = argparse.ArgumentParser(prog="python -m agulhas all", description="Run every step, in order.")
    p.add_argument("--from", dest="start", metavar="STEP", help="resume at this step")
    p.add_argument("--box", nargs=4, metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"), help="another region")
    p.add_argument("--data", help="that region's subset, from fetch_gdp_subset")
    args = p.parse_args(argv)
    if bool(args.box) != bool(args.data):
        raise SystemExit("--box and --data go together: another region needs its own subset from fetch_gdp_subset.")
    region = []
    if args.data:
        if not os.path.exists(args.data):
            raise SystemExit(f"{args.data} not found: fetch it first, e.g. python -m agulhas fetch_gdp_subset "
                             f"--fetch-box -90 -30 15 55 --out {args.data}")
        region = ["--box", *args.box, "--data", args.data]
    names = [s.name for s in STEPS]
    if args.start and args.start not in names:
        raise SystemExit(f"No step called {args.start} (try: python -m agulhas list)")

    os.makedirs(LOGS, exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    first = names.index(args.start) if args.start else 0
    for step in STEPS[first:]:
        if args.data and step.agulhas_only:
            continue
        if not ready(step):
            continue
        print(f"== {step.name} {time.strftime('%H:%M:%S')}", flush=True)
        log = f"{LOGS}/{step.name}.log"
        command = [sys.executable, "-W", "ignore", "-m", f"agulhas.{step.module}", *step.options, *region]
        with open(log, "w") as f:
            result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            with open(log, errors="replace") as f:
                print(f"FAILED at step {step.name}; the end of {log}:\n" + "".join(f.readlines()[-20:]))
            raise SystemExit(f"Resume with: python -m agulhas all --from {step.name}")
    print(f"== done {time.strftime('%H:%M:%S')}; logs in {LOGS}/")


def run_one(step, argv):
    """Run one step in this process, with its own options on top of the step's."""
    sys.argv[0] = f"python -m agulhas {step.name}"     # the name a step's --help shows
    importlib.import_module(f"agulhas.{step.module}").main([*step.options, *argv])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return
    os.chdir(ROOT)
    command, rest = argv[0], argv[1:]
    known = {s.name: s for s in STEPS + TOOLS}
    if command == "list":
        for s in STEPS:
            print(" ".join([s.name.ljust(26), *s.options]) + ("  (Agulhas only)" if s.agulhas_only else ""))
        print("\nNot part of a full run: " + ", ".join(s.name for s in TOOLS))
    elif command == "all":
        run_all(rest)
    elif command in known:
        run_one(known[command], rest)
    else:
        raise SystemExit(f"No step called {command} (try: python -m agulhas list)")
