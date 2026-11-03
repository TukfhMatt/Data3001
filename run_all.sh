#!/usr/bin/env bash
# Runs the whole pipeline in order, from the repository root, with one log per
# step in data/logs/. It stops at the first step that fails. Downloads run only
# when their file is missing (ERA5 needs ~/.cdsapirc and takes many hours).
#
#   ./run_all.sh                 every step, in order
#   ./run_all.sh --from 23       resume at a step (its first run, if it has several)
#   ./run_all.sh --list          print the steps
#   ./run_all.sh --box -80 -40 25 45 --data data/gulfstream_gdp1h_subset.nc
#                                another region: the steps that do not need the
#                                Agulhas coasts, routes or ERA5 winds
#
# PYTHON overrides the interpreter (default .venv/bin/python).
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-.venv/bin/python}"
SUBSET="data/agulhas_gdp1h_subset.nc"
SHIP_ZIP="data/shipping/shipdensity_commercial.zip"
SHIP_URL="https://datacatalogfiles.worldbank.org/ddh-published/0037580/5/DR0045405/shipdensity_commercial_.zip"
WIND="data/drifter_wind10m.nc"
LOGS="data/logs"

# ── 1. The steps: label, then script and options; "*" marks Agulhas-only steps ─
STEPS=(
  "00 |*|00_fetch_agulhas_subset"
  "01 ||01_agulhas_gdp_exploration"
  "02 ||02_grid_coverage"
  "03 ||03_transition_matrix"
  "03_tau ||03_transition_matrix --res 1 --tau 2 5 7"
  "04 ||04_shipping_overlay"
  "05 |*|05_box_selection"
  "06 ||06_bootstrap"
  "07 ||07_wind_slip"
  "08 |*|08_fetch_era5"
  "09 |*|09_oil_wind"
  "10 ||10_validation"
  "10_tau2 ||10_validation --res 1 --tau 2"
  "10_tau5 ||10_validation --res 1 --tau 5"
  "10_res05 ||10_validation --res 0.5 --tau 3.5"
  "10_res2 ||10_validation --res 2 --tau 3.5"
  "11 ||11_whole_box_analysis"
  "12 ||12_coastal_coverage"
  "13 ||13_trap_stability"
  "14 ||14_residence_time"
  "15 ||15_exit_zones"
  "16 ||16_almost_invariant_regions"
  "17 ||17_release_point_maps"
  "18 ||18_sensitivity"
  "19 ||19_seasonal_matrices"
  "19_res2 ||19_seasonal_matrices --res 2 --tau 3.5"
  "20 ||20_chapman_kolmogorov"
  "21 ||21_coastal_pooling"
  "22 ||22_drogue_loss"
  "23 |*|23_coastal_stranding"
  "24 |*|24_traffic_exposure"
  "25 |*|25_risk_atlas"
  "26 ||26_hypothesis_tests"
  "27 ||27_decorrelation_time"
  "28 |*|28_oil_windage"
  "29 |*|29_spill_hindcasts"
)

# ── 2. Options ───────────────────────────────────────────────────────────────
FROM=""
BOX=()
DATA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --box) BOX=(--box "$2" "$3" "$4" "$5"); shift 5 ;;
    --data) DATA="$2"; shift 2 ;;
    --list) for s in "${STEPS[@]}"; do echo "${s%% *}: ${s##*|}"; done; exit 0 ;;
    -h|--help) sed -n 2,13p "$0" | cut -c3-; exit 0 ;;
    *) echo "Unknown option: $1 (try --help)"; exit 2 ;;
  esac
done
if { [ ${#BOX[@]} -gt 0 ] && [ -z "$DATA" ]; } || { [ ${#BOX[@]} -eq 0 ] && [ -n "$DATA" ]; }; then
  echo "--box and --data go together: another region needs its own subset from 00."
  exit 2
fi
REGION=()
if [ -n "$DATA" ]; then
  [ -f "$DATA" ] || { echo "$DATA not found: fetch it first, e.g. python scripts/00_fetch_agulhas_subset.py --fetch-box -90 -30 15 55 --out $DATA"; exit 1; }
  REGION=("${BOX[@]}" --data "$DATA")
fi
if [ -n "$FROM" ]; then
  known=0
  for s in "${STEPS[@]}"; do [ "${s%% *}" = "$FROM" ] && known=1; done
  [ "$known" = 1 ] || { echo "No step called $FROM (try --list)"; exit 2; }
fi

# ── 3. Run ───────────────────────────────────────────────────────────────────
mkdir -p "$LOGS" figures
started=$([ -z "$FROM" ] && echo 1 || echo 0)
for s in "${STEPS[@]}"; do
  label="${s%% *}"
  rest="${s#* |}"
  only="${rest%%|*}"
  cmd="${rest#*|}"
  [ "$label" = "$FROM" ] && started=1
  [ "$started" = 1 ] || continue
  [ -n "$DATA" ] && [ "$only" = "*" ] && continue
  case "$label" in
    00) [ -f "$SUBSET" ] && { echo "== 00 skipped: $SUBSET exists"; continue; } ;;
    04|05)
      if [ ! -f "$SHIP_ZIP" ]; then
        echo "== shipping data ($SHIP_ZIP, 458 MB)"
        mkdir -p "$(dirname "$SHIP_ZIP")"
        curl -fL -o "$SHIP_ZIP" "$SHIP_URL"
      fi ;;
    08)
      if [ -f "$WIND" ]; then echo "== 08 skipped: $WIND exists"; continue; fi
      [ -f "$HOME/.cdsapirc" ] || { echo "08 needs ERA5 winds: create ~/.cdsapirc (see README), or supply $WIND."; exit 1; } ;;
  esac
  script="${cmd%% *}"
  options=""
  [ "$script" = "$cmd" ] || options="${cmd#* }"
  echo "== $label $(date +%H:%M:%S)"
  # shellcheck disable=SC2086
  if ! "$PY" -W ignore "scripts/$script.py" $options ${REGION[@]+"${REGION[@]}"} > "$LOGS/$label.log" 2>&1; then
    echo "FAILED at step $label; the end of $LOGS/$label.log:"
    tail -20 "$LOGS/$label.log"
    echo "Resume with: ./run_all.sh --from $label"
    exit 1
  fi
done
echo "== done $(date +%H:%M:%S); logs in $LOGS/"
