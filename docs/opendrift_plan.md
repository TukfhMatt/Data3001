# OpenDrift comparison: settings and rules, fixed before any OpenDrift result was seen

Model: OpenDrift OceanDrift, no weathering. Currents: GLORYS12 surface uo/vo, daily, 1/12°. Wind: ERA5 10 m at 3.5%, 0.5°, 6-hourly.
No Stokes drift field. Horizontal diffusivity 10 m²/s. Runge-Kutta, 1 h step. Coast action: stranding, on OpenDrift's default landmask.

## 1. Spread and fate
Years 2000, 2005, 2010, 2015, 2020. Starts on the 1st of each month (60 starts). The 13 named release points, 500 particles per point per start,
seeded uniformly over the ocean part of the point's 1° cell, the same information the matrix has. Horizons: day 7 and day 31.5 (matrix steps 2 and 9).
First crossing of an edge of R counts as an exit, and first landfall inside R as stranded; neither can return, as in the matrix. Stranded particles get a coast from coast_of.
Compared with the `oil` matrix. A point agrees at day 31.5 if all three hold:
  a. TVD over the ocean cells, four exits and stranded is at most config.DISAGREE_TVD (0.3)
  b. the stranding intervals overlap (OpenDrift: 95% from resampling start dates; matrix: atlas)
  c. the largest of in-box, exit W/E/S/N and stranded is the same
Reference: how much OpenDrift differs from itself between years (TVD of each year against the other four).
The five years do not match the matrix's 1995–2022 period, so that is reported as a limit.

## 2. Treasure, 2000
Start 23 June 2000 at 18.33167°E, 33.67167°S, 5,000 particles from the wreck position. Uses the same three rules as the spill hindcast step,
with wind at 3.5% and the same 30-day and 14-day horizons. Also reported: particles reaching Robben and Dassen islands.
Castillo de Bellver (1983) is before GLORYS begins.

## 3. Held-out drifters (after 1 and 2)
Undrogued test starts, 2% wind, position error against the matrix at the same starts.

Interpretation: OpenDrift knows each date's real currents and winds and the matrix only the average, so this measures what averaging costs, not which is better.

## 3. Held-out drifters: scoring rule (fixed 2026-11-09, before any run was scored)
Starts: held_out's undrogued test starts in the five years whose 28-day horizon ends before the forcing does. OpenDrift at 2% wind, seeded at the exact position (10 particles) and over the 1° cell (20).
Score: distance from the forecast's centre (of what is still in R) to the drifter's actual position, at every whole step of τ up to 28 days, for starts the drifter is still in R at that step. Compared: matrix, OpenDrift exact, OpenDrift cell, persistence.
OpenDrift beats the matrix at a horizon if the 95% interval of the difference in median error (matrix − OpenDrift) excludes zero; intervals resample whole drifters. Starts with no OpenDrift particle in R are reported separately, not dropped.
Not covered by the rule: results are added to Research.md whichever way they fall.

## 4. Added after parts 1–3 (asked for 2026-11-09; rules fixed before any of these ran)
A. Interval coverage. For each release point, horizon (day 7, 31.5) and outcome (in R, stranded, exit W/E/S/N), count how often OpenDrift's mean share lies inside the oil matrix's 95% bootstrap interval (risk atlas). Outcomes where both shares are below 1% are left out. Reported overall and for the 6 South African points against the 7 lanes. No pass mark: inside means the gap is within the matrix's sampling noise, outside means data or physics differ by more.
B. Settings sensitivity.
  1. Windage 3% and 4% at the 13 points (the same 60 starts), against the oil_3pct and oil_4pct matrices, under part 1's three rules at day 31.5. Reported: points agreeing, and OpenDrift's stranding at Algoa Bay, Richards Bay and the 27.5°E lane.
  2. Horizontal diffusivity 1 and 50 m²/s (10 is the base) on the held-out drifters at 2% wind, scored by part 3's rule. OpenDrift's lead holds if the interval excludes zero at 6 or more of the 8 horizons for both seedings.
C. Weathering. OpenOil with the NOAA weathering model, wind at 3.5%, real ERA5 wind, no wave field and no Stokes drift, constant water temperature 15 and 25 °C. Oils: GENERIC INTERMEDIATE FUEL OIL 300 (bunker fuel), AMSA Average Very Low Sulfur Fuel Oil, and GENERIC DIESEL as a light bound. 13 points, 12 monthly starts in 2010, 100 particles per point. The mass left is 1 − (evaporated + dispersed)/initial, over all particles, at days 1, 3, 7, 14 and 30. Fit m(t)/m0 = exp(−λt) by least squares on the log over days 0–30. The working λ = 1/14 per day fits an oil if its fitted 1/λ lies between 7 and 28 days, a factor of 2 either side; otherwise it does not fit. The half-life is also reported. Weathering depends strongly on temperature and waves, which are not in the forcing, so the range over the two temperatures is the result, not one number.

Note (2026-11-09): part C (OpenOil weathering) was not used. Its fuel-oil results (about half the oil dispersed in a day) contradict Mackay et al. (1980), where heavy fuel oil evaporates at most about 15% and its mousse suppresses dispersion, and OpenOil's dispersion needs wave height and water temperature that the forcing lacks. The oil mass left is config.oil_mass_left instead.
