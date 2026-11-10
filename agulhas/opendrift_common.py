"""
Agulhas Region — OpenDrift Common
What the OpenDrift comparison steps share: the settings, one simulation, and
sorting particles into the same states as the matrix. OpenDrift is imported only
inside simulate, so the sorting also works in the project's main environment.
"""

import os
from datetime import timedelta

import numpy as np

from agulhas.transport import coast_of

OUT = "data/opendrift"
FORCING = f"{OUT}/forcing"
YEARS = [2000, 2005, 2010, 2015, 2020]
HOURS = {7: 7 * 24, 31.5: int(31.5 * 24)}     # the matrix's steps 2 and 9
DIFFUSIVITY = 10.0                            # m²/s, for mixing that daily 1/12° currents miss
OIL_WIND = 0.035                              # no Stokes field and no drifter windage to subtract, so the full share
N_COASTS = 6


def merged_wind(year, next_january=True):
    """One ERA5 file for a year's runs, with the names and attributes OpenDrift's reader expects."""
    import xarray as xr

    out = f"{FORCING}/era5_merged_{year}{'' if next_january else '_only'}.nc"
    if os.path.exists(out):
        return out
    parts = [xr.open_dataset(f"{FORCING}/era5_{y}.nc").load() for y in ([year, f"{year + 1}01"] if next_january else [year])]
    ds = xr.concat(parts, dim="valid_time").rename({"valid_time": "time", "longitude": "lon", "latitude": "lat"})
    ds = ds[["u10", "v10"]].sortby("lat").sortby("time")
    ds = ds.drop_vars([c for c in ds.coords if c not in ("time", "lon", "lat")])
    ds["u10"].attrs.update(standard_name="x_wind", units="m s-1")
    ds["v10"].attrs.update(standard_name="y_wind", units="m s-1")
    ds["lon"].attrs.update(standard_name="longitude", units="degrees_east")
    ds["lat"].attrs.update(standard_name="latitude", units="degrees_north")
    ds.to_netcdf(out + ".tmp")
    os.replace(out + ".tmp", out)
    return out


def simulate(start, lons, lats, markers, hours, seed, wind_file=None, release_hours=0, times=None, wind=OIL_WIND, max_age_hours=None, diffusivity=DIFFUSIVITY):
    """Run OceanDrift with hourly output. Returns hourly lon and lat (particles, hours + 1), empty before
    a particle is released and after it stops, with each particle's marker, whether it stranded, and its last hour.
    times gives each particle its own release time; max_age_hours stops each particle that long after its release."""
    from opendrift.models.oceandrift import OceanDrift
    from opendrift.readers import reader_netCDF_CF_generic

    np.random.seed(seed)
    m = OceanDrift(loglevel=50)
    m.add_reader([reader_netCDF_CF_generic.Reader(f"{FORCING}/glorys_{start.year}.nc"),
                  reader_netCDF_CF_generic.Reader(wind_file or merged_wind(start.year))])
    m.set_config("general:coastline_action", "stranding")
    m.set_config("general:use_auto_landmask", True)
    m.set_config("drift:advection_scheme", "runge-kutta")
    m.set_config("drift:stokes_drift", False)
    m.set_config("drift:vertical_mixing", False)
    m.set_config("drift:current_uncertainty", 0)
    m.set_config("drift:wind_uncertainty", 0)
    m.set_config("environment:constant:horizontal_diffusivity", diffusivity)
    if max_age_hours:
        m.set_config("drift:max_age_seconds", max_age_hours * 3600)
    when = list(times) if times is not None else [start, start + timedelta(hours=release_hours)] if release_hours else start
    m.seed_elements(lon=lons, lat=lats, radius=0, number=len(lons), time=when, wind_drift_factor=wind, origin_marker=markers)
    m.run(duration=timedelta(hours=hours), time_step=3600, time_step_output=3600,
          export_variables=["lon", "lat", "status", "origin_marker"])
    r = m.result
    lon, lat, status = r.lon.values, r.lat.values, r.status.values
    n, t = lon.shape
    if t > hours + 1:
        raise RuntimeError(f"expected at most {hours + 1} outputs, got {t}")
    last = t - 1 - np.argmax(~np.isnan(lon[:, ::-1]), axis=1)
    meanings = r.status.attrs["flag_meanings"].split()
    stranded = status[np.arange(n), last] == meanings.index("stranded") if "stranded" in meanings else np.zeros(n, bool)
    marker = np.array(r.origin_marker.values[np.arange(n), last], dtype=int)
    if t < hours + 1:    # OpenDrift stops once no particle is active
        pad = np.full((n, hours + 1 - t), np.nan)
        lon, lat = np.hstack([lon, pad]), np.hstack([lat, pad])
    return lon, lat, marker, stranded, last


def classify(lon, lat, stranded, last, hour, cell_state, n_cells, box, n_lon):
    """Each particle's state at an hour, counted as the matrix counts them: the first crossing of an edge of R is
    an exit and the first landfall in R is stranded, and neither returns. States are 0..n_cells-1 for ocean cells,
    then exits W, E, S, N, then the six coasts, then one for a position the matrix has no state for."""
    lon0, lon1, lat0, lat1 = box
    state = np.empty(lon.shape[0], int)
    for i in range(lon.shape[0]):
        upto = min(hour, last[i])
        x, y = lon[i, :upto + 1], lat[i, :upto + 1]
        out = (x < lon0) | (x >= lon1) | (y < lat0) | (y >= lat1)
        if out.any():
            j = int(np.argmax(out))
            state[i] = n_cells + (0 if x[j] < lon0 else 1 if x[j] >= lon1 else 2 if y[j] < lat0 else 3)
        elif stranded[i] and last[i] <= hour:
            state[i] = n_cells + 4 + int(coast_of(x[-1], y[-1]))
        elif last[i] < hour:
            state[i] = n_cells + 4 + N_COASTS     # stopped for another reason; none are expected inside R
        else:
            flat = int(np.floor(y[-1] - lat0)) * n_lon + int(np.floor(x[-1] - lon0))
            state[i] = cell_state.get(flat, n_cells + 4 + N_COASTS)
    return state


def hourly_states(lon, lat, stranded, last, cell_state, n_cells, box, n_lon):
    """classify at every hour at once: (particles, hours) states, for counting moves."""
    lon0, lon1, lat0, lat1 = box
    n, t = lon.shape
    none = n_cells + 4 + N_COASTS
    hour, row = np.arange(t)[None, :], np.arange(n)
    alive = hour <= last[:, None]
    with np.errstate(invalid="ignore"):
        out = alive & ((lon < lon0) | (lon >= lon1) | (lat < lat0) | (lat >= lat1))
        flat = np.floor(lat - lat0) * n_lon + np.floor(lon - lon0)
    lookup = np.full(max(cell_state) + 1, none)
    lookup[list(cell_state)] = list(cell_state.values())
    known = alive & ~out & np.isfinite(flat) & (flat >= 0) & (flat < len(lookup))
    state = np.full((n, t), none)
    state[known] = lookup[flat[known].astype(int)]
    end_lon, end_lat = lon[row, last], lat[row, last]
    coast = n_cells + 4 + coast_of(end_lon, end_lat)
    state = np.where(stranded[:, None] & (hour >= last[:, None]), coast[:, None], state)
    first = np.where(out.any(axis=1), np.argmax(out, axis=1), t)
    x, y = lon[row, np.minimum(first, t - 1)], lat[row, np.minimum(first, t - 1)]
    side = np.select([x < lon0, x >= lon1, y < lat0], [0, 1, 2], 3)
    return np.where(hour >= first[:, None], (n_cells + side)[:, None], state)
