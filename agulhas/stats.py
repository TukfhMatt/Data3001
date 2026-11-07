"""Small statistics shared by the steps."""

import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine_km(lo1, la1, lo2, la2):
    """Great-circle distance in km."""
    lo1, la1, lo2, la2 = map(np.radians, (lo1, la1, lo2, la2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h))


def interval(draws, axis=0):
    """95% interval of resampled values, ignoring NaN: (low, high)."""
    with np.errstate(all="ignore"):
        return np.nanpercentile(draws, [2.5, 97.5], axis=axis)
