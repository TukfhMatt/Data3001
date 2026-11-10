"""The published ERA5 winds: downloads keep only a file whose checksum matches, the
published winds are used only for the subset they were built for, and oil_wind
refuses winds built for another subset. No test touches the network."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from agulhas import download as dl
from agulhas.config import GDP_LOCAL
from agulhas.drifters import Drifters
from agulhas.operators.oil_wind import load_winds
from agulhas.sources import fetch_era5

CONTENT = b"winds" * 1000


class Response:
    """A stand-in for requests' streamed response."""

    def __init__(self, blocks):
        self.blocks = blocks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for block in self.blocks:
            if isinstance(block, Exception):
                raise block
            yield block


def serve(monkeypatch, blocks):
    monkeypatch.setattr(dl.requests, "get", lambda url, stream, timeout: Response(blocks))


def test_a_download_is_kept_when_its_checksum_matches(tmp_path, monkeypatch):
    serve(monkeypatch, [CONTENT[:2000], CONTENT[2000:]])
    path = str(tmp_path / "sub" / "wind.nc")
    dl.download("https://example.org/wind.nc", path, hashlib.sha256(CONTENT).hexdigest())
    assert open(path, "rb").read() == CONTENT
    assert dl.sha256(path) == hashlib.sha256(CONTENT).hexdigest()


def test_a_download_with_the_wrong_checksum_is_not_kept(tmp_path, monkeypatch):
    serve(monkeypatch, [CONTENT])
    path = tmp_path / "wind.nc"
    with pytest.raises(SystemExit):
        dl.download("https://example.org/wind.nc", str(path), "0" * 64)
    assert list(tmp_path.iterdir()) == []


def test_an_interrupted_download_never_looks_finished(tmp_path, monkeypatch):
    serve(monkeypatch, [CONTENT[:100], ConnectionError("dropped")])
    path = tmp_path / "wind.nc"
    with pytest.raises(ConnectionError):
        dl.download("https://example.org/wind.nc", str(path))
    assert not path.exists()


def write_subset(path, lon):
    t = np.datetime64("2001-03-01T00:00") + np.arange(len(lon)).astype("timedelta64[h]")
    xr.Dataset({"lon": ("obs", np.asarray(lon, np.float32)), "lat": ("obs", np.full(len(lon), -30.0, np.float32)),
                "time": ("obs", t)}).to_netcdf(path)
    return Drifters(str(path))


def test_the_fingerprint_follows_positions_and_times(tmp_path):
    a = write_subset(tmp_path / "a.nc", [20.0, 20.1, 20.2])
    same = write_subset(tmp_path / "b.nc", [20.0, 20.1, 20.2])
    moved = write_subset(tmp_path / "c.nc", [20.0, 20.1, 20.3])
    assert a.fingerprint == same.fingerprint != moved.fingerprint


@pytest.mark.parametrize("data, cds, fits, used", [
    (GDP_LOCAL, False, True, True),
    (GDP_LOCAL, True, True, False),                         # --cds
    (GDP_LOCAL, False, False, False),                       # the subset has changed since the winds were built
    ("data/gulfstream_gdp1h_subset.nc", False, True, False),   # another subset
])
def test_the_published_winds_are_used_only_for_the_subset_they_fit(monkeypatch, data, cds, fits, used):
    calls = []
    monkeypatch.setattr(fetch_era5, "download", lambda url, path, checksum: calls.append((url, path, checksum)))
    drifters = SimpleNamespace(fingerprint=fetch_era5.WIND_SUBSET if fits else "another")
    assert fetch_era5.published(SimpleNamespace(data=data, cds=cds), drifters, "data/w.nc") is used
    assert calls == ([(fetch_era5.WIND_URL, "data/w.nc", fetch_era5.WIND_SHA256)] if used else [])


@pytest.mark.parametrize("built_for, ok", [("match", True), ("another", False), (None, True)])
def test_oil_wind_refuses_winds_built_for_another_subset(tmp_path, built_for, ok):
    drifters = write_subset(tmp_path / "subset.nc", [20.0, 20.1, 20.2])
    attrs = {} if built_for is None else {"subset_fingerprint": drifters.fingerprint if built_for == "match" else "x"}
    path = tmp_path / "wind.nc"
    xr.Dataset({"u10": ("obs", np.ones(3, np.float32)), "v10": ("obs", np.zeros(3, np.float32))},
               attrs=attrs).to_netcdf(path)
    if ok:
        u10, v10 = load_winds(str(path), drifters)
        assert u10.tolist() == [1.0, 1.0, 1.0]
    else:
        with pytest.raises(SystemExit):
            load_winds(str(path), drifters)
