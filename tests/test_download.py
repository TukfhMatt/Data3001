"""Downloads and the published ERA5 winds: a dropped connection is retried from where
it stopped, a file is kept only if its checksum matches, the published winds are used
only for the subset they were built for, and oil_wind refuses winds built for another
subset. No test touches the network."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import requests
import xarray as xr

from agulhas import download as dl
from agulhas.config import GDP_LOCAL
from agulhas.drifters import Drifters
from agulhas.operators.oil_wind import load_winds
from agulhas.sources import fetch_era5

CONTENT = bytes(range(256)) * 20
SUM = hashlib.sha256(CONTENT).hexdigest()


class Response:
    """A stand-in for requests' streamed response, which can drop part way."""

    def __init__(self, status_code, body, drop_after=None):
        self.status_code, self.body, self.drop_after = status_code, body, drop_after

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(self.status_code)

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), 500):
            if self.drop_after is not None and i >= self.drop_after:
                raise requests.exceptions.ChunkedEncodingError("Connection reset by peer")
            yield self.body[i:i + 500]


class Server:
    """Serves CONTENT, honouring byte ranges unless told not to, and drops the
    connection after drop_after bytes on each of its first `drops` requests."""

    def __init__(self, drops=0, drop_after=1500, ranges=True):
        self.drops, self.drop_after, self.ranges, self.starts = drops, drop_after, ranges, []

    def get(self, url, stream, timeout, headers=None):
        start = int(headers["Range"][6:-1]) if headers else 0
        self.starts.append(start)
        drop = self.drop_after if len(self.starts) <= self.drops else None
        if not self.ranges:
            return Response(200, CONTENT, drop)
        if start >= len(CONTENT):
            return Response(416, b"")
        return Response(206 if start else 200, CONTENT[start:], drop)


@pytest.fixture
def server(monkeypatch):
    """Install a fake server; retries do not wait."""
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)

    def install(**kw):
        s = Server(**kw)
        monkeypatch.setattr(dl.requests, "get", s.get)
        return s
    return install


def test_a_download_is_kept_when_its_checksum_matches(tmp_path, server):
    server()
    path = str(tmp_path / "sub" / "wind.nc")
    dl.download("https://example.org/wind.nc", path, SUM)
    assert open(path, "rb").read() == CONTENT
    assert dl.sha256(path) == SUM


def test_a_download_with_the_wrong_checksum_is_not_kept(tmp_path, server):
    server()
    with pytest.raises(SystemExit):
        dl.download("https://example.org/wind.nc", str(tmp_path / "wind.nc"), "0" * 64)
    assert list(tmp_path.iterdir()) == []


def test_a_dropped_connection_continues_from_where_it_stopped(tmp_path, server):
    s = server(drops=2)
    path = tmp_path / "wind.nc"
    dl.download("https://example.org/wind.nc", str(path), SUM)
    assert path.read_bytes() == CONTENT
    assert s.starts == [0, 1500, 3000]


def test_a_server_that_ignores_ranges_sends_the_file_again(tmp_path, server):
    server(drops=1, ranges=False)
    path = tmp_path / "wind.nc"
    dl.download("https://example.org/wind.nc", str(path), SUM)
    assert path.read_bytes() == CONTENT


def test_a_partial_file_from_an_earlier_run_is_continued(tmp_path, server):
    s = server()
    path = tmp_path / "wind.nc"
    (tmp_path / "wind.nc.part").write_bytes(CONTENT[:2000])
    dl.download("https://example.org/wind.nc", str(path), SUM)
    assert path.read_bytes() == CONTENT and s.starts == [2000]


def test_after_every_try_fails_the_run_stops_and_keeps_the_part(tmp_path, server):
    server(drops=dl.MAX_TRIES, drop_after=0)
    path = tmp_path / "wind.nc"
    with pytest.raises(SystemExit, match="Run the same command again"):
        dl.download("https://example.org/wind.nc", str(path))
    assert not path.exists() and (tmp_path / "wind.nc.part").exists()


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
