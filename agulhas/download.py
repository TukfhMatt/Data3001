"""
Downloads the pipeline makes: the shipping zip and the published ERA5 winds. Each
file goes to a temporary name first, so an interrupted download never looks like
the finished file, and is kept only if its checksum, when one is given, matches.
"""

import hashlib
import os

import requests


def sha256(path):
    """SHA-256 of a file, read in blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url, path, checksum=None):
    """Fetch url to path. With a checksum, a file that does not match is deleted and the run stops."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    part = path + ".part"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(part, "wb") as f:
            for block in r.iter_content(chunk_size=1 << 20):
                f.write(block)
    if checksum and sha256(part) != checksum:
        os.remove(part)
        raise SystemExit(f"{url} did not match its checksum, so it was not kept; try again.")
    os.replace(part, path)
