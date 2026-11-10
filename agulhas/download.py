"""
Downloads the pipeline makes: the shipping zip and the published ERA5 winds. Each
file goes to a temporary name first, so an interrupted download never looks like
the finished file, and is kept only if its checksum, when one is given, matches.
A dropped connection is retried from where it stopped (both servers accept byte
ranges), and a partial file left by an earlier run is continued, not restarted.
"""

import hashlib
import os
import time

import requests

MAX_TRIES = 5
WAIT_SECONDS = 10      # before a retry, times the number of the try


def sha256(path):
    """SHA-256 of a file, read in blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_part(url, part):
    """Add to the partial file from where it stops. A server that ignores the range sends it all again."""
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, stream=True, timeout=60, headers=headers) as r:
        if have and r.status_code == 416:            # the partial file is already complete
            return
        r.raise_for_status()
        with open(part, "ab" if have and r.status_code == 206 else "wb") as f:
            for block in r.iter_content(chunk_size=1 << 20):
                f.write(block)


def download(url, path, checksum=None):
    """Fetch url to path, retrying a dropped connection. With a checksum, a file that
    does not match is deleted and the run stops."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    part = path + ".part"
    for attempt in range(1, MAX_TRIES + 1):
        try:
            fetch_part(url, part)
            break
        except requests.RequestException as e:
            if attempt == MAX_TRIES:
                raise SystemExit(f"Downloading {url} failed {MAX_TRIES} times ({type(e).__name__}). "
                                 "Run the same command again to continue from where it stopped.")
            have = os.path.getsize(part) / 1e6 if os.path.exists(part) else 0
            print(f"Download interrupted ({type(e).__name__}); retrying from {have:.0f} MB.", flush=True)
            time.sleep(WAIT_SECONDS * attempt)
    if checksum and sha256(part) != checksum:
        os.remove(part)
        raise SystemExit(f"{url} did not match its checksum, so it was not kept; try again.")
    os.replace(part, path)
