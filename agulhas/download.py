"""
Downloads the pipeline makes: the shipping zip, the published ERA5 winds and the
OpenDrift forcing. Each file goes to a temporary name first, so an interrupted
download never looks like the finished file, and is kept only if its checksum, when
one is given, matches. A dropped connection is retried from where it stopped (all
servers accept byte ranges), and the download stops after MAX_TRIES tries in a row
that add nothing. A partial file left by an earlier run is continued, not restarted.
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
    attempt, last = 0, -1
    while True:
        try:
            fetch_part(url, part)
            break
        except requests.RequestException as e:
            size = os.path.getsize(part) if os.path.exists(part) else 0
            attempt = attempt + 1 if size <= last else 1       # only tries that bring nothing new count
            last = size
            if attempt == MAX_TRIES:
                raise SystemExit(f"Downloading {url} failed {MAX_TRIES} times in a row ({type(e).__name__}). "
                                 "Run the same command again to continue from where it stopped.")
            print(f"Download interrupted ({type(e).__name__}); retrying from {size / 1e6:.0f} MB.", flush=True)
            time.sleep(WAIT_SECONDS * attempt)
    if checksum and sha256(part) != checksum:
        os.remove(part)
        raise SystemExit(f"{url} did not match its checksum, so it was not kept; try again.")
    os.replace(part, path)
