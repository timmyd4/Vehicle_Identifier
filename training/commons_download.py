"""
Step 2 of the Commons pipeline: download the photos that commons_list.py found.

THE BIG PICTURE
---------------
manifest.csv (from commons_list.py) lists every usable photo: its license, its author and
the address of a ~800 px copy. This script picks up to `max_per_category` photos per
category and downloads those copies into data/commons/images/.

Run it only after commons_list.py has printed "Done": the pick depends on the full list.

WHICH PHOTOS: a seeded random pick per category, not the first 200 alphabetically. Sorted
by name, files starting with "1955 ..." or "2019 ..." would crowd the start of the list, so
the pick would lean toward some years. A random pick spreads over the whole category, and
seeding it with the category's name gives the SAME pick every run (needed for resuming).

WHERE THEY GO: each photo is named by a hash of its Commons title (local_name() below),
e.g. "File:2018 Toyota Camry.jpg" -> images/3f/3f9a1c...jpg. Hashing gives short, safe
names (titles can hold characters Windows doesn't allow), and the two-letter subfolders
keep ~120k files from landing in one folder, which makes Explorer and some tools crawl.
Later steps find a photo's file by calling local_name(title) again; no lookup table needed.

SPEED: each photo is small (~150 KB), so the time goes to waiting, not transferring. Two
tricks cut the wait (~0.7 s per photo -> see the measured rate it prints):
  - KEEP-ALIVE: each worker opens ONE connection to Wikimedia's image server and reuses it,
    instead of a new secure connection (several round trips of handshake) per photo.
  - A FEW WORKERS: 4 threads download at once, so while one waits for an answer the
    others are working. Kept at 4 on purpose: more would be hammering a non-profit's
    servers, and Wikimedia answers "429 Too Many Requests" or blocks scripts.

SLOWING DOWN TOGETHER: when Wikimedia answers 429 to ANY worker, ALL workers pause (see
pause_everyone()). If each worker only backed off on its own, the other three would keep
asking during its wait, Wikimedia would keep saying no, and the 429s would never stop.
Each 429 that comes soon after the last one doubles the pause (30 s, 60 s, ... up to 10
min), and a long quiet spell resets it. A photo that still fails after many tries is
skipped and counted as "failed"; the next run tries it again, so one bad photo can't stop
an overnight run.

RESUMING: a photo already on disk is skipped, so stop it any time (Ctrl+C) and run it
again. Each download is written to a ".part" file and renamed only once it's complete, so
a stop halfway through a file never leaves a broken image that looks finished.

Usage (from the project root):
    python training/commons_download.py
"""

import csv
import hashlib
import http.client
import random
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import load_config, project_path
from wikimedia_check import USER_AGENT

WORKERS = 4

# Each thread keeps its own connection here. threading.local() gives every thread its own
# separate copy of the attributes, like a Java ThreadLocal: connections can't be shared,
# because one connection carries one request at a time.
_thread = threading.local()

# The shared "everyone wait" state. All workers read and write it, so a Lock guards it
# (like a Java synchronized block): without one, two workers could update it at once.
_pause_lock = threading.Lock()
_pause_until = 0.0     # time.time() when workers may send again
_last_429 = 0.0        # when the last 429 came, to tell a burst from a one-off
_pause_seconds = 30    # the next pause's length; doubles while 429s keep coming
FIRST_PAUSE = 30
LONGEST_PAUSE = 600
QUIET_RESET = 900      # 15 min with no 429 -> start again from FIRST_PAUSE
TRIES = 10             # per photo, before it's skipped until the next run


class TooManyRequests(ConnectionError):
    """Wikimedia said 429. retry_after = the seconds it asked us to wait (or None)."""
    def __init__(self, retry_after):
        super().__init__("HTTP 429")
        self.retry_after = retry_after


def pause_everyone(retry_after):
    """A 429 arrived: make ALL workers wait before their next request."""
    global _pause_until, _last_429, _pause_seconds
    with _pause_lock:
        now = time.time()
        if now < _pause_until:
            return  # already pausing: this request was sent before the pause began
        if now - _last_429 > QUIET_RESET:
            _pause_seconds = FIRST_PAUSE  # a long time since the last one: start small
        wait = max(_pause_seconds, retry_after or 0)
        _pause_until = now + wait
        _last_429 = now
        _pause_seconds = min(_pause_seconds * 2, LONGEST_PAUSE)
        print(f"    Wikimedia says slow down (429): all workers pause {wait:.0f} s")


def wait_while_paused():
    """Sleep until the shared pause (if any) is over. Called before every request."""
    while True:
        with _pause_lock:
            left = _pause_until - time.time()
        if left <= 0:
            return
        time.sleep(left)


def local_name(title, url):
    """Where a photo lives under images/: a hash of its title, plus the file type from its URL."""
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:20]
    extension = ".png" if url.lower().endswith(".png") else ".jpg"
    return f"{digest[:2]}/{digest}{extension}"


def pick_files(rows, max_per_category):
    """Choose up to max_per_category rows per category; return them (each photo once).

    A photo in two categories can be picked by both. It's only downloaded once, but it
    counts toward both caps, so popular shared photos don't use up extra space.
    """
    by_category = {}
    for row in rows:
        for category in row["categories"].split(";"):
            by_category.setdefault(category, []).append(row)

    picked = {}
    for category, members in sorted(by_category.items()):
        members = sorted(members, key=lambda r: r["title"])  # same starting order every run
        random.Random(category).shuffle(members)              # a string seed is repeatable
        for row in members[:max_per_category]:
            picked[row["title"]] = row
    return list(picked.values())


def download(url, path):
    """Fetch one photo into path over this thread's open connection.

    Returns False if Commons no longer has the photo. Trouble worth retrying (network
    errors, 5xx server errors) raises ConnectionError instead, and 429 "Too Many Requests"
    raises TooManyRequests; fetch() answers both by waiting and trying again.
    """
    parts = urllib.parse.urlsplit(url)  # "https://upload.wikimedia.org/a/b.jpg" -> host + path
    if getattr(_thread, "host", None) != parts.netloc:
        _thread.connection = http.client.HTTPSConnection(parts.netloc, timeout=60)
        _thread.host = parts.netloc
    try:
        _thread.connection.request("GET", parts.path, headers={"User-Agent": USER_AGENT})
        response = _thread.connection.getresponse()
        data = response.read()  # read it all, or the connection can't be reused
    except (http.client.HTTPException, OSError) as error:
        _thread.connection.close()  # broken: the next request reconnects by itself
        raise ConnectionError(error)

    if response.status == 429:
        retry_after = response.getheader("Retry-After", "")
        raise TooManyRequests(int(retry_after) if retry_after.isdigit() else None)
    if response.status >= 500:
        raise ConnectionError(f"HTTP {response.status}")
    if response.status != 200 or not response.getheader("Content-Type", "").startswith("image/"):
        return False  # 404 = deleted since it was listed; anything else isn't a photo
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    part.replace(path)  # the rename is what marks the photo as finished
    return True


def main():
    config = load_config()
    settings = config["commons"]
    commons_dir = project_path(config["paths"]["commons_dir"])
    images_dir = commons_dir / "images"

    with open(commons_dir / "manifest.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    picked = pick_files(rows, settings["max_per_category"])
    todo = [r for r in picked if not (images_dir / local_name(r["title"], r["thumb_url"])).exists()]
    print(f"Manifest: {len(rows):,} usable photos. Picked {len(picked):,} "
          f"(max {settings['max_per_category']} per category); {len(picked) - len(todo):,} "
          f"already downloaded, {len(todo):,} to go")

    done, gone, failed = download_all(todo, images_dir)
    print(f"\nDone: {done:,} downloaded, {gone:,} no longer on Commons. Photos in {images_dir}")
    if failed:
        print(f"{failed:,} kept failing and were skipped: run this again later to fetch them.")


def fetch(row, images_dir):
    """One worker's job: download one photo. Runs on a worker thread.

    Returns True (downloaded), False (gone from Commons) or None (kept failing; skipped
    until the next run). A 429 pauses every worker; other trouble (a dropped connection, a
    5xx server error) only makes this worker wait, longer each time (5, 10, 20 ... 120 s).
    """
    path = images_dir / local_name(row["title"], row["thumb_url"])
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(TRIES):
        wait_while_paused()
        try:
            found = download(row["thumb_url"], path)
            time.sleep(0.1)  # be polite: a short pause between each worker's downloads
            return found
        except TooManyRequests as error:
            pause_everyone(error.retry_after)
        except ConnectionError:
            time.sleep(min(5 * 2 ** attempt, 120))
    return None


def download_all(rows, images_dir):
    """Download rows with WORKERS threads; print progress. Returns (downloaded, gone, failed).

    ThreadPoolExecutor is Python's version of Java's ExecutorService: submit() hands a job
    to the pool and returns a Future; as_completed() hands back each Future as it finishes.
    Only this main thread prints and counts, so the counters need no locking.
    """
    done = gone = failed = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(fetch, row, images_dir) for row in rows]
        try:
            for future in as_completed(futures):
                result = future.result()  # result() also re-raises a worker's error here
                if result is None:
                    failed += 1
                elif result:
                    done += 1
                else:
                    gone += 1
                finished = done + gone + failed
                if finished % 500 == 0:
                    rate = finished / (time.time() - started)
                    hours_left = (len(rows) - finished) / rate / 3600
                    print(f"  {finished:,}/{len(rows):,}  {rate:.1f} photos/s  "
                          f"({gone} gone from Commons, {failed} failed)  ~{hours_left:.1f} h left")
        except KeyboardInterrupt:
            # Ctrl+C: drop the queued jobs so we don't wait for 100k of them to run.
            print("\nStopping after the downloads in progress (run again to resume)...")
            pool.shutdown(cancel_futures=True)
            raise
    return done, gone, failed


if __name__ == "__main__":
    main()
