"""
Step 1 of the Commons pipeline: list every file in our Commons categories, with its license
and author. Downloads no images.

THE BIG PICTURE
---------------
generations.csv names a Commons category for every look ("Toyota Highlander (XU50)").
This script asks the Commons API two things:

  1. Which files are in each category (and its subcategories, like "... Hybrid (XU50)")?
  2. For each file: how big is it, what license, who made it, and where is an 800 px copy?

and writes the answers to data/commons/manifest.csv, one row per file. That manifest is
the source of truth for everything after it: commons_download.py fetches the thumbnails
it lists, and if you ever publish the model, it's the record of where the photos came
from and whom to credit. (CC BY and CC BY-SA licenses REQUIRE crediting the author.)

RESUMING: both steps save as they go. Stop it any time (Ctrl+C) and run it again: finished
categories are read from categories.json and finished files from manifest.csv, so it
carries on where it stopped instead of starting over.

BEING POLITE: Commons is run by a non-profit. api_get() sends a User-Agent with the repo
URL, asks one thing at a time with a short pause, and passes maxlag so Wikimedia can tell
us to back off when its servers are busy.

Usage (from the project root):
    python training/commons_list.py
"""

import csv
import html
import json
import re
import sys
import time
import urllib.error

from common import load_config, load_generations, project_path
from wikimedia_check import api_get, files_in_tree

# The manifest's columns, in order.
COLUMNS = ["title", "categories", "thumb_url", "width", "height", "mime",
           "license", "license_url", "artist", "uploader", "uploaded", "taken"]


# ---------------------------------------------------------------------------
# Step 1: which files are in each category
# ---------------------------------------------------------------------------
def list_categories(categories, cache_path):
    """Return {category: [file titles]}, reading finished categories from the cache.

    files_in_tree(category, depth=1) also walks subcategories two levels down, skipping
    ones about interiors, engines, toy cars and so on (see wikimedia_check.SKIP_WORDS).
    """
    done = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = [c for c in categories if c not in done]
    print(f"Categories: {len(categories)} total, {len(done)} already listed, {len(todo)} to go")

    for i, category in enumerate(todo, 1):
        files = sorted(with_retries(files_in_tree, f"Category:{category}", 1))
        done[category] = files
        print(f"  [{i}/{len(todo)}] {category}: {len(files)} files")
        # Save after EVERY category: a crash or Ctrl+C loses at most one category's work.
        cache_path.write_text(json.dumps(done, indent=0), encoding="utf-8")
    return done


# ---------------------------------------------------------------------------
# Step 2: details for each file
# ---------------------------------------------------------------------------
def file_details(titles, thumb_width):
    """Ask Commons about up to 50 files at once and return one manifest row per file.

    prop=imageinfo gives size, type, uploader and URLs. iiurlwidth asks Commons to make an
    800 px wide copy and tell us its address (thumburl). extmetadata holds the license and
    author, already extracted from the file's description page.
    """
    data = with_retries(
        api_get, action="query", prop="imageinfo", titles="|".join(titles),
        iiprop="url|size|mime|user|timestamp|extmetadata", iiurlwidth=thumb_width,
        iiextmetadatafilter="License|LicenseUrl|Artist|DateTimeOriginal")
    rows = []
    for page in data["query"]["pages"].values():
        if "imageinfo" not in page:
            continue  # the file was deleted since the category was listed
        info = page["imageinfo"][0]
        # A file with no metadata comes back as [] instead of {} (PHP, which runs Commons,
        # has one type for both). `or {}` turns an empty list into an empty dict. Such a
        # file has no license, so usable() skips it.
        meta = info.get("extmetadata") or {}
        rows.append({
            "title": page["title"],
            "thumb_url": info.get("thumburl", info["url"]),  # small files have no thumbnail
            "width": info["width"],
            "height": info["height"],
            "mime": info["mime"],
            "license": meta.get("License", {}).get("value", "").lower(),
            "license_url": meta.get("LicenseUrl", {}).get("value", ""),
            "artist": plain_text(meta.get("Artist", {}).get("value", "")),
            "uploader": info.get("user", ""),
            "uploaded": info.get("timestamp", ""),
            "taken": plain_text(meta.get("DateTimeOriginal", {}).get("value", "")),
        })
    return rows


def plain_text(value):
    """The author field is HTML (often a link to their user page): keep only the text."""
    text = re.sub(r"<[^>]+>", "", value)    # drop tags: <a href="...">Name</a> -> Name
    return " ".join(html.unescape(text).split())  # &amp; -> &, and tidy the spaces


def usable(row, settings):
    """Keep photos only: JPEG/PNG, big enough, with a license we accept."""
    return (row["mime"] in ("image/jpeg", "image/png")
            and row["width"] >= settings["min_width"]
            and row["license"].startswith(tuple(settings["allowed_licenses"])))


def with_retries(function, *args, **kwargs):
    """Call function; on a network error, wait and try again (up to 5 times).

    Waiting longer each time (5, 10, 20, 40 s) is called EXPONENTIAL BACKOFF: if the server
    is overloaded, hammering it again right away only makes things worse. ("Server busy"
    answers are already handled inside api_get.)
    """
    for attempt in range(5):
        try:
            return function(*args, **kwargs)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            # HTTP 4xx means "your request is wrong": asking again can't help, so stop.
            # Except 429 "Too Many Requests", where waiting is exactly the cure.
            code = getattr(error, "code", None)
            if code and 400 <= code < 500 and code != 429:
                raise
            wait = 5 * 2 ** attempt
            print(f"    {error}; retrying in {wait} s")
            time.sleep(wait)
    sys.exit("Commons kept failing; run again later (finished work is saved).")


def main():
    config = load_config()
    settings = config["commons"]
    out_dir = project_path(config["paths"]["commons_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Every distinct category in the table. One look can name several, separated by ";".
    table = load_generations()
    categories = sorted({c for cats in table["commons_category"] for c in cats.split(";") if c})
    files_by_category = list_categories(categories, out_dir / "categories.json")

    # The same photo often sits in several categories: remember all of them for each file,
    # so later steps know which looks it could belong to.
    categories_of = {}
    for category, titles in files_by_category.items():
        for title in titles:
            categories_of.setdefault(title, []).append(category)

    manifest = out_dir / "manifest.csv"
    seen = set()
    if manifest.exists():
        with open(manifest, encoding="utf-8", newline="") as f:
            seen = {row["title"] for row in csv.DictReader(f)}
    todo = sorted(t for t in categories_of if t not in seen)
    print(f"\nFiles: {len(categories_of):,} unique, {len(seen):,} already checked, {len(todo):,} to go")

    # "a" = append. newline="" is what the csv module wants on Windows (no blank lines).
    kept = skipped = 0
    with open(manifest, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not seen:
            writer.writeheader()
        for start in range(0, len(todo), 50):
            for row in file_details(todo[start:start + 50], settings["thumb_width"]):
                seen.add(row["title"])
                if usable(row, settings):
                    row["categories"] = ";".join(categories_of[row["title"]])
                    writer.writerow(row)
                    kept += 1
                else:
                    skipped += 1
            f.flush()  # write to disk now, so a stop loses nothing
            if start % 5000 == 0:
                print(f"  {start + 50:,}/{len(todo):,} checked: {kept:,} kept, {skipped:,} skipped")

    # Skipped files aren't written to the manifest, so a rerun checks them again. That's
    # only a few extra API calls, and it keeps the manifest a clean list of usable photos.
    print(f"\nDone: {kept:,} usable files added ({skipped:,} skipped: wrong type, too small, "
          f"or a license we don't accept). Manifest: {manifest}")


if __name__ == "__main__":
    main()
