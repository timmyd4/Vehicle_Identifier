"""
Check how many car photos Wikimedia Commons has for some models, before downloading anything.

Wikimedia Commons (commons.wikimedia.org) is a library of freely licensed images. It sorts
files into categories, and car categories are usually split by GENERATION, e.g.

    Category:Toyota Highlander
        Category:Toyota Highlander (XU50)          <- 3rd generation, 164 files
            Category:Toyota Highlander Hybrid (XU50)
            Category:Toyota Highlander (XU50) taxis
        ...

That's close to the labels we want (make + model + generation), so this script walks the
category tree for each model and reports, per generation category:
  - how many unique image files it has (including its subcategories)
  - how many filenames contain a model year (e.g. "2018 Toyota Highlander XLE.jpg"),
    which is how we could split a generation into pre-facelift and facelift

This script only reads category listings through the official API. It downloads no images.
Not every file is a usable photo of the outside of the car (some are interiors, engines,
dashboards). A later step will run our car detector on the images to count real exteriors.

Usage (from the project root):
    python training/wikimedia_check.py
    python training/wikimedia_check.py "Toyota Highlander" "Honda Civic"
    python training/wikimedia_check.py --top 50     # the 50 models with the most VMMRdb images
"""

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

import pandas as pd

from common import project_path

API_URL = "https://commons.wikimedia.org/w/api.php"

# Wikimedia asks every script to send a User-Agent that says what it is and how to reach
# the owner (https://meta.wikimedia.org/wiki/User-Agent_policy). Scripts without one can be blocked.
USER_AGENT = "VehicleIdentifierDataCheck/0.1 (https://github.com/timmyd4/Vehicle_Identifier)"

# Models to check when none are given on the command line.
DEFAULT_MODELS = ["Toyota Highlander", "Toyota Corolla", "Honda Civic"]

# How deep to follow subcategories below the model's category:
# depth 1 = generations, 2 = things inside a generation (hybrid, taxis, ...).
MAX_DEPTH = 3

# Subcategories with these words are about something other than the outside of the car,
# so we don't follow them. Careful with short words: "toy" would also match "Toyota".
SKIP_WORDS = ["interior", "engine", "dashboard", "model car", "scale model", "toy car",
              "logo", "advertis"]

# Years in filenames: a full year like "2018", or a short one at the start like "'17" / "17-19".
FULL_YEAR = re.compile(r"(?<!\d)(19[5-9]\d|20[0-2]\d)(?!\d)")
SHORT_YEAR_AT_START = re.compile(r"^File:'?(\d{2})(?=[\s\-'])")


def api_get(api_url=API_URL, **params):
    """
    Call a Wikimedia API once and return the decoded JSON.

    Commons, Wikipedia and Wikidata all run the same software, so the same call works for all
    three; only the address changes (check_generations.py passes Wikipedia's and Wikidata's).
    """
    params.update(format="json", maxlag=5)  # maxlag: back off when Wikimedia's servers are busy
    query = urllib.parse.urlencode(params)
    if len(query) < 4000:
        # GET: the parameters ride in the address. Preferred, because answers can be cached.
        request = urllib.request.Request(api_url + "?" + query, headers={"User-Agent": USER_AGENT})
    else:
        # POST: the parameters go in the request body, which has no length limit. Needed
        # when 50 long titles (accents become %C3%A9...) make the address too long (HTTP 414).
        request = urllib.request.Request(api_url, data=query.encode("ascii"),
                                         headers={"User-Agent": USER_AGENT})
    for _ in range(10):
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)
        time.sleep(0.2)  # be polite: one request at a time, with a short pause
        # When the servers are busy, the answer is an error saying "maxlag" instead of data.
        # The polite reaction is to wait and ask again, not to give up or retry at once.
        if data.get("error", {}).get("code") != "maxlag":
            return data
        print("    Wikimedia is busy (maxlag); waiting 10 s")
        time.sleep(10)
    raise urllib.error.URLError("Wikimedia stayed busy (maxlag)")


def category_members(category, member_type):
    """
    Return every member title of a category ("file" or "subcat").

    The API returns at most 500 results per call. When there are more, it includes a
    "continue" token, and we call again with that token to get the next page.
    """
    titles = []
    params = dict(action="query", list="categorymembers", cmtitle=category,
                  cmtype=member_type, cmlimit=500)
    while True:
        data = api_get(**params)
        titles += [m["title"] for m in data["query"]["categorymembers"]]
        if "continue" not in data:
            return titles
        params.update(data["continue"])


def files_in_tree(category, depth):
    """Return the set of file titles in a category and its subcategories (down to MAX_DEPTH)."""
    files = set(category_members(category, "file"))
    if depth < MAX_DEPTH:
        for sub in category_members(category, "subcat"):
            if not any(word in sub.lower() for word in SKIP_WORDS):
                files |= files_in_tree(sub, depth + 1)
    # A set, because the same photo is often in several categories and should count once.
    return files


def year_from_filename(title):
    """Return the model year written in a filename, or None if there isn't one."""
    match = FULL_YEAR.search(title)
    if match:
        return int(match.group(1))
    match = SHORT_YEAR_AT_START.search(title)
    if match:
        # '17 -> 2017, '98 -> 1998
        short = int(match.group(1))
        return 2000 + short if short < 50 else 1900 + short
    return None


def describe_years(files):
    """Turn filenames into a short year summary like '2014:12 2015:3 ...'."""
    years = Counter(y for y in map(year_from_filename, files) if y)
    return " ".join(f"{y}:{n}" for y, n in sorted(years.items())), sum(years.values())


def check_model(model):
    top = f"Category:{model}"
    print(f"\n=== {model} ===")
    subcats = category_members(top, "subcat")
    if not subcats and not category_members(top, "file"):
        print("  category not found (Commons may spell this model differently)")
        return None

    # Generation categories usually look like "Toyota Highlander (XU50)" or
    # "Honda Civic (tenth generation)": the model name followed by something in brackets.
    generations = [c for c in subcats if c.startswith(f"{top} (")]
    others = [c for c in subcats if c not in generations]

    total = set()
    sizes = []
    for gen in generations:
        files = files_in_tree(gen, depth=1)
        total |= files
        sizes.append(len(files))
        years, with_year = describe_years(files)
        name = gen.removeprefix(top).strip()
        print(f"  {name:32} {len(files):5} files  ({with_year} with a year)  {years}")

    # Files sitting directly in the model's category, or under other subcategories (by
    # country, by color, ...), aren't sorted into a generation yet. Report how many there are.
    print(f"  {'generation categories total':32} {len(total):5} files")
    print(f"  other subcategories not counted: {len(others)}"
          + (f" (e.g. {', '.join(o.removeprefix('Category:') for o in others[:3])})" if others else ""))
    return sizes, len(total)


def biggest_vmmrdb_models(count):
    """
    Return the display names ("Toyota Corolla") of the `count` make+models with the most
    images in our training subset.

    classes.csv has one row per class (make + model + year bucket) with its image count, so we
    add up the buckets per make+model. labels.json has the nicely capitalized names
    ("Toyota Corolla (2007-2009)"); we cut off the year part.
    """
    classes = pd.read_csv(project_path("data/splits/classes.csv"))
    labels = json.loads(project_path("models/labels.json").read_text(encoding="utf-8"))
    names = {(l["make"], l["model"]): l["display_name"].rsplit(" (", 1)[0] for l in labels}
    totals = classes.groupby(["make", "model"])["num_images"].sum().nlargest(count)
    return [names[key] for key in totals.index]


def main():
    args = sys.argv[1:]
    if args[:1] == ["--top"]:
        models = biggest_vmmrdb_models(int(args[1]))
    else:
        models = args or DEFAULT_MODELS

    summary = {model: check_model(model) for model in models}

    # One line per model, so a long run is easy to read at the end.
    # "gens >=100" = generation categories with at least 100 files, a rough "enough to train on".
    print(f"\n{'model':30} {'gens':>5} {'gens >=100':>11} {'files':>7}")
    for model, result in summary.items():
        if result is None:
            print(f"{model:30} {'not found':>25}")
        else:
            sizes, total = result
            print(f"{model:30} {len(sizes):5} {sum(s >= 100 for s in sizes):11} {total:7}")


if __name__ == "__main__":
    main()
