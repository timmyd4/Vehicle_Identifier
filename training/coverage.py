"""
Coverage report: which car models sold in the US does our label table NOT know yet?

THE BIG PICTURE
---------------
"Know every car" needs a definition, or we never know when we're done. Ours: every model
the US government's vehicle database (NHTSA vPIC) lists for the brands below, model years
FIRST_YEAR to LAST_YEAR. vPIC is free and needs no key; it's what VIN decoders use.

The script:
  1. downloads the model list per brand and year (cached in data/coverage/nhtsa_models.csv,
     so a second run is instant; delete the file to refresh)
  2. compares every model with generations.csv (and the VMMRdb folder names that map to it)
  3. writes data/coverage/report.csv: each model, its years, "covered" or "missing", and how
     many VMMRdb photos exist for it (photos decide whether a missing model CAN be learned)

The comparison is by name, so it's approximate: "F-150" and our "f150" match after removing
spaces and dashes, but vPIC sometimes names trims ("328i") or uses odd names. The report is
for a human (or me) to read, not something training reads.

Usage (from the project root):
    python training/coverage.py
"""

import json
import os
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

from common import car_model, load_generations, load_merges, project_path
from wikimedia_check import USER_AGENT

FIRST_YEAR, LAST_YEAR = 1990, 2026
API = "https://vpic.nhtsa.dot.gov/api/vehicles/GetModelsForMakeYear/make/{make}/modelyear/{year}?format=json"

# Our make name (as in generations.csv / VMMRdb folders) -> vPIC's make name.
# The 40 makes we have, plus brands sold in the US that VMMRdb has (almost) no photos of.
MAKES = {
    "acura": "acura", "audi": "audi", "bmw": "bmw", "buick": "buick", "cadillac": "cadillac",
    "chevrolet": "chevrolet", "chrysler": "chrysler", "dodge": "dodge", "ford": "ford", "gmc": "gmc",
    "honda": "honda", "hummer": "hummer", "hyundai": "hyundai", "infiniti": "infiniti",
    "isuzu": "isuzu", "jaguar": "jaguar", "jeep": "jeep", "kia": "kia", "landrover": "land rover",
    "lexus": "lexus", "lincoln": "lincoln", "mazda": "mazda", "mercedes benz": "mercedes-benz",
    "mercury": "mercury", "mini": "mini", "mitsubishi": "mitsubishi", "nissan": "nissan",
    "oldsmobile": "oldsmobile", "plymouth": "plymouth", "pontiac": "pontiac", "porsche": "porsche",
    "saab": "saab", "saturn": "saturn", "scion": "scion", "smart": "smart", "subaru": "subaru",
    "suzuki": "suzuki", "toyota": "toyota", "volkswagen": "volkswagen", "volvo": "volvo",
    # Not in our table yet:
    "tesla": "tesla", "genesis": "genesis", "rivian": "rivian", "polestar": "polestar",
    "lucid": "lucid", "ram": "ram", "fiat": "fiat", "alfa romeo": "alfa romeo",
    "maserati": "maserati", "geo": "geo", "eagle": "eagle", "daewoo": "daewoo",
}


def normalize(name):
    """"F-150" -> "f150", "3 Series" -> "3series": compare names without case, spaces or dashes."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def fetch_models(cache):
    """Every (make, model, year) vPIC lists. Resumable: makes already in the cache are skipped."""
    rows = pd.read_csv(cache).to_dict("records") if cache.exists() else []
    done = {r["make"] for r in rows}
    for make, vpic_make in MAKES.items():
        if make in done:
            continue
        print(f"  {make}...")
        for year in range(FIRST_YEAR, LAST_YEAR + 1):
            url = API.format(make=urllib.parse.quote(vpic_make), year=year)
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                results = json.load(response)["Results"]
            rows += [{"make": make, "model": r["Model_Name"].strip(), "year": year} for r in results]
            time.sleep(0.2)  # be polite: one request at a time
        # Save after every make, so an interrupted run picks up where it stopped.
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(cache, index=False)
    return pd.DataFrame(rows)


def fetch_road_models(cache):
    """{(make, model)} that vPIC files as a car, SUV/van ("mpv") or pickup ("truck").

    WHY: the per-year list also has every motorcycle (Honda alone: 305), bus and bare
    chassis the brand makes. Asking per vehicle type (all years at once) keeps cars only.
    """
    if not cache.exists():
        rows = []
        for make, vpic_make in MAKES.items():
            for vehicle_type in ["car", "mpv", "truck"]:
                url = (f"https://vpic.nhtsa.dot.gov/api/vehicles/GetModelsForMakeYear/make/"
                       f"{urllib.parse.quote(vpic_make)}/vehicletype/{vehicle_type}?format=json")
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=60) as response:
                    rows += [{"make": make, "model": r["Model_Name"].strip(), "type": vehicle_type}
                             for r in json.load(response)["Results"]]
                time.sleep(0.2)
        pd.DataFrame(rows).to_csv(cache, index=False)
    types = pd.read_csv(cache)
    return set(zip(types["make"], types["model"]))


def vmmrdb_photos(dataset_dir):
    """{(make, normalized folder model): photo count}, e.g. ("subaru", "brz"): 62."""
    counts = {}
    for folder in os.listdir(dataset_dir):
        parts = folder.split("_")
        key = (parts[0], normalize("_".join(parts[1:-1])))
        counts[key] = counts.get(key, 0) + len(os.listdir(dataset_dir / folder))
    return counts


def known_names(dataset_dir):
    """{(make, normalized name)} for every name we already cover: the table's models, plus
    every VMMRdb folder model that maps onto one (so "is350" counts once IS350 -> IS is merged)."""
    table = load_generations()
    models_of_make = table.groupby("make")["model"].apply(list).to_dict()
    merges = load_merges()
    known = {(make, normalize(model)) for make, model in zip(table["make"], table["model"])}
    for folder in os.listdir(dataset_dir):
        parts = folder.split("_")
        make, model = parts[0], "_".join(parts[1:-1])
        if car_model(make, model, models_of_make.get(make, []), merges):
            known.add((make, normalize(model)))
    return known


def is_covered(make, model, known):
    """Same normalized name, or one starts with the other ("mx5" vs our "mx5miata")."""
    name = normalize(model)
    return any(m == make and (k == name or (min(len(k), len(name)) >= 3 and
                                            (k.startswith(name) or name.startswith(k))))
               for m, k in known)


def main():
    out_dir = project_path("data/coverage")
    dataset_dir = project_path("data/VMMRdb")
    print("Model lists from NHTSA vPIC:")
    models = fetch_models(out_dir / "nhtsa_models.csv")
    road = fetch_road_models(out_dir / "nhtsa_types.csv")
    keep = [(make, model) in road for make, model in zip(models["make"], models["model"])]
    print(f"  {len(models):,} (make, model, year) rows; {sum(keep):,} are cars, SUVs, vans or pickups")
    models = models[keep]

    # One row per (make, model), with the years vPIC lists it.
    report = (models.groupby(["make", "model"])["year"].agg(first_year="min", last_year="max", years="nunique")
              .reset_index())
    known = known_names(dataset_dir)
    photos = vmmrdb_photos(dataset_dir)
    report["status"] = ["covered" if is_covered(make, model, known) else "missing"
                        for make, model in zip(report["make"], report["model"])]
    report["vmmrdb_photos"] = [photos.get((make, normalize(model)), 0)
                               for make, model in zip(report["make"], report["model"])]
    report = report.sort_values(["status", "make", "model"])
    report.to_csv(out_dir / "report.csv", index=False)

    covered = (report["status"] == "covered").sum()
    print(f"\nCovered {covered:,} of {len(report):,} US models ({covered / len(report):.0%}), "
          f"model years {FIRST_YEAR}-{LAST_YEAR}")
    missing = report[report["status"] == "missing"]
    print(f"Missing per make (most first):")
    for make, n in missing["make"].value_counts().head(20).items():
        print(f"  {make:15} {n:4}")
    print(f"Saved {out_dir / 'report.csv'}")


if __name__ == "__main__":
    main()
