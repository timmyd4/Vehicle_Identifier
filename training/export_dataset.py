"""
Package ALL the downloaded Wikimedia Commons photos as a public dataset for Hugging Face.

THE BIG PICTURE
---------------
VMMRdb is easy to learn from because it's just folders: honda_civic_2005/ full of photos.
This script builds the same kind of thing from our Commons photos, so other people
learning ML can use them:

    data/hf_dataset/
        metadata.csv              one row per photo: labels, filter result, author, license
        images/Toyota.zip         inside: Toyota/Corolla/E140 (2007-2013)/<photo>.jpg
        images/Ford.zip
        ...
        images/_Multiple cars.zip photos whose Commons categories name different cars

EVERY photo we downloaded goes in, even the ones the filter or the pre-sort rejected.
metadata.csv says what happened to each one (kept or dropped by the filter, and why), so a
learner can practice cleaning data, or just use the photos we kept.

WHICH FOLDER a photo goes in (Make/Model/Generation): its Commons category, through
generations.csv, names the car. Most categories cover one generation ("Toyota Corolla
(E140)"). Some cover a whole model ("Chevrolet El Camino") -> Model/_Unknown generation.
A photo in categories of DIFFERENT cars (a Jeep next to a Hummer) -> _Multiple cars.
If the pre-sort labeled the photo (or a human did), its answer picks the generation.

The facelift (pre / facelift) is a column, not a folder: we only know it for photos the
pre-sort could label, and an extra folder level that's mostly "unknown" would be clutter.

WHY ZIPS and not 222k loose files: Hugging Face recommends under ~100k files per repo, and
uploading/downloading one file per photo is very slow. Unzipping gives the folders back.
Zips are "stored" (not compressed): JPEGs are already compressed, so zipping them again
would only cost time.

LICENSES: every photo keeps its own license (CC0, public domain, CC BY, CC BY-SA). CC BY
and CC BY-SA require crediting the author, so metadata.csv has author, license and a link
to the photo's Commons page for every row. Never share the photos without that file.

Usage (from the project root):
    python training/export_dataset.py --dry-run   # only metadata.csv + counts (seconds)
    python training/export_dataset.py             # also the zips (~40 GB, a while)
"""

import argparse
import json
import re
import urllib.parse
import zipfile

import pandas as pd

from common import load_config, load_generations, project_path
from commons_presort import load_looks
from inference import format_model_word, make_name

# The status the pre-sort gave a photo -> what metadata.csv calls that label.
LOOK_SOURCES = {"category": "auto: category", "year": "auto: year in filename",
                "model": "auto: model prediction"}


def model_text(model):
    """"corolla" -> "Corolla", "rav4" -> "RAV4" (same rule as the app's names)."""
    return " ".join(format_model_word(w) for w in model.split())


def safe(name):
    """Make a name usable as a folder on Windows too (no / \\ : * ? " < > |)."""
    return re.sub(r'[\\/:*?"<>|]', "-", name).strip()


def commons_url(title):
    """"File:My car.jpg" -> its page on Commons (where the author and license are shown)."""
    return "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))


def load_labels(commons_dir):
    """The pre-sort's answer per photo, with the human answers from commons_looks.py on top.

    Same rule as build_subset.py: a look a human picked replaces the pre-sort's answer.
    """
    presort = pd.read_csv(commons_dir / "presort.csv", keep_default_na=False)
    answers_file = commons_dir / "look_answers.json"
    answers = json.loads(answers_file.read_text(encoding="utf-8")) if answers_file.exists() else {}
    picked = {title: look for title, look in answers.items() if look is not None}
    is_picked = presort["title"].isin(picked)
    if is_picked.any():
        presort.loc[is_picked, ["make", "model", "generation", "phase"]] = [
            picked[t] for t in presort.loc[is_picked, "title"]]
        presort.loc[is_picked, "status"] = "human"
    return presort.set_index("title")


def run6_splits():
    """{photo file: "train"/"val"/"test"} for the Commons photos run 6 trained and tested on."""
    split_of = {}
    for split in ["train", "val", "test"]:
        df = pd.read_csv(project_path("data/splits_run6_newcars") / f"{split}.csv",
                         usecols=["path", "source"])
        commons = df[df["source"] == "commons"]
        split_of.update({p.split("images/", 1)[1]: split for p in commons["path"]})
    return split_of


def label_photo(categories, sorted_row, looks_of_category, generation_years):
    """Decide one photo's folder and label columns. Returns a dict."""
    # Every look the photo's categories could mean, as (make, model, generation, phase).
    candidates = list(dict.fromkeys(
        key for c in categories.split(";") for key in looks_of_category.get(c, [])))
    cars = {k[:2] for k in candidates}
    generations = {k[:3] for k in candidates}

    label = dict(look_label="", phase="")
    if sorted_row is not None and sorted_row["status"] == "review":
        label["look_label"] = "needs review: " + sorted_row["reason"]
    # 1. The pre-sort (or a human) named the look: trust it for the generation AND facelift.
    if sorted_row is not None and sorted_row["status"] in ("auto", "human"):
        make, model, generation = sorted_row["make"], sorted_row["model"], sorted_row["generation"]
        label.update(phase=sorted_row["phase"], look_label="human" if sorted_row["status"] == "human"
                     else LOOK_SOURCES.get(sorted_row["reason"], "auto"))
    # 2. Otherwise the category alone: one generation, one model, or several cars.
    elif len(cars) > 1:
        return dict(label, folder="_Multiple cars", make="", model="", generation="", years="")
    elif len(generations) == 1:
        make, model, generation = candidates[0][:3]
        # A generation with only one look (no facelift) tells us the phase too.
        if len(candidates) == 1:
            label["phase"] = candidates[0][3]
    else:
        make, model = next(iter(cars))
        generation = ""

    if generation:
        first, last = generation_years[(make, model, generation)]
        years = f"{first}-{last}"
        generation_folder = safe(f"{generation} ({years})")
    else:
        years, generation_folder = "", "_Unknown generation"
    folder = "/".join([safe(make_name(make)), safe(model_text(model)), generation_folder])
    return dict(label, folder=folder, make=make_name(make), model=model_text(model),
                generation=generation, years=years)


def main():
    parser = argparse.ArgumentParser(description="Package the Commons photos as a dataset.")
    parser.add_argument("--dry-run", action="store_true",
                        help="write metadata.csv and print counts, but don't build the zips")
    args = parser.parse_args()

    config = load_config()
    commons_dir = project_path(config["paths"]["commons_dir"])
    images_dir = commons_dir / "images"
    out_dir = project_path("data/hf_dataset")
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    # filtered.csv has one row per DOWNLOADED photo (kept or not), so it's our list of photos.
    photos = pd.read_csv(commons_dir / "filtered.csv", keep_default_na=False)
    manifest = pd.read_csv(commons_dir / "manifest.csv", keep_default_na=False).set_index("title")
    sorted_photos = load_labels(commons_dir)
    looks_of_category, _ = load_looks()
    # Each generation's years = from its first look's first year to its last look's last year.
    table = load_generations()
    spans = table.groupby(["make", "model", "generation"]).agg(first=("first_year", "min"),
                                                                last=("last_year", "max"))
    generation_years = {key: (int(r["first"]), int(r["last"])) for key, r in spans.iterrows()}
    split_of = run6_splits()

    rows = []
    for photo in photos.itertuples():
        info = manifest.loc[photo.title]
        sorted_row = sorted_photos.loc[photo.title] if photo.title in sorted_photos.index else None
        label = label_photo(info["categories"], sorted_row, looks_of_category, generation_years)
        # Inside the zip, the photo keeps its unique hashed name ("6b/6bd84...jpg" -> "6bd84...jpg").
        file_name = f"{label['folder']}/{photo.file.split('/')[-1]}"
        rows.append({
            "file_name": file_name,
            "zip": safe(file_name.split("/")[0]) + ".zip",
            "make": label["make"], "model": label["model"],
            "generation": label["generation"], "years": label["years"], "phase": label["phase"],
            "look_label": label["look_label"],
            "filter": photo.reason,              # "ok" = kept; otherwise why it was dropped
            "detector_score": photo.score,
            "car_box": photo.box,                # left top right bottom, in pixels
            "used_in_run6": split_of.get(photo.file, ""),
            "commons_category": info["categories"],
            "commons_title": photo.title,
            "commons_url": commons_url(photo.title),
            "author": info["artist"] or info["uploader"],
            "license": info["license"],
            "license_url": info["license_url"],
            "date_taken": info["taken"],
            "original_width": info["width"],     # the full photo on Commons (ours is 800 px wide)
            "original_height": info["height"],
            "_source": photo.file,               # where it is on this PC (not saved in the CSV)
        })
    meta = pd.DataFrame(rows)
    meta.drop(columns="_source").to_csv(out_dir / "metadata.csv", index=False)

    # Summary, so the numbers can go straight into the dataset card.
    print(f"Photos: {len(meta):,}  (kept by the filter: {(meta['filter'] == 'ok').sum():,})")
    print(f"Makes: {meta.loc[meta['make'] != '', 'make'].nunique()}, "
          f"models: {meta.loc[meta['make'] != ''].groupby(['make', 'model']).ngroups}, "
          f"generations: {meta.loc[meta['generation'] != ''].groupby(['make', 'model', 'generation']).ngroups}")
    print(f"Unknown generation: {meta['file_name'].str.contains('/_Unknown generation/').sum():,}, "
          f"multiple cars: {meta['file_name'].str.startswith('_Multiple cars/').sum():,}")
    print("Look labels:\n" + meta["look_label"].replace("", "(none)").value_counts().to_string())
    print(f"Wrote {out_dir / 'metadata.csv'}")
    if args.dry_run:
        return

    # One zip per make. Writing a zip again from the start each run keeps it simple: if the
    # script stops halfway, just run it again.
    for zip_name, group in meta.groupby("zip"):
        print(f"  {zip_name}: {len(group):,} photos")
        with zipfile.ZipFile(out_dir / "images" / zip_name, "w", zipfile.ZIP_STORED) as archive:
            for source, file_name in zip(group["_source"], group["file_name"]):
                archive.write(images_dir / source, file_name)
    print(f"Done: {meta['zip'].nunique()} zips in {out_dir / 'images'}")


if __name__ == "__main__":
    main()
