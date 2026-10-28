"""
Step 4 of the Commons pipeline: give every kept photo a label, and decide which ones a
human has to check.

THE BIG PICTURE
---------------
A Commons photo comes with a category ("Toyota Corolla (E140)"), and generations.csv
says which of our LOOKS that category covers: E140 pre-facelift AND E140 facelift.
So the category alone often can't name one look. Three clues can narrow it down:

  1. the CATEGORY:  which looks are possible at all (the "candidates")
  2. the DATE TAKEN: rules out looks newer than the photo (see decide())
  3. the MODEL:     run 2 looks at the photo and says how likely each look is
  4. the YEAR:      many filenames start with the model year ("2010 Corolla LE.jpg")

AGREEMENT FILTERING: when independent clues agree, the label is very likely right, so
we accept it automatically ("auto"). When they disagree, or no clue can decide, a human
looks at it ("review"). Your time goes only to the hard photos.

The rules, in order (the first that applies wins):

  a. the model is fairly sure it's a DIFFERENT car         -> review ("different car?")
     (only asked when the model knows one of the candidate looks, see decide())
     (catches wrong photos the detector let through: a Golf parked next to the Audi)
  b. the category covers only one look                     -> auto (category)
  c. the filename year fits exactly one candidate          -> auto (year)
     ...unless the model strongly prefers another one      -> review ("year vs model")
  d. the model strongly prefers one candidate              -> auto (model)
     (c and d only trust the model about looks it HAS a class for: it can't vote for
      a look it has never learned, so its "preference" against one means nothing)
  e. anything else                                         -> review (e.g. pre or facelift?)

"Strongly prefers" = that look gets at least `auto_share` of the probability the model
gives to all the candidates together.

WHY RUN 2 CAN'T DECIDE EVERYTHING: it only knows the 836 looks VMMRdb had 50+ photos of.
Newer looks (VMMRdb stops at 2016) and rare ones get no probability at all, so for those
the category and the year have to do the work, or a human does.

Writes data/commons/presort.csv. The review page (step 5) reads it.

Usage (from the project root, after commons_filter.py is done):
    python training/commons_presort.py
"""

import csv
import re

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from common import car_model, load_config, load_generations, load_merges, project_path
from detect import crop_to_car, open_photo
from inference import eval_transform, load_trained_model, pick_device
from wikimedia_check import year_from_filename

COLUMNS = ["title", "file", "status", "reason", "make", "model", "generation", "phase",
           "p_car", "p_look", "year", "model_guess", "candidates"]

# "2016-08-24 Car show.jpg" or "20160824_123456.jpg" is the day the photo was TAKEN, not the
# car's model year. Remove dates like these before looking for a year.
DATES = re.compile(r"(19|20)\d\d[-_.]?\d\d[-_.]?\d\d")


def look_key(make, model, generation, phase):
    """One look, the unit we label: ("toyota", "corolla", "E140", "pre")."""
    return (make, model, generation, phase)


def look_text(key):
    make, model, generation, phase = key
    return f"{model}/{generation}/{phase}"


def load_looks():
    """From generations.csv: {category: [look keys]} and {look key: (first_year, last_year)}."""
    looks_of_category, years_of_look = {}, {}
    for row in load_generations().itertuples():
        key = look_key(row.make, row.model, row.generation, row.phase)
        years_of_look[key] = (int(row.first_year), int(row.last_year))
        for category in filter(None, row.commons_category.split(";")):
            looks_of_category.setdefault(category, []).append(key)
    return looks_of_category, years_of_look


def class_looks(class_names, splits_dir):
    """For each of run 2's output classes, the look it belongs to in TODAY's table.

    Run 2 was trained before the merges, so its classes say "bmw_325i" and "bmw_328i".
    car_model() turns both into "3 series", so both point at the same look, and their
    probabilities get added together (the same idea as the app's layered answer).
    """
    classes = pd.read_csv(splits_dir / "classes.csv", keep_default_na=False).set_index("class_name")
    table = load_generations()
    merges = load_merges()
    models_of_make = table.groupby("make")["model"].unique().to_dict()
    keys = []
    for name in class_names:
        row = classes.loc[name]
        model = car_model(row["make"], row["model"], models_of_make.get(row["make"], []), merges)
        keys.append(look_key(row["make"], model, row["generation"], row["phase"]))
    return keys


def model_year(title, candidates, years_of_look):
    """The model year in the filename, if it's a plausible one for these candidates."""
    year = year_from_filename(DATES.sub(" ", title))
    first = min(years_of_look[c][0] for c in candidates)
    last = max(years_of_look[c][1] for c in candidates)
    return year if year and first <= year <= last else None


def decide(title, categories, taken, probs, keys, looks_of_category, years_of_look, settings):
    """Apply rules a-e (see the top of the file) to one photo. Returns a dict of results."""
    candidates = list(dict.fromkeys(  # dict.fromkeys: drop repeats, keep the order
        key for c in categories.split(";") for key in looks_of_category.get(c, [])))

    # The day the photo was TAKEN rules out looks that didn't exist yet: a car photographed
    # in 2014 can't be the 2017 facelift. One year of slack, because model-year 2017 cars
    # go on sale in 2016. (It never proves the NEWER look: a 2015 car photographed in 2019
    # is still a 2015.) If it would rule out everything, the date is wrong: ignore it.
    taken_year = int(taken[:4]) if taken[:4].isdigit() else None
    if taken_year:
        possible = [c for c in candidates if years_of_look[c][0] <= taken_year + 1]
        candidates = possible or candidates

    # Add up the model's probability per look, and per car (make + model).
    p_look, p_car = {}, {}
    for key, p in zip(keys, probs.tolist()):
        p_look[key] = p_look.get(key, 0.0) + p
        p_car[key[:2]] = p_car.get(key[:2], 0.0) + p
    best_look = max(p_look, key=p_look.get)
    car_share = sum(p_car.get(car, 0.0) for car in {c[:2] for c in candidates})  # saved for review

    def share(key, among):
        """key's part of the probability the model gives to the looks in `among`."""
        total = sum(p_look.get(k, 0.0) for k in among)
        return p_look.get(key, 0.0) / total if total > 0 else 0.0

    year = model_year(title, candidates, years_of_look)
    result = dict(p_car=f"{car_share:.3f}", year=year or "", model_guess=look_text(best_look),
                  candidates=";".join(look_text(c) for c in candidates))

    def done(status, reason, key):
        result.update(status=status, reason=reason, p_look=f"{p_look.get(key, 0.0):.3f}")
        result.update(zip(["make", "model", "generation", "phase"], key))
        return result

    # The model's favorite among the candidates (falls back to the first candidate when
    # the model knows none of them: then every share is 0).
    favorite = max(candidates, key=lambda k: p_look.get(k, 0.0))

    # a. Different car? Only askable if run 2 knows at least one candidate LOOK. Knowing the
    #    car isn't enough: shown a generation it never saw (a 2022 C-Class W206 when it only
    #    knows older C-Classes), run 2 guesses something random, so a low score proves nothing.
    #    And it must be fairly sure of ANOTHER car: just being unsure of the right one (an
    #    odd angle, half the car hidden) isn't evidence of a wrong photo.
    knows_look = any(c in p_look for c in candidates)
    candidate_cars = {c[:2] for c in candidates}
    other_car = max((p for car, p in p_car.items() if car not in candidate_cars), default=0.0)
    if knows_look and other_car >= settings["min_other_car"]:
        return done("review", "different car?", favorite)
    # b. Only one possible look.
    if len(candidates) == 1:
        return done("auto", "category", candidates[0])
    # c. The year picks exactly one.
    by_year = [c for c in candidates if year and years_of_look[c][0] <= year <= years_of_look[c][1]]
    if len(by_year) == 1:
        # The model may only overrule the year if it knows the year's look: otherwise it's
        # "disagreeing" with an answer it can't give (same trap as rule d below).
        if (favorite != by_year[0] and by_year[0] in p_look
                and share(favorite, candidates) >= settings["auto_share"]):
            return done("review", "year vs model", favorite)
        return done("auto", "year", by_year[0])
    # d. The model strongly prefers one (among the year's candidates, if the year narrowed it).
    #    ...but only if it KNOWS every one of them. A look without a class always gets 0%,
    #    so "pre 100%, facelift 0%" for a model with no facelift class isn't a choice: it
    #    would label every facelift photo "pre", and the facelift would never get the 50
    #    photos it needs to become a class (the bug that hid the Highlander XU50 facelift).
    among = by_year or candidates
    best = max(among, key=lambda k: p_look.get(k, 0.0))
    knows_all = all(k in p_look for k in among)
    if knows_all and share(best, among) >= settings["auto_share"]:
        return done("auto", "model", best)
    # e. Nobody can decide.
    return done("review", "which look?" if knows_all else "look unknown to model", best)


class Crops(Dataset):
    """Each kept photo, cropped to the detector's box (as the app does), as a model input."""

    def __init__(self, rows, images_dir, transform, margin):
        self.rows, self.images_dir, self.transform, self.margin = rows, images_dir, transform, margin

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = open_photo(self.images_dir / row["file"])
        box = [float(v) for v in row["box"].split()]
        return i, self.transform(crop_to_car(image, box, self.margin))


def main():
    config = load_config()
    settings = config["commons"]["presort"]
    commons_dir = project_path(config["paths"]["commons_dir"])

    with open(commons_dir / "filtered.csv", encoding="utf-8", newline="") as f:
        kept = [r for r in csv.DictReader(f) if r["keep"] == "1"]
    with open(commons_dir / "manifest.csv", encoding="utf-8", newline="") as f:
        manifest = {r["title"]: r for r in csv.DictReader(f)}
    categories_of = {title: r["categories"] for title, r in manifest.items()}
    looks_of_category, years_of_look = load_looks()
    print(f"Kept photos to sort: {len(kept):,}")

    device = pick_device()
    model, info = load_trained_model(project_path(settings["checkpoint"]), device)
    keys = class_looks(info["class_names"], project_path(settings["splits_dir"]))
    loader = DataLoader(Crops(kept, commons_dir / "images", eval_transform(info),
                              config["detector"]["crop_margin"]),
                        batch_size=64, num_workers=config["training"]["num_workers"])

    results, counts = [], {}
    with torch.no_grad():
        for batch_number, (indexes, images) in enumerate(loader):
            probs = model(images.to(device)).softmax(dim=1).cpu()
            for i, p in zip(indexes.tolist(), probs):
                row = kept[i]
                if not any(c in looks_of_category for c in categories_of[row["title"]].split(";")):
                    continue  # its category left the table since listing (renamed row)
                out = decide(row["title"], categories_of[row["title"]], manifest[row["title"]]["taken"],
                             p, keys, looks_of_category, years_of_look, settings)
                results.append(dict(out, title=row["title"], file=row["file"]))
                label = f"{out['status']}: {out['reason']}"
                counts[label] = counts.get(label, 0) + 1
            if batch_number % 200 == 0:
                print(f"  {len(results):,}/{len(kept):,}")

    with open(commons_dir / "presort.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(results)

    print("\nResults:")
    for label, n in sorted(counts.items()):
        print(f"  {label:32} {n:7,}  ({n / len(results):.0%})")
    print(f"Saved {commons_dir / 'presort.csv'}")


if __name__ == "__main__":
    main()
