"""
Build the training subset of VMMRdb and split it into train/val/test CSV files.

THE BIG PICTURE
---------------
A model learns like a student studying flashcards: photo on the front, answer
("honda_civic_2001-2003") on the back. Training needs three separate piles:

    train (80%)  - the cards the model studies and learns from
    val   (10%)  - a practice quiz, checked after every round of training to see
                   whether the model is still improving
    test  (10%)  - the final exam, used ONCE at the end. The model must never have
                   seen these cards, or the score is meaningless.

This script builds those three piles as CSV files (one row = one card: an image
path plus its answer). train.py, evaluate.py, etc. just read the CSVs.

WHY ALL THE CLEANING?
---------------------
The model learns whatever we show it, including mistakes; it cannot tell good
data from bad ("garbage in, garbage out"). VMMRdb comes from used-car listings
where sellers typed their own labels, so it contains wrong years, tiny images,
and the same photo filed under several cars. Each step below removes one kind
of bad card before it can teach the model something wrong or inflate the exam
score.

WHY A SEPARATE SCRIPT (not part of training)?
---------------------------------------------
  - It's slow (reads all 285K files, ~1.5 min). Run it once; training then
    starts instantly every time.
  - The piles must stay FIXED. If every training run re-split the data, the
    final exam would change each time and you could never compare two models.
    A fixed random seed makes the split identical on every run.

No images are copied or modified. The CSVs only list paths relative to the
dataset folder. Settings come from the `subset` section of training/config.yaml.

COMMONS PHOTOS (run 3+, include_commons in config.yaml)
------------------------------------------------------
The photos commons_presort.py labeled "auto" join the VMMRdb photos. Their paths
start with "../commons/images/": relative to data/VMMRdb, that's data/commons/images,
so every script that opens dataset_dir / path finds them without changes.
Each row also gets a "source" column (vmmrdb or commons).

Two rules keep the exam honest when data is added:
  - FROZEN TEST SET (keep_split_from): every VMMRdb photo goes into the same pile
    it had in an earlier split. Run 2 and run 3 then take exactly the same exam on
    the VMMRdb part, so their scores can be compared directly.
  - GROUP SPLIT for Commons: all photos one person took on one day go into the
    same pile. Someone who shot one car 30 times at a show would otherwise put near-
    identical photos in train AND test, and the model would pass the exam from memory.

Usage (from the project root):
    python training/build_subset.py

Output (in data/splits/):
    train.csv, val.csv, test.csv   the three piles
    classes.csv                    one row per class (the list of possible answers)
    subset_settings.json           the exact settings used, for traceability
"""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from common import car_model, load_config, load_generations, load_merges, parse_class_name, project_path

# Where the Commons photos are, relative to the dataset folder (data/VMMRdb).
COMMONS_PREFIX = "../commons/images/"


# ---------------------------------------------------------------------------
# Step 1: find every photo and work out its answer
# ---------------------------------------------------------------------------
def list_images(dataset_dir):
    """Return a DataFrame with one row per image: path, folder, make, model, year.

    WHY: the answer for each photo isn't stored anywhere except the name of the
    folder it's in. A photo inside honda_civic_2002/ is a 2002 Honda Civic.
    So we walk every folder and build a table of "this photo -> this car".

    A DataFrame is pandas' table type: think of it as an in-memory SQL table,
    or a List<Row> where every column can be filtered/grouped in one line.
    """
    rows = []  # build a plain Python list of dicts first, then turn it into a table at the end

    # os.scandir lists a folder's contents. `with` closes the listing when done,
    # like Java's try-with-resources.
    with os.scandir(dataset_dir) as class_entries:
        for class_entry in class_entries:
            # Only folders are classes; skip any stray files at the top level.
            if not class_entry.is_dir():
                continue

            # Parse the answer out of the folder name once per folder,
            # e.g. "honda_civic_2002" -> ("honda", "civic", 2002).
            make, model, year = parse_class_name(class_entry.name)

            with os.scandir(class_entry.path) as file_entries:
                for f in file_entries:
                    if f.is_file():
                        rows.append({
                            # Store the path RELATIVE to the dataset folder, with forward
                            # slashes. That way the CSV still works if the dataset lives
                            # somewhere else (another PC, Linux, Kaggle).
                            "path": f"{class_entry.name}/{f.name}",
                            "folder": class_entry.name,
                            "make": make,
                            "model": model,
                            "year": year,
                        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2: measure and fingerprint every photo
# ---------------------------------------------------------------------------
def read_image_info(full_path):
    """Return (md5 hash, width, height) for one image file.

    The MD5 hash is a fingerprint of the file's bytes: two files with the same
    hash are the same photo, even if their names differ. Step 5 uses it to find
    duplicates. The size is used in step 3 to drop tiny images.
    """
    with open(full_path, "rb") as f:  # "rb" = read raw bytes
        md5 = hashlib.md5(f.read()).hexdigest()

    # Image.open is lazy: it only reads the file header (where width/height are
    # stored), not the pixels, so this is fast.
    with Image.open(full_path) as img:
        width, height = img.size
    return md5, width, height


def add_image_info(df, dataset_dir):
    """Add md5, width, and height columns to the table.

    WHY: this step doesn't remove anything itself. It gathers the facts that
    later steps need to decide what to remove.

    HOW: reading 285K files one at a time would be slow, mostly because each read
    waits on the disk. A thread pool (Python's version of Java's ExecutorService)
    keeps 16 reads in flight at once, so the waiting overlaps.
    """
    full_paths = [dataset_dir / p for p in df["path"]]  # `/` on Path objects joins paths

    with ThreadPoolExecutor(max_workers=16) as pool:
        # pool.map runs read_image_info on every path and returns results IN ORDER,
        # so result #i belongs to row #i. chunksize=256 hands each thread 256 paths
        # at a time instead of 1, which cuts overhead.
        # tqdm(...) just wraps the results to draw a progress bar.
        results = list(tqdm(pool.map(read_image_info, full_paths, chunksize=256),
                            total=len(full_paths), desc="Reading images", unit="img"))

    # `results` is a list of (md5, width, height) tuples. Turn it into a 3-column
    # table and attach those columns to df. index=df.index lines rows up exactly.
    df[["md5", "width", "height"]] = pd.DataFrame(results, index=df.index)
    return df


# ---------------------------------------------------------------------------
# Step 4: decide what the "answer" on each card is (year buckets)
# ---------------------------------------------------------------------------
def assign_classes(df, bucket_size, max_year):
    """Add year_start, year_end, and class_name columns based on the year bucket size.

    WHY: adjacent model years usually look identical, and single years have too
    few photos. Grouping years gives each class more photos and asks the model a
    question it can actually answer. (See the year_bucket_size comment in config.yaml.)

    These operations work on whole columns at once ("vectorized"): one line
    updates all ~285K rows, with no explicit loop. Pandas runs the loop in C.
    """
    if bucket_size == 0:
        # Merge every year of a model into one class, e.g. "honda_civic".
        # transform("min") computes the min year per (make, model) group and writes
        # it back onto every row of that group, so each row knows its group's range.
        df["year_start"] = df.groupby(["make", "model"])["year"].transform("min")
        df["year_end"] = df.groupby(["make", "model"])["year"].transform("max")
        df["class_name"] = df["make"] + "_" + df["model"]
        return df

    # Integer division (//) snaps each year down to the start of its bucket.
    # With bucket_size 3:  2001 // 3 * 3 = 2001
    #                      2002 // 3 * 3 = 2001
    #                      2003 // 3 * 3 = 2001
    #                      2004 // 3 * 3 = 2004   <- next bucket
    df["year_start"] = (df["year"] // bucket_size) * bucket_size

    # The bucket ends bucket_size-1 years later, but never past the dataset's last
    # year, so we don't label a class "2016-2018" when we have no 2017/2018 photos.
    df["year_end"] = (df["year_start"] + bucket_size - 1).clip(upper=max_year)

    # Build the year part of the name: "2001-2003", or just "2016" when the
    # bucket was clipped down to a single year.
    years = df["year_start"].astype(str)
    has_range = df["year_end"] > df["year_start"]          # True/False for every row
    # Only on rows where has_range is True, append "-<year_end>".
    years[has_range] = years[has_range] + "-" + df.loc[has_range, "year_end"].astype(str)

    df["class_name"] = df["make"] + "_" + df["model"] + "_" + years
    return df


def assign_generation_classes(df, table):
    """Label every photo with its look from generations.csv; drop photos the table doesn't cover.

    WHY: year buckets made the model guess differences that often don't exist
    (a 2014 and a 2016 Highlander look identical). Each row of the table is one
    look the eye CAN tell apart, so those are the answers worth learning.

    The class name keeps the old "make_model_years" shape, using the row's own
    year range (toyota_highlander_2017-2019 = XU50 facelift). That way
    display_name(), export_onnx.py and the Java app keep working unchanged. Rows
    never overlap within a model, so the years alone identify the look. The
    generation and phase are kept as extra columns for classes.csv.
    """
    # Step A: map each folder model to its model in our labels: trims folded in
    # (corolla_s -> corolla), look-alikes merged (325i -> 3 series, model_merges.csv).
    # Done once per distinct (make, folder model) pair, not once per photo.
    table_models = table.groupby("make")["model"].apply(list).to_dict()  # make -> [models]
    merges = load_merges()
    pairs = df[["make", "model"]].drop_duplicates()
    pairs["base_model"] = [car_model(make, model, table_models.get(make, []), merges)
                           for make, model in zip(pairs["make"], pairs["model"])]
    df = df.merge(pairs, on=["make", "model"])

    not_in_table = df["base_model"].isna()
    print(f"Dropped {not_in_table.sum():,} images of models that aren't in generations.csv "
          f"({df.loc[not_in_table, 'folder'].nunique():,} folders)")
    df = df[~not_in_table]

    # Step B: join every photo to every row of its model, then keep the one row whose
    # years contain the photo's year. Like SQL:
    #   SELECT ... FROM photos p JOIN generations g ON p.make = g.make AND p.base_model = g.model
    #   WHERE p.year BETWEEN g.first_year AND g.last_year
    rows = table[["make", "model", "generation", "phase", "first_year", "last_year"]]
    rows = rows.rename(columns={"model": "base_model"})
    joined = df.merge(rows, on=["make", "base_model"])
    joined = joined[joined["year"].between(joined["first_year"], joined["last_year"])]

    # Photos whose year no row covers simply didn't survive the join: years where two
    # looks were sold side by side, unclear changeover years, or a year with no US model.
    print(f"Dropped {len(df) - len(joined):,} images from years no table row covers")

    # Step C: build the answer. The base model replaces the trim folder name, so the
    # make+model grouping in the app sees one Corolla, not corolla + corolla_s.
    joined = joined.copy()
    joined["model"] = joined["base_model"]
    return add_look_class_names(joined)


def add_look_class_names(df):
    """Add year_start, year_end and class_name from each row's look (first_year, last_year).

    Shared by VMMRdb and Commons photos, so the same look always gets the same name.
    """
    df["year_start"] = df["first_year"]
    df["year_end"] = df["last_year"]
    years = df["year_start"].astype(str)
    has_range = df["year_end"] > df["year_start"]
    years[has_range] = years[has_range] + "-" + df.loc[has_range, "year_end"].astype(str)
    df["class_name"] = df["make"] + "_" + df["model"] + "_" + years
    return df


# ---------------------------------------------------------------------------
# Commons photos (see COMMONS PHOTOS at the top)
# ---------------------------------------------------------------------------
def list_commons_images(commons_dir, table):
    """Return the Commons photos commons_presort.py labeled "auto", with their looks' classes.

    presort.csv already names each photo's look (make, model, generation, phase), so
    labeling is one join with the table to get the look's years.
    """
    presort = pd.read_csv(commons_dir / "presort.csv", keep_default_na=False)

    # Looks a human picked on commons_looks.py replace the pre-sort's answer for that photo
    # (a skipped photo stays out). They go in as status "human", next to the "auto" ones.
    answers_file = commons_dir / "look_answers.json"
    answers = json.loads(answers_file.read_text(encoding="utf-8")) if answers_file.exists() else {}
    picked = {title: look for title, look in answers.items() if look is not None}
    is_picked = presort["title"].isin(picked)
    if is_picked.any():
        looks = [picked[t] for t in presort.loc[is_picked, "title"]]
        presort.loc[is_picked, ["make", "model", "generation", "phase"]] = looks
        presort.loc[is_picked, "status"] = "human"

    auto = presort[presort["status"].isin(["auto", "human"])]
    print(f"Commons: {len(auto):,} labeled photos ({is_picked.sum():,} by a human) of {len(presort):,} "
          f"sorted; the rest wait for the next model to sort them")

    # The group for the split: who TOOK the photo, on which day ("Jane|2019-05-04").
    # Not who uploaded it: bots like "Flickr upload bot" copy thousands of different
    # photographers' pictures to Commons in one day, which would lump strangers together.
    # Falls back to the uploader / upload day for the ~1% of files missing the author or date.
    manifest = pd.read_csv(commons_dir / "manifest.csv", keep_default_na=False,
                           usecols=["title", "artist", "uploader", "uploaded", "taken"])
    df = auto.merge(manifest, on="title")
    who = df["artist"].where(df["artist"] != "", df["uploader"])
    day = df["taken"].str.extract(r"(\d{4}-\d{2}-\d{2})")[0].fillna(df["uploaded"].str[:10])
    df["group"] = who + "|" + day

    # Join on all four parts of the look. A look renamed or removed from the table since
    # the pre-sort simply doesn't match, and its photos are left out.
    rows = table[["make", "model", "generation", "phase", "first_year", "last_year"]]
    joined = df.merge(rows, on=["make", "model", "generation", "phase"])
    if len(joined) < len(df):
        print(f"Commons: dropped {len(df) - len(joined):,} photos whose look isn't in the table "
              f"anymore (rerun commons_presort.py after changing generations.csv)")

    joined["path"] = COMMONS_PREFIX + joined["file"]
    joined["folder"] = "commons"
    # The filename year, when the pre-sort found one (only kept for reference).
    joined["year"] = pd.to_numeric(joined["year"], errors="coerce").astype("Int64")
    joined["source"] = "commons"
    return add_look_class_names(joined)


# ---------------------------------------------------------------------------
# Step 5: duplicate photos
# ---------------------------------------------------------------------------
def remove_duplicates(df):
    """Keep one copy of each duplicated photo; drop photos whose copies disagree on the class.

    WHY, problem 1 - cheating on the exam: if the same photo is in both the train
    pile and the test pile, the model just recognizes it from memory. The test
    score goes up, but the model isn't actually better. This is called "data
    leakage" and it's one of the most common ways ML results end up overstated.
    Keeping only one copy of each photo makes leakage impossible.

    WHY, problem 2 - contradictory answers: VMMRdb has photos filed under two
    different cars (e.g. both acura_integra_1987 and acura_integra_2001). At least
    one label is wrong and we can't tell which, so we drop every copy rather than
    risk teaching the wrong answer.

    Note that this runs AFTER year bucketing: copies in acura_integra_1994 and
    acura_integra_1995 both land in class acura_integra_1993-1995, so they agree
    and we keep one. Only copies that end up in DIFFERENT final classes conflict.
    """
    # For each fingerprint, count how many different classes its copies are in.
    # Result: a Series (a column with an index) mapping md5 -> number of classes.
    classes_per_hash = df.groupby("md5")["class_name"].nunique()

    # The fingerprints that appear in more than one class are the conflicts.
    conflicting = classes_per_hash[classes_per_hash > 1].index

    # True for every row whose photo is one of the conflicting ones.
    conflict_mask = df["md5"].isin(conflicting)

    dropped_conflicts = int(conflict_mask.sum())  # True counts as 1, so sum = number of rows
    df = df[~conflict_mask]  # ~ means NOT: keep rows that are NOT conflicts

    # What's left: duplicates that all agree on the class. Keep the first, drop the rest.
    before = len(df)
    df = df.drop_duplicates(subset="md5", keep="first")
    dropped_copies = before - len(df)
    return df, dropped_conflicts, dropped_copies


# ---------------------------------------------------------------------------
# Step 7: deal the cards into three piles
# ---------------------------------------------------------------------------
def split_stratified(df, fractions, seed):
    """Split into train/val/test so every class keeps the same proportions in each split.

    WHY "stratified": a plain random split might put 48 of a rare car's 50 photos
    in train and 0 in test, so that car would never be tested. Stratifying splits
    EACH CLASS 80/10/10 separately, so every car appears in every pile.

    WHY a seed: the split is random but repeatable. The same seed always produces
    the same split (like new Random(42) in Java), so the final exam never changes
    between runs and scores stay comparable.

    HOW: sklearn's train_test_split only cuts in two, so we cut twice:
        all data  -> 80% train | 20% "rest"
        "rest"    -> half val  | half test      (10% + 10% of the original)
    """
    holdout = fractions["val"] + fractions["test"]  # 0.2

    train, rest = train_test_split(df, test_size=holdout, stratify=df["class_name"], random_state=seed)

    # test_size here is a fraction of `rest`, not of the whole dataset:
    # 0.1 / 0.2 = 0.5, i.e. half of the 20% -> 10% of the original.
    val, test = train_test_split(rest, test_size=fractions["test"] / holdout,
                                 stratify=rest["class_name"], random_state=seed)
    return train, val, test


def pile_by_hash(key, fractions):
    """Put `key` in train, val or test based on a hash of it.

    WHY a hash: the same key always lands in the same pile, on every run and on every
    computer, whatever else is in the data. (Python's built-in hash() is NOT like this:
    it changes every run, on purpose, for security. md5 never changes.)

    HOW: the first 8 hex digits of md5 are a number from 0 to 16^8. Divided by 16^8,
    that's a "random" fraction between 0 and 1: below 0.8 = train, below 0.9 = val, else test.
    """
    fraction = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) / 16 ** 8
    if fraction < fractions["train"]:
        return "train"
    return "val" if fraction < fractions["train"] + fractions["val"] else "test"


def split_all(df, settings):
    """Deal every photo into a pile. Returns (train, val, test).

    VMMRdb photos: the pile they had in keep_split_from (frozen test set); photos that
    weren't in that split (e.g. folders a new merge brought in) go by a hash of their path.
    Without keep_split_from, VMMRdb is split stratified, as in runs 1 and 2.
    Commons photos: by a hash of their uploader + day group (group split).
    """
    vmmrdb = df[df["source"] == "vmmrdb"]
    commons = df[df["source"] == "commons"].copy()

    if settings.get("keep_split_from"):
        old_dir = project_path(settings["keep_split_from"])
        old_pile = {}
        for pile in ("train", "val", "test"):
            for path in pd.read_csv(old_dir / f"{pile}.csv")["path"]:
                old_pile[path] = pile
        vmmrdb = vmmrdb.copy()
        vmmrdb["pile"] = [old_pile.get(p) or pile_by_hash(p, settings["split"]) for p in vmmrdb["path"]]
        kept = vmmrdb["path"].isin(old_pile.keys()).sum()
        print(f"VMMRdb: {kept:,} photos keep their pile from {settings['keep_split_from']}, "
              f"{len(vmmrdb) - kept:,} new ones split by hash")
    else:
        train, val, test = split_stratified(vmmrdb, settings["split"], settings["seed"])
        vmmrdb = pd.concat([train.assign(pile="train"), val.assign(pile="val"), test.assign(pile="test")])

    commons["pile"] = [pile_by_hash(g, settings["split"]) for g in commons["group"]]
    print(f"Commons: {commons['group'].nunique():,} photographer+day groups")

    df = pd.concat([vmmrdb, commons])
    return [df[df["pile"] == pile] for pile in ("train", "val", "test")]


def main():
    config = load_config()
    settings = config["subset"]
    dataset_dir = project_path(config["paths"]["dataset_dir"])
    splits_dir = project_path(config["paths"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)  # create data/splits/ if missing

    # Step 1 + 2: build the full table of photos, then add size and fingerprint.
    df = list_images(dataset_dir)
    print(f"Found {len(df):,} images in {df['folder'].nunique():,} folders")
    df = add_image_info(df, dataset_dir)

    # Step 3a: drop folders with impossible years.
    # WHY: VMMRdb covers 1950-2016. Folders like toyota_camry_1938 (there was no
    # 1938 Camry) or the "1900" placeholders are wrong answers on the card.
    # between() gives a True/False column; df[mask] keeps only the True rows.
    # .copy() makes df its own table rather than a "view" of the old one, which
    # avoids pandas warnings when we add columns to it later.
    in_range = df["year"].between(settings["min_year"], settings["max_year"])
    print(f"Dropped {(~in_range).sum():,} images with years outside "
          f"{settings['min_year']}-{settings['max_year']}")
    df = df[in_range].copy()

    # Step 3b: drop tiny images.
    # WHY: training resizes everything to about 224x224 pixels. A 60x40 photo
    # blown up that far is mostly blur, so it teaches the model nothing useful.
    # min(axis=1) takes the smaller of width/height for each row (axis=1 = across columns).
    big_enough = (df[["width", "height"]].min(axis=1) >= settings["min_image_side"])
    print(f"Dropped {(~big_enough).sum():,} images smaller than {settings['min_image_side']}px")
    df = df[big_enough].copy()

    # Step 4: decide each photo's answer (see label_mode in config.yaml).
    if settings["label_mode"] == "generations":
        df = assign_generation_classes(df, load_generations())
    else:
        df = assign_classes(df, settings["year_bucket_size"], settings["max_year"])
    df["source"] = "vmmrdb"

    # Step 4b: add the Commons photos (already labeled; same size check as VMMRdb).
    # VMMRdb comes FIRST in the table, so when step 5 finds the same photo in both,
    # it keeps the VMMRdb copy and the frozen split stays intact.
    if settings.get("include_commons"):
        commons = list_commons_images(project_path(config["paths"]["commons_dir"]), load_generations())
        commons = add_image_info(commons, dataset_dir)
        big_enough = commons[["width", "height"]].min(axis=1) >= settings["min_image_side"]
        df = pd.concat([df, commons[big_enough]], ignore_index=True)

    # Step 5: duplicates. This MUST happen before the split (step 7); otherwise
    # copies of the same photo could be dealt into different piles.
    df, dropped_conflicts, dropped_copies = remove_duplicates(df)
    print(f"Dropped {dropped_conflicts:,} duplicate photos with conflicting labels")
    print(f"Dropped {dropped_copies:,} extra copies of duplicate photos")

    # Step 6: drop classes with too few photos.
    # WHY: the model can't learn a car from a handful of photos, and 10% of a
    # small class is too few test photos to measure anything. This happens AFTER
    # cleaning, so a class is judged by how many GOOD photos it has.
    # value_counts() counts rows per class_name, like SQL GROUP BY ... COUNT(*).
    class_sizes = df["class_name"].value_counts()
    keep = class_sizes[class_sizes >= settings["min_images_per_class"]].index
    # Save every look's count, the dropped ones too: a look with 40 photos needs only 10
    # more to become a class. commons_looks.py reads this to pick which looks to ask about.
    if "generation" in df:
        counts = (df.groupby(["make", "model", "generation", "phase", "class_name"]).size()
                  .rename("num_images").reset_index().sort_values("num_images"))
        counts.to_csv(splits_dir / "look_counts.csv", index=False)
    print(f"Dropped {len(class_sizes) - len(keep):,} classes with fewer than "
          f"{settings['min_images_per_class']} images ({(~df['class_name'].isin(keep)).sum():,} images)")
    df = df[df["class_name"].isin(keep)]

    # Step 7: deal the cards into three piles (see split_all for the rules).
    train, val, test = split_all(df, settings)

    # Write the three piles. Only these columns are needed for training; sorting
    # by path makes the files easy to browse and diff.
    columns = ["path", "class_name", "make", "model", "year", "source"]
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.sort_values("path")[columns].to_csv(splits_dir / f"{name}.csv", index=False)

    # The list of possible answers: one row per class, sorted alphabetically.
    # IMPORTANT: a model doesn't output names, it outputs a number per class.
    # Training will use this row order as the class numbering (row 0 = class 0, ...),
    # and the Java app will use the same order to turn numbers back into names.
    # groupby(...).agg(...) is like SQL: GROUP BY class_name, then for each group
    # take the first make/model/years and count the rows.
    details = dict(make=("make", "first"), model=("model", "first"),
                   year_start=("year_start", "first"), year_end=("year_end", "first"))
    if settings["label_mode"] == "generations":
        # Extra columns only generation labels have (e.g. "XU50", "facelift").
        details.update(generation=("generation", "first"), phase=("phase", "first"))
    classes = (
        df.groupby("class_name")
        .agg(**details, num_images=("path", "size"))   # ** unpacks the dict into keyword arguments
        .reset_index()          # turn class_name from the group key back into a normal column
        .sort_values("class_name")
    )
    classes.to_csv(splits_dir / "classes.csv", index=False)

    # Save the exact settings used. If you change config.yaml later, you can
    # still tell which settings produced the CSVs currently on disk.
    with open(splits_dir / "subset_settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

    print(f"\nFinal subset: {len(classes):,} classes, {len(df):,} images")
    print(f"  train {len(train):>7,}   val {len(val):>6,}   test {len(test):>6,}")
    for source in ("vmmrdb", "commons"):
        counts = [(part["source"] == source).sum() for part in (train, val, test)]
        print(f"    {source:8} train {counts[0]:>7,}   val {counts[1]:>6,}   test {counts[2]:>6,}")
    print(f"  images per class: min {classes['num_images'].min()}, "
          f"median {int(classes['num_images'].median())}, max {classes['num_images'].max()}")
    print(f"Wrote CSVs to {splits_dir}")


# Runs main() only when this file is executed directly, not when imported
# (Python's equivalent of Java's `public static void main`).
if __name__ == "__main__":
    main()
