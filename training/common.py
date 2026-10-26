"""
Helpers shared by the training scripts.

Python finds this module because the scripts in training/ are run directly
(e.g. `python training/build_subset.py`), which puts training/ on the import path.
"""

from pathlib import Path

import pandas as pd
import yaml

# The project root is the folder above training/. Building paths from here
# (instead of the current directory) means scripts work no matter where you run them from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config():
    """Read training/config.yaml into a nested dict."""
    with open(PROJECT_ROOT / "training" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative_path):
    """Turn a path from config.yaml (relative to the project root) into an absolute Path."""
    return PROJECT_ROOT / relative_path


def parse_class_name(folder_name):
    """Split a VMMRdb folder name into (make, model, year).

    Makes use spaces, not underscores ("mercedes benz"), but models can contain
    underscores ("a4_avant_wagon"). So: make = first part, year = last part,
    model = everything in between.

        "honda_civic_2005"         -> ("honda", "civic", 2005)
        "audi_a4_avant_wagon_2001" -> ("audi", "a4_avant_wagon", 2001)
    """
    parts = folder_name.split("_")
    make = parts[0]
    model = "_".join(parts[1:-1])
    year = int(parts[-1])
    return make, model, year


def base_model(folder_model, table_models):
    """Return the table model a VMMRdb folder model belongs to, or None.

    WHY: VMMRdb splits one car over trim and body-style folders (corolla, corolla_s,
    corolla_le). They look the same, so the table only has the base model and
    every trim folder is merged into it. build_subset.py uses this to label photos,
    evaluate.py to score "right car?" the same way for every run.

    HOW: a folder belongs to model M when it IS M or starts with "M_". If several
    table models match, the longest wins, so "ram_1500_hd" goes to ram_1500, not ram.
    """
    matches = [m for m in table_models
               if folder_model == m or folder_model.startswith(m + "_")]
    return max(matches, key=len) if matches else None


def load_merges():
    """Read training/model_merges.csv into {(make, model): merged_model}.

        {("bmw", "325i"): "3 series", ("ford", "f350"): "f250-f350", ...}

    A dict keyed by a tuple, like a Map<Pair<String, String>, String> in Java.
    """
    df = pd.read_csv(PROJECT_ROOT / "training" / "model_merges.csv", comment="#",
                     skipinitialspace=True, keep_default_na=False)
    return {(make, model): merged for make, model, merged in zip(df["make"], df["model"], df["merged_model"])}


def car_model(make, folder_model, table_models, merges):
    """The model a VMMRdb folder belongs to in OUR labels, or None if we don't cover it.

    The ONE place that answers "which car is this folder?" for every script, in two steps:

      1. trims:  "325i_convertible" -> "325i"      (base_model's prefix rule)
      2. merges: "325i" -> "3 series"              (model_merges.csv)

    Step 1 matches against the table's models AND the merge list's old names, because
    after merging, "325i" is no longer in the table, but its folders still say 325i.

    table_models: the table's models for this make. merges: from load_merges().
    """
    names = list(table_models) + [old for (m, old) in merges if m == make]
    model = base_model(folder_model, names)
    if model is None:
        return None
    return merges.get((make, model), model)  # unchanged if it isn't merged


def load_generations():
    """Read training/generations.csv (one row = one look of a car) into a DataFrame.

    comment="#" skips the explanation lines at the top of the file.
    keep_default_na=False stops pandas from turning the text "NA" into a missing
    value. "NA" is a real generation name (the first Mazda Miata), and empty notes
    then come through as "" instead of NaN.
    """
    return pd.read_csv(PROJECT_ROOT / "training" / "generations.csv", comment="#",
                       skipinitialspace=True, keep_default_na=False)
