"""
Helpers shared by the training scripts.

Python finds this module because the scripts in training/ are run directly
(e.g. `python training/build_subset.py`), which puts training/ on the import path.
"""

from pathlib import Path

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
