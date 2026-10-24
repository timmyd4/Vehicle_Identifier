"""
Explore the VMMRdb dataset before building a training subset.

Reports:
  - number of classes, makes, and images
  - file types (by extension)
  - how many images each class has (distribution + chart)
  - how many classes would survive different minimum-image thresholds
  - the newest year we have for each make/model
  - folder names with suspicious years (e.g. 1900, a placeholder)

This script only reads the dataset; it never changes it.
Output files go to training/runs/explore/.

Usage (from the project root):
    python training/explore_data.py
"""

import os
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw charts straight to image files, no pop-up window
import matplotlib.pyplot as plt
import pandas as pd

from common import load_config, parse_class_name, project_path

# VMMRdb says it covers 1950-2016; anything outside that is worth a look.
EXPECTED_YEARS = range(1950, 2017)

# Thresholds to compare when choosing the minimum images per class (step 4).
THRESHOLDS = [10, 20, 50, 100, 200]


def scan_dataset(dataset_dir):
    """Return a DataFrame with one row per class folder, plus a Counter of file extensions."""
    rows = []
    extensions = Counter()

    # os.scandir is much faster than Path.iterdir() for ~291K files on Windows,
    # because it gets "is this a file?" from the directory listing for free.
    with os.scandir(dataset_dir) as class_entries:
        for class_entry in class_entries:
            if not class_entry.is_dir():
                continue

            with os.scandir(class_entry.path) as file_entries:
                file_names = [f.name for f in file_entries if f.is_file()]

            extensions.update(Path(name).suffix.lower() for name in file_names)
            make, model, year = parse_class_name(class_entry.name)
            rows.append({
                "class_name": class_entry.name,
                "make": make,
                "model": model,
                "year": year,
                "num_images": len(file_names),
            })

    df = pd.DataFrame(rows).sort_values("class_name").reset_index(drop=True)
    return df, extensions


def print_header(title):
    print(f"\n=== {title} ===")


def report_overview(df, extensions):
    print_header("Overview")
    print(f"Classes (make_model_year folders): {len(df):,}")
    print(f"Make/model pairs (years merged):   {df.groupby(['make', 'model']).ngroups:,}")
    print(f"Makes:                             {df['make'].nunique():,}")
    print(f"Images:                            {df['num_images'].sum():,}")

    print_header("File types")
    for ext, count in extensions.most_common():
        print(f"  {ext or '(no extension)':8} {count:>8,}")


def report_distribution(df):
    print_header("Images per class")
    # describe() gives count, mean, min, max and percentiles in one call.
    stats = df["num_images"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    print(stats.round(1).to_string())

    print("\nLargest classes:")
    print(df.nlargest(5, "num_images")[["class_name", "num_images"]].to_string(index=False))
    print("\nSmallest classes:")
    print(df.nsmallest(5, "num_images")[["class_name", "num_images"]].to_string(index=False))

    # How much data each threshold keeps, with and without merging model years.
    # Merging years combines e.g. honda_civic_2004 + honda_civic_2005 into one class.
    merged = df.groupby(["make", "model"])["num_images"].sum()
    print_header("Classes kept at each minimum-images threshold")
    print(f"{'min images':>10} | {'classes':>8} {'images':>9} | {'merged-year classes':>19} {'images':>9}")
    for t in THRESHOLDS:
        kept = df[df["num_images"] >= t]
        kept_merged = merged[merged >= t]
        print(f"{t:>10} | {len(kept):>8,} {kept['num_images'].sum():>9,} | "
              f"{len(kept_merged):>19,} {kept_merged.sum():>9,}")


def report_newest_years(df, out_dir):
    print_header("Newest year per make/model")
    newest = (
        df.groupby(["make", "model"])["year"]
        .max()
        .rename("newest_year")
        .reset_index()
        .sort_values(["make", "model"])
    )
    newest.to_csv(out_dir / "newest_year_per_model.csv", index=False)

    # Group by decade so the summary fits on screen; the full list is in the CSV.
    decade = (newest["newest_year"] // 10 * 10).astype(str) + "s"
    print("Make/models by the decade of their newest year:")
    print(decade.value_counts().sort_index().to_string())

    print("\nNewest year per make:")
    per_make = df.groupby("make")["year"].max().sort_values(ascending=False)
    print(per_make.to_string())
    print(f"\nFull list: {out_dir / 'newest_year_per_model.csv'}")


def report_suspicious_years(df):
    print_header(f"Years outside {EXPECTED_YEARS.start}-{EXPECTED_YEARS.stop - 1}")
    odd = df[~df["year"].isin(EXPECTED_YEARS)]
    if odd.empty:
        print("None")
        return
    print(f"{len(odd)} classes, {odd['num_images'].sum():,} images:")
    print(odd[["class_name", "num_images"]].to_string(index=False))


def plot_distribution(df, out_path):
    """Bar chart: how many classes fall into each images-per-class range."""
    bins = [0, 10, 20, 50, 100, 200, 500, float("inf")]
    labels = ["0-9", "10-19", "20-49", "50-99", "100-199", "200-499", "500+"]
    counts = pd.cut(df["num_images"], bins=bins, labels=labels, right=False).value_counts(sort=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, counts.values, width=0.8, color="#2a78d6")
    ax.bar_label(bars, labels=[f"{v:,}" for v in counts.values], padding=3, color="#52514e", fontsize=9)

    ax.set_title("VMMRdb: classes by number of images", loc="left", color="#0b0b0b")
    ax.set_xlabel("Images in class", color="#52514e")
    ax.set_ylabel("Number of classes", color="#52514e")

    # Keep the chart quiet: light horizontal gridlines, no box around the plot.
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", length=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nChart saved: {out_path}")


def main():
    config = load_config()
    dataset_dir = project_path(config["paths"]["dataset_dir"])
    out_dir = project_path(config["paths"]["runs_dir"]) / "explore"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {dataset_dir} ...")
    df, extensions = scan_dataset(dataset_dir)
    df.to_csv(out_dir / "class_counts.csv", index=False)

    report_overview(df, extensions)
    report_distribution(df)
    report_suspicious_years(df)
    report_newest_years(df, out_dir)
    plot_distribution(df, out_dir / "images_per_class.png")


# This block only runs when the file is executed directly, not when imported.
# It's Python's equivalent of Java's `public static void main`.
if __name__ == "__main__":
    main()
