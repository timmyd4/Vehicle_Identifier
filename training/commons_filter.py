"""
Step 3 of the Commons pipeline: look at the pixels and keep only photos worth training on.

THE BIG PICTURE
---------------
Steps 1-2 checked the paperwork (file type, size, license). A photo of a steering wheel,
an engine, a car show crowd or a politician next to a car passes all of that. This script
runs the car detector (the same one the app uses, see detect.py) on every downloaded
photo and keeps it only if:

  1. it has a vehicle the detector is sure about (min_score)
     -> drops interiors, engines, badges, brochures, people
  2. that vehicle fills a decent part of the photo (min_area)
     -> drops street scenes where "the" car is a speck
  3. no second vehicle is nearly as big (max_second_car)
     -> a photo of two cars side by side: the category names one, but which one?
        A wrong-car label is worse than no photo, so we drop it
  4. it isn't a duplicate of a photo we already kept
     -> the same picture is often uploaded twice (cropped, resized, re-saved), and two
        copies could end up one in train and one in test: the exam would contain a
        photo the model studied (LEAKAGE, the test score would look too good)

It writes data/commons/filtered.csv: one row per photo with keep/drop, the reason, and the
car's box (so later steps can crop to it). It downloads nothing and changes no images.

DUPLICATES, HOW: a "difference hash" (dHash). Shrink the photo to 9x8 gray pixels and
record, for each row, whether each pixel is brighter than its right neighbor: 8x8 = 64
yes/no answers = one 64-bit number. Resizing, re-compressing or small color changes
don't change those coarse brighter/darker patterns, so copies get the same number, while
different photos almost never do. (Only exact hash matches count here: cheap, and it
catches the common case. Near-misses are left for the review step.)

RESUMING: photos already in filtered.csv are skipped, and it only looks at photos that
are fully downloaded, so you can even run it WHILE commons_download.py is still going
and run it again later for the rest.

Usage (from the project root):
    python training/commons_filter.py
"""

import csv

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

from common import load_config, project_path
from commons_download import local_name
from detect import cars_in_output, detector_from_config, open_photo

COLUMNS = ["title", "file", "keep", "reason", "score", "area", "second", "box", "dhash"]
BATCH_SIZE = 8


def dhash(image):
    """The 64-bit difference hash of a PIL image (see the docstring at the top)."""
    small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)  # L = grayscale
    pixels = list(small.getdata())  # 72 brightness values, row by row
    bits = 0
    for row in range(8):
        for col in range(8):
            left, right = pixels[row * 9 + col], pixels[row * 9 + col + 1]
            bits = (bits << 1) | (left > right)  # shift in one yes/no answer
    return f"{bits:016x}"  # as 16 hex digits, easy to store in a CSV


class Photos(Dataset):
    """Loads photos for the DataLoader: (row, image tensor, dhash), or (row, None, None)
    for a file PIL can't open.

    WHY a Dataset + DataLoader instead of a plain loop: decoding JPEGs is CPU work, and
    the DataLoader's worker processes do it in parallel while the GPU runs the detector.
    Same trick train.py uses.
    """

    def __init__(self, rows, images_dir):
        self.rows, self.images_dir = rows, images_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        try:
            image = open_photo(self.images_dir / row["file"])
        except OSError:  # truncated or not really an image
            return row, None, None
        return row, to_tensor(image), dhash(image)


def keep_as_list(batch):
    """The DataLoader normally stacks a batch into one tensor, which needs equal-size
    images. The detector takes a list of any sizes, so keep the batch as a list.
    (A named function, not a lambda: on Windows, worker processes must be able to import it.)
    """
    return batch


def merge_same_object(cars):
    """Merge boxes that are really one vehicle found twice, keeping the higher score.

    WHY: the detector removes repeated boxes with "non-maximum suppression" (keep the best
    box, delete boxes overlapping it), but only WITHIN a class. A box called "car" and one
    called "truck" never compete, so a wagon or SUV often comes back twice, once as each.
    Without this, judge() would see "two vehicles" in a photo of one car.
    """
    kept = []
    for car in sorted(cars, key=lambda c: c["score"], reverse=True):  # best first
        if all(overlap(car["box"], other["box"]) < 0.6 for other in kept):
            kept.append(car)
    return sorted(kept, key=lambda c: box_area(c["box"]), reverse=True)  # biggest first again


def overlap(a, b):
    """Intersection over union (IoU): shared area / combined area. 1 = same box, 0 = apart."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    shared = max(0, width) * max(0, height)
    return shared / (box_area(a) + box_area(b) - shared)


def judge(row, cars, width, height, settings):
    """Decide keep/drop for one photo; fill in row's reason, score, area, second and box."""
    cars = merge_same_object(cars)
    if not cars or cars[0]["score"] < settings["min_score"]:
        row["reason"] = "no clear vehicle"
        return False
    main = cars[0]  # biggest first (see detect.cars_in_output)
    area = box_area(main["box"]) / (width * height)
    second = box_area(cars[1]["box"]) / box_area(main["box"]) if len(cars) > 1 else 0.0
    row.update(score=f"{main['score']:.3f}", area=f"{area:.3f}", second=f"{second:.2f}",
               box=" ".join(str(round(v)) for v in main["box"]))
    if area < settings["min_area"]:
        row["reason"] = "vehicle too small"
        return False
    if second >= settings["max_second_car"]:
        row["reason"] = "two vehicles"
        return False
    row["reason"] = "ok"
    return True


def box_area(box):
    left, top, right, bottom = box
    return (right - left) * (bottom - top)


def main():
    config = load_config()
    settings = config["commons"]["filter"]
    commons_dir = project_path(config["paths"]["commons_dir"])
    images_dir = commons_dir / "images"
    out_path = commons_dir / "filtered.csv"

    # What's already judged, and the hashes of photos already kept (for duplicates).
    judged, kept_hashes = set(), set()
    if out_path.exists():
        with open(out_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                judged.add(row["title"])
                if row["keep"] == "1":
                    kept_hashes.add(row["dhash"])

    # Photos on disk that aren't judged yet. sorted() so a rerun goes in the same order.
    with open(commons_dir / "manifest.csv", encoding="utf-8", newline="") as f:
        todo = [{"title": r["title"], "file": local_name(r["title"], r["thumb_url"])}
                for r in csv.DictReader(f) if r["title"] not in judged]
    todo = sorted((r for r in todo if (images_dir / r["file"]).exists()), key=lambda r: r["title"])
    print(f"Already judged: {len(judged):,}. On disk and not judged yet: {len(todo):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector, categories, detector_settings = detector_from_config(device)
    loader = DataLoader(Photos(todo, images_dir), batch_size=BATCH_SIZE,
                        num_workers=config["training"]["num_workers"], collate_fn=keep_as_list)

    counts = {}
    with open(out_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not judged:
            writer.writeheader()
        with torch.no_grad():  # inference only: don't keep what training would need
            for batch_number, batch in enumerate(loader):
                readable = [(row, tensor, h) for row, tensor, h in batch if tensor is not None]
                outputs = detector([t.to(device) for _, t, _ in readable]) if readable else []

                for (row, tensor, h), output in zip(readable, outputs):
                    cars = cars_in_output(output, categories, detector_settings["labels"],
                                          detector_settings["min_score"])
                    row["dhash"] = h
                    keep = judge(row, cars, tensor.shape[2], tensor.shape[1], settings)
                    if keep and h in kept_hashes:
                        row["reason"], keep = "duplicate", False
                    if keep:
                        kept_hashes.add(h)
                    row["keep"] = int(keep)
                    writer.writerow(row)
                    counts[row["reason"]] = counts.get(row["reason"], 0) + 1
                for row, tensor, _ in batch:
                    if tensor is None:
                        writer.writerow(dict(row, keep=0, reason="unreadable"))
                        counts["unreadable"] = counts.get("unreadable", 0) + 1

                if batch_number % 250 == 0:
                    f.flush()
                    print(f"  {(batch_number + 1) * BATCH_SIZE:,}/{len(todo):,}  " +
                          "  ".join(f"{k}: {v:,}" for k, v in sorted(counts.items())))

    print("\nDone. This run: " + ", ".join(f"{k} {v:,}" for k, v in sorted(counts.items())))
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()
