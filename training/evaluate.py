"""
The final exam: measure the trained model on the TEST pile and explain its mistakes.

THE BIG PICTURE
---------------
During training we watched val accuracy and kept the epoch that did best on it.
That means val influenced which model we picked, so its score is slightly
optimistic. The TEST pile has never influenced anything, so its score is the
honest number to put in the README.

Beyond a single accuracy number, this script answers "HOW is it wrong?":

  - Accuracy at three levels. Getting the make right but the model wrong is a
    different kind of mistake from picking the right car one year bucket off:
        make          Honda == Honda
        make + model  Honda Civic == Honda Civic (any years)
        exact class   Honda Civic (2001-2003) exactly
  - Most-confused pairs: which classes it mixes up most often. This tells us
    what to fix next (e.g. adjacent year buckets, or "civic" vs "civic_coupe").
  - Worst classes: the cars it struggles with most.
  - A picture of high-confidence mistakes, so you can SEE whether the model is
    wrong, or the label is (VMMRdb has mislabeled photos).

Output (training/runs/evaluate/):
    metrics.json        the headline numbers (for the README)
    confused_pairs.csv  every (true, predicted) mix-up, most frequent first
    per_class.csv       accuracy for each class
    mistakes.png        16 confident mistakes with photo, true label, and guess

Usage (from the project root):
    python training/evaluate.py
    python training/evaluate.py --checkpoint training/checkpoints/quick/best.pt
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import load_config, project_path
from dataset import CarDataset
from inference import display_name, eval_transform, load_trained_model, pick_device


@torch.no_grad()
def predict_all(model, loader, device):
    """Run the model on every test image. Returns (top-5 class indices, top-5 probabilities) per image."""
    all_indices, all_probs = [], []
    use_amp = device.type == "cuda"
    for images, _ in tqdm(loader, desc="Testing", unit="batch"):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(images)
        # .float() converts back from bfloat16 before softmax, for accurate probabilities.
        top = logits.float().softmax(dim=1).topk(5, dim=1)
        all_indices.append(top.indices.cpu())
        all_probs.append(top.values.cpu())
    # torch.cat joins the per-batch results into one big tensor: [num_images, 5].
    return torch.cat(all_indices), torch.cat(all_probs)


def accuracy_report(test, class_names, top5_indices):
    """Add prediction columns to the test table and compute accuracy at each level."""
    classes = pd.Series(class_names)
    # Column 0 of top5_indices is the best guess for each image. Map number -> class name.
    test["pred_class"] = classes[top5_indices[:, 0].numpy()].values

    # Look up make and model of each predicted class from classes.csv-style info,
    # by splitting the class name the same way build_subset.py built it.
    # Example: "honda_civic_2001-2003" -> make "honda", model "civic".
    def make_and_model(name):
        parts = name.split("_")
        return parts[0], "_".join(parts[1:-1])

    pred_make_model = test["pred_class"].map(make_and_model)
    true_make_model = test["class_name"].map(make_and_model)

    # Convert each image's true class name to its number, to compare against the top 5.
    class_to_index = {name: i for i, name in enumerate(class_names)}
    true_index = torch.tensor(test["class_name"].map(class_to_index).values)

    test["correct_top1"] = test["pred_class"] == test["class_name"]
    test["correct_top5"] = (top5_indices == true_index.unsqueeze(1)).any(dim=1).numpy()

    return {
        "test_images": len(test),
        "classes": len(class_names),
        "exact_top1": float(test["correct_top1"].mean()),   # mean of True/False = fraction True
        "exact_top5": float(test["correct_top5"].mean()),
        "make_model_top1": float((pred_make_model == true_make_model).mean()),
        "make_top1": float((pred_make_model.str[0] == true_make_model.str[0]).mean()),  # .str[0] = first item of each tuple
    }


def confused_pairs(test):
    """Count every (true class, predicted class) mix-up, most frequent first."""
    wrong = test[~test["correct_top1"]]
    pairs = wrong.groupby(["class_name", "pred_class"]).size().rename("count").reset_index()
    return pairs.sort_values("count", ascending=False)


def per_class_accuracy(test):
    """Top-1 accuracy for each class, worst first."""
    return (test.groupby("class_name")["correct_top1"]
            .agg(accuracy="mean", test_images="size")
            .reset_index()
            .sort_values("accuracy"))


def save_mistakes_grid(test, dataset_dir, out_path, count=16):
    """Save the model's most CONFIDENT mistakes as a picture grid.

    Confident mistakes are the most informative: either the model has learned
    something wrong, or the photo's label is wrong. Looking at them is how you
    find out which.
    """
    worst = test[~test["correct_top1"]].nlargest(count, "pred_confidence")
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    # zip pairs each subplot with one mistake; axes.flat walks the 4x4 grid row by row.
    for ax, (_, row) in zip(axes.flat, worst.iterrows()):
        with Image.open(dataset_dir / row["path"]) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"True: {display_name(row['class_name'])}\n"
                     f"Guess: {display_name(row['pred_class'])} ({row['pred_confidence']:.0%})",
                     fontsize=9, loc="left")
        ax.axis("off")
    for ax in list(axes.flat)[len(worst):]:  # hide unused cells if there were fewer mistakes
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description="Evaluate the trained model on the test set.")
    parser.add_argument("--checkpoint", default=project_path(config["paths"]["checkpoints_dir"]) / "best.pt")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    args = parser.parse_args()

    device = pick_device(args.device)
    model, info = load_trained_model(args.checkpoint, device)
    class_names = info["class_names"]
    print(f"Model from epoch {info['epoch']} (val top-1 {info['val_top1']:.1%}), on {device}")

    splits_dir = project_path(config["paths"]["splits_dir"])
    dataset_dir = project_path(config["paths"]["dataset_dir"])
    test_csv = splits_dir / "test.csv"

    # shuffle=False keeps results in the same order as test.csv, so row i of the
    # predictions belongs to row i of the table.
    test_set = CarDataset(test_csv, dataset_dir, class_names, eval_transform(info))
    loader = DataLoader(test_set, batch_size=256, shuffle=False,
                        num_workers=config["training"]["num_workers"], pin_memory=device.type == "cuda")
    top5_indices, top5_probs = predict_all(model, loader, device)

    test = pd.read_csv(test_csv)
    test["pred_confidence"] = top5_probs[:, 0].numpy()
    metrics = accuracy_report(test, class_names, top5_indices)

    out_dir = project_path(config["paths"]["runs_dir"]) / "evaluate"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pairs = confused_pairs(test)
    pairs.to_csv(out_dir / "confused_pairs.csv", index=False)
    per_class = per_class_accuracy(test)
    per_class.to_csv(out_dir / "per_class.csv", index=False)
    save_mistakes_grid(test, dataset_dir, out_dir / "mistakes.png")

    print(f"\n=== Test set: {metrics['test_images']:,} images, {metrics['classes']:,} classes ===")
    print(f"Exact class    top-1 {metrics['exact_top1']:.1%}   top-5 {metrics['exact_top5']:.1%}")
    print(f"Make + model   top-1 {metrics['make_model_top1']:.1%}   (year range ignored)")
    print(f"Make only      top-1 {metrics['make_top1']:.1%}")

    print("\nMost-confused pairs (true -> guessed, times):")
    for _, row in pairs.head(15).iterrows():
        print(f"  {display_name(row['class_name']):<40} -> {display_name(row['pred_class']):<40} {row['count']}")

    print("\nHardest classes (lowest top-1):")
    for _, row in per_class.head(10).iterrows():
        print(f"  {display_name(row['class_name']):<40} {row['accuracy']:.0%} of {row['test_images']}")

    print(f"\nSaved metrics, CSVs, and mistakes.png to {out_dir}")


if __name__ == "__main__":
    main()
