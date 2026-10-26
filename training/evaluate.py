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
        make + model  Honda Civic == Honda Civic (any years; trims like civic_coupe merged)
        exact class   Honda Civic (2001-2003) exactly
    Make + model is scored two ways: the car of the single best class, and the car
    with the most probability after adding up its classes (how the app answers).
    The second one is the number to compare between runs with different classes,
    e.g. run 1 (year buckets) vs run 2 (generations).
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

from common import base_model, load_config, load_generations, project_path
from dataset import CarDataset
from inference import display_name, eval_transform, load_trained_model, pick_device


def car_of_class(class_name, table_models):
    """Which car (make, base model) a class belongs to, ignoring years, generation and trim.

        "honda_civic_2001-2003"        -> ("honda", "civic")
        "toyota_corolla_s_2005-2007"   -> ("toyota", "corolla")   trim merged (run-1 classes)

    WHY merge trims here: run 1 had separate classes for corolla and corolla_s. If
    "corolla_s" for a corolla photo counted as the wrong car, run 1 would look worse
    than run 2 for a reason that has nothing to do with how well it sees. Scoring
    both runs with the same trim rule (common.base_model) keeps the comparison fair.
    Models that aren't in generations.csv keep their folder name.
    """
    parts = class_name.split("_")
    make, model = parts[0], "_".join(parts[1:-1])
    return make, base_model(model, table_models.get(make, [])) or model


@torch.no_grad()
def predict_all(model, loader, device, class_to_car, num_cars):
    """Run the model on every test image.

    Returns:
        top-5 class indices and top-5 probabilities per image, and
        the best CAR per image after adding up each car's classes.

    WHY add up per car: this is how the app will answer "which car is it?". If the
    model says Highlander XU40 pre 40% and XU40 facelift 35%, its best single class
    is only 40% sure, but it's 75% sure the car is a Highlander. The best class and
    the best car can even disagree, so the app's way needs its own score.

    HOW: index_add_ adds column j of `probs` into column class_to_car[j] of `per_car`.
    Like: for each class j: per_car[:, car(j)] += probs[:, j]  - done for all j at once.
    """
    all_indices, all_probs, all_cars = [], [], []
    use_amp = device.type == "cuda"
    class_to_car = class_to_car.to(device)
    for images, _ in tqdm(loader, desc="Testing", unit="batch"):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(images)
        # .float() converts back from bfloat16 before softmax, for accurate probabilities.
        probs = logits.float().softmax(dim=1)
        top = probs.topk(5, dim=1)
        per_car = torch.zeros(len(probs), num_cars, device=device).index_add_(1, class_to_car, probs)
        all_indices.append(top.indices.cpu())
        all_probs.append(top.values.cpu())
        all_cars.append(per_car.argmax(dim=1).cpu())
    # torch.cat joins the per-batch results into one big tensor: [num_images, 5] (or [num_images]).
    return torch.cat(all_indices), torch.cat(all_probs), torch.cat(all_cars)


def accuracy_report(test, class_names, top5_indices, car_names, best_car, table_models):
    """Add prediction columns to the test table and compute accuracy at each level."""
    classes = pd.Series(class_names)
    # Column 0 of top5_indices is the best guess for each image. Map number -> class name.
    test["pred_class"] = classes[top5_indices[:, 0].numpy()].values

    true_car = test["class_name"].map(lambda name: car_of_class(name, table_models))
    pred_car = test["pred_class"].map(lambda name: car_of_class(name, table_models))
    summed_car = pd.Series(car_names)[best_car.numpy()].values   # the app's way: probabilities added per car

    # Convert each image's true class name to its number, to compare against the top 5.
    class_to_index = {name: i for i, name in enumerate(class_names)}
    true_index = torch.tensor(test["class_name"].map(class_to_index).values)

    test["correct_top1"] = test["pred_class"] == test["class_name"]
    test["correct_top5"] = (top5_indices == true_index.unsqueeze(1)).any(dim=1).numpy()
    car_right = pred_car == true_car

    return {
        "test_images": len(test),
        "classes": len(class_names),
        "exact_top1": float(test["correct_top1"].mean()),   # mean of True/False = fraction True
        "exact_top5": float(test["correct_top5"].mean()),
        "make_model_top1": float(car_right.mean()),
        "make_model_summed_top1": float((summed_car == true_car.values).mean()),
        # Of the photos where the best class is the right car, how often is it also the right
        # years/look? This isolates "can it tell generations apart" from "can it tell cars apart".
        "class_given_car": float(test["correct_top1"].sum() / car_right.sum()),
        "make_top1": float((pred_car.str[0] == true_car.str[0]).mean()),  # .str[0] = first item of each tuple
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
    # An older model must be tested on the exam it was built for (its own classes),
    # e.g. --splits-dir data/splits_run1_year_buckets for run 1.
    parser.add_argument("--splits-dir", default=config["paths"]["splits_dir"])
    parser.add_argument("--out-name", default="evaluate",
                        help="folder under training/runs/ for the results, e.g. evaluate_run1")
    args = parser.parse_args()

    device = pick_device(args.device)
    model, info = load_trained_model(args.checkpoint, device)
    class_names = info["class_names"]
    print(f"Model from epoch {info['epoch']} (val top-1 {info['val_top1']:.1%}), on {device}")

    splits_dir = project_path(args.splits_dir)
    dataset_dir = project_path(config["paths"]["dataset_dir"])
    test_csv = splits_dir / "test.csv"

    # Give every car (make + base model) a number, and note which car each class belongs to.
    table_models = load_generations().groupby("make")["model"].apply(list).to_dict()
    cars = [car_of_class(name, table_models) for name in class_names]
    car_names = sorted(set(cars))
    car_to_index = {car: i for i, car in enumerate(car_names)}
    class_to_car = torch.tensor([car_to_index[car] for car in cars])

    # shuffle=False keeps results in the same order as test.csv, so row i of the
    # predictions belongs to row i of the table.
    test_set = CarDataset(test_csv, dataset_dir, class_names, eval_transform(info))
    loader = DataLoader(test_set, batch_size=256, shuffle=False,
                        num_workers=config["training"]["num_workers"], pin_memory=device.type == "cuda")
    top5_indices, top5_probs, best_car = predict_all(model, loader, device, class_to_car, len(car_names))

    test = pd.read_csv(test_csv)
    test["pred_confidence"] = top5_probs[:, 0].numpy()
    metrics = accuracy_report(test, class_names, top5_indices, car_names, best_car, table_models)

    out_dir = project_path(config["paths"]["runs_dir"]) / args.out_name
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
    print(f"Make + model   top-1 {metrics['make_model_top1']:.1%}   (best class's car; trims merged)")
    print(f"Make + model   top-1 {metrics['make_model_summed_top1']:.1%}   (probabilities added per car, "
          f"like the app)  <- compare runs with this")
    print(f"Class | car    top-1 {metrics['class_given_car']:.1%}   (right years/look, when the car is right)")
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
