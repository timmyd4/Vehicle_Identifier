"""
Export the trained classifier and the car detector to ONNX for the Java app, then verify them.

THE BIG PICTURE
---------------
PyTorch models only run inside Python. ONNX (Open Neural Network Exchange) is a
standard file format for models - think of it as a .jar for neural networks:
the structure and the weights in one file that other languages can run.
The Java app uses Microsoft's ONNX Runtime library to run these files.

This script writes four files to models/ - everything the Java app needs:

    car_model.onnx       our classifier: 224x224 image in -> 1,168 scores out
    detector.onnx        the car detector: photo in -> boxes, labels, scores out
    labels.json          class number -> class name + readable name ("Honda Civic (2001-2003)")
    preprocessing.json   EXACTLY how to prepare a photo for each model. If Java prepares
                         images even slightly differently (wrong size, BGR instead of RGB,
                         wrong mean/std), the model still runs but gives worse answers,
                         with no error. Writing it down once avoids that.

Then it VERIFIES the export: the same images go through PyTorch and through
ONNX Runtime, and the answers must match. If this passes, any difference the
Java app shows later must come from Java's own image preparation, not from the
model files - which makes Java bugs much easier to find.

Usage (from the project root):
    python training/export_onnx.py
"""

import json
import sys
import warnings

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from torchvision.transforms.functional import to_tensor

from common import load_config, project_path
from detect import crop_to_car, load_detector, open_photo
from inference import display_name, eval_transform, load_trained_model

# PyTorch's exporter prints emoji in its progress messages. The default Windows
# console encoding (cp1252) can't print emoji and would crash, so switch to UTF-8.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_classifier(model, image_size, out_path):
    """Export the classifier with input "input" [batch, 3, H, W] and output "logits" [batch, classes].

    dynamo=True is PyTorch's current exporter. It reads the model's code and records
    every operation as an ONNX graph.

    dynamic_shapes marks dimension 0 (the batch size) as "can vary". Without it, the
    file would only accept exactly the batch size of the example input below.
    Java will send batches of 1, but live mode might send several crops at once.

    external_data=False keeps the weights INSIDE the .onnx file, instead of a separate
    .onnx.data file next to it - one file is simpler to copy and to load from Java.
    """
    example = torch.randn(2, 3, image_size, image_size)  # any values work; only the shape matters
    torch.onnx.export(
        model, (example,), out_path,
        input_names=["input"], output_names=["logits"],
        dynamic_shapes=({0: torch.export.Dim("batch")},),
        dynamo=True, external_data=False,
    )


def export_detector(detector, example_photo, out_path):
    """Export the detector with input "image" [3, H, W] and outputs "boxes", "labels", "scores".

    WHY THE OLDER EXPORTER (dynamo=False): the detector's last step, NMS
    ("non-maximum suppression", which merges overlapping boxes), keeps a number of
    boxes that depends on the image. The new exporter can't handle that yet.

    The older exporter works by TRACING: it runs the model once on an example and
    records the operations it actually performed. So the example must be a REAL
    photo with a car in it. With a blank image, no boxes survive, and the recording
    misses the steps a real photo needs (we hit exactly that bug while testing).

    dynamic_axes marks which dimensions can vary: photo height/width, and the
    number of boxes found (n).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tracing prints many harmless warnings about Python control flow
        torch.onnx.export(
            detector, ([example_photo],), out_path,
            input_names=["image"], output_names=["boxes", "labels", "scores"],
            dynamic_axes={"image": {1: "height", 2: "width"},
                          "boxes": {0: "n"}, "labels": {0: "n"}, "scores": {0: "n"}},
            dynamo=False, opset_version=17,
        )


def write_labels(class_names, splits_dir, out_path):
    """Write labels.json: a list where position i describes class i.

    The class ORDER comes from the checkpoint (the order the model was trained with),
    never re-sorted. Details like make/model/years come from classes.csv.
    """
    details = pd.read_csv(splits_dir / "classes.csv").set_index("class_name")
    labels = []
    for index, name in enumerate(class_names):
        row = details.loc[name]  # look up this class's row by name
        labels.append({
            "index": index,
            "class_name": name,
            "display_name": display_name(name),
            "make": row["make"],
            "model": row["model"],
            "year_start": int(row["year_start"]),  # int(): numpy integers aren't JSON-serializable
            "year_end": int(row["year_end"]),
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)


def write_preprocessing(info, det_settings, det_categories, out_path):
    """Write preprocessing.json: the exact recipe for preparing a photo, for Python and Java alike."""
    recipe = {
        "photo": {
            "apply_exif_orientation": True,   # rotate per the phone's EXIF flag before anything else
            "color_order": "RGB",             # NOT BGR (Java's BufferedImage often stores BGR internally)
        },
        "detector": {
            "file": "detector.onnx",
            "input": {
                "name": "image",
                "layout": "CHW",              # [3 channels, height, width], no batch dimension
                "dtype": "float32",
                "value_range": "pixel / 255, so 0.0 to 1.0",
                "size": "the whole photo at its original size (the model resizes internally)",
            },
            "outputs": {
                "boxes": "[n, 4] float32: left, top, right, bottom in photo pixels",
                "labels": "[n] int64: COCO category id",
                "scores": "[n] float32: confidence 0-1",
            },
            # COCO category ids for the labels we accept, e.g. {"car": 3, "truck": 8}.
            "accept_labels": {name: det_categories.index(name) for name in det_settings["labels"]},
            "min_score": det_settings["min_score"],
            "pick": "the accepted box with the largest area",
            "crop_margin": det_settings["crop_margin"],
            "crop": ("widen the box by crop_margin x its width/height on each side, clamp to the photo, "
                     "truncate to whole pixels; if no box is accepted, use the whole photo"),
        },
        "classifier": {
            "file": "car_model.onnx",
            "input": {"name": "input", "layout": "NCHW", "dtype": "float32"},  # [batch, 3, height, width]
            "resize": {
                "width": info["image_size"],
                "height": info["image_size"],
                "keep_aspect_ratio": False,   # squash the whole crop to a square, no center crop
                "interpolation": "bilinear",
                # torchvision's Resize on a PIL image calls PIL's resize, which smooths ("antialiases")
                # when shrinking. Java must shrink the same way to match closely.
                "reference_implementation": "PIL Image.resize(size, Image.BILINEAR)",
            },
            "scale": "pixel / 255, so 0.0 to 1.0",
            "mean": list(info["mean"]),       # then (value - mean[c]) / std[c] for each channel c = R, G, B
            "std": list(info["std"]),
            "output": {"name": "logits", "shape": ["batch", len(info["class_names"])],
                       "postprocess": "softmax over the class dimension -> probabilities"},
        },
        "labels_file": "labels.json",
        "trained_model": {"model_name": info["model_name"], "epoch": info["epoch"], "val_top1": info["val_top1"]},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2)


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
def verify_classifier(model, info, onnx_path, image_paths):
    """Run the same images through PyTorch and ONNX Runtime; their answers must match."""
    session = ort.InferenceSession(onnx_path)
    transform = eval_transform(info)
    batch = torch.stack([transform(open_photo(p)) for p in image_paths])

    with torch.no_grad():
        torch_logits = model(batch).numpy()
    onnx_logits = session.run(["logits"], {"input": batch.numpy()})[0]

    # argsort sorts ascending; [:, ::-1] reverses each row, [:, :5] keeps the 5 highest.
    torch_top5 = np.argsort(torch_logits, axis=1)[:, ::-1][:, :5]
    onnx_top5 = np.argsort(onnx_logits, axis=1)[:, ::-1][:, :5]

    max_diff = float(np.abs(torch_logits - onnx_logits).max())
    same_top5 = float((torch_top5 == onnx_top5).all(axis=1).mean())
    print(f"  classifier: {len(image_paths)} images, max logit difference {max_diff:.2e}, "
          f"identical top-5: {same_top5:.0%}")
    # Tiny differences (~1e-5) are normal: the two engines add numbers up in slightly
    # different orders. Anything near 1e-2 would mean the export is broken.
    return max_diff < 1e-3 and same_top5 == 1.0


def box_overlap(a, b):
    """IoU ("intersection over union"): 1.0 = identical boxes, 0.0 = no overlap."""
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    area = lambda box: (box[2] - box[0]) * (box[3] - box[1])
    return inter / (area(a) + area(b) - inter)


def chosen_box(boxes, labels, scores, accept_ids, min_score):
    """Apply the same rule as predict.py: the biggest accepted box, or None."""
    keep = [(b, l) for b, l, s in zip(boxes, labels, scores) if l in accept_ids and s >= min_score]
    if not keep:
        return None
    return max(keep, key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]))


def verify_detector(detector, onnx_path, image_paths, accept_ids, min_score, model, info, crop_margin):
    """Check that ONNX leads to the same FINAL ANSWER as PyTorch.

    WHY not compare every box: the detector also returns many faint guesses (5-50%
    confidence) that we throw away. Their exact count can differ slightly between
    the two engines, which doesn't matter.

    WHY not demand pixel-identical boxes: when several overlapping boxes cover the
    same car (common with pickups, which COCO sometimes calls "car" and sometimes
    "truck"), tiny numeric differences between the engines can shift the final box
    by a few percent. The box is only used to crop, so what matters is:
      1. the same kind of vehicle is chosen, with the boxes mostly overlapping, and
      2. after cropping with each box, the classifier gives the same #1 answer.
    """
    session = ort.InferenceSession(onnx_path)
    transform = eval_transform(info)
    matches, overlaps = 0, []
    for path in image_paths:
        photo = open_photo(path)
        image = to_tensor(photo)
        with torch.no_grad():
            ref = detector([image])[0]
        boxes, labels, scores = session.run(None, {"image": image.numpy()})

        torch_pick = chosen_box(ref["boxes"].numpy(), ref["labels"].numpy(), ref["scores"].numpy(), accept_ids, min_score)
        onnx_pick = chosen_box(boxes, labels, scores, accept_ids, min_score)

        if torch_pick is None and onnx_pick is None:
            matches += 1  # both found no car: that's agreement too
        elif torch_pick is not None and onnx_pick is not None and torch_pick[1] == onnx_pick[1]:
            overlap = box_overlap(torch_pick[0], onnx_pick[0])
            overlaps.append(overlap)
            # Crop with each engine's box and compare the classifier's top answer.
            crops = torch.stack([transform(crop_to_car(photo, pick[0], crop_margin)) for pick in (torch_pick, onnx_pick)])
            with torch.no_grad():
                top1 = model(crops).argmax(dim=1)
            matches += overlap > 0.85 and top1[0] == top1[1]
    print(f"  detector:   {len(image_paths)} images, same final answer: {matches}/{len(image_paths)}, "
          f"lowest box overlap {min(overlaps) if overlaps else 1:.3f}")
    return matches == len(image_paths)


def main():
    config = load_config()
    paths = config["paths"]
    det_settings = config["detector"]
    models_dir = project_path(paths["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = project_path(paths["splits_dir"])
    dataset_dir = project_path(paths["dataset_dir"])

    # Export on the CPU: the exported graph doesn't depend on the device, and the
    # Java app will run on the CPU by default.
    cpu = torch.device("cpu")
    model, info = load_trained_model(project_path(paths["checkpoints_dir"]) / "best.pt", cpu)
    detector, categories = load_detector(det_settings["model"], cpu)

    # Test photos for tracing and verifying. A fixed random_state picks the same ones every run.
    test_paths = [dataset_dir / p for p in pd.read_csv(splits_dir / "test.csv")["path"].sample(64, random_state=0)]

    print("Exporting classifier...")
    export_classifier(model, info["image_size"], models_dir / "car_model.onnx")
    print("Exporting detector...")
    export_detector(detector, to_tensor(open_photo(test_paths[0])), models_dir / "detector.onnx")
    write_labels(info["class_names"], splits_dir, models_dir / "labels.json")
    write_preprocessing(info, det_settings, categories, models_dir / "preprocessing.json")

    print("\nVerifying PyTorch vs ONNX Runtime on test images:")
    accept_ids = [categories.index(name) for name in det_settings["labels"]]
    classifier_ok = verify_classifier(model, info, models_dir / "car_model.onnx", test_paths)
    detector_ok = verify_detector(detector, models_dir / "detector.onnx", test_paths[:24],
                                  accept_ids, det_settings["min_score"], model, info, det_settings["crop_margin"])

    print(f"\nWrote to {models_dir}:")
    for f in sorted(models_dir.iterdir()):
        print(f"  {f.name:<20} {f.stat().st_size / 1e6:7.1f} MB")

    if not (classifier_ok and detector_ok):
        print("\nVERIFICATION FAILED - don't use these files in the Java app.")
        sys.exit(1)  # non-zero exit code = failure, like System.exit(1)
    print("\nVerification passed.")


if __name__ == "__main__":
    main()
