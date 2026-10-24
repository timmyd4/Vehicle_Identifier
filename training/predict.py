"""
Identify the car in one or more photos and print the top 5 guesses.

THE BIG PICTURE
---------------
This is the whole app in miniature: photo in, top-5 car names with confidence
out. It's the quickest way to try the model on your own phone photos, and the
Java app in app/ will do exactly the same steps:

    1. open the photo the right way up (EXIF rotation), as RGB
    2. DETECT: find the biggest car/truck with the detector (detect.py)
    3. CROP to that box plus a small margin, so the car fills the frame like
       the listing photos the classifier learned from
       (if no car is found, use the whole photo)
    4. resize to 224x224 and normalize (the evaluation recipe from dataset.py)
    5. run the classifier -> 1,168 scores ("logits")
    6. softmax -> turn the scores into probabilities that add up to 100%
    7. take the 5 highest

Usage (from the project root):
    python training/predict.py path/to/photo.jpg
    python training/predict.py photo1.jpg photo2.jpg
    python training/predict.py photo.jpg --no-crop      # skip the detector, use the whole photo
    python training/predict.py photo.jpg --device cpu   # run without the GPU
"""

import argparse

import torch

from common import load_config, project_path
from detect import crop_to_car, detector_from_config, find_cars, open_photo
from inference import display_name, eval_transform, load_trained_model, pick_device


def classify(model, info, image, device, top_k=5):
    """Return a list of (display name, confidence 0-1) for the top_k guesses on a PIL image."""
    tensor = eval_transform(info)(image)  # shape [3, 224, 224]

    # The model expects a BATCH of images: shape [batch, 3, 224, 224].
    # unsqueeze(0) adds the batch dimension, making a batch of one.
    batch = tensor.unsqueeze(0).to(device)

    with torch.no_grad():  # not training, so don't track gradients
        logits = model(batch)[0]  # [0] = the first (only) image's 1,168 scores

    # SOFTMAX turns raw scores into probabilities: each becomes e^score / sum(e^all scores).
    # Big scores get most of the probability, and they all add up to 1.0.
    probabilities = logits.softmax(dim=0)
    top = probabilities.topk(top_k)  # .values = the 5 biggest probabilities, .indices = their class numbers

    return [(display_name(info["class_names"][i]), p.item())
            for p, i in zip(top.values, top.indices.tolist())]


def main():
    config = load_config()
    default_checkpoint = project_path(config["paths"]["checkpoints_dir"]) / "best.pt"

    parser = argparse.ArgumentParser(description="Identify the car in a photo.")
    parser.add_argument("images", nargs="+", help="one or more image files")
    parser.add_argument("--checkpoint", default=default_checkpoint, help="model file (default: best.pt)")
    parser.add_argument("--no-crop", action="store_true", help="skip the detector and classify the whole photo")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    args = parser.parse_args()

    device = pick_device(args.device)
    model, info = load_trained_model(args.checkpoint, device)
    if not args.no_crop:
        detector, categories, det_settings = detector_from_config(device)

    for image_path in args.images:
        print(f"\n{image_path}")
        image = open_photo(image_path)

        if not args.no_crop:
            cars = find_cars(detector, categories, image, device,
                             det_settings["labels"], det_settings["min_score"])
            if cars:
                best = cars[0]  # biggest vehicle in the photo
                image = crop_to_car(image, best["box"], det_settings["crop_margin"])
                print(f"  (found a {best['label']}, {best['score']:.0%} sure - cropped to it)")
            else:
                print("  (no car found - using the whole photo)")

        for rank, (name, confidence) in enumerate(classify(model, info, image, device), start=1):
            # A simple text bar: one # per 5% confidence.
            bar = "#" * round(confidence * 20)
            print(f"  {rank}. {name:<42} {confidence:6.1%}  {bar}")


if __name__ == "__main__":
    main()
