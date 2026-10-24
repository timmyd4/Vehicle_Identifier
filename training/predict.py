"""
Identify the car in one or more photos and print the top 5 guesses.

THE BIG PICTURE
---------------
This is the whole app in miniature: photo in, top-5 car names with confidence
out. It's the quickest way to try the model on your own phone photos, and the
Java app in app/ will do exactly the same steps:

    1. open the photo, convert to RGB
    2. resize to 224x224 and normalize (the evaluation recipe from dataset.py)
    3. run the model -> 1,168 scores ("logits")
    4. softmax -> turn the scores into probabilities that add up to 100%
    5. take the 5 highest

Usage (from the project root):
    python training/predict.py path/to/photo.jpg
    python training/predict.py photo1.jpg photo2.jpg
    python training/predict.py photo.jpg --device cpu     # run without the GPU
"""

import argparse

import torch
from PIL import Image

from common import load_config, project_path
from inference import display_name, eval_transform, load_trained_model, pick_device


def predict(model, info, image_path, device, top_k=5):
    """Return a list of (display name, confidence 0-1) for the top_k guesses."""
    with Image.open(image_path) as img:
        tensor = eval_transform(info)(img.convert("RGB"))  # shape [3, 224, 224]

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
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    args = parser.parse_args()

    device = pick_device(args.device)
    model, info = load_trained_model(args.checkpoint, device)

    for image_path in args.images:
        print(f"\n{image_path}")
        for rank, (name, confidence) in enumerate(predict(model, info, image_path, device), start=1):
            # A simple text bar: one # per 5% confidence.
            bar = "#" * round(confidence * 20)
            print(f"  {rank}. {name:<42} {confidence:6.1%}  {bar}")


if __name__ == "__main__":
    main()
