"""
Find the car in a photo and crop to it, so the classifier sees what it was trained on.

THE BIG PICTURE
---------------
Our classifier learned from used-car listing photos: landscape, with the car
filling most of the frame. Casual phone photos are different: often portrait,
with the car small and lots of background (garage, sky, driveway). When we
tested your Corolla, the classifier was unsure mostly because of that.

The fix is a second model, an OBJECT DETECTOR. A classifier answers "what is
this photo of?"; a detector answers "WHERE are the objects, and what kind are
they?" by returning boxes. So the pipeline becomes:

    phone photo -> detector: "car at (x1, y1, x2, y2)" -> crop to that box
                -> classifier: "Toyota Corolla (2007-2009)"

We don't train the detector. torchvision ships Faster R-CNN detectors already
trained on COCO, a dataset of everyday scenes with 80 kinds of objects
including "car" and "truck". Finding cars is a solved problem; telling a 2005
Civic from a 2005 Corolla is the part we trained ourselves.

The weights download once from download.pytorch.org (torchvision's official
source) and are cached in ~/.cache/torch.

Run directly to see what the detector finds (saves a copy with boxes drawn):
    python training/detect.py data/my_photos/corolla_2009.jpg
"""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision.models import detection
from torchvision.transforms.functional import to_tensor

from common import load_config, project_path

# Maps the model name in config.yaml to torchvision's builder function and its weights.
# DEFAULT = the best COCO weights torchvision provides for that model.
DETECTORS = {
    "fasterrcnn_mobilenet_v3_large_fpn": (
        detection.fasterrcnn_mobilenet_v3_large_fpn,
        detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
    ),
}


def open_photo(path):
    """Open a photo the right way up, as RGB.

    Phones often save the pixels sideways and store a separate EXIF "Orientation"
    flag that says how to rotate them for display. Image viewers apply the flag;
    PIL doesn't, unless we call exif_transpose. Without this, some phone photos
    would reach the models rotated 90 degrees. (Java must do the same.)
    """
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def load_detector(model_name, device):
    """Return (detector model, list of COCO category names by label number)."""
    build, weights = DETECTORS[model_name]
    model = build(weights=weights)
    model.to(device)
    model.eval()
    # weights.meta["categories"] is a list where position = label number,
    # e.g. categories[3] == "car", categories[8] == "truck".
    return model, weights.meta["categories"]


@torch.no_grad()
def find_cars(detector, categories, image, device, labels=("car", "truck"), min_score=0.5):
    """Return detected vehicles as dicts {box, score, label}, biggest first.

    box = (left, top, right, bottom) in pixels of the original image.
    """
    # Unlike our classifier, torchvision detectors take a LIST of images with
    # values 0.0-1.0 and do their own resizing and normalization internally.
    output = detector([to_tensor(image).to(device)])[0]

    # output has three parallel tensors, one entry per detected object:
    #   boxes [N, 4], labels [N] (COCO category numbers), scores [N] (confidence 0-1)
    cars = []
    for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
        name = categories[label]
        if name in labels and score >= min_score:
            cars.append({"box": tuple(box.tolist()), "score": score.item(), "label": name})

    # Biggest box first: in a photo of "my car", the car you meant is usually the
    # largest one, while cars in the background are smaller.
    def area(car):
        left, top, right, bottom = car["box"]
        return (right - left) * (bottom - top)

    return sorted(cars, key=area, reverse=True)


def crop_to_car(image, box, margin):
    """Crop the image to the box, widened by `margin` (a fraction of the box size) on each side.

    WHY a margin: listing photos (what the classifier learned from) show the whole
    car with some surroundings. A crop that's exactly the box could cut off a
    mirror or bumper, and would look tighter than anything seen in training.
    """
    left, top, right, bottom = box
    pad_x = (right - left) * margin
    pad_y = (bottom - top) * margin
    # max/min keep the widened box inside the photo's edges.
    return image.crop((
        int(max(0, left - pad_x)),
        int(max(0, top - pad_y)),
        int(min(image.width, right + pad_x)),
        int(min(image.height, bottom + pad_y)),
    ))


def detector_from_config(device):
    """Load the detector and its settings from config.yaml. Returns (detector, categories, settings)."""
    settings = load_config()["detector"]
    detector, categories = load_detector(settings["model"], device)
    return detector, categories, settings


def main():
    parser = argparse.ArgumentParser(description="Show which vehicles the detector finds in photos.")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    detector, categories, settings = detector_from_config(device)
    out_dir = project_path(load_config()["paths"]["runs_dir"]) / "detect"
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in args.images:
        image = open_photo(path)
        cars = find_cars(detector, categories, image, device, settings["labels"], settings["min_score"])
        print(f"\n{path}: {len(cars)} vehicle(s) found")

        # Draw every box; the first (biggest) one is the one predict.py will use.
        draw = ImageDraw.Draw(image)
        line = max(3, image.width // 200)  # thicker lines on big photos so they stay visible
        for i, car in enumerate(cars):
            color = "#2a78d6" if i == 0 else "#898781"   # blue = used, gray = ignored
            draw.rectangle(car["box"], outline=color, width=line)
            print(f"  {'USED ' if i == 0 else 'other'} {car['label']:<5} {car['score']:.0%}  box {tuple(round(v) for v in car['box'])}")

        out_path = out_dir / Path(path).name
        image.save(out_path, quality=90)
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
