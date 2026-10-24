"""
Turn the train/val/test CSV files into images the model can learn from.

THE BIG PICTURE
---------------
build_subset.py produced CSV "piles" of (image path, class name) rows. A model
can't read a file path or a class name. It needs:

  - the image as a TENSOR: a grid of numbers shaped [3, 224, 224]
    (3 color channels - red, green, blue - each 224 x 224 pixels)
  - the answer as a NUMBER: the class index (0 .. 1167), not "honda_civic_2001-2003"

This file does that conversion. It has three parts:

  1. load_class_names() - the list of possible answers, in a fixed order.
     Position in the list = the number the model uses for that class.
  2. build_transforms() - the recipe that turns a photo into a tensor,
     including random "augmentation" for training (explained below).
  3. CarDataset - PyTorch's standard way to say "here is item #i".
     PyTorch's DataLoader (used in train.py) calls it in parallel to feed the GPU.

Run this file directly to save a preview image of augmented training samples:
    python training/dataset.py
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_class_names(splits_dir):
    """Return the list of class names, in the order that defines each class's number.

    WHY a fixed order matters: the model's output is just 1,168 numbers - one
    score per class, by position. Score #0 means "class #0". If the order ever
    changed between training and the Java app, every prediction would be
    labeled with the wrong car. classes.csv is sorted alphabetically by
    build_subset.py, and we never re-sort it, so the order is stable.
    """
    classes = pd.read_csv(Path(splits_dir) / "classes.csv")
    return classes["class_name"].tolist()


def build_transforms(image_size, mean, std, train):
    """Return the image -> tensor recipe for training (train=True) or evaluation (train=False).

    WHY two different recipes:

    TRAINING uses random AUGMENTATION. Each time the model sees a photo, it gets
    a slightly different version: zoomed into a random region, maybe mirrored,
    slightly brighter or darker. The model can't just memorize specific photos,
    so it's pushed to learn what a Civic actually looks like. This matters a lot
    for us: listing photos are tidy, but your phone photos will have different
    framing, lighting, and angles. Augmentation simulates that variety.

    EVALUATION (val/test/predict, and the Java app) uses NO randomness. The same
    photo must always produce exactly the same tensor, so scores are repeatable
    and Java can reproduce the result exactly.

    WHY resize to a square without cropping (for evaluation): VMMRdb photos are
    mostly 4:3 landscape. Cropping the center square would cut off the front and
    back of the car - the grille and tail lights, which are some of the most
    useful clues. Squashing the whole photo to 224x224 keeps everything visible.
    The model learns from squashed photos during training too (RandomResizedCrop
    also outputs a square), so it's used to the stretched look. It's also simple
    to reproduce exactly in Java, which Phase 2.5 depends on.

    WHY normalize with mean/std: the pretrained model was trained on images where
    each color channel was shifted and scaled by these exact numbers. Feeding it
    images prepared any other way would be like changing the units of its input
    (inches instead of cm) - it would still run, but give worse answers.
    """
    normalize = [
        # PIL image (H x W x RGB, values 0-255) -> tensor [3, H, W] with values 0.0-1.0.
        # Note the order change: PyTorch wants channels FIRST ("CHW" layout).
        transforms.ToTensor(),
        # For each channel c: value = (value - mean[c]) / std[c]
        transforms.Normalize(mean=mean, std=std),
    ]

    if train:
        return transforms.Compose([
            # Pick a random rectangle covering 50-100% of the photo, then resize it to a square.
            # Simulates the car being closer, further, or off-center in the frame.
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            # Mirror the photo half the time: a car facing left is the same car facing right.
            transforms.RandomHorizontalFlip(),
            # Randomly vary brightness/contrast/saturation by up to 30%: different light and cameras.
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            *normalize,  # * unpacks the list, like passing its two items one by one
        ])

    return transforms.Compose([
        # (height, width) = the whole photo squashed to a square. Bilinear resizing
        # is the default and the easiest method to match exactly in Java.
        transforms.Resize((image_size, image_size)),
        *normalize,
    ])


class CarDataset(Dataset):
    """One split (train, val, or test) as a PyTorch Dataset.

    A PyTorch Dataset is like a read-only List: it must answer two questions,
      __len__()        -> how many items are there?        (Java: size())
      __getitem__(i)   -> give me item i                   (Java: get(i))
    Python calls these "dunder" (double-underscore) methods; len(ds) and ds[i]
    call them for you.

    It loads images LAZILY: only the CSV (paths + labels) is kept in memory, and
    each photo is read from disk when it's asked for. All 194K images at once
    would need tens of GB of RAM.
    """

    def __init__(self, csv_path, dataset_dir, class_names, transform):
        rows = pd.read_csv(csv_path)
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform

        # Map each class name to its number, e.g. {"acura_cl_1998-2000": 0, ...}.
        # This is a dict comprehension - like building a HashMap in a loop:
        #   for (i, name) in enumerate(class_names): map.put(name, i)
        class_to_index = {name: i for i, name in enumerate(class_names)}

        # Store plain Python lists rather than the DataFrame: they're fast to index,
        # and cheap to send to the worker processes the DataLoader starts.
        self.paths = rows["path"].tolist()
        self.labels = [class_to_index[name] for name in rows["class_name"]]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # convert("RGB") guarantees 3 channels: a few VMMRdb photos are grayscale (1 channel),
        # and PNGs can have a 4th transparency channel. The model always expects exactly 3.
        with Image.open(self.dataset_dir / self.paths[index]) as img:
            image = self.transform(img.convert("RGB"))
        return image, self.labels[index]


def save_augmentation_preview(out_path, num_images=16):
    """Save a grid of augmented training images, so you can SEE what the model is shown."""
    # Imported here because only this preview needs them.
    import random

    from torchvision.utils import save_image

    from common import load_config, project_path

    config = load_config()
    splits_dir = project_path(config["paths"]["splits_dir"])
    # The standard ImageNet mean/std, the same values the pretrained model's config reports.
    # (train.py reads them from the model itself rather than hard-coding them.)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    dataset = CarDataset(
        csv_path=splits_dir / "train.csv",
        dataset_dir=project_path(config["paths"]["dataset_dir"]),
        class_names=load_class_names(splits_dir),
        transform=build_transforms(config["training"]["image_size"], mean, std, train=True),
    )

    # Show one photo 4 times (to see how augmentation varies it) plus 12 random photos.
    picks = [0] * 4 + random.sample(range(len(dataset)), num_images - 4)
    images = torch.stack([dataset[i][0] for i in picks])  # stack: list of [3,224,224] -> [16,3,224,224]

    # Undo the normalization so the colors look normal to a human: value * std + mean.
    # view(1, 3, 1, 1) reshapes the 3 numbers so they line up with the channel dimension
    # and apply to every pixel ("broadcasting").
    images = images * torch.tensor(std).view(1, 3, 1, 1) + torch.tensor(mean).view(1, 3, 1, 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(images, out_path, nrow=4)
    print(f"Dataset size: {len(dataset):,} training images, {len(set(dataset.labels)):,} classes")
    print(f"Saved augmentation preview: {out_path}")
    print("Top row = the same photo augmented 4 different ways.")


if __name__ == "__main__":
    from common import load_config, project_path

    runs_dir = project_path(load_config()["paths"]["runs_dir"])
    save_augmentation_preview(runs_dir / "dataset_preview.png")
