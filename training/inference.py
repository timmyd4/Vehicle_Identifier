"""
Load a trained model and turn class names into readable car names.

THE BIG PICTURE
---------------
train.py saves best.pt. Three scripts then need to USE that model:
evaluate.py (final exam), predict.py (one photo), and export_onnx.py (convert for Java).
They all need the same two things, so they live here once:

  1. load_trained_model() - rebuild the model from best.pt, along with the
     facts saved with it (class order, image size, mean/std).
  2. display_name() - "honda_civic_2001-2003" -> "Honda Civic (2001-2003)".

WHY rebuild the model from facts saved in the checkpoint (rather than from
config.yaml): config.yaml might have changed since this model was trained
(e.g. you tried a different image size). The checkpoint records how THIS model
was trained, so it always loads and prepares images correctly.
"""

import timm
import torch

from dataset import build_transforms


def pick_device(requested="auto"):
    """'auto' = the GPU if there is one, otherwise the CPU. Or force 'cpu' / 'cuda'."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_trained_model(checkpoint_path, device):
    """Return (model, info). model is ready for predictions; info holds the saved facts.

    info keys: class_names, model_name, image_size, mean, std, epoch, val_top1.
    """
    # weights_only=True: only plain data (tensors, numbers, strings, lists) may be
    # loaded, never arbitrary code. Loading untrusted .pt files without it is unsafe.
    info = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Build the same architecture with the same number of outputs, WITHOUT
    # downloading pretrained weights (pretrained=False) - we're about to overwrite
    # every weight with our trained ones anyway.
    model = timm.create_model(info["model_name"], pretrained=False, num_classes=len(info["class_names"]))
    model.load_state_dict(info.pop("model_state"))  # pop: take the weights out of info, it's big
    model.to(device)
    model.eval()  # prediction mode: no training-only randomness
    return model, info


def eval_transform(info):
    """The image -> tensor recipe this model expects (no augmentation)."""
    return build_transforms(info["image_size"], info["mean"], info["std"], train=False)


# A few makes don't look right with simple capitalization.
MAKE_NAMES = {
    "bmw": "BMW",
    "gmc": "GMC",
    "mercedes benz": "Mercedes-Benz",
    "landrover": "Land Rover",
    "mini": "MINI",
}


def display_name(class_name):
    """Turn a class name into something a person would read.

        "honda_civic_2001-2003"     -> "Honda Civic (2001-2003)"
        "volvo_xc90_2004-2006"      -> "Volvo XC90 (2004-2006)"
        "mercedes benz_e320_2001-2003" -> "Mercedes-Benz E320 (2001-2003)"

    It's a simple rule, so a few names come out imperfect (e.g. "328I" instead
    of "328i"). Phase 2.5 writes these names into labels.json, so the Java app
    uses exactly the same text without re-implementing this.
    """
    parts = class_name.split("_")
    make, years = parts[0], parts[-1]
    model_words = parts[1:-1]

    make_text = make_name(make)
    # Some model names use spaces as well as underscores ("accord_2 door"),
    # so join everything with spaces and then split into single words.
    words = " ".join(model_words).split()
    model_text = " ".join(format_model_word(w) for w in words)
    return f"{make_text} {model_text} ({years})"


def make_name(make):
    """"honda" -> "Honda", "mercedes benz" -> "Mercedes-Benz" (the make part of display_name)."""
    return MAKE_NAMES.get(make, make.title())  # dict.get(key, default) - like Map.getOrDefault


def format_model_word(word):
    """Capitalize one word of a model name.

    Model codes get uppercased: words with a digit (xc90 -> XC90, rav4 -> RAV4) and
    short words with no vowels (mdx -> MDX, tl -> TL, cts -> CTS).
    Everything else gets a capital first letter (cherokee -> Cherokee, fit -> Fit).
    """
    has_digit = any(ch.isdigit() for ch in word)
    is_letter_code = len(word) <= 3 and not any(ch in "aeiou" for ch in word)
    return word.upper() if has_digit or is_letter_code else word.title()
