"""
Write test data that proves the Java app prepares images exactly like Python.

THE BIG PICTURE
---------------
The classifier only gives good answers if every photo is resized and normalized
exactly as it was during training. Training ran in Python (PIL + torchvision);
the app runs in Java (app/server/.../Preprocessing.java), which re-implements
the same steps. If the two drift apart, accuracy drops silently: no error, just
worse answers. So we lock them together with a test:

  1. This script makes test images and runs them through the REAL Python
     evaluation recipe (dataset.build_transforms, the code used for val/test),
     saving the exact results ("expected outputs").
  2. A JUnit test (PreprocessingParityTest) runs the same images through Java
     and fails if a single byte or float differs.

The files go into app/server/src/test/resources/parity/, so Maven puts them on
the test classpath. Rerun this script whenever preprocessing changes in Python.

WHY synthetic images instead of real photos:
  - Random noise is the hardest case for a resize (every pixel differs from its
    neighbors), so any mismatch in the math shows up immediately.
  - PNG is lossless. JPEG decoders differ slightly between PIL and Java, which
    would hide resize differences behind decoding differences. Decoding is a
    separate concern; this test checks resize + normalize only.

Run (from the project root, with the venv active):
    python training/make_parity_fixture.py
"""

import json

import numpy as np
from PIL import Image

from common import project_path
from dataset import build_transforms

OUT_DIR = project_path("app/server/src/test/resources/parity")

# (name, width, height): odd sizes so the scale factors aren't whole numbers.
CASES = [
    ("shrink", 517, 389),   # a photo bigger than 224: the usual case (antialiased shrink)
    ("grow", 150, 97),      # a small crop smaller than 224: enlarging
]


def make_image(width, height, seed):
    """Random noise plus a smooth gradient, as an RGB PIL image.

    The gradient gives large smooth areas (like sky or car paint), the noise
    gives sharp detail. A fixed seed makes the image identical on every run.
    """
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(height, width, 3), dtype=np.int32)
    # np.linspace = evenly spaced values; [None, :, None] reshapes so it broadcasts
    # across rows and channels (a horizontal gradient from 0 to 255).
    gradient = np.linspace(0, 255, width)[None, :, None]
    pixels = ((noise + gradient) / 2).astype(np.uint8)
    # A uint8 array shaped (height, width, 3) becomes an RGB image automatically.
    return Image.fromarray(pixels)


def main():
    # Use the exact settings the Java app reads, from the exported config.
    prep = json.loads(project_path("models/preprocessing.json").read_text())
    clf = prep["classifier"]
    size = (clf["resize"]["width"], clf["resize"]["height"])
    # The real evaluation recipe: Resize -> ToTensor -> Normalize.
    transform = build_transforms(size[1], clf["mean"], clf["std"], train=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for seed, (name, width, height) in enumerate(CASES):
        image = make_image(width, height, seed)
        image.save(OUT_DIR / f"{name}_input.png")

        # Expected resized pixels. transform.transforms[0] is the Resize step on its
        # own; on a PIL image it calls PIL's resize(BILINEAR), which Java mirrors.
        resized = transform.transforms[0](image)
        resized.save(OUT_DIR / f"{name}_resized.png")

        # Expected classifier input: float32, CHW, little-endian raw bytes.
        # Java reads it back with a ByteBuffer set to LITTLE_ENDIAN.
        tensor = transform(image).numpy().astype("<f4")
        (OUT_DIR / f"{name}_tensor.f32").write_bytes(tensor.tobytes())

        print(f"{name}: {width}x{height} -> {size[0]}x{size[1]}, tensor {tensor.shape}")

    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
