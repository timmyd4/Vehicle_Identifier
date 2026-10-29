# Car Identifier

Identifies a car's **make, model, generation and facelift** from a photo or a live camera feed,
e.g. *"Toyota Highlander, XU50, facelift (2017-2019)"*. It only answers as deep as it's sure
(make → model → generation → facelift) and says "Not sure" instead of guessing.

## Highlights

- **95.8% make + model accuracy** on a held-out test set of 37,964 photos (97.7% make only);
  when the car is right, the generation/facelift is right **96.9%** of the time. 1,465 classes.
  Adding 343 new looks cost nothing on the frozen exam every run takes (97.2% before and after).
- **Live camera mode:** boxes and names every car in view in real time.
- **Full pipeline, built from scratch:** a car detector (Faster R-CNN) crops each vehicle, and a
  ConvNeXt-Tiny classifier (PyTorch, trained on an RTX 4090) names it. The model is exported to
  ONNX and served by a **Java 17 Spring Boot** API with a GWT/SmartGWT web client.
- **380k photos:** VMMRdb plus ~130k freely licensed Wikimedia Commons photos, which I
  downloaded, filtered with the detector, de-duplicated and labeled automatically. Only photos
  where the clues disagree go to a human.
- **Hand-built label table** of ~2,100 car "looks" (generation + facelift years) for 560+ models,
  checked against the US government's NHTSA vehicle database for coverage.
- **Leak-free evaluation:** a frozen test set shared by every training run, and Commons photos
  split by photographer and day. Java-vs-Python parity tests and golden-value regression tests.

## Repo layout

| Folder | What's in it |
|---|---|
| `training/` | Python: data pipeline, label table (`generations.csv`), training, evaluation, ONNX export |
| `app/` | Java: Spring Boot server (`server/`) and GWT/SmartGWT web client (`client/`), Maven build |
| `docs/` | Notes |

## Screenshots

*Coming soon: photos and a short clip of the upload page and live camera mode.*

## Data

*A full list of where every image came from is coming soon.* In short:

- **VMMRdb** (Vehicle Make and Model Recognition dataset): ~285k photos, used for training only
  and not redistributed here.
- **Wikimedia Commons**: freely licensed photos (CC0 / public domain / CC BY / CC BY-SA). Each
  photo's author, license and source page are recorded. I plan to publish these photos with
  their labels and credits as a public dataset.
- **NHTSA vPIC**: the US model list used to measure coverage.

## Status

Work in progress: expanding to every car model sold in the US since 1990, then the next
training run.
