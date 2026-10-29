# Car Identifier

Point your phone at a car and it tells you what it is: make, model, generation and whether
it's the facelift. For example: *"Toyota Highlander, XU50, facelift (2017-2019)"*.

It works on a single photo or on a live camera feed, where it draws a box around every car
it sees. If it isn't sure about a detail, it stops before that detail. So you might get
"Toyota Highlander" without a generation, or just "Not sure", instead of a confident wrong
answer.

## At a glance

- 95.8% make + model accuracy (97.7% make only) on 37,964 test photos, across 1,465 classes
- Gets the generation and facelift right 96.9% of the time when the make + model is right
- Live camera mode on a phone: boxes and names every car in view in real time
- Two models: a Faster R-CNN detector finds the cars, and a ConvNeXt-Tiny classifier I
  trained names them
- Trained in Python/PyTorch on an RTX 4090, served by a Java 17 Spring Boot app through
  ONNX Runtime
- 382k training photos: VMMRdb plus about 130k Wikimedia Commons photos that I downloaded,
  cleaned and labeled automatically
- Hand-written table of about 2,100 car generations and facelifts for 560+ models, checked
  against the US government's NHTSA vehicle list

## How it works

1. A car detector (Faster R-CNN, pretrained) finds each car in the picture and crops it out.
2. A classifier (ConvNeXt-Tiny, which I trained) looks at each crop and names the car.
3. Both models are exported to ONNX and run inside a Java server, which also serves the web
   page you use on your phone or PC.

Training is done in Python with PyTorch. The app is Java.

## Results

Run 5, the current model: 1,465 classes, trained on 382,635 photos.

| Test | Accuracy |
|---|---|
| Make only | 97.7% |
| Make + model | 95.8% |
| Exact class (make, model, generation, facelift) | 92.8% |
| Generation/facelift, when make + model is right | 96.9% |

The test set is 37,964 photos the model never saw during training.

There's also a fixed exam set of VMMRdb photos that every training run is graded on, so runs
can be compared fairly. Run 5 scores 97.2% make + model on it, the same as run 4, even though
run 5 learned 343 extra car looks. On Wikimedia Commons photos alone (harder, more varied
pictures) it scores 92.5%.

To keep the test honest, photos from the same photographer on the same day all go in the same
split. Otherwise the model could be tested on near-copies of photos it trained on.

## Tech stack

- **Training:** Python 3.13, PyTorch, timm (ConvNeXt-Tiny), torchvision (Faster R-CNN), ONNX
- **Server:** Java 17, Spring Boot 4, Jetty, ONNX Runtime, built with Maven
- **Web page:** GWT + SmartGWT (Java compiled to JavaScript)
- **Phone access:** Tailscale, which gives the server a private HTTPS link so the phone
  browser allows camera access
- **Hardware:** trained on an RTX 4090 (24 GB), Ryzen 9 9950X3D, 61 GB RAM. One epoch takes
  about 5.5 minutes and a full training run about 2 hours. The app itself runs fine without
  a GPU.

## Data

- **VMMRdb** (Vehicle Make and Model Recognition dataset): about 285k photos. Used for
  training only and not shared here.
- **Wikimedia Commons:** about 130k freely licensed photos (CC0, public domain, CC BY,
  CC BY-SA). A script downloads them, the car detector throws out ones without a clear car
  (interiors, engines, badges), duplicates are removed, and they are labeled automatically.
  Only photos where the clues disagree get checked by hand. Each photo's author, license and
  source page are saved, and I plan to publish these photos with their labels as a public
  dataset.
- **Label table** (`training/generations.csv`): about 2,100 car "looks" (a generation or a
  facelift, with its years) for 560+ models, written by hand.
- **NHTSA vPIC:** the US government's vehicle list, used to check which models are still
  missing.

A full list of image sources is coming soon.

## Running it

You need Java 17+ and Maven. The trained model files go in `models/` (they're too big for
git). `training/export_onnx.py` writes them there.

```
cd app
mvn clean package
java -jar server/target/car-identifier-server-0.1.0-SNAPSHOT.jar
```

Then open http://localhost:8080. The server only listens on your own PC. To use it from your
phone, install Tailscale on both devices and run:

```
tailscale serve --bg 8080
```

To train your own model, see `training/requirements.txt` for setup. The settings are in
`training/config.yaml`.

## Folders

| Folder | What's in it |
|---|---|
| `training/` | Python: data download and cleanup, label table, training, evaluation, ONNX export |
| `app/` | Java: Spring Boot server (`server/`) and web page (`client/`) |
| `models/` | Exported model files the server loads (not in git) |
| `docs/` | Notes |

## Screenshots

Coming soon.

## Status

Work in progress. Next up: more photos of 2015+ cars (the current model is weaker on new
cars because there are fewer photos of them), then adding every car model sold in the US
since 1990.
