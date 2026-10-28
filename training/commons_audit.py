"""
Step 5 of the Commons pipeline: an AUDIT of the automatic labels.

THE BIG PICTURE
---------------
commons_presort.py labeled ~81% of the Commons photos automatically, because the clues
(category, model, year) agreed. "Agreed" isn't the same as "right", so before training on
88k labels nobody looked at, we measure how often they're wrong:

  - take a RANDOM SAMPLE: 100 photos from each automatic rule (category, model, year)
  - a human clicks every photo whose label is wrong
  - wrong / 100 = that rule's error rate, give or take a few points

Random matters: picking photos yourself (the first 100, or ones that look interesting)
would measure those photos, not the 88k. The sample is also SHUFFLED and doesn't show
which rule labeled each photo, so you judge every photo the same way (a BLIND audit).

What counts as wrong: wrong make or model, wrong generation, wrong pre/facelift (only
when you can tell), or not a usable photo of the car's outside (interior, toy, drawing,
mostly hidden). The box shows which car the label is about. Not sure? Leave it.

Clicks are saved as you go (training/runs/review/commons_audit_answers.json), so you can
close the page and come back. Press Ctrl+C in the terminal when you're done: it prints
the error rate of each rule.

Usage (from the project root):
    python training/commons_audit.py
"""

import csv
import html
import json
import random
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

from common import load_config, project_path
from commons_presort import load_looks
from inference import display_name

PORT = 8766
PER_RULE = 100
RULES = ["category", "model", "year"]
ANSWERS_FILE = project_path("training/runs/review/commons_audit_answers.json")
PHASE_TEXT = {"single": "", "pre": "pre-facelift", "facelift": "facelift",
              "facelift2": "2nd facelift", "facelift3": "3rd facelift"}


def pick_sample(presort_rows):
    """PER_RULE random auto-labeled photos per rule, shuffled together. Same pick every run."""
    rng = random.Random(42)
    sample = []
    for rule in RULES:
        rows = [r for r in presort_rows if r["status"] == "auto" and r["reason"] == rule]
        sample += rng.sample(rows, min(PER_RULE, len(rows)))
    rng.shuffle(sample)  # mix the rules, so the page doesn't reveal which is which
    return sample


def label_text(row, years_of_look):
    """("toyota", "corolla", "E140", "pre") -> "Toyota Corolla E140 pre-facelift (2009-2010)"."""
    key = (row["make"], row["model"], row["generation"], row["phase"])
    first, last = years_of_look[key]
    name = display_name(f"{row['make']}_{row['model']}_{first}-{last}")  # "Toyota Corolla (2009-2010)"
    car, years = name.rsplit(" (", 1)
    return " ".join(filter(None, [car, row["generation"], PHASE_TEXT[row["phase"]]])) + f" ({years}"


def box_style(row, size):
    """CSS that puts the box over the photo, in % of its size (works at any display size)."""
    left, top, right, bottom = (float(v) for v in row["box"].split())
    width, height = size
    return (f"left:{left / width:.1%};top:{top / height:.1%};"
            f"width:{(right - left) / width:.1%};height:{(bottom - top) / height:.1%}")


def build_page(sample, wrong):
    tiles = "".join(f"""
      <div class="tile{' wrong' if r['title'] in wrong else ''}" data-t="{html.escape(r['title'])}">
        <div class="photo"><img src="/img/{i}" loading="lazy"><div class="box" style="{r['style']}"></div></div>
        <p>{html.escape(r['label'])}</p>
      </div>""" for i, r in enumerate(sample))
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Commons audit</title>
<style>
  body {{ font: 14px system-ui, sans-serif; margin: 16px; background: #f4f4f2; color: #222; }}
  header {{ position: sticky; top: 0; background: #f4f4f2; padding: 8px 0; z-index: 1; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }}
  .tile {{ background: white; border: 4px solid transparent; border-radius: 8px; cursor: pointer; }}
  .tile.wrong {{ border-color: #d33; background: #fde8e8; }}
  .photo {{ position: relative; }}
  .photo img {{ width: 100%; display: block; border-radius: 4px 4px 0 0; }}
  .box {{ position: absolute; border: 2px solid #2a78d6; pointer-events: none; }}
  .tile p {{ margin: 6px 8px 8px; }}
</style></head><body>
<header><b>Click every photo whose label is WRONG</b> (wrong car, generation, pre/facelift
if you can tell, or not a usable outside photo). Not sure? Leave it.
&nbsp; <span id="count"></span></header>
<div class="grid">{tiles}</div>
<script>
  const count = () => document.getElementById("count").textContent =
      document.querySelectorAll(".wrong").length + " marked wrong of {len(sample)}";
  document.querySelectorAll(".tile").forEach(tile => tile.addEventListener("click", () => {{
    tile.classList.toggle("wrong");
    fetch("/answer", {{method: "POST", body: JSON.stringify(
        {{title: tile.dataset.t, wrong: tile.classList.contains("wrong")}})}});
    count();
  }}));
  count();
</script></body></html>"""


def load_wrong():
    return set(json.loads(ANSWERS_FILE.read_text(encoding="utf-8"))) if ANSWERS_FILE.exists() else set()


def make_handler(sample, files):
    """Like review_boundaries.py: GET serves the page and photos (by number only, never by
    path, so the page can't read other files), POST saves one click."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.send_bytes(build_page(sample, load_wrong()).encode(), "text/html; charset=utf-8")
            elif self.path.startswith("/img/"):
                path = files[int(self.path.removeprefix("/img/"))]
                self.send_bytes(path.read_bytes(), "image/png" if path.suffix == ".png" else "image/jpeg")
            else:
                self.send_error(404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            wrong = load_wrong()
            if body["wrong"]:
                wrong.add(body["title"])
            else:
                wrong.discard(body["title"])
            ANSWERS_FILE.parent.mkdir(parents=True, exist_ok=True)
            ANSWERS_FILE.write_text(json.dumps(sorted(wrong), indent=1), encoding="utf-8")
            self.send_bytes(b"ok", "text/plain")

        def send_bytes(self, data, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass  # don't print a line for every photo

    return Handler


def report(sample):
    """Error rate per rule. 'Give or take' is a rough 95% margin for a sample of this size."""
    wrong = load_wrong()
    print("\nAudit results (wrong / checked):")
    for rule in RULES:
        rows = [r for r in sample if r["reason"] == rule]
        bad = sum(r["title"] in wrong for r in rows)
        rate = bad / len(rows)
        margin = 1.96 * (rate * (1 - rate) / len(rows)) ** 0.5  # binomial standard error x 1.96
        print(f"  {rule:9} {bad:3} / {len(rows)}  = {rate:.0%}  (give or take {margin:.0%})")


def main():
    commons_dir = project_path(load_config()["paths"]["commons_dir"])
    with open(commons_dir / "presort.csv", encoding="utf-8", newline="") as f:
        presort = list(csv.DictReader(f))
    with open(commons_dir / "filtered.csv", encoding="utf-8", newline="") as f:
        boxes = {r["title"]: r["box"] for r in csv.DictReader(f) if r["keep"] == "1"}
    _, years_of_look = load_looks()

    sample = pick_sample(presort)
    files = []
    for row in sample:
        path = commons_dir / "images" / row["file"]
        files.append(path)
        row["box"] = boxes[row["title"]]
        with Image.open(path) as image:  # opening reads only the header: fast
            row["style"] = box_style(row, image.size)
        row["label"] = label_text(row, years_of_look)

    server = ThreadingHTTPServer(("127.0.0.1", PORT), make_handler(sample, files))
    url = f"http://127.0.0.1:{PORT}/"
    print(f"Open {url}  ({len(sample)} photos; clicks saved to {ANSWERS_FILE})\n"
          f"Press Ctrl+C here when you've gone through all of them.")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        report(sample)


if __name__ == "__main__":
    main()
