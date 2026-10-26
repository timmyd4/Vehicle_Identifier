"""
Show photos from both sides of an uncertain facelift, so a human can decide if it's visible.

check_generations.py flags rows where Wikipedia says "the car got a light facelift" but not
whether you'd notice it in a photo. This script starts a small web page on your own computer:
for each question it shows photos from the years BEFORE the possible change next to photos
from the years AFTER it, with three buttons:

    Split     - the two sides look different, keep them as separate labels
    One look  - they look the same, merge them into one label
    Skip      - can't tell; leave the boundary out of training

Each batch of models has its own questions file, training/runs/review/<batch>_questions.json,
a list of questions like:

    {"id": "accord_6th_2001", "make": "honda", "model": "accord",
     "before": [1998, 2000], "after": [2001, 2002], "source": "vmmrdb",
     "hint": "What Wikipedia says about the change"}

"source" is where the photos come from: "vmmrdb" (on disk), or "commons" plus a
"commons_category" for years VMMRdb doesn't have (your browser loads those from Commons).

Answers are saved to training/runs/review/<batch>_answers.json as you click, so you can close
the page any time. The page only listens on 127.0.0.1, so nothing outside your computer can
reach it.

Usage (from the project root):
    python training/review_boundaries.py batch2
Then press Ctrl+C in the terminal when you're done.
"""

import html
import json
import os
import random
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from common import load_config, parse_class_name, project_path
from wikimedia_check import api_get, files_in_tree, year_from_filename

PORT = 8765
PHOTOS_PER_SIDE = 8
REVIEW_DIR = project_path("training/runs/review")


# ---------------------------------------------------------------------------------------
# Picking photos
# ---------------------------------------------------------------------------------------

def vmmrdb_photos(dataset_dir, make, model, years, rng):
    """
    Pick random photo paths from the base model's VMMRdb folders for a range of years.

    Only the base model (corolla, not corolla_s), so trims don't muddy the comparison.
    `rng` is a seeded random generator, so the same photos come up every time you run it.
    """
    paths = []
    for folder in os.listdir(dataset_dir):
        try:
            f_make, f_model, year = parse_class_name(folder)
        except ValueError:
            continue
        if f_make == make and f_model == model and years[0] <= year <= years[1]:
            paths += [dataset_dir / folder / name for name in os.listdir(dataset_dir / folder)]
    return rng.sample(sorted(paths), min(PHOTOS_PER_SIDE, len(paths)))


def commons_photos(category, years, rng):
    """
    Pick Commons files whose filename has a year in range, and return their thumbnail URLs.

    The filename year is only a hint (it can be the date the photo was taken), which is why
    the page tells you so. Thumbnails are 300 px wide images Commons makes on request.
    """
    files = [f for f in files_in_tree(category, depth=1)
             if (y := year_from_filename(f)) and years[0] <= y <= years[1]]
    chosen = rng.sample(sorted(files), min(PHOTOS_PER_SIDE, len(files)))
    urls = []
    for title in chosen:
        data = api_get(action="query", titles=title, prop="imageinfo", iiprop="url",
                       iiurlwidth=300)
        page = next(iter(data["query"]["pages"].values()))
        urls.append(page["imageinfo"][0]["thumburl"])
    return urls


# ---------------------------------------------------------------------------------------
# The web page
# ---------------------------------------------------------------------------------------

def build_page(questions, answers):
    """Return the whole page as one HTML string."""
    sections = []
    for q in questions:
        sides = []
        for side in ("before", "after"):
            first, last = q[side]
            imgs = "".join(f'<img src="{html.escape(src)}" loading="lazy">' for src in q[side + "_srcs"])
            sides.append(f'<div class="side"><h3>{first}-{last}</h3>{imgs or "<p>no photos</p>"}</div>')
        chosen = answers.get(q["id"], "")
        buttons = "".join(
            f'<button data-q="{q["id"]}" data-a="{a}" class="{"chosen" if a == chosen else ""}">{label}</button>'
            for a, label in [("split", "Split (they look different)"),
                             ("one", "One look (same)"), ("skip", "Skip (can't tell)")])
        sections.append(f"""
        <section>
          <h2>{q["make"].title()} {q["model"].title()}: change at {q["after"][0]}?</h2>
          <p class="hint">{html.escape(q["hint"])}</p>
          <div class="sides">{"".join(sides)}</div>
          <div class="buttons">{buttons}</div>
        </section>""")

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Facelift review</title>
<style>
  body {{ font-family: sans-serif; margin: 16px; background: #f4f4f4; color: #222; }}
  section {{ background: white; padding: 12px 16px; margin-bottom: 20px; border-radius: 8px; }}
  .hint {{ color: #555; }}
  .sides {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .side {{ flex: 1; min-width: 300px; }}
  .side img {{ width: 48%; margin: 1%; border-radius: 4px; }}
  button {{ font-size: 16px; padding: 8px 14px; margin-right: 8px; cursor: pointer; }}
  button.chosen {{ background: #2a7; color: white; }}
</style></head><body>
<h1>Facelift review</h1>
<p>Compare the left and right photos. Look at the front: headlights, grille, bumper.
Answers save as you click.</p>
{"".join(sections)}
<script>
  // Each click sends {{question, answer}} to the Python server, which writes it to disk.
  document.querySelectorAll("button").forEach(b => b.onclick = async () => {{
    await fetch("/answer", {{method: "POST", body: JSON.stringify({{q: b.dataset.q, a: b.dataset.a}})}});
    b.parentElement.querySelectorAll("button").forEach(x => x.classList.toggle("chosen", x === b));
  }});
</script>
</body></html>"""


def load_answers(answers_file):
    if answers_file.exists():
        return json.loads(answers_file.read_text(encoding="utf-8"))
    return {}


def make_handler(questions, local_files, answers_file):
    """
    Build the request handler class for http.server.

    Like a tiny servlet: do_GET answers page and image requests, do_POST saves an answer.
    Local photos are served by NUMBER (/img/3), never by path, so the page can only ever
    load the photos we picked, not any other file on your disk.
    """
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                page = build_page(questions, load_answers(answers_file))
                self.send_bytes(page.encode(), "text/html; charset=utf-8")
            elif self.path.startswith("/img/"):
                index = int(self.path.removeprefix("/img/"))
                self.send_bytes(local_files[index].read_bytes(), "image/jpeg")
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/answer":
                return self.send_error(404)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            answers = load_answers(answers_file)
            answers[body["q"]] = body["a"]
            answers_file.write_text(json.dumps(answers, indent=2), encoding="utf-8")
            print(f"  saved: {body['q']} = {body['a']}")
            self.send_bytes(b"ok", "text/plain")

        def send_bytes(self, data, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass  # don't print a line for every image request

    return Handler


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python training/review_boundaries.py <batch>   (e.g. batch2)")
    batch = sys.argv[1]
    questions = json.loads((REVIEW_DIR / f"{batch}_questions.json").read_text(encoding="utf-8"))
    answers_file = REVIEW_DIR / f"{batch}_answers.json"

    dataset_dir = project_path(load_config()["paths"]["dataset_dir"])
    rng = random.Random(42)
    local_files = []  # photos on disk, served as /img/<index>

    print(f"Picking photos for {len(questions)} questions...")
    for q in questions:
        for side in ("before", "after"):
            if q["source"] == "vmmrdb":
                srcs = []
                for path in vmmrdb_photos(dataset_dir, q["make"], q["model"], q[side], rng):
                    srcs.append(f"/img/{len(local_files)}")
                    local_files.append(path)
            else:
                srcs = commons_photos(q["commons_category"], q[side], rng)
            q[side + "_srcs"] = srcs

    handler = make_handler(questions, local_files, answers_file)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"Open {url}  (answers go to {answers_file})\nPress Ctrl+C to stop.")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
