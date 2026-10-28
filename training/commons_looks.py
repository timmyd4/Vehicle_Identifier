"""
Step 6 of the Commons pipeline: a human names the look of photos no clue could decide,
for the MISSING looks only (the "facelift gap").

THE BIG PICTURE
---------------
A look becomes a class (an answer the model can give) only with 50+ photos. Some looks
never get there by themselves: Commons files the Highlander XU50 pre-facelift AND facelift
under one category, and when the filename has no year, nothing can prove which one a photo
is. The model can't help either: it has no class for the facelift, so it can't vote for it.
Those photos wait in presort.csv as "review: look unknown to model".

Here you click the right look under each photo. Your answers count as labels, and once a
look reaches 50 photos, the next build_subset.py makes it a class.

TO KEEP IT SHORT, the page only asks about:
  - looks in config `looks.always` (your Highlander XU50 facelift), plus
  - looks that need at most `looks.max_need` more photos, AND have enough waiting photos
    to get there (a look with 3 photos and 10 waiting never will, so it's not worth a click).
And it stops showing a car's photos as soon as all of ITS asked-about looks reach 50: if 12
photos bring the WRX GD to 50, you never see the other 195.

Buttons under each photo: one per possible look, and "Skip" (can't tell, not this car, not a
usable outside photo). Every click is saved right away (data/commons/look_answers.json), so
you can close the page and come back. Reload the page for the next batch.

Usage (from the project root, after build_subset.py wrote data/splits/look_counts.csv):
    python training/commons_looks.py
"""

import csv
import html
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

from common import load_config, project_path
from commons_audit import box_style, label_text
from commons_presort import load_looks, look_text

PORT = 8767
PER_PAGE = 40
NEEDED = 50          # same as subset.min_images_per_class: photos a look needs to become a class
REASONS = {"look unknown to model", "which look?"}


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_answers(path):
    """{title: [make, model, generation, phase]}, or {title: None} for a skipped photo."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def candidate_keys(row, years_of_look):
    """presort.csv stores candidates as "highlander/XU50/pre;highlander/XU50/facelift".
    Turn them back into full look keys (the make is the same for all of a photo's candidates)."""
    by_text = {look_text(k): k for k in years_of_look if k[0] == row["make"]}
    return [by_text[t] for t in row["candidates"].split(";") if t in by_text]


class Queue:
    """Which looks to ask about, which photos show them, and how far each look is."""

    def __init__(self, presort, counts, years_of_look, settings, answers):
        self.answers = answers
        self.have = {(c["make"], c["model"], c["generation"], c["phase"]): int(c["num_images"])
                     for c in counts}

        # Every review photo, with its possible looks. pending = how many photos MIGHT be each look.
        self.photos, pending = [], {}
        for row in presort:
            if row["status"] == "review" and row["reason"] in REASONS:
                row["keys"] = candidate_keys(row, years_of_look)
                self.photos.append(row)
                for key in row["keys"]:
                    pending[key] = pending.get(key, 0) + 1

        # The looks worth asking about (see the top of the file).
        always = {tuple(text.split("/")) for text in settings["always"]}
        self.asked = {key for key, n in pending.items()
                      if self.have.get(key, 0) < NEEDED
                      and (key in always or
                           NEEDED - self.have.get(key, 0) <= min(settings["max_need"], n))}
        # Photos of the always-looks first, then the looks that need the fewest.
        need = lambda key: -1 if key in always else NEEDED - self.have.get(key, 0)
        self.photos = [p for p in self.photos if self.asked & set(p["keys"])]
        self.photos.sort(key=lambda p: (min(need(k) for k in self.asked & set(p["keys"])), p["candidates"]))

    def count(self, key):
        """Photos the look has now: from the last build, plus your clicks since."""
        return self.have.get(key, 0) + sum(a is not None and tuple(a) == key for a in self.answers.values())

    def still_needed(self, photo):
        """Does this photo still help? Only if one of its asked-about looks is below 50."""
        return any(self.count(k) < NEEDED for k in self.asked & set(photo["keys"]))

    def next_page(self):
        return [p for p in self.photos if p["title"] not in self.answers and self.still_needed(p)][:PER_PAGE]

    def progress(self):
        done = sum(self.count(k) >= NEEDED for k in self.asked)
        return f"{done} of {len(self.asked)} looks reached {NEEDED} photos | {len(self.answers)} photos answered"


def build_page(queue, page, years_of_look):
    def button(key, title):
        row = dict(zip(["make", "model", "generation", "phase"], key))
        return (f'<button data-t="{html.escape(title)}" data-k="{html.escape(json.dumps(key))}">'
                f'{html.escape(label_text(row, years_of_look))} &middot; {queue.count(key)}</button>')

    tiles = "".join(f"""
      <div class="tile">
        <div class="photo"><img src="/img/{i}" loading="lazy"><div class="box" style="{p['style']}"></div></div>
        {"".join(button(k, p["title"]) for k in p["keys"])}
        <button class="skip" data-t="{html.escape(p['title'])}" data-k="null">Skip</button>
      </div>""" for i, p in enumerate(page))
    body = tiles or "<p><b>All done!</b> Every look on the list has 50 photos, or no photos are left to ask about.</p>"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Missing looks</title>
<style>
  body {{ font: 14px system-ui, sans-serif; margin: 16px; background: #f4f4f2; color: #222; }}
  header {{ position: sticky; top: 0; background: #f4f4f2; padding: 8px 0; z-index: 1; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px; }}
  .tile {{ background: white; border-radius: 8px; padding-bottom: 6px; }}
  .tile.answered {{ opacity: 0.3; }}
  .photo {{ position: relative; }}
  .photo img {{ width: 100%; display: block; border-radius: 8px 8px 0 0; }}
  .box {{ position: absolute; border: 2px solid #2a78d6; pointer-events: none; }}
  button {{ display: block; width: calc(100% - 12px); margin: 6px; padding: 8px; text-align: left;
           font: inherit; border: 1px solid #bbb; border-radius: 6px; background: #fafafa; cursor: pointer; }}
  button.picked {{ background: #2a78d6; color: white; border-color: #2a78d6; }}
  button.skip {{ color: #777; }}
</style></head><body>
<header><b>Click the right look for each photo</b> (the box shows which car). Can't tell,
wrong car, or not a usable outside photo? Skip. Then reload for the next batch.
The number on a button = photos that look has so far.<br>{queue.progress()}</header>
<div class="grid">{body}</div>
<script>
  document.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {{
    const tile = b.closest(".tile");
    tile.querySelectorAll("button").forEach(x => x.classList.remove("picked"));
    b.classList.add("picked");
    tile.classList.add("answered");
    fetch("/answer", {{method: "POST", body: JSON.stringify({{title: b.dataset.t, look: JSON.parse(b.dataset.k)}})}});
  }}));
</script></body></html>"""


def make_handler(queue, commons_dir, answers_file, years_of_look):
    """Same idea as commons_audit.py: photos are served by number only, never by path.
    Each page load picks the next batch and remembers it, so /img/<n> means that batch's photo n."""
    current = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                current[:] = queue.next_page()
                for p in current:
                    with Image.open(commons_dir / "images" / p["file"]) as image:  # header only: fast
                        p["style"] = box_style(p, image.size)
                self.send_bytes(build_page(queue, current, years_of_look).encode(), "text/html; charset=utf-8")
            elif self.path.startswith("/img/"):
                path = commons_dir / "images" / current[int(self.path.removeprefix("/img/"))]["file"]
                self.send_bytes(path.read_bytes(), "image/png" if path.suffix == ".png" else "image/jpeg")
            else:
                self.send_error(404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            queue.answers[body["title"]] = body["look"]   # clicking again just replaces the answer
            answers_file.write_text(json.dumps(queue.answers, indent=1), encoding="utf-8")
            self.send_bytes(b"ok", "text/plain")

        def send_bytes(self, data, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    return Handler


def main():
    config = load_config()
    commons_dir = project_path(config["paths"]["commons_dir"])
    answers_file = commons_dir / "look_answers.json"
    _, years_of_look = load_looks()

    presort = read_csv(commons_dir / "presort.csv")
    boxes = {r["title"]: r["box"] for r in read_csv(commons_dir / "filtered.csv") if r["keep"] == "1"}
    for row in presort:
        row["box"] = boxes.get(row["title"], "0 0 1 1")
    counts = read_csv(project_path(config["paths"]["splits_dir"]) / "look_counts.csv")

    queue = Queue(presort, counts, years_of_look, config["commons"]["looks"], load_answers(answers_file))
    print(f"Asking about {len(queue.asked)} missing looks, {len(queue.photos):,} photos could show them "
          f"(you'll see far fewer: a car stops once its looks reach {NEEDED}).")
    print(queue.progress())

    server = ThreadingHTTPServer(("127.0.0.1", PORT), make_handler(queue, commons_dir, answers_file, years_of_look))
    url = f"http://127.0.0.1:{PORT}/"
    print(f"Open {url}  (answers saved to {answers_file}). Press Ctrl+C here when you're done.")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n" + queue.progress())


if __name__ == "__main__":
    main()
