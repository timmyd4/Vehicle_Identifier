"""
Helper for writing new generations.csv rows: what does Commons call each model's generations?

For every model in a list ("Tesla Model 3"), prints the model's Commons subcategories with
how many files and subcategories each one holds directly, e.g.

    === Tesla Model 3 ===
      Category:Tesla Model 3 Highland        files 212  subcats 3
      Category:Tesla Model 3 in Germany      files  40  subcats 0

The counts are NOT recursive (photos in deeper subcategories aren't counted), so they're only
a rough "is there anything here". commons_list.py does the real, complete listing later.
If the category doesn't exist, it prints the closest category names Commons' search finds.

Only reads category listings through the official API (same polite rules as wikimedia_check).

Usage (from the project root):
    python training/commons_discover.py data/coverage/wave1_models.txt > data/coverage/wave1_discover.txt
The list file has one Commons category name per line ("Tesla Model 3"); # starts a comment.
"""

import sys

from wikimedia_check import api_get, category_members


def category_info(titles):
    """{title: (files, subcats)} for up to 50 categories in one call."""
    data = api_get(action="query", prop="categoryinfo", titles="|".join(titles))
    info = {}
    for page in data["query"]["pages"].values():
        counts = page.get("categoryinfo", {})
        info[page["title"]] = (counts.get("files", 0), counts.get("subcats", 0))
    return info


def search_categories(text):
    """Category names Commons' own search finds for some text (namespace 14 = categories)."""
    data = api_get(action="query", list="search", srsearch=text, srnamespace=14, srlimit=6)
    return [hit["title"] for hit in data["query"]["search"]]


def discover(name):
    top = f"Category:{name}"
    print(f"\n=== {name} ===")
    subcats = category_members(top, "subcat")
    own = category_info([top])[top]
    if not subcats and own == (0, 0):
        print("  NOT FOUND. Closest: " + " | ".join(search_categories(name)))
        return
    print(f"  (itself)  files {own[0]:4}  subcats {own[1]:3}")
    for start in range(0, len(subcats), 50):
        info = category_info(subcats[start:start + 50])
        for title in subcats[start:start + 50]:
            files, subs = info.get(title, (0, 0))
            print(f"  {title.removeprefix('Category:'):60} files {files:4}  subcats {subs:3}")


def main():
    with open(sys.argv[1], encoding="utf-8") as f:
        names = [line.split("#")[0].strip() for line in f]
    for name in filter(None, names):
        discover(name)


if __name__ == "__main__":
    main()
