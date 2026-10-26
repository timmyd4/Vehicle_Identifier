"""
Check training/generations.csv for mistakes, and show only the rows worth a human look.

generations.csv is hand-written (mostly from memory), and a wrong year there means wrong
training labels. This script runs three checks and prints a report:

  1. The table itself: per model, do the rows cover the years without gaps or overlaps?
  2. VMMRdb: how many photos each row gets, and which VMMRdb years no row covers.
  3. Wikipedia: for each generation, what years Wikipedia gives and what it says about
     facelifts/updates. If Wikipedia doesn't back up a row's years, the row is flagged.

Rows that pass everything get a short "ok" line. Flagged rows show the evidence, so you can
decide: keep the table, use Wikipedia's year, or skip the uncertain year.

Wikipedia is only evidence, not the answer. It often lists the year a facelift was SHOWN
("2016 facelift") while our table uses US MODEL years (2017 models), so "one year earlier
than the table" counts as agreeing.

Usage (from the project root):
    python training/check_generations.py
    python training/check_generations.py highlander      # only models containing this text
"""

import os
import re
import sys
from collections import Counter

import pandas as pd

from common import car_model, load_config, load_merges, parse_class_name, project_path
from wikimedia_check import api_get

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# Current generations are written as ending in 2026 in the table; Wikipedia says "present".
CURRENT_YEAR = 2026

# Words Wikipedia uses for a mid-generation change of looks.
UPDATE_WORDS = r"facelift|refresh|update|restyl|redesign"

# "10th gen" in our table -> "tenth generation" in Wikipedia's headings.
ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth",
            7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth",
            13: "thirteenth", 14: "fourteenth", 15: "fifteenth"}

YEAR = re.compile(r"(?<!\d)(19[5-9]\d|20[0-3]\d)(?!\d)")


# ---------------------------------------------------------------------------------------
# Check 1: the table itself
# ---------------------------------------------------------------------------------------

def check_table(rows):
    """Return problems in one model's rows: gaps, overlaps, backwards years."""
    problems = []
    rows = rows.sort_values("first_year")
    previous = None
    for _, row in rows.iterrows():
        if row.first_year > row.last_year:
            problems.append(f"{row.generation} {row.phase}: first_year is after last_year")
        if previous is not None:
            if row.first_year > previous.last_year + 1:
                problems.append(f"gap: no row covers {previous.last_year + 1}-{row.first_year - 1}")
            elif row.first_year <= previous.last_year:
                problems.append(f"overlap: {previous.generation} {previous.phase} and "
                                f"{row.generation} {row.phase} both cover {row.first_year}")
        previous = row
    return problems


# ---------------------------------------------------------------------------------------
# Check 2: VMMRdb photo counts
# ---------------------------------------------------------------------------------------

def vmmrdb_year_counts(dataset_dir):
    """
    Return {(make, model): Counter(year -> number of photos)} for every VMMRdb folder.

    Trims and body styles are folded into the base model here (corolla_s -> corolla), the
    same merge the table assumes. A folder model counts for base model M when it is M, or
    starts with "M_". We can't know the base yet, so we store the raw model and merge later.
    """
    counts = {}
    for folder in os.listdir(dataset_dir):
        try:
            make, model, year = parse_class_name(folder)
        except ValueError:
            continue  # not a "make_model_year" folder
        n = len(os.listdir(dataset_dir / folder))
        counts.setdefault((make, model), Counter())[year] += n
    return counts


def photos_for_model(counts, make, model, merges):
    """Add up the year counts of every folder that becomes this model in our labels.

    corolla + corolla_s + corolla_le for "corolla"; 325i + 328i + ... for "3 series"
    (common.car_model decides, the same rule build_subset.py uses).
    """
    total = Counter()
    for (m, folder_model), years in counts.items():
        if m == make and car_model(make, folder_model, [model], merges) == model:
            total += years
    return total


# ---------------------------------------------------------------------------------------
# Check 3: Wikipedia
# ---------------------------------------------------------------------------------------

def wikitext(title):
    """Return the source text of a Wikipedia article (following redirects), or None."""
    data = api_get(WIKIPEDIA_API, action="parse", page=title, prop="wikitext", redirects=1)
    if "error" in data:
        return None
    return data["parse"]["wikitext"]["*"]


def wikipedia_title_for_commons(category):
    """
    Find the Wikipedia article about the same thing as a Commons category, or None.

    Commons categories are linked to a Wikidata item (a database entry shared by all
    Wikimedia sites), and the Wikidata item lists the matching article on each Wikipedia.
    So: Commons category -> Wikidata item -> English Wikipedia title.
    """
    data = api_get(action="query", titles=f"Category:{category}", prop="pageprops")
    page = next(iter(data["query"]["pages"].values()))
    item = page.get("pageprops", {}).get("wikibase_item")
    if not item:
        return None
    data = api_get(WIKIDATA_API, action="wbgetentities", ids=item, props="sitelinks",
                   sitefilter="enwiki")
    return data["entities"][item].get("sitelinks", {}).get("enwiki", {}).get("title")


def split_sections(text):
    """Split wikitext into [(heading, body)] at level-2 headings ("== Third generation ==")."""
    parts = re.split(r"^==([^=].*?)==\s*$", text, flags=re.MULTILINE)
    # re.split with a group gives: [intro, heading1, body1, heading2, body2, ...]
    return [("(intro)", parts[0])] + list(zip(parts[1::2], parts[2::2]))


def generation_text(rows, main_text):
    """
    Return (source description, wikitext) about one generation (all its rows).

    First choice: a separate article for the generation, found through Commons (works for
    e.g. "Honda Civic (tenth generation)"). We try every Commons category of every row,
    because small ones like "Ford Mustang (1979-1986)" often aren't linked to Wikipedia.
    Otherwise: the section of the model's main article whose heading names the generation
    ("Third generation (XU50; 2013)").
    """
    row = rows.iloc[0]
    main_title = f"{row.make.title()} {row.model.title()}"
    categories = dict.fromkeys(c for cats in rows.commons_category for c in cats.split(";"))
    for category in categories:  # dict.fromkeys = remove duplicates but keep the order
        title = wikipedia_title_for_commons(category)
        if title and title != main_title:
            text = wikitext(title)
            if text:
                return f"article '{title}'", text

    if main_text is None:
        return None, None
    wanted = [row.generation.lower()]
    number = re.match(r"(\d+)\w* gen", row.generation)
    if number:
        wanted.append(f"{ORDINALS[int(number.group(1))]} generation")
    for heading, body in split_sections(main_text):
        if any(w in heading.lower() for w in wanted):
            heading = re.sub(r"<[^>]+>", "", heading).strip()  # drop <span id=...> anchors
            return f"'{main_title}' section '{heading}'", body
    return None, None


def model_years(text):
    """
    Return (first, last) from the infobox "model_years = 2014-2019" line, or None.

    Infoboxes are the summary boxes at the top right of an article. The value can be messy
    ("2008-2013 (North America)", lists for several countries), so we just take the lowest
    and highest year in it. "present" means the generation is still on sale.
    """
    match = re.search(r"\|\s*model_years\s*=(.*?)(?=\n\s*\||\n\}\})", text, flags=re.DOTALL)
    if not match:
        return None
    value = match.group(1)
    years = [int(y) for y in YEAR.findall(value)]
    if "present" in value.lower():
        years.append(CURRENT_YEAR)
    return (min(years), max(years)) if years else None


def update_mentions(text):
    """Return short snippets that mention a facelift/update together with a year."""
    snippets = []
    # Subheadings like "=== 2016 facelift ===" are the clearest evidence.
    for heading in re.findall(r"^===+(.*?)===+\s*$", text, flags=re.MULTILINE):
        if re.search(UPDATE_WORDS, heading, re.I) and YEAR.search(heading):
            snippets.append(f"heading: {heading.strip()}")
    # Then sentences. Strip references and markup first so the snippets are readable.
    clean = re.sub(r"<ref[^>]*/>|<ref.*?</ref>", "", text, flags=re.DOTALL)
    clean = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", clean)  # [[link|text]] -> text
    clean = re.sub(r"\{\{.*?\}\}|'''?", "", clean, flags=re.DOTALL)
    for sentence in re.split(r"(?<=\.)\s+", clean):
        # Tables ({| ... |}) and photo galleries aren't prose; their years are noise.
        if "{|" in sentence or "<gallery" in sentence:
            continue
        if re.search(UPDATE_WORDS, sentence, re.I) and YEAR.search(sentence):
            snippets.append(sentence.strip().replace("\n", " ")[:160])
    return snippets


def missing_commons_categories(categories):
    """
    Return the Commons categories that don't exist (usually a typo in the table).

    The API takes up to 50 titles per call, so we ask in chunks. A page that doesn't exist
    comes back with a "missing" key.
    """
    categories = sorted(set(categories))
    missing = []
    for start in range(0, len(categories), 50):
        chunk = categories[start:start + 50]
        data = api_get(action="query", titles="|".join(f"Category:{c}" for c in chunk))
        missing += [p["title"].removeprefix("Category:")
                    for p in data["query"]["pages"].values() if "missing" in p]
    return missing


def mentioned_years(snippets):
    return {int(y) for s in snippets for y in YEAR.findall(s)}


# ---------------------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------------------

def check_generation(rows, main_text):
    """Return (flags, evidence lines) for one generation (its pre/facelift/single rows)."""
    flags, evidence = [], []
    source, text = generation_text(rows, main_text)
    # Rows already settled have "photo review" (a human compared photos in
    # review_boundaries.py) or "decided" (settled from Wikipedia's text) in their notes.
    # Their answer wins, so we only show evidence for them, no flags.
    settled = any(re.search(r"photo review|decided", str(n), re.I) for n in rows.notes)
    if text is None:
        return ([] if settled else ["no Wikipedia text found for this generation"]), []
    evidence.append(f"source: {source}")

    gen_first, gen_last = rows.first_year.min(), rows.last_year.max()
    years = model_years(text)
    if years:
        evidence.append(f"Wikipedia model years: {years[0]}-{years[1]}")
        # Wikipedia's box often covers several countries, so it may be WIDER than the US
        # years. Only flag when our range sticks out beyond Wikipedia's.
        if not settled and (gen_first < years[0] or gen_last > years[1]):
            flags.append(f"table says {gen_first}-{gen_last}, outside Wikipedia's {years[0]}-{years[1]}")
    else:
        evidence.append("Wikipedia model years: not found")

    snippets = update_mentions(text)
    found = mentioned_years(snippets)
    for _, row in rows[rows.phase.str.startswith("facelift")].iterrows():
        # Facelifts are usually shown the year before the model year they're sold as.
        if not settled and not found & {row.first_year - 1, row.first_year}:
            flags.append(f"facelift {row.first_year}: Wikipedia mentions no update in "
                         f"{row.first_year - 1} or {row.first_year}")
    # For one-look rows, only words that mean "the looks changed" count. Plain "update" is
    # too common ("updated engine", "updated safety ratings").
    if not settled and (rows.phase == "single").all() and any(
            re.search(r"facelift|refresh|restyl", s, re.I) for s in snippets):
        flags.append("table says one look, but Wikipedia mentions a facelift (was it visible?)")
    if not settled and any("CHECK" in str(n) for n in rows.notes):
        flags.append("marked CHECK in the table")
    evidence += snippets[:6]
    return flags, evidence


def main():
    # keep_default_na=False: pandas would otherwise turn the text "NA" into a missing value,
    # and "NA" is a real generation name (the first Mazda Miata).
    table = pd.read_csv(project_path("training/generations.csv"), comment="#",
                        skipinitialspace=True, keep_default_na=False)
    if len(sys.argv) > 1:
        table = table[table.model.str.contains(sys.argv[1])]

    dataset_dir = project_path(load_config()["paths"]["dataset_dir"])
    max_year = load_config()["subset"]["max_year"]
    print("Checking that every Commons category exists...")
    for category in missing_commons_categories(
            c for cats in table.commons_category for c in cats.split(";")):
        print(f"  ⚠ COMMONS: no category named '{category}'")

    print("Counting VMMRdb photos...")
    counts = vmmrdb_year_counts(dataset_dir)
    merges = load_merges()

    flagged = 0
    for (make, model), rows in table.groupby(["make", "model"], sort=False):
        print(f"\n{'=' * 70}\n{make.title()} {model.title()}\n{'=' * 70}")

        for problem in check_table(rows):
            print(f"  ⚠ TABLE: {problem}")

        photos = photos_for_model(counts, make, model, merges)
        covered = {y for _, r in rows.iterrows() for y in range(r.first_year, r.last_year + 1)}
        uncovered = {y: n for y, n in photos.items() if y not in covered and y <= max_year}
        if uncovered:
            print(f"  VMMRdb years no row covers (photos): {dict(sorted(uncovered.items()))}")

        main_text = wikitext(f"{make.title()} {model.title()}")
        for generation, gen_rows in rows.groupby("generation", sort=False):
            flags, evidence = check_generation(gen_rows, main_text)
            for _, r in gen_rows.iterrows():
                n = sum(photos[y] for y in range(r.first_year, r.last_year + 1))
                print(f"  {r.generation:9} {r.phase:9} {r.first_year}-{r.last_year}  "
                      f"VMMRdb photos: {n}")
            if flags:
                flagged += 1
                for flag in flags:
                    print(f"    ⚠ {flag}")
                for line in evidence:
                    print(f"      | {line}")
            else:
                print("    ok")

    print(f"\n{flagged} generation(s) flagged for a look.")


if __name__ == "__main__":
    main()
