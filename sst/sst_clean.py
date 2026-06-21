# coding=utf-8
"""
sst_clean.py — Self-contained rebuild of the Stanford Sentiment Treebank (SST).

Standalone reimplementation of HuggingFace's `sst.py` that needs NO third-party
libraries (no `datasets`, no `pyarrow`). Run it and it will:

    1. Download the raw archives from Stanford NLP (skipped if already cached).
    2. Extract them.
    3. Clean + align the raw files (encoding repair, bracket restore, phrase
       dictionary join, split bucketing).
    4. Write ready-to-use CSVs for all three configs.

Configs (matching the HF dataset card):
    default     sentence , label(0.0-1.0) , tokens , tree   -> train/dev/test
    dictionary  phrase   , label(0.0-1.0)                    -> dictionary
    ptb         ptb_tree (root label 0-4 baked in)           -> train/dev/test

Usage:
    py -3.12 sst_clean.py
    py -3.12 sst_clean.py --output ./sst/clean --cache ./sst/raw
    py -3.12 sst_clean.py --configs default          # only the default config

Only the Python standard library is required.
"""

import argparse
import csv
import os
import sys
import urllib.request
import zipfile

# ---------------------------------------------------------------------------
# Source archives (the URLs the original HF sst.py downloads from)
# ---------------------------------------------------------------------------
DEFAULT_URL = "https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip"
PTB_URL = "https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip"

DEFAULT_ZIP = "stanfordSentimentTreebank.zip"
PTB_ZIP = "trainDevTestTrees_PTB.zip"

# Stanford's split codes: 1 = train, 2 = test, 3 = dev
SPLIT_ID_TO_NAME = {"1": "train", "2": "test", "3": "dev"}

# csv field-size: parse trees / dictionary lines can be long
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


# ---------------------------------------------------------------------------
# Download + extract
# ---------------------------------------------------------------------------
def download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"[skip] already cached: {os.path.basename(dest)}")
        return
    print(f"[get ] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 sst_clean.py"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        out.write(resp.read())
    print(f"       -> {dest} ({os.path.getsize(dest):,} bytes)")


def extract(zip_path: str, dest_dir: str) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        # skip macOS resource-fork junk
        members = [m for m in zf.namelist() if not m.startswith("__MACOSX")]
        zf.extractall(dest_dir, members=members)


def ensure_raw(cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    default_zip = os.path.join(cache_dir, DEFAULT_ZIP)
    ptb_zip = os.path.join(cache_dir, PTB_ZIP)

    download(DEFAULT_URL, default_zip)
    download(PTB_URL, ptb_zip)

    if not os.path.isdir(os.path.join(cache_dir, "stanfordSentimentTreebank")):
        print("[unzip] stanfordSentimentTreebank.zip")
        extract(default_zip, cache_dir)
    if not os.path.isdir(os.path.join(cache_dir, "trees")):
        print("[unzip] trainDevTestTrees_PTB.zip")
        extract(ptb_zip, cache_dir)


# ---------------------------------------------------------------------------
# Cleaning helper
# ---------------------------------------------------------------------------
def clean_sentence(sentence: str) -> str:
    """Repair historic double-encoded UTF-8 bytes and restore PTB brackets,
    so the sentence text can be matched against the phrase dictionary."""
    sentence = (
        sentence.encode("utf-8")
        .replace(b"\xc3\x83\xc2", b"\xc3")
        .replace(b"\xc3\x82\xc2", b"\xc2")
        .decode("utf-8")
    )
    return sentence.replace("-LRB-", "(").replace("-RRB-", ")")


# ---------------------------------------------------------------------------
# Shared loaders
# ---------------------------------------------------------------------------
def load_phrase_scores(stb_dir: str):
    """Return (labels, phrases):
    labels   : phrase id   -> score(float)
    phrases  : phrase text -> score(float)   (the bridge used by `default`)
    """
    labels = {}
    with open(os.path.join(stb_dir, "sentiment_labels.txt"), encoding="utf-8") as g:
        reader = csv.DictReader(g, delimiter="|", quoting=csv.QUOTE_NONE)
        for row in reader:
            labels[row["phrase ids"]] = float(row["sentiment values"])

    phrases = {}
    with open(os.path.join(stb_dir, "dictionary.txt"), encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)
        for row in reader:
            phrases[row[0]] = labels[row[1]]
    return labels, phrases


def write_csv(out_path: str, fieldnames, rows) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"       {os.path.basename(out_path):16s} {len(rows):6d} rows")


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------
def build_default(cache_dir: str, out_dir: str) -> None:
    print("[build] default (sentence, label, tokens, tree)")
    stb = os.path.join(cache_dir, "stanfordSentimentTreebank")
    _, phrases = load_phrase_scores(stb)

    # per-sentence tokens & parse tree (line i -> sentence i)
    trees = {}
    with open(os.path.join(stb, "SOStr.txt"), encoding="utf-8") as tok, \
         open(os.path.join(stb, "STree.txt"), encoding="utf-8") as tr:
        for i, row in enumerate(csv.reader(tok, delimiter="\t", quoting=csv.QUOTE_NONE), start=1):
            trees[i] = {"tokens": row[0]}
        for i, row in enumerate(csv.reader(tr, delimiter="\t", quoting=csv.QUOTE_NONE), start=1):
            trees[i]["tree"] = row[0]

    with open(os.path.join(stb, "datasetSplit.txt"), encoding="utf-8") as spl:
        reader = csv.DictReader(spl, delimiter=",", quoting=csv.QUOTE_NONE)
        splits = {row["sentence_index"]: row["splitset_label"] for row in reader}

    buckets = {"train": [], "dev": [], "test": []}
    missed = 0
    with open(os.path.join(stb, "datasetSentences.txt"), encoding="utf-8") as snt:
        reader = csv.DictReader(snt, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            sentence = clean_sentence(row["sentence"])
            if sentence not in phrases:
                missed += 1
                continue
            idx = int(row["sentence_index"])
            split_name = SPLIT_ID_TO_NAME[splits[row["sentence_index"]]]
            buckets[split_name].append({
                "sentence": sentence,
                "label": phrases[sentence],
                "tokens": trees[idx]["tokens"],
                "tree": trees[idx]["tree"],
            })

    fields = ["sentence", "label", "tokens", "tree"]
    for split_name in ("train", "dev", "test"):
        write_csv(os.path.join(out_dir, f"default_{split_name}.csv"), fields, buckets[split_name])
    if missed:
        print(f"       [warn] {missed} sentence(s) unmatched")
    else:
        print("       [ok] every sentence matched")


def build_dictionary(cache_dir: str, out_dir: str) -> None:
    print("[build] dictionary (phrase, label)")
    stb = os.path.join(cache_dir, "stanfordSentimentTreebank")
    labels, _ = load_phrase_scores(stb)

    rows = []
    with open(os.path.join(stb, "dictionary.txt"), encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE):
            rows.append({"phrase": row[0], "label": labels[row[1]]})
    write_csv(os.path.join(out_dir, "dictionary.csv"), ["phrase", "label"], rows)


def build_ptb(cache_dir: str, out_dir: str) -> None:
    print("[build] ptb (ptb_tree, labels 0-4 baked in)")
    trees_dir = os.path.join(cache_dir, "trees")
    for split_name, fname in (("train", "train.txt"), ("dev", "dev.txt"), ("test", "test.txt")):
        rows = []
        with open(os.path.join(trees_dir, fname), encoding="utf-8") as fp:
            for row in csv.reader(fp, delimiter="\t", quoting=csv.QUOTE_NONE):
                rows.append({"ptb_tree": row[0]})
        write_csv(os.path.join(out_dir, f"ptb_{split_name}.csv"), ["ptb_tree"], rows)


BUILDERS = {
    "default": build_default,
    "dictionary": build_dictionary,
    "ptb": build_ptb,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Download + clean + rebuild SST (no `datasets` needed).")
    parser.add_argument("--cache", default=os.path.join(here, "raw"),
                        help="dir for raw downloads/extraction (default: <script_dir>/raw)")
    parser.add_argument("--output", default=os.path.join(here, "clean"),
                        help="dir for the cleaned CSVs (default: <script_dir>/clean)")
    parser.add_argument("--configs", nargs="+", default=list(BUILDERS),
                        choices=list(BUILDERS), help="which configs to build")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    ensure_raw(args.cache)
    for name in args.configs:
        BUILDERS[name](args.cache, args.output)
    print(f"[done] CSVs written to {args.output}")


if __name__ == "__main__":
    main()
