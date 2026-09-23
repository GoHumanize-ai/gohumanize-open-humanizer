"""Step 4: assemble the publishable dataset.

Reads the AI-fied pairs, drops rows whose generation failed, writes the
release files (JSONL + CSV per split), a preview file that is committed to
git, and a statistics JSON used by the dataset card and the paper.

Each row:
    id, input (AI-styled text), output (human original), generator, style,
    gutenberg_id, title, author, category, input_words, output_words

Run:
    python pipeline/04_build_dataset.py --pairs-dir data --out-dir dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = ["id", "input", "output", "generator", "style", "source", "gutenberg_id", "title",
          "author", "category", "url", "input_words", "output_words"]
DEFAULTS = {"source": "gutenberg", "gutenberg_id": 0, "url": ""}

# Anything in the human target that the AI-styled input does not carry is something the model
# learns to ADD. Measured on the first modern build: press-release datelines ("GAITHERSBURG,
# Md. - ") and footnote numbers glued to sentence ends ("...monetary policy.12") were in the
# targets and mostly absent from the inputs, and the retrained model duly invented both. They
# are stripped from both sides, and passages that are media notices rather than prose are dropped.
DATELINE = re.compile(r"^[A-Z][A-Z.\- ]{2,25}(,\s*(?:D\.C\.|[A-Z][a-z]{1,3}\.|[A-Z]{2}))?\s*[\u2014\u2013]\s*")
FOOTNOTE = re.compile(r"(?<=[a-z\)\"\u201d\u2019%][.,;:])[0-9]{1,2}(?=\s|$)")
JUNK = re.compile(r"(https?://|www\.|\bcookie\b|\bAPOD\b|media accreditation|\bRSVP\b|"
                  r"watch (online|live) at|newsroom at)", re.I)


def clean(text: str) -> str:
    return FOOTNOTE.sub("", DATELINE.sub("", text))


def artifacts(text: str) -> list[str]:
    """Names of the patterns a target must never contain after cleaning."""
    return [name for name, pat in (("dateline", DATELINE), ("footnote", FOOTNOTE), ("junk", JUNK))
            if pat.search(text)]


def hold_out_by_group(train: list[dict], test: list[dict], key: str, rows: int,
                      seed: int) -> tuple[list[dict], list[dict]]:
    """Move whole groups (articles) into the test split, so no test passage shares a source
    article with training. Rows without the key (the books) keep their split: for books the
    shared-author test is deliberate, it measures the rewrite rather than the author."""
    keyed = [r for r in train + test if r[key]]
    other_train = [r for r in train if not r[key]]
    other_test = [r for r in test if not r[key]]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in keyed:
        groups[r[key]].append(r)
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    held: list[dict] = []
    kept: list[dict] = []
    for name in names:
        (held if len(held) < rows else kept).extend(groups[name])
    return other_train + kept, other_test + held


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        r["input_words"] = len(r["input"].split())
        r["output_words"] = len(r["output"].split())
        rows.append({k: r.get(k, DEFAULTS.get(k, "")) for k in FIELDS})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("dataset"))
    ap.add_argument("--prefixes", default="pairs",
                    help="comma-separated file prefixes to concatenate, e.g. pairs,pairs_modern")
    ap.add_argument("--holdout-by", default="",
                    help="field whose groups go wholly to test, e.g. url; rows without it keep "
                         "their split")
    ap.add_argument("--holdout-rows", type=int, default=100,
                    help="roughly how many grouped rows to hold out")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip()]

    splits = {s: [r for prefix in prefixes for r in load(args.pairs_dir / f"{prefix}_{s}.jsonl")]
              for s in ("train", "test")}
    dropped = 0
    for s, rows in splits.items():
        kept_rows = []
        for r in rows:
            if JUNK.search(r["output"]) or JUNK.search(r["input"]):
                dropped += 1
                continue
            r["input"], r["output"] = clean(r["input"]), clean(r["output"])
            r["input_words"], r["output_words"] = len(r["input"].split()), len(r["output"].split())
            kept_rows.append(r)
        splits[s] = kept_rows
    if args.holdout_by:
        splits["train"], splits["test"] = hold_out_by_group(
            splits["train"], splits["test"], args.holdout_by, args.holdout_rows, args.seed)
    bad = [(r["id"], artifacts(r["output"])) for s in splits.values() for r in s if artifacts(r["output"])]
    if bad:
        raise SystemExit(f"{len(bad)} targets still carry artifacts the model would learn to "
                         f"invent, first: {bad[:3]}")
    print(f"dropped {dropped} media-notice rows; no target carries a dateline, footnote or link")

    stats: dict = {"splits": {}}
    for split in ("train", "test"):
        rows = splits[split]
        rows.sort(key=lambda r: r["id"])
        with (args.out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with (args.out_dir / f"{split}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        stats["splits"][split] = {
            "rows": len(rows),
            "books": len({r["gutenberg_id"] for r in rows if r["gutenberg_id"]}),
            "articles": len({r["url"] for r in rows if r["url"]}),
            "source": dict(Counter(r["source"] for r in rows)),
            "category": dict(Counter(r["category"] for r in rows)),
            "generator": dict(Counter(r["generator"] for r in rows)),
            "style": dict(Counter(r["style"] for r in rows)),
            "avg_input_words": round(sum(r["input_words"] for r in rows) / len(rows), 1),
            "avg_output_words": round(sum(r["output_words"] for r in rows) / len(rows), 1),
        }
    train = splits["train"]
    # Books are listed one by one; agency articles are far too many for that, so they are
    # summarised by agency instead.
    books = Counter((r["gutenberg_id"], r["title"], r["author"], r["category"])
                    for r in train if r["gutenberg_id"])
    stats["books"] = [{"gutenberg_id": g, "title": t, "author": a, "category": c, "train_rows": n}
                      for (g, t, a, c), n in sorted(books.items(), key=lambda x: x[0][1])]
    agencies = Counter(r["author"] for r in train if not r["gutenberg_id"])
    stats["agencies"] = [{"agency": a, "train_rows": n} for a, n in sorted(agencies.items())]
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    preview = sorted(splits["test"], key=lambda r: r["id"])[:20]
    with (args.pairs_dir / "samples_preview.jsonl").open("w", encoding="utf-8") as f:
        for r in preview:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in stats["splits"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
