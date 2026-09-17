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
from collections import Counter
from pathlib import Path

FIELDS = ["id", "input", "output", "generator", "style", "gutenberg_id", "title", "author",
          "category", "input_words", "output_words"]


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        r["input_words"] = len(r["input"].split())
        r["output_words"] = len(r["output"].split())
        rows.append({k: r[k] for k in FIELDS})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("dataset"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stats: dict = {"splits": {}}
    for split in ("train", "test"):
        rows = load(args.pairs_dir / f"pairs_{split}.jsonl")
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
            "books": len({r["gutenberg_id"] for r in rows}),
            "category": dict(Counter(r["category"] for r in rows)),
            "generator": dict(Counter(r["generator"] for r in rows)),
            "style": dict(Counter(r["style"] for r in rows)),
            "avg_input_words": round(sum(r["input_words"] for r in rows) / len(rows), 1),
            "avg_output_words": round(sum(r["output_words"] for r in rows) / len(rows), 1),
        }
    train = load(args.pairs_dir / "pairs_train.jsonl")
    books = Counter((r["gutenberg_id"], r["title"], r["author"], r["category"]) for r in train)
    stats["books"] = [{"gutenberg_id": g, "title": t, "author": a, "category": c, "train_rows": n}
                      for (g, t, a, c), n in sorted(books.items(), key=lambda x: x[0][1])]
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    preview = sorted(load(args.pairs_dir / "pairs_test.jsonl"), key=lambda r: r["id"])[:20]
    with (args.pairs_dir / "samples_preview.jsonl").open("w", encoding="utf-8") as f:
        for r in preview:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in stats["splits"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
