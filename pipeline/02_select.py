"""Step 2: select the training and test passages.

Takes the raw passages from step 1 and picks a fixed-size set, balanced
across books so no single author dominates, with a held-out test split
whose books also appear in training (we test the transformation, not
author generalisation). Deterministic given the seed.

Run:
    python pipeline/02_select.py --raw data/passages_raw.jsonl \
        --train 2000 --test 200 --out-dir data
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in args.raw.open(encoding="utf-8")]
    by_book: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_book[r["gutenberg_id"]].append(r)
    for v in by_book.values():
        rng.shuffle(v)

    want = args.train + args.test
    # round-robin across books so each contributes evenly until it runs out
    picked: list[dict] = []
    books = list(by_book)
    rng.shuffle(books)
    i = 0
    while len(picked) < want and any(by_book.values()):
        b = books[i % len(books)]
        if by_book[b]:
            picked.append(by_book[b].pop())
        i += 1
    if len(picked) < want:
        raise SystemExit(f"only {len(picked)} passages available, need {want}")

    rng.shuffle(picked)
    test, train = picked[: args.test], picked[args.test:]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("test", test)):
        with (args.out_dir / f"human_{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    per_book = defaultdict(int)
    for r in train:
        per_book[r["title"]] += 1
    fiction = sum(1 for r in train if r["category"] == "fiction")
    words = sum(r["word_count"] for r in train) / len(train)
    print(f"train={len(train)} test={len(test)} fiction={fiction} nonfiction={len(train)-fiction} "
          f"avg_words={words:.0f} books={len(per_book)} max_per_book={max(per_book.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
