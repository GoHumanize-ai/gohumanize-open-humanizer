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
    ap.add_argument("--group-key", default="gutenberg_id",
                    help="passages are drawn round-robin across this field (a book, an article)")
    ap.add_argument("--prefix", default="human", help="output file prefix, <prefix>_train.jsonl")
    ap.add_argument("--cap-per", default="", metavar="FIELD:N",
                    help="keep at most N passages per value of FIELD before selecting, so a "
                         "source with far more material than the others cannot dominate")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in args.raw.open(encoding="utf-8")]
    if args.cap_per:
        field, cap = args.cap_per.rsplit(":", 1)
        kept: list[dict] = []
        seen: dict[object, int] = defaultdict(int)
        rng.shuffle(rows)
        for r in rows:
            if seen[r[field]] < int(cap):
                seen[r[field]] += 1
                kept.append(r)
        print(f"cap {field}<={cap}: {len(rows)} -> {len(kept)} passages {dict(seen)}")
        rows = kept
    by_book: dict[object, list[dict]] = defaultdict(list)
    for r in rows:
        by_book[r[args.group_key]].append(r)
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
        with (args.out_dir / f"{args.prefix}_{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Books carry a title, agency articles only a URL; group by whatever the run used.
    per_group = defaultdict(int)
    for r in train:
        per_group[r.get("title") or r.get("url") or r[args.group_key]] += 1
    categories = defaultdict(int)
    for r in train:
        categories[r["category"]] += 1
    words = sum(r.get("word_count") or len(r["text"].split()) for r in train) / len(train)
    print(f"train={len(train)} test={len(test)} {dict(categories)} avg_words={words:.0f} "
          f"groups={len(per_group)} max_per_group={max(per_group.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
