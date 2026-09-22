"""Repository checks that need no GPU, no API keys and no network.

Run locally or in CI:
    python3 scripts/check_repo.py

Checks:
  1. every Python file compiles
  2. the Python package version is the same in pyproject.toml and __init__.py
  3. the released dataset matches dataset/stats.json (row counts, fields, splits)
  4. the JSON records in train/runs and eval/results parse and have the expected keys
  5. published links and names stay consistent (no dead project URL, no API host named)
"""

from __future__ import annotations

import compileall
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def python_compiles() -> None:
    ok = compileall.compile_dir(str(ROOT), quiet=2, force=True,
                                rx=re.compile(r"(\.git|node_modules|__pycache__|\.venv)"))
    check(bool(ok), "some Python files do not compile")


def package_version() -> None:
    pyproject = (ROOT / "packages/python/pyproject.toml").read_text()
    init = (ROOT / "packages/python/gohumanize_open_humanizer/__init__.py").read_text()
    in_toml = re.search(r'^version = "([^"]+)"', pyproject, re.M)
    in_init = re.search(r'__version__ = "([^"]+)"', init)
    check(bool(in_toml and in_init), "version missing from pyproject.toml or __init__.py")
    if in_toml and in_init:
        check(in_toml.group(1) == in_init.group(1),
              f"version mismatch: pyproject {in_toml.group(1)} vs __init__ {in_init.group(1)}")


def dataset_matches_stats() -> None:
    stats = json.loads((ROOT / "dataset/stats.json").read_text())
    fields = {"id", "input", "output", "generator", "style", "gutenberg_id", "title",
              "author", "category", "input_words", "output_words"}
    for split, expected in stats["splits"].items():
        rows = [json.loads(line) for line in
                (ROOT / f"dataset/{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        check(len(rows) == expected["rows"],
              f"dataset/{split}.jsonl has {len(rows)} rows, stats.json says {expected['rows']}")
        check(all(fields <= set(r) for r in rows), f"dataset/{split}.jsonl: rows with missing fields")
        check(len({r["id"] for r in rows}) == len(rows), f"dataset/{split}.jsonl: duplicate ids")
        check(all(r["input"].strip() and r["output"].strip() for r in rows),
              f"dataset/{split}.jsonl: empty input or output")
        for key in ("generator", "style", "category"):
            counts: dict[str, int] = {}
            for r in rows:
                counts[r[key]] = counts.get(r[key], 0) + 1
            check(counts == expected[key], f"dataset/{split}.jsonl: {key} counts differ from stats.json")
        csv_lines = (ROOT / f"dataset/{split}.csv").read_text(encoding="utf-8").splitlines()
        check(len(csv_lines) > expected["rows"], f"dataset/{split}.csv looks shorter than the JSONL")


def run_records() -> None:
    for path in sorted((ROOT / "train/runs").glob("*.json")):
        record = json.loads(path.read_text())
        missing = {"base_model", "epochs", "learning_rate", "train_loss", "eval_loss",
                   "train_runtime_s", "gpu", "wandb_url"} - set(record)
        check(not missing, f"{path.name}: missing keys {sorted(missing)}")
    for path in sorted((ROOT / "eval/results").glob("*.json")):
        if path.name.endswith("-samples.json"):
            continue
        result = json.loads(path.read_text())
        check("systems" in result and {"base", "finetuned"} <= set(result["systems"]),
              f"{path.name}: expected systems base and finetuned")


def links_and_names() -> None:
    docs = [p for p in ROOT.rglob("*.md") if ".git" not in p.parts and "node_modules" not in p.parts]
    code = [ROOT / "pipeline/03_aify.py", ROOT / ".env.example", ROOT / "scripts/zenodo_deposit.py"]
    for path in docs + code:
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = path.relative_to(ROOT)
        # The project page moved in September 2026; the old path only survives as a redirect.
        check("gohumanize.ai/research" not in text, f"{rel}: links to the old project page URL")
        # The AI-fication generator is named by model, never by the API host it was called through.
        check("openrouter" not in text.lower(), f"{rel}: names an API host that should not be mentioned")


def main() -> int:
    python_compiles()
    package_version()
    dataset_matches_stats()
    run_records()
    links_and_names()
    if FAILURES:
        print("FAILED")
        for failure in FAILURES:
            print(" -", failure)
        return 1
    print("all repository checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
