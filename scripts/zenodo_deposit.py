"""Create (or update) the Zenodo deposit for the GoHumanize Open Humanizer release.

Uploads a source snapshot of this repository (code, docs, dataset, evaluation
results; no weights, they live on Hugging Face) plus the paper and the dataset
files as separate downloads, with citation metadata. The deposit is left as a
DRAFT; publishing (which mints the DOI and is irreversible) is a separate,
explicit step:

    python scripts/zenodo_deposit.py --create            # draft + files + metadata
    python scripts/zenodo_deposit.py --refresh <id>      # replace files + metadata on a draft
    python scripts/zenodo_deposit.py --publish <id>      # mint the DOI

Token: ~/.zenodo_token or ZENODO_TOKEN.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

API = "https://zenodo.org/api"
ROOT = Path(__file__).resolve().parent.parent
VERSION = "0.1.0"  # matches the PyPI and npm packages

METADATA = {
    "metadata": {
        "title": "GoHumanize Open Humanizer: an open text-humanization model, dataset and pipeline built from public-domain data",
        "upload_type": "software",
        "description": (
            "<p>The GoHumanize Open Humanizer is a public research and educational model: a Qwen3-4B "
            "fine-tune (QLoRA) that rewrites AI-styled English prose into more natural human writing. "
            "This record archives the complete pipeline (Project Gutenberg sourcing and cleaning, "
            "AI-fication of inputs with three generators, training, evaluation, serving), the dataset "
            "of 2,000 training and 200 test pairs (CC-BY 4.0), the evaluation results and the paper-style "
            "write-up explaining every step and service used.</p>"
            "<p>Model weights and GGUF builds: https://huggingface.co/gohumanize/gohumanize-open-humanizer. "
            "Dataset: https://huggingface.co/datasets/gohumanize/gohumanize-open-humanizer-dataset. "
            "Code: https://github.com/GoHumanize-ai/gohumanize-open-humanizer. "
            "Project page: https://gohumanize.ai/research.</p>"
            "<p>The Open Humanizer is separate from the production models used by GoHumanize.ai and makes "
            "no claim about AI detectors.</p>"
        ),
        "creators": [{"name": "GoHumanize team", "affiliation": "GoHumanize"}],
        "version": VERSION,
        "license": "apache-2.0",
        "keywords": ["humanizer", "text rewriting", "style transfer", "Qwen3", "LoRA", "public domain",
                     "Project Gutenberg", "dataset", "open model"],
        "related_identifiers": [
            {"identifier": "https://github.com/GoHumanize-ai/gohumanize-open-humanizer", "relation": "isSupplementTo", "scheme": "url"},
            {"identifier": "https://huggingface.co/gohumanize/gohumanize-open-humanizer", "relation": "isSupplementedBy", "scheme": "url"},
            {"identifier": "https://huggingface.co/datasets/gohumanize/gohumanize-open-humanizer-dataset", "relation": "isSupplementedBy", "scheme": "url"},
            {"identifier": "https://gohumanize.ai/research", "relation": "isDescribedBy", "scheme": "url"},
        ],
    }
}


def token() -> str:
    t = os.getenv("ZENODO_TOKEN") or Path.home().joinpath(".zenodo_token").read_text().strip()
    if not t:
        sys.exit("no Zenodo token")
    return t


def snapshot_zip(dest: Path) -> Path:
    """git archive of HEAD (tracked files only, so .env and caches never leak)."""
    out = dest / f"gohumanize-open-humanizer-{VERSION}-source.zip"
    subprocess.run(["git", "archive", "--format=zip", f"--prefix=gohumanize-open-humanizer-{VERSION}/",
                    "-o", str(out), "HEAD"], cwd=ROOT, check=True)
    return out


def upload_all(tok: str, dep: dict) -> None:
    """Upload the snapshot, paper and dataset files into the deposit's bucket.

    Zenodo overwrites a bucket object with the same name, so re-running this on a
    draft replaces the files in place.
    """
    h = {"Authorization": f"Bearer {tok}"}
    bucket = dep["links"]["bucket"]
    with tempfile.TemporaryDirectory() as tmp:
        files = [snapshot_zip(Path(tmp)), ROOT / "docs" / "paper.md", ROOT / "dataset" / "train.jsonl",
                 ROOT / "dataset" / "test.jsonl", ROOT / "dataset" / "README.md"]
        for f in files:
            name = f.name if f.parent.name != "dataset" else f"dataset-{f.name}"
            with f.open("rb") as fh:
                up = requests.put(f"{bucket}/{name}", data=fh, headers=h, timeout=600)
                up.raise_for_status()
            print("uploaded", name)


def set_metadata(tok: str, dep: dict) -> dict:
    h = {"Authorization": f"Bearer {tok}"}
    r = requests.put(f"{API}/deposit/depositions/{dep['id']}", json=METADATA, headers=h, timeout=60)
    r.raise_for_status()
    out = r.json()
    print(json.dumps({"deposition_id": dep["id"], "draft_url": dep["links"]["html"],
                      "version": out.get("metadata", {}).get("version"),
                      "prereserved_doi": out.get("metadata", {}).get("prereserve_doi", {}).get("doi")}, indent=2))
    return out


def create(tok: str) -> None:
    h = {"Authorization": f"Bearer {tok}"}
    r = requests.post(f"{API}/deposit/depositions", json={}, headers=h, timeout=60)
    r.raise_for_status()
    dep = r.json()
    upload_all(tok, dep)
    set_metadata(tok, dep)


def refresh(tok: str, dep_id: int) -> None:
    """Replace the files and metadata on an existing, still unpublished draft."""
    h = {"Authorization": f"Bearer {tok}"}
    r = requests.get(f"{API}/deposit/depositions/{dep_id}", headers=h, timeout=60)
    r.raise_for_status()
    dep = r.json()
    if dep.get("submitted"):
        sys.exit(f"deposit {dep_id} is already published; make a new version instead")
    # Files from an earlier run under a different name would otherwise linger.
    for f in dep.get("files", []):
        requests.delete(f["links"]["self"], headers=h, timeout=60).raise_for_status()
        print("removed", f["filename"])
    upload_all(tok, dep)
    set_metadata(tok, dep)


def publish(tok: str, dep_id: int) -> None:
    h = {"Authorization": f"Bearer {tok}"}
    r = requests.post(f"{API}/deposit/depositions/{dep_id}/actions/publish", headers=h, timeout=60)
    r.raise_for_status()
    print("published:", r.json()["doi_url"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--create", action="store_true")
    g.add_argument("--refresh", type=int, metavar="DEPOSITION_ID")
    g.add_argument("--publish", type=int, metavar="DEPOSITION_ID")
    a = ap.parse_args()
    t = token()
    if a.create:
        create(t)
    elif a.refresh:
        refresh(t, a.refresh)
    else:
        publish(t, a.publish)
