"""Create (or update) the Zenodo deposit for the GoHumanize Open Humanizer release.

Uploads a source snapshot of this repository (code, docs, dataset, evaluation
results; no weights, they live on Hugging Face) plus the paper and the dataset
files as separate downloads, with citation metadata. The deposit is left as a
DRAFT; publishing (which mints the DOI and is irreversible) is a separate,
explicit step:

    python scripts/zenodo_deposit.py --create            # draft + files + metadata
    python scripts/zenodo_deposit.py --refresh <id>      # replace files + metadata on a draft
    python scripts/zenodo_deposit.py --publish <id>      # mint the DOI
    python scripts/zenodo_deposit.py --update-metadata <id>   # fix metadata on a published record
    python scripts/zenodo_deposit.py --new-version <id>  # draft a new version of a published record

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
VERSION = "0.2.0"  # the research release: model and dataset version 2. The PyPI and npm clients
                   # are versioned on their own and did not change for it.

METADATA = {
    "metadata": {
        "title": "GoHumanize Open Humanizer: an open text-humanization model, dataset and pipeline built from public-domain data",
        "upload_type": "software",
        "description": (
            "<p>The GoHumanize Open Humanizer is a public research and educational model: a Qwen3-4B "
            "fine-tune that rewrites AI-styled English prose into more natural human writing. "
            "This record archives the complete pipeline (sourcing and cleaning of public-domain text from "
            "Project Gutenberg and US federal agencies, AI-fication of inputs with three generators, "
            "training, evaluation, serving), the dataset of 2,957 training and 300 test pairs (CC-BY 4.0), "
            "the evaluation results and the paper-style write-up explaining every step and service used.</p>"
            "<p><b>Version 0.2.0.</b> Version 1 often returned modern text nearly unchanged. The cause was in "
            "how the AI-styled side of the training pairs was generated; version 2 fixes it and adds modern "
            "public-domain prose, which takes the near-copy rate on unseen modern articles from 38% to 17% "
            "(write-up, section 6).</p>"
            "<p><b>Links</b></p><ul>"
            "<li>Project page and browser demo: https://gohumanize.ai/open-model</li>"
            "<li>GoHumanize, the product this research comes from: https://gohumanize.ai/</li>"
            "<li>Model weights and GGUF builds (version 2 full fine-tune): https://huggingface.co/gohumanize/gohumanize-open-humanizer</li>"
            "<li>Version 1 QLoRA and LoRA adapter: https://huggingface.co/gohumanize/gohumanize-open-humanizer-qlora</li>"
            "<li>Dataset (CC-BY 4.0): https://huggingface.co/datasets/gohumanize/gohumanize-open-humanizer-dataset</li>"
            "<li>Code, pipeline and write-up: https://github.com/GoHumanize-ai/gohumanize-open-humanizer</li>"
            "<li>Paper: https://github.com/GoHumanize-ai/gohumanize-open-humanizer/blob/main/docs/paper.md</li>"
            "<li>Python client and CLI: https://pypi.org/project/gohumanize-open-humanizer/</li>"
            "<li>MCP server: https://www.npmjs.com/package/gohumanize-open-humanizer-mcp "
            "(source: https://github.com/GoHumanize-ai/gohumanize-open-humanizer-mcp)</li>"
            "<li>Training runs: https://wandb.ai/gohumanize/gohumanize-open-humanizer "
            "(version 2 na5tpdrj; version 1 full fine-tune khrhh8sl, QLoRA 95wi8tdg)</li></ul>"
            "<p>The Open Humanizer is separate from the production models used by GoHumanize.ai and makes "
            "no claim about AI detectors.</p>"
        ),
        "creators": [{"name": "GoHumanize team", "affiliation": "GoHumanize"}],
        "version": VERSION,
        "license": "apache-2.0",
        "keywords": ["humanizer", "text rewriting", "style transfer", "Qwen3", "LoRA", "public domain",
                     "Project Gutenberg", "US government", "dataset", "open model"],
        "related_identifiers": [
            {"identifier": "https://github.com/GoHumanize-ai/gohumanize-open-humanizer", "relation": "isSupplementTo", "scheme": "url"},
            {"identifier": "https://huggingface.co/gohumanize/gohumanize-open-humanizer", "relation": "isSupplementedBy", "scheme": "url"},
            {"identifier": "https://huggingface.co/datasets/gohumanize/gohumanize-open-humanizer-dataset", "relation": "isSupplementedBy", "scheme": "url"},
            {"identifier": "https://gohumanize.ai/open-model", "relation": "isDescribedBy", "scheme": "url"},
            {"identifier": "https://pypi.org/project/gohumanize-open-humanizer/", "relation": "isSupplementedBy", "scheme": "url"},
            {"identifier": "https://www.npmjs.com/package/gohumanize-open-humanizer-mcp", "relation": "isSupplementedBy", "scheme": "url"},
            {"identifier": "https://github.com/GoHumanize-ai/gohumanize-open-humanizer-mcp", "relation": "isSupplementedBy", "scheme": "url"},
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


def update_metadata(tok: str, dep_id: int) -> None:
    """Change the metadata of a PUBLISHED record in place (no new version or DOI)."""
    h = {"Authorization": f"Bearer {tok}"}
    r = requests.post(f"{API}/deposit/depositions/{dep_id}/actions/edit", headers=h, timeout=60)
    if r.status_code not in (201, 400):  # 400 = already in edit mode
        r.raise_for_status()
    r = requests.put(f"{API}/deposit/depositions/{dep_id}", json=METADATA, headers=h, timeout=60)
    r.raise_for_status()
    r = requests.post(f"{API}/deposit/depositions/{dep_id}/actions/publish", headers=h, timeout=60)
    r.raise_for_status()
    print("metadata updated:", r.json()["doi_url"])


def new_version(tok: str, dep_id: int) -> None:
    """Start a new version of a PUBLISHED record: a draft with this repository's current
    files and metadata, and its own DOI. Left unpublished; run --publish when happy."""
    h = {"Authorization": f"Bearer {tok}"}
    r = requests.post(f"{API}/deposit/depositions/{dep_id}/actions/newversion", headers=h, timeout=60)
    r.raise_for_status()
    draft_url = r.json()["links"]["latest_draft"]
    dep = requests.get(draft_url, headers=h, timeout=60).json()
    # The draft starts as a copy of the previous version's files; replace them all.
    for f in dep.get("files", []):
        requests.delete(f"{API}/deposit/depositions/{dep['id']}/files/{f['id']}", headers=h, timeout=60)
        print("removed", f["filename"])
    dep = requests.get(draft_url, headers=h, timeout=60).json()
    upload_all(tok, dep)
    set_metadata(tok, dep)
    print("new draft:", dep["links"]["html"], "-> publish with --publish", dep["id"])


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
    g.add_argument("--update-metadata", type=int, metavar="DEPOSITION_ID")
    g.add_argument("--new-version", type=int, metavar="PUBLISHED_DEPOSITION_ID")
    a = ap.parse_args()
    t = token()
    if a.create:
        create(t)
    elif a.refresh:
        refresh(t, a.refresh)
    elif a.update_metadata:
        update_metadata(t, a.update_metadata)
    elif a.new_version:
        new_version(t, a.new_version)
    else:
        publish(t, a.publish)
