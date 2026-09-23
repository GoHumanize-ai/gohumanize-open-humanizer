"""How often does a model hand back its input instead of rewriting it?

The standard evaluation (modal_eval.py) compares outputs with the human original. It cannot
see the failure people actually hit in the demo: an output that is simply the input again.
This measures that directly, together with the fabrications a bad training set teaches.

For each model, every test passage is rewritten three times at temperature 0.9 (the demo's
setting) through vLLM (the demo's engine). Per output it records:

  near-copy     more than 90% of the input's words survive into the output
  lost number   a figure from the input is missing, counted on real rewrites only, since a
                near-copy keeps every number without trying
  invented      a press-release dateline, a footnote number glued to a sentence end, or a link
                that the input did not have

Run (the models must be on the Modal volume):
    DATASET_DIR=dataset modal run --detach eval/modal_copy_rate.py \\
        --runs open-humanizer-full-v5-s13,open-humanizer-full-lr2e5
Writes eval/results/copy-rate/<runs>.json with the summary per model.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / os.environ.get("DATASET_DIR", "dataset")
VOLUME = modal.Volume.from_name("gohumanize-open-humanizer-models")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("vllm")
         .add_local_file(DATASET_DIR / "test.jsonl", remote_path="/data/test.jsonl"))
app = modal.App("gohumanize-open-humanizer-copy-rate")

SYSTEM = ("Rewrite the following text so that it reads as if a person wrote it: varied sentence "
          "length, concrete wording, natural rhythm, no filler transitions. Keep the meaning, the "
          "facts and the order of ideas. Return only the rewritten text.")
SAMPLES = 3
NEAR_COPY = 0.9

DATELINE = re.compile(r"^\s*[A-Z][A-Z.\- ]{2,25}(,\s*(?:D\.C\.|[A-Z][a-z]{1,3}\.|[A-Z]{2}))?\s*[—–]\s*")
FOOTNOTE = re.compile(r"(?<=[a-z\)\"”’%][.,;:])[0-9]{1,2}(?=\s|$)")
LINK = re.compile(r"https?://|www\.")
NUMBER = re.compile(r"\d[\d,.]*")


def overlap(source: str, output: str) -> float:
    """Share of the source's words still present in the output, counted with repeats."""
    counts: dict[str, int] = {}
    for w in source.lower().split():
        counts[w] = counts.get(w, 0) + 1
    total = sum(counts.values()) or 1
    kept = 0
    for w in output.lower().split():
        if counts.get(w, 0) > 0:
            counts[w] -= 1
            kept += 1
    return kept / total


def lost_number(source: str, output: str) -> bool | None:
    wanted = [n.rstrip(".,") for n in NUMBER.findall(source)]
    if not wanted:
        return None
    present = {n.rstrip(".,") for n in NUMBER.findall(output)}
    return any(n not in present for n in wanted)


def invented(source: str, output: str) -> bool:
    return bool(DATELINE.match(output) and not DATELINE.match(source)
                or FOOTNOTE.search(output) and not FOOTNOTE.search(source)
                or LINK.search(output) and not LINK.search(source))


def summarise(rows: list[dict]) -> dict:
    def pct(flags: list[bool]) -> float:
        return round(100 * sum(flags) / max(len(flags), 1), 1)

    out: dict = {}
    for label, source in (("modern", "us-federal"), ("literary", "gutenberg")):
        pairs = [(r["input"], o) for r in rows if r["source"] == source for o in r["outputs"]]
        if not pairs:
            continue
        rewrites = [(i, o) for i, o in pairs if overlap(i, o) <= NEAR_COPY]
        lost = [x for x in (lost_number(i, o) for i, o in rewrites) if x is not None]
        out[label] = {"outputs": len(pairs),
                      "near_copy_pct": pct([overlap(i, o) > NEAR_COPY for i, o in pairs]),
                      "rewrites_losing_a_number_pct": pct(lost),
                      "invented": sum(invented(i, o) for i, o in pairs)}
    return out


@app.function(image=image, gpu="A10G", volumes={"/models": VOLUME}, timeout=90 * 60)
def generate(run_name: str) -> list[dict]:
    from vllm import LLM, SamplingParams

    rows = [json.loads(line) for line in open("/data/test.jsonl", encoding="utf-8")]
    llm = LLM(model=f"/models/{run_name}/merged-16bit", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.9)
    params = SamplingParams(temperature=0.9, top_p=0.9, max_tokens=1500, n=SAMPLES)
    messages = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["input"]}]
                for r in rows]
    results = llm.chat(messages, params, chat_template_kwargs={"enable_thinking": False})
    return [{"id": r["id"], "source": r["source"], "input": r["input"],
             "outputs": [c.text.strip() for c in g.outputs]} for r, g in zip(rows, results)]


@app.local_entrypoint()
def main(runs: str):
    names = [r.strip() for r in runs.split(",") if r.strip()]
    report = {name: summarise(rows) for name, rows in zip(names, generate.map(names))}
    out = REPO_ROOT / "eval" / "results" / "copy-rate"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{'+'.join(names)}.json"
    path.write_text(json.dumps({"samples_per_passage": SAMPLES, "temperature": 0.9,
                                "models": report}, indent=2))
    print(json.dumps(report, indent=2))
    print("wrote", path)
