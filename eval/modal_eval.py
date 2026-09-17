"""Evaluate the fine-tuned model against the untouched base model on the held-out pairs.

For each of the 200 test rows, both models rewrite the AI-styled input. We then
measure, against the human original:

  faithfulness  BERTScore F1 and ROUGE-L (does the meaning survive?),
                capitalised-token recall (are names and places kept?)
  style         length ratio, contractions per 100 words, transition-word
                rate, and a count of stock LLM phrases ("it is worth noting",
                "delve", "furthermore", ...)

These describe how far the output moved from AI-styled prose towards the human
target. They are NOT detector scores and make no claim about AI detectors.

Run:
    modal run eval/modal_eval.py --run-name open-humanizer-v1
Outputs eval/results/<run-name>.json and eval/results/<run-name>-samples.jsonl locally.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal

BASE_MODEL = "Qwen/Qwen3-4B"
VOLUME_NAME = "gohumanize-open-humanizer-models"
REMOTE_DATA = "/data"
REMOTE_MODELS = "/models"
REPO_ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = (
    "Rewrite the following text so that it reads as if a person wrote it: varied sentence "
    "length, concrete wording, natural rhythm, no filler transitions. Keep the meaning, the "
    "facts and the order of ideas. Return only the rewritten text."
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "bert-score", "rouge-score", "huggingface_hub")
    .add_local_dir(REPO_ROOT / "dataset", remote_path=REMOTE_DATA)
)
app = modal.App("gohumanize-open-humanizer-eval")
volume = modal.Volume.from_name(VOLUME_NAME)

TRANSITIONS = ["however", "moreover", "furthermore", "additionally", "in addition", "consequently",
               "therefore", "thus", "ultimately", "in essence", "in conclusion", "notably"]
STOCK_PHRASES = ["it is worth noting", "it is important to note", "delve", "tapestry", "testament to",
                 "plays a crucial role", "in today's", "a wide range of", "serves as a", "underscores",
                 "highlights the importance", "fosters", "landscape of", "embark", "navigate the"]
CONTRACTION_RE = re.compile(r"\b\w+'(s|t|re|ve|ll|d|m)\b", re.I)


def style_metrics(text: str) -> dict:
    words = text.split()
    n = max(len(words), 1)
    low = text.lower()
    return {
        "words": len(words),
        "contractions_per_100w": 100 * len(CONTRACTION_RE.findall(text)) / n,
        "transitions_per_100w": 100 * sum(low.count(t) for t in TRANSITIONS) / n,
        "stock_phrases": sum(low.count(p) for p in STOCK_PHRASES),
        "avg_sentence_len": n / max(len(re.findall(r"[.!?]+", text)), 1),
    }


def cap_recall(reference: str, candidate: str) -> float:
    """Share of capitalised tokens (names, places) from the reference that appear in the candidate."""
    caps = {w.strip(".,;:!?\"'()") for w in reference.split()[1:] if w[:1].isupper()}
    caps = {c for c in caps if len(c) > 1}
    if not caps:
        return 1.0
    cand = set(w.strip(".,;:!?\"'()") for w in candidate.split())
    return len(caps & cand) / len(caps)


@app.function(image=image, gpu="A10G", timeout=2 * 60 * 60, volumes={REMOTE_MODELS: volume})
def evaluate(run_name: str, max_rows: int = 200) -> dict:
    from bert_score import score as bert_score
    from rouge_score import rouge_scorer
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(f"{REMOTE_DATA}/test.jsonl")][:max_rows]
    models = {"base": BASE_MODEL, "finetuned": f"{REMOTE_MODELS}/{run_name}/merged-16bit"}
    outputs: dict[str, list[str]] = {}
    for name, path in models.items():
        llm = LLM(model=path, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85)
        tok = llm.get_tokenizer()
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": r["input"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
        gen = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.9, max_tokens=700, seed=13))
        outputs[name] = [g.outputs[0].text.strip() for g in gen]
        del llm
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()

    refs = [r["output"] for r in rows]
    inputs = [r["input"] for r in rows]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    report: dict = {"run_name": run_name, "rows": len(rows), "systems": {}}
    systems = {"ai_input": inputs, **outputs}
    for name, cands in systems.items():
        _, _, f1 = bert_score(cands, refs, lang="en", model_type="roberta-large", verbose=False)
        rouge = [scorer.score(r, c)["rougeL"].fmeasure for r, c in zip(refs, cands)]
        st = [style_metrics(c) for c in cands]
        report["systems"][name] = {
            "bertscore_f1_vs_human": round(float(f1.mean()), 4),
            "rougeL_vs_human": round(sum(rouge) / len(rouge), 4),
            "cap_token_recall": round(sum(cap_recall(r, c) for r, c in zip(refs, cands)) / len(refs), 4),
            "length_ratio_vs_human": round(sum(s["words"] for s in st) / sum(len(r.split()) for r in refs), 3),
            "contractions_per_100w": round(sum(s["contractions_per_100w"] for s in st) / len(st), 3),
            "transitions_per_100w": round(sum(s["transitions_per_100w"] for s in st) / len(st), 3),
            "stock_phrases_per_text": round(sum(s["stock_phrases"] for s in st) / len(st), 3),
            "avg_sentence_len": round(sum(s["avg_sentence_len"] for s in st) / len(st), 2),
        }
    st_h = [style_metrics(r) for r in refs]
    report["systems"]["human_target"] = {
        "contractions_per_100w": round(sum(s["contractions_per_100w"] for s in st_h) / len(st_h), 3),
        "transitions_per_100w": round(sum(s["transitions_per_100w"] for s in st_h) / len(st_h), 3),
        "stock_phrases_per_text": round(sum(s["stock_phrases"] for s in st_h) / len(st_h), 3),
        "avg_sentence_len": round(sum(s["avg_sentence_len"] for s in st_h) / len(st_h), 2),
    }
    report["samples"] = [{"id": r["id"], "title": r["title"], "input": r["input"], "human": r["output"],
                          "base": outputs["base"][i], "finetuned": outputs["finetuned"][i]}
                         for i, r in enumerate(rows)]
    return report


@app.local_entrypoint()
def main(run_name: str = "open-humanizer-v1", max_rows: int = 200):
    report = evaluate.remote(run_name, max_rows)
    out = REPO_ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    samples = report.pop("samples")
    (out / f"{run_name}.json").write_text(json.dumps(report, indent=2))
    with (out / f"{run_name}-samples.jsonl").open("w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))
