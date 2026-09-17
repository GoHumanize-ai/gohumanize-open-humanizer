# GoHumanize Open Humanizer

An open, educational text-humanization model: a Qwen3-4B fine-tune that rewrites
AI-styled English prose into more natural human writing, trained on 2,000 pairs
built from public-domain books (Project Gutenberg).

This repository holds the complete pipeline: data sourcing and cleaning, the
AI-fication step that creates training inputs, training (Unsloth QLoRA on Modal),
evaluation, serving, and the write-up explaining every step.

- Model, dataset and demo: links will be added at release.
- Paper: `docs/paper.md`
- MCP server: https://github.com/GoHumanize-ai/gohumanize-open-humanizer-mcp

The Open Humanizer is separate from the production models used by GoHumanize.ai
and makes no claim about AI detectors. Licences: code and model Apache-2.0,
dataset CC-BY 4.0.

## Layout

```
pipeline/   01 source Gutenberg -> 02 select -> 03 AI-fy -> 04 build dataset
train/      modal_train.py   QLoRA fine-tune of Qwen3-4B on Modal
eval/       modal_eval.py    base vs fine-tuned on the held-out pairs
serve/      modal_serve.py   OpenAI-compatible vLLM endpoint
dataset/    released train/test files and stats (added at release)
docs/       paper and notes
```

## Reproduce

```bash
pip install -r requirements.txt
cp .env.example .env            # OPENAI_API_KEY, OPENROUTER_API_KEY, DEEPSEEK_API_KEY, WANDB_API_KEY, HF_TOKEN
python pipeline/01_source_gutenberg.py --out data/passages_raw.jsonl
python pipeline/02_select.py --raw data/passages_raw.jsonl --train 2000 --test 200 --out-dir data
python pipeline/03_aify.py --human data/human_train.jsonl --out data/pairs_train.jsonl
python pipeline/03_aify.py --human data/human_test.jsonl  --out data/pairs_test.jsonl
python pipeline/04_build_dataset.py --pairs-dir data --out-dir dataset
modal run train/modal_train.py
modal run eval/modal_eval.py --run-name open-humanizer-v1
modal deploy serve/modal_serve.py
```
