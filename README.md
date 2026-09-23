# GoHumanize Open Humanizer

[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-model-yellow)](https://huggingface.co/gohumanize/gohumanize-open-humanizer)
[![Dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-dataset-yellow)](https://huggingface.co/datasets/gohumanize/gohumanize-open-humanizer-dataset)
[![PyPI](https://img.shields.io/pypi/v/gohumanize-open-humanizer)](https://pypi.org/project/gohumanize-open-humanizer/)
[![npm](https://img.shields.io/npm/v/gohumanize-open-humanizer-mcp)](https://www.npmjs.com/package/gohumanize-open-humanizer-mcp)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22843083.svg)](https://doi.org/10.5281/zenodo.22843083)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

An open, educational text-humanization model: a **full fine-tune of Qwen3-4B**
(Apache-2.0) that rewrites AI-styled English prose into more natural human writing,
trained on 2,957 pairs whose human side is public domain: passages from 47 books on
Project Gutenberg and modern prose from US federal agencies.

This repository is the complete, reproducible pipeline: sourcing and cleaning the
text, creating the AI-styled inputs, training, evaluation, serving, and the write-up
that explains every step and every service used.

| Resource | Where |
| --- | --- |
| Write-up (paper) | [`docs/paper.md`](docs/paper.md) |
| Model weights and GGUF | [huggingface.co/gohumanize/gohumanize-open-humanizer](https://huggingface.co/gohumanize/gohumanize-open-humanizer) |
| Model card | [`docs/model-card.md`](docs/model-card.md) |
| Dataset | [huggingface.co/datasets/gohumanize/gohumanize-open-humanizer-dataset](https://huggingface.co/datasets/gohumanize/gohumanize-open-humanizer-dataset), card and files in [`dataset/`](dataset/) (2,957 train / 300 test, CC-BY 4.0) |
| Archived release, DOI | [10.5281/zenodo.22843083](https://doi.org/10.5281/zenodo.22843083) |
| Evaluation results | [`eval/results/`](eval/results/) |
| Training runs | [Weights & Biases](https://wandb.ai/gohumanize/gohumanize-open-humanizer) |
| MCP server | [npm: gohumanize-open-humanizer-mcp](https://www.npmjs.com/package/gohumanize-open-humanizer-mcp), [source](https://github.com/GoHumanize-ai/gohumanize-open-humanizer-mcp) |
| Python client | [PyPI: gohumanize-open-humanizer](https://pypi.org/project/gohumanize-open-humanizer/), source in [`packages/python`](packages/python) |
| Project page and demo | https://gohumanize.ai/open-model |
| GoHumanize (the product this research comes from) | https://gohumanize.ai/ |

Model weights, GGUF files and the dataset are published on Hugging Face under
https://huggingface.co/gohumanize; the browser demo is on the project page.

The Open Humanizer is separate from the production models used by GoHumanize.ai and
**makes no claim about AI detectors**. Its purpose is to show how such a tool is
built so that developers and researchers can learn from and build on the work.

## Results

On 300 held-out passages, three tries each at temperature 0.9. The modern passages come
from 59 articles never seen in training.

| | |
|---|---|
| Modern passages returned nearly unchanged (more than 90% of the words kept) | 17.0% |
| Book passages returned nearly unchanged | 13.3% |
| Modern rewrites that lose a number | 28.5% |
| Invented datelines, footnote numbers or links (of 900 outputs) | 0 |

Against the human original, on the same 300 passages:

| Measure | AI-styled input | Base Qwen3-4B | **Open Humanizer** | Human |
|---|---|---|---|---|
| BERTScore F1 vs human | 0.918 | 0.903 | **0.929** | |
| ROUGE-L vs human | 0.522 | 0.434 | **0.580** | |
| Names kept (recall) | 0.601 | 0.610 | **0.652** | |
| Length ratio vs human | 1.16 | 0.85 | **0.95** | 1.00 |
| Transition words / 100 words | 1.13 | 0.02 | **0.11** | 0.15 |
| Stock LLM phrases / text | 0.70 | 0.01 | **0.03** | 0.02 |
| Average sentence length | 23.2 | 15.2 | **21.5** | 25.8 |

## Layout

```
pipeline/   01 source Gutenberg, 01b source US federal prose -> 02 select -> 03 AI-fy -> 04 build dataset
train/      modal_train.py (QLoRA or full fine-tune on Modal), push_to_hub.py (HF upload + GGUF), runs/ (summaries)
eval/       modal_eval.py (base vs fine-tuned), results/
serve/      modal_serve.py (OpenAI-compatible vLLM endpoint)
demo/       Gradio app (optional self-hosted demo)
packages/   python client + CLI
dataset/    released files, stats and dataset card
scripts/    zenodo_deposit.py (archive the release and mint a DOI)
docs/       paper, model card, notes
```

## Reproduce

Requirements: Python 3.10+, a [Modal](https://modal.com) account for the GPU steps,
API keys for the AI-fication generators (OpenAI, any OpenAI-compatible host for Llama 3.3 70B, DeepSeek), a Weights &
Biases key and a Hugging Face token. Total cost under $10.

```bash
pip install -r requirements.txt
cp .env.example .env            # fill in the keys
python pipeline/01_source_gutenberg.py --out data/passages_raw.jsonl
python pipeline/02_select.py --raw data/passages_raw.jsonl --train 2000 --test 200 --out-dir data
python pipeline/03_aify.py --human data/human_train.jsonl --out data/pairs_train.jsonl
python pipeline/03_aify.py --human data/human_test.jsonl  --out data/pairs_test.jsonl
python pipeline/01b_source_gov.py --out data/passages_gov.jsonl --per-source 200
python pipeline/02_select.py --raw data/passages_gov.jsonl --group-key url --cap-per source:450 --prefix human_modern --train 1000 --test 100 --out-dir data
python pipeline/03_aify.py --human data/human_modern_train.jsonl --out data/pairs_modernh_train.jsonl --register hard
python pipeline/03_aify.py --human data/human_modern_test.jsonl  --out data/pairs_modernh_test.jsonl  --register hard
python pipeline/04_build_dataset.py --pairs-dir data --prefixes pairs,pairs_modernh --holdout-by url --out-dir dataset
modal run train/modal_train.py --max-steps 3 --run-name smoke     # validate the setup first
modal run --detach train/modal_train.py --method full --learning-rate 2e-5 --seed 13 --run-name open-humanizer-full-v5-s13
modal run eval/modal_eval.py --run-name open-humanizer-full-v5-s13 --max-rows 300
modal run --detach eval/modal_copy_rate.py --runs open-humanizer-full-v5-s13
modal run train/push_to_hub.py::push --run-name open-humanizer-full-v5-s13 --repo <org>/<model>
modal run train/push_to_hub.py::gguf --run-name open-humanizer-full-v5-s13 --repo <org>/<model>
modal deploy serve/modal_serve.py
```

## Licences and citation

Code and model: Apache-2.0. Dataset: CC-BY 4.0. See [`CITATION.cff`](CITATION.cff).

> GoHumanize team (2026). *GoHumanize Open Humanizer: an open text-humanization model, dataset and pipeline built from public-domain data.* Version 0.2.0. Zenodo. https://doi.org/10.5281/zenodo.22843083
