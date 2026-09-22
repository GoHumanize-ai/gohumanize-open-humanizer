# GoHumanize Open Humanizer

[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-model-yellow)](https://huggingface.co/gohumanize/gohumanize-open-humanizer)
[![Dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-dataset-yellow)](https://huggingface.co/datasets/gohumanize/gohumanize-open-humanizer-dataset)
[![PyPI](https://img.shields.io/pypi/v/gohumanize-open-humanizer)](https://pypi.org/project/gohumanize-open-humanizer/)
[![npm](https://img.shields.io/npm/v/gohumanize-open-humanizer-mcp)](https://www.npmjs.com/package/gohumanize-open-humanizer-mcp)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22843083.svg)](https://doi.org/10.5281/zenodo.22843083)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

An open, educational text-humanization model: a **full fine-tune of Qwen3-4B**
(Apache-2.0) that rewrites AI-styled English prose into more natural human writing,
trained on 2,000 pairs built from 47 public-domain books (Project Gutenberg). A QLoRA
version, trained in 22 minutes on one rented GPU for about a dollar, scores the same
and is published alongside it.

This repository is the complete, reproducible pipeline: sourcing and cleaning the
text, creating the AI-styled inputs, training, evaluation, serving, and the write-up
that explains every step and every service used.

| Resource | Where |
| --- | --- |
| Write-up (paper) | [`docs/paper.md`](docs/paper.md) |
| Model weights and GGUF (full fine-tune) | [huggingface.co/gohumanize/gohumanize-open-humanizer](https://huggingface.co/gohumanize/gohumanize-open-humanizer) |
| QLoRA version and LoRA adapter | [huggingface.co/gohumanize/gohumanize-open-humanizer-qlora](https://huggingface.co/gohumanize/gohumanize-open-humanizer-qlora) |
| Model card | [`docs/model-card.md`](docs/model-card.md) |
| Dataset | [huggingface.co/datasets/gohumanize/gohumanize-open-humanizer-dataset](https://huggingface.co/datasets/gohumanize/gohumanize-open-humanizer-dataset), card and files in [`dataset/`](dataset/) (2,000 train / 200 test, CC-BY 4.0) |
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

## Results in one table

Base Qwen3-4B vs the fine-tuned model on 200 held-out pairs, measured against the
human original (details and examples in the paper):

| Measure | AI-styled input | Base Qwen3-4B | Open Humanizer (full) | QLoRA version | Human |
|---|---|---|---|---|---|
| BERTScore F1 vs human | 0.914 | 0.900 | **0.920** | 0.921 | |
| ROUGE-L vs human | 0.493 | 0.425 | **0.535** | 0.540 | |
| Names kept (recall) | 0.587 | 0.593 | **0.614** | 0.623 | |
| Length ratio vs human | 1.03 | 0.83 | **0.95** | 0.93 | 1.00 |
| Transition words / 100 words | 0.76 | 0.02 | **0.11** | 0.11 | 0.14 |
| Stock LLM phrases / text | 0.19 | 0.01 | **0.00** | 0.00 | 0.00 |
| Average sentence length | 21.6 | 15.1 | **22.6** | 22.5 | 28.1 |

## Layout

```
pipeline/   01 source Gutenberg -> 02 select -> 03 AI-fy -> 04 build dataset
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
Biases key and a Hugging Face token. Total cost under $5.

```bash
pip install -r requirements.txt
cp .env.example .env            # fill in the keys
python pipeline/01_source_gutenberg.py --out data/passages_raw.jsonl
python pipeline/02_select.py --raw data/passages_raw.jsonl --train 2000 --test 200 --out-dir data
python pipeline/03_aify.py --human data/human_train.jsonl --out data/pairs_train.jsonl
python pipeline/03_aify.py --human data/human_test.jsonl  --out data/pairs_test.jsonl
python pipeline/04_build_dataset.py --pairs-dir data --out-dir dataset
modal run train/modal_train.py --max-steps 3 --run-name smoke     # validate the setup first
modal run train/modal_train.py --run-name open-humanizer-v1
modal run eval/modal_eval.py --run-name open-humanizer-v1
modal run train/push_to_hub.py::push --run-name open-humanizer-v1 --repo <org>/<model>
modal run train/push_to_hub.py::gguf --run-name open-humanizer-v1 --repo <org>/<model>
modal deploy serve/modal_serve.py
```

## Licences and citation

Code and model: Apache-2.0. Dataset: CC-BY 4.0. See [`CITATION.cff`](CITATION.cff).

> GoHumanize team (2026). *GoHumanize Open Humanizer: an open text-humanization model, dataset and pipeline built from public-domain data.* Version 0.1.0. Zenodo. https://doi.org/10.5281/zenodo.22843083
