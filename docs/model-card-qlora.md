---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen3-4B
pipeline_tag: text-generation
library_name: transformers
tags:
- humanizer
- text-rewriting
- style-transfer
- qwen3
- unsloth
- lora
- qlora
datasets:
- gohumanize/gohumanize-open-humanizer-dataset
---

# GoHumanize Open Humanizer (QLoRA version)

The QLoRA version of the [GoHumanize Open Humanizer](https://huggingface.co/gohumanize/gohumanize-open-humanizer),
a small open model that rewrites AI-styled English prose into more natural human
writing. The main published model is a full fine-tune of Qwen3-4B; this repository
holds the same model trained with QLoRA instead, for anyone who wants the small
adapter or wants to reproduce training on a 24 GB GPU. The two score the same on
every measure we use (write-up, section 5.4).

Educational release, separate from the production systems of GoHumanize.ai.
**It makes no claim about AI detectors.** Project page: https://gohumanize.ai/open-model

## Files

- Root: merged 16-bit weights (the adapter already applied to Qwen3-4B), usable like any `transformers` model.
- `lora/`: the LoRA adapter on its own (about 66 MB), to apply to `Qwen/Qwen3-4B` with PEFT.
- `gguf/`: Q4_K_M and Q8_0 builds for Ollama, LM Studio and llama.cpp
  (`ollama run hf.co/gohumanize/gohumanize-open-humanizer-qlora:Q4_K_M`).

Usage, system prompt and chat format are the same as for the
[main model](https://huggingface.co/gohumanize/gohumanize-open-humanizer).

## Training

| | |
|---|---|
| Base model | Qwen/Qwen3-4B (Apache-2.0) |
| Method | QLoRA with Unsloth: 4-bit base, LoRA rank 16, alpha 32, all attention and MLP projections |
| Data | 2,000 train / 200 test pairs, see the dataset card |
| Loss | on the assistant (human target) tokens only |
| Epochs, LR, batch | 2 epochs (250 steps), 2e-4 cosine, effective batch 16, max 1,024 tokens |
| Hardware | 1x NVIDIA A10G (24 GB) on Modal, 21.7 minutes |
| Tracking | [Weights & Biases run](https://wandb.ai/gohumanize/gohumanize-open-humanizer/runs/95wi8tdg) |
| Final losses | train 1.30, eval 1.39 (3.17 before training) |

## Evaluation

| Measure (200 held-out pairs) | Base Qwen3-4B | QLoRA (this repo) | Full fine-tune (main model) | Human |
|---|---|---|---|---|
| BERTScore F1 vs human | 0.900 | 0.921 | 0.920 | |
| ROUGE-L vs human | 0.424 | 0.540 | 0.535 | |
| Names kept (recall) | 0.594 | 0.623 | 0.614 | |
| Length ratio vs human | 0.83 | 0.93 | 0.95 | 1.00 |
| Stock LLM phrases / text | 0.01 | 0.00 | 0.00 | 0.00 |
| Average sentence length | 15.2 | 22.5 | 22.6 | 28.1 |

## Licence and citation

Apache-2.0. Cite as:

> GoHumanize team (2026). *GoHumanize Open Humanizer: an open text-humanization model, dataset and pipeline built from public-domain data.* Version 0.1.0. Zenodo. https://doi.org/10.5281/zenodo.22843083

GoHumanize, the product this research comes from: https://gohumanize.ai/
