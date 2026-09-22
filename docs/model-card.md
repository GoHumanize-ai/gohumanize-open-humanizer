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
- full-fine-tune
datasets:
- gohumanize/gohumanize-open-humanizer-dataset
---

# GoHumanize Open Humanizer

A small open model that rewrites AI-styled English prose into more natural human
writing. It is a **full fine-tune of Qwen3-4B** (all four billion weights
updated) on 2,000 pairs of (AI-styled passage, human original) built from
public-domain books (Project Gutenberg).

The Open Humanizer is a public research and educational model created to
demonstrate the general approach used to develop AI text humanization systems.
It is separate from the production models used by GoHumanize.ai, but it reflects
many of the same high-level principles: careful dataset preparation,
transformation of source text into training pairs, model fine-tuning,
evaluation and iterative improvement. It is not intended to reproduce the
architecture, data, configuration or performance of the production systems,
and **it makes no claim about AI detectors**.

Everything about the project (dataset, code, evaluation, write-up):
https://gohumanize.ai/open-model

## Use

Chat format, thinking disabled. The system prompt below is the one used in training.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

repo = "gohumanize/gohumanize-open-humanizer"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map="auto")

SYSTEM = ("Rewrite the following text so that it reads as if a person wrote it: varied sentence "
          "length, concrete wording, natural rhythm, no filler transitions. Keep the meaning, the "
          "facts and the order of ideas. Return only the rewritten text.")
messages = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "It is worth noting that the committee ultimately reached a consensus."}]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=600, temperature=0.7, top_p=0.9, do_sample=True)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

Run it locally without a GPU using the GGUF builds in `gguf/`:

```bash
ollama run hf.co/gohumanize/gohumanize-open-humanizer:Q4_K_M
```

Also available: an MCP server (`npx gohumanize-open-humanizer-mcp`), a Python
client (`pip install gohumanize-open-humanizer`) and a browser demo at
https://gohumanize.ai/open-model.

## Training

| | |
|---|---|
| Base model | Qwen/Qwen3-4B (Apache-2.0) |
| Method | Full fine-tune with Unsloth: every weight updated in bf16, 8-bit AdamW, gradient checkpointing |
| Data | 2,000 train / 200 test pairs, see the dataset card |
| Loss | on the assistant (human target) tokens only |
| Epochs, LR, batch | 2 epochs (250 steps), 2e-5 cosine, effective batch 16, max 1,024 tokens |
| Hardware | 1x NVIDIA H100 (80 GB) on Modal, 7.2 minutes of training |
| Tracking | [Weights & Biases run](https://wandb.ai/gohumanize/gohumanize-open-humanizer/runs/khrhh8sl) |
| Final losses | train 1.27, eval 1.38 (3.17 before training) |

We also trained the same model with QLoRA (a rank-16 adapter on a 4-bit base, on a
24 GB GPU). A full fine-tune rewrites every weight, so it can shift a model's writing
further than an adapter of 33 million numbers on a frozen 4-bit base; with a dataset
this small that ceiling never came into play, and the two score the same on every
measure below. The full fine-tune reached the lowest held-out loss, so it is the
published model. The QLoRA version,
including its small adapter, is at
[gohumanize/gohumanize-open-humanizer-qlora](https://huggingface.co/gohumanize/gohumanize-open-humanizer-qlora).
The write-up (section 5.4) compares them in detail.

## Evaluation

Base Qwen3-4B vs this model on the 200 held-out pairs, against the human original.

| Measure (200 held-out pairs) | AI-styled input | Base Qwen3-4B | Open Humanizer | Human target |
|---|---|---|---|---|
| BERTScore F1 vs human (higher = closer meaning) | 0.914 | 0.900 | 0.920 |  |
| ROUGE-L vs human (higher = closer wording) | 0.493 | 0.425 | 0.535 |  |
| Names/capitalised tokens kept (recall) | 0.587 | 0.593 | 0.614 |  |
| Length ratio vs human | 1.027 | 0.831 | 0.952 | 1.000 |
| Contractions per 100 words | 0.279 | 0.761 | 0.016 | 0.118 |
| Transition words per 100 words | 0.757 | 0.023 | 0.110 | 0.143 |
| Stock LLM phrases per text | 0.19 | 0.01 | 0.00 | 0.00 |
| Average sentence length (words) | 21.6 | 15.1 | 22.6 | 28.1 |

These measure how far the output moves from AI-styled prose towards the human
target. They are not detector scores.

## Limitations

- The human targets are pre-1929 prose, so the model leans towards a literary,
  slightly old-fashioned register.
- Trained on 80 to 300 word passages; rewrite long documents paragraph by paragraph.
- English only. Can drop or alter details on inputs unlike its training data;
  check facts in the output.
- Educational demo; no claims about AI detectors.

## Links

| Resource | Link |
| --- | --- |
| Project page and browser demo | [gohumanize.ai/open-model](https://gohumanize.ai/open-model) |
| Model weights and GGUF builds (full fine-tune) | [gohumanize/gohumanize-open-humanizer](https://huggingface.co/gohumanize/gohumanize-open-humanizer) |
| QLoRA version and LoRA adapter | [gohumanize/gohumanize-open-humanizer-qlora](https://huggingface.co/gohumanize/gohumanize-open-humanizer-qlora) |
| Dataset, 2,200 pairs (CC-BY 4.0) | [gohumanize/gohumanize-open-humanizer-dataset](https://huggingface.co/datasets/gohumanize/gohumanize-open-humanizer-dataset) |
| Code and full pipeline | [GoHumanize-ai/gohumanize-open-humanizer](https://github.com/GoHumanize-ai/gohumanize-open-humanizer) |
| Write-up: every step, service and result | [docs/paper.md](https://github.com/GoHumanize-ai/gohumanize-open-humanizer/blob/main/docs/paper.md) |
| Archived release, citable DOI | [10.5281/zenodo.22843083](https://doi.org/10.5281/zenodo.22843083) |
| Python client and CLI | [pypi.org/project/gohumanize-open-humanizer](https://pypi.org/project/gohumanize-open-humanizer/) |
| MCP server for AI assistants | [npm](https://www.npmjs.com/package/gohumanize-open-humanizer-mcp) · [source](https://github.com/GoHumanize-ai/gohumanize-open-humanizer-mcp) |
| Training run, loss curves and config | [Weights & Biases](https://wandb.ai/gohumanize/gohumanize-open-humanizer/runs/khrhh8sl) |

## Licence and citation

Apache-2.0. Cite as:

> GoHumanize team (2026). *GoHumanize Open Humanizer: an open text-humanization model, dataset and pipeline built from public-domain data.* Version 0.1.0. Zenodo. https://doi.org/10.5281/zenodo.22843083
