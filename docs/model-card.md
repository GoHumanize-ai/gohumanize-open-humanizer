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
writing. It is a **full fine-tune of Qwen3-4B** (all four billion weights updated)
on 2,957 pairs of (AI-styled passage, human original). The human side is
public domain: 2,000 passages from books on Project Gutenberg and, since version 2,
957 passages of modern prose from US federal agencies.

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

## What changed in version 2

Version 1 often returned modern text almost unchanged: on AI-styled versions of
modern articles it gave back a near-copy (more than 90% of the input's words kept)
38% of the time. The cause was in how the training pairs were made. Rewriting a
modern factual passage "in AI style" while keeping every fact and the length left
so little to change that the pairs taught the model to leave modern text alone.
Version 2 generates that side with a stricter instruction that keeps the facts but
forbids keeping the sentence structure, adds modern public-domain prose, and
removes press-release datelines and footnote numbers from the targets so the model
does not learn to invent them. The write-up (section 6) tells the whole story.

Near-copy rate on 300 held-out passages, three tries each at temperature 0.9:

| | Version 1 | **Version 2** |
|---|---|---|
| Modern passages (from 59 articles never seen in training) | 37.7% | **17.0%** |
| Book passages | 16.7% | **13.3%** |
| Modern rewrites that lose a number | 31.9% | **28.5%** |
| Invented datelines, footnote numbers or links, of 900 outputs | 0 | **0** |

Version 1 remains in this repository's history.

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
out = model.generate(ids, max_new_tokens=600, temperature=0.9, top_p=0.9, do_sample=True)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

On modern text, a single try still comes back nearly unchanged about one time in
six. Generating a few tries and keeping the one that changed most (the MCP server
and Python client do this for you) takes that close to zero.

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
| Data | 2,957 train / 300 test pairs (dataset version 2), see the dataset card |
| Loss | on the assistant (human target) tokens only |
| Epochs, LR, batch | 2 epochs, 2e-5 cosine, effective batch 16, max 1,024 tokens, seed 13 |
| Hardware | 1x NVIDIA H100 (80 GB) on Modal, 11 minutes of training |
| Tracking | [Weights & Biases run](https://wandb.ai/gohumanize/gohumanize-open-humanizer/runs/na5tpdrj) |
| Final losses | train 1.02, eval 1.19 (not comparable with version 1: the test set changed) |

The version 1 QLoRA build (a rank-16 adapter on a 4-bit base, trained on a 24 GB GPU
for about a dollar), including its small adapter, is at
[gohumanize/gohumanize-open-humanizer-qlora](https://huggingface.co/gohumanize/gohumanize-open-humanizer-qlora).
The write-up compares QLoRA and full fine-tuning in detail (sections 5.4 and 6.9).

## Evaluation

Base Qwen3-4B, version 1 and version 2 on the 300 held-out passages (100 modern, 200
from books), against the human original. The base model scored the same in both
evaluation runs (BERTScore 0.9033 and 0.9034), so the columns are comparable.

| Measure (300 held-out pairs) | AI-styled input | Base Qwen3-4B | Version 1 | **Version 2** | Human target |
|---|---|---|---|---|---|
| BERTScore F1 vs human (higher = closer meaning) | 0.918 | 0.903 | 0.927 | **0.929** | |
| ROUGE-L vs human (higher = closer wording) | 0.522 | 0.434 | 0.569 | **0.580** | |
| Names/capitalised tokens kept (recall) | 0.601 | 0.610 | 0.640 | **0.652** | |
| Length ratio vs human | 1.16 | 0.85 | 0.98 | **0.95** | 1.00 |
| Contractions per 100 words | 0.33 | 0.69 | 0.07 | **0.08** | 0.16 |
| Transition words per 100 words | 1.13 | 0.02 | 0.11 | **0.11** | 0.15 |
| Stock LLM phrases per text | 0.70 | 0.01 | 0.03 | **0.03** | 0.02 |
| Average sentence length (words) | 23.2 | 15.2 | 22.2 | **21.5** | 25.8 |

Version 2 keeps meaning, wording and names better than version 1 and is about level
on style; the large difference between them is the copy rate above, which these
measures cannot see: an output identical to the AI-styled input scores well on
meaning and wording without having rewritten anything.

These measure how far the output moves from AI-styled prose towards the human
target. They are not detector scores.

## Limitations

- Two thirds of the human targets are pre-1929 books and the rest is US government
  writing, so the model leans literary on one side and official on the other.
  Casual modern writing is the gap.
- About three in ten rewrites of number-heavy text lose at least one figure. Some of
  that is rewording ("20 percent" as "a fifth"), some is loss. Check figures and
  other facts against the source.
- Trained on 80 to 300 word passages; rewrite long documents paragraph by paragraph.
- English only.
- Educational demo; no claims about AI detectors.

## Links

| Resource | Link |
| --- | --- |
| Project page and browser demo | [gohumanize.ai/open-model](https://gohumanize.ai/open-model) |
| GoHumanize (the product this research comes from) | [gohumanize.ai](https://gohumanize.ai/) |
| Model weights and GGUF builds (version 2 full fine-tune) | [gohumanize/gohumanize-open-humanizer](https://huggingface.co/gohumanize/gohumanize-open-humanizer) |
| Version 1 QLoRA and LoRA adapter | [gohumanize/gohumanize-open-humanizer-qlora](https://huggingface.co/gohumanize/gohumanize-open-humanizer-qlora) |
| Dataset, 3,257 pairs (CC-BY 4.0) | [gohumanize/gohumanize-open-humanizer-dataset](https://huggingface.co/datasets/gohumanize/gohumanize-open-humanizer-dataset) |
| Code and full pipeline | [GoHumanize-ai/gohumanize-open-humanizer](https://github.com/GoHumanize-ai/gohumanize-open-humanizer) |
| Write-up: every step, service and result | [docs/paper.md](https://github.com/GoHumanize-ai/gohumanize-open-humanizer/blob/main/docs/paper.md) |
| Archived release, citable DOI | [10.5281/zenodo.22843083](https://doi.org/10.5281/zenodo.22843083) |
| Python client and CLI | [pypi.org/project/gohumanize-open-humanizer](https://pypi.org/project/gohumanize-open-humanizer/) |
| MCP server for AI assistants | [npm](https://www.npmjs.com/package/gohumanize-open-humanizer-mcp) · [source](https://github.com/GoHumanize-ai/gohumanize-open-humanizer-mcp) |
| Training run, loss curves and config | [Weights & Biases](https://wandb.ai/gohumanize/gohumanize-open-humanizer/runs/na5tpdrj) |

## Licence and citation

Apache-2.0. Cite as:

> GoHumanize team (2026). *GoHumanize Open Humanizer: an open text-humanization model, dataset and pipeline built from public-domain data.* Version 0.2.0. Zenodo. https://doi.org/10.5281/zenodo.22843083
