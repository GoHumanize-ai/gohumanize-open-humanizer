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
public domain: 2,000 passages from books on Project Gutenberg and 957 passages of
modern prose from US federal agencies. The AI-styled side was written by OpenAI
gpt-4o-mini and DeepSeek-chat.

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

## How often it rewrites

A humanizer can fail by handing its input back almost unchanged. On 300 held-out
passages, three tries each at temperature 0.9:

| | |
|---|---|
| Modern passages returned nearly unchanged (more than 90% of the words kept), from 59 articles never seen in training | 25.3% |
| Book passages returned nearly unchanged | 17.5% |
| Modern rewrites that lose a number | 18.3% |
| Invented datelines, footnote numbers or links, of 900 outputs | 0 |

The AI-styled test inputs were regenerated for version 3 and are milder rewrites than before,
which leaves the model less to change. On the previous test set the same model returns 17.7% of
modern and 12.8% of book passages nearly unchanged.

The training pairs were built so the model learns to restructure sentences, not just swap
words, and press-release datelines and footnote numbers were removed from the targets so
the model does not learn to invent them. The write-up explains how.

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
four. Generating a few tries and keeping the one that changed most (the MCP server
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
| Data | 2,957 train / 300 test pairs, see the dataset card |
| Loss | on the assistant (human target) tokens only |
| Epochs, LR, batch | 2 epochs, 2e-5 cosine, effective batch 16, max 1,024 tokens, seed 13 |
| Hardware | 1x NVIDIA H100 (80 GB) on Modal, 10 minutes of training |
| Tracking | [Weights & Biases run](https://wandb.ai/gohumanize/gohumanize-open-humanizer/runs/vub8gt0w) |
| Final losses | train 0.92, eval 1.10 |

## Evaluation

The untouched base model and the Open Humanizer on the 300 held-out passages (100
modern, 200 from books), against the human original.

| Measure (300 held-out pairs) | AI-styled input | Base Qwen3-4B | **Open Humanizer** | Human target |
|---|---|---|---|---|
| BERTScore F1 vs human (higher = closer meaning) | 0.925 | 0.907 | **0.935** | |
| ROUGE-L vs human (higher = closer wording) | 0.560 | 0.454 | **0.616** | |
| Names/capitalised tokens kept (recall) | 0.637 | 0.621 | **0.692** | |
| Length ratio vs human | 1.11 | 0.84 | **0.97** | 1.00 |
| Contractions per 100 words | 0.24 | 0.59 | **0.06** | 0.16 |
| Transition words per 100 words | 0.97 | 0.02 | **0.09** | 0.15 |
| Stock LLM phrases per text | 0.53 | 0.01 | **0.03** | 0.02 |
| Average sentence length (words) | 22.9 | 15.3 | **22.6** | 25.8 |

These measures cannot see whether the model rewrote at all: an output identical to the
AI-styled input scores well on meaning and wording. That is why the rewrite rate above
is measured separately.

These measure how far the output moves from AI-styled prose towards the human
target. They are not detector scores.

## Limitations

- Two thirds of the human targets are pre-1929 books and the rest is US government
  writing, so the model leans literary on one side and official on the other.
  Casual modern writing is the gap.
- About one in five rewrites of number-heavy text loses at least one figure. Some of
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
| Model weights and GGUF builds | [gohumanize/gohumanize-open-humanizer](https://huggingface.co/gohumanize/gohumanize-open-humanizer) |
| Dataset, 3,257 pairs (CC-BY 4.0) | [gohumanize/gohumanize-open-humanizer-dataset](https://huggingface.co/datasets/gohumanize/gohumanize-open-humanizer-dataset) |
| Code and full pipeline | [GoHumanize-ai/gohumanize-open-humanizer](https://github.com/GoHumanize-ai/gohumanize-open-humanizer) |
| Write-up: every step, service and result | [docs/paper.md](https://github.com/GoHumanize-ai/gohumanize-open-humanizer/blob/main/docs/paper.md) |
| Archived release, citable DOI | [10.5281/zenodo.22843083](https://doi.org/10.5281/zenodo.22843083) |
| Python client and CLI | [pypi.org/project/gohumanize-open-humanizer](https://pypi.org/project/gohumanize-open-humanizer/) |
| MCP server for AI assistants | [npm](https://www.npmjs.com/package/gohumanize-open-humanizer-mcp) · [source](https://github.com/GoHumanize-ai/gohumanize-open-humanizer-mcp) |
| Training run, loss curves and config | [Weights & Biases](https://wandb.ai/gohumanize/gohumanize-open-humanizer/runs/vub8gt0w) |

## Licence and citation

Apache-2.0. Cite as:

> GoHumanize team (2026). *GoHumanize Open Humanizer: an open text-humanization model, dataset and pipeline built from public-domain data.* Version 0.3.0. Zenodo. https://doi.org/10.5281/zenodo.22843083
