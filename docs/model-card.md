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
datasets:
- GoHumanize-ai/gohumanize-open-humanizer-dataset
---

# GoHumanize Open Humanizer

A small open model that rewrites AI-styled English prose into more natural human
writing. It is a QLoRA fine-tune of **Qwen3-4B** on 2,000 pairs of (AI-styled
passage, human original) built from public-domain books (Project Gutenberg).

The Open Humanizer is a public research and educational model created to
demonstrate the general approach used to develop AI text humanization systems.
It is separate from the production models used by GoHumanize.ai, but it reflects
many of the same high-level principles: careful dataset preparation,
transformation of source text into training pairs, model fine-tuning,
evaluation and iterative improvement. It is not intended to reproduce the
architecture, data, configuration or performance of the production systems,
and **it makes no claim about AI detectors**.

Everything about the project (dataset, code, evaluation, write-up):
https://gohumanize.ai/research

## Use

Chat format, thinking disabled. The system prompt below is the one used in training.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

repo = "GoHumanize-ai/gohumanize-open-humanizer"
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

Also available: a GGUF build in `gguf/` for Ollama / LM Studio / llama.cpp, an
MCP server (`npx gohumanize-open-humanizer-mcp`) and a Python client
(`pip install gohumanize-open-humanizer`).

## Training

| | |
|---|---|
| Base model | Qwen/Qwen3-4B (Apache-2.0) |
| Method | QLoRA with Unsloth: 4-bit base, LoRA rank 16, alpha 32, all attention and MLP projections |
| Data | 2,000 train / 200 test pairs, see the dataset card |
| Loss | on the assistant (human target) tokens only |
| Epochs, LR, batch | TBD |
| Hardware | 1x NVIDIA A10G (24 GB) on Modal, TBD minutes |
| Tracking | Weights & Biases, run link TBD |
| Final losses | train TBD, eval TBD |

The LoRA adapter is in `lora/`; the main files are the merged 16-bit weights.

## Evaluation

Base Qwen3-4B vs this model on the 200 held-out pairs, against the human original.
TBD table (BERTScore F1, ROUGE-L, name recall, length ratio, contractions,
transition words, stock phrases, sentence length).

These measure how far the output moves from AI-styled prose towards the human
target. They are not detector scores.

## Limitations

- The human targets are pre-1929 prose, so the model leans towards a literary,
  slightly old-fashioned register.
- Trained on 80 to 300 word passages; rewrite long documents paragraph by paragraph.
- English only. Can drop or alter details on inputs unlike its training data;
  check facts in the output.
- Educational demo; no claims about AI detectors.

## Licence and citation

Apache-2.0. Cite as: GoHumanize team, "GoHumanize Open Humanizer", 2026,
https://gohumanize.ai/research
