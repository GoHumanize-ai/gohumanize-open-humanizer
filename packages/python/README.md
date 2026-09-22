# gohumanize-open-humanizer

[![PyPI](https://img.shields.io/pypi/v/gohumanize-open-humanizer)](https://pypi.org/project/gohumanize-open-humanizer/)
[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-model-yellow)](https://huggingface.co/gohumanize/gohumanize-open-humanizer)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22843083.svg)](https://doi.org/10.5281/zenodo.22843083)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Python client and CLI for the **GoHumanize Open Humanizer**, an open Qwen3-4B
fine-tune (Apache-2.0) that rewrites AI-styled English text into more natural
human prose. Trained on 2,000 pairs built from public-domain books.

```bash
pip install gohumanize-open-humanizer
open-humanizer "It is worth noting that the committee reached a consensus."
```

```python
from gohumanize_open_humanizer import Humanizer

# Your own copy of the model, which needs no key:
#   ollama pull hf.co/gohumanize/gohumanize-open-humanizer:Q4_K_M
h = Humanizer(base_url="http://localhost:11434/v1",
              model="hf.co/gohumanize/gohumanize-open-humanizer:Q4_K_M")
print(h.humanize("It is worth noting that the committee reached a consensus."))

# Humanizer() with no arguments points at the endpoint behind the browser demo,
# which is rate-limited and needs OPEN_HUMANIZER_API_KEY. To try the model without
# installing anything, use the demo at https://gohumanize.ai/open-model
```

The hosted endpoint sleeps when idle: the first request after a quiet period waits for
a GPU cold start, one to two minutes, and later calls take a second or two. The default
timeout is 300 seconds to allow for that.

Environment variables `OPEN_HUMANIZER_URL`, `OPEN_HUMANIZER_MODEL` and
`OPEN_HUMANIZER_API_KEY` set the same options. No dependencies beyond the
standard library.

The model is an educational release: it is separate from the production models
of GoHumanize.ai and makes no claim about AI detectors.

## Links

## Several rewrites, best one returned

The model learned from pre-1929 books, so on modern prose it sometimes plays safe and hands
the text back almost unchanged. Each call therefore asks the endpoint for five rewrites (it
generates them in parallel, so the wait is the same) and returns the one that moved furthest
from the input while keeping a sensible length. Endpoints that cannot generate several at
once, such as Ollama, are asked again only when the rewrite is barely a rewrite.

```python
Humanizer(samples=1)   # one request, no choosing
```

```bash
open-humanizer --samples 1 "It is worth noting that ..."
```

| Resource | Link |
| --- | --- |
| Project page and browser demo | [gohumanize.ai/open-model](https://gohumanize.ai/open-model) |
| GoHumanize (the product this research comes from) | [gohumanize.ai](https://gohumanize.ai/) |
| Model weights and GGUF builds (full fine-tune) | [gohumanize/gohumanize-open-humanizer](https://huggingface.co/gohumanize/gohumanize-open-humanizer) |
| QLoRA version and LoRA adapter | [gohumanize/gohumanize-open-humanizer-qlora](https://huggingface.co/gohumanize/gohumanize-open-humanizer-qlora) |
| Dataset, 2,200 pairs (CC-BY 4.0) | [gohumanize/gohumanize-open-humanizer-dataset](https://huggingface.co/datasets/gohumanize/gohumanize-open-humanizer-dataset) |
| Code and full pipeline | [GoHumanize-ai/gohumanize-open-humanizer](https://github.com/GoHumanize-ai/gohumanize-open-humanizer) |
| Write-up: every step, service and result | [docs/paper.md](https://github.com/GoHumanize-ai/gohumanize-open-humanizer/blob/main/docs/paper.md) |
| Archived release, citable DOI | [10.5281/zenodo.22843083](https://doi.org/10.5281/zenodo.22843083) |
| Python client and CLI | [pypi.org/project/gohumanize-open-humanizer](https://pypi.org/project/gohumanize-open-humanizer/) |
| MCP server for AI assistants | [npm](https://www.npmjs.com/package/gohumanize-open-humanizer-mcp) · [source](https://github.com/GoHumanize-ai/gohumanize-open-humanizer-mcp) |
| Training runs, loss curves and config | [Weights & Biases](https://wandb.ai/gohumanize/gohumanize-open-humanizer) |
