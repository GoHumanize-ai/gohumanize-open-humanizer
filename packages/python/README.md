# gohumanize-open-humanizer

Python client and CLI for the **GoHumanize Open Humanizer**, an open Qwen3-4B
fine-tune (Apache-2.0) that rewrites AI-styled English text into more natural
human prose. Trained on 2,000 pairs built from public-domain books.

```bash
pip install gohumanize-open-humanizer
open-humanizer "It is worth noting that the committee reached a consensus."
```

```python
from gohumanize_open_humanizer import Humanizer

h = Humanizer()  # public demo endpoint
print(h.humanize("It is worth noting that the committee reached a consensus."))

# your own copy of the model, e.g. Ollama with the GGUF build
h = Humanizer(base_url="http://localhost:11434/v1", model="gohumanize/open-humanizer")
```

Environment variables `OPEN_HUMANIZER_URL`, `OPEN_HUMANIZER_MODEL` and
`OPEN_HUMANIZER_API_KEY` set the same options. No dependencies beyond the
standard library.

The model is an educational release: it is separate from the production models
of GoHumanize.ai and makes no claim about AI detectors.

## Links

- Project page: https://gohumanize.ai/research
- Model weights, LoRA adapter and GGUF builds (Hugging Face): https://huggingface.co/gohumanize/gohumanize-open-humanizer
- Dataset (Hugging Face, CC-BY 4.0): https://huggingface.co/datasets/gohumanize/gohumanize-open-humanizer-dataset
- Code, pipeline and write-up (GitHub): https://github.com/GoHumanize-ai/gohumanize-open-humanizer
- Paper: https://github.com/GoHumanize-ai/gohumanize-open-humanizer/blob/main/docs/paper.md
- Archived release with DOI (Zenodo): https://doi.org/10.5281/zenodo.22843083
- Python client and CLI (PyPI): https://pypi.org/project/gohumanize-open-humanizer/
- MCP server (npm): https://www.npmjs.com/package/gohumanize-open-humanizer-mcp, source: https://github.com/GoHumanize-ai/gohumanize-open-humanizer-mcp
- Training run (Weights & Biases): https://wandb.ai/gohumanize/gohumanize-open-humanizer/runs/95wi8tdg
