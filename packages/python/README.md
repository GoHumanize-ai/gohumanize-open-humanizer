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
of GoHumanize.ai and makes no claim about AI detectors. Model, dataset, code and
write-up: https://gohumanize.ai/research
