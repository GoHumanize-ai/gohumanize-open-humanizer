"""Client for the GoHumanize Open Humanizer.

The model is served behind any OpenAI-compatible chat endpoint (the public
demo endpoint by default, or a local Ollama / vLLM running the open weights).

    from gohumanize_open_humanizer import Humanizer
    h = Humanizer()                       # or Humanizer(base_url="http://localhost:11434/v1")
    print(h.humanize("It is worth noting that ..."))

Educational open model. It rewrites style and makes no claim about AI detectors.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:  # the single source of truth is pyproject.toml
    __version__ = _pkg_version("gohumanize-open-humanizer")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.1.5"

DEFAULT_URL = "https://gohumanize--gohumanize-open-humanizer-serve-serve.modal.run/v1"
DEFAULT_MODEL = "gohumanize-open-humanizer"

# The default endpoint is the one behind the browser demo; it is rate-limited and
# needs a key, so it is not open for general use. Say what to do instead of
# surfacing a bare 401.
ENDPOINT_HELP = (
    "The hosted endpoint refused the request: it requires an API key and is not open "
    "for general use.\n"
    "Run the model yourself, which needs no key:\n"
    "  ollama pull hf.co/gohumanize/gohumanize-open-humanizer:Q4_K_M\n"
    '  Humanizer(base_url="http://localhost:11434/v1", '
    'model="hf.co/gohumanize/gohumanize-open-humanizer:Q4_K_M")\n'
    "or pass api_key=..., or try the model in a browser at https://gohumanize.ai/open-model"
)
SYSTEM_PROMPT = (
    "Rewrite the following text so that it reads as if a person wrote it: varied sentence "
    "length, concrete wording, natural rhythm, no filler transitions. Keep the meaning, the "
    "facts and the order of ideas. Return only the rewritten text."
)


class Humanizer:
    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, timeout: float = 300.0):
        # 300 s, not 120: the hosted endpoint runs on a GPU that scales to zero, and a
        # cold start plus generation can exceed two minutes. A warm call takes a second
        # or two, and a local endpoint is immediate.
        self.base_url = (base_url or os.getenv("OPEN_HUMANIZER_URL") or DEFAULT_URL).rstrip("/")
        self.model = model or os.getenv("OPEN_HUMANIZER_MODEL") or DEFAULT_MODEL
        self.api_key = api_key or os.getenv("OPEN_HUMANIZER_API_KEY") or ""
        self.timeout = timeout

    def humanize(self, text: str, temperature: float = 0.7) -> str:
        """Rewrite one passage (best results with 50 to 400 words)."""
        body = {
            "model": self.model,
            "temperature": temperature,
            "top_p": 0.9,
            "max_tokens": 1500,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": text}],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        # No key and the default endpoint: the request will be refused, but only after
        # the hosted container has booted, which can take a couple of minutes.
        if not self.api_key and self.base_url == DEFAULT_URL.rstrip("/"):
            raise RuntimeError(ENDPOINT_HELP)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                raise RuntimeError(ENDPOINT_HELP) from error
            raise
        return data["choices"][0]["message"]["content"].strip()

    def humanize_document(self, text: str, temperature: float = 0.7) -> str:
        """Rewrite a longer text paragraph by paragraph."""
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        return "\n\n".join(self.humanize(p, temperature) for p in parts)


def main() -> None:
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog="open-humanizer", description="Rewrite AI-styled text with the GoHumanize Open Humanizer")
    ap.add_argument("text", nargs="?", help="text to rewrite; reads stdin when omitted")
    ap.add_argument("--url", help="OpenAI-compatible base URL (default: public demo endpoint)")
    ap.add_argument("--model", help="served model name")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()
    text = args.text if args.text is not None else sys.stdin.read()
    print(Humanizer(base_url=args.url, model=args.model).humanize_document(text, args.temperature))
