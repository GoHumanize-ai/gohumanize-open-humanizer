"""Serve the GoHumanize Open Humanizer as an OpenAI-compatible API on Modal.

vLLM loads the merged 16-bit model from the training volume and exposes
`/v1/chat/completions` behind a bearer token. The container scales to zero
when idle, so the demo costs nothing while nobody uses it and roughly the
price of one A10G-hour per hour of active use.

The MCP server, the demo Space and the site page all talk to this endpoint
(or to any other OpenAI-compatible server running the same weights, such as
a local Ollama with the GGUF build).

Deploy:
    modal deploy serve/modal_serve.py
Test:
    curl -s $URL/v1/chat/completions -H "Authorization: Bearer $OPEN_HUMANIZER_API_KEY" \
      -H "Content-Type: application/json" -d '{"model":"gohumanize-open-humanizer",
      "messages":[{"role":"user","content":"<AI-styled text>"}]}'
"""

from __future__ import annotations

from pathlib import Path

import modal

VOLUME_NAME = "gohumanize-open-humanizer-models"
RUN_NAME = "open-humanizer-v1"
MODEL_DIR = f"/models/{RUN_NAME}/merged-16bit"
SERVED_NAME = "gohumanize-open-humanizer"
REPO_ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = (
    "Rewrite the following text so that it reads as if a person wrote it: varied sentence "
    "length, concrete wording, natural rhythm, no filler transitions. Keep the meaning, the "
    "facts and the order of ideas. Return only the rewritten text."
)


def _dotenv(path: Path, keys: tuple[str, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() in keys and v.strip():
                out[k.strip()] = v.strip().strip('"')
    return out


image = modal.Image.debian_slim(python_version="3.11").pip_install("vllm", "fastapi[standard]")
app = modal.App("gohumanize-open-humanizer-serve")
volume = modal.Volume.from_name(VOLUME_NAME)
secrets = [modal.Secret.from_dict(_dotenv(REPO_ROOT / ".env", ("OPEN_HUMANIZER_API_KEY",)))]


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/models": volume},
    secrets=secrets,
    scaledown_window=300,
    timeout=10 * 60,
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=8000, startup_timeout=15 * 60)
def serve():
    import os
    import subprocess

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_DIR,
        "--served-model-name", SERVED_NAME,
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.9",
        "--host", "0.0.0.0", "--port", "8000",
        "--api-key", os.environ["OPEN_HUMANIZER_API_KEY"],
    ]
    subprocess.Popen(cmd)
