"""Hugging Face Space: browser demo for the GoHumanize Open Humanizer.

A thin Gradio front end over the OpenAI-compatible endpoint (vLLM on Modal).
Set OPEN_HUMANIZER_URL and OPEN_HUMANIZER_API_KEY as Space secrets.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import gradio as gr

BASE_URL = os.environ.get("OPEN_HUMANIZER_URL", "https://gohumanize--gohumanize-open-humanizer-serve-serve.modal.run/v1").rstrip("/")
API_KEY = os.environ.get("OPEN_HUMANIZER_API_KEY", "")
MODEL = os.environ.get("OPEN_HUMANIZER_MODEL", "gohumanize-open-humanizer")
SYSTEM_PROMPT = (
    "Rewrite the following text so that it reads as if a person wrote it: varied sentence "
    "length, concrete wording, natural rhythm, no filler transitions. Keep the meaning, the "
    "facts and the order of ideas. Return only the rewritten text."
)
MAX_WORDS = 600

EXAMPLE = (
    "It is worth noting that the committee ultimately reached a consensus regarding the proposed "
    "changes. Furthermore, the members expressed a considerable degree of satisfaction with the "
    "collaborative process, which facilitated a comprehensive examination of the relevant factors. "
    "Additionally, it was agreed that the implementation would commence in the subsequent quarter."
)


def humanize(text: str, temperature: float) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if len(text.split()) > MAX_WORDS:
        return f"Please keep the demo input under {MAX_WORDS} words."
    body = {
        "model": MODEL, "temperature": temperature, "top_p": 0.9, "max_tokens": 1200,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.load(resp)
        return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        return f"The model endpoint returned an error ({e.code}). Please try again in a minute."
    except Exception:  # noqa: BLE001
        return "The model endpoint did not respond. It scales to zero when idle; the first request can take up to two minutes. Please try again."


with gr.Blocks(title="GoHumanize Open Humanizer") as demo:
    gr.Markdown(
        "# GoHumanize Open Humanizer\n"
        "An open, educational Qwen3-4B fine-tune that rewrites AI-styled English text into more "
        "natural prose. Trained on 2,000 pairs built from public-domain books. "
        "It is separate from the production models of GoHumanize.ai and makes no claim about AI detectors. "
        "[Model, dataset, code and write-up](https://gohumanize.ai/research)\n\n"
        "The endpoint scales to zero when idle, so the first request after a pause can take up to two minutes."
    )
    with gr.Row():
        inp = gr.Textbox(label="AI-styled text (English, up to ~600 words)", lines=10, value=EXAMPLE)
        out = gr.Textbox(label="Rewritten", lines=10)
    temp = gr.Slider(0.1, 1.2, value=0.7, step=0.1, label="Temperature (lower = more literal)")
    gr.Button("Humanize", variant="primary").click(humanize, [inp, temp], out)

if __name__ == "__main__":
    demo.launch()
