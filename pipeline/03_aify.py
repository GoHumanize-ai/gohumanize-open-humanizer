"""Step 3: produce the AI-styled side of each pair.

For every human passage we ask a large language model to rewrite it the way
LLMs typically write: smooth, formal, hedged, generic word choice, no
contractions, connective phrases, every sentence roughly the same length.
The rewrite becomes the model INPUT; the original human passage stays the
TARGET. The fine-tuned model therefore learns the reverse direction:
AI-styled prose -> the human original.

Generators are mixed (OpenAI + Google, and Anthropic when a key is set) so
the model does not learn the quirks of one provider. Several style prompts
are rotated for the same reason. Results are cached per passage id, so the
script can be re-run safely.

Run:
    python pipeline/03_aify.py --human data/human_train.jsonl --out data/pairs_train.jsonl
    python pipeline/03_aify.py --human data/human_test.jsonl  --out data/pairs_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("aify")

SYSTEM = (
    "You rewrite passages so that they read like typical output of a large language model. "
    "Keep every fact, event, name and the order of ideas. Keep roughly the same length "
    "(within 20 percent). Do not add headings, lists, quotation marks around the whole text, "
    "commentary, or a preamble. Return only the rewritten passage."
)

STYLES = {
    "formal_polished": (
        "Rewrite in a polished, formal register: expand contractions, prefer general and "
        "abstract vocabulary over concrete or unusual words, smooth out irregular sentence "
        "rhythms so sentences are of similar length, and open sentences with connective phrases "
        "such as 'Additionally', 'Furthermore', 'Moreover', 'In essence', 'It is worth noting that'."
    ),
    "explanatory": (
        "Rewrite as a clear, neutral explanation: state things plainly, add light framing such as "
        "'This highlights', 'This suggests', 'Ultimately', remove idiosyncratic phrasing and dialect, "
        "and keep a calm, even, slightly impersonal tone throughout."
    ),
    "hedged_helpful": (
        "Rewrite in a careful, helpful assistant tone: soften strong statements with hedges "
        "('may', 'can often', 'tends to'), use balanced sentence structures, replace vivid or "
        "archaic words with common modern equivalents, and close with a tidy summarising sentence."
    ),
    "corporate_clean": (
        "Rewrite in clean, efficient modern prose as a professional writing assistant would produce: "
        "short paragraphs, consistent sentence length, generic descriptive adjectives, transitional "
        "words between sentences, no contractions, no slang, no fragments."
    ),
}

# Gemini free tier allows 20 requests/day, so it is kept only as an optional generator.
PROVIDER_WEIGHTS = {"openai": 0.5, "openrouter": 0.25, "deepseek": 0.25, "gemini": 0.0, "anthropic": 0.0}
MODELS = {"openai": "gpt-4o-mini", "gemini": "gemini-2.5-flash", "anthropic": "claude-haiku-4-5",
          "openrouter": "meta-llama/llama-3.3-70b-instruct", "deepseek": "deepseek-chat"}


def call_openai(prompt: str, text: str) -> str:
    import openai

    r = openai.OpenAI().chat.completions.create(
        model=MODELS["openai"], temperature=0.8, max_tokens=1200,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"{prompt}\n\nPassage:\n{text}"}],
    )
    return r.choices[0].message.content or ""


def _openai_compatible(base_url: str, key_env: str, model: str, prompt: str, text: str) -> str:
    import openai

    r = openai.OpenAI(api_key=os.environ[key_env], base_url=base_url).chat.completions.create(
        model=model, temperature=0.8, max_tokens=1200,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"{prompt}\n\nPassage:\n{text}"}],
    )
    return r.choices[0].message.content or ""


def call_openrouter(prompt: str, text: str) -> str:
    return _openai_compatible("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", MODELS["openrouter"], prompt, text)


def call_deepseek(prompt: str, text: str) -> str:
    return _openai_compatible("https://api.deepseek.com", "DEEPSEEK_API_KEY", MODELS["deepseek"], prompt, text)


def call_anthropic(prompt: str, text: str) -> str:
    import anthropic

    r = anthropic.Anthropic().messages.create(
        model=MODELS["anthropic"], max_tokens=1200, temperature=0.8, system=SYSTEM,
        messages=[{"role": "user", "content": f"{prompt}\n\nPassage:\n{text}"}],
    )
    return "".join(b.text for b in r.content if getattr(b, "text", None))


def call_gemini(prompt: str, text: str) -> str:
    key = os.environ["GOOGLE_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELS['gemini']}:generateContent?key={key}"
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"parts": [{"text": f"{prompt}\n\nPassage:\n{text}"}]}],
        # thinking tokens count against maxOutputTokens on 2.5 models; disable thinking
        # for this plain rewrite task so the answer is never cut short.
        "generationConfig": {"temperature": 0.8, "maxOutputTokens": 4000, "thinkingConfig": {"thinkingBudget": 0}},
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with GEMINI_LOCK:  # free-tier quota: keep Gemini to one request every ~6.5 s
        global _gemini_last
        wait = _gemini_last + 6.5 - time.time()
        if wait > 0:
            time.sleep(wait)
        _gemini_last = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])


import threading
GEMINI_LOCK = threading.Lock()
_gemini_last = 0.0

CALLERS = {"openai": call_openai, "gemini": call_gemini, "anthropic": call_anthropic,
           "openrouter": call_openrouter, "deepseek": call_deepseek}


def clean_output(s: str) -> str:
    s = s.strip()
    for pre in ("Here is the rewritten passage:", "Rewritten passage:", "Passage:"):
        if s.lower().startswith(pre.lower()):
            s = s[len(pre):].strip()
    if len(s) > 2 and s[0] == s[-1] == '"':
        s = s[1:-1].strip()
    return s


def reject_reason(src: str, out: str) -> str | None:
    if not out:
        return "empty"
    a, b = len(src.split()), len(out.split())
    if b < 0.6 * a or b > 1.6 * a:
        return f"length {b}/{a}"
    if out.lstrip().startswith(("#", "-", "*", "1.")):
        return "markdown"
    return None


def aify(row: dict, provider: str, style: str) -> dict | None:
    prompt = STYLES[style]
    for attempt in range(5):
        try:
            out = clean_output(CALLERS[provider](prompt, row["text"]))
            why = reject_reason(row["text"], out)
            if why is None:
                return {
                    "id": row["id"], "input": out, "output": row["text"],
                    "generator": MODELS[provider], "style": style,
                    "gutenberg_id": row["gutenberg_id"], "title": row["title"],
                    "author": row["author"], "category": row["category"],
                }
            log.warning("rejected %s (%s/%s) attempt %d: %s", row["id"], provider, style, attempt + 1, why)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            wait = 15 * (attempt + 1) if "429" in msg else 3 * (attempt + 1)
            log.warning("%s error for %s: %s (retry in %ds)", provider, row["id"], msg[:100], wait)
            time.sleep(wait)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=0, help="only process the first N rows (smoke test)")
    ap.add_argument("--only-provider", default="", help="force one provider (testing)")
    args = ap.parse_args()

    providers = {p: w for p, w in PROVIDER_WEIGHTS.items() if w > 0}
    if args.only_provider:
        providers = {args.only_provider: 1.0}
    total_w = sum(providers.values())
    names, weights = list(providers), [providers[p] / total_w for p in providers]
    log.info("providers: %s", {p: round(w, 2) for p, w in zip(names, weights)})

    rows = [json.loads(l) for l in args.human.open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    done: dict[str, dict] = {}
    if args.out.exists():
        for l in args.out.open(encoding="utf-8"):
            r = json.loads(l)
            done[r["id"]] = r
    todo = [r for r in rows if r["id"] not in done]
    log.info("%d rows, %d already done, %d to do", len(rows), len(done), len(todo))

    rng = random.Random(args.seed)
    plan = []
    styles = list(STYLES)
    for i, r in enumerate(todo):
        plan.append((r, rng.choices(names, weights)[0], styles[i % len(styles)]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    with args.out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(aify, r, p, s): r["id"] for r, p, s in plan}
        for n, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res is None:
                failed += 1
            else:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()
            if n % 50 == 0:
                log.info("%d/%d done (%d failed)", n, len(plan), failed)
    log.info("finished: %d written, %d failed -> %s", len(plan) - failed, failed, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
