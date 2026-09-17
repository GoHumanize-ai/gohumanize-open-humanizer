# GoHumanize Open Humanizer: building a small text-humanization model from public-domain data

*GoHumanize team, September 2026. Draft outline; sections are filled in as results come in.*

## Abstract
<!-- one paragraph: what, with what data, what it shows, what it does not claim -->

## 1. Purpose and scope
- Educational, open release: model, dataset, code, this document.
- Framing text agreed with Yaro (see paper-notes.md): reflects high-level principles, not the production system.
- Explicit non-claim: no statement about AI-detector results.

## 2. The task
- Input: prose in the register typical of LLM output. Target: the same content as a person wrote it.
- Why "reverse distillation" from human originals works better than asking an LLM to "write like a human".

## 3. Data
### 3.1 Why public domain, and how it was checked
- Project Gutenberg, English originals only, published before 1929; translations excluded and why.
- List of the 47 books with authors and years (from dataset/stats.json).
### 3.2 Cleaning and passage extraction
- Boilerplate removal, paragraph grouping to 80-300 words, prose filters. Numbers: 3,644 passages.
### 3.3 Selection
- 2,000 train + 200 test, at most 46 per book, fiction/non-fiction split.
### 3.4 AI-fication (creating the input side)
- Three generators (gpt-4o-mini via OpenAI, Llama 3.3 70B via OpenRouter, DeepSeek-chat) and four style prompts; why mixing matters.
- Rejection rules (length window, markdown), retry behaviour, cost.
- What Gemini and Claude were not used for and why (quota, key).
### 3.5 Dataset card and licence (CC-BY 4.0)

## 4. Model and training
- Base: Qwen3-4B (Apache 2.0); why a 4B model for a demo.
- QLoRA with Unsloth: what LoRA is, rank 16, 4-bit base, loss on the answer only, chat template with thinking off.
- Hyperparameters table, hardware (Modal A10G), wall-clock and cost.
- Weights & Biases: what it tracks and the run link.

## 5. Evaluation
- Method: base vs fine-tuned on 200 held-out pairs; BERTScore/ROUGE-L for meaning, capitalised-token recall for names, style measures (contractions, transitions, stock phrases, sentence length) relative to the human target.
- Results table and 3-4 side-by-side examples.
- What these numbers do and do not mean.

## 6. Serving and integration
- Merged weights on Hugging Face; GGUF for Ollama/LM Studio.
- Modal + vLLM OpenAI-compatible endpoint, scale to zero; cost model.
- MCP server (npm) and Python package (PyPI): how a developer calls the model.
- Demo Space.

## 7. Services used and why
- OpenAI, OpenRouter, DeepSeek (data), Modal (GPU training and serving), Weights & Biases (tracking), Hugging Face (weights, dataset, demo), Zenodo (DOI), GitHub, npm, PyPI, Kaggle.

## 8. Limitations
- Old-fashioned prose bias from pre-1929 sources; small data; no detector evaluation; English only; hallucination risk on out-of-distribution input.

## 9. Reproducing the work
- Step-by-step commands, expected cost and time.

## References and links
