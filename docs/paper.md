# GoHumanize Open Humanizer: building a small text-humanization model from public-domain data

*GoHumanize team, September 2026*

## Abstract

We describe, end to end, how we built a small open model that rewrites AI-styled English prose into more natural human writing. The human side of every training example comes from public-domain books on Project Gutenberg; the AI side was produced by asking three large language models to rewrite those passages in their own characteristic register. From 2,000 such pairs we fine-tuned Qwen3-4B with QLoRA in 22 minutes on a single rented GPU, for about one dollar. We explain each step, each service we used and why, how we evaluated the result against the untouched base model, and what the model can and cannot do. The model, dataset, code and this document are published under open licences so that developers and researchers can study, reproduce and extend the work. The Open Humanizer is an educational release: it is separate from the production systems of GoHumanize.ai and it makes no claim about AI detectors.

## 1. Purpose and scope

The GoHumanize Open Humanizer is a public research and educational model created to demonstrate the general approach used to develop AI text humanization systems. It is separate from the production models used by GoHumanize.ai, but it reflects many of the same high-level principles we follow when developing our technology, including careful dataset preparation, transformation of source text into training pairs, model fine-tuning, evaluation, and iterative improvement.

By publishing the model, dataset, code, methodology, and development process, we aim to provide developers and researchers with a practical example of how a humanization model can be built and studied. The open model is not intended to reproduce the exact architecture, datasets, training configuration, or performance of GoHumanize.ai's production systems.

Two boundaries are deliberate:

- **No detector claims.** We do not measure the model against AI-detection services and we make no statement about how detectors treat its output. The goal is to show how such a tool is developed, not to certify an outcome.
- **Public-domain data only.** Every human-written sentence in the dataset comes from a book that is out of copyright in the United States. This keeps the dataset redistributable under a permissive licence without asking anyone's permission.

## 2. The task

Text produced by large language models has a recognisable register: even sentence lengths, formal and general vocabulary, connective phrases at the start of sentences ("Furthermore", "It is worth noting that"), hedging, no contractions, a tidy summarising sentence at the end. A humanizer takes such text and rewrites it so that it keeps the meaning but reads as a person's writing: uneven rhythm, concrete words, contractions, fewer connectives, an occasional fragment.

The obvious way to get training data, asking an LLM to "write like a human", does not work well: the output is still LLM prose with contractions added. Our earlier internal work confirmed this repeatedly. What works is the reverse: start from text a person actually wrote, ask an LLM to rewrite it in its own style, and train the model to undo that rewrite. The human original is the target; the LLM rewrite is the input. We call the second step **AI-fication**, and the overall approach is sometimes called reverse distillation.

## 3. Data

### 3.1 Sources and the public-domain check

We used 47 books from Project Gutenberg, listed in the dataset card. Selection rules:

- English-language originals only. Translations were excluded even when the original is ancient, because the translation itself can carry a newer copyright. Two candidates on our first list were dropped for this reason.
- Published before 1929, so they are in the public domain in the United States under the rule that applied when we built the dataset. Most are 19th-century; the newest is from 1925.
- A mix of genres: 34 works of fiction (Austen, Dickens, Melville, Twain, the Brontës, Conrad, Wells, Wilde, Joyce, Fitzgerald and others) and 13 works of non-fiction: essays (Emerson, Thoreau), autobiography (Franklin, Douglass), speeches and political writing (Lincoln, Paine, the Federalist Papers), science and philosophy (Darwin, Mill, Hobbes, Russell, Smith, Du Bois), and satire (Swift).

Gutenberg files record title, author and language in a header; we parse those and keep them with every passage so the provenance of each row is auditable.

### 3.2 Cleaning and passage extraction

Gutenberg files come with a licence header and footer, and the text contains artefacts that would otherwise leak into the model: illustration tags, footnote markers, underscores used for italics, doubled hyphens. The pipeline (`pipeline/01_source_gutenberg.py`) removes the header and footer, normalises whitespace and punctuation, drops chapter headings, tables of contents, lines that are mostly capitals, and anything that mentions Gutenberg or transcription. What remains is grouped into passages of consecutive paragraphs of 80 to 300 words. This produced 3,644 candidate passages.

### 3.3 Selection

From the candidates we drew 2,000 training passages and 200 test passages (`pipeline/02_select.py`), round-robin across books so that no author dominates: at most 46 passages per book in training, average 147 words per passage, 1,468 fiction and 532 non-fiction. The test passages come from the same books. This tests the transformation the model learns, not whether it generalises to unseen authors, which would need a different split and a larger dataset.

### 3.4 AI-fication: creating the input side

For each passage we asked a large language model to rewrite it "so that it reads like typical output of a large language model", keeping every fact, name and the order of ideas, and roughly the same length (`pipeline/03_aify.py`). Two choices matter here.

**Several generators, not one.** If all inputs came from one model, the humanizer would learn that model's habits rather than the general LLM register. We used three, chosen for cost and variety: OpenAI's gpt-4o-mini (1,009 training rows), Meta's Llama 3.3 70B through OpenRouter (504) and DeepSeek-chat (487). We had also planned to use Google Gemini and Anthropic Claude; the Gemini free tier allows 20 requests per day, and our Anthropic key was not active, so both were left out. The code keeps them as optional generators.

**Several style prompts.** Four prompts rotate across rows: formal and polished, neutral explanatory, hedged assistant tone, and clean corporate prose. Each describes the register in concrete terms (expand contractions, prefer general vocabulary, even out sentence length, open with connectives, add a summarising sentence). The prompt name is stored with each row.

Outputs were rejected and regenerated if they were shorter than 60% or longer than 160% of the original, or came back as markdown or a list. About 1% of rows needed a second generator after five failed attempts, typically because Llama compressed a long literary passage too much. Generating all 2,200 pairs cost roughly one US dollar across the three providers and took 20 minutes with eight parallel requests.

### 3.5 What the pairs look like

A typical pair, from Frederick Douglass's *Narrative* (AI-styled input first, human original second):

> In the morning, our breakfasts were typically delivered through a small opening in the door, served in specially designed tin pans that held about a pint of chocolate, along with some brown bread and an iron spoon. When the time came to collect the containers...

> In the morning our breakfasts were passed in to us through the hole in the door in small square tin pans, made to fit, and holding a pint of chocolate, with brown bread, and an iron spoon...

The input is longer, smoother and vaguer ("typically delivered", "specially designed"); the target is shorter and concrete. That gap is what the model learns to close.

### 3.6 Dataset release

The dataset is published as JSONL and CSV with a card describing sources, licence, statistics and the full book list (`dataset/README.md`). Fields: `id`, `input`, `output`, `generator`, `style`, `gutenberg_id`, `title`, `author`, `category`, `input_words`, `output_words`. Licence: CC-BY 4.0.

## 4. Model and training

### 4.1 Base model

We fine-tuned **Qwen3-4B**, a 4-billion-parameter open model released under Apache-2.0. A model this size is a deliberate choice for an educational release: it trains on one consumer-class GPU in minutes, runs on a laptop after quantisation, and is small enough that anyone can repeat the experiment. It is not the size one would choose for the best possible quality.

### 4.2 QLoRA with Unsloth

Rather than updating all four billion weights, which would need far more GPU memory, we used **QLoRA**: the base model is loaded quantised to 4 bits and kept frozen, and a small set of trainable low-rank matrices (a **LoRA** adapter, rank 16, alpha 32) is added to every attention and MLP projection. Only the adapter is trained, then merged back into a full-precision copy of the base for serving. **Unsloth** is a library that implements this efficiently; it wraps Hugging Face's `transformers` and the **TRL** `SFTTrainer`.

Each pair is rendered with the Qwen3 chat template: a fixed system prompt ("Rewrite the following text so that it reads as if a person wrote it: varied sentence length, concrete wording, natural rhythm, no filler transitions. Keep the meaning, the facts and the order of ideas. Return only the rewritten text."), the AI-styled text as the user turn and the human original as the assistant turn, with Qwen3's thinking mode disabled. The loss is computed only on the assistant turn, so the model is trained to produce the human text, not to reproduce the prompt.

### 4.3 Settings and run

| Setting | Value |
|---|---|
| Base model | Qwen/Qwen3-4B, 4-bit base (bitsandbytes) |
| Adapter | LoRA rank 16, alpha 32, dropout 0, all q/k/v/o/gate/up/down projections |
| Data | 2,000 train, 200 test pairs |
| Epochs, steps | 2 epochs, 250 optimizer steps |
| Batch | 4 per device × 4 gradient accumulation = 16 |
| Learning rate | 2e-4, cosine schedule, 10 warm-up steps, weight decay 0.01 |
| Sequence length | up to 1,024 tokens |
| Precision | bf16 |
| Hardware | 1 × NVIDIA A10G (24 GB) on Modal |
| Wall-clock | 21.7 minutes of training, plus model download and merge |
| Cost | about $1 of GPU time |
| Software | torch 2.12.1, transformers 5.5.0, trl 0.24.0, peft 0.21.0, unsloth 2026.9.6, bitsandbytes 0.50.2 |
| Result | training loss 1.30, evaluation loss 1.39 (3.17 before training) |
| Tracking | Weights & Biases run `95wi8tdg` in project `gohumanize/gohumanize-open-humanizer` |

The evaluation loss fell from 3.17 (base model, measured on the same pairs before training) to 1.43 after the first 50 steps and 1.39 at the end; most of the learning happens early, which is common for style tasks with a strong base model. The training script is `train/modal_train.py`.

### 4.4 A note on getting the environment right

The first attempt at training failed four times before a single step ran, every time on library plumbing rather than on the data or the model: a missing local package, a file read that only works on the developer's machine, and two variants of a version clash between TRL and Unsloth (the fix was simply to import Unsloth before TRL so that its patches apply). We mention this because it is the normal experience with fast-moving ML libraries, and it is why the script supports a 3-step smoke run: spend a minute of GPU time to validate the setup before spending an hour.

## 5. Evaluation

### 5.1 Method

We compare the fine-tuned model with the untouched base model on the 200 held-out pairs (`eval/modal_eval.py`). Both receive the same system prompt and the same AI-styled input and generate a rewrite (temperature 0.7, top-p 0.9). We then measure two things against the human original:

- **Faithfulness**: BERTScore F1 and ROUGE-L between the output and the human original (does the meaning and wording survive?), and the share of capitalised tokens from the original (names, places) that appear in the output.
- **Style**: how the output's surface statistics compare with the human target: length ratio, contractions per 100 words, transition words per 100 words, count of stock LLM phrases ("it is worth noting", "delve", "testament to" and similar), and average sentence length.

We also report the same statistics for the AI-styled input itself and for the human target, which give the two ends of the scale. These numbers describe how far an output has moved from LLM prose towards the human original. They are not detector scores.

### 5.2 Results

*To be filled in from `eval/results/open-humanizer-v1.json`.*

### 5.3 Examples

*Three or four side-by-side examples from `eval/results/open-humanizer-v1-samples.jsonl`.*

## 6. Serving and integration

- **Weights** are on Hugging Face: merged 16-bit safetensors, the LoRA adapter, and GGUF files (Q4_K_M and Q8_0) built with llama.cpp so the model runs locally in Ollama, LM Studio or llama.cpp.
- **Endpoint**: `serve/modal_serve.py` runs vLLM on Modal behind an OpenAI-compatible API (`/v1/chat/completions`). The container scales to zero when idle, so the demo costs nothing while unused and roughly one A10G-hour per hour of use.
- **MCP server** (`npx gohumanize-open-humanizer-mcp`) exposes a `humanize_text` tool to AI assistants. It calls the endpoint above by default, or any OpenAI-compatible server you point it at, including a local Ollama running the GGUF.
- **Python client** (`pip install gohumanize-open-humanizer`) with an `open-humanizer` command, same options.
- **Demo Space** on Hugging Face for trying the model in a browser.

## 7. Services used, and why

| Service | Role | Why this one |
|---|---|---|
| Project Gutenberg | Human text | Large, free, clearly public domain, direct downloads. |
| OpenAI, OpenRouter, DeepSeek | AI-fication | Three different model families for variety; all cheap at this volume (about $1 in total). |
| Modal | GPU for training, evaluation and serving | Pay-per-second GPUs from a Python script, no servers to manage, scale-to-zero serving. |
| Weights & Biases | Experiment tracking | Every run's settings and loss curves are recorded and shareable. |
| Hugging Face | Hosting weights, dataset, demo | The standard place developers look for open models; free hosting. |
| llama.cpp | GGUF conversion | Lets the model run on CPUs and laptops. |
| GitHub | Code and this document | |
| npm, PyPI | MCP server and Python client | One-command install for developers. |
| Zenodo | Archived release with a DOI | Permanent, citable snapshot independent of any company account. |

## 8. Limitations

- **Old prose.** The human targets are pre-1929 books, so the model's idea of "human" leans literary and slightly old-fashioned. A production system would use contemporary human writing, which is much harder to obtain with a clean licence.
- **Small data, small model.** 2,000 pairs and a 4B model are enough to learn the style shift, not to handle every domain. Expect weaker results on technical or marketing text.
- **Faithfulness is not guaranteed.** Like any rewriting model it can drop or alter details; outputs should be checked against the source.
- **English only; passage-length inputs.** Trained on 80 to 300 word passages; longer texts should be processed paragraph by paragraph (the MCP server and Python client do this).
- **No detector evaluation**, by design.

## 9. Reproducing the work

```bash
git clone https://github.com/GoHumanize-ai/gohumanize-open-humanizer && cd gohumanize-open-humanizer
pip install -r requirements.txt && cp .env.example .env   # add API keys
python pipeline/01_source_gutenberg.py --out data/passages_raw.jsonl          # ~3 min
python pipeline/02_select.py --raw data/passages_raw.jsonl --train 2000 --test 200 --out-dir data
python pipeline/03_aify.py --human data/human_train.jsonl --out data/pairs_train.jsonl  # ~20 min, ~$1
python pipeline/03_aify.py --human data/human_test.jsonl  --out data/pairs_test.jsonl
python pipeline/04_build_dataset.py --pairs-dir data --out-dir dataset
modal run train/modal_train.py --max-steps 3 --run-name smoke                  # validate setup
modal run train/modal_train.py --run-name open-humanizer-v1                    # ~25 min, ~$1
modal run eval/modal_eval.py --run-name open-humanizer-v1
modal run train/push_to_hub.py::push --run-name open-humanizer-v1 --repo <org>/<model>
modal run train/push_to_hub.py::gguf --run-name open-humanizer-v1 --repo <org>/<model>
modal deploy serve/modal_serve.py
```

Total cost of one full reproduction: under $5. Total time: about two hours including waiting.

## Links

- Project page: https://gohumanize.ai/research
- Code and this document: https://github.com/GoHumanize-ai/gohumanize-open-humanizer
- Model, dataset, demo: Hugging Face, organisation GoHumanize-ai
- Training run: https://wandb.ai/gohumanize/gohumanize-open-humanizer/runs/95wi8tdg

## Citation

GoHumanize team (2026). *GoHumanize Open Humanizer: building a small text-humanization model from public-domain data.* https://gohumanize.ai/research

Licences: model and code Apache-2.0; dataset CC-BY 4.0.
