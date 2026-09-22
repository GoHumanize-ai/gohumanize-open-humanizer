# GoHumanize Open Humanizer: building a small text-humanization model from public-domain data

*GoHumanize team, September 2026*

## Abstract

We describe, end to end, how we built a small open model that rewrites AI-styled English prose into more natural human writing. The human side of every training example comes from public-domain books on Project Gutenberg; the AI side was produced by asking three large language models to rewrite those passages in their own characteristic register. From 2,000 such pairs we fine-tuned Qwen3-4B in two ways: with QLoRA, in 22 minutes on a single rented 24 GB GPU for about one dollar, and as a full fine-tune of all four billion weights on an 80 GB GPU. The two score the same. The full fine-tune, which reached the lowest held-out loss, is the published model; the QLoRA version is released alongside it as the cheaper recipe to reproduce. We explain each step, each service we used and why, how we evaluated the result against the untouched base model, and what the model can and cannot do. The model, dataset, code and this document are published under open licences so that developers and researchers can study, reproduce and extend the work. The Open Humanizer is an educational release: it is separate from the production systems of GoHumanize.ai and it makes no claim about AI detectors.

## 1. Purpose and scope

The GoHumanize Open Humanizer is a public research and educational model created to demonstrate the general approach used to develop AI text humanization systems. It is separate from the production models used by GoHumanize.ai, but it reflects many of the same high-level principles we follow when developing our technology, including careful dataset preparation, transformation of source text into training pairs, model fine-tuning, evaluation, and iterative improvement.

By publishing the model, dataset, code, methodology, and development process, we aim to provide developers and researchers with a practical example of how a humanization model can be built and studied. The open model is not intended to reproduce the exact architecture, datasets, training configuration, or performance of GoHumanize.ai's production systems.

The write-up is aimed at developers who have not fine-tuned a model before. Each section explains not only what we did but why we chose it over the alternatives, and terms are explained as they come up. If you only want to run the model, the links at the end are enough; if you want to build something similar, read on.

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

**Why this step exists.** To teach a model to turn AI-styled text into human text, we need pairs: an AI-styled passage and the human version of the same passage. Human text is easy to find. The matching AI version does not exist anywhere, so we make it: we take each human passage and ask a large language model to rewrite it in the style LLMs typically write in. That rewrite becomes the training input, and the untouched human original is the answer the model has to learn to produce. We call this AI-fication. The direction matters: the target is always real human writing, never an LLM's imitation of it.

**How one passage is AI-fied** (`pipeline/03_aify.py`), step by step:

1. Pick a generator at random, with weights: gpt-4o-mini 50%, Llama 3.3 70B 25%, DeepSeek-chat 25%. The random seed is fixed (13), so a re-run makes the same choices.
2. Pick one of four style prompts, in rotation, so each style covers a quarter of the rows.
3. Send the passage with a fixed system instruction and the style prompt, at temperature 0.8. Temperature controls randomness; 0.8 gives some variety so the same instruction does not produce identical phrasing every time.
4. Clean the answer (strip preambles such as "Here is the rewritten passage:" and wrapping quotes) and check it. Reject it if it is shorter than 60% or longer than 160% of the original word count, or looks like markdown or a list. On a rejection or an API error, retry up to five times, waiting longer after rate-limit errors.
5. Save the pair straight away to a JSONL file together with the generator and style names. The script skips ids already in the file, so it can be stopped and resumed without paying twice.

The system instruction is the same for every call:

> You rewrite passages so that they read like typical output of a large language model. Keep every fact, event, name and the order of ideas. Keep roughly the same length (within 20 percent). Do not add headings, lists, quotation marks around the whole text, commentary, or a preamble. Return only the rewritten passage.

"Keep every fact, event, name and the order of ideas" is the important line. If the AI version dropped or invented content, the humanizer would learn to drop or invent content too. We want it to learn only the change of *style*.

**Four style prompts.** Each describes one flavour of LLM prose in concrete, checkable terms:

| Style | What it asks the generator to do | Training rows |
|---|---|---|
| `formal_polished` | expand contractions, prefer general vocabulary, even out sentence length, open sentences with "Additionally", "Furthermore", "It is worth noting that" | 498 |
| `explanatory` | plain neutral explanation, framing such as "This highlights" or "Ultimately", remove dialect and odd phrasing | 503 |
| `hedged_helpful` | soften claims ("may", "tends to"), swap vivid or old words for common modern ones, end with a tidy summarising sentence | 497 |
| `corporate_clean` | short paragraphs, consistent sentence length, generic adjectives, transition words, no contractions or fragments | 502 |

With one prompt the model would learn to undo one pattern. Four prompts cover the main registers people actually paste into a humanizer.

**Three generators, and why these.** If every input came from one model, the humanizer would learn to undo that model's particular habits (its favourite words, its sentence shapes) rather than LLM style in general. So we mixed three model families from three different labs:

- **OpenAI gpt-4o-mini** (1,009 training rows): the style most people recognise as "ChatGPT writing", reliable and very cheap. It got the largest share for that reason.
- **Meta Llama 3.3 70B Instruct** (504 rows): an open-weights model with a different training recipe and different habits. Many hosts serve it through an OpenAI-compatible API, and you can also run it yourself; the script only needs a base URL and key.
- **DeepSeek-chat** (487 rows): a third, independent family with its own phrasing, also very cheap.

AI-fication does not need the strongest available model, only typical ones, because the goal is typical LLM prose. We had planned to add Google Gemini and Anthropic Claude as well; the Gemini free tier allowed only 20 requests per day and our Anthropic key was not active at the time, so both were left out. The code keeps them as optional generators (give them a weight above zero).

**Numbers.** About 1% of rows failed five times with their first generator, typically because Llama compressed a long literary passage too much, and were redone with another. Generating all 2,200 pairs cost roughly one US dollar and took about 20 minutes with eight parallel requests.

### 3.5 What the pairs look like

A typical pair, from Frederick Douglass's *Narrative* (AI-styled input first, human original second):

> In the morning, our breakfasts were typically delivered through a small opening in the door, served in specially designed tin pans that held about a pint of chocolate, along with some brown bread and an iron spoon. When the time came to collect the containers...

> In the morning our breakfasts were passed in to us through the hole in the door in small square tin pans, made to fit, and holding a pint of chocolate, with brown bread, and an iron spoon...

The input is longer, smoother and vaguer ("typically delivered", "specially designed"); the target is shorter and concrete. That gap is what the model learns to close.

### 3.6 Dataset release

The dataset is published as JSONL and CSV with a card describing sources, licence, statistics and the full book list (`dataset/README.md`). Fields: `id`, `input`, `output`, `generator`, `style`, `gutenberg_id`, `title`, `author`, `category`, `input_words`, `output_words`. Licence: CC-BY 4.0.

## 4. Model and training

**Terms used in this section.**

- *Parameter* (or *weight*): one of the numbers inside a model, learned during its original training. Qwen3-4B has about four billion.
- *Fine-tuning*: continuing the training of an existing model on a small, specific dataset so that it picks up one new behaviour.
- *Token*: the unit a model reads and writes, roughly three quarters of an English word.
- *Loss*: a number that says how badly the model predicts the target text. Training is the process of making it smaller.
- *Step, batch, epoch*: one step updates the weights using a batch of examples (16 here). An epoch is one pass through all 2,000 training examples, which is 125 steps.
- *Learning rate*: how big each update is. Too big and training becomes unstable; too small and the model barely changes.
- *Held-out (test) set*: examples kept aside and never trained on, used to check that the model learned the skill rather than memorised the examples.
- *Quantisation*: storing each weight in fewer bits (4 instead of 16) to save memory, at a small cost in precision.

### 4.1 Choosing the base model: why Qwen3-4B

Fine-tuning starts from a model that already writes good English and teaches it one new habit. The first decision is which model to start from. Three questions settled it.

**Why an open model, and not OpenAI's fine-tuning?** OpenAI offers fine-tuning of its own models, and it would have been the quickest way to get a working humanizer. We did not use it, because the result could not be an open release:

- The weights stay on OpenAI's servers. We could not publish them, and nobody could download, inspect or run the model on their own machine.
- Every use is billed per token for as long as the model exists, and only while OpenAI keeps offering that model version.
- Nobody could reproduce the training, because the fine-tuning process itself is not visible.

The point of this project is that a developer can repeat every step and own the result, so the base had to be a model with open weights and a permissive licence. OpenAI's models were still useful as one of the AI-fication generators, where they only produce data.

**Why 4 billion parameters, and not a bigger model?** Memory is the constraint that decides almost everything in fine-tuning, and it grows with the parameter count. Rough memory for the weights alone:

| Model size | 16-bit weights | 4-bit weights | In practice |
|---|---|---|---|
| **4B (ours)** | ~8 GB | ~2.5 GB | trains on one 24 GB GPU in minutes; runs on a laptop |
| 8B | ~16 GB | ~5 GB | trains on 24 GB with QLoRA; about twice the time and serving cost |
| 14B | ~28 GB | ~9 GB | needs a larger GPU to train comfortably |
| 32B | ~64 GB | ~19 GB | needs an 80 GB GPU and hours of training; expensive to serve |
| 70B | ~140 GB | ~40 GB | needs several GPUs |

Training needs more than the weights (the adapter, the optimizer state, and the intermediate values of each batch); our QLoRA run peaked at 8.6 GB on a 24 GB card (section 4.5). Three reasons made the smallest sensible size the right one:

1. **The task does not need knowledge.** A humanizer does not have to know facts: everything it needs is in the input text. It needs fluent English and the ability to follow a style. Larger models mostly add knowledge and reasoning, which this task does not use, and section 5 shows that a 4B model learns the style shift well.
2. **Anyone can repeat it.** A 4B model trains in about 22 minutes for about a dollar. A 32B model would take several hours on an 80 GB GPU and cost tens of dollars per attempt, and every failed attempt (section 4.4) would cost the same again.
3. **Anyone can run it.** The 4-bit GGUF build is about 2.5 GB and runs in Ollama or LM Studio on an ordinary laptop, and the hosted demo needs only a mid-range GPU.

For a production system where quality matters more than cost, a larger base model would be worth testing. That is outside the purpose of this release.

**Why Qwen3-4B among the small models?** Several open models exist around this size, for example Llama 3.2 3B, Gemma 3 4B and Phi-4-mini. We chose Qwen3-4B because of:

- **Licence.** Apache-2.0, a standard permissive licence, so the fine-tuned model can be released under Apache-2.0 as well. Llama and Gemma come with their own custom licence terms that a derived model has to carry.
- **Quality for its size.** At release it was among the strongest open models in its size class, and its English is fluent.
- **Tooling.** First-class support in Unsloth (fast, memory-efficient training), vLLM (serving) and llama.cpp (GGUF builds for laptops), all of which we use.
- **Switchable "thinking".** Qwen3 can reason step by step before answering. For a rewrite that only adds delay and cost, and the chat template lets us switch it off cleanly, which we do.

### 4.2 QLoRA with Unsloth

Rather than updating all four billion weights, which would need far more GPU memory, we used **QLoRA**: the base model is loaded quantised to 4 bits and kept frozen, and a small set of trainable low-rank matrices (a **LoRA** adapter, rank 16, alpha 32) is added to every attention and MLP projection. Only the adapter is trained, then merged back into a full-precision copy of the base for serving. In our case the adapter is about 33 million parameters, under 1% of the model. The idea behind LoRA is that adapting a model to a new style needs a small, low-dimensional change, not a new model; section 5.4 tests that idea by training all the weights instead. **Unsloth** is a library that implements this efficiently; it wraps Hugging Face's `transformers` and the **TRL** `SFTTrainer`.

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
| Result | training loss 1.30 averaged over the run (about 1.13 in the last steps), evaluation loss 1.39 (3.17 before training) |
| Tracking | Weights & Biases run `95wi8tdg` in project `gohumanize/gohumanize-open-humanizer` |

The evaluation loss fell from 3.17 (base model, measured on the same pairs before training) to 1.43 after the first 50 steps and 1.39 at the end; most of the learning happens early, which is common for style tasks with a strong base model. The training script is `train/modal_train.py`.

This section describes the QLoRA run, which we built first and which is the cheaper recipe to follow. The model we publish is the full fine-tune of the same data, described and compared in section 5.4; the two score the same.

### 4.4 A note on getting the environment right

The first attempt at training failed four times before a single step ran, every time on library plumbing rather than on the data or the model: a missing local package, a file read that only works on the developer's machine, and two variants of a version clash between TRL and Unsloth (the fix was simply to import Unsloth before TRL so that its patches apply). We mention this because it is the normal experience with fast-moving ML libraries, and it is why the script supports a 3-step smoke run: spend a minute of GPU time to validate the setup before spending an hour.

### 4.5 What we watched in Weights & Biases

Weights & Biases (W&B) is a dashboard that records a training run while it happens. The training script needs only `report_to="wandb"` and an API key. Every 10 steps it logs the training numbers, every 50 steps it scores the 200 held-out pairs, and in the background it samples the state of the GPU. All our runs are public in the [W&B project](https://wandb.ai/gohumanize/gohumanize-open-humanizer), and each run also leaves a JSON summary in `train/runs/`. This is what each chart shows, what we saw in the QLoRA run, and what would have been a warning sign:

| Chart | What it measures | What we saw | Warning sign |
|---|---|---|---|
| `train/loss` | how badly the model predicts the human text on the batches it trains on (lower is better) | 2.43 at step 10, 1.38 by step 20, then a slow slide to about 1.13 at step 250 | not falling (learning rate too low, or broken data); sudden spikes or `NaN` (learning rate too high) |
| `eval/loss` | the same measure on the 200 held-out pairs, which the model never trains on | 3.17 before training, then 1.43, 1.40, 1.39, 1.40, 1.39 at steps 50 to 250 | rising while `train/loss` keeps falling: overfitting, i.e. memorising instead of learning |
| `train/grad_norm` | the size of each update | 0.84 at the start, then settling between 0.3 and 0.6 | large spikes: unstable training |
| `train/learning_rate` | the step size in use | up to 2e-4 over 10 warm-up steps, then a cosine curve down to zero at step 250 | a shape that does not match the configuration |
| `train/epoch` | passes through the training set | 0 to 2 | |
| GPU utilisation | how busy the GPU is | 93% on average | low values: the GPU is waiting for data and you pay for idle time |
| GPU memory allocated | memory in use | peak 8.6 GB of 24 GB | near 100%: risk of an out-of-memory crash; far below: room for a bigger batch |
| GPU temperature and power | hardware health | normal | throttling, which slows training |

How we read them together:

- **Most of the learning happens in the first 50 steps.** Held-out loss fell from 3.17 to 1.43 in the first fifth of training, and only to 1.39 over the remaining 200 steps. That is typical when a strong base model learns a style: it already knows English and only has to learn which way to rewrite.
- **No overfitting.** Training loss ended near 1.13 and held-out loss at 1.39. A gap is normal, since the model has seen the training passages. What matters is that the held-out loss stayed flat through the second epoch instead of climbing back up. That is also why we stopped at two epochs: a third would cost more and risk memorisation without improving the held-out loss.
- **What a loss of 1.39 means.** The loss is the average of the natural log of how surprised the model is by each next token of the human original. e^1.39 is about 4, so after training the model is, on average, about as unsure as someone choosing between four equally likely next words; before training it was about 24 (e^3.17). It can never reach zero, because the same idea can be phrased in many valid ways and the model cannot know which one the author picked.
- **Comparing runs.** Because every run logs the same charts, the full fine-tune runs (section 5.4) can be laid over the QLoRA run on one plot. Their update sizes are larger (a gradient norm of about 3.5), which is expected when every weight moves, and they used 31 GB of the H100's 80 GB.
- **The smoke run.** Before the real run we trained for 3 steps (`--max-steps 3`). It costs a minute, and in W&B it shows the starting held-out loss (3.17) and proves that logging, the GPU and the data all work.

## 5. Evaluation

### 5.1 Method

We compare the fine-tuned model with the untouched base model on the 200 held-out pairs (`eval/modal_eval.py`). Both receive the same system prompt and the same AI-styled input and generate a rewrite (temperature 0.7, top-p 0.9). We then measure two things against the human original:

- **Faithfulness**: BERTScore F1 and ROUGE-L between the output and the human original (does the meaning and wording survive?), and the share of capitalised tokens from the original (names, places) that appear in the output.
- **Style**: how the output's surface statistics compare with the human target: length ratio, contractions per 100 words, transition words per 100 words, count of stock LLM phrases ("it is worth noting", "delve", "testament to" and similar), and average sentence length.

We also report the same statistics for the AI-styled input itself and for the human target, which give the two ends of the scale. These numbers describe how far an output has moved from LLM prose towards the human original. They are not detector scores.

### 5.2 Results

| Measure (200 held-out pairs) | AI-styled input | Base Qwen3-4B | Open Humanizer (QLoRA) | Human target |
|---|---|---|---|---|
| BERTScore F1 vs human (higher = closer meaning) | 0.914 | 0.900 | 0.921 |  |
| ROUGE-L vs human (higher = closer wording) | 0.493 | 0.424 | 0.540 |  |
| Names/capitalised tokens kept (recall) | 0.587 | 0.594 | 0.623 |  |
| Length ratio vs human | 1.027 | 0.830 | 0.928 | 1.000 |
| Contractions per 100 words | 0.279 | 0.772 | 0.044 | 0.118 |
| Transition words per 100 words | 0.757 | 0.023 | 0.106 | 0.143 |
| Stock LLM phrases per text | 0.19 | 0.01 | 0.00 | 0.00 |
| Average sentence length (words) | 21.6 | 15.2 | 22.5 | 28.1 |

How to read this. The first column is the AI-styled input itself, i.e. what a model would score by changing nothing. The base model, given the same instruction, moves *away* from the human original: meaning and wording drift (BERTScore 0.900 vs 0.915 for the untouched input, ROUGE-L 0.424 vs 0.493), it shortens the text by 17%, and it over-corrects on style, adding contractions at six times the rate of the human authors and cutting sentences to 15 words when the originals average 28. The fine-tuned model moves *towards* the original on every measure: highest meaning and wording overlap of the three (0.921 / 0.540), more names kept (0.623), length within 7% of the human text, transition words and stock phrases brought down to the human level (0.11 per 100 words vs 0.14; zero stock phrases), and sentence length moving in the right direction (22.5 words). Its one clear deviation from the human target is that it uses fewer contractions than the authors did; the training data is dominated by 19th-century prose, where contractions are rare in narration.

In short, 22 minutes of fine-tuning turned a general model that "simplifies" into one that reconstructs the register of the source authors while keeping the content. The measures say nothing about AI detectors; they describe distance from LLM prose and closeness to the human original.

### 5.3 Examples

Held-out pairs, all generated at temperature 0.7. The input is the AI-styled text; the target is the sentence a person originally wrote.

**Swift, *A Modest Proposal* (1729)**

- *Input:* "I am assured by our merchants that a child, whether male or female, under the age of twelve years is not a marketable commodity. Furthermore, even upon reaching this age, such individuals will not fetch more than three pounds, or at most three pounds and half a crown in the marketplace. This situation proves disadvantageous for both the parents and the kingdom, as the expenses of sustenance and clothing have amounted to at least four times that value. In essence, I shall now humbly present my own thoughts on the matter, which I trust will not encounter any objections."
- *Base model:* "Our merchants say a child, boy or girl, under twelve is not something you can sell. Even when they reach twelve, they won't be worth more than three pounds, maybe three pounds and a half crown. This is bad for the parents and the country, because the cost of food and clothes has already been four times that amount. Now I will share my own thoughts on this, and I hope they won't be objected to."
- *Open Humanizer:* "I am assured by our merchants, that a child, male or female, under twelve years old, is not a commodity to be sold in the market, and that when they come to that age, they will not fetch more than three pounds, or three pounds and a half crown in the market; which is a great loss to both parents and the kingdom, for the charges of keeping them in food and raiment have been at least four times that value. I shall now humbly offer my own thoughts upon the matter, which I hope will not be objected to."
- *Human original:* "I am assured by our merchants, that a boy or a girl, before twelve years old, is no saleable commodity, and even when they come to this age, they will not yield above three pounds, or three pounds and half a crown at most, on the exchange; which cannot turn to account either to the parents or kingdom, the charge of nutriments and rags having been at least four times that value. I shall now therefore humbly propose my own thoughts, which I hope will not be liable to the least objection."

**Burnett, *The Secret Garden* (1911)**

- *Input:* "He very seldom engaged in conversation and at times did not even respond to Mary's inquiries, except with a grunt; however, this morning he spoke more than usual. He stood up and placed one hobnailed boot upon the top of his spade while he scanned her appearance. Moreover, it is worth noting that Mary was not vain, and since she had never given much consideration to her looks, she was not significantly disturbed by the observation."
- *Base model:* "He rarely talked and sometimes didn't answer Mary's questions, just grunting. But this morning he spoke more. He stood up, placed one hobnailed boot on top of his spade, and looked at her. Mary wasn't vain. She never thought much about her looks, so she wasn't too bothered by the look."
- *Open Humanizer:* "He seldom spoke, and sometimes did not answer Mary's questions, except with a grunt; but to-day he spoke more than he had spoken for a long time. He got up, and putting one hobnailed boot on the top of his spade, looked at her face. Mary was not vain, and as she had never thought much about her looks, she was not much hurt by the remark."
- *Human original:* "He very seldom talked much and sometimes did not even answer Mary's questions except by a grunt, but this morning he said more than usual. He stood up and rested one hobnailed boot on the top of his spade while he looked her over. Mary was not vain and as she had never thought much of her looks she was not greatly disturbed."

The pattern repeats across the test set: the base model produces clean, short, modern sentences that read like a summary; the fine-tuned model restores the sentence shapes, connectives and vocabulary of the period, occasionally too eagerly ("raiment", "to-day"), which is the flip side of training on old books.


### 5.4 QLoRA versus full fine-tuning

Is a full fine-tune worth it here? We trained the same model a second way, updating
all four billion weights in bf16 instead of a rank-16 adapter on a 4-bit model, with
the same data, two epochs and effective batch of 16. Full fine-tuning needs roughly
three times the GPU memory, so it ran on an H100 (80 GB) rather than the A10G, with
8-bit AdamW and gradient checkpointing (`train/modal_train.py --method full`).

The learning rate is the one setting that has to change: a full fine-tune moves every
weight, so it wants something like one tenth to one twentieth of the LoRA rate. At
1e-5 the model was still improving when training ended (held-out loss 1.410, worse
than QLoRA's 1.391); at 2e-5 it levelled off at 1.381, the best of the three.

That run was then evaluated exactly like the QLoRA model, on the same 200 held-out
pairs, with the base model regenerated under the same seed (it scored 0.8998 against
0.8997 the first time, so the two evaluations are comparable):

| Measure | Base Qwen3-4B | QLoRA | Full fine-tune | Human |
|---|---|---|---|---|
| BERTScore F1 vs human | 0.900 | **0.921** | 0.920 | |
| ROUGE-L vs human | 0.424 | **0.540** | 0.535 | |
| Names kept (recall) | 0.594 | **0.623** | 0.614 | |
| Length ratio vs human | 0.83 | 0.93 | **0.95** | 1.00 |
| Stock LLM phrases / text | 0.01 | **0.00** | **0.00** | 0.00 |
| Average sentence length | 15.2 | 22.5 | 22.6 | 28.1 |

The two are indistinguishable. The differences are of the order of 0.001 in BERTScore
and 0.005 in ROUGE-L, in both directions, which on 200 pairs is noise; the full
fine-tune's slightly lower held-out loss did not become better rewrites. This is the
usual outcome for style transfer with a small dataset: 2,000 pairs are not enough
signal to need more than a low-rank update, and the adapter captures what there is
to learn.

**Which one we publish.** Both. The full fine-tune is the main published model
([gohumanize/gohumanize-open-humanizer](https://huggingface.co/gohumanize/gohumanize-open-humanizer)):
it reached the lowest held-out loss, and updating every weight is the most direct form
of fine-tuning, so it is the natural reference model. The QLoRA version is published
next to it ([gohumanize/gohumanize-open-humanizer-qlora](https://huggingface.co/gohumanize/gohumanize-open-humanizer-qlora)),
with its 66 MB adapter, because it is the cheaper recipe to reproduce: the same quality
on a 24 GB GPU for about a dollar, where the full fine-tune needs an 80 GB one.

**Which one to use for your own project.** Start with QLoRA: it is cheaper, faster to
iterate on, and here it gave the same result. A full fine-tune becomes worth its cost
with much more data, or when the task needs the model to learn new knowledge rather
than a new register. The run records are in `train/runs/`
(`open-humanizer-full-lr1e5.json`, `open-humanizer-full-lr2e5.json`) and the
evaluation in `eval/results/open-humanizer-full-lr2e5.json`.

## 6. Serving and integration

- **Weights** are on Hugging Face: the full fine-tune as 16-bit safetensors plus GGUF files (Q4_K_M and Q8_0) built with llama.cpp, so the model runs locally in Ollama, LM Studio or llama.cpp. The QLoRA version, with its adapter and its own GGUF files, is in a second repository.
- **Endpoint model**: the hosted demo serves the full fine-tune.
- **Endpoint**: `serve/modal_serve.py` runs vLLM on Modal behind an OpenAI-compatible API (`/v1/chat/completions`). The container scales to zero when idle, so the demo costs nothing while unused and roughly one A10G-hour per hour of use.
- **MCP server** (`npx gohumanize-open-humanizer-mcp`) exposes a `humanize_text` tool to AI assistants. It calls the endpoint above by default, or any OpenAI-compatible server you point it at, including a local Ollama running the GGUF.
- **Python client** (`pip install gohumanize-open-humanizer`) with an `open-humanizer` command, same options.
- **Browser demo** on the project page (gohumanize.ai/open-model): a small form that calls the endpoint through the site's own server route, so the key stays server-side. A Gradio app for a Hugging Face Space is included in `demo/` for anyone who wants to host their own copy.

## 7. Services used, and why

| Service | Role | Why this one |
|---|---|---|
| Project Gutenberg | Human text | Large, free, clearly public domain, direct downloads. |
| OpenAI gpt-4o-mini, Meta Llama 3.3 70B, DeepSeek-chat | AI-fication | Three model families from three labs, so the humanizer learns LLM style in general rather than one model's habits; all cheap at this volume (about $1 in total). |
| Qwen3-4B (Qwen team, Alibaba) | Base model | Open weights under Apache-2.0, strong for its size, small enough to train for a dollar and run on a laptop (section 4.1). |
| Unsloth, TRL, vLLM | Training and serving libraries | Unsloth makes QLoRA fast and memory-efficient; TRL provides the supervised fine-tuning loop; vLLM serves the model behind an OpenAI-compatible API. |
| Modal | GPU for training, evaluation and serving | Pay-per-second GPUs from a Python script, no servers to manage, scale-to-zero serving. |
| Weights & Biases | Experiment tracking | Every run's settings, loss curves and GPU usage are recorded and public, so any number in this document can be checked (section 4.5). |
| Hugging Face | Hosting weights and dataset | The standard place developers look for open models; free hosting for public repositories. |
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
modal run train/modal_train.py --run-name open-humanizer-v1                    # QLoRA, A10G, ~25 min, ~$1
modal run train/modal_train.py --method full --learning-rate 2e-5 --run-name open-humanizer-full-lr2e5   # full, H100
modal run eval/modal_eval.py --run-name open-humanizer-full-lr2e5
modal run train/push_to_hub.py::push --run-name open-humanizer-full-lr2e5 --repo <org>/<model>
modal run train/push_to_hub.py::gguf --run-name open-humanizer-full-lr2e5 --repo <org>/<model>
modal deploy serve/modal_serve.py
```

Total cost of one full reproduction: under $5. Total time: about two hours including waiting.

## Links

| Resource | Link |
| --- | --- |
| Project page and browser demo | [gohumanize.ai/open-model](https://gohumanize.ai/open-model) |
| Model weights and GGUF builds (full fine-tune) | [gohumanize/gohumanize-open-humanizer](https://huggingface.co/gohumanize/gohumanize-open-humanizer) |
| QLoRA version and LoRA adapter | [gohumanize/gohumanize-open-humanizer-qlora](https://huggingface.co/gohumanize/gohumanize-open-humanizer-qlora) |
| Dataset, 2,200 pairs (CC-BY 4.0) | [gohumanize/gohumanize-open-humanizer-dataset](https://huggingface.co/datasets/gohumanize/gohumanize-open-humanizer-dataset) |
| Code and full pipeline | [GoHumanize-ai/gohumanize-open-humanizer](https://github.com/GoHumanize-ai/gohumanize-open-humanizer) |
| Write-up: every step, service and result | [docs/paper.md](https://github.com/GoHumanize-ai/gohumanize-open-humanizer/blob/main/docs/paper.md) |
| Archived release, citable DOI | [10.5281/zenodo.22843083](https://doi.org/10.5281/zenodo.22843083) |
| Python client and CLI | [pypi.org/project/gohumanize-open-humanizer](https://pypi.org/project/gohumanize-open-humanizer/) |
| MCP server for AI assistants | [npm](https://www.npmjs.com/package/gohumanize-open-humanizer-mcp) · [source](https://github.com/GoHumanize-ai/gohumanize-open-humanizer-mcp) |
| Training runs, loss curves and config | [Weights & Biases](https://wandb.ai/gohumanize/gohumanize-open-humanizer) (full fine-tune `khrhh8sl`, QLoRA `95wi8tdg`) |

## Citation

GoHumanize team (2026). *GoHumanize Open Humanizer: an open text-humanization model, dataset and pipeline built from public-domain data.* Version 0.1.0. Zenodo. https://doi.org/10.5281/zenodo.22843083

Licences: model and code Apache-2.0; dataset CC-BY 4.0.
