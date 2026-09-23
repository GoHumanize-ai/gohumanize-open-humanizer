# Adding modern human prose did not stop the model copying modern input

Status: resolved. The first attempt failed and the reasons are kept below, because each wrong
turn ruled something out. The fix is in the final section.

## The problem

On modern prose the published model often returns its input almost unchanged. Measured on 100
held-out modern passages through the serving stack, one sample per passage:

    near-copies (word overlap with the input above 0.9): 57 of 100
    of those, genuinely unchanged (sequence similarity above 0.95): 45
    byte-identical to the input: 20

On the literary held-out set the same model copies 19 of 200. So the behaviour is specific to
modern text, not general.

## The hypothesis

The training pairs came only from pre-1929 books, so the model had never seen contemporary
writing. Give it modern prose and it should learn to rewrite modern prose.

## What was tried

Works of the US federal government carry no copyright (17 U.S.C. 105), so agency writing is the
one large source of modern prose that can be redistributed. `pipeline/01b_source_gov.py`
collected 3,118 passages from NASA, the Federal Reserve, the Department of Energy and NIST.
Capped per source so no single agency dominated, 999 train and 100 test pairs were built with the
same AI-fication step as the books, giving a 2,999-pair dataset that is one third modern.

A full fine-tune with the published recipe (Qwen3-4B, 2 epochs, lr 2e-5) reached a lower loss
than the published model: train 1.023 against 1.271, eval 1.172 against 1.381.

## Result

Both models scored on the same held-out sets, same engine, same sampling, one sample per passage:

| model | modern near-copies | literary near-copies |
| --- | --- | --- |
| published (books only) | 57% | 19.0% |
| retrained (books + modern) | 55% | 13.5% |

The literary set improved. Modern did not move.

## Why

Measure how large a rewrite each pair actually teaches, as word overlap between the AI-styled
input and the human target. Lower means a bigger transformation:

| source | pairs | input/output overlap |
| --- | --- | --- |
| Project Gutenberg books | 2,000 | 0.54 |
| NIST | 407 | 0.68 |
| NASA | 234 | 0.68 |
| Federal Reserve | 347 | 0.67 |
| Department of Energy | 11 | 0.64 |

And the model's copy rate on the test set tracks it directly:

    test rows whose human target differs a lot (target overlap below 0.6):  40% copied
    test rows whose human target is already close (target overlap 0.7 up):  77% copied

Federal prose is already written in the register the AI-fication step produces: formal, hedged,
transition-heavy, even sentence lengths. AI-fying it barely changes it, so those pairs do not
teach the model to rewrite modern text. They teach it that modern text needs almost no change.
Adding more of them cannot fix the copying, and mildly reinforces it.

This also explains why pre-1929 books work so well: their prose is stylistically far from LLM
output, so every pair teaches a large transformation.

## What to try next

The dataset needs modern human prose in a plainer, less institutional voice, and pairs that
teach a real transformation.

1. A plainer federal source. Voice of America is US federal, public domain, and writes ordinary
   journalistic prose rather than press-release prose. NASA blogs are another.
2. A harder AI-fication prompt for source text that is already formal, so the gap between input
   and target is real rather than nominal.
3. Drop pairs whose input and target overlap above about 0.6, whatever the source: a pair that
   teaches no transformation is worse than no pair.

Until then, the demo and both clients generate several rewrites and return the one that moved
furthest from the input, which takes the user-visible copy rate to near zero. That is a
mitigation, not a fix: the underlying per-sample rate is unchanged.

## Update: what actually fixed it

Two further findings, each measured before acting on it.

**The source was never the problem.** Voice of America, plain journalistic prose and nothing like a
press release, measured the same 0.68 input-to-target overlap as agency text. So the register of the
human side does not matter here.

**The AI-fication instruction was.** It asked for every fact kept, the order kept and the length within
20 percent. On a modern factual passage that leaves almost nothing to change, so the machine version came
back nearly identical to the original. A hard register that keeps every fact, name, number and quotation
but forbids keeping the sentence structure took the mean overlap on the same federal passages from 0.63
to 0.49, below the books' 0.54 (`03_aify.py --register hard`).

**Then a data review found the model inventing things.** Anything the human target carries and the
machine input lacks becomes something the model learns to add. 190 Federal Reserve targets had footnote
numbers glued to sentence ends and 39 had press-release datelines; the retrained model began inventing
both. The build now strips them, drops media notices, and refuses to write a dataset where a target still
carries one; CI checks the released dataset the same way. The review also found 70 of 96 modern test
articles had sibling passages in training, so test articles are now held out whole.

## Final result

Article-held-out test set (100 modern passages from 59 articles never seen in training, 200 literary),
three samples per passage, two training seeds:

| | published | clean, seed 13 | clean, seed 29 |
| --- | --- | --- | --- |
| modern near-copies | 37.7% | 17.0% | 16.3% |
| literary near-copies | 16.7% | 13.3% | 14.5% |
| invented datelines (of 900) | 0 | 0 | 0 |
| invented footnote numbers (of 900) | 0 | 0 | 0 |
| invented links (of 900) | 0 | 0 | 1 |
| real rewrites that drop a number | 31.9% | 28.5% | 28.5% |

The last row counts only outputs that actually rewrite the passage. Counting copies too makes the published
model look better at keeping numbers (17.1% against 23.0%), but only because a copy keeps every number
trivially and the published model copies twice as often.

Loss told us nothing throughout: runs with the same loss differed by twelve points in copy rate.

## Known limitation, all versions

About three in ten real rewrites of number-heavy text drop at least one number, for the published model
and the retrain alike. Some of that is rewording ("20 percent" to "a fifth"), some is loss. The best-of-N
selector in the clients could prefer candidates that keep every number of the input; not done yet.

## QLoRA on the same data

The version 1 QLoRA recipe (rank 16, 4-bit base, lr 2e-4, 2 epochs) trained on dataset version 2 at the same
two seeds, scored on the same test in the same way:

| | QLoRA seed 13 | QLoRA seed 29 | full seed 13 (published) | full seed 29 |
| --- | --- | --- | --- | --- |
| modern near-copies | 7.0% | 5.7% | 17.0% | 16.3% |
| literary near-copies | 12.5% | 12.2% | 13.3% | 14.5% |
| real rewrites that drop a number | 26.8% | 26.2% | 28.5% | 28.5% |
| inventions (of 900) | 1 | 1 | 0 | 1 |
| held-out loss | 1.195 | 1.195 | 1.188 | |

QLoRA copies well under half as often at both seeds. Version 1 showed the same gap (QLoRA 20%, full 38%),
which the standard evaluation could not see. The one QLoRA invention is the same Federal Reserve passage at both
seeds (a footnote-style "12" glued to a sentence end). Full results: `eval/results/copy-rate/v1-v2-qlora-full.json`.
The full fine-tune stays the published model for version 2; the paper, section 6.9, records the comparison.
