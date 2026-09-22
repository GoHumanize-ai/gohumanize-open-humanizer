# Adding modern human prose did not stop the model copying modern input

Status: negative result. Kept because it rules out the obvious explanation and points at the
real one.

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
