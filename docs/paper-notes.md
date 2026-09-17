# Notes for the write-up

## Framing agreed with Yaro (2026-09-17)

Use this positioning (agreed with Yaro). Do NOT say "this is how
GoHumanize started" or that it uses the "same approach" as production.

> The GoHumanize Open Humanizer is a public research and educational model created
> to demonstrate the general approach used to develop AI text humanization systems.
> It is separate from the production models used by GoHumanize.ai, but it reflects
> many of the same high-level principles we follow when developing our technology,
> including careful dataset preparation, transformation of source text into training
> pairs, model fine-tuning, evaluation, and iterative improvement.
>
> By publishing the model, dataset, code, methodology, and development process, we
> aim to provide developers and researchers with a practical example of how a
> humanization model can be built and studied. The open model is not intended to
> reproduce the exact architecture, datasets, training configuration, or performance
> of GoHumanize.ai's production systems.

Other constraints from the brief:
- Public-domain data only (Project Gutenberg, English originals, no translations).
- 2,000 training pairs, AI-fied with a mix of providers.
- Base model Qwen3-4B (Apache 2.0).
- No claim that the model passes AI detectors. Goal: show developers how such a
  tool is built so they can reuse the knowledge.
- Explain each service used and why: OpenAI/Gemini (AI-fication), W&B (training
  tracking), RunPod or Modal (GPU), Hugging Face (hosting), Zenodo (DOI), etc.
- Authors: "GoHumanize team". Licences: dataset CC-BY 4.0, model + code Apache 2.0.
- Site page: gohumanize.ai, title "Open Model & Research", linked in the footer.
