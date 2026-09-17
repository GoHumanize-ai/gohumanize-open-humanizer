"""Fine-tune Qwen3-4B into the GoHumanize Open Humanizer on a rented GPU (Modal).

What happens here, in order:
  1. Modal builds a container image with the training libraries (Unsloth, TRL,
     transformers) and copies dataset/train.jsonl + dataset/test.jsonl into it.
  2. On an A10G (24 GB) GPU the base model Qwen/Qwen3-4B is loaded in 4-bit
     (QLoRA) and a LoRA adapter of rank 16 is trained on the pairs. The prompt
     side (system + AI-styled input) is masked out, so the loss only covers the
     human target text.
  3. Training metrics stream to Weights & Biases. The LoRA adapter and a merged
     16-bit copy of the model are written to a Modal volume; a later step pushes
     them to Hugging Face.

Run from the repo root:
    modal run train/modal_train.py            # train with defaults below
    modal run train/modal_train.py --epochs 3 # override a hyperparameter
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "gohumanize-open-humanizer-train"
BASE_MODEL = "Qwen/Qwen3-4B"
VOLUME_NAME = "gohumanize-open-humanizer-models"
REMOTE_DATA = "/data"
REMOTE_OUT = "/models"

SYSTEM_PROMPT = (
    "Rewrite the following text so that it reads as if a person wrote it: varied sentence "
    "length, concrete wording, natural rhythm, no filler transitions. Keep the meaning, the "
    "facts and the order of ideas. Return only the rewritten text."
)

REPO_ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Unsloth pins the transformers/TRL/PEFT versions it is tested with, so we let it
    # resolve them. The exact versions used are recorded in train_summary.json.
    .pip_install("unsloth", "wandb", "huggingface_hub")
    .add_local_dir(REPO_ROOT / "dataset", remote_path=REMOTE_DATA)
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
def _dotenv(path: Path, keys: tuple[str, ...]) -> dict[str, str]:
    """Read selected KEY=value lines from .env without extra dependencies."""
    out: dict[str, str] = {}
    if not path.exists():  # inside the Modal container the module is re-imported without .env
        return out
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() in keys and v.strip():
                out[k.strip()] = v.strip().strip('"')
    return out


secrets = [modal.Secret.from_dict(_dotenv(REPO_ROOT / ".env", ("HF_TOKEN", "WANDB_API_KEY")))]


def to_messages(row: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]},
        ]
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=4 * 60 * 60,
    volumes={REMOTE_OUT: volume},
    secrets=secrets,
)
def train(
    epochs: int = 2,
    learning_rate: float = 2e-4,
    lora_rank: int = 16,
    max_seq_length: int = 1024,
    batch_size: int = 4,
    grad_accum: int = 4,
    run_name: str = "open-humanizer-v1",
    max_steps: int = -1,
) -> str:
    import json
    import os

    import torch
    import wandb
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    os.environ["WANDB_PROJECT"] = "gohumanize-open-humanizer"
    wandb.login(key=os.environ["WANDB_API_KEY"])

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=13,
    )

    ds = load_dataset("json", data_files={"train": f"{REMOTE_DATA}/train.jsonl",
                                          "test": f"{REMOTE_DATA}/test.jsonl"})
    ds = ds.map(to_messages, remove_columns=ds["train"].column_names)

    def render(batch):
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False,
                                          enable_thinking=False)
            for m in batch["messages"]
        ]
        return {"text": texts}

    ds = ds.map(render, batched=True, remove_columns=["messages"])

    out_dir = f"{REMOTE_OUT}/{run_name}"
    import dataclasses
    cfg_fields = {f.name for f in dataclasses.fields(SFTConfig)}
    seq_kw = {"max_length": max_seq_length} if "max_length" in cfg_fields else {"max_seq_length": max_seq_length}
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        args=SFTConfig(
            **seq_kw,
            max_steps=max_steps,
            output_dir=f"{out_dir}/checkpoints",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            bf16=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="no",
            dataset_text_field="text",
            packing=False,
            report_to="wandb",
            run_name=run_name,
            seed=13,
        ),
    )
    # Qwen3 chat template markers: loss only on the assistant turn.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    result = trainer.train()
    metrics = trainer.evaluate()
    summary = {
        "base_model": BASE_MODEL,
        "epochs": epochs, "learning_rate": learning_rate, "lora_rank": lora_rank,
        "max_seq_length": max_seq_length, "effective_batch": batch_size * grad_accum,
        "train_rows": len(ds["train"]), "test_rows": len(ds["test"]),
        "train_loss": result.training_loss, "eval_loss": metrics.get("eval_loss"),
        "train_runtime_s": result.metrics.get("train_runtime"),
        "gpu": torch.cuda.get_device_name(0),
        "wandb_url": wandb.run.url if wandb.run else None,
        "library_versions": {m: __import__(m).__version__ for m in
                             ("torch", "transformers", "trl", "peft", "unsloth", "bitsandbytes")},
    }

    model.save_pretrained(f"{out_dir}/lora")
    tokenizer.save_pretrained(f"{out_dir}/lora")
    model.save_pretrained_merged(f"{out_dir}/merged-16bit", tokenizer, save_method="merged_16bit")
    with open(f"{out_dir}/train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    volume.commit()
    wandb.finish()
    return json.dumps(summary, indent=2)


@app.local_entrypoint()
def main(epochs: int = 2, learning_rate: float = 2e-4, lora_rank: int = 16,
         run_name: str = "open-humanizer-v1", max_steps: int = -1):
    print(train.remote(epochs=epochs, learning_rate=learning_rate, lora_rank=lora_rank,
                       run_name=run_name, max_steps=max_steps))
