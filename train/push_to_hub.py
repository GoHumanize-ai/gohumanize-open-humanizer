"""Publish the trained model to Hugging Face and build a GGUF for local use.

Runs on Modal because the weights live on the Modal volume. Two functions:

  push        uploads the merged 16-bit model (plus the LoRA adapter as a
              subfolder) to <org>/<model-repo> with the model card from
              docs/model-card.md.
  gguf        converts the merged model with llama.cpp to GGUF (Q4_K_M and
              Q8_0) and uploads the files to the same repo, so the model runs
              in Ollama / LM Studio / llama.cpp.

Run:
    modal run train/push_to_hub.py::push --run-name open-humanizer-v1 --repo GoHumanize-ai/gohumanize-open-humanizer
    modal run train/push_to_hub.py::gguf --run-name open-humanizer-v1 --repo GoHumanize-ai/gohumanize-open-humanizer
"""

from __future__ import annotations

from pathlib import Path

import modal

VOLUME_NAME = "gohumanize-open-humanizer-models"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _dotenv(path: Path, keys: tuple[str, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() in keys and v.strip():
                out[k.strip()] = v.strip().strip('"')
    return out


secrets = [modal.Secret.from_dict(_dotenv(REPO_ROOT / ".env", ("HF_TOKEN",)))]
volume = modal.Volume.from_name(VOLUME_NAME)
app = modal.App("gohumanize-open-humanizer-publish")

push_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub")
    .add_local_dir(REPO_ROOT / "docs", remote_path="/docs")
)

gguf_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake")
    .run_commands(
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /llama.cpp",
        "pip install -r /llama.cpp/requirements/requirements-convert_hf_to_gguf.txt",
        "cmake -S /llama.cpp -B /llama.cpp/build -DGGML_NATIVE=OFF && cmake --build /llama.cpp/build --target llama-quantize -j",
    )
    .pip_install("huggingface_hub")
)


@app.function(image=push_image, volumes={"/models": volume}, secrets=secrets, timeout=60 * 60)
def push(run_name: str, repo: str, private: bool = True) -> str:
    import os

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    src = f"/models/{run_name}"
    api.upload_folder(repo_id=repo, folder_path=f"{src}/merged-16bit", path_in_repo=".",
                      commit_message="Upload merged 16-bit weights")
    api.upload_folder(repo_id=repo, folder_path=f"{src}/lora", path_in_repo="lora",
                      commit_message="Upload LoRA adapter")
    api.upload_file(repo_id=repo, path_or_fileobj=f"{src}/train_summary.json",
                    path_in_repo="train_summary.json", commit_message="Add training summary")
    card = Path("/docs/model-card.md")
    if card.exists():
        api.upload_file(repo_id=repo, path_or_fileobj=str(card), path_in_repo="README.md",
                        commit_message="Add model card")
    return f"https://huggingface.co/{repo}"


@app.function(image=gguf_image, volumes={"/models": volume}, secrets=secrets,
              timeout=2 * 60 * 60, cpu=8, memory=32768)
def gguf(run_name: str, repo: str, quants: str = "Q4_K_M,Q8_0") -> list[str]:
    import os
    import subprocess

    from huggingface_hub import HfApi

    src = f"/models/{run_name}/merged-16bit"
    out_dir = f"/models/{run_name}/gguf"
    os.makedirs(out_dir, exist_ok=True)
    f16 = f"{out_dir}/gohumanize-open-humanizer-f16.gguf"
    subprocess.run(["python", "/llama.cpp/convert_hf_to_gguf.py", src, "--outfile", f16,
                    "--outtype", "f16"], check=True)
    uploaded = []
    api = HfApi(token=os.environ["HF_TOKEN"])
    for q in quants.split(","):
        q = q.strip()
        path = f"{out_dir}/gohumanize-open-humanizer-{q}.gguf"
        subprocess.run(["/llama.cpp/build/bin/llama-quantize", f16, path, q], check=True)
        api.upload_file(repo_id=repo, path_or_fileobj=path, path_in_repo=f"gguf/{os.path.basename(path)}",
                        commit_message=f"Add {q} GGUF")
        uploaded.append(path)
    volume.commit()
    return uploaded
