"""Download a model's safetensors weights + tokenizer from Hugging Face into ./models/<name>.

    python scripts/download_model.py                      # Qwen/Qwen2.5-0.5B-Instruct (~1 GB)
    python scripts/download_model.py --repo Qwen/Qwen3-0.6B

If huggingface.co is slow or blocked, set a mirror first, e.g.
    set HF_ENDPOINT=https://hf-mirror.com        (Windows cmd)
    $env:HF_ENDPOINT="https://hf-mirror.com"     (Windows PowerShell)
"""

import argparse
import os

from huggingface_hub import snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dir", default=None, help="target folder (default: models/<repo name>)")
    args = ap.parse_args()
    target = args.dir or os.path.join(ROOT, "models", args.repo.split("/")[-1])
    path = snapshot_download(args.repo, local_dir=target,
                             allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json", "*.txt"])
    print(f"downloaded to {path}")


if __name__ == "__main__":
    main()
