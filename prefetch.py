#!/usr/bin/env python3
"""Pre-download HuggingFace models before running inference.

Self-bootstrapping: creates a temporary venv and installs dependencies
automatically. Just run with any system python3.

Usage:
    python3 prefetch.py
    python3 prefetch.py --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import os
import subprocess
import sys

VENV_DIR = "/tmp/hf-prefetch-venv"
REQUIRED_PACKAGES = ["transformers", "huggingface_hub"]


def ensure_venv():
    venv_python = os.path.join(VENV_DIR, "bin", "python")
    if sys.executable == venv_python or sys.prefix != sys.base_prefix:
        return
    if not os.path.exists(venv_python):
        print(f">>> Setting up venv at {VENV_DIR}...")
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
        subprocess.check_call(
            [os.path.join(VENV_DIR, "bin", "pip"), "install", "-q"] + REQUIRED_PACKAGES
        )
        print()
    os.execv(venv_python, [venv_python] + sys.argv)


def main():
    parser = argparse.ArgumentParser(description="Pre-download HF models")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model to download (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--cache-dir",
        default=".hf_cache",
        help="Local cache directory (default: .hf_cache)",
    )
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)
    os.environ["HF_HOME"] = cache_dir
    print(f"Cache directory: {cache_dir}")
    print()

    print(f">>> Downloading tokenizer: {args.model}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"    OK — vocab size: {tokenizer.vocab_size}")

    print(f"\n>>> Downloading model weights: {args.model}")
    from huggingface_hub import snapshot_download
    path = snapshot_download(args.model)
    print(f"    OK — cached at: {path}")

    print(f"\n{'='*60}")
    print("All assets cached. Run the container with:")
    print("  docker run -it --gpus all --ipc=host \\")
    print("    -v $(pwd):/workspace -w /workspace \\")
    print(f"    -v $(pwd)/{args.cache_dir}:/root/.cache/huggingface \\")
    print("    llm-inf")
    print(f"{'='*60}")


if __name__ == "__main__":
    ensure_venv()
    main()
