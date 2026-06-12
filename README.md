# LLM Inference Learning Project

Progressive scripts for learning LLM inference with HuggingFace, Ollama, and vLLM.
Runs on NVIDIA DGX Spark (GB10, 128GB unified memory).

## Scripts

| Script | What you learn |
|--------|---------------|
| 01_hf_pipeline.py | HuggingFace pipeline() — simplest inference |
| 02_hf_generate.py | Manual tokenizer + generate() with sampling control |
| 03_ollama_inference.py | Ollama HTTP API — managed runtime |
| 04_vllm_serve.py | vLLM — production serving with OpenAI-compatible API |
| 05_benchmark.py | Compare all three frameworks |
| 06_scaling_limits.py | Push larger models, find DGX Spark limits |

## Quick start (local, CPU)

    bash setup_env.sh
    source .venv/bin/activate
    python 01_hf_pipeline.py

## DGX Spark (GPU)

    python3 prefetch.py
    docker build -t llm-inf .
    docker run -it --gpus all --ipc=host \
      -v $(pwd):/workspace -w /workspace \
      -v $(pwd)/.hf_cache:/root/.cache/huggingface \
      llm-inf
    python 01_hf_pipeline.py
