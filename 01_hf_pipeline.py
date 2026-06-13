"""01 - HuggingFace Pipeline: the simplest way to run an LLM.

pipeline() handles everything: model loading, tokenization, generation, and decoding, One function call from text in to text out.

Run:
    python 01_hf_pipeline.py

What to observe:
    - How long model loading takes (first run downloads weights) 
    - The generated text quality
    - GPU memory usage (watch with: nvidia-smi in another terminal)
"""

import time
from transformers import pipeline

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    print(f"Loading model: {MODEL}")
    t0 = time.time()

    gen = pipeline("text-generation", model=MODEL)

    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    prompt = "Explain what LLM inference is in two sentences."
    print(f"\nPrompt: {prompt}")

    t0 = time.time()
    result = gen(prompt, max_new_tokens=128)
    gen_time = time.time() - t0

    text = result[0]["generated_text"]
    prompt_tokens = len(gen.tokenizer.encode(prompt))
    total_tokens = len(gen.tokenizer.encode(text))
    new_tokens = total_tokens - prompt_tokens

    print(f"\nResponse:\n{text}")
    print(f"\nStats:")
    print(f"  Generated tokens : {new_tokens}")
    print(f"  Wall time        : {gen_time:.1f}s")
    print(f"  Tokens/sec       : {new_tokens / gen_time:.1f}")

if __name__ == "__main__":
    main()


