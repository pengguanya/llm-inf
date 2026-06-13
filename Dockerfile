FROM nvcr.io/nvidia/pytorch:25.12-py3

WORKDIR /workspace

COPY pyproject.toml ./
RUN pip install --no-cache-dir transformers accelerate requests openai && \
    pip install --no-cache-dir vllm || true && \
    pip uninstall -y torchao || true

COPY 0*.py prefetch.py ./

CMD ["bash"]
