#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

ARCH=$(uname -m)

echo "=== LLM Inference Environment Setup ==="
echo "Architecture: ${ARCH}"

if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
fi

if ! command -v uv &>/dev/null; then
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

echo ">>> Ensuring Python is available..."
uv python install

if [ "${ARCH}" = "aarch64" ]; then
    echo ">>> aarch64 detected — using system PyTorch, installing other deps..."
    uv sync --no-install-package torch
else
    echo ">>> Installing dependencies (uv sync)..."
    uv sync
fi

echo ""
echo "=== Verification ==="
uv run python -c "
import torch
print(f'PyTorch:       {torch.__version__}')
print(f'CUDA:          {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:           {torch.cuda.get_device_name(0)}')
    p = torch.cuda.get_device_properties(0)
    mem = getattr(p, 'total_memory', getattr(p, 'total_mem', 0)) / 1e9
    print(f'GPU memory:    {mem:.1f} GB')

import transformers
print(f'Transformers:  {transformers.__version__}')
print()
print('Setup complete!')
"

echo ""
echo "=== Done ==="
echo "Run with:  uv run python 01_hf_pipeline.py"
echo "Or activate:  source .venv/bin/activate"
