#!/usr/bin/env bash
# Dev-only "brain" for macOS: llama.cpp llama-server (Metal) serving Qwen3-VL-2B on
# http://127.0.0.1:8080/v1, the same OpenAI-compatible API that `geniex serve` exposes on
# the Snapdragon NPU. Downloads llama.cpp + the model into ~/.cache/netra on first run.
set -euo pipefail

CACHE="${NETRA_CACHE:-$HOME/.cache/netra}"
LLAMA_BUILD="${LLAMA_BUILD:-b11158}"
MODEL_REPO="https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main"
MODEL="$CACHE/models/Qwen3VL-2B-Instruct-Q4_K_M.gguf"
MMPROJ="$CACHE/models/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
mkdir -p "$CACHE/bin/llama" "$CACHE/models"

SERVER="$(find "$CACHE/bin/llama" -name llama-server -type f 2>/dev/null | head -1)"
if [[ -z "$SERVER" ]]; then
  echo "Downloading llama.cpp $LLAMA_BUILD (macOS arm64)..."
  curl -fL "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_BUILD/llama-$LLAMA_BUILD-bin-macos-arm64.tar.gz" \
    | tar -xz -C "$CACHE/bin/llama"
  SERVER="$(find "$CACHE/bin/llama" -name llama-server -type f | head -1)"
fi
[[ -s "$MODEL" ]] || curl -fL -C - -o "$MODEL" "$MODEL_REPO/$(basename "$MODEL")"
[[ -s "$MMPROJ" ]] || curl -fL -C - -o "$MMPROJ" "$MODEL_REPO/$(basename "$MMPROJ")"

exec "$SERVER" -m "$MODEL" --mmproj "$MMPROJ" --alias qwen3-vl-2b \
  --host 127.0.0.1 --port 8080 -c 8192 -ngl 99 --jinja "$@"
