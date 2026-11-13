#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
HF=${HF:-$(command -v hf)}

mkdir -p "$ROOT/models" "$ROOT/data"

"$HF" download Qwen/Qwen3.5-4B \
  --local-dir "$ROOT/models/Qwen3.5-4B"
"$HF" download Qwen/Qwen3-8B \
  --local-dir "$ROOT/models/Qwen3-8B"
"$HF" download haowu89/tmax-rl-800 \
  --repo-type dataset \
  --local-dir "$ROOT/data/tmax-rl-800"

echo "Assets downloaded under $ROOT/models and $ROOT/data"
