#!/usr/bin/env bash
# Run from a NAS checkout so environments, downloads, and compiler caches stay there.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/diskstation/uv-cache}"
export UV_PYTHON_INSTALL_DIR="$PWD/.runtime/python"
export UV_LINK_MODE=hardlink
export TMPDIR="$PWD/.runtime/tmp"
export XDG_CACHE_HOME="$PWD/.runtime/cache"
export GS_CACHE_FILE_PATH="$XDG_CACHE_HOME/genesis"
export QD_OFFLINE_CACHE_FILE_PATH="$XDG_CACHE_HOME/quadrants"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU:-1}"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$QD_OFFLINE_CACHE_FILE_PATH" "$CUDA_CACHE_PATH"
uv sync --extra cuda
exec uv run --no-sync --extra cuda python "$@"
