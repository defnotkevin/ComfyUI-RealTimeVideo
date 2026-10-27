#!/usr/bin/env bash
set -euo pipefail
# Use the runtime's normal Hub downloader, not a host-only optional accelerator.
export HF_HUB_ENABLE_HF_TRANSFER=0
NODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${KREA_INSTALL_PYTHON:-python3.11}"
RUNTIME_DIR="${KREA_VENV:-$NODE_ROOT/.runtime}"
MODEL_DIR="${KREA_MODELS:-$NODE_ROOT/models}"
"$PYTHON_BIN" -c 'import sys,platform; assert platform.system()=="Linux", "Install this runtime on the Linux NVIDIA GPU host"; assert sys.version_info[:2] == (3,11), "Use Python 3.11"'
"$PYTHON_BIN" -m venv "$RUNTIME_DIR"
"$RUNTIME_DIR/bin/python" -m pip install --upgrade 'pip==25.3'
"$RUNTIME_DIR/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.8.0' 'torchvision==0.23.0'
"$RUNTIME_DIR/bin/python" -m pip install -r "$NODE_ROOT/requirements-worker.txt"
"$RUNTIME_DIR/bin/python" "$NODE_ROOT/scripts/download_models.py" --directory "$MODEL_DIR" --runtime-config "$NODE_ROOT/runtime.json"
HF_HUB_OFFLINE=1 "$RUNTIME_DIR/bin/python" "$NODE_ROOT/scripts/check_model_api.py" --model "$MODEL_DIR/krea"
"$RUNTIME_DIR/bin/python" -m pip check
"$RUNTIME_DIR/bin/python" -m pip freeze > "$NODE_ROOT/runtime-freeze.txt"
printf '\nRuntime installed. Restart ComfyUI, import workflows/01-live-camera.json, and open Live Studio.\n'
