#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=0
NODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${LTX_INSTALL_PYTHON:-python3.11}"
RUNTIME_DIR="$NODE_ROOT/.runtime-ltx"
"$PYTHON_BIN" -c 'import sys,platform; assert platform.system()=="Linux"; assert sys.version_info[:2]==(3,11), "Use Python 3.11"'
"$PYTHON_BIN" -m venv "$RUNTIME_DIR"
"$RUNTIME_DIR/bin/python" -m pip install 'pip==25.3'
"$RUNTIME_DIR/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.8.0' 'torchvision==0.23.0'
"$RUNTIME_DIR/bin/python" -m pip install -r "$NODE_ROOT/requirements-worker.txt"
"$RUNTIME_DIR/bin/python" "$NODE_ROOT/scripts/download_ltx_models.py" --directory "${LTX_MODELS:-$NODE_ROOT/models/ltx}" --runtime-config "$NODE_ROOT/runtime-ltx.json" --offload "${LTX_OFFLOAD:-model}"
"$RUNTIME_DIR/bin/python" -m pip check
"$RUNTIME_DIR/bin/python" -m pip freeze > "$NODE_ROOT/runtime-ltx-freeze.txt"
printf '\nLTX runtime installed separately from Krea. Restart ComfyUI, open Live Studio and select LTX.\n'
