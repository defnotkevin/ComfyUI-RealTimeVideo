#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=0
NODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${LTX25_INSTALL_PYTHON:-python3.11}"
RUNTIME_DIR="$NODE_ROOT/.runtime-ltx25"
SOURCE_DIR="$NODE_ROOT/.ltx25-source"
"$PYTHON_BIN" -c 'import sys,platform; assert platform.system()=="Linux", "Install on Linux GPU host"; assert sys.version_info[:2]==(3,11), "Use Python 3.11"'
REVISION="$($PYTHON_BIN -c 'import sys; sys.path.insert(0, sys.argv[1]); from ltx25_config import SOURCE_REVISION; print(SOURCE_REVISION)' "$NODE_ROOT/realtime")"
if [ ! -d "$SOURCE_DIR" ]; then
    git init "$SOURCE_DIR"
    git -C "$SOURCE_DIR" remote add origin https://github.com/Lightricks/LTX-2.git
    git -C "$SOURCE_DIR" fetch --depth 1 origin "$REVISION"
    git -C "$SOURCE_DIR" checkout --detach "$REVISION"
fi
if [ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" != "$REVISION" ] || [ -n "$(git -C "$SOURCE_DIR" status --porcelain)" ]; then
    echo 'Native LTX source differs from the pinned clean revision. Preserve your changes and move it aside before retrying.' >&2
    exit 1
fi
"$PYTHON_BIN" -m venv "$RUNTIME_DIR"
"$RUNTIME_DIR/bin/python" -m pip install 'pip==25.3'
"$RUNTIME_DIR/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.8.0' 'torchvision==0.23.0' 'torchaudio==2.8.0'
"$RUNTIME_DIR/bin/python" -m pip install -r "$NODE_ROOT/requirements-ltx25.txt"
# Constraints keep the native package resolver from replacing the tested torch trio.
"$RUNTIME_DIR/bin/python" -m pip install -c "$NODE_ROOT/requirements-ltx25-constraints.txt" "$SOURCE_DIR/packages/ltx-core" "$SOURCE_DIR/packages/ltx-pipelines"
"$RUNTIME_DIR/bin/python" -m pip check
"$RUNTIME_DIR/bin/python" "$NODE_ROOT/scripts/check_ltx25_api.py"
"$RUNTIME_DIR/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable; check RunPod driver/runtime"; print(torch.cuda.get_device_name(0)); x=torch.ones(1, device="cuda"); torch.cuda.synchronize()'
"$RUNTIME_DIR/bin/python" "$NODE_ROOT/scripts/download_ltx25_models.py" --directory "${LTX25_MODELS:-$NODE_ROOT/models/ltx25}" --runtime-config "$NODE_ROOT/runtime-ltx25.json" --offload "${LTX25_OFFLOAD:-none}"
"$RUNTIME_DIR/bin/python" -m pip freeze > "$NODE_ROOT/runtime-ltx25-freeze.txt"
printf '\nLTX-2.5 installed separately. Restart ComfyUI and select LTX-2.5 in Studio. Test GPU output before judging speed.\n'
