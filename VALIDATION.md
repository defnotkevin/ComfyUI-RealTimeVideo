# Validation record — 2026-10-26

## Passed locally

- **21 automated tests**: protocol validation, malformed/oversized frame lists, origin rejection, chunk padding, workflow link consistency, complete WebSocket session, exclusive ownership, cancellation during work, endpoint headers, subprocess IPC, failure propagation, timeout termination, pre-launch cancellation, prompt-cache invalidation, seed reset, fresh session caches and image dimension checks.
- Real subprocess transport tests use a small fake worker. Model-control tests use a fake inference pipeline. Neither is a GPU generation benchmark, and neither fake is exposed in the shipped application.
- Python source compilation, both JavaScript modules' syntax checks, and installer Bash syntax check.
- Pinned worker dependencies installed in a disposable macOS/Python 3.13 compatibility environment; `pip check` reported no broken requirements. The deployment installer targets Linux/Python 3.11/CUDA 12.8 and was not run on this Mac.
- The pinned official Krea `WanModularPipeline` imports with Diffusers 0.36.0. Its required streaming inputs were checked. Its 40-block transformer was instantiated on meta tensors and QKV projection fusion succeeded without loading model weights.
- Compatibility testing exposed a Diffusers 0.36 local custom-code subfolder lookup issue. The worker loads the custom transformer directly from its local directory and injects it into the pipeline; the model API check uses the same direct-directory import.
- Live Studio was opened in the browser. Starting without a source displayed the expected error; appearance/VFX/background fields correctly composed the full prompt. No browser warning/error logs were reported during those checks. `studio-preview.png` records the interface; its blank output is intentional, not a generated result.

## Still requires the GPU host

- Full model downloads, actual safetensors weight loading and CUDA execution.
- Fresh ComfyUI import and execution of both supplied workflows against the deployed ComfyUI version.
- Camera permission/capture and complete browser → ComfyUI → GPU → browser inference.
- Real rendered quality, temporal consistency, prompt responsiveness and character/motion fidelity.
- Sustained latency, measured throughput, peak VRAM, thermal behavior and long-session stability.
- Browser recording with actual generated output and proxy disconnect/reconnect behavior.

## Reproduce CPU tests

Use a disposable Python environment with Pillow and aiohttp installed:

```bash
python -m pip install Pillow aiohttp
python -m unittest discover -s tests -v
node --check studio/studio.js
node --check web/extension.js
bash -n scripts/install_runtime.sh
```

The GPU installation and smoke test are documented in README.md. Model-dependent API validation uses `scripts/check_model_api.py`; GPU inference uses `scripts/smoke_gpu.py`.
