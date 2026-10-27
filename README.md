# ComfyUI · Krea Realtime Video

Competition review: see `SUBMISSION_CHECKLIST.txt` and `THIRD_PARTY_NOTICES.txt`.
This is a candidate entry, not a certified or submitted entry. GPU execution,
reproducibility, publication and submission materials remain outstanding.

Custom ComfyUI nodes and a browser studio for **Krea streaming video-to-video**: connect a camera or video, describe a different appearance, visual effects and background, and receive generated frames incrementally.

**Status: implementation complete for first GPU validation; inference speed and visual quality have not yet been tested on a GPU.** CPU protocol tests, dependency checks, official model API imports/meta construction, and browser interface checks are recorded in [VALIDATION.md](VALIDATION.md). There is no simulated inference mode in the shipped application.

![Live Studio controls](studio-preview.png)

## What is included

- **Krea · Prompt & Settings**: prompt, transformation strength, denoising steps and seed.
- **Krea · Live Camera Studio**: opens a separate browser panel on the same ComfyUI server. No additional exposed service or local tunnel.
- **Krea · Load Video Clip (30s max)**: samples a source clip at up to 12 fps, center-cropped to 832 × 480.
- **Krea · Streaming Video to Video**: processes IMAGE batches through the same stateful pipeline and returns IMAGE frames plus measured timing data.
- [Live camera workflow](workflows/01-live-camera.json) and [video-to-video workflow](workflows/02-video-to-video.json).
- An isolated, persistent GPU worker; pinned installer; model manifest; GPU smoke-test script.

The studio includes camera/local-video input, appearance/VFX/background prompt composition, live prompt/strength/steps updates, context reset, side-by-side preview, silent WebM recording, generated FPS, block round-trip latency, GPU allocated memory, and stale-frame counts.

## GPU-host installation

Copy this entire folder to `ComfyUI/custom_nodes/ComfyUI-RealTimeVideo` on your **Linux NVIDIA GPU host**. Leave the local Mac copy as source code. Do not copy a runtime environment from another machine.

Start with an **RTX PRO 6000 Blackwell 96 GB** or another CUDA GPU with substantial VRAM. This is a sensible capacity target, not a measured speed guarantee. The default runtime uses CUDA 12.8 wheels and requires a compatible host driver. Budget around 50 GB for model artifacts plus space for the environment and temporary downloads; allocate at least 80–100 GB free disk. Host memory must also accommodate model loading and frame batches; 64 GB or more is a practical starting point.

Inside the GPU container:

```bash
cd /workspace/ComfyUI/custom_nodes/ComfyUI-RealTimeVideo
bash scripts/install_runtime.sh
```

The installer requires Python 3.11 and creates `.runtime` inside this node directory. It **does not upgrade or install model dependencies into ComfyUI's Python environment**. ComfyUI itself already provides Torch, NumPy, Pillow, aiohttp and PyAV. The clip workflow uses the core `SaveWEBM` node.

Optional installation locations:

```bash
KREA_INSTALL_PYTHON=/usr/bin/python3.11 \
KREA_VENV=/workspace/krea-runtime \
KREA_MODELS=/workspace/ComfyUI/models/krea-realtime \
bash scripts/install_runtime.sh
```

The installer writes `runtime.json` with absolute paths. Re-run it after moving environments/model folders. It downloads only the Krea transformer/code and Wan text encoder, tokenizer, VAE and scheduler, using fixed repository revisions. No model downloads occur merely by starting ComfyUI.

The installer disables the optional `HF_HUB_ENABLE_HF_TRANSFER` flag inherited
from some GPU container images. Installing `hf_transfer` with the container's
global `pip` does not install it in `.runtime`. If an older copy fails with
this error, retry using `HF_HUB_ENABLE_HF_TRANSFER=0 bash scripts/install_runtime.sh`.
The pip update notice is informational; it is not the cause of that failure.

If Hugging Face authentication is required, authenticate privately on the GPU host using its CLI. Never put tokens in workflows or browser fields. The installer uses Hugging Face's normal cached authentication.

Restart ComfyUI after copying the extension. Keep ComfyUI behind your existing authenticated HTTPS proxy. Use a regular Chrome/Edge window if an embedded browser blocks camera permission. HTTPS or localhost is necessary for browser camera access.

## Run the live workflow

1. Import `workflows/01-live-camera.json` into ComfyUI.
2. Edit **Prompt & Settings**, then click ComfyUI's **Run/Queue** once to pass those values to the studio node. This does not start the camera or model.
3. Click **Open Live Studio** on the node. You can also open `krea-live/studio` relative to the ComfyUI base URL, e.g. `https://your-host/proxy/8188/krea-live/studio`.
4. Click **Enable camera** and grant permission, or choose **Open video**. Source video files stay in the browser; sampled image frames are sent during generation.
5. Click **Start generation**. Wait for model loading and first-block kernel compilation. Subsequent blocks reuse loaded weights and temporal state.
6. Change the full prompt, or compose it from appearance/VFX/background fields. Click **Apply on next block**. Prompt, strength and step changes apply at the next submission; changing seed resets temporal context.
7. Use **Reset temporal context** between unrelated scenes. It becomes available between requests. **Stop** closes the camera, disconnects the session and terminates the worker to release its GPU memory. Starting again reloads the model.

The live studio runs independently of ComfyUI's job queue. Do not queue other GPU-heavy ComfyUI workflows while it is active; the separate worker is not part of ComfyUI's VRAM accounting. Start with ComfyUI freshly restarted if other models are still resident. Only one Krea live or batch session can own the worker at a time.

## Run a video clip

Put an MP4/MOV/WebM/MKV in `ComfyUI/input`, refresh the UI, import `02-video-to-video.json`, and select the file in **Load Video Clip**. Start with five seconds, 12 fps, strength 0.3 and six steps. Run the workflow. Output saves under `ComfyUI/output/Krea` using the core SaveWEBM node.

The loader caps source duration at 30 seconds and samples no more than 360 frames. Input and output image batches remain in host RAM; a 360-frame RGB float32 batch alone is roughly 1.6 GiB. Live streaming uses bounded buffers instead of accumulating a full video.

Output is silent. The clip workflow does not carry audio or promise lip synchronization. The initial decoded block has nine generated frames; batch mode resamples those to twelve to preserve source clip duration. Timing metrics count actual generated frames, not duplicates.

## First GPU test

After installation, validate model APIs without loading weights:

```bash
.runtime/bin/python scripts/check_model_api.py --model models/krea
```

Then use a non-sensitive test image for three real generated blocks:

```bash
.runtime/bin/python scripts/smoke_gpu.py \
  --model models/krea \
  --image /workspace/test-person.jpg \
  --output benchmark-output
```

Use the configured runtime/model paths instead if you customized the installer. The smoke test saves generated JPEGs and `metrics.json`. It repeats a still image to test execution, cold/warm block time and memory; it does not validate movement fidelity. Next test the live workflow with actual motion and prompt changes. Inspect `worker.log` for CUDA, memory or component loading errors.

## How the integration works

```text
Browser camera/video → 12-frame crop/encode buffer → same-origin WebSocket
  → ComfyUI route → isolated worker stdin/stdout → Krea ModularPipeline
  → persistent causal/context + VAE caches → JPEG frame block
  → browser bounded playback queue / optional local WebM recording
```

The backend uses the official model's `video_stream` input and a persistent `PipelineState`. It explicitly clears cached text embeddings and cross-attention context on prompt changes, starts every session with a fresh input-frame cache, and restores the VAE cache-clearing method on reset. A serial state subclass avoids Diffusers deep-copying all GPU caches each block.

Only 832 × 480 is exposed initially because the published pipeline config fixes attention token geometry. It accepts 12 input frames per block, emits nine on the first block and typically twelve thereafter. At 12 capture fps, collecting a block already takes approximately one second, before inference and transport. This is **chunked streaming**, not a zero-latency per-frame filter.

There is one request in flight and at most twelve pending input frames. If inference falls behind, stale input is dropped; playback is bounded to two blocks. Low latency is prioritized over retaining every live source frame. A faster GPU improves throughput but does not remove block collection delay. No 24/30-fps claim is made.

The appearance, VFX and background controls build one prompt. They are not independent segmentation layers. This model may restyle faces, clothing and scenery but does not guarantee exact reference identity, pose preservation, character replacement, independent multi-character control or lip movement transfer. No reference-image adapter or audio model is included.

## Security and dependency boundary

- Browser routes use the existing ComfyUI listener; no additional worker port, shell endpoint, tunnel or arbitrary URL fetch.
- WebSocket handshakes require matching Origin/Host. If a reverse proxy rewrites Host, configure it to preserve the public Host; do not disable the check.
- **Origin checks are not authentication.** ComfyUI must remain behind authentication; anyone authorized to access this instance can use these nodes. This package is not a multi-tenant sandbox.
- Worker executable/model locations come from administrator-owned `runtime.json`, never browser input. Subprocesses launch with an argument list, not a shell.
- Payloads, image dimensions, frame queues and session ownership are bounded. Stop/disconnect kills the worker; no output accumulation on the server for live sessions.
- Model code is explicitly trusted only during the operator-initiated installation/load of the pinned official release. `trust_remote_code=True` executes that local code; repository pinning reduces unexpected changes but is not an independent security audit or sandbox.
- Worker inference uses offline Hugging Face settings and disables automatic Hub kernel downloads. This is an application setting, not OS-level network isolation.
- Camera microphone access is disabled. Frames go only to this ComfyUI host; logs may include errors and prompts, so treat the node folder as private. Recording stays in the user's browser until downloaded.

## Sources and versions

- [Krea model and streaming examples](https://huggingface.co/krea/krea-realtime-video/blob/9a2ceea280e5d584c6a28d384a9c19e38c810f8c/README.md)
- Krea revision: `9a2ceea280e5d584c6a28d384a9c19e38c810f8c`
- [Wan components](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/tree/38ec498cb3208fb688890f8cc7e94ede2cbd7f68)
- Wan revision: `38ec498cb3208fb688890f8cc7e94ede2cbd7f68`
- Diffusers 0.36.0, PyTorch 2.8.0/CUDA 12.8, Transformers 4.57.1. Direct worker dependencies are pinned in `requirements-worker.txt`; the installer records resolved transitive versions in `runtime-freeze.txt`.

Extension code is MIT. Downloaded model weights, code and dependencies keep their own licenses. The Hugging Face Krea model card identifies Apache-2.0; review its pinned model card and the Wan component terms for your use. This package does not vendor the separate legacy Krea GitHub server.

### Reverse-proxy WebSocket origin configuration

If Studio opens but Start reports a connection failure and ComfyUI logs an origin
rejection, the reverse proxy may rewrite the Host header. Set `KREA_PUBLIC_ORIGIN`
in the **ComfyUI server process** to the exact browser origin (scheme and hostname,
plus port if present; no path, `/api`, query string or token), then restart ComfyUI:

```bash
export KREA_PUBLIC_ORIGIN='https://YOUR-POD-8188.proxy.runpod.net'
# Start ComfyUI with your usual command in this same shell.
```

This explicitly pins the accepted browser origin; arbitrary forwarded headers and
other origins remain rejected. Update it if the pod URL changes. The `/api/` prefix
is supported by the relative Studio WebSocket URL. No installer rerun or model
download is required for this configuration. Origin rejection occurs before the
worker starts, so it appears in ComfyUI's log rather than `worker.log`.
