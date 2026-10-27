"""GPU-only entry point. Invoked in a separate Python environment by manager.py."""
import base64
from collections import deque
import contextlib
import io
import json
import sys
import time
import traceback
from pathlib import Path

# Executable script, not a ComfyUI module import.
sys.path.insert(0, str(Path(__file__).parent))
from protocol import WIDTH, HEIGHT, MAX_MESSAGE, settings, frames


class Engine:
    def __init__(self, model):
        self.model = model
        self.pipe = None
        self.state = None
        self.index = 0
        self.prompt = None

    def load(self):
        import torch
        from diffusers import ModularPipeline, AutoModel
        if not torch.cuda.is_available():
            raise RuntimeError('An NVIDIA CUDA GPU is required; no CPU fallback is enabled.')
        self.torch = torch
        self.pipe = ModularPipeline.from_pretrained(self.model, trust_remote_code=True, local_files_only=True)
        # Diffusers 0.36 custom-code loading does not resolve local subfolders.
        # Load the transformer directly from its folder, then inject it.
        transformer = AutoModel.from_pretrained(str(Path(self.model) / 'transformer'), trust_remote_code=True, local_files_only=True, device_map='cuda', torch_dtype=torch.bfloat16, use_safetensors=True)
        self.pipe.update_components(transformer=transformer)
        self.pipe.load_components(names=['text_encoder', 'tokenizer', 'vae', 'scheduler'], local_files_only=True, device_map='cuda', torch_dtype={'default': torch.bfloat16, 'vae': torch.float16})
        for name in ('transformer', 'text_encoder', 'tokenizer', 'vae', 'scheduler'):
            if getattr(self.pipe, name, None) is None:
                raise RuntimeError(f'Failed to load {name}. Inspect worker.log for the component loader traceback.')
        for block in self.pipe.transformer.blocks:
            block.self_attn.fuse_projections()
        self.clear_vae = self.pipe.vae.clear_cache
        return {'gpu': torch.cuda.get_device_name(0), 'vram_gib': round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)}

    def reset(self, config):
        from diffusers.modular_pipelines import PipelineState
        # Diffusers deep-copies PipelineState by default, including all CUDA caches.
        # This worker is strictly serial; mutating one state avoids doubling cache VRAM.
        class InPlaceStreamState(PipelineState):
            def __deepcopy__(self, memo):
                return self
        self.state = None
        self.pipe.vae.clear_cache = self.clear_vae
        self.clear_vae()
        self.state = InPlaceStreamState()
        self.state.set('input_frames_cache', deque(maxlen=24))
        self.index, self.prompt = 0, None
        self.seed = config['seed']
        self.generator = self.torch.Generator('cpu').manual_seed(self.seed)
        self.torch.cuda.reset_peak_memory_stats()
        return {}

    def block(self, request):
        from PIL import Image
        config = settings(request['settings'])
        if self.state is None:
            raise RuntimeError('Start or reset a session before sending frames')
        if config['seed'] != self.seed:
            self.reset(config)
        images = []
        for encoded in frames(request['frames']):
            raw = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(raw)) as im:
                if im.size != (WIDTH, HEIGHT) or im.format not in ('JPEG', 'PNG'):
                    raise ValueError('Input must be an 832 × 480 JPEG or PNG')
                images.append(im.convert('RGB'))
        changed = config['prompt'] != self.prompt
        if changed:
            self.state.set('prompt_embeds', None)
            self.state.set('negative_prompt_embeds', None)
        start = time.perf_counter()
        with self.torch.inference_mode():
            self.state = self.pipe(self.state, video_stream=images, prompt=[config['prompt']], num_inference_steps=config['steps'], strength=config['strength'], block_idx=self.index, generator=self.generator, height=HEIGHT, width=WIDTH, output_type='pil', update_prompt_embeds=changed)
        self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        output = self.state.values['videos'][0]
        # Wan emits 9 frames for the initial 3-latent block, then 12 thereafter.
        # Keep the actual output count; never pretend duplicated frames were inferred.
        result = []
        for im in output:
            buf = io.BytesIO()
            im.save(buf, format='JPEG', quality=90)
            result.append(base64.b64encode(buf.getvalue()).decode('ascii'))
        self.prompt = config['prompt']
        self.index += 1
        return {'frames': result, 'block': self.index, 'seconds': elapsed, 'generated_fps': len(result) / max(elapsed, 1e-6), 'peak_vram_gib': self.torch.cuda.max_memory_allocated() / 2**30, 'prompt_changed': changed}


def main():
    transport = sys.stdout
    engine = Engine(sys.argv[1])
    while True:
        line = sys.stdin.readline(MAX_MESSAGE + 1)
        if not line:
            break
        try:
            if len(line) > MAX_MESSAGE:
                raise ValueError('Request too large')
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request['op'] == 'load':
                    result = engine.load()
                elif request['op'] == 'reset':
                    result = engine.reset(settings(request['settings']))
                elif request['op'] == 'block':
                    result = engine.block(request)
                else:
                    raise ValueError('Unknown operation')
            response = {'ok': True, **result}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
        transport.write(json.dumps(response, separators=(',', ':')) + '\n')
        transport.flush()


if __name__ == '__main__':
    main()
