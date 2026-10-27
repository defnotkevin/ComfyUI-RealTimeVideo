"""Experimental LTX video-conditioned windows; not a causal/real-time guarantee.

The 12-frame wire protocol is retained. Each call uses 17 model frames: one
continuity anchor, 12 new inputs, and four tail-padding frames. Only the 12 new
outputs are delivered. No Reactor API or remote inference is used.
"""
import base64
import io
import os
import time
from PIL import Image, ImageOps
from protocol import WIDTH, HEIGHT, BLOCK, settings, frames

LTX_WIDTH, LTX_HEIGHT = 512, 320


class LTXEngine:
    def __init__(self, model):
        self.model = model
        self.pipe = None
        self.initialized = False

    def load(self):
        import torch
        from diffusers import LTXConditionPipeline
        if not torch.cuda.is_available():
            raise RuntimeError('LTX requires an NVIDIA CUDA GPU')
        self.torch = torch
        self.pipe = LTXConditionPipeline.from_pretrained(
            self.model, torch_dtype=torch.bfloat16, local_files_only=True,
            use_safetensors=True)
        self.pipe.vae.enable_tiling()
        mode = os.environ.get('LTX_OFFLOAD', 'model')
        if mode == 'model':
            self.pipe.enable_model_cpu_offload()
        elif mode == 'sequential':
            self.pipe.enable_sequential_cpu_offload()
        elif mode == 'none':
            self.pipe.to('cuda')
        else:
            raise ValueError('Invalid LTX_OFFLOAD mode')
        return dict(gpu=torch.cuda.get_device_name(0),
                    vram_gib=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
                    backend='ltx', width=LTX_WIDTH, height=LTX_HEIGHT,
                    mode='experimental-video-windows', offload=mode)

    def reset(self, config):
        self.seed = config['seed']
        self.index = 0
        self.previous = None
        self.prompt = None
        self.embeddings = None
        self.initialized = True
        self.torch.cuda.reset_peak_memory_stats()
        return {}

    def block(self, request):
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
        config = settings(request['settings'])
        if not self.initialized:
            raise ValueError('Reset the LTX session before sending frames')
        images = []
        for encoded in frames(request['frames']):
            with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as im:
                if im.size != (WIDTH, HEIGHT) or im.format not in ('JPEG', 'PNG'):
                    raise ValueError('Input must be an 832 × 480 JPEG or PNG')
                images.append(ImageOps.fit(im.convert('RGB'), (LTX_WIDTH, LTX_HEIGHT)))
        if config['seed'] != self.seed:
            self.reset(config)
        anchor = self.previous if self.previous is not None else images[0]
        window = [anchor] + images + [images[-1]] * 4
        # Higher UI strength means more change; LTX conditioning strength is inverse.
        conditions = [LTXVideoCondition(video=window, frame_index=0, strength=1-config['strength'])]
        if self.previous is not None:
            conditions.append(LTXVideoCondition(image=anchor, frame_index=0, strength=1.0))
        changed = config['prompt'] != self.prompt
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            if changed:
                self.embeddings = self.pipe.encode_prompt(
                    prompt=config['prompt'], do_classifier_free_guidance=False)
            embedding, mask, _, _ = self.embeddings
            result = self.pipe(
                conditions=conditions, prompt_embeds=embedding, prompt_attention_mask=mask,
                width=LTX_WIDTH, height=LTX_HEIGHT, num_frames=17,
                frame_rate=config['fps'], num_inference_steps=config['steps'],
                guidance_scale=1.0, decode_timestep=0.05, decode_noise_scale=0.025,
                generator=self.torch.Generator('cpu').manual_seed((self.seed+self.index) % 2**32),
                output_type='pil').frames[0]
        self.torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        if len(result) != 17:
            raise RuntimeError(f'LTX returned {len(result)} frames; expected 17')
        delivered = result[1:1+BLOCK]
        self.previous = delivered[-1].copy()
        self.prompt = config['prompt']
        self.index += 1
        encoded = []
        for im in delivered:
            buf = io.BytesIO()
            im.save(buf, format='JPEG', quality=90)
            encoded.append(base64.b64encode(buf.getvalue()).decode('ascii'))
        return dict(frames=encoded, block=self.index, seconds=elapsed,
                    generated_fps=BLOCK/max(elapsed, 1e-6), model_frames=17,
                    delivered_frames=BLOCK, backend='ltx', width=LTX_WIDTH, height=LTX_HEIGHT,
                    peak_vram_gib=self.torch.cuda.max_memory_allocated()/2**30,
                    prompt_changed=changed)
