"""LTX-2.5 B200 test backend; 12 delivered frames per 17-frame window."""
import base64
import io
import os
import time
from PIL import Image, ImageOps
from protocol import WIDTH, HEIGHT, BLOCK, settings, frames

LTX25_WIDTH, LTX25_HEIGHT = 768, 448


class LTX25Engine:
    def __init__(self, model):
        self.model = model
        self.initialized = False

    def load(self):
        import torch
        from ltx25_native import NativeLTX25
        if not torch.cuda.is_available():
            raise RuntimeError('LTX-2.5 requires an NVIDIA CUDA GPU')
        self.torch = torch
        mode = os.environ.get('LTX25_OFFLOAD', 'none')
        if mode not in ('none', 'cpu', 'disk'):
            raise ValueError('LTX25_OFFLOAD must be none, cpu or disk')
        self.pipe = NativeLTX25(self.model, mode)
        return dict(gpu=torch.cuda.get_device_name(0),
                    vram_gib=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
                    backend='ltx25', width=LTX25_WIDTH, height=LTX25_HEIGHT,
                    mode='experimental-video-latent-denoising', offload=mode,
                    message='LTX-2.5 configured. Model weights load on the first block.')

    def reset(self, config):
        self.seed = config['seed']
        self.index = 0
        self.previous = None
        self.prompt = None
        self.initialized = True
        self.torch.cuda.reset_peak_memory_stats()
        return {}

    def block(self, request):
        config = settings(request['settings'])
        if not self.initialized:
            raise ValueError('Reset the LTX-2.5 session before sending frames')
        images = []
        for encoded in frames(request['frames']):
            with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as im:
                if im.size != (WIDTH, HEIGHT) or im.format not in ('JPEG', 'PNG'):
                    raise ValueError('Input must be an 832 × 480 JPEG or PNG')
                images.append(ImageOps.fit(im.convert('RGB'), (LTX25_WIDTH, LTX25_HEIGHT)))
        if config['seed'] != self.seed:
            self.reset(config)
        anchor = self.previous if self.previous is not None else images[0]
        window = [anchor] + images + [images[-1]] * 4
        changed = config['prompt'] != self.prompt
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            result, schedule = self.pipe.generate(
                window, prompt=config['prompt'], seed=(self.seed+self.index) % 2**32,
                fps=config['fps'], strength=config['strength'], anchor=self.previous is not None)
        self.torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        if len(result) != 17:
            raise RuntimeError(f'LTX-2.5 returned {len(result)} frames; expected 17')
        delivered = result[1:1+BLOCK]
        self.previous = delivered[-1].copy()
        self.prompt = config['prompt']
        self.index += 1
        encoded = []
        for im in delivered:
            buf = io.BytesIO(); im.save(buf, format='JPEG', quality=90)
            encoded.append(base64.b64encode(buf.getvalue()).decode('ascii'))
        bypass = len(schedule) == 1
        return dict(frames=encoded, block=self.index, seconds=elapsed,
                    generated_fps=0.0 if bypass else BLOCK/max(elapsed, 1e-6),
                    model_frames=0 if bypass else 17, delivered_frames=BLOCK,
                    passthrough=bypass, effective_noise=schedule[0], denoising_steps=len(schedule)-1,
                    backend='ltx25', width=LTX25_WIDTH, height=LTX25_HEIGHT,
                    peak_vram_gib=self.torch.cuda.max_memory_allocated()/2**30,
                    prompt_changed=changed)
