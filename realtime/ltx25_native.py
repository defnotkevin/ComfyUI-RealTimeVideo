"""Experimental video-only img2img-style denoising using pinned native LTX blocks.

No IC-LoRA or performance-transfer claim: input frames are VAE-encoded, noised,
then denoised with text. A hard first-latent anchor carries the previous output.
"""
import json
from pathlib import Path
from ltx25_config import FILES, MODEL_REVISION, SOURCE_REVISION, schedule_for_strength


class NativeLTX25:
    def __init__(self, model, offload='none'):
        import torch
        from ltx_core.loader.registry import ModelRegistry
        from ltx_pipelines.utils.blocks import PromptEncoder, ImageConditioner, DiffusionStage, VideoDecoder
        from ltx_pipelines.utils.model_paths import ModelPaths
        from ltx_pipelines.utils.types import OffloadMode
        self.torch = torch
        self.device = torch.device('cuda')
        self.dtype = torch.bfloat16
        root = Path(model)
        manifest = json.loads((root / 'ltx25-models.json').read_text())
        if manifest.get('revision') != MODEL_REVISION or manifest.get('source_revision') != SOURCE_REVISION:
            raise ValueError('LTX-2.5 installation revision mismatch; rerun its installer')
        paths = {name: str(root / file) for name, file in FILES.items()}
        for path in paths.values():
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        model_paths = ModelPaths.from_split(**paths)
        # Cache weights/model shells across short windows. GPU lifecycle is
        # owned by the upstream blocks, not ComfyUI's memory manager.
        self.registry = ModelRegistry()
        common = dict(dtype=self.dtype, device=self.device, registry=self.registry)
        mode = OffloadMode(offload)
        self.prompt_encoder = PromptEncoder(model_paths, offload_mode=mode, **common)
        self.encoder = ImageConditioner(model_paths.video_vae(), **common)
        self.stage = DiffusionStage.from_checkpoint(model_paths.transformer(), offload_mode=mode, **common)
        self.decoder = VideoDecoder(model_paths.video_vae(), **common)
        self.prompt = None
        self.context = None

    def generate(self, images, prompt, seed, fps, strength, anchor):
        import numpy as np
        from PIL import Image
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.conditioning import VideoConditionByLatentIndex
        from ltx_pipelines.distilled import ancestral_sampler_kwargs, ANCESTRAL_NOISE_SEED_OFFSET
        from ltx_pipelines.utils.denoisers import SimpleDenoiser
        from ltx_pipelines.utils.types import ModalitySpec, VideoAudio
        torch = self.torch
        schedule = schedule_for_strength(strength)
        if len(schedule) == 1:
            return [image.copy() for image in images], schedule
        generator = torch.Generator(device=self.device).manual_seed(seed)
        if prompt != self.prompt:
            self.context = self.prompt_encoder([prompt], enhance_first_prompt=False)[0].video_encoding
            self.prompt = prompt
        # Native encoder expects B,C,F,H,W in [-1,1]. No temporary video file.
        pixels = torch.from_numpy(np.stack([np.asarray(im.convert('RGB')) for im in images])).to(
            device=self.device, dtype=self.dtype).permute(3, 0, 1, 2).unsqueeze(0) / 127.5 - 1
        latent = self.encoder(lambda enc: enc(pixels))
        del pixels
        conditionings = []
        if anchor:
            conditionings.append(VideoConditionByLatentIndex(latent=latent[:, :, :1].clone(), strength=1.0, latent_idx=0))
        sigmas = torch.tensor(schedule, device=self.device, dtype=torch.float32)
        video, _ = self.stage(
            denoiser=SimpleDenoiser(v_context=self.context, a_context=None),
            sigmas=sigmas, noiser=GaussianNoiser(generator),
            modalities=VideoAudio(video=ModalitySpec(
                latent=latent, conditioning_fps=fps, context=self.context,
                noise_scale=schedule[0], conditionings=conditionings), audio=None),
            **ancestral_sampler_kwargs(seed, self.dtype, ANCESTRAL_NOISE_SEED_OFFSET))
        del latent
        # Native decoders yield F,H,W,C floats in [0,1], possibly in several chunks.
        output = []
        for chunk in self.decoder(video.latent, generator=generator):
            array = (chunk.detach().float().clamp(0, 1).cpu().numpy() * 255).round().astype('uint8')
            output.extend(Image.fromarray(frame) for frame in array)
        return output, schedule
