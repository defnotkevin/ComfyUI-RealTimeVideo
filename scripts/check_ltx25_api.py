"""Import/signature checks only; does not load weights or claim GPU inference."""
import inspect
from ltx_core.loader.registry import ModelRegistry
from ltx_core.conditioning import VideoConditionByLatentIndex
from ltx_pipelines.utils.blocks import PromptEncoder, ImageConditioner, DiffusionStage, VideoDecoder
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import ModalitySpec, VideoAudio, OffloadMode
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.distilled import ancestral_sampler_kwargs

for cls, fields in [(DiffusionStage.__call__, ['sigmas', 'noiser', 'modalities', 'loop']),
                    (VideoDecoder.__call__, ['latent', 'generator']),
                    (ModalitySpec, ['latent', 'noise_scale', 'conditioning_fps'])]:
    missing = set(fields) - set(inspect.signature(cls).parameters)
    if missing:
        raise RuntimeError(f'Native LTX API mismatch: {missing}')
print('Native LTX-2.5 imports and API contract verified; GPU inference not tested.')
