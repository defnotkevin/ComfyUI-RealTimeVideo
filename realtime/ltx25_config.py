"""Pinned native LTX-2.5 model/source contract, shared by installer and worker."""
SOURCE_REVISION = 'e98f6d927d83c92ef2be6aec76c8440b19306e39'
MODEL_REPO = 'Lightricks/LTX-2.5'
MODEL_REVISION = '49b093a39c19169a8a81721dcc541e07e945232f'
FILES = {
    'transformer_path': 'diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors',
    'text_encoder_path': 'text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors',
    'video_vae_path': 'vae/ltx-2.5-video-vae-conv-bf16.safetensors',
}
SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)


def schedule_for_strength(strength):
    """Use a suffix of the trained schedule, rounding UP to a supported noise level."""
    if not 0 <= strength <= 1:
        raise ValueError('Strength must be 0–1')
    if strength == 0:
        return (0.0,)
    start = max(i for i, sigma in enumerate(SIGMAS[:-1]) if sigma >= strength)
    return SIGMAS[start:]
