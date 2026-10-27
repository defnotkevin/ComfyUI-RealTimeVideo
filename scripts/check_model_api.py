"""Validate installed official model code against pinned libraries without loading weights."""
import argparse
from pathlib import Path
from diffusers import ModularPipeline
from diffusers.utils.dynamic_modules_utils import get_class_from_dynamic_module
from accelerate import init_empty_weights
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
args = parser.parse_args()
pipeline = ModularPipeline.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
required = {'video_stream', 'block_idx', 'input_frames_cache', 'prompt_embeds', 'update_prompt_embeds', 'strength'}
assert required <= {x.name for x in pipeline.blocks.inputs}, 'Required streaming inputs missing'
model_class = get_class_from_dynamic_module(str(Path(args.model) / 'transformer'), 'causal_model.py', class_name='CausalWanModel', local_files_only=True)
with init_empty_weights():
    config = model_class.load_config(args.model, subfolder='transformer')
    transformer = model_class.from_config(config)
    for block in transformer.blocks:
        block.self_attn.fuse_projections()
assert len(transformer.blocks) == 40
print('Official pipeline and transformer APIs validated on meta tensors; no inference performed.')
