"""Download a pinned official Diffusers checkpoint; never run at server startup."""
import argparse
import json
import os
from pathlib import Path
import sys

REPO = 'Lightricks/LTX-Video-0.9.7-distilled'
REVISION = '057509edea1493cae5e62e9d8f780ebda3fb4333'


def main():
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    from huggingface_hub import snapshot_download
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--runtime-config', type=Path, required=True)
    parser.add_argument('--offload', choices=['model', 'sequential', 'none'], default='model')
    args = parser.parse_args()
    model = args.directory.expanduser().resolve()
    snapshot_download(REPO, revision=REVISION, local_dir=model,
                      allow_patterns=['model_index.json', 'scheduler/*', 'tokenizer/*',
                                      'text_encoder/*', 'transformer/*', 'vae/*', '*.txt', 'README.md'],
                      ignore_patterns=['*.bin', '*.pt', '*.pth', '*.py'])
    # Validate all sharded weights referenced by the pinned component indices.
    for component in ['transformer', 'text_encoder', 'vae']:
        folder = model / component
        if not (folder / 'config.json').is_file():
            raise RuntimeError(f'Missing {component} config')
        if not list(folder.glob('*.safetensors')):
            raise RuntimeError(f'Missing {component} weights')
        for index in folder.glob('*.index.json'):
            for shard in set(json.loads(index.read_text())['weight_map'].values()):
                if not (folder / shard).is_file():
                    raise RuntimeError(f'Missing shard: {component}/{shard}')
    args.runtime_config.write_text(json.dumps(dict(python=sys.executable, model=str(model),
        repo=REPO, revision=REVISION, offload=args.offload), indent=2))
    print('LTX downloaded. Run the GPU smoke test before judging speed or quality.')


if __name__ == '__main__':
    main()
