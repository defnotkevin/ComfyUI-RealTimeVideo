"""Explicit installation only. No network downloads when ComfyUI starts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

KREA = 'krea/krea-realtime-video'
KREA_REV = '9a2ceea280e5d584c6a28d384a9c19e38c810f8c'
WAN = 'Wan-AI/Wan2.1-T2V-14B-Diffusers'
WAN_REV = '38ec498cb3208fb688890f8cc7e94ede2cbd7f68'


def main():
    from huggingface_hub import snapshot_download
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--runtime-config', type=Path, required=True)
    args = parser.parse_args()
    root = args.directory.expanduser().resolve()
    krea, wan = root / 'krea', root / 'wan-components'
    print('Downloading pinned official Krea code/weights and Wan text encoder, tokenizer, VAE, scheduler.', flush=True)
    snapshot_download(KREA, revision=KREA_REV, local_dir=krea, allow_patterns=['*.py', '*.json', '*.md', 'LICENSE*', 'NOTICE*', 'COPYING*', 'transformer/*'], ignore_patterns=['*.bin', '*.pt', '*.pth'])
    snapshot_download(WAN, revision=WAN_REV, local_dir=wan, allow_patterns=['README.md', 'LICENSE*', 'NOTICE*', 'COPYING*', 'text_encoder/*', 'tokenizer/*', 'vae/*', 'scheduler/*'], ignore_patterns=['*.bin', '*.pt', '*.pth'])
    # Resolve every component to local storage, including the custom transformer.
    # Its repo must equal the pipeline path so Diffusers forwards trust_remote_code.
    index_path = krea / 'modular_model_index.json'
    index = json.loads(index_path.read_text())
    for name in ['transformer', 'text_encoder', 'tokenizer', 'vae', 'scheduler']:
        spec = index[name][2]
        spec['repo'] = str(krea if name == 'transformer' else wan)
        spec['revision'] = None
    index_path.write_text(json.dumps(index, indent=2))
    manifest = {'krea_revision': KREA_REV, 'wan_revision': WAN_REV, 'files': []}
    for directory in [krea, wan]:
        for path in sorted(directory.rglob('*')):
            if not path.is_file() or '.cache' in path.parts:
                continue
            digest = hashlib.sha256()
            with path.open('rb') as source:
                for chunk in iter(lambda: source.read(16 * 1024 * 1024), b''):
                    digest.update(chunk)
            manifest['files'].append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()})
    (root / 'installed-manifest.json').write_text(json.dumps(manifest, indent=2))
    args.runtime_config.write_text(json.dumps({'python': sys.executable, 'model': str(krea)}, indent=2))
    print('Installed and hashed local files. GPU inference still needs validation.', flush=True)


if __name__ == '__main__':
    main()
