"""Explicit LTX-2.5 download, size/hash verification and local runtime config."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'realtime'))
from ltx25_config import FILES, MODEL_REPO, MODEL_REVISION, SOURCE_REVISION


def main():
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    from huggingface_hub import hf_hub_download, get_hf_file_metadata, hf_hub_url
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--runtime-config', type=Path, required=True)
    parser.add_argument('--offload', choices=['none', 'cpu', 'disk'], default='none')
    args = parser.parse_args()
    root = args.directory.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for file in FILES.values():
        print('Downloading/verifying', file, flush=True)
        metadata = get_hf_file_metadata(hf_hub_url(MODEL_REPO, file, revision=MODEL_REVISION))
        path = Path(hf_hub_download(MODEL_REPO, file, revision=MODEL_REVISION, local_dir=root))
        if metadata.size is not None and path.stat().st_size != metadata.size:
            raise RuntimeError(f'Size mismatch: {file}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(16*1024*1024), b''):
                digest.update(chunk)
        checksum = digest.hexdigest()
        etag = (metadata.etag or '').strip('"')
        if len(etag) == 64 and checksum != etag:
            raise RuntimeError(f'Checksum mismatch: {file}')
        records.append(dict(file=file, bytes=path.stat().st_size, sha256=checksum))
    manifest = dict(repo=MODEL_REPO, revision=MODEL_REVISION, source_revision=SOURCE_REVISION, files=records)
    target = root / 'ltx25-models.json'
    temp = target.with_suffix('.tmp'); temp.write_text(json.dumps(manifest, indent=2)); temp.replace(target)
    temp = args.runtime_config.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(python=sys.executable, model=str(root), offload=args.offload,
                                   source_revision=SOURCE_REVISION), indent=2))
    temp.replace(args.runtime_config)
    print('LTX-2.5 files verified. GPU inference still needs testing.', flush=True)


if __name__ == '__main__':
    main()
