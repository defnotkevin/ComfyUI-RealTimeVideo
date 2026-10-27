"""Download verification tested with a local Hub double, never real model weights."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ltx25_download', ROOT/'scripts/download_ltx25_models.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class DownloadTests(unittest.TestCase):
    def execute(self, corrupt=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = root/'runtime.json'; model = root/'model'
            payload = b'fake weights for download verification only'
            calls = []
            def download(repo, file, revision, local_dir):
                calls.append((repo, revision, file))
                p = Path(local_dir)/file; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(payload); return str(p)
            fake = types.SimpleNamespace(hf_hub_download=download,
                hf_hub_url=lambda *args, **kwargs: 'unused',
                get_hf_file_metadata=lambda _: types.SimpleNamespace(size=len(payload), etag='0'*64 if corrupt else hashlib.sha256(payload).hexdigest()))
            with patch.dict(sys.modules, {'huggingface_hub': fake}), patch.object(sys, 'argv', ['download', '--directory', str(model), '--runtime-config', str(config)]):
                if corrupt:
                    with self.assertRaisesRegex(RuntimeError, 'Checksum mismatch'): module.main()
                    self.assertFalse(config.exists())
                else:
                    module.main()
                    manifest = json.loads((model/'ltx25-models.json').read_text())
                    self.assertEqual(len(manifest['files']), 3)
                    self.assertEqual(json.loads(config.read_text())['offload'], 'none')
                    self.assertTrue(all(x[1] == module.MODEL_REVISION for x in calls))
    def test_pinned_success(self): self.execute()
    def test_corruption_not_published(self): self.execute(corrupt=True)
