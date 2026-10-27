"""The downloader must override host transfer flags before importing Hub."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/download_models.py'


class DownloaderEnvironmentTests(unittest.TestCase):
    def test_direct_invocation_overrides_host_setting_before_hub_import(self):
        with tempfile.TemporaryDirectory() as directory:
            # Emulate Hub's import-time environment capture without network/weights.
            Path(directory, 'huggingface_hub.py').write_text(
                "import os\n"
                "assert os.environ.get('HF_HUB_ENABLE_HF_TRANSFER') == '0', 'host flag leaked into Hub'\n"
                "def snapshot_download(*args, **kwargs):\n"
                "    raise AssertionError('help must not download files')\n"
            )
            for inherited in ('1', '0', None):
                with self.subTest(inherited=inherited):
                    env = os.environ.copy()
                    env['PYTHONPATH'] = directory
                    env['PYTHONDONTWRITEBYTECODE'] = '1'
                    if inherited is None:
                        env.pop('HF_HUB_ENABLE_HF_TRANSFER', None)
                    else:
                        env['HF_HUB_ENABLE_HF_TRANSFER'] = inherited
                    result = subprocess.run(
                        [sys.executable, str(SCRIPT), '--help'],
                        env=env, capture_output=True, text=True, timeout=10,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('--runtime-config', result.stdout)


if __name__ == '__main__':
    unittest.main()
