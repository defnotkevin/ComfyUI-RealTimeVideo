"""Real subprocess IPC with a tiny fake worker; never invokes the model runtime."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import realtime.manager as module


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'realtime').mkdir()
        (self.root / 'model').mkdir()
        (self.root / 'model/modular_model_index.json').write_text('{}')
        (self.root / 'runtime.json').write_text(json.dumps({'python': sys.executable, 'model': str(self.root / 'model')}))
        (self.root / 'realtime/worker.py').write_text('''import sys,json,time
for line in sys.stdin:
 c=json.loads(line)
 if c['op']=='wait': time.sleep(20)
 if c['op']=='fail': print(json.dumps({'ok':False,'error':'test failure'}),flush=True)
 else: print(json.dumps({'ok':True,'op':c['op']}),flush=True)
''')
        self.patch = patch.object(module, 'ROOT', self.root)
        self.patch.start()
        self.env = patch.dict(os.environ, {'KREA_RUNTIME_CONFIG': str(self.root / 'runtime.json')})
        self.env.start()
        self.manager = module.WorkerManager()
        self.manager.acquire()

    def tearDown(self):
        self.manager.release()
        self.patch.stop(); self.env.stop(); self.temp.cleanup()

    def test_lifecycle(self):
        self.assertEqual(self.manager.start()['op'], 'load')
        process = self.manager.process
        self.assertEqual(self.manager.call({'op':'block'})['op'], 'block')
        self.manager.shutdown()
        self.assertIsNotNone(process.poll())

    def test_ltx_runtime_is_separate(self):
        (self.root / 'model/model_index.json').write_text('{}')
        config = self.root / 'runtime-ltx.json'
        config.write_text(json.dumps({'python': sys.executable, 'model': str(self.root / 'model'), 'offload': 'model'}))
        with patch.dict(os.environ, {'LTX_RUNTIME_CONFIG': str(config)}):
            self.assertEqual(self.manager.start('ltx')['op'], 'load')
            self.assertEqual(self.manager.process.args[-1], 'ltx')

    def test_ltx25_runtime_marker_and_worker_selection(self):
        (self.root / 'model/ltx25-models.json').write_text('{}')
        config = self.root / 'runtime-ltx25.json'
        config.write_text(json.dumps({'python': sys.executable, 'model': str(self.root / 'model'), 'offload': 'none'}))
        with patch.dict(os.environ, {'LTX25_RUNTIME_CONFIG': str(config)}):
            self.assertEqual(self.manager.start('ltx25')['op'], 'load')
            self.assertEqual(self.manager.process.args[-1], 'ltx25')

    def test_timeout_terminates(self):
        self.manager.start()
        process = self.manager.process
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            self.manager.call({'op':'wait'}, timeout=0.02)
        self.assertIsNotNone(process.poll())

    def test_worker_failure_propagates(self):
        self.manager.start()
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            self.manager.call({'op':'fail'})

    def test_cancel_before_start_prevents_launch(self):
        self.manager.shutdown()
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            self.manager.start()
        self.assertIsNone(self.manager.process)


if __name__ == '__main__': unittest.main()
