"""LTX window protocol tests with fake inference; no GPU performance claims."""
import base64
import contextlib
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'realtime'))
from ltx_worker import LTXEngine
from protocol import settings


class FakePipe:
    def __init__(self):
        self.calls = []
        self.encodes = 0
    def encode_prompt(self, **kwargs):
        self.encodes += 1
        return ('embedding', 'mask', None, None)
    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(frames=[[Image.new('RGB', (512, 320), (i, 0, 0)) for i in range(17)]])


class LTXTests(unittest.TestCase):
    def setUp(self):
        self.engine = LTXEngine('unused')
        self.engine.pipe = FakePipe()
        self.engine.torch = types.SimpleNamespace(
            Generator=lambda _: types.SimpleNamespace(manual_seed=lambda seed: seed),
            inference_mode=contextlib.nullcontext,
            cuda=types.SimpleNamespace(reset_peak_memory_stats=lambda: None, synchronize=lambda: None,
                                      max_memory_allocated=lambda: 0))
        self.patch = patch.dict(sys.modules, {'diffusers.pipelines.ltx.pipeline_ltx_condition': types.SimpleNamespace(LTXVideoCondition=lambda **kw: types.SimpleNamespace(**kw))})
        self.patch.start()
        self.config = settings(dict(prompt='test', backend='ltx', seed=42, fps=24))
        self.engine.reset(self.config)
        buf = io.BytesIO(); Image.new('RGB', (832, 480)).save(buf, format='JPEG')
        self.frames = [base64.b64encode(buf.getvalue()).decode()] * 12
    def tearDown(self):
        self.patch.stop()
    def block(self, config=None):
        return self.engine.block(dict(settings=config or self.config, frames=self.frames))
    def test_window_padding_anchor_and_actual_output_count(self):
        result = self.block()
        first = self.engine.pipe.calls[0]
        self.assertEqual(len(first['conditions'][0].video), 17)
        self.assertEqual(first['frame_rate'], 24)
        self.assertAlmostEqual(first['conditions'][0].strength, .7)
        self.assertEqual(len(result['frames']), 12)
        self.assertEqual(self.engine.previous.getpixel((0, 0)), (12, 0, 0))
        self.block()
        conditions = self.engine.pipe.calls[1]['conditions']
        self.assertEqual(len(conditions), 2)
        self.assertEqual(conditions[1].strength, 1)
        self.assertEqual(conditions[1].image.getpixel((0, 0)), (12, 0, 0))
    def test_cache_prompt_and_seed_reset(self):
        self.block(); self.block()
        self.assertEqual(self.engine.pipe.encodes, 1)
        self.block({**self.config, 'prompt': 'changed'})
        self.assertEqual(self.engine.pipe.encodes, 2)
        self.block({**self.config, 'seed': 43})
        self.assertEqual(self.engine.index, 1)
        self.assertEqual(len(self.engine.pipe.calls[-1]['conditions']), 1)
    def test_input_validation_precedes_inference(self):
        self.frames = ['bad'] * 12
        with self.assertRaises(ValueError): self.block()
        self.assertEqual(self.engine.pipe.calls, [])
    def test_backend_and_fps_validation(self):
        self.assertEqual(settings({'prompt':'x'})['backend'], 'krea')
        for extra in [{'backend':'evil'}, {'fps':0}, {'fps':True}]:
            with self.assertRaises(ValueError): settings({'prompt':'x', **extra})
