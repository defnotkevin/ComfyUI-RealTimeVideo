"""CPU transport/schedule tests. Fake inference is NOT GPU validation."""
import base64
import contextlib
import io
from pathlib import Path
import sys
import types
import unittest
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'realtime'))
from ltx25_worker import LTX25Engine
from ltx25_config import schedule_for_strength, SIGMAS
from protocol import settings


class FakeNative:
    def __init__(self): self.calls = []
    def generate(self, window, **kwargs):
        self.calls.append((window, kwargs))
        return [Image.new('RGB', (768, 448), (i, 0, 0)) for i in range(17)], schedule_for_strength(kwargs['strength'])


class LTX25Tests(unittest.TestCase):
    def setUp(self):
        self.engine = LTX25Engine('unused')
        self.engine.pipe = FakeNative()
        self.engine.torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext,
            cuda=types.SimpleNamespace(reset_peak_memory_stats=lambda: None, synchronize=lambda: None, max_memory_allocated=lambda: 0))
        self.config = settings(dict(prompt='wizard', backend='ltx25', strength=.3, fps=24))
        self.engine.reset(self.config)
        buffer = io.BytesIO(); Image.new('RGB', (832, 480)).save(buffer, format='JPEG')
        self.frames = [base64.b64encode(buffer.getvalue()).decode()] * 12
    def block(self, **extra):
        return self.engine.block({'settings': {**self.config, **extra}, 'frames': self.frames})
    def test_schedule_is_trained_suffix_and_zero_is_bypass(self):
        self.assertEqual(schedule_for_strength(1), SIGMAS)
        self.assertEqual(schedule_for_strength(.3), (.421875, 0.))
        self.assertEqual(schedule_for_strength(0), (0.,))
        for strength in [.001, .4, .8, .99]:
            schedule = schedule_for_strength(strength)
            self.assertGreaterEqual(schedule[0], strength)
            self.assertTrue(any(schedule == SIGMAS[i:] for i in range(8)))
    def test_trim_padding_anchor_and_dimensions(self):
        result = self.block()
        self.assertEqual(len(result['frames']), 12)
        self.assertEqual(result['width'], 768)
        self.assertEqual(result['denoising_steps'], 1)
        self.assertEqual(self.engine.previous.getpixel((0,0)), (12,0,0))
        self.block()
        window, kwargs = self.engine.pipe.calls[-1]
        self.assertEqual(len(window), 17)
        self.assertEqual(window[0].getpixel((0,0)), (12,0,0))
        self.assertTrue(kwargs['anchor'])
        self.assertEqual(kwargs['fps'], 24)
    def test_zero_strength_not_counted_as_generated(self):
        result = self.block(strength=0)
        self.assertTrue(result['passthrough'])
        self.assertEqual(result['generated_fps'], 0)
        self.assertEqual(result['model_frames'], 0)
    def test_seed_reset_removes_anchor(self):
        self.block(); self.block(seed=999)
        self.assertFalse(self.engine.pipe.calls[-1][1]['anchor'])
        self.assertEqual(self.engine.index, 1)
    def test_bad_input_before_native_call(self):
        self.frames = ['bad'] * 12
        with self.assertRaises(ValueError): self.block()
        self.assertEqual(self.engine.pipe.calls, [])
