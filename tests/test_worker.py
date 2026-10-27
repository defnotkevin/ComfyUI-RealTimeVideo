"""Exercise real worker state/control code with an explicitly fake inference pipeline."""
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
from worker import Engine


class State:
    def __init__(self): self.values = {}
    def set(self, name, value): self.values[name] = value


class Generator:
    def manual_seed(self, seed): return self


class Pipeline:
    def __init__(self):
        self.calls = []
        self.vae = types.SimpleNamespace(clear_cache=lambda: None)
    def __call__(self, state, **kwargs):
        self.calls.append((dict(state.values), kwargs))
        state.values['prompt_embeds'] = 'cached'
        state.values['videos'] = [[Image.new('RGB', (832, 480))] * (9 if kwargs['block_idx'] == 0 else 12)]
        return state


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.engine = Engine('not-used')
        self.engine.pipe = Pipeline()
        self.engine.clear_vae = lambda: None
        self.engine.torch = types.SimpleNamespace(Generator=lambda device: Generator(), inference_mode=contextlib.nullcontext, cuda=types.SimpleNamespace(reset_peak_memory_stats=lambda: None, synchronize=lambda: None, max_memory_allocated=lambda: 0))
        fake = types.ModuleType('diffusers.modular_pipelines'); fake.PipelineState = State
        self.patch = patch.dict(sys.modules, {'diffusers.modular_pipelines': fake})
        self.patch.start()
        self.config = dict(prompt='a', seed=42, strength=0.3, steps=6)
        self.engine.reset(self.config)
        buf = io.BytesIO(); Image.new('RGB', (832, 480)).save(buf, format='JPEG')
        self.frames = [base64.b64encode(buf.getvalue()).decode()] * 12

    def tearDown(self): self.patch.stop()

    def block(self, config=None):
        return self.engine.block({'settings': config or self.config, 'frames': self.frames})

    def test_prompt_cache_invalidated_only_when_changed(self):
        first = self.block()
        second = self.block()
        third = self.block({**self.config, 'prompt': 'new'})
        self.assertEqual(len(first['frames']), 9)
        self.assertEqual(len(second['frames']), 12)
        self.assertTrue(first['prompt_changed'])
        self.assertFalse(second['prompt_changed'])
        self.assertTrue(third['prompt_changed'])
        state, kwargs = self.engine.pipe.calls[-1]
        self.assertIsNone(state['prompt_embeds'])
        self.assertTrue(kwargs['update_prompt_embeds'])
        self.assertEqual(kwargs['block_idx'], 2)

    def test_seed_change_resets_state(self):
        self.block()
        self.block({**self.config, 'seed': 100})
        self.assertEqual(self.engine.pipe.calls[-1][1]['block_idx'], 0)
        self.assertEqual(self.engine.seed, 100)

    def test_independent_input_cache_after_reset(self):
        old = self.engine.state.values['input_frames_cache']
        old.append('old session')
        self.engine.reset(self.config)
        self.assertEqual(len(self.engine.state.values['input_frames_cache']), 0)
        self.assertIsNot(old, self.engine.state.values['input_frames_cache'])

    def test_bad_image_rejected_before_inference(self):
        buf = io.BytesIO(); Image.new('RGB', (20, 20)).save(buf, format='JPEG')
        self.frames = [base64.b64encode(buf.getvalue()).decode()] * 12
        with self.assertRaises(ValueError): self.block()
        self.assertEqual(self.engine.pipe.calls, [])


if __name__ == '__main__': unittest.main()
