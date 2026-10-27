"""Optional real-torch/native-API test with model blocks replaced; no weights/GPU."""
import importlib.util
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'realtime'))


@unittest.skipUnless(importlib.util.find_spec('ltx_core'), 'Native LTX runtime not installed in CPU test environment')
class NativeContractTests(unittest.TestCase):
    def test_source_latents_sigma_anchor_and_decoding(self):
        import torch
        from PIL import Image
        from types import SimpleNamespace
        from ltx25_native import NativeLTX25
        from ltx_pipelines.utils.types import VideoAudio
        native = NativeLTX25.__new__(NativeLTX25)
        native.torch = torch
        native.device = torch.device('cpu')
        native.dtype = torch.float32
        native.prompt = None
        encoded_prompts, captures = [], []
        def prompts(texts, **kwargs):
            encoded_prompts.append(texts)
            return [SimpleNamespace(video_encoding=torch.ones(1, 2, 4))]
        native.prompt_encoder = prompts
        def encoder(callback):
            def encode(pixels):
                self.assertEqual(tuple(pixels.shape), (1,3,17,32,32))
                self.assertEqual(float(pixels.min()), -1.)
                return torch.ones(1,128,3,1,1)
            return callback(encode)
        native.encoder = encoder
        def stage(**kwargs):
            captures.append(kwargs)
            return VideoAudio(SimpleNamespace(latent=kwargs['modalities'].video.latent), None)
        native.stage = stage
        native.decoder = lambda latent, **kwargs: iter([torch.zeros(8,32,32,3), torch.ones(9,32,32,3)])
        images = [Image.new('RGB', (32,32))]*17
        result, sigmas = native.generate(images, 'test', 42, 12, .3, True)
        self.assertEqual(len(result), 17)
        self.assertEqual(result[-1].getpixel((0,0)), (255,255,255))
        self.assertEqual(sigmas, (.421875,0.))
        self.assertEqual(captures[-1]['modalities'].video.noise_scale, sigmas[0])
        self.assertEqual(captures[-1]['modalities'].video.conditionings[0].strength, 1.)
        self.assertIsNone(captures[-1]['modalities'].audio)
        self.assertTrue(callable(captures[-1]['loop']))
        native.generate(images, 'test', 43, 12, .3, False)
        self.assertEqual(len(encoded_prompts), 1)
        native.generate(images, 'new', 44, 12, .3, False)
        self.assertEqual(len(encoded_prompts), 2)
        result, sigmas = native.generate(images, 'new', 44, 12, 0, False)
        self.assertEqual(sigmas, (0.,))
        self.assertEqual(len(captures), 3)


if __name__ == '__main__': unittest.main()
