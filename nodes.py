import base64
import io
import json
from .realtime.protocol import settings, padded_blocks, WIDTH, HEIGHT
from .realtime.manager import manager


class KreaSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'prompt': ('STRING', {'multiline': True, 'default': 'A cinematic portrait of a wizard, glowing magical sparks around their hands, an ancient stone hall in the background.'}),
            'strength': ('FLOAT', {'default': 0.3, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
            'steps': ('INT', {'default': 6, 'min': 2, 'max': 12}),
            'seed': ('INT', {'default': 42, 'min': 0, 'max': 4294967295, 'control_after_generate': False}),
        }}
    RETURN_TYPES = ('KREA_SETTINGS',)
    FUNCTION = 'configure'
    CATEGORY = 'Krea Realtime'

    def configure(self, **kwargs):
        return (settings(kwargs),)


class KreaLiveStudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'settings': ('KREA_SETTINGS',)}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = 'studio'
    CATEGORY = 'Krea Realtime'

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def studio(self, settings):
        # Running this node only prepares the studio link, never starts a camera.
        return {'ui': {'krea_settings': [settings]}}


class KreaVideoToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',), 'settings': ('KREA_SETTINGS',)}}
    RETURN_TYPES = ('IMAGE', 'STRING')
    RETURN_NAMES = ('images', 'metrics_json')
    FUNCTION = 'render'
    CATEGORY = 'Krea Realtime'
    DESCRIPTION = 'Restyle up to 360 frames (30 seconds at 12 fps). Use a video loader to provide IMAGE frames. The same persistent streaming pipeline is used as Live Studio.'

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def render(self, images, settings):
        import numpy as np
        import torch
        from PIL import Image, ImageOps
        import comfy.model_management as mm
        from comfy.utils import ProgressBar
        config = globals()['settings'](settings)
        if not 1 <= len(images) <= 360:
            raise ValueError('Provide 1–360 frames; sample at 12 fps for a maximum 30-second clip.')
        manager.acquire()
        try:
            # The separate worker is not visible to ComfyUI's VRAM accounting.
            mm.unload_all_models()
            mm.soft_empty_cache()
            manager.start()
            manager.call({'op': 'reset', 'settings': config})
            generated, metrics = [], []
            progress = ProgressBar((len(images) + 11) // 12)
            for index, (chunk, valid) in enumerate(padded_blocks(images)):
                mm.throw_exception_if_processing_interrupted()
                encoded = []
                for tensor in chunk:
                    im = Image.fromarray((tensor.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)).convert('RGB')
                    im = ImageOps.fit(im, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    im.save(buf, format='JPEG', quality=95)
                    encoded.append(base64.b64encode(buf.getvalue()).decode())
                result = manager.call({'op': 'block', 'settings': config, 'frames': encoded})
                batch = []
                for value in result.pop('frames'):
                    with Image.open(io.BytesIO(base64.b64decode(value))) as im:
                        batch.append(np.asarray(im.convert('RGB'), dtype=np.float32) / 255)
                if not batch:
                    raise RuntimeError('The model returned no frames')
                # Resample the first decoded 9-frame block to 12 to preserve clip duration.
                # This does NOT count as additional generated FPS in metrics.
                indices = np.linspace(0, len(batch) - 1, 12).round().astype(int)
                generated.extend(batch[i] for i in indices[:valid])
                metrics.append(result)
                progress.update_absolute(index + 1)
            return (torch.from_numpy(np.stack(generated)), json.dumps(metrics, indent=2))
        finally:
            manager.release()



class KreaLoadClip:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        from pathlib import Path
        files = sorted(p.name for p in Path(folder_paths.get_input_directory()).iterdir() if p.is_file() and p.suffix.lower() in ('.mp4', '.webm', '.mov', '.mkv'))
        return {'required': {'video': (files, {'video_upload': True}), 'duration_seconds': ('FLOAT', {'default': 5, 'min': 0.1, 'max': 30, 'step': 0.1}), 'fps': ('INT', {'default': 12, 'min': 2, 'max': 12})}}
    RETURN_TYPES = ('IMAGE', 'FLOAT')
    RETURN_NAMES = ('images', 'fps')
    FUNCTION = 'load'
    CATEGORY = 'Krea Realtime'

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        import folder_paths
        from pathlib import Path
        p = Path(folder_paths.get_input_directory()) / video
        return (p.stat().st_mtime_ns, p.stat().st_size)

    def load(self, video, duration_seconds, fps):
        import av
        import folder_paths
        import numpy as np
        import torch
        from PIL import Image, ImageOps
        from pathlib import Path
        import comfy.model_management as mm
        root = Path(folder_paths.get_input_directory()).resolve()
        path = (root / video).resolve()
        if path.parent != root or not path.is_file():
            raise ValueError('Select a video in the ComfyUI input directory')
        if not 0.1 <= duration_seconds <= 30 or not 2 <= fps <= 12:
            raise ValueError('Duration must be 0.1–30 seconds and FPS 2–12')
        limit = min(360, max(1, int(duration_seconds * fps)))
        result = []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            source_fps = float(stream.average_rate or 30)
            first_time = None
            for i, frame in enumerate(container.decode(stream)):
                mm.throw_exception_if_processing_interrupted()
                timestamp = float(frame.time) if frame.time is not None else i / source_fps
                if first_time is None:
                    first_time = timestamp
                timestamp -= first_time
                if timestamp >= duration_seconds:
                    break
                if timestamp + 1e-6 < len(result) / fps:
                    continue
                image = ImageOps.fit(frame.to_image().convert('RGB'), (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS)
                array = np.asarray(image)
                # Reuse a frame for slower source rates; never decode unbounded input.
                while len(result) < limit and len(result) / fps <= timestamp + 1e-6:
                    result.append(array)
                if len(result) >= limit:
                    break
        if not result:
            raise ValueError('No decodable video frames')
        return (torch.from_numpy(np.stack(result).astype(np.float32) / 255), float(fps))


NODE_CLASS_MAPPINGS = {'KreaRTLoadClip': KreaLoadClip, 'KreaRTSettings': KreaSettings, 'KreaRTLiveStudio': KreaLiveStudio, 'KreaRTVideoToVideo': KreaVideoToVideo}
NODE_DISPLAY_NAME_MAPPINGS = {'KreaRTLoadClip': 'Krea · Load Video Clip (30s max)', 'KreaRTSettings': 'Krea · Prompt & Settings', 'KreaRTLiveStudio': 'Krea · Live Camera Studio', 'KreaRTVideoToVideo': 'Krea · Streaming Video to Video'}
