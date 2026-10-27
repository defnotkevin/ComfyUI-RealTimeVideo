"""Run three real GPU blocks with a still image; save frames and measured timings."""
import argparse
import base64
import io
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'realtime'))
from worker import Engine
from PIL import Image, ImageOps

parser = argparse.ArgumentParser()
parser.add_argument('--backend', choices=['krea', 'ltx'], default='krea')
parser.add_argument('--offload', choices=['model', 'sequential', 'none'], default='model')
parser.add_argument('--image', type=Path, required=True)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--output', type=Path, default=Path('benchmark-output'))
parser.add_argument('--prompt', default='A cinematic portrait of a wizard with glowing sparks, in a stone hall.')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
image = ImageOps.fit(Image.open(args.image).convert('RGB'), (832, 480))
buf = io.BytesIO(); image.save(buf, format='JPEG', quality=95)
encoded = base64.b64encode(buf.getvalue()).decode()
config = dict(prompt=args.prompt, strength=0.3, steps=6, seed=42)
if args.backend == 'ltx':
    import os
    os.environ['LTX_OFFLOAD'] = args.offload
    from ltx_worker import LTXEngine
    Engine = LTXEngine
    config.update(backend='ltx', fps=12, steps=7)
engine = Engine(str(args.model.resolve()))
info = engine.load(); engine.reset(config)
metrics = []
for index in range(3):
    result = engine.block({'settings': config, 'frames': [encoded] * 12})
    frames = result.pop('frames')
    for j, frame in enumerate(frames):
        (args.output / f'block-{index:02d}-frame-{j:02d}.jpg').write_bytes(base64.b64decode(frame))
    metrics.append(result)
    print(json.dumps(result), flush=True)
(args.output / 'metrics.json').write_text(json.dumps({'hardware': info, 'blocks': metrics, 'note': 'Repeated still-image input. Tests execution and warm/cold throughput, not motion fidelity.'}, indent=2))
