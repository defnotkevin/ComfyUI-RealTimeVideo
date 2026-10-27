"""Regenerate standard LiteGraph workflows with explicit links and widget values."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def node(id, type, pos, inputs=None, outputs=None, widgets=None, title=None):
    return dict(id=id, type=type, pos=pos, size=[360, 220], flags={}, order=id-1, mode=0, inputs=inputs or [], outputs=outputs or [], properties={'Node name for S&R': type}, widgets_values=widgets or [], **({'title': title} if title else {}))


def socket(name, type, link=None, links=None):
    result = {'name': name, 'type': type}
    if links is not None:
        result['links'] = links
    else:
        result['link'] = link
    return result


def save(name, nodes, links):
    graph = dict(last_node_id=max(n['id'] for n in nodes), last_link_id=max((x[0] for x in links), default=0), nodes=nodes, links=links, groups=[], config={}, extra={'ds': {'scale': 0.85, 'offset': [30, 50]}}, version=0.4)
    (ROOT / 'workflows' / name).write_text(json.dumps(graph, indent=2))


prompt = 'A cinematic portrait of a wizard, glowing magical sparks around their hands, an ancient stone hall in the background.'
config = node(1, 'KreaRTSettings', [30, 100], outputs=[socket('KREA_SETTINGS', 'KREA_SETTINGS', links=[1])], widgets=[prompt, 0.3, 6, 42])
live = node(2, 'KreaRTLiveStudio', [450, 100], inputs=[socket('settings', 'KREA_SETTINGS', 1)])
note = node(3, 'Note', [30, 380], widgets=['1. Queue this workflow once to pass settings to the studio.\n2. Click Open Live Studio on the Krea node.\n3. Enable camera or open a video, then Start generation.\n4. Edit prompt/strength and Apply on next block.\nStop closes the camera and releases the worker.\nInstall the GPU runtime first; see README.md.'])
save('01-live-camera.json', [config, live, note], [[1, 1, 0, 2, 0, 'KREA_SETTINGS']])
config = node(1, 'KreaRTSettings', [30, 80], outputs=[socket('KREA_SETTINGS', 'KREA_SETTINGS', links=[1])], widgets=[prompt, 0.3, 6, 42])
clip = node(2, 'KreaRTLoadClip', [30, 380], outputs=[socket('images', 'IMAGE', links=[2]), socket('fps', 'FLOAT', links=[4])], widgets=['', 5.0, 12])
render = node(3, 'KreaRTVideoToVideo', [460, 120], inputs=[socket('images', 'IMAGE', 2), socket('settings', 'KREA_SETTINGS', 1)], outputs=[socket('images', 'IMAGE', links=[3]), socket('metrics_json', 'STRING', links=[])])
savevideo = node(4, 'SaveWEBM', [890, 120], inputs=[socket('images', 'IMAGE', 3), {'name': 'fps', 'type': 'FLOAT', 'link': 4, 'widget': {'name': 'fps'}}], widgets=['Krea/clip', 'vp9', 12.0, 24.0])
note = node(5, 'Note', [460, 400], widgets=['Put a video in ComfyUI/input, refresh, and select it in Load Video Clip.\nStart with 5 seconds. Maximum: 30 seconds at 12 fps.\nThe source is center-cropped to 832 × 480. Output is silent.\nStop Live Studio before running this workflow.\nSaveWEBM is a ComfyUI core node; update ComfyUI if missing.'])
save('02-video-to-video.json', [config, clip, render, savevideo, note], [[1, 1, 0, 3, 1, 'KREA_SETTINGS'], [2, 2, 0, 3, 0, 'IMAGE'], [3, 3, 0, 4, 0, 'IMAGE'], [4, 2, 1, 4, 1, 'FLOAT']])

config = node(1, 'RTVideoSettings', [30, 100], outputs=[socket('KREA_SETTINGS', 'KREA_SETTINGS', links=[1])], widgets=[prompt, 0.3, 7, 42, 'ltx', 12])
live = node(2, 'KreaRTLiveStudio', [450, 100], inputs=[socket('settings', 'KREA_SETTINGS', 1)], title='Realtime Video · Open Studio')
note = node(3, 'Note', [30, 380], widgets=['Install scripts/install_ltx_runtime.sh on the GPU host first. Queue once, then open Live Studio. LTX uses experimental 512x320 video-conditioned chunks, with a previous-frame anchor. It is not a causal streaming model. Switch back to Krea after stopping. GPU speed and continuity require testing.'])
save('03-ltx-live-camera.json', [config, live, note], [[1, 1, 0, 2, 0, 'KREA_SETTINGS']])
