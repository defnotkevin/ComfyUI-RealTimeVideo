import json
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from realtime.protocol import settings, frames, same_origin, padded_blocks


class ProtocolTests(unittest.TestCase):
    def test_valid_settings(self):
        self.assertEqual(settings({'prompt': '  hello  '})['prompt'], 'hello')

    def test_invalid_settings(self):
        for value in [None, [], {'prompt': ''}, {'prompt': 'x', 'strength': float('nan')}, {'prompt': 'x', 'strength': 2}, {'prompt': 'x', 'steps': True}, {'prompt': 'x', 'steps': 1}, {'prompt': 'x', 'seed': -1}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                settings(value)

    def test_frame_limits(self):
        self.assertEqual(len(frames(['a'] * 12)), 12)
        for value in [None, [], ['a'] * 13, [123] * 12, ['x' * (1024 * 1024 + 1)] * 12]:
            with self.assertRaises(ValueError):
                frames(value)

    def test_origin(self):
        self.assertTrue(same_origin('https://pod.proxy.runpod.net', 'pod.proxy.runpod.net'))
        self.assertTrue(same_origin('http://localhost:8188', 'localhost:8188'))
        for origin in [None, 'null', 'https://evil.example', 'https://pod.proxy.runpod.net.evil.test', 'https://user@pod.proxy.runpod.net', 'https://pod.proxy.runpod.net/path']:
            self.assertFalse(same_origin(origin, 'pod.proxy.runpod.net'))

    def test_chunk_tail(self):
        parts = list(padded_blocks(list(range(14))))
        self.assertEqual(parts[0], (list(range(12)), 12))
        self.assertEqual(parts[1], ([12] + [13] * 11, 2))
        self.assertEqual(list(padded_blocks([])), [])

    def test_workflow_links(self):
        for file in (ROOT / 'workflows').glob('*.json'):
            graph = json.loads(file.read_text())
            nodes = {n['id']: n for n in graph['nodes']}
            self.assertEqual(len(nodes), len(graph['nodes']))
            for id, source, outslot, dest, inslot, kind in graph['links']:
                self.assertIn(id, nodes[source]['outputs'][outslot]['links'])
                self.assertEqual(id, nodes[dest]['inputs'][inslot]['link'])
                self.assertEqual(kind, nodes[source]['outputs'][outslot]['type'])
                self.assertEqual(kind, nodes[dest]['inputs'][inslot]['type'])


if __name__ == '__main__':
    unittest.main()
