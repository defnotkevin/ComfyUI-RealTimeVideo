"""HTTP/WebSocket integration tests with a fake GPU worker; no model inference."""
import asyncio
import importlib
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch
from aiohttp import web, WSServerHandshakeError
from aiohttp.test_utils import TestClient, TestServer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
server = types.ModuleType('server')
server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=web.RouteTableDef()))
sys.modules['server'] = server
routes = importlib.import_module('realtime.routes')


class FakeManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.commands = []
    def acquire(self):
        if not self.lock.acquire(False):
            raise RuntimeError('busy')
    def start(self, backend="krea"):
        self.backend = backend
        return {'gpu': 'TEST FAKE', 'vram_gib': 0}
    def call(self, command):
        self.commands.append(command)
        return {'ok': True, 'frames': [], 'block': 1}
    def shutdown(self):
        self.stopped.set()
    def release(self):
        self.shutdown()
        self.lock.release()


class RouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fake = FakeManager()
        self.patch = patch.object(routes, 'manager', self.fake)
        self.patch.start()
        app = web.Application()
        app.add_routes(server.PromptServer.instance.routes)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.origin = str(self.client.make_url('/')).rstrip('/')

    async def asyncTearDown(self):
        await self.client.close()
        self.patch.stop()

    async def connect(self):
        return await self.client.ws_connect('/krea-live/ws', headers={'Origin': self.origin})

    async def start(self, ws):
        await ws.send_json({'op': 'start', 'settings': {'prompt': 'test'}})
        self.assertEqual((await ws.receive_json())['type'], 'loading')
        self.assertEqual((await ws.receive_json())['type'], 'ready')

    async def test_full_session(self):
        ws = await self.connect()
        await self.start(ws)
        await ws.send_json({'op': 'block', 'settings': {'prompt': 'updated', 'strength': 0.4}, 'frames': ['a'] * 12})
        self.assertEqual((await ws.receive_json())['type'], 'frames')
        self.assertEqual(self.fake.commands[-1]['settings']['prompt'], 'updated')
        await ws.send_json({'op': 'reset', 'settings': {'prompt': 'reset'}})
        self.assertEqual((await ws.receive_json())['type'], 'reset')
        await ws.send_json({'op': 'stop'})
        await ws.receive()
        self.assertTrue(self.fake.stopped.is_set())
        self.assertFalse(self.fake.lock.locked())

    async def test_ltx_selection_and_mid_session_switch_denied(self):
        ws = await self.connect()
        await ws.send_json({'op': 'start', 'settings': {'prompt': 'test', 'backend': 'ltx25'}})
        self.assertEqual((await ws.receive_json())['type'], 'loading')
        self.assertEqual((await ws.receive_json())['type'], 'ready')
        self.assertEqual(self.fake.backend, 'ltx25')
        await ws.send_json({'op': 'reset', 'settings': {'prompt': 'test', 'backend': 'krea'}})
        self.assertIn('switching backend', (await ws.receive_json())['message'])
        await ws.receive()
        self.assertFalse(self.fake.lock.locked())

    async def test_foreign_origin_denied(self):
        with self.assertRaises(WSServerHandshakeError) as exc:
            await self.client.ws_connect('/krea-live/ws', headers={'Origin': 'https://evil.example'})
        self.assertEqual(exc.exception.status, 403)

    async def test_missing_origin_denied(self):
        with self.assertRaises(WSServerHandshakeError):
            await self.client.ws_connect('/krea-live/ws')

    async def test_explicit_proxy_origin(self):
        with patch.dict('os.environ', {'KREA_PUBLIC_ORIGIN': 'https://pod.example'}):
            ws = await self.client.ws_connect('/krea-live/ws', headers={'Origin': 'https://pod.example'})
            await self.start(ws)
            await ws.close()
            for origin in ['https://evil.example', 'http://pod.example', self.origin]:
                with self.assertRaises(WSServerHandshakeError) as exc:
                    await self.client.ws_connect('/krea-live/ws', headers={
                        'Origin': origin, 'X-Forwarded-Host': 'pod.example'})
                self.assertEqual(exc.exception.status, 403)

    async def test_exclusive_owner(self):
        first = await self.connect()
        second = await self.connect()
        self.assertEqual((await second.receive_json())['type'], 'error')
        await first.close()

    async def test_invalid_frame_count_closes(self):
        ws = await self.connect()
        await self.start(ws)
        await ws.send_json({'op': 'block', 'settings': {'prompt': 'x'}, 'frames': ['a']})
        self.assertEqual((await ws.receive_json())['type'], 'error')
        await ws.receive()
        self.assertFalse(self.fake.lock.locked())

    async def test_stop_interrupts_busy_worker(self):
        original = self.fake.call
        def blocking(command):
            if command['op'] == 'block':
                self.fake.stopped.wait(3)
                raise RuntimeError('cancelled')
            return original(command)
        self.fake.call = blocking
        ws = await self.connect()
        await self.start(ws)
        await ws.send_json({'op': 'block', 'settings': {'prompt': 'x'}, 'frames': ['a'] * 12})
        await asyncio.sleep(0.02)
        await ws.send_json({'op': 'stop'})
        await asyncio.wait_for(ws.receive(), timeout=1)
        self.assertTrue(self.fake.stopped.is_set())

    async def test_studio_security_headers(self):
        response = await self.client.get('/krea-live/studio')
        self.assertEqual(response.status, 200)
        self.assertIn("frame-ancestors 'self'", response.headers['Content-Security-Policy'])
        self.assertIn('getUserMedia', await (await self.client.get('/krea-live/studio.js')).text())


if __name__ == '__main__':
    unittest.main()
