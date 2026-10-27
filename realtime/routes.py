import asyncio
import contextlib
import json
from pathlib import Path
from aiohttp import web, WSMsgType
from server import PromptServer
from .manager import manager
from .protocol import MAX_MESSAGE, frames, settings, same_origin

STUDIO = Path(__file__).resolve().parents[1] / 'studio'
routes = PromptServer.instance.routes


@routes.get('/krea-live/studio')
async def studio(request):
    return web.FileResponse(STUDIO / 'index.html', headers={'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff', 'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'self'; base-uri 'none'; object-src 'none'"})


@routes.get('/krea-live/studio.js')
async def studio_js(request):
    return web.FileResponse(STUDIO / 'studio.js')


@routes.get('/krea-live/studio.css')
async def studio_css(request):
    return web.FileResponse(STUDIO / 'studio.css')


@routes.get('/krea-live/ws')
async def stream(request):
    if not same_origin(request.headers.get('Origin'), request.host):
        raise web.HTTPForbidden(text='Krea requires a same-origin browser connection.')
    ws = web.WebSocketResponse(max_msg_size=MAX_MESSAGE, heartbeat=30, receive_timeout=120)
    await ws.prepare(request)
    try:
        manager.acquire()
    except RuntimeError as exc:
        await ws.send_json({'type': 'error', 'message': str(exc)})
        await ws.close()
        return ws
    job = None
    ready = False

    async def work(command):
        nonlocal ready
        try:
            if command['op'] == 'start':
                await ws.send_json({'type': 'loading', 'message': 'Loading Krea on GPU. First load and kernel compilation can take several minutes.'})
                info = await asyncio.to_thread(manager.start)
                await asyncio.to_thread(manager.call, {'op': 'reset', 'settings': command['settings']})
                ready = True
                await ws.send_json({'type': 'ready', **info})
            else:
                response = await asyncio.to_thread(manager.call, command)
                await ws.send_json({'type': 'frames' if command['op'] == 'block' else 'reset', **response})
        except Exception as exc:
            ready = False
            if not ws.closed:
                await ws.send_json({'type': 'error', 'message': str(exc)})
                await ws.close()

    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                break
            value = json.loads(msg.data)
            if not isinstance(value, dict):
                raise ValueError('Expected an object')
            op = value.get('op')
            if op == 'stop':
                break
            if job and not job.done():
                raise ValueError('Only one request may be in flight')
            if op == 'start' and not ready:
                command = {'op': op, 'settings': settings(value.get('settings'))}
            elif op in ('block', 'reset') and ready:
                command = {'op': op, 'settings': settings(value.get('settings'))}
                if op == 'block':
                    command['frames'] = frames(value.get('frames'))
            else:
                raise ValueError('Start a session before sending frames')
            job = asyncio.create_task(work(command))
    except (ValueError, TypeError, asyncio.TimeoutError) as exc:
        if not ws.closed:
            await ws.send_json({'type': 'error', 'message': str(exc)})
    finally:
        # Kill first to unblock the reader thread. Then join the request before a new owner.
        await asyncio.to_thread(manager.shutdown)
        if job:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await job
        manager.release()
        await ws.close()
    return ws
