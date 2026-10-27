"""Small, dependency-free protocol shared by ComfyUI and its GPU worker."""
import math
from urllib.parse import urlsplit

WIDTH, HEIGHT, BLOCK = 832, 480, 12
MAX_MESSAGE = 16 * 1024 * 1024
MAX_FRAME = 1024 * 1024


def settings(value):
    if not isinstance(value, dict):
        raise ValueError('Settings must be an object')
    backend = value.get('backend', 'krea')
    if backend not in ('krea', 'ltx'):
        raise ValueError('Backend must be krea or ltx')
    fps = value.get('fps', 12)
    if type(fps) is not int or not 2 <= fps <= 30:
        raise ValueError('FPS must be an integer from 2 to 30')
    prompt = value.get('prompt', '')
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 4000:
        raise ValueError('Enter a prompt of 1–4000 characters')
    strength = value.get('strength', 0.3)
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Strength must be between 0 and 1')
    steps = value.get('steps', 6)
    seed = value.get('seed', 42)
    for name, number, low, high in [('steps', steps, 2, 12), ('seed', seed, 0, 2**32 - 1)]:
        if type(number) is not int or not low <= number <= high:
            raise ValueError(f'{name} must be an integer from {low} to {high}')
    return dict(prompt=prompt.strip(), strength=float(strength), steps=steps, seed=seed, backend=backend, fps=fps)


def frames(value):
    if not isinstance(value, list) or len(value) != BLOCK:
        raise ValueError('Send exactly 12 frames per block')
    if any(not isinstance(x, str) or not 1 <= len(x) <= MAX_FRAME for x in value):
        raise ValueError('Invalid or oversized frame')
    return value


def same_origin(origin, host):
    """Never trust X-Forwarded-Host supplied by a client."""
    try:
        p = urlsplit(origin or '')
        return p.scheme in ('http', 'https') and p.netloc.lower() == host.lower() and not p.username and not p.password and p.path in ('', '/') and not p.query and not p.fragment
    except ValueError:
        return False


def allowed_origin(origin, host, public_origin=''):
    """An explicit public origin pins access when a reverse proxy rewrites Host.

    Forwarded headers are intentionally never used as an authorization source.
    """
    if not public_origin:
        return same_origin(origin, host)
    try:
        expected = urlsplit(public_origin)
        if not same_origin(public_origin, expected.netloc):
            return False
        actual = urlsplit(origin or '')
        return same_origin(origin, expected.netloc) and actual.scheme == expected.scheme
    except ValueError:
        return False


def padded_blocks(sequence):
    for offset in range(0, len(sequence), BLOCK):
        part = list(sequence[offset:offset + BLOCK])
        valid = len(part)
        yield part + [part[-1]] * (BLOCK - valid), valid
