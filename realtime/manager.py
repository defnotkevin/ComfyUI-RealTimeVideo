"""One GPU process, one owner, bounded IPC. No listening worker port."""
import atexit
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
from .protocol import MAX_MESSAGE

ROOT = Path(__file__).resolve().parents[1]


class WorkerManager:
    def __init__(self):
        self.lease = threading.Lock()
        self.lifecycle = threading.Lock()
        self.cancelled = threading.Event()
        self.process = None
        self.log = None
        self.responses = None

    def acquire(self):
        if not self.lease.acquire(blocking=False):
            raise RuntimeError('Krea GPU worker is busy. Stop the other live session or queued Krea job.')
        self.cancelled.clear()

    def release(self):
        self.shutdown()
        self.lease.release()

    def start(self):
        config_path = Path(os.environ.get('KREA_RUNTIME_CONFIG', ROOT / 'runtime.json'))
        if not config_path.is_file():
            raise RuntimeError('Krea runtime is not installed. Follow README.md and run scripts/install_runtime.sh on the GPU host.')
        config = json.loads(config_path.read_text())
        python = Path(config['python'])
        model = Path(config['model'])
        if not python.is_file() or not (model / 'modular_model_index.json').is_file():
            raise RuntimeError('Runtime paths are missing. Re-run the installer on this host.')
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', DIFFUSERS_ENABLE_HUB_KERNELS='0', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
        with self.lifecycle:
            if self.cancelled.is_set():
                raise RuntimeError('Session cancelled')
            self.log = (ROOT / 'worker.log').open('w', encoding='utf-8')
            self.process = subprocess.Popen([str(python), '-u', str(ROOT / 'realtime/worker.py'), str(model)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, env=env, bufsize=1)
        responses = queue.Queue(maxsize=2)
        self.responses = responses
        process = self.process
        def read():
            try:
                while True:
                    line = process.stdout.readline(MAX_MESSAGE + 1)
                    if not line:
                        responses.put(RuntimeError('GPU worker exited. See worker.log in the node package.'))
                        return
                    if len(line) > MAX_MESSAGE:
                        raise RuntimeError('Oversized worker response')
                    responses.put(json.loads(line))
            except Exception as exc:
                try:
                    responses.put_nowait(exc)
                except queue.Full:
                    pass
        threading.Thread(target=read, daemon=True).start()
        return self.call({'op': 'load'}, timeout=900)

    def call(self, command, timeout=300):
        process, responses = self.process, self.responses
        if process is None or process.poll() is not None:
            raise RuntimeError('GPU worker is not running')
        try:
            process.stdin.write(json.dumps(command, separators=(',', ':')) + '\n')
            process.stdin.flush()
            result = responses.get(timeout=timeout)
            if isinstance(result, Exception):
                raise result
            if not result.get('ok'):
                raise RuntimeError(result.get('error', 'GPU worker failed'))
            return result
        except queue.Empty as exc:
            self.shutdown()
            raise RuntimeError('GPU worker timed out. See worker.log; lower load or check CUDA compatibility.') from exc

    def shutdown(self):
        self.cancelled.set()
        with self.lifecycle:
            process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()
        if self.log:
            self.log.close()
            self.log = None


manager = WorkerManager()
atexit.register(manager.shutdown)
