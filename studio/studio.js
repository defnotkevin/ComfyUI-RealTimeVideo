const $ = id => document.getElementById(id);
const video = $('source'), capture = $('capture'), output = $('output');
const ctx = capture.getContext('2d'), out = output.getContext('2d');
let sourceStream, sourceURL, socket, timer, playbackTimer, ready = false, busy = false;
let inputs = [], outputs = [], dropped = 0, sentAt = 0, active, recorder, recording = [];
let generation = 0;
const status = message => { $('status').textContent = message; };
function readSettings() {
    const value = {prompt: $('prompt').value.trim(), strength: Number($('strength').value), steps: Number($('steps').value), seed: Number($('seed').value)};
    if (!value.prompt || value.prompt.length > 4000) throw Error('Enter a prompt of 1–4000 characters.');
    if (!Number.isInteger(value.steps) || value.steps < 2 || value.steps > 12) throw Error('Steps must be 2–12.');
    if (!Number.isInteger(value.seed) || value.seed < 0 || value.seed > 4294967295) throw Error('Invalid seed.');
    return value;
}
try {
    if (location.hash.length > 1) {
        const initial = JSON.parse(decodeURIComponent(location.hash.slice(1)));
        for (const key of ['prompt', 'strength', 'steps', 'seed']) if (initial[key] !== undefined) $(key).value = initial[key];
        history.replaceState(null, '', location.pathname + location.search);
    }
} catch { status('Could not read workflow settings; using defaults.'); }
$('strength-value').value = Number($('strength').value).toFixed(2);
$('strength').oninput = () => { $('strength-value').value = Number($('strength').value).toFixed(2); };
function releaseSource() {
    sourceStream?.getTracks().forEach(track => track.stop());
    sourceStream = null;
    video.pause(); video.srcObject = null;
    video.removeAttribute('src'); video.load();
    if (sourceURL) URL.revokeObjectURL(sourceURL);
    sourceURL = null;
}
function toggle(running) {
    $('start').disabled = running;
    $('stop').disabled = !running;
    $('camera').disabled = running;
    $('video-file').disabled = running;
    $('fps').disabled = running;
    $('reset').disabled = !ready;
    $('record').disabled = !ready;
}
$('camera').onclick = async () => {
    try {
        if (!navigator.mediaDevices?.getUserMedia) throw Error('Camera access requires HTTPS or localhost.');
        releaseSource();
        sourceStream = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 832}, height: {ideal: 480}}, audio: false});
        video.srcObject = sourceStream; await video.play();
        $('source-label').textContent = 'Camera · audio off';
        status('Camera ready. Click Start generation to send frames to your ComfyUI host.');
    } catch (error) { status(error.message); }
};
$('video-file').onchange = async event => {
    if (!event.target.files[0]) return;
    try {
        releaseSource(); sourceURL = URL.createObjectURL(event.target.files[0]);
        video.src = sourceURL; video.loop = true; await video.play();
        $('source-label').textContent = 'Local video · looping';
        status('Video ready. Only sampled frames are sent when generation starts.');
    } catch (error) { status(error.message); }
};
$('compose').onclick = () => {
    const pieces = [$('appearance').value, $('effects').value, $('background').value].map(x => x.trim()).filter(Boolean);
    if (pieces.length) $('prompt').value = `Cinematic video. ${pieces.join('. ')}.`;
};
$('apply').onclick = () => {
    try { active = readSettings(); status(ready ? 'Updated settings will apply to the next submitted block.' : 'Settings ready.'); }
    catch (error) { status(error.message); }
};
function sendBlock() {
    if (!ready || busy || inputs.length < 12 || socket?.readyState !== WebSocket.OPEN) return;
    busy = true; sentAt = performance.now();
    socket.send(JSON.stringify({op: 'block', settings: active, frames: inputs.splice(0, 12)}));
    $('reset').disabled = true;
}
function sample() {
    if (!ready || video.readyState < 2) return;
    // Center crop, matching the batch node. No stretching body proportions.
    const ratio = Math.max(832 / video.videoWidth, 480 / video.videoHeight);
    const width = video.videoWidth * ratio, height = video.videoHeight * ratio;
    ctx.drawImage(video, (832 - width) / 2, (480 - height) / 2, width, height);
    inputs.push(capture.toDataURL('image/jpeg', 0.85).split(',')[1]);
    if (inputs.length > 12) { inputs.shift(); dropped++; }
    $('dropped').textContent = String(dropped);
    sendBlock();
}
async function showFrames(frames, epoch) {
    const decoded = [];
    for (const raw of frames) {
        const image = new Image(); image.src = `data:image/jpeg;base64,${raw}`;
        await image.decode(); decoded.push(image);
    }
    if (epoch !== generation) return;
    // Bound playback to two blocks. Drop stale output instead of increasing delay.
    outputs.push(...decoded); if (outputs.length > 24) outputs.splice(0, outputs.length - 24);
}
function stop(message = 'Stopped. Camera closed and GPU worker released.') {
    generation++; ready = false; busy = false;
    clearInterval(timer); clearInterval(playbackTimer);
    inputs = []; outputs = [];
    if (recorder?.state === 'recording') recorder.stop();
    const oldSocket = socket; socket = null;
    if (oldSocket?.readyState === WebSocket.OPEN) oldSocket.send(JSON.stringify({op: 'stop'}));
    oldSocket?.close(); releaseSource(); toggle(false); status(message);
}
$('start').onclick = () => {
    try {
        active = readSettings();
        const fps = Number($('fps').value);
        if (!Number.isInteger(fps) || fps < 2 || fps > 30) throw Error('Capture FPS must be 2–30.');
        if (video.readyState < 2) throw Error('Enable a camera or open a video first.');
        generation++; const epoch = generation;
        const endpoint = new URL('ws', location.href); endpoint.hash = ''; endpoint.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        socket = new WebSocket(endpoint); const current = socket;
        dropped = 0; inputs = []; outputs = []; busy = true; toggle(true);
        status('Connecting to your ComfyUI GPU host…');
        current.onopen = () => current.send(JSON.stringify({op: 'start', settings: active}));
        current.onmessage = async event => {
            if (epoch !== generation) return;
            try {
                const msg = JSON.parse(event.data);
                if (msg.type === 'loading') status(msg.message);
                if (msg.type === 'error') { stop(msg.message); return; }
                if (msg.type === 'ready') {
                    ready = true; busy = false; toggle(true);
                    $('gpu').textContent = `${msg.gpu} · ${msg.vram_gib} GiB`;
                    status('Streaming. Waiting for the first generated block; kernel compilation may take several minutes.');
                    timer = setInterval(sample, 1000 / fps);
                    playbackTimer = setInterval(() => { const frame = outputs.shift(); if (frame) out.drawImage(frame, 0, 0, 832, 480); }, 1000 / fps);
                }
                if (msg.type === 'frames') {
                    $('render-fps').textContent = msg.generated_fps.toFixed(1);
                    $('latency').textContent = `${((performance.now() - sentAt) / 1000).toFixed(2)} s`;
                    $('memory').textContent = `${msg.peak_vram_gib.toFixed(1)} GiB`;
                    await showFrames(msg.frames, epoch);
                    if (epoch !== generation) return;
                    busy = false; $('reset').disabled = false;
                    status(`Streaming · block ${msg.block}. Prompt and strength changes apply at block boundaries.`);
                    sendBlock();
                }
                if (msg.type === 'reset') { busy = false; inputs = []; outputs = []; $('reset').disabled = false; status('Temporal context reset.'); }
            } catch (error) { stop(error.message); }
        };
        current.onerror = () => { if (epoch === generation) stop('WebSocket connection failed before generation. Check the ComfyUI log for an origin rejection; behind a proxy, configure KREA_PUBLIC_ORIGIN to match this page’s origin. Also check proxy WebSocket support.'); };
        current.onclose = () => { if (epoch === generation) stop('Server disconnected. Check worker.log and restart the session.'); };
    } catch (error) { status(error.message); }
};
$('stop').onclick = () => stop();
$('reset').onclick = () => {
    if (!ready || busy) return;
    busy = true; inputs = []; outputs = []; $('reset').disabled = true;
    socket.send(JSON.stringify({op: 'reset', settings: active}));
};
$('record').onclick = () => {
    if (recorder?.state === 'recording') { recorder.stop(); return; }
    try {
        const mimeType = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'].find(x => MediaRecorder.isTypeSupported(x));
        if (!mimeType) throw Error('This browser cannot record WebM. Try Chrome or Edge.');
        const stream = output.captureStream(Number($('fps').value));
        recorder = new MediaRecorder(stream, {mimeType}); recording = [];
        let bytes = 0;
        recorder.ondataavailable = event => {
            if (event.data.size) { recording.push(event.data); bytes += event.data.size; }
            if (bytes > 256 * 1024 * 1024 && recorder.state === 'recording') recorder.stop();
        };
        recorder.onstop = () => {
            const url = URL.createObjectURL(new Blob(recording, {type: mimeType}));
            const link = document.createElement('a'); link.href = url; link.download = `krea-live-${Date.now()}.webm`; link.click();
            setTimeout(() => URL.revokeObjectURL(url), 10000); stream.getTracks().forEach(track => track.stop());
            recording = []; $('record').textContent = 'Record output';
        };
        recorder.start(1000); $('record').textContent = 'Save recording';
    } catch (error) { status(error.message); }
};
window.addEventListener('pagehide', () => stop());
