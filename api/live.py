import asyncio
import base64
import io
import json
import os
import time

import numpy as np
import soundfile as sf
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.inference import _load, audio_score, bio_features, tabular_score, _fuse, recommend, predict_detailed, threshold
from api.dataset import AUDIO_DIR

router = APIRouter()

SR = 8000
WINDOW_S = 8.0
TABULAR_MIN_S = 10.0
TABULAR_EVERY_S = 6.0
TICK_S = 1.0


class LiveCall:
    def __init__(self, sr=SR):
        self.sr = sr
        self.caller = np.zeros(0, dtype=np.float32)
        self.agent = np.zeros(0, dtype=np.float32)
        self.last_tab = None
        self.last_tab_t = -1e9
        self.busy = False

    def append(self, caller, agent=None):
        self.caller = np.concatenate([self.caller, caller.astype(np.float32)])
        if agent is None:
            agent = np.zeros_like(caller, dtype=np.float32)
        self.agent = np.concatenate([self.agent, agent.astype(np.float32)])

    @property
    def t(self):
        return len(self.caller) / self.sr

    def frame(self):
        n = int(WINDOW_S * self.sr)
        window = self.caller[-n:]
        if len(window) < self.sr:
            return None
        p_audio = float(audio_score(window, self.sr))
        if self.t >= TABULAR_MIN_S and self.t - self.last_tab_t >= TABULAR_EVERY_S:
            self.last_tab, _ = tabular_score(self.caller, self.agent, self.sr)
            self.last_tab_t = self.t
        p_tab = self.last_tab
        if p_tab is not None:
            final, thr = _fuse(p_tab, p_audio)
        else:
            final, thr = p_audio, threshold()
        return {
            'type': 'frame',
            't': round(self.t, 2),
            'p_audio': round(p_audio, 4),
            'p_tabular': round(p_tab, 4) if p_tab is not None else None,
            'p_final': round(float(final), 4),
            'threshold': round(float(thr), 4),
            'bio': bio_features(window, self.sr),
            'recommendation': recommend(float(final), thr),
        }

    def final(self, include_audio):
        if len(self.caller) < self.sr:
            return {'type': 'error', 'detail': 'audio insuficiente para veredicto'}
        data = np.stack([self.caller, self.agent], axis=1)
        d = predict_detailed(data, self.sr)
        d['type'] = 'final'
        if include_audio:
            buf = io.BytesIO()
            sf.write(buf, data, self.sr, format='WAV', subtype='PCM_16')
            d['audio_base64'] = base64.b64encode(buf.getvalue()).decode('ascii')
        return d


async def _emit_frame(ws, call):
    if call.busy:
        return
    call.busy = True
    try:
        msg = await asyncio.to_thread(call.frame)
        if msg:
            await ws.send_text(json.dumps(msg))
    except Exception:
        pass
    finally:
        call.busy = False


@router.websocket('/ws/call')
async def ws_call(ws: WebSocket, source: str = 'mic', anon_id: str | None = None, speed: float = 1.0):
    await ws.accept()
    _load()
    call = LiveCall()
    await ws.send_text(json.dumps({'type': 'ready', 'threshold': threshold(), 'sr': SR}))
    try:
        if source == 'dataset':
            await _run_dataset(ws, call, anon_id, speed)
        else:
            await _run_mic(ws, call)
    except WebSocketDisconnect:
        return


async def _run_mic(ws, call):
    acc = 0.0
    while True:
        msg = await ws.receive()
        if msg.get('type') == 'websocket.disconnect':
            return
        if msg.get('bytes') is not None:
            pcm = np.frombuffer(msg['bytes'], dtype=np.int16).astype(np.float32) / 32768.0
            call.append(pcm)
            acc += len(pcm) / call.sr
            if acc >= TICK_S:
                acc = 0.0
                asyncio.create_task(_emit_frame(ws, call))
        elif msg.get('text'):
            cmd = json.loads(msg['text'])
            if cmd.get('type') == 'hangup':
                while call.busy:
                    await asyncio.sleep(0.05)
                final = await asyncio.to_thread(call.final, True)
                await ws.send_text(json.dumps(final))
                await ws.close()
                return


async def _run_dataset(ws, call, anon_id, speed):
    path = os.path.join(AUDIO_DIR, f'{anon_id}.wav')
    if not anon_id or not os.path.exists(path):
        await ws.send_text(json.dumps({'type': 'error', 'detail': 'audio del dataset no disponible'}))
        await ws.close()
        return
    data, sr = sf.read(path, always_2d=True, dtype='float32')
    call.sr = sr
    chunk = int(TICK_S * sr)
    hung_up = asyncio.Event()

    async def listen():
        try:
            while True:
                msg = await ws.receive()
                if msg.get('type') == 'websocket.disconnect':
                    hung_up.set()
                    return
                if msg.get('text') and json.loads(msg['text']).get('type') == 'hangup':
                    hung_up.set()
                    return
        except WebSocketDisconnect:
            hung_up.set()

    listener = asyncio.create_task(listen())
    started = time.monotonic()
    pos = 0
    while pos < len(data) and not hung_up.is_set():
        seg = data[pos:pos + chunk]
        call.append(seg[:, 0], seg[:, 1] if seg.shape[1] > 1 else None)
        pos += chunk
        asyncio.create_task(_emit_frame(ws, call))
        target = started + (pos / sr) / max(speed, 0.1)
        delay = target - time.monotonic()
        if delay > 0:
            try:
                await asyncio.wait_for(hung_up.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
    while call.busy:
        await asyncio.sleep(0.05)
    final = await asyncio.to_thread(call.final, False)
    try:
        await ws.send_text(json.dumps(final))
        await ws.close()
    except Exception:
        pass
    listener.cancel()
