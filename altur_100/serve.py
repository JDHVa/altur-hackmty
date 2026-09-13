import io
import os
import sys
import json
import base64
import binascii
import subprocess
import threading
import urllib.request
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pipeline import predict, explain, audio_only

app = FastAPI(title='Altur 100 - A+B')

PI_URL = os.environ.get('ALTUR_PI_URL', '')


def notify_pi(is_syn, confidence):
    if not PI_URL:
        return
    state = 'bot' if is_syn else 'human'
    label = 'VOZ IA' if is_syn else 'HUMANO'

    def go():
        try:
            body = json.dumps({'state': state, 'confidence': float(confidence), 'label': label}).encode()
            req = urllib.request.Request(PI_URL, data=body, headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req, timeout=1.5).read()
        except Exception:
            pass
    threading.Thread(target=go, daemon=True).start()


class DetectRequest(BaseModel):
    audio_base64: str


def decode(audio_base64):
    raw = base64.b64decode(audio_base64, validate=True)
    try:
        return sf.read(io.BytesIO(raw), always_2d=True, dtype='float32')
    except Exception:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        p = subprocess.run([ff, '-hide_banner', '-loglevel', 'error', '-i', 'pipe:0', '-f', 'wav', 'pipe:1'],
                           input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return sf.read(io.BytesIO(p.stdout), always_2d=True, dtype='float32')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/detect')
def detect(payload: DetectRequest):
    try:
        data, sr = decode(payload.audio_base64)
    except (binascii.Error, ValueError, RuntimeError):
        raise HTTPException(status_code=400, detail='audio invalido')
    d = predict(data, sr)
    notify_pi(d['is_synthetic'], d['confidence'])
    return {'is_synthetic': d['is_synthetic'], 'confidence': d['confidence']}


@app.post('/detect/detailed')
def detect_detailed(payload: DetectRequest):
    try:
        data, sr = decode(payload.audio_base64)
    except (binascii.Error, ValueError, RuntimeError):
        raise HTTPException(status_code=400, detail='audio invalido')
    d = predict(data, sr)
    notify_pi(d['is_synthetic'], d['confidence'])
    return d


@app.post('/detect/audio')
def detect_audio(payload: DetectRequest):
    try:
        data, sr = decode(payload.audio_base64)
    except (binascii.Error, ValueError, RuntimeError):
        raise HTTPException(status_code=400, detail='audio invalido')
    d = audio_only(data, sr)
    p = d['p_synthetic']
    notify_pi(d['is_synthetic'], p if d['is_synthetic'] else 1.0 - p)
    return d


@app.post('/explain')
def explain_route(payload: DetectRequest):
    try:
        data, sr = decode(payload.audio_base64)
    except (binascii.Error, ValueError, RuntimeError):
        raise HTTPException(status_code=400, detail='audio invalido')
    return explain(data, sr)


_STATIC = os.path.join(HERE, 'static')
if os.path.isdir(_STATIC):
    from fastapi.staticfiles import StaticFiles
    app.mount('/console', StaticFiles(directory=_STATIC, html=True), name='console')
