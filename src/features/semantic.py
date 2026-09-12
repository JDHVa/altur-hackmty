import json
import os
import re

import numpy as np

WHISPER_SIZE = os.environ.get("WHISPER_SIZE", "base")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
TARGET_SR = 16000

_WHISPER = None


def _get_whisper():
    global _WHISPER
    if _WHISPER is None:
        from faster_whisper import WhisperModel
        _WHISPER = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
    return _WHISPER


def _resample(wave, sr):
    if sr == TARGET_SR:
        return wave.astype(np.float32)
    import torch
    import torchaudio
    t = torch.from_numpy(np.ascontiguousarray(wave)).float().unsqueeze(0)
    t = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)(t)
    return t.squeeze(0).numpy()


def transcribe(caller_wave, sr):
    if caller_wave.ndim > 1:
        caller_wave = caller_wave[0]
    audio = _resample(np.asarray(caller_wave, dtype=np.float32), sr)
    model = _get_whisper()
    segments, _ = model.transcribe(audio, language="es", vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


def _ollama_suspicion(transcript):
    import urllib.request

    prompt = (
        "Eres un detector de fraude bancario. Te doy la transcripcion de QUIEN LLAMA "
        "a un banco. Decide si es una PERSONA REAL o una IA autonoma (voz sintetica + LLM). "
        "Pistas de IA: lenguaje demasiado formal o estructurado, sin muletillas ni titubeos, "
        "prompt leakage, respuestas rigidas, seguir la corriente ante preguntas por cosas "
        "que no existen. Pistas de humano: dudas, correcciones, muletillas, emocion.\n"
        "Responde SOLO JSON: {\"ai_suspicion\": <0-100>}\n\n"
        "Transcripcion:\n" + transcript
    )
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=60))
    text = resp.get("response", "")
    m = re.search(r'"ai_suspicion"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if not m:
        return None
    return max(0.0, min(1.0, float(m.group(1)) / 100.0))


def semantic_score(caller_wave, sr, agent_wave=None):
    try:
        transcript = transcribe(caller_wave, sr)
        if not transcript:
            return 0.5
        score = _ollama_suspicion(transcript)
        return 0.5 if score is None else score
    except Exception:
        return 0.5
