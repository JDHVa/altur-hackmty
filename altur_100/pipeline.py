import os
import sys
import json
import numpy as np
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from features.conversational import extract_features_from_turns, turns_from_audio
from features.audio import audio_score
from features.heavy_audio import xlsr_sls_score

MODEL_PATH = os.path.join(SRC, 'models', 'saved', 'altur_ab.joblib')
STATS_PATH = os.path.join(SRC, 'models', 'saved', 'feature_stats.json')
_AB = None
_STATS = None

REASON_TXT = {
    'time_to_first_caller_s': lambda v, h: f'Tardo {v:.1f}s en empezar a hablar (humanos ~{h:.1f}s)',
    'caller_latency_mean': lambda v, h: f'Responde en promedio a {v:.1f}s de latencia (humanos ~{h:.1f}s)',
    'caller_latency_median': lambda v, h: f'Latencia tipica de respuesta {v:.1f}s (humanos ~{h:.1f}s)',
    'n_overlaps': lambda v, h: f'Interrumpio/solapo {v:.0f} veces (humanos ~{h:.0f})',
    'overlap_total_s': lambda v, h: f'{v:.1f}s hablando encima del agente (humanos ~{h:.1f}s)',
    'n_turns_caller': lambda v, h: f'{v:.0f} turnos del que llama (humanos ~{h:.0f})',
    'caller_turns_per_min': lambda v, h: f'{v:.1f} turnos por minuto (humanos ~{h:.1f})',
    'silence_mean': lambda v, h: f'Silencios promedio de {v:.1f}s (humanos ~{h:.1f}s)',
    'xlsr': lambda v, h: f'La voz suena sintetica (detector de audio {v * 100:.0f}%)',
    'wavlm': lambda v, h: f'Timbre poco natural (WavLM {v * 100:.0f}%)',
}


def _load():
    global _AB
    if _AB is None:
        _AB = joblib.load(MODEL_PATH)
    return _AB


def _stats():
    global _STATS
    if _STATS is None:
        _STATS = json.load(open(STATS_PATH, encoding='utf-8')) if os.path.exists(STATS_PATH) else {}
    return _STATS


def audio_signals(caller, sr):
    return {
        'wavlm': float(audio_score(caller, sr)),
        'xlsr': float(xlsr_sls_score(caller, sr)),
    }


def feature_vector(caller, agent, sr):
    ab = _load()
    duration_s = len(caller) / sr if sr else 0.0
    turns = turns_from_audio(caller, agent, sr)
    feats = extract_features_from_turns(turns, duration_s)
    sig = audio_signals(caller, sr)
    feats['wavlm'] = sig['wavlm']
    feats['xlsr'] = sig['xlsr']
    x = np.array([[float(feats.get(k, 0.0)) for k in ab['features']]])
    return x, sig, turns


def predict(data, sr):
    ab = _load()
    if data.ndim == 1:
        data = data[:, None]
    caller = data[:, 0]
    agent = data[:, 1] if data.shape[1] > 1 else np.zeros_like(caller)
    x, sig, turns = feature_vector(caller, agent, sr)
    p = float(ab['model'].predict_proba(x)[0, 1])
    is_synth = bool(p >= 0.5)
    rec = 'hangup' if p >= 0.5 else ('verify' if p >= 0.35 else 'continue')
    w = {'wavlm': 0.4, 'xlsr': 0.6}
    live = {k: v for k, v in sig.items() if v != 0.5}
    p_audio = sum(live[k] * w[k] for k in live) / sum(w[k] for k in live) if live else 0.5
    duration_s = len(caller) / sr if sr else 0.0
    return {
        'is_synthetic': is_synth,
        'confidence': round(p if is_synth else 1.0 - p, 4),
        'p_synthetic': round(p, 4),
        'p_final': round(p, 4),
        'p_audio': round(float(p_audio), 4),
        'threshold': 0.5,
        'recommendation': rec,
        'signals': {k: round(v, 4) for k, v in sig.items()},
        'duration_s': round(duration_s, 2),
        'n_turns': len(turns),
        'turns': [{'channel': int(t['channel']), 'start': round(float(t['start']), 2), 'end': round(float(t['end']), 2)} for t in turns[:400]],
    }


def audio_only(data, sr):
    if data.ndim == 1:
        data = data[:, None]
    caller = data[:, 0]
    sig = audio_signals(caller, sr)
    w = {'wavlm': 0.4, 'xlsr': 0.6}
    live = {k: v for k, v in sig.items() if v != 0.5}
    p = sum(live[k] * w[k] for k in live) / sum(w[k] for k in live) if live else 0.5
    return {
        'p_synthetic': round(float(p), 4),
        'is_synthetic': bool(p >= 0.5),
        'signals': {k: round(v, 4) for k, v in sig.items()},
        'duration_s': round(len(caller) / sr if sr else 0.0, 2),
    }


def _reasons(feats):
    st = _stats()
    out = []
    for f, tmpl in REASON_TXT.items():
        if f not in st or f not in feats:
            continue
        v = float(feats[f])
        s = st[f]
        d_h = abs(v - s['human_mean']) / s['human_std']
        d_s = abs(v - s['synth_mean']) / s['synth_std']
        out.append({'text': tmpl(v, s['human_mean']), 'ia_like': bool(d_s < d_h), 'strength': float(abs(d_h - d_s))})
    out.sort(key=lambda r: -r['strength'])
    return [{'text': r['text'], 'ia_like': r['ia_like']} for r in out[:6]]


def explain(data, sr):
    ab = _load()
    if data.ndim == 1:
        data = data[:, None]
    caller = data[:, 0]
    agent = data[:, 1] if data.shape[1] > 1 else np.zeros_like(caller)
    duration_s = len(caller) / sr if sr else 0.0
    turns = turns_from_audio(caller, agent, sr)
    feats = extract_features_from_turns(turns, duration_s)
    sig = audio_signals(caller, sr)
    feats['wavlm'] = sig['wavlm']
    feats['xlsr'] = sig['xlsr']
    x = np.array([[float(feats.get(k, 0.0)) for k in ab['features']]])
    p = float(ab['model'].predict_proba(x)[0, 1])

    from features.transcribe import transcribe_segments
    segs = transcribe_segments(caller, sr)
    out_segs = []
    for s in segs:
        a = int(max(0, s['start']) * sr)
        b = int(min(duration_s, s['end']) * sr)
        chunk = caller[a:b]
        score = float(xlsr_sls_score(chunk, sr)) if b - a >= sr // 2 else 0.5
        out_segs.append({'start': round(s['start'], 2), 'end': round(s['end'], 2), 'text': s['text'], 'score': round(score, 3)})

    return {
        'is_synthetic': bool(p >= 0.5),
        'p_synthetic': round(p, 4),
        'reasons': _reasons(feats),
        'transcript': ' '.join(s['text'] for s in segs),
        'segments': out_segs,
    }
