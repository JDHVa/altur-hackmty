import os
import sys
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
_AB = None


def _load():
    global _AB
    if _AB is None:
        _AB = joblib.load(MODEL_PATH)
    return _AB


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
