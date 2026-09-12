import io
import json
import os
import base64
import binascii

import numpy as np
import pandas as pd
import soundfile as sf
import joblib

from src.features.conversational import extract_features_from_turns, turns_from_audio


class InvalidAudioError(ValueError):
    pass

try:
    from src.features.audio import audio_score
except Exception:
    def audio_score(caller_wave, sr):
        return 0.5

TABULAR_PATH = 'src/models/saved/lgbm_tabular.pkl'
FEATURES_PATH = 'src/models/saved/tabular_features.json'
ENSEMBLE_PATH = 'src/models/saved/ensemble.pkl'

_state = None


def _load():
    global _state
    if _state is None:
        ensemble = joblib.load(ENSEMBLE_PATH) if os.path.exists(ENSEMBLE_PATH) else None
        if ensemble is not None:
            _state = {
                'tabular': ensemble['model'],
                'feature_cols': ensemble['feature_cols'],
                'ensemble': ensemble,
            }
        else:
            tabular = joblib.load(TABULAR_PATH)
            with open(FEATURES_PATH, encoding='utf-8') as f:
                feature_cols = json.load(f)
            _state = {'tabular': tabular, 'feature_cols': feature_cols, 'ensemble': None}
    return _state


def decode_wav(audio_base64):
    if not isinstance(audio_base64, str) or not audio_base64.strip():
        raise InvalidAudioError('audio_base64 vacio o no es texto')
    try:
        raw = base64.b64decode(audio_base64, validate=True)
    except (binascii.Error, ValueError):
        raise InvalidAudioError('base64 invalido')
    if not raw:
        raise InvalidAudioError('audio vacio tras decodificar base64')
    try:
        data, sr = sf.read(io.BytesIO(raw), always_2d=True, dtype='float32')
    except Exception as exc:
        raise InvalidAudioError(f'no se pudo leer el WAV: {str(exc)[:120]}')
    if data.size == 0 or sr <= 0:
        raise InvalidAudioError('audio sin muestras o sample rate invalido')
    return data, sr


VERIFY_FLOOR = 0.35

try:
    from src.features.prosody import prosody_features
except Exception:
    def prosody_features(caller_wave, sr):
        return {}


def recommend(p, threshold):
    if p >= threshold:
        return 'hangup'
    if p >= VERIFY_FLOOR:
        return 'verify'
    return 'continue'


def threshold():
    ensemble = _load()['ensemble']
    return float(ensemble['threshold']) if ensemble is not None else 0.5


def _fuse(p_tabular, p_audio):
    ensemble = _load()['ensemble']
    if ensemble is None:
        return p_tabular, 0.5
    signals = ensemble['signals']
    fusion = ensemble['fusion']
    cal_tab = float(ensemble['calibrator'].transform([p_tabular])[0])
    if 'audio' in signals and fusion['mode'] == 'avg' and p_audio is not None and p_audio != 0.5:
        w = fusion['w_tab']
        final = w * cal_tab + (1 - w) * p_audio
    else:
        final = cal_tab
    return float(final), float(ensemble['threshold'])


def tabular_score(caller, agent, sr):
    state = _load()
    duration_s = len(caller) / sr if sr else 0.0
    turns = turns_from_audio(caller, agent, sr)
    feats = extract_features_from_turns(turns, duration_s)
    x = pd.DataFrame([[feats[c] for c in state['feature_cols']]], columns=state['feature_cols'])
    return float(state['tabular'].predict_proba(x)[0, 1]), turns


def bio_features(caller, sr):
    f = prosody_features(caller, sr)
    keys = ('hnr', 'shimmer_local', 'jitter_local', 'voiced_frac', 'f0_mean', 'f0_std')
    return {k: round(float(f[k]), 4) for k in keys if k in f}


def predict_detailed(data, sr):
    caller = data[:, 0]
    agent = data[:, 1] if data.shape[1] > 1 else np.zeros_like(caller)
    duration_s = len(caller) / sr if sr else 0.0
    p_tabular, turns = tabular_score(caller, agent, sr)
    p_audio = float(audio_score(caller, sr))
    final, thr = _fuse(p_tabular, p_audio)
    is_synth = bool(final > thr)
    confidence = final if is_synth else 1 - final
    return {
        'is_synthetic': is_synth,
        'confidence': round(float(confidence), 4),
        'p_final': round(final, 4),
        'p_tabular': round(p_tabular, 4),
        'p_audio': round(p_audio, 4),
        'threshold': round(thr, 4),
        'duration_s': round(duration_s, 2),
        'n_turns': len(turns),
        'bio': bio_features(caller, sr),
        'recommendation': recommend(final, thr),
    }


def predict(audio_base64):
    data, sr = decode_wav(audio_base64)
    d = predict_detailed(data, sr)
    return {'is_synthetic': d['is_synthetic'], 'confidence': d['confidence']}
