import io
import json
import os
import base64

import numpy as np
import pandas as pd
import soundfile as sf
import joblib

from src.features.conversational import extract_features_from_turns, turns_from_audio

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
    raw = base64.b64decode(audio_base64)
    data, sr = sf.read(io.BytesIO(raw), always_2d=True)
    return data, sr


def predict(audio_base64):
    state = _load()
    tabular = state['tabular']
    feature_cols = state['feature_cols']
    ensemble = state['ensemble']

    data, sr = decode_wav(audio_base64)
    caller = data[:, 0]
    agent = data[:, 1] if data.shape[1] > 1 else np.zeros_like(caller)
    duration_s = len(caller) / sr if sr else 0.0

    turns = turns_from_audio(caller, agent, sr)
    feats = extract_features_from_turns(turns, duration_s)
    x = pd.DataFrame([[feats[c] for c in feature_cols]], columns=feature_cols)
    p_tabular = float(tabular.predict_proba(x)[0, 1])

    p_audio = float(audio_score(caller, sr))

    if ensemble is not None:
        signals = ensemble['signals']
        fusion = ensemble['fusion']
        cal_tab = float(ensemble['calibrator'].transform([p_tabular])[0])
        if 'audio' in signals and fusion['mode'] == 'avg' and p_audio != 0.5:
            w = fusion['w_tab']
            final = w * cal_tab + (1 - w) * p_audio
        else:
            final = cal_tab
        threshold = ensemble['threshold']
    else:
        final = p_tabular
        threshold = 0.5

    is_synth = bool(final > threshold)
    confidence = final if is_synth else 1 - final
    return {'is_synthetic': is_synth, 'confidence': round(float(confidence), 4)}
