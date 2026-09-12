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

MODEL_PATH = 'src/models/saved/lgbm_tabular.pkl'
FEATURES_PATH = 'src/models/saved/tabular_features.json'
TABULAR_WEIGHT = 0.7
THRESHOLD = 0.5

_model = None
_feature_cols = None


def _load():
    global _model, _feature_cols
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH, encoding='utf-8') as f:
            _feature_cols = json.load(f)
    return _model, _feature_cols


def decode_wav(audio_base64):
    raw = base64.b64decode(audio_base64)
    data, sr = sf.read(io.BytesIO(raw), always_2d=True)
    return data, sr


def predict(audio_base64):
    model, feature_cols = _load()
    data, sr = decode_wav(audio_base64)

    caller = data[:, 0]
    agent = data[:, 1] if data.shape[1] > 1 else np.zeros_like(caller)
    duration_s = len(caller) / sr if sr else 0.0

    turns = turns_from_audio(caller, agent, sr)
    feats = extract_features_from_turns(turns, duration_s)
    x = pd.DataFrame([[feats[c] for c in feature_cols]], columns=feature_cols)

    p_tabular = float(model.predict_proba(x)[0, 1])
    p_audio = float(audio_score(caller, sr))

    if p_audio == 0.5:
        final = p_tabular
    else:
        final = TABULAR_WEIGHT * p_tabular + (1 - TABULAR_WEIGHT) * p_audio

    return {
        'is_synthetic': bool(final > THRESHOLD),
        'confidence': round(final if final > 0.5 else 1 - final, 4),
    }
