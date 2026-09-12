# -*- coding: utf-8 -*-
import numpy as np
import parselmouth
from parselmouth.praat import call

_KEYS = ['jitter_local', 'shimmer_local', 'hnr', 'f0_mean', 'f0_std', 'voiced_frac']


def prosody_features(caller_wave: np.ndarray, sr: int) -> dict:
    if caller_wave.ndim > 1:
        caller_wave = caller_wave[0]
    x = np.ascontiguousarray(caller_wave, dtype=np.float64)
    out = {k: 0.0 for k in _KEYS}
    try:
        snd = parselmouth.Sound(x, sampling_frequency=float(sr))
        pitch = snd.to_pitch(pitch_floor=75.0, pitch_ceiling=400.0)
        f0 = pitch.selected_array['frequency']
        voiced = f0[f0 > 0]
        out['f0_mean'] = float(np.mean(voiced)) if voiced.size else 0.0
        out['f0_std'] = float(np.std(voiced)) if voiced.size > 1 else 0.0
        out['voiced_frac'] = float(voiced.size / f0.size) if f0.size else 0.0

        pp = call(snd, 'To PointProcess (periodic, cc)', 75.0, 400.0)
        out['jitter_local'] = float(call(pp, 'Get jitter (local)', 0, 0, 0.0001, 0.02, 1.3))
        out['shimmer_local'] = float(call([snd, pp], 'Get shimmer (local)', 0, 0, 0.0001, 0.02, 1.3, 1.6))
        harm = call(snd, 'To Harmonicity (cc)', 0.01, 75.0, 0.1, 1.0)
        out['hnr'] = float(call(harm, 'Get mean', 0, 0))
    except Exception:
        pass
    for k in _KEYS:
        v = out[k]
        if not np.isfinite(v):
            out[k] = 0.0
    return out


def prosody_vector(caller_wave: np.ndarray, sr: int) -> np.ndarray:
    f = prosody_features(caller_wave, sr)
    return np.array([f[k] for k in _KEYS], dtype=np.float32)
