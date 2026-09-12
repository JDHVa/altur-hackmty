import os

import numpy as np
import pandas as pd
import soundfile as sf

from src.features.conversational import (
    turns_from_audio, extract_features_from_turns, FEATURE_ORDER,
)

MANIFEST = 'hackmty26/manifest.csv'
AUDIO_DIR = 'hackmty26/audio'
CACHE = 'src/models/saved/vad_features.csv'


def _features_for_call(anon_id):
    data, sr = sf.read(os.path.join(AUDIO_DIR, anon_id + '.wav'), always_2d=True)
    caller = data[:, 0]
    agent = data[:, 1] if data.shape[1] > 1 else np.zeros_like(caller)
    turns = turns_from_audio(caller, agent, sr)
    return extract_features_from_turns(turns, len(caller) / sr if sr else 0.0)


def build(force=False):
    if os.path.exists(CACHE) and not force:
        return pd.read_csv(CACHE)

    manifest = pd.read_csv(MANIFEST)
    rows = []
    for _, r in manifest.iterrows():
        feats = _features_for_call(r['anon_id'])
        feats['anon_id'] = r['anon_id']
        feats['label_num'] = 1 if r['label'] == 'synthetic' else 0
        feats['split'] = r['split']
        rows.append(feats)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


def load(split, force=False):
    df = build(force=force)
    if split:
        df = df[df['split'] == split].reset_index(drop=True)
    return df, list(FEATURE_ORDER)


if __name__ == '__main__':
    df = build(force=True)
    print('VAD features:', df.shape, '->', CACHE)
