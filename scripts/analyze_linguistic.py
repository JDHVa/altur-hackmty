# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from features.stt import transcribe
from features.linguistic import linguistic_features, LING_KEYS

MANIFEST = os.path.join(ROOT, 'hackmty26', 'manifest.csv')
AUD = os.path.join(ROOT, 'hackmty26', 'audio')
CACHE = os.path.join(ROOT, 'models', 'ssl_cache', 'stt')
os.makedirs(CACHE, exist_ok=True)
MAX_S = 40


def stt_cached(aid):
    p = os.path.join(CACHE, aid + '.json')
    if os.path.exists(p):
        return json.load(open(p, encoding='utf-8'))
    w, sr = sf.read(os.path.join(AUD, aid + '.wav'))
    c = (w[:, 0] if w.ndim > 1 else w).astype(np.float32)[:sr * MAX_S]
    r = transcribe(c, sr)
    json.dump(r, open(p, 'w', encoding='utf-8'), ensure_ascii=False)
    return r


def main():
    mani = pd.read_csv(MANIFEST)
    rows, y, split = [], [], []
    for i, r in mani.iterrows():
        st = stt_cached(r['anon_id'])
        feats = linguistic_features(st)
        rows.append([feats[k] for k in LING_KEYS])
        y.append(1 if r['label'] == 'synthetic' else 0)
        split.append(r['split'])
        if (i + 1) % 40 == 0:
            print(f'[{i+1}/{len(mani)}]', flush=True)
    X = pd.DataFrame(rows, columns=LING_KEYS)
    y = np.array(y); split = np.array(split)
    tr = split == 'train'
    print(f'\nN={len(y)} train={tr.sum()}\n')
    print(f"{'feature':24s} {'AUC':>6s} {'human':>10s} {'synth':>10s}")
    res = []
    for c in LING_KEYS:
        v = X[c].values[tr]; yy = y[tr]
        auc = 0.5 if np.std(v) == 0 else max(roc_auc_score(yy, v), 1 - roc_auc_score(yy, v))
        res.append((c, auc, np.mean(v[yy == 0]), np.mean(v[yy == 1])))
    for c, a, mh, ms in sorted(res, key=lambda z: -z[1]):
        print(f'{c:24s} {a:6.3f} {mh:10.3f} {ms:10.3f}')
    X.assign(label=y, split=split, anon_id=mani['anon_id'].values).to_csv(os.path.join(ROOT, 'models', 'altur_linguistic.csv'), index=False)
    print('\nGuardado models/altur_linguistic.csv')


if __name__ == '__main__':
    main()
