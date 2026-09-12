# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
from src.features.conversational import extract_features_from_turns, FEATURE_ORDER

TURNS = os.path.join(ROOT, 'hackmty26', 'turns')
MANIFEST = os.path.join(ROOT, 'hackmty26', 'manifest.csv')


def build_matrix():
    mani = pd.read_csv(MANIFEST)
    rows, y, split, ids = [], [], [], []
    for _, r in mani.iterrows():
        jp = os.path.join(TURNS, r['anon_id'] + '.json')
        if not os.path.exists(jp):
            continue
        turns = json.load(open(jp, encoding='utf-8')).get('turns', [])
        feats = extract_features_from_turns(turns, r['duration_s'])
        rows.append([feats[k] for k in FEATURE_ORDER])
        y.append(1 if r['label'] == 'synthetic' else 0)
        split.append(r['split'])
        ids.append(r['anon_id'])
    X = pd.DataFrame(rows, columns=FEATURE_ORDER)
    return X, np.array(y), np.array(split), np.array(ids)


def main():
    X, y, split, ids = build_matrix()
    tr = split == 'train'
    print(f'N={len(y)} | train={tr.sum()} val={(~tr).sum()} | features={X.shape[1]}')
    print(f'train: human={ (y[tr]==0).sum() } synthetic={ (y[tr]==1).sum() }\n')

    rows = []
    for c in FEATURE_ORDER:
        v = X[c].values[tr]
        yy = y[tr]
        if np.std(v) == 0:
            auc = 0.5
        else:
            auc = roc_auc_score(yy, v)
            auc = max(auc, 1 - auc)
        mh = np.mean(v[yy == 0]); ms = np.mean(v[yy == 1])
        rows.append((c, auc, mh, ms))
    rows.sort(key=lambda r: -r[1])
    print(f"{'feature':30s} {'AUC':>6s} {'human_mean':>12s} {'synth_mean':>12s}")
    for c, auc, mh, ms in rows:
        print(f'{c:30s} {auc:6.3f} {mh:12.3f} {ms:12.3f}')

    X.assign(label=y, split=split, anon_id=ids).to_csv(os.path.join(ROOT, 'models', 'altur_features.csv'), index=False)
    print('\nMatriz guardada en models/altur_features.csv')


if __name__ == '__main__':
    main()
