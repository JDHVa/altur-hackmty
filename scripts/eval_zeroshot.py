# -*- coding: utf-8 -*-
import os
import sys
import csv
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from features.audio_zeroshot import zeroshot_score

ROOT = os.path.join(os.path.dirname(__file__), '..', 'hackmty26')
AUDIO = os.path.join(ROOT, 'audio')


def load_val():
    rows = []
    with open(os.path.join(ROOT, 'manifest.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['split'] == 'val':
                rows.append((r['anon_id'], 1 if r['label'] == 'synthetic' else 0))
    return rows


def eer(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2


def main():
    from sklearn.metrics import roc_auc_score
    val = load_val()
    ys, ss = [], []
    for i, (aid, y) in enumerate(val):
        path = os.path.join(AUDIO, aid + '.wav')
        w, sr = sf.read(path)
        caller = (w[:, 0] if w.ndim > 1 else w).astype(np.float32)
        s = zeroshot_score(caller, sr)
        ys.append(y)
        ss.append(s)
        print(f"[{i+1}/{len(val)}] {aid} y={y} score={s:.3f}", flush=True)
    ys, ss = np.array(ys), np.array(ss)
    auc = roc_auc_score(ys, ss)
    e = eer(ys, ss)
    acc = ((ss >= 0.5).astype(int) == ys).mean()
    print("=" * 40)
    print(f"N={len(ys)}  synthetic={ys.sum()}  human={(1-ys).sum()}")
    print(f"AUC={auc:.4f}  EER={e:.4f}  ACC@0.5={acc:.4f}")
    print(f"mean score synthetic={ss[ys==1].mean():.3f}  human={ss[ys==0].mean():.3f}")


if __name__ == '__main__':
    main()
