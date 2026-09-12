# -*- coding: utf-8 -*-
import os
import sys
import glob
import numpy as np
import soundfile as sf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from features.prosody import prosody_features, _KEYS

HUMAN_DIRS = ['datasets_externos/Human_ES_train_8kHz', 'datasets_externos/Human_ES_test_8kHz',
              'datasets_externos/Human_MX_8kHz']
SYNTH_DIRS = ['datasets_externos/Synthetic_MX_8kHz', 'datasets_externos/Synthetic_Multi_8kHz',
              'datasets_externos/Synthetic_Piper_8kHz']


def collect(dirs):
    rows = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, '*.wav'))):
            w, sr = sf.read(f)
            c = (w[:, 0] if w.ndim > 1 else w).astype(np.float32)
            rows.append(prosody_features(c, sr))
    return rows


def main():
    hum = collect(HUMAN_DIRS)
    syn = collect(SYNTH_DIRS)
    print(f'humanos={len(hum)} sinteticos={len(syn)}\n')
    y = np.array([0] * len(hum) + [1] * len(syn))
    print(f"{'feature':16s} {'human_media':>12s} {'synth_media':>12s} {'AUC_univar':>11s}")
    for k in _KEYS:
        vh = np.array([r[k] for r in hum])
        vs = np.array([r[k] for r in syn])
        v = np.r_[vh, vs]
        auc = roc_auc_score(y, v)
        auc = max(auc, 1 - auc)
        print(f'{k:16s} {np.mean(vh):12.3f} {np.mean(vs):12.3f} {auc:11.3f}')


if __name__ == '__main__':
    main()
